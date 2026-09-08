"""Federated client: local training of FedFuse under degradation heterogeneity.

Per communication round the client
  1. keeps its personalized LoRA factors {A_e, B_e} and router rho_phi from
     the previous round (personalization path is continuous across rounds),
  2. receives the server-stacked global anchor (C, D) -- FROZEN locally,
  3. (re)estimates its client-level degradation descriptor g_k (EMA),
  4. routes pi_k = softmax(rho_phi(g_k)/tau), injects into all MoLoRA modules,
  5. optimizes  L1 + l_ssim L_SSIM + l_edge L_edge + l_bal L_bal(pi_k)
     over {A_e, B_e}, router, and the LOCAL image-domain branch psi_k,
  6. uploads {A_e, B_e} and the router (anchors never leave the server).

Delta W_k^tot = scaling * ( personalized sum_e pi B_e A_e
                            + frozen global anchor C @ D )   (Sec. III-D)
"""
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..losses import CompositeLoss, router_balance_loss
from ..physics import global_descriptor


class FedFuseClient:
    def __init__(self, cid, dataset, device, batch_size=1, lr=1e-4,
                 local_epochs=1, lambda_bal=0.01, desc_samples=16,
                 g_ema=0.5, num_workers=2, seed=0):
        self.cid = cid
        self.device = device
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.lambda_bal = lambda_bal
        self.desc_samples = desc_samples
        self.g_ema = g_ema
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, drop_last=True,
                                 generator=torch.Generator().manual_seed(seed))
        self.n_samples = len(dataset)
        self.g_k = None                       # client descriptor (EMA)
        self.local_state = None               # psi_k: noise branch/fusion/decoder
        self.personal_state = None            # {A_e, B_e} of every q/v module
        self.router_state = None
        self.optim_state = None
        self.loss_fn = CompositeLoss()

    # ------------------------------------------------------------------ #
    def estimate_descriptor(self, model):
        model.eval()
        idxs = np.random.default_rng(self.cid).choice(
            len(self.loader.dataset), min(self.desc_samples, len(self.loader.dataset)),
            replace=False)
        gs = []
        with torch.no_grad():
            for i in idxs:
                x, _ = self.loader.dataset[i]
                g = global_descriptor(x[None].to(self.device))
                gs.append(g.cpu())
        g_new = torch.cat(gs).mean(0)
        if self.g_k is None:
            self.g_k = g_new
        else:
            self.g_k = self.g_ema * self.g_k + (1 - self.g_ema) * g_new
        return self.g_k

    # ------------------------------------------------------------------ #
    def _restore_personal(self, model):
        """Restore this client's personalized factors {A_e,B_e} into model."""
        if self.personal_state is None:
            return
        with torch.no_grad():
            for i, m in enumerate(model.backbone.lora_modules()):
                st = self.personal_state.get(f"lora{i}")
                if st is not None:
                    m.A.copy_(st["A"].to(self.device))
                    m.B.copy_(st["B"].to(self.device))

    def train_round(self, model, router, round_idx=0, server_C=None, server_D=None):
        """Returns upload dict {A,B per lora, router state, n, pi}.

        server_C/server_D: stacked global anchor from this round's broadcast.
        """
        # server anchor (frozen) first -- overrides any stale local copy
        if server_C is not None:
            model.set_global_anchor(server_C, server_D)
        # restore client-local state (personalized modules stay local)
        if self.local_state is not None:
            _load_local(model, self.local_state, self.device)
        if self.router_state is not None:
            router.load_state_dict(self.router_state)
        self._restore_personal(model)
        model.to(self.device).train()
        router.to(self.device).train()

        g_k = self.estimate_descriptor(model).to(self.device)

        # trainable: personalized A,B + router + local CNN modules.
        # global anchor C/D are buffers -> never in the optimizer.
        params = ([p for m in model.backbone.lora_modules() for p in (m.A, m.B)]
                  + list(router.parameters())
                  + [p for n, p in model.named_parameters() if _is_local(n)])
        opt = torch.optim.AdamW(params, lr=self._lr(), weight_decay=1e-5)
        if self.optim_state is not None:
            try:
                opt.load_state_dict(self.optim_state)
            except ValueError:
                pass

        logs = []
        pi = None
        for _ in range(self.local_epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                pi = router(g_k[None])[0]          # fresh graph each step
                model.set_pi(pi)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=self.device == "cuda"):
                    pred = model(x)
                    loss, parts = self.loss_fn(pred.float(), y)
                    loss = loss + self.lambda_bal * router_balance_loss(pi)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                logs.append(parts)

        self.local_state = _dump_local(model)
        self.router_state = copy.deepcopy(router.state_dict())
        self.optim_state = copy.deepcopy(opt.state_dict())
        # personalized factors persist locally across rounds
        self.personal_state = {f"lora{i}":
                               {"A": m.A.detach().cpu().clone(),
                                "B": m.B.detach().cpu().clone()}
                               for i, m in enumerate(model.backbone.lora_modules())}
        upload = dict(
            A={f"lora{i}": m.A.detach().cpu().clone()
               for i, m in enumerate(model.backbone.lora_modules())},
            B={f"lora{i}": m.B.detach().cpu().clone()
               for i, m in enumerate(model.backbone.lora_modules())},
            n=self.n_samples,
        )
        return upload, _mean_logs(logs)

    def _lr(self):
        return 1e-4


def _is_local(name):
    return (name.startswith("noise_branch") or name.startswith("fusions")
            or name.startswith("decoder"))


def _dump_local(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if _is_local(n)}


def _load_local(model, state, device):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.copy_(state[n].to(device))


def _mean_logs(logs):
    if not logs:
        return {}
    return {k: float(np.mean([l[k] for l in logs])) for k in logs[0]}
