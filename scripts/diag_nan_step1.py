"""Diagnose where NaN appears on step 1 of the real-DINOv3 training loop.

Hypothesis space (fedfuse-project.md): autocast/bf16 path, guided-filter /
variance stats, MoLoRA fp32 einsum, AdamW after step 0, router pi update.
We bisect: run step 0 -> step 1 forward, and check finiteness of
  - model params (before/after step)
  - backbone hidden states (per tap layer)
  - noise branch / fusion / decoder intermediate tensors
  - raw DINOv3 hidden_states from the HF model
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fedfuse.data import MayoLDCTDataset, TRAIN_PATIENTS
from fedfuse.losses import CompositeLoss
from fedfuse.models import FedFuseNet, LoRARouter
from fedfuse.physics import global_descriptor, noise_residual, local_variance_map

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np = __import__("numpy")

net = FedFuseNet("models/dinov3", lora_r=8, n_experts=3).to(DEV)
router = LoRARouter(d_in=288, n_experts=3).to(DEV)
loss_fn = CompositeLoss()
trainable = ([m.B for m in net.backbone.lora_modules()]
             + list(router.parameters())
             + [p for n, p in net.named_parameters()
                 if n.startswith(("noise_branch", "fusions", "decoder"))])
opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-5)

ds = MayoLDCTDataset("/mnt/d/icassp/datas/mayo_2016_npy",
                     TRAIN_PATIENTS[:2], max_slices=6, patch_size=256)
ld = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, drop_last=True)


def fin(name, t):
    bad = (~torch.isfinite(t)).sum().item()
    if bad:
        print(f"  !! NONFINITE {name}: {bad}/{t.numel()}  "
              f"nan={(~torch.isfinite(t)).sum().item() and t.isnan().sum().item()} "
              f"inf={t.isinf().sum().item()}  absmax={t.abs().max().item():.3e}")
        return True
    return False


def run_step(step):
    x, y = next(iter(ld))
    x, y = x.to(DEV), y.to(DEV)
    g = global_descriptor(x)[0]
    pi = router(g[None])[0]
    net.set_pi(pi)

    # ---- inputs to branches
    r = noise_residual(x)
    v = local_variance_map(r)
    for nm, t in (("x", x), ("r", r), ("v", v), ("y", y)):
        fin(f"input {nm}", t)

    # ---- backbone internals: raw hidden states
    x3 = x.repeat(1, 3, 1, 1)
    x3 = (x3 - net.backbone.mean) / net.backbone.std
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
        out = net.backbone.vit(pixel_values=x3, output_hidden_states=True)
    for li, hs in enumerate(out.hidden_states):
        if fin(f"vit hs[{li}]", hs):
            break
    feats = net.backbone(x)
    for i, f in enumerate(feats):
        if fin(f"tap feat {i}", f):
            break

    # ---- full forward under autocast, all internals recorded
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
        pred = net(x)
        fin("pred", pred)
        loss, parts = loss_fn(pred.float(), y)
        fin("loss", loss)
    print(f"step {step}: loss={loss.item():.4f} parts={parts} pi={pi.detach().cpu().numpy().round(3)}")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    print(f"  grad_norm(pre-clip scan)={gn.item():.4f}")
    opt.step()
    bad = sum(fin(f"param {n}", p) for n, p in net.named_parameters() if p.requires_grad)
    print(f"  nonfinite trainable params after step: {bad}")
    return loss.item()


run_step(0)
print("---- after step 0 ----")
# inspect B stats after one AdamW update
B0 = net.backbone.lora_modules()[0].B.detach()
print(f"  B[0] absmax={B0.abs().max().item():.3e} mean={B0.mean().item():.3e}")
run_step(1)
