"""Federated server: stacking-based aggregation with global anchor (Sec. III-D).

Every client participates every round. Per expert e of client k the server
folds routing mass into the up-projection,  B_hat_{k,e} = omega_{k,e} B_{k,e},
with omega_{k,e} = N_k pi_{k,e} / sum_j N_j pi_{j,e}, then builds the global
anchor by STACKING (no averaging -> no multiplicative LoRA bias):

    D = [A_{1,1}; ...; A_{K,E}]   in R^{(KEr) x d_in}
    C = [B_hat_{1,1} ... B_hat_{K,E}]  in R^{d_out x (KEr)}
    C @ D  ==  sum_{k,e} omega_{k,e} B_{k,e} A_{k,e}      (exact, no bias)

(C, D) are broadcast to all clients as a frozen global-anchor branch. A plain
FedAvg/FedProx path for full-parameter baseline networks is kept below.
"""
import torch


class FedFuseServer:
    def __init__(self, n_lora_modules: int, n_experts: int = 3,
                 r: int = 8, d_in: int = 1024, d_out: int = 1024):
        self.n_lora = n_lora_modules
        self.n_experts = n_experts
        self.r = r
        self.d_in = d_in
        self.d_out = d_out
        self.global_C = None          # (n_lora, d_out, KEr) after aggregate
        self.global_D = None          # (n_lora, KEr, d_in)
        self.aggregated = False

    def aggregate(self, uploads):
        """uploads: list of dicts {A: {lora_i: (E,r,d_in)}, B: {...},
        n: int} -> stacked global anchor (C, D).

        NOTE (2026-09-03, user decision): pi is NOT uploaded -- routing
        weights stay private to each client (they only blend that client's
        local experts in Delta W_k^tot). Server aggregation therefore uses
        DATA-SIZE weighting omega_k = N_k / sum_j N_j, identical for every
        expert block, and stacks all client-expert factors:
            D = [A_{1,1}; ...; A_{K,E}]  rows (k,e,r)
            C = [w_1 B_{1,1} ... w_K B_{K,E}]  cols (k,e,r)
            C@D == sum_{k,e} w_k B_{k,e} A_{k,e}   (exact, no avg bias)
        """
        K = len(uploads)
        E = self.n_experts
        r = self.r
        d_out, d_in = self.d_out, self.d_in
        ns = torch.tensor([u["n"] for u in uploads], dtype=torch.float64)
        w = (ns / ns.sum())                                     # (K,) data-size
        w = w[:, None, None, None]                              # (K,1,1,1)

        # per adapted projection index i, stack across clients & experts
        Cs, Ds = [], []
        for i in range(self.n_lora):
            A_all = torch.stack([u["A"][f"lora{i}"].double() for u in uploads])  # (K,E,r,d_in)
            B_all = torch.stack([u["B"][f"lora{i}"].double() for u in uploads])  # (K,E,d_out,r)
            # fold data-size weight into B, then stack. Row order of D and
            # column order of C must BOTH be (k outer, e inner, r innermost)
            # so that C@D == sum_{k,e} w_k B_{k,e} A_{k,e} exactly.
            D_i = A_all.reshape(K * E * r, d_in)          # rows: (k,e,r)
            # (K,E,d_out,r) -> permute to (d_out,K,E,r) -> cols (k,e,r)
            Bw = (B_all * w).permute(2, 0, 1, 3).reshape(d_out, K * E * r)
            Cs.append(Bw)
            Ds.append(D_i)
        # Each adapted projection owns its experts -> one anchor pair per
        # module slot (n_lora of them), sharing the same omega weighting.
        self.global_C = torch.stack(Cs).float()   # (n_lora, d_out, KEr)
        self.global_D = torch.stack(Ds).float()   # (n_lora, KEr, d_in)
        self.aggregated = True
        return self.global_C, self.global_D

    def broadcast(self, model):
        """Push stacked anchors into every injected module of the model."""
        assert self.aggregated, "aggregate() before broadcast()"
        for i, m in enumerate(model.backbone.lora_modules()):
            m.set_global_anchor(self.global_C[i], self.global_D[i])


# --------------------------------------------------------------------------- #
# Plain FedAvg / FedProx for baseline networks (full-parameter exchange)
# --------------------------------------------------------------------------- #
def fedavg_aggregate(state_dicts, ns):
    w = torch.tensor(ns, dtype=torch.float64)
    w = (w / w.sum()).float()
    out = {}
    for k in state_dicts[0]:
        out[k] = sum(w[i] * sd[k].float() for i, sd in enumerate(state_dicts))
    return out
