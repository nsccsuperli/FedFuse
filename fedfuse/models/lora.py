"""Physics-guided mixture-of-LoRA with global-anchor stacking (Sec. III-D).

Each adapted projection keeps E parallel LoRA experts {(A_e, B_e)} of rank r.
Unlike the earlier design, BOTH A_e and B_e are trainable per client. The
server does NOT average factors (multiplicative bias bar(B)bar(A) !=
overline(BA)); instead it forms the global anchor by STACKING:

    B_hat_{k,e} = omega_{k,e} B_{k,e},   omega_{k,e} = N_k pi_{k,e} / sum_j ...
    D = [A_{1,1}; ...; A_{K,E}]           (c x d_in),   c = K*E*r
    C = [B_hat_{1,1} ... B_hat_{K,E}]     (d_out x c)
    global update  C @ D  =  sum_{k,e} omega_{k,e} B_{k,e} A_{k,e}   (exact)

The pair (C, D) is broadcast to every client and appended as a FROZEN
parallel branch (no gradient flows into it):

    Delta W_k^tot = scaling * ( sum_e pi_{k,e} B_e A_e   +   C @ D )
                    |______ personalized ______|   |_ global anchor _|

C and D are stored as non-trainable buffers: locally frozen by design, only
the server rewrites them via set_global_anchor(). The anchor starts at zero
(round 0 = pure personalized training) and becomes meaningful after the
first aggregation.

Numerical note: all adapter math runs in fp32 (autocast disabled) -- bf16
downcasting of these small einsums caused NaN gradients (see skill).
"""
import torch
import torch.nn as nn


def _fp32():
    return torch.autocast("cuda", enabled=False)


class MoLoRALinear(nn.Module):
    """Linear whose output is adapted by a routed mixture of LoRA experts
    plus a frozen global-anchor branch (stacked across clients)."""

    def __init__(self, base: nn.Linear, r: int = 8, n_experts: int = 3,
                 n_clients: int = 1, alpha: float = 16.0, seed: int = 1234):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_in, d_out = base.in_features, base.out_features
        g = torch.Generator().manual_seed(seed)   # same init across clients
        # both factors trainable (paper: {(A_{k,e}, B_{k,e})}, rank r)
        A = torch.randn(n_experts, r, d_in, generator=g) * (1.0 / r ** 0.5)
        B = torch.zeros(n_experts, d_out, r)
        self.A = nn.Parameter(A)                  # (E, r, d_in)
        self.B = nn.Parameter(B)                  # (E, d_out, r)
        self.scaling = alpha / r
        self.r = r
        self.n_experts = n_experts
        # global anchor: c = K*E*r ; frozen buffers, zero-initialized.
        c = n_clients * n_experts * r
        self.register_buffer("anchor_C", torch.zeros(d_out, c))   # (d_out, c)
        self.register_buffer("anchor_D", torch.zeros(c, d_in))    # (c, d_in)
        # pi: NOT a buffer. A plain attribute holding the CURRENT routing
        # weights -- with graph attached during training (see set_pi). A
        # buffer + copy_() severs autograd, starving the router of task
        # gradient and collapsing pi to uniform.
        self._pi = None

    # ------------------------------------------------------------------ #
    def set_pi(self, pi: torch.Tensor):
        """Store routing weights BY REFERENCE (graph preserved).

        Training: pass router output directly (pi.requires_grad True) so the
        task loss back-propagates into the router through every MoLoRA
        module. Eval: call inside torch.no_grad() or pass a detached tensor.
        Caller must keep pi on the same device as the module.
        """
        self._pi = pi

    def set_global_anchor(self, C: torch.Tensor, D: torch.Tensor):
        """Server broadcast: rewrite the frozen global branch."""
        with torch.no_grad():
            self.anchor_C.copy_(C.to(self.anchor_C.device, self.anchor_C.dtype))
            self.anchor_D.copy_(D.to(self.anchor_D.device, self.anchor_D.dtype))

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, pi: torch.Tensor | None = None) -> torch.Tensor:
        """pi: optional routing weights with graph attached (training).
        When None, the detached buffer set by set_pi() is used (eval).
        Passing pi directly is REQUIRED in training so gradients reach the
        router -- copying through the buffer severs the graph and the router
        then only sees the entropy-regularizer gradient, collapsing pi to
        uniform (observed: pi stayed at 0.333 forever)."""
        uniform = torch.full((self.n_experts,), 1.0 / self.n_experts,
                             device=x.device, dtype=x.dtype)
        if pi is None:
            pi = self._pi if self._pi is not None else uniform
        y = self.base(x)
        with _fp32():
            xf = x.float()
            # personalized branch:  sum_e pi_e B_e A_e x
            xa = torch.einsum("...d,erd->...re", xf, self.A.float())  # (...,E,r)
            delta_p = torch.einsum("...re,edr,e->...d",
                                   xa, self.B.float(), pi.float())
            # global-anchor branch:  (C @ D) x = C @ (D @ x)   [frozen]
            h = torch.einsum("...d,cd->...c", xf, self.anchor_D.float())
            delta_g = torch.einsum("...c,dc->...d", h, self.anchor_C.float())
        s = self.scaling
        return y + s * (delta_p + delta_g).to(y.dtype)

    # ------------------------------------------------------------------ #
    def expert_parameters(self):
        """Uploaded per round: BOTH factors (A and B) plus local pi handled
        by the router; anchors never leave the server."""
        return {"A": self.A, "B": self.B}

    def load_expert(self, state):
        with torch.no_grad():
            self.A.copy_(state["A"])
            self.B.copy_(state["B"])


class LoRARouter(nn.Module):
    """Two-layer MLP: global descriptor g_k -> expert weights pi_k (Eq. 9)."""

    def __init__(self, d_in: int = 288, n_experts: int = 3, hidden: int = 64,
                 tau: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_experts))
        self.tau = tau

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(g) / self.tau, dim=-1)
