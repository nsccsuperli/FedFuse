"""Training losses: L1 + SSIM + edge (manuscript, Fig. 1).

Numerical-safety note (root cause of the FedFuse NaN bug, 2026-09):
ssim/edge contain catastrophic-cancellation terms -- f(x*x) - mx**2 -- and
divisions by near-zero denominators. If these convs run under bf16 autocast
(which casts even .float() conv inputs to bf16), ~3 significant digits are
lost, sxx/syy go wrong/negative, gradients explode to NaN and poison every
parameter on the backward chain. Loss must therefore ALWAYS compute in fp32:
every entry point here disables autocast explicitly, regardless of what the
training loop does around it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_FP32 = lambda: torch.autocast("cuda", enabled=False)


def _conv2d_fp32(t, kern, padding=0, groups=1):
    """conv2d guaranteed fp32 (autocast would downcast a .float() input)."""
    with _FP32():
        return F.conv2d(t.float(), kern.float(), padding=padding, groups=groups)


def ssim_map(x, y, win: int = 11, sigma: float = 1.5, data_range: float = 1.0):
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    g = torch.exp(-0.5 * ((torch.arange(win, device=x.device) - win // 2) / sigma) ** 2)
    g = (g / g.sum())
    kern = (g[:, None] * g[None, :]).expand(x.shape[1], 1, win, win)
    mx, my = _conv2d_fp32(x, kern), _conv2d_fp32(y, kern)
    sxx, syy = _conv2d_fp32(x * x, kern) - mx ** 2, _conv2d_fp32(y * y, kern) - my ** 2
    sxy = _conv2d_fp32(x * y, kern) - mx * my
    num = (2 * mx * my + c1) * (2 * sxy + c2)
    den = (mx ** 2 + my ** 2 + c1) * (sxx + syy + c2)
    return num / den.clamp_min(1e-8)


def ssim(x, y, **kw) -> torch.Tensor:
    return ssim_map(x, y, **kw).mean()


def sobel(t: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=t.device)
    ky = kx.t()
    k = torch.stack([kx, ky])[:, None].expand(2, t.shape[1], 3, 3).reshape(2 * t.shape[1], t.shape[1], 3, 3)
    g = _conv2d_fp32(t, k, padding=1)
    g = g.reshape(t.shape[0], 2, t.shape[1], *t.shape[2:]).pow(2).sum(1, keepdim=True).sqrt()
    return g


class CompositeLoss(nn.Module):
    def __init__(self, lambda_ssim: float = 0.1, lambda_edge: float = 0.05):
        super().__init__()
        self.l_ssim, self.l_edge = lambda_ssim, lambda_edge

    def forward(self, pred, target):
        with _FP32():
            p, t = pred.float(), target.float()
            l1 = F.l1_loss(p, t)
            l_s = 1.0 - ssim(p, t)
            l_e = F.l1_loss(sobel(p), sobel(t))
            total = l1 + self.l_ssim * l_s + self.l_edge * l_e
            parts = dict(l1=l1.item(), ssim=l_s.item(), edge=l_e.item())
        return total, parts


def router_balance_loss(pi: torch.Tensor) -> torch.Tensor:
    """Entropy regularizer preventing expert collapse (Eq. 12). pi: (E,) or (B,E)."""
    p = pi.mean(dim=0) if pi.dim() > 1 else pi
    return (p * torch.log(p.clamp_min(1e-8))).sum()
