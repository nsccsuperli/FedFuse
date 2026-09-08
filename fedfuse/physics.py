"""Physics-driven multi-dimensional degradation representation (Sec. III-C).

All descriptors are computed from the low-dose image alone, on-GPU, and are
*statistics of the noise field, not of the anatomy*:

    r(x)  = x - GF(x)                 noise residual (guided filter)
    V(x)  sliding-window variance of r  -> local noise-strength map
    h(x)  normalized 32-bin intensity histogram over the diagnostic window
    NPS   radially averaged noise power spectrum of r over near-uniform patches

The global descriptor g = [h; NPS] in R^{32+256} feeds the MoE-LoRA router;
r and V(x) enter the image-domain branch as pixel-level conditioning.
"""
import numpy as np
import torch
import torch.nn.functional as F

HU_LO, HU_HI = -1024.0, 3072.0
DIAG_LO, DIAG_HI = -1024.0, 1024.0
BODY_TH_HU = -500.0


def x01_to_hu(x: torch.Tensor) -> torch.Tensor:
    return x * (HU_HI - HU_LO) + HU_LO


def guided_filter(x: torch.Tensor, r: int = 4, eps: float = 1e-3) -> torch.Tensor:
    """Edge-preserving guided filter, guidance = input itself. (B,C,H,W)."""
    k = 2 * r + 1
    box = lambda t: F.avg_pool2d(t, k, stride=1, padding=r)
    mean_x = box(x)
    var = box(x * x) - mean_x ** 2
    a = var / (var + eps)
    b = mean_x - a * mean_x
    return box(a) * x + box(b)


def noise_residual(x: torch.Tensor) -> torch.Tensor:
    return x - guided_filter(x)


def local_variance_map(residual: torch.Tensor, win: int = 15) -> torch.Tensor:
    m1 = F.avg_pool2d(residual, win, stride=1, padding=win // 2)
    m2 = F.avg_pool2d(residual ** 2, win, stride=1, padding=win // 2)
    return (m2 - m1 ** 2).clamp_min(0)


def body_mask(x: torch.Tensor) -> torch.Tensor:
    return x01_to_hu(x) > BODY_TH_HU


@torch.no_grad()
def intensity_histogram(x: torch.Tensor, bins: int = 32,
                        lo: float = DIAG_LO, hi: float = DIAG_HI) -> torch.Tensor:
    """Normalized histogram over the diagnostic HU window, body region. (B,bins)"""
    hu = x01_to_hu(x)
    mask = body_mask(x)
    edges = torch.linspace(lo, hi, bins + 1, device=x.device)
    outs = []
    for b in range(hu.shape[0]):
        vals = hu[b][mask[b]]
        h = torch.histc(vals, bins=bins, min=lo, max=hi)
        outs.append(h / h.sum().clamp_min(1e-8))
    return torch.stack(outs)


@torch.no_grad()
def nps_profile(residual: torch.Tensor, x: torch.Tensor, patch: int = 96,
                n_bins: int = 256, max_patches: int = 8) -> torch.Tensor:
    """Radial NPS profile (B,n_bins). Near-uniform body patches, Hann window."""
    B, _, H, W = residual.shape
    device = residual.device
    gy, gx = torch.gradient(residual[:, 0], dim=(1, 2))
    gmag = torch.hypot(gx, gy)
    mask = body_mask(x)[:, 0].float()

    def box(t):
        return F.avg_pool2d(t[:, None], patch, stride=1, padding=0)[:, 0]

    # candidate patch scores: body coverage high, gradient low
    body_frac = box(mask)
    grad_mean = box(gmag)
    step = patch // 2
    ys = list(range(0, H - patch, step))
    xs = list(range(0, W - patch, step))
    win = torch.outer(torch.hamming_window(patch, device=device),
                      torch.hamming_window(patch, device=device))
    freqs = torch.fft.fftfreq(patch, device=device)
    fy, fx = torch.meshgrid(freqs, freqs, indexing="ij")
    rho = torch.hypot(fx, fy)
    rbins = torch.linspace(0, rho.max().item(), n_bins + 1, device=device)
    ridx = torch.bucketize(rho, rbins) - 1

    outs = []
    for b in range(B):
        cands = []
        for yy in ys:
            for xx in xs:
                if body_frac[b, yy, xx] > 0.98:
                    cands.append((grad_mean[b, yy, xx].item(), yy, xx))
        cands.sort(key=lambda t: t[0])
        acc, cnt = None, 0
        for _, yy, xx in cands[:max_patches]:
            p = residual[b, 0, yy:yy + patch, xx:xx + patch]
            p = (p - p.mean()) * win
            ps = torch.fft.fft2(p).abs() ** 2
            prof = torch.zeros(n_bins, device=device)
            cnt_per = torch.zeros(n_bins, device=device)
            prof.scatter_add_(0, ridx.flatten().clamp(0, n_bins - 1), ps.flatten())
            cnt_per.scatter_add_(0, ridx.flatten().clamp(0, n_bins - 1),
                                 torch.ones_like(ps.flatten()))
            prof = prof / cnt_per.clamp_min(1)
            acc = prof if acc is None else acc + prof
            cnt += 1
        if acc is None:
            acc = torch.zeros(n_bins, device=device)
        else:
            acc = acc / max(cnt, 1)
        acc = acc / acc.max().clamp_min(1e-12)          # normalize for scale invariance
        outs.append(acc)
    return torch.stack(outs)


@torch.no_grad()
def global_descriptor(x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
    """g = [h(x); NPS] in R^{32+256} per image. (B, 288)"""
    residual = noise_residual(x) if residual is None else residual
    h = intensity_histogram(x)
    nps = nps_profile(residual, x)
    return torch.cat([h, nps], dim=1)
