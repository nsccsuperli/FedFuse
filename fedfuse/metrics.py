"""Evaluation metrics for LDCT enhancement.

Convention (2026-09-03, aligned with ProFed/SCAN-PhysFed/FedFDD et al.):
PSNR/SSIM are computed on images normalized over the FULL HU dynamic range
[-1024, 3072] -> [0,1], body-masked. The abdominal display window [-160,240]
is ONLY for visualization -- using it as the PSNR normalization range
inflates errors ~10x (400 vs 4096 HU) and makes numbers ~20 dB lower than
every federated-LDCT paper, which was a real comparability bug.

SSIM here returns a fraction in [0,1]; papers often print it as percent
(0.976 == 97.6%).
"""
import numpy as np

from .losses import ssim as _ssim
import torch

HU_LO, HU_HI = -1024.0, 3072.0      # full range -> [0,1]
DISP_LO, DISP_HI = -160.0, 240.0     # display window (visualization only)


def _to01_full(x_hu):
    return np.clip((x_hu - HU_LO) / (HU_HI - HU_LO), 0.0, 1.0)


def psnr_ssim_rmse(pred_hu: np.ndarray, tgt_hu: np.ndarray, body: np.ndarray | None = None):
    """PSNR (dB), SSIM, RMSE (HU). Full-range normalization, body-masked.

    All inputs in HU. body mask (bool array) restricts PSNR/RMSE to the
    body region (convention: tgt_hu > -500).
    """
    m = np.ones_like(tgt_hu, bool) if body is None else body
    p, t = _to01_full(pred_hu)[m], _to01_full(tgt_hu)[m]
    mse = float(np.mean((p - t) ** 2))
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    rmse_hu = float(np.sqrt(np.mean((pred_hu[m] - tgt_hu[m]) ** 2)))
    pt = torch.from_numpy(_to01_full(pred_hu))[None, None].float()
    tt = torch.from_numpy(_to01_full(tgt_hu))[None, None].float()
    ss = float(_ssim(pt, tt))
    return dict(psnr=psnr, ssim=ss, rmse=rmse_hu)


def psnr_ssim_rmse_window(pred_hu, tgt_hu, body=None, lo=DISP_LO, hi=DISP_HI):
    """Legacy display-window variant, kept for figure-side reporting only."""
    m = np.ones_like(tgt_hu, bool) if body is None else body
    p = np.clip((pred_hu - lo) / (hi - lo), 0.0, 1.0)[m]
    t = np.clip((tgt_hu - lo) / (hi - lo), 0.0, 1.0)[m]
    mse = float(np.mean((p - t) ** 2))
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    rmse_hu = float(np.sqrt(np.mean((pred_hu[m] - tgt_hu[m]) ** 2)))
    return dict(psnr=psnr, ssim=0.0, rmse=rmse_hu)
