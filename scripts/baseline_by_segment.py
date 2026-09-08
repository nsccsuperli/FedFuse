"""Input LDCT baseline PSNR/SSIM by slice segment (L310 vs L506, no GPU).
Determines whether eval-set slice position/patient difficulty explains the
40-slice (0.819) vs 200-slice (0.738) SSIM gap of the trained model.
"""
import re
import numpy as np
import torch
from torch.utils.data import DataLoader

from fedfuse.data import MayoLDCTDataset, TEST_PATIENTS
from fedfuse.metrics import psnr_ssim_rmse


def baseline_for(items, tag):
    ps, ss = [], []
    for a, b in items:
        ld_u = np.load(a)
        nd_u = np.load(b)
        ld = (ld_u.astype(np.float32) - 1024.0 - (-1024.0)) / 4096.0
        nd = (nd_u.astype(np.float32) - 1024.0 - (-1024.0)) / 4096.0
        pred_hu = ld * 4096.0 - 1024.0
        tgt_hu = nd * 4096.0 - 1024.0
        r = psnr_ssim_rmse(pred_hu, tgt_hu, tgt_hu > -500)
        ps.append(r["psnr"]); ss.append(r["ssim"])
    ps, ss = np.array(ps), np.array(ss)
    print(f"{tag}: n={len(ps)} input_PSNR={ps.mean():.2f}+/-{ps.std():.2f} "
          f"input_SSIM={ss.mean():.4f}+/-{ss.std():.4f}")
    return ps, ss


ds = MayoLDCTDataset("/mnt/d/icassp/datas/mayo_2016_npy", TEST_PATIENTS,
                     max_slices=1059, preload=False)
by_pat = {}
for a, b in ds.items:
    m = re.search(r"(L\d+)_(\d+)_(\d+)_img", a)
    by_pat.setdefault(m.group(1), []).append((int(m.group(3)), a, b))
for p, v in by_pat.items():
    v.sort()

# L310 segments: 0-39 (what training eval used), 0-199, 40-199, 200-532
l310 = [(a, b) for _, a, b in by_pat.get("L310", [])]
l506 = [(a, b) for _, a, b in by_pat.get("L506", [])]
baseline_for(l310[:40], "L310 slice0-39    ")
baseline_for(l310[40:80], "L310 slice40-79   ")
baseline_for(l310[200:240], "L310 slice200-239 ")
baseline_for(l506[:40], "L506 slice0-39    ")

