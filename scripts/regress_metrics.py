"""Regression: eval.py path with FIXED metrics (full-range) on best ckpt.
Expect L310 ~45.6/0.984, L506 ~45.4/0.986 (matches metric_conventions B)."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from fedfuse.data import MayoLDCTDataset, TEST_PATIENTS
from fedfuse.metrics import psnr_ssim_rmse
from fedfuse.models import FedFuseNet

m = FedFuseNet("models/dinov3", lora_r=8, n_experts=1).to("cuda")
m.load_state_dict(torch.load("runs/train_verify_1k_v2/verify_best.pt", map_location="cuda"))
m.eval()
for pat in TEST_PATIENTS:
    ds = MayoLDCTDataset("/mnt/d/icassp/datas/mayo_2016_npy", [pat], max_slices=40)
    rows = []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=1, num_workers=2):
            x = x.to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = m(x)
            ph = (pred[0, 0].float().cpu().numpy() * 4096.0) - 1024.0
            th = (y[0, 0].numpy() * 4096.0) - 1024.0
            rows.append(psnr_ssim_rmse(ph, th, th > -500))
    r = {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}
    print(f"{pat}: PSNR {r['psnr']:.2f}  SSIM {r['ssim']:.4f}  RMSE {r['rmse']:.2f} HU")
