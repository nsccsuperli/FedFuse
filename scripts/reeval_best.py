"""Re-evaluate a verify_best.pt checkpoint on held-out TEST slices,
per patient (L310 AND L506) so both patients are covered fairly.
"""
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from fedfuse.data import MayoLDCTDataset, TEST_PATIENTS
from fedfuse.metrics import psnr_ssim_rmse
from fedfuse.models import FedFuseNet

ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/train_verify_1k/verify_best.pt"
per_patient = int(sys.argv[2]) if len(sys.argv) > 2 else 100
DEV = "cuda"
m = FedFuseNet("models/dinov3", lora_r=8, n_experts=1).to(DEV)
m.load_state_dict(torch.load(ckpt, map_location=DEV))
m.eval()

for p in TEST_PATIENTS:
    ds = MayoLDCTDataset("/mnt/d/icassp/datas/mayo_2016_npy", [p],
                         max_slices=per_patient)
    ps, ss = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=1, num_workers=2):
            x = x.to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = m(x)
            ph = (pred[0, 0].float().cpu().numpy() * 4096) - 1024
            th = (y[0, 0].numpy() * 4096) - 1024
            r = psnr_ssim_rmse(ph, th, th > -500)
            ps.append(r["psnr"]); ss.append(r["ssim"])
    ps, ss = np.array(ps), np.array(ss)
    print(f"{p} n={len(ps)} PSNR={ps.mean():.3f}+/-{ps.std():.3f}  "
          f"SSIM={ss.mean():.4f}+/-{ss.std():.4f}")
