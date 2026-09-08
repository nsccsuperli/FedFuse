"""Same trained model, three metric conventions -> shows how much of the gap
vs ProFed/SCAN-PhysFed (PSNR 41-45 / SSIM 0.97) is pure eval-protocol.

Convention A (current repo):   display window [-160,240] HU, body-masked
Convention B (common in lit):  full range [-1024,3072] -> [0,1], body-masked
Convention C (common in lit2): [0,1] over full uint16 (0..4095 raw -> /4095)
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from fedfuse.data import MayoLDCTDataset, TEST_PATIENTS, HU_OFFSET
from fedfuse.metrics import _to01 as win_to01
from fedfuse.models import FedFuseNet

CKPT = "runs/train_verify_1k_v2/verify_best.pt"
DEV = "cuda"
W_LO, W_HI = -160.0, 240.0
F_LO, F_HI = -1024.0, 3072.0

m = FedFuseNet("models/dinov3", lora_r=8, n_experts=1).to(DEV)
m.load_state_dict(torch.load(CKPT, map_location=DEV))
m.eval()


def to01_full(hu):
    return np.clip((hu - F_LO) / (F_HI - F_LO), 0, 1)


def psnr_ssim(pred01, tgt01, mask):
    p, t = pred01[mask], tgt01[mask]
    mse = float(np.mean((p - t) ** 2))
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    pt = torch.from_numpy(pred01)[None, None].float()
    tt = torch.from_numpy(tgt01)[None, None].float()
    from fedfuse.losses import ssim as _ssim
    ss = float(_ssim(pt, tt))
    return psnr, ss


for pat in TEST_PATIENTS:
    ds = MayoLDCTDataset("/mnt/d/icassp/datas/mayo_2016_npy", [pat], max_slices=40)
    acc = {k: [] for k in ("A_psnr", "A_ssim", "B_psnr", "B_ssim", "C_psnr", "C_ssim")}
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=1, num_workers=2):
            x = x.to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = m(x)
            pred_hu = (pred[0, 0].float().cpu().numpy() * (F_HI - F_LO)) + F_LO
            tgt_hu = (y[0, 0].numpy() * (F_HI - F_LO)) + F_LO
            body = tgt_hu > -500
            # A: display window
            pa = np.clip((pred_hu - W_LO) / (W_HI - W_LO), 0, 1)
            ta = np.clip((tgt_hu - W_LO) / (W_HI - W_LO), 0, 1)
            p1, s1 = psnr_ssim(pa, ta, body)
            # B: full HU range
            pb, tb = to01_full(pred_hu), to01_full(tgt_hu)
            p2, s2 = psnr_ssim(pb, tb, body)
            # C: raw uint16 range (HU+1024)/4096
            pc = np.clip((pred_hu + 1024.0) / 4096.0, 0, 1)
            tc = np.clip((tgt_hu + 1024.0) / 4096.0, 0, 1)
            p3, s3 = psnr_ssim(pc, tc, body)
            acc["A_psnr"].append(p1); acc["A_ssim"].append(s1)
            acc["B_psnr"].append(p2); acc["B_ssim"].append(s2)
            acc["C_psnr"].append(p3); acc["C_ssim"].append(s3)
    print(f"\n{pat} (n={len(acc['A_psnr'])}):")
    print(f"  A window[-160,240] : PSNR {np.mean(acc['A_psnr']):6.2f}  SSIM {np.mean(acc['A_ssim']):.4f}")
    print(f"  B full [-1024,3072]: PSNR {np.mean(acc['B_psnr']):6.2f}  SSIM {np.mean(acc['B_ssim']):.4f}")
    print(f"  C full (HU+1024)/4096: PSNR {np.mean(acc['C_psnr']):6.2f}  SSIM {np.mean(acc['C_ssim']):.4f}")
