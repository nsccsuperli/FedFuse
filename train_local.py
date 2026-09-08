#!/usr/bin/env python3
"""Local single-center training of FedFuse T1+T2 (dual-path fusion + LoRA).

Scope (manuscript Sec. III-B/III-C only, NO federated machinery, NO MoE
router): freeze DINOv3 ViT-L, adapt q/v with rank-r LoRA (n_experts=1 ->
standard LoRA, A frozen / B trained, uniform pi), run the dual-path fusion
network (semantic backbone + noise branch on [x, r, V(x)] + gated fusion +
residual decoder), train with L1+SSIM+edge on local Mayo data.

Usage:
  python train_local.py --smoke                  # tiny closed-loop check
  python train_local.py --epochs 30 --batch 2    # real training
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from fedfuse.data import MayoLDCTDataset, TRAIN_PATIENTS, build_test_dataset
from fedfuse.losses import CompositeLoss
from fedfuse.metrics import psnr_ssim_rmse
from fedfuse.models import FedFuseNet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WIN_LO, WIN_HI = -160.0, 240.0


def evaluate(model, loader):
    """Body-masked PSNR/SSIM on full slices (HU domain)."""
    model.eval()
    rows = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                pred = model(x)
            pred_hu = (pred[0, 0].float().cpu().numpy()
                       * (3072.0 - (-1024.0)) + (-1024.0))
            tgt_hu = (y[0, 0].cpu().numpy() * (3072.0 - (-1024.0)) + (-1024.0))
            body = tgt_hu > -500
            rows.append(psnr_ssim_rmse(pred_hu, tgt_hu, body))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/mnt/d/icassp/datas/mayo_2016_npy")
    ap.add_argument("--dinov3-path", default="models/dinov3")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--max-slices", type=int, default=None,
                    help="limit training slices (smoke / quick check)")
    ap.add_argument("--val-slices", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/train_local")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    max_tr = 12 if args.smoke else args.max_slices
    ds = MayoLDCTDataset(args.data_root, TRAIN_PATIENTS,
                         max_slices=max_tr, patch_size=args.patch)
    n_val = min(args.val_slices, len(ds) // 10)
    n_train = len(ds) - n_val
    tr, va = random_split(ds, [n_train, n_val],
                          generator=torch.Generator().manual_seed(0))
    if args.patch is not None:   # validation on full slices
        va.dataset = MayoLDCTDataset(args.data_root, TRAIN_PATIENTS,
                                     max_slices=max_tr, patch_size=None)
        va.indices = list(range(n_train, n_train + n_val))
    train_ld = DataLoader(tr, batch_size=args.batch, shuffle=True,
                          num_workers=2, drop_last=True)
    val_ld = DataLoader(va, batch_size=1, num_workers=2)

    model = FedFuseNet(args.dinov3_path, lora_r=args.lora_r,
                       n_experts=1).to(DEV)      # E=1: plain LoRA, no router
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params: total {n_all/1e6:.2f}M, trainable {n_tr/1e6:.3f}M "
          f"({100*n_tr/n_all:.2f}%)  dev={DEV}")
    print(f"train={n_train} val={n_val} patch={args.patch} batch={args.batch}")

    # trainable: LoRA B (all q/v modules) + local CNN/fusion/decoder
    trainable = ([m.B for m in model.backbone.lora_modules()]
                 + [p for n, p in model.named_parameters()
                     if n.startswith(("noise_branch", "fusions", "decoder"))])
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = CompositeLoss()

    best = -1.0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for x, y in train_ld:
            x, y = x.to(DEV), y.to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                pred = model(x)
                loss, parts = loss_fn(pred.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
        sched.step()
        m = evaluate(model, val_ld)
        print(f"[local] ep{ep:02d} loss={ep_loss/max(nb,1):.4f} "
              f"val_psnr={m['psnr']:.2f} val_ssim={m['ssim']:.4f} "
              f"({(time.time()-t0)/60:.1f} min)")
        if m["psnr"] > best:
            best = m["psnr"]
            torch.save(model.state_dict(), os.path.join(args.out, "fedfuse_local_best.pt"))
    torch.save(dict(model=model.state_dict(), args=vars(args)),
               os.path.join(args.out, "fedfuse_local_last.pt"))
    with open(os.path.join(args.out, "train_local_history.json"), "w") as f:
        json.dump(dict(best_psnr=best), f)
    print(f"done in {(time.time()-t0)/60:.1f} min, best_val_psnr={best:.2f}")


if __name__ == "__main__":
    main()
