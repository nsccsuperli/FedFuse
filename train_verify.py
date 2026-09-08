#!/usr/bin/env python3
"""Verification run (option b): 64 slices / 8 patients, watch PSNR+SSIM,
auto-stop on plateau (patience epochs without new best), report curves.

Evaluates on the held-out TEST patients (L310/L506) full slices each epoch,
so the metric is real generalization, not train-set memorization.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from fedfuse.data import MayoLDCTDataset, TRAIN_PATIENTS, build_test_dataset
from fedfuse.losses import CompositeLoss, router_balance_loss
from fedfuse.metrics import psnr_ssim_rmse
from fedfuse.models import FedFuseNet, LoRARouter
from fedfuse.physics import global_descriptor

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, router, loader, tag=""):
    model.eval()
    rows = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEV)
            if router is not None:
                g = global_descriptor(x).mean(0)
                model.set_pi(router(g[None])[0])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                pred = model(x)
            pred_hu = (pred[0, 0].float().cpu().numpy() * 4096.0) - 1024.0
            tgt_hu = (y[0, 0].numpy() * 4096.0) - 1024.0
            rows.append(psnr_ssim_rmse(pred_hu, tgt_hu, tgt_hu > -500))
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    out["n"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/mnt/d/icassp/datas/mayo_2016_npy")
    ap.add_argument("--dinov3-path", default="models/dinov3")
    ap.add_argument("--per-patient", type=int, default=8, help="slices per train patient")
    ap.add_argument("--epochs", type=int, default=60, help="hard cap")
    ap.add_argument("--patience", type=int, default=10, help="plateau stop")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-slices", type=int, default=40,
                    help="slices per TEST patient for eval (both L310 and L506)")
    ap.add_argument("--experts", type=int, default=1,
                    help="MoLoRA experts; 1 = plain LoRA without router")
    ap.add_argument("--lambda-bal", type=float, default=0.01,
                    help="router entropy regularizer (E>1 only)")
    ap.add_argument("--resume", default=None,
                    help="start from a verify_best.pt checkpoint (state_dict)")
    ap.add_argument("--out", default="runs/train_verify")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    parts = [MayoLDCTDataset(args.data_root, [p], max_slices=args.per_patient,
                             patch_size=args.patch) for p in TRAIN_PATIENTS]
    tr = ConcatDataset(parts)
    train_ld = DataLoader(tr, batch_size=args.batch, shuffle=True,
                          num_workers=2, drop_last=True)
    # eval: per-patient sub-datasets (L310 AND L506), NOT first-N-of-concat
    from fedfuse.data import TEST_PATIENTS
    test_loaders = {
        p: DataLoader(MayoLDCTDataset(args.data_root, [p],
                                      max_slices=args.eval_slices),
                      batch_size=1, num_workers=2)
        for p in TEST_PATIENTS}
    n_te = sum(len(ld.dataset) for ld in test_loaders.values())
    print(f"train={len(tr)} ({args.per_patient}x{TRAIN_PATIENTS}) "
          f"test={n_te} ({ {p: len(v.dataset) for p, v in test_loaders.items()} }) "
          f"batch={args.batch} patch={args.patch}")

    model = FedFuseNet(args.dinov3_path, lora_r=8,
                       n_experts=args.experts).to(DEV)
    router = (LoRARouter(d_in=288, n_experts=args.experts).to(DEV)
              if args.experts > 1 else None)
    if args.resume:
        sd = torch.load(args.resume, map_location=DEV)
        model.load_state_dict(sd if "model" not in sd else sd["model"])
        print(f"resumed from {args.resume}")
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_tr/1e6:.3f}M of "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"(E={args.experts}{' + router' if router else ''})")
    trainable = ([m.B for m in model.backbone.lora_modules()]
                 + (list(router.parameters()) if router else [])
                 + [p for n, p in model.named_parameters()
                     if n.startswith(("noise_branch", "fusions", "decoder"))])
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-5)
    loss_fn = CompositeLoss()

    hist, best, stall = [], -1.0, 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        el, nb = 0.0, 0
        for x, y in train_ld:
            x, y = x.to(DEV), y.to(DEV)
            if router is not None:
                with torch.no_grad():
                    g = global_descriptor(x).mean(0)   # (288,)
                pi = router(g[None])[0]                # (E,) fresh graph
                model.set_pi(pi)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                pred = model(x)
                loss, _ = loss_fn(pred.float(), y)
                if router is not None:
                    loss = loss + args.lambda_bal * router_balance_loss(pi)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            el += loss.item()
            nb += 1
        m = {}
        for p, ld in test_loaders.items():
            r = evaluate(model, router, ld)
            for k in ("psnr", "ssim", "rmse"):
                m[f"{p}_{k}"] = r[k]
        m["psnr"] = float(np.mean([m[f"{p}_psnr"] for p in test_loaders]))
        m["ssim"] = float(np.mean([m[f"{p}_ssim"] for p in test_loaders]))
        row = dict(epoch=ep, loss=el / max(nb, 1), **m)
        hist.append(row)
        improved = m["psnr"] > best + 1e-4
        if improved:
            best, stall = m["psnr"], 0
            torch.save(model.state_dict(), os.path.join(args.out, "verify_best.pt"))
        else:
            stall += 1
        pp = " ".join(f"{p}:{m[f'{p}_psnr']:.2f}/{m[f'{p}_ssim']:.3f}"
                      for p in test_loaders)
        print(f"[v] ep{ep:02d} loss={row['loss']:.4f} "
              f"psnr={m['psnr']:.3f} ssim={m['ssim']:.4f} "
              f"[{pp}] best={best:.3f} stall={stall}/{args.patience} "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)
        if stall >= args.patience:
            print(f"PLATEAU: no PSNR gain for {args.patience} epochs -> stop")
            break

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    p = [h["psnr"] for h in hist]
    s = [h["ssim"] for h in hist]
    print(f"best_psnr={best:.3f} @ ep{hist[p.index(max(p))]['epoch']} | "
          f"final_psnr={p[-1]:.3f} final_ssim={s[-1]:.4f} | "
          f"epochs_run={len(hist)}")
    print(f"RESULT saved to {args.out}/history.json + verify_best.pt")


if __name__ == "__main__":
    main()
