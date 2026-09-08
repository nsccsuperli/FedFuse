#!/usr/bin/env python3
"""Evaluation: PSNR/SSIM/RMSE table on the Mayo test patients + figure export.

Usage:
  python eval.py --method redcnn --ckpt runs/redcnn_best.pt
  python eval.py --method fedfuse --ckpt runs/fedfuse_latest.pt --save-figs
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fedfuse.data import build_test_dataset, denormalize_hu, TEST_PATIENTS
from fedfuse.metrics import psnr_ssim_rmse
from fedfuse.models import FedFuseNet, LoRARouter
from fedfuse.models.baselines import REDCNN, UNetGen, DUGAN, CTformer
from fedfuse.physics import global_descriptor

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WIN_LO, WIN_HI = -160.0, 240.0


def load_method(args):
    if args.method == "redcnn":
        m = REDCNN().to(DEV)
        m.load_state_dict(torch.load(args.ckpt, map_location=DEV))
        return m, None
    if args.method == "wgan_vgg":
        m = UNetGen().to(DEV)
        m.load_state_dict(torch.load(args.ckpt, map_location=DEV))
        return m, None
    if args.method == "dugan":
        m = DUGAN().to(DEV)
        m.G.load_state_dict(torch.load(args.ckpt, map_location=DEV))
        return m, None
    if args.method == "ctformer":
        m = CTformer().to(DEV)
        m.load_state_dict(torch.load(args.ckpt, map_location=DEV))
        return m, "patch"
    if args.method in ("fedavg", "fedprox"):
        m = REDCNN().to(DEV)
        sd = torch.load(args.ckpt, map_location=DEV)
        m.load_state_dict(sd["model"] if "model" in sd else sd)
        return m, None
    if args.method == "fedfuse":
        m = FedFuseNet(args.dinov3_path, lora_r=args.lora_r,
                       n_experts=args.experts,
                       n_clients=args.clients or 8).to(DEV)
        router = LoRARouter(d_in=288, n_experts=args.experts).to(DEV)
        sd = torch.load(args.ckpt, map_location="cpu")
        # pick client: explicit --client-id, else the first one present
        if args.client_id is not None:
            cid = str(args.client_id)
        elif "personal" in sd and sd["personal"]:
            cid = sorted(sd["personal"])[0]
        elif "router" in sd and isinstance(sd["router"], dict) \
                and "net" not in sd["router"]:
            cid = sorted(sd["router"])[0]
        else:  # legacy single-router checkpoint
            cid = None
        if cid is not None:
            with torch.no_grad():
                for i, mod in enumerate(m.backbone.lora_modules()):
                    st = sd["personal"][cid].get(f"lora{i}")
                    if st is not None:
                        mod.A.copy_(st["A"])
                        mod.B.copy_(st["B"])
            if cid in sd.get("router", {}):
                router.load_state_dict(sd["router"][cid])
            loc = sd.get("local", {}).get(cid)
            if loc:
                with torch.no_grad():
                    for n, p in m.named_parameters():
                        if n in loc:
                            p.copy_(loc[n])
        return (m, router), "fedfuse"
    raise KeyError(args.method)


def save_png(arr01, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(arr01.shape[1] / 100, arr01.shape[0] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(arr01, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dinov3-path", default="models/dinov3")
    ap.add_argument("--data-root", default="/mnt/d/icassp/datas/mayo_2016_npy")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--experts", type=int, default=3)
    ap.add_argument("--clients", type=int, default=None)
    ap.add_argument("--client-id", type=int, default=None,
                    help="federated client to evaluate (fedfuse checkpoints)")
    ap.add_argument("--max-slices", type=int, default=None)
    ap.add_argument("--save-figs", action="store_true")
    ap.add_argument("--fig-dir", default="eval_figs")
    ap.add_argument("--manifest", default=None,
                    help="optional json manifest to restrict eval slices")
    args = ap.parse_args()

    model, mode = load_method(args)
    ds = build_test_dataset(args.data_root, max_slices=args.max_slices)
    ld = DataLoader(ds, batch_size=1, num_workers=2)
    os.makedirs(args.fig_dir, exist_ok=True)

    rows = []
    for i, (x, y) in enumerate(ld):
        x, y = x.to(DEV), y.to(DEV)
        with torch.no_grad():
            if mode == "patch":
                pred = model.forward_full(x)
            elif mode == "fedfuse":
                m, router = model
                g = global_descriptor(x).mean(0)     # route this slice
                m.set_pi(router(g[None])[0])
                pred = m(x)
            else:
                pred = model(x)
        pred_hu = denormalize_hu(pred[0, 0].cpu().numpy())
        tgt_hu = denormalize_hu(y[0, 0].cpu().numpy())
        body = tgt_hu > -500
        r = psnr_ssim_rmse(pred_hu, tgt_hu, body)
        rows.append(r)
        if args.save_figs:
            disp = lambda a: np.clip((a - WIN_LO) / (WIN_HI - WIN_LO), 0, 1)
            save_png(disp(pred_hu), os.path.join(args.fig_dir, f"{args.method}_{i:04d}_pred.png"))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ds)}")

    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    std = {k: float(np.std([r[k] for r in rows])) for k in rows[0]}
    print(f"\n{args.method}: PSNR {agg['psnr']:.2f}±{std['psnr']:.2f} | "
          f"SSIM {agg['ssim']:.4f}±{std['ssim']:.4f} | RMSE {agg['rmse']:.2f}±{std['rmse']:.2f} HU")
    with open(os.path.join(args.fig_dir, f"{args.method}_metrics.json"), "w") as f:
        json.dump(dict(mean=agg, std=std, n=len(rows)), f, indent=2)


if __name__ == "__main__":
    main()
