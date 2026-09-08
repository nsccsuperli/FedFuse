#!/usr/bin/env python3
"""Centralized training for baseline methods (RED-CNN / WGAN-VGG / DUGAN / CTformer).

Usage:
  python train_central.py --method redcnn --epochs 60 --batch 8
  python train_central.py --method ctformer --epochs 40 --batch 16   # patch-based
  python train_central.py --method wgan_vgg --epochs 60 --batch 4 --smoke
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from fedfuse.data import MayoLDCTDataset, TRAIN_PATIENTS, TEST_PATIENTS
from fedfuse.losses import CompositeLoss, ssim
from fedfuse.models.baselines import (REDCNN, UNetGen, PatchCritic, VGGPerceptual,
                                      gradient_penalty, DUGAN, CTformer)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, loader, patch_mode=False):
    model.eval()
    ps = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            pred = model.forward_full(x) if patch_mode else model(x)
            mse = F.mse_loss(pred, y).item()
            ps.append(10 * np.log10(1.0 / max(mse, 1e-12)))
    return float(np.mean(ps))


def train_redcnn(args, train_ld, val_ld):
    model = REDCNN().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = CompositeLoss(0.0, 0.0) if args.pure_l1 else CompositeLoss(0.1, 0.05)
    best = -1
    for ep in range(args.epochs):
        model.train()
        for x, y in train_ld:
            x, y = x.to(DEV), y.to(DEV)
            loss, _ = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        psnr = evaluate(model, val_ld)
        print(f"[redcnn] ep{ep:03d} val_psnr={psnr:.2f}")
        if psnr > best:
            best = psnr
            torch.save(model.state_dict(), os.path.join(args.out, "redcnn_best.pt"))
    return model


def train_ctformer(args, train_ds, val_ld):
    model = CTformer().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    p = model.patch
    best = -1
    for ep in range(args.epochs):
        model.train()
        n = len(train_ds)
        idxs = np.random.permutation(n)
        for bi in range(0, n - args.batch, args.batch):
            xs, ys = [], []
            for i in idxs[bi:bi + args.batch]:
                x, y = train_ds[i]
                H, W = x.shape[-2:]
                yy, xx = np.random.randint(0, H - p), np.random.randint(0, W - p)
                xs.append(x[None, :, yy:yy + p, xx:xx + p])
                ys.append(y[None, :, yy:yy + p, xx:xx + p])
            x = torch.cat(xs).to(DEV)
            y = torch.cat(ys).to(DEV)
            pred = model(x)
            loss = F.l1_loss(pred, y) + 0.1 * (1 - ssim(pred, y))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        psnr = evaluate(model, val_ld, patch_mode=True)
        print(f"[ctformer] ep{ep:03d} val_psnr={psnr:.2f}")
        if psnr > best:
            best = psnr
            torch.save(model.state_dict(), os.path.join(args.out, "ctformer_best.pt"))
    return model


def train_wgan_vgg(args, train_ld, val_ld):
    G = UNetGen().to(DEV)
    D = PatchCritic().to(DEV)
    vgg = VGGPerceptual().to(DEV)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))
    best = -1
    for ep in range(args.epochs):
        G.train()
        for x, y in train_ld:
            x, y = x.to(DEV), y.to(DEV)
            for _ in range(2):
                fake = G(x).detach()
                d_loss = D(fake).mean() - D(y).mean() + 10 * gradient_penalty(D, y, fake)
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()
            fake = G(x)
            g_loss = -D(fake).mean() + 0.1 * vgg(fake, y) + F.l1_loss(fake, y)
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
        psnr = evaluate(G, val_ld)
        print(f"[wgan_vgg] ep{ep:03d} val_psnr={psnr:.2f} d={d_loss.item():.3f} g={g_loss.item():.3f}")
        if psnr > best:
            best = psnr
            torch.save(G.state_dict(), os.path.join(args.out, "wgan_vgg_best.pt"))
    return G


def train_dugan(args, train_ld, val_ld):
    net = DUGAN().to(DEV)
    G, Di, Dg = net.G, net.D_img, net.D_grad
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(list(Di.parameters()) + list(Dg.parameters()),
                             lr=args.lr, betas=(0.5, 0.9))
    bce = torch.nn.BCEWithLogitsLoss()
    best = -1
    for ep in range(args.epochs):
        net.train()
        for x, y in train_ld:
            x, y = x.to(DEV), y.to(DEV)
            fake = G(x).detach()
            di_real, di_fake = Di(y), Di(fake)
            dg_real, dg_fake = Dg(DUGAN.grad_pair(y)), Dg(DUGAN.grad_pair(fake))
            d_loss = 0.5 * (bce(di_real, torch.ones_like(di_real))
                            + bce(di_fake, torch.zeros_like(di_fake))
                            + bce(dg_real, torch.ones_like(dg_real))
                            + bce(dg_fake, torch.zeros_like(dg_fake)))
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            fake = G(x)
            g_adv = 0.5 * (bce(Di(fake), torch.ones_like(di_real))
                           + bce(Dg(DUGAN.grad_pair(fake)), torch.ones_like(dg_real)))
            g_loss = g_adv + 50 * F.mse_loss(fake, y) + 10 * DUGAN.gradient_loss(fake, y)
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
        psnr = evaluate(net, val_ld)
        print(f"[dugan] ep{ep:03d} val_psnr={psnr:.2f}")
        if psnr > best:
            best = psnr
            torch.save(G.state_dict(), os.path.join(args.out, "dugan_best.pt"))
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["redcnn", "wgan_vgg", "dugan", "ctformer"])
    ap.add_argument("--data-root", default="/mnt/d/icassp/datas/mayo_2016_npy")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patch", type=int, default=128, help="training crop size")
    ap.add_argument("--pure-l1", action="store_true")
    ap.add_argument("--val-slices", type=int, default=60)
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify code")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    max_tr = 64 if args.smoke else None
    patch = None if args.method == "ctformer" else args.patch
    ds = MayoLDCTDataset(args.data_root, TRAIN_PATIENTS, max_slices=max_tr,
                         patch_size=patch)
    n_val = min(args.val_slices, len(ds) // 10)
    n_train = len(ds) - n_val
    tr, va = random_split(ds, [n_train, n_val],
                          generator=torch.Generator().manual_seed(0))
    if patch is not None:
        va.dataset = MayoLDCTDataset(args.data_root, TRAIN_PATIENTS,
                                     max_slices=max_tr, patch_size=None)
        va.indices = list(range(n_train, n_train + n_val))
    train_ld = DataLoader(tr, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_ld = DataLoader(va, batch_size=args.batch, num_workers=args.workers)
    print(f"train={n_train} val={n_val} dev={DEV} patch={patch}")

    t0 = time.time()
    dict(redcnn=lambda: train_redcnn(args, train_ld, val_ld),
         wgan_vgg=lambda: train_wgan_vgg(args, train_ld, val_ld),
         dugan=lambda: train_dugan(args, train_ld, val_ld),
         ctformer=lambda: train_ctformer(args, ds, val_ld))[args.method]()
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
