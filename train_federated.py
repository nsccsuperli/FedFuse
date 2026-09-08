#!/usr/bin/env python3
"""Federated training: FedFuse stacking aggregation + global anchor (paper).

Protocol (Sec. III-D, updated):
  - K=8 clients, one Mayo train patient each, per-client image-domain
    degradation spec (dose/kernel ladder, no CTLib projection sim).
  - Client keeps personalized LoRA factors {A_e,B_e} + router + local
    CNN modules across rounds; server broadcasts the FROZEN global anchor
    (C, D) = stacking of all clients' weighted factors each round.
  - Metrics: full-range PSNR/SSIM, body-masked, PER TEST PATIENT (L310,
    L506). Evaluation is per-client: load client k's personalized factors
    and route test slices with its own g_k (client-personalized model).

Usage:
  python train_federated.py --method fedfuse --rounds 3 --smoke
  python train_federated.py --method fedfuse --rounds 50 --local-epochs 2
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fedfuse.data import (MayoLDCTDataset, TEST_PATIENTS, build_client_datasets,
                          build_test_dataset, client8_patients)
from fedfuse.federated import FedFuseClient, FedFuseServer, fedavg_aggregate
from fedfuse.losses import CompositeLoss
from fedfuse.metrics import psnr_ssim_rmse
from fedfuse.models import FedFuseNet, LoRARouter
from fedfuse.models.baselines import REDCNN

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_client_model(model, router, g_k, loader):
    """Evaluate client k's personalized model on a loader (per-patient)."""
    model.eval()
    router.eval()
    rows = []
    with torch.no_grad():
        pi = router(g_k[None].to(DEV))[0]
        model.set_pi(pi)
        for x, y in loader:
            x = x.to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                pred = model(x)
            pred_hu = (pred[0, 0].float().cpu().numpy() * 4096.0) - 1024.0
            tgt_hu = (y[0, 0].numpy() * 4096.0) - 1024.0
            rows.append(psnr_ssim_rmse(pred_hu, tgt_hu, tgt_hu > -500))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def restore_client_into(model, client):
    """Load client's personalized A/B, local CNN and router state into model."""
    with torch.no_grad():
        if client.personal_state is not None:
            for i, m in enumerate(model.backbone.lora_modules()):
                st = client.personal_state.get(f"lora{i}")
                if st is not None:
                    m.A.copy_(st["A"].to(DEV))
                    m.B.copy_(st["B"].to(DEV))
        if client.local_state is not None:
            for n, p in model.named_parameters():
                if n in client.local_state:
                    p.copy_(client.local_state[n].to(DEV))


def train_fedfuse(args):
    clients_meta = build_client_datasets(
        args.data_root, n_clients=args.clients,
        max_slices_per_client=args.max_slices if args.smoke else None,
        patch_size=args.patch, patients_per_client=client8_patients())
    specs = [c["spec"].name for c in clients_meta]
    n_per = [len(c["dataset"]) for c in clients_meta]
    print(f"clients={args.clients} per-client-slices={n_per}")
    print(f"degradation specs = {specs}")

    model = FedFuseNet(args.dinov3_path, lora_r=args.lora_r,
                       n_experts=args.experts, n_clients=args.clients).to(DEV)
    router = LoRARouter(d_in=288, n_experts=args.experts).to(DEV)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_tr/1e6:.3f}M of "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"(E={args.experts} K={args.clients})")
    model.zero_anchors()          # round 0: no global branch yet

    server = FedFuseServer(len(model.backbone.lora_modules()),
                           args.experts, args.lora_r,
                           model.backbone.embed_dim, model.backbone.embed_dim)

    clients = [FedFuseClient(m["cid"], m["dataset"], DEV, batch_size=args.batch,
                             local_epochs=args.local_epochs,
                             lambda_bal=args.lambda_bal, seed=m["cid"])
               for m in clients_meta]

    # per-client test loaders (each test patient separately)
    test_loaders = {
        p: DataLoader(MayoLDCTDataset(args.data_root, [p],
                                      max_slices=args.test_slices),
                      batch_size=1, num_workers=2)
        for p in TEST_PATIENTS}

    start_rnd, history = 0, []
    if args.resume:
        sd = torch.load(args.resume, map_location="cpu", weights_only=False)
        for c in clients:
            cid = c.cid
            c.personal_state = sd["personal"].get(cid)
            c.local_state = sd["local"].get(cid)
            c.router_state = sd["router"].get(cid)
        if sd.get("C") is not None:
            for i, m in enumerate(model.backbone.lora_modules()):
                m.set_global_anchor(sd["C"][i], sd["D"][i])
            server.global_C, server.global_D = sd["C"], sd["D"]
            server.aggregated = True
        history = list(sd.get("history", []))
        start_rnd = len(history)
        print(f"resumed from {args.resume}: continuing at round {start_rnd}")

    for rnd in range(start_rnd, args.rounds):
        t0 = time.time()
        uploads = []
        for c in clients:
            if server.aggregated:
                server.broadcast(model)          # push (C,D) into model
            else:
                model.zero_anchors()
            # NOTE: same shared `model` object is reused per client; the
            # client restores its own personalized factors before training.
            up, logs = c.train_round(model, router, rnd)
            uploads.append(up)
            print(f"  r{rnd:03d} c{c.cid} loss={logs}", flush=True)
        server.aggregate(uploads)
        server.broadcast(model)

        row = dict(round=rnd, time=round(time.time() - t0, 1))
        if (rnd + 1) % args.eval_every == 0 or rnd == args.rounds - 1:
            # per-client personalized evaluation on BOTH test patients
            per_patient = {p: [] for p in TEST_PATIENTS}
            for c in clients:
                if c.g_k is None:
                    continue
                restore_client_into(model, c)
                if c.router_state is not None:
                    router.load_state_dict(c.router_state)
                vals = {}
                for p, ld in test_loaders.items():
                    m_ = evaluate_client_model(model, router, c.g_k, ld)
                    vals[f"{p}_psnr"], vals[f"{p}_ssim"] = m_["psnr"], m_["ssim"]
                    per_patient[p].append(m_["psnr"])
                psnr_mean = float(np.mean([vals[f"{p}_psnr"] for p in TEST_PATIENTS]))
                print(f"  eval c{c.cid} psnr={psnr_mean:.2f} "
                      f"ssim_mean={float(np.mean([vals[f'{p}_ssim'] for p in TEST_PATIENTS])):.4f}")
                row[f"c{c.cid}_psnr"] = psnr_mean
            for p in TEST_PATIENTS:
                row[f"{p}_psnr_mean"] = float(np.mean(per_patient[p]))
            print(f"[fedfuse] r{rnd:03d} L310={row['L310_psnr_mean']:.2f} "
                  f"L506={row['L506_psnr_mean']:.2f}")
        history.append(row)
        torch.save(dict(C=server.global_C, D=server.global_D,
                        personal={c.cid: c.personal_state for c in clients},
                        local={c.cid: c.local_state for c in clients},
                        router={c.cid: c.router_state for c in clients},
                        history=history),
                   os.path.join(args.out, "fedfuse_latest.pt"))
    return history


def train_fedavg(args):
    clients_meta = build_client_datasets(
        args.data_root, n_clients=args.clients,
        max_slices_per_client=args.max_slices if args.smoke else None,
        patch_size=args.patch, patients_per_client=client8_patients())
    global_model = REDCNN().to(DEV)
    mu = args.mu if args.method == "fedprox" else 0.0
    loss_fn = CompositeLoss(0.1, 0.05)
    test_ld = DataLoader(build_test_dataset(args.data_root,
                                            max_slices=args.test_slices),
                         batch_size=1, num_workers=2)
    history = []
    for rnd in range(args.rounds):
        gstate = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
        states, ns = [], []
        for m in clients_meta:
            model = REDCNN().to(DEV)
            model.load_state_dict(gstate)
            opt = torch.optim.Adam(model.parameters(), lr=1e-4)
            ld = DataLoader(m["dataset"], batch_size=8, shuffle=True,
                            num_workers=2, drop_last=True)
            model.train()
            for _ in range(args.local_epochs):
                for x, y in ld:
                    x, y = x.to(DEV), y.to(DEV)
                    loss, _ = loss_fn(model(x), y)
                    if mu > 0:
                        prox = sum(((p - gstate[n].to(DEV)) ** 2).sum()
                                   for n, p in model.named_parameters())
                        loss = loss + (mu / 2) * prox
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            states.append({k: v.cpu() for k, v in model.state_dict().items()})
            ns.append(len(m["dataset"]))
        global_model.load_state_dict(fedavg_aggregate(states, ns))
        row = dict(round=rnd)
        if (rnd + 1) % args.eval_every == 0 or rnd == args.rounds - 1:
            global_model.eval()
            ps = []
            with torch.no_grad():
                for x, y in test_ld:
                    x, y = x.to(DEV), y.to(DEV)
                    pred = global_model(x)
                    ph = (pred[0, 0].cpu().numpy() * 4096.0) - 1024.0
                    th = (y[0, 0].numpy() * 4096.0) - 1024.0
                    ps.append(psnr_ssim_rmse(ph, th, th > -500)["psnr"])
            row["test_psnr_mean"] = float(np.mean(ps))
            print(f"[{args.method}] r{rnd:03d} psnr={row['test_psnr_mean']:.2f}")
        history.append(row)
        torch.save(dict(model=global_model.state_dict(), history=history),
                   os.path.join(args.out, f"{args.method}_latest.pt"))
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="fedfuse", choices=["fedfuse", "fedavg", "fedprox"])
    ap.add_argument("--data-root", default="/mnt/d/icassp/datas/mayo_2016_npy")
    ap.add_argument("--dinov3-path", default="models/dinov3")
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--patch", type=int, default=None)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--experts", type=int, default=3)
    ap.add_argument("--lambda-bal", type=float, default=0.01)
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--test-slices", type=int, default=40)
    ap.add_argument("--max-slices", type=int, default=24, help="smoke: slices/client")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", default=None,
                    help="resume from a fedfuse_latest.pt checkpoint")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    if args.patch is None:
        args.patch = 256 if args.method == "fedfuse" else 128
    os.makedirs(args.out, exist_ok=True)

    if args.method == "fedfuse":
        if not os.path.isdir(args.dinov3_path):
            raise SystemExit(f"DINOv3 weights not found at {args.dinov3_path}")
        train_fedfuse(args)
    else:
        train_fedavg(args)


if __name__ == "__main__":
    main()
