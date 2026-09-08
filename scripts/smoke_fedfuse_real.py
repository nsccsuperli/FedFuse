"""Smoke: full FedFuseNet on REAL DINOv3 ViT-L -- multi-step local training loop.

Extends smoke_backbone_real to the whole network (noise branch + fusion +
decoder) and runs K optimization steps exactly as FedFuseClient.train_round
would (fresh pi per step, set_pi, bf16 autocast, CompositeLoss, grad clip,
AdamW) WITHOUT any federated machinery. NaN here would reproduce the loop bug
locally; finite + falling loss means the loop itself is clean.

Also validates T2 inputs: r = x - GF(x), V(r) enter the branch; global
descriptor g is computed (interface kept) but NOT routed (T3 deferred).
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from fedfuse.data import MayoLDCTDataset, TRAIN_PATIENTS
from fedfuse.losses import CompositeLoss
from fedfuse.models import FedFuseNet, LoRARouter
from fedfuse.physics import global_descriptor

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=6)
p.add_argument("--batch", type=int, default=1)
p.add_argument("--patch", type=int, default=256)
p.add_argument("--experts", type=int, default=3)
p.add_argument("--dinov3-path", default="models/dinov3")
args = p.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
print(f"dev={DEV} steps={args.steps} patch={args.patch} experts={args.experts}")

net = FedFuseNet(args.dinov3_path, lora_r=8, n_experts=args.experts).to(DEV)
router = LoRARouter(d_in=288, n_experts=args.experts).to(DEV)
print(f"FedFuseNet trainable: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")

trainable = ([m.B for m in net.backbone.lora_modules()]
             + list(router.parameters())
             + [p for n, p in net.named_parameters()
                 if n.startswith(("noise_branch", "fusions", "decoder"))])
opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-5)
loss_fn = CompositeLoss()

# smoke data: 6 slices from 2 train patients, random 256 crops on the fly
ds = MayoLDCTDataset("/mnt/d/icassp/datas/mayo_2016_npy",
                     TRAIN_PATIENTS[:2], max_slices=6, patch_size=args.patch)
ld = DataLoader(ds, batch_size=args.batch, shuffle=True,
                num_workers=0, drop_last=True)
g_k = None
logs = []
for step, (x, y) in enumerate(ld):
    if step >= args.steps:
        break
    x, y = x.to(DEV), y.to(DEV)
    g_k = global_descriptor(x)[0]           # (288,) -- client keeps 1D descriptor
    pi = router(g_k[None].to(DEV))[0]
    net.set_pi(pi)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
        pred = net(x)
        loss, parts = loss_fn(pred.float(), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    print(f"step {step}: loss={loss.item():.4f}  l1={parts['l1']:.4f} "
          f"ssim={parts['ssim']:.4f} edge={parts['edge']:.4f} pi={pi.detach().cpu().numpy().round(3)}")
    assert torch.isfinite(loss), f"NaN at step {step}"
    logs.append(loss.item())

print(f"first={logs[0]:.4f} last={logs[-1]:.4f} "
      f"falling={logs[-1] < logs[0]:} finite_all={all(np.isfinite(logs))}")
print("SMOKE FEDFUSE-REAL PASS")
