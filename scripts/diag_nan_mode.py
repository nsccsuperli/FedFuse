"""Bisect NaN: A) pure fp32 (no autocast) vs B) bf16 autocast, few steps each.

Run: python scripts/diag_nan_mode.py {fp32|bf16}
"""
import sys
import torch
from torch.utils.data import DataLoader

from fedfuse.data import MayoLDCTDataset, TRAIN_PATIENTS
from fedfuse.losses import CompositeLoss
from fedfuse.models import FedFuseNet, LoRARouter
from fedfuse.physics import global_descriptor

mode = sys.argv[1] if len(sys.argv) > 1 else "fp32"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

net = FedFuseNet("models/dinov3", lora_r=8, n_experts=3).to(DEV)
router = LoRARouter(d_in=288, n_experts=3).to(DEV)
loss_fn = CompositeLoss()
trainable = ([m.B for m in net.backbone.lora_modules()]
             + list(router.parameters())
             + [p for n, p in net.named_parameters()
                 if n.startswith(("noise_branch", "fusions", "decoder"))])
opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-5)

ds = MayoLDCTDataset("/mnt/d/icassp/datas/mayo_2016_npy",
                     TRAIN_PATIENTS[:2], max_slices=6, patch_size=256)
ld = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

ok = True
for step in range(4):
    x, y = next(iter(ld))
    x, y = x.to(DEV), y.to(DEV)
    g = global_descriptor(x)[0]
    pi = router(g[None])[0]
    net.set_pi(pi)
    import contextlib
    cm = (torch.autocast("cuda", dtype=torch.bfloat16) if mode == "bf16"
          else contextlib.nullcontext())
    with cm:
        pred = net(x)
        loss, parts = loss_fn(pred.float(), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    print(f"step {step} [{mode}]: loss={loss.item():.4f} grad_norm={gn.item():.3f} "
          f"finite={torch.isfinite(gn).item()}")
    if not torch.isfinite(gn):
        ok = False
        break
    opt.step()

print(f"RESULT[{mode}]: {'CLEAN' if ok else 'NAN'}")
