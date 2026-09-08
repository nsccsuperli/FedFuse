"""Pinpoint the exact op whose backward produces NaN on step 1 (real DINOv3)."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fedfuse.data import MayoLDCTDataset, TRAIN_PATIENTS
from fedfuse.losses import CompositeLoss
from fedfuse.models import FedFuseNet, LoRARouter
from fedfuse.physics import global_descriptor

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

torch.autograd.set_detect_anomaly(True)   # trace first NaN backward op

for step in range(2):
    x, y = next(iter(ld))
    x, y = x.to(DEV), y.to(DEV)
    g = global_descriptor(x)[0]
    pi = router(g[None])[0]
    net.set_pi(pi)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
        pred = net(x)
        loss, parts = loss_fn(pred.float(), y)
    print(f"step {step}: loss={loss.item():.4f}")
    opt.zero_grad(set_to_none=True)
    try:
        loss.backward()
        print(f"  backward OK")
    except RuntimeError as e:
        print(f"  BACKWARD RAISED: {type(e).__name__}: {str(e)[:500]}")
        break
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    print(f"  grad_norm={gn.item():.4f}")
    opt.step()
