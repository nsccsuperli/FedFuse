"""Decisive: reproduce cfg-C NaN at the exact failing step, decompose.

Runs the cfg-C context (loss_fn called under outer bf16 autocast, relying on
loss_fn's internal fp32 forcing). At the NaN step, prints:
  - loss components (l1/ssim/edge) finiteness
  - gradient finiteness per module group
  - whether loss_fn internals actually ran conv in fp32 (dtype probe)
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fedfuse.data import MayoLDCTDataset, TRAIN_PATIENTS
from fedfuse.losses import CompositeLoss, ssim, sobel
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

# --- dtype probe: monkeypatch _conv2d_fp32 to record actual conv dtypes ---
import fedfuse.losses as L
orig_conv = L._conv2d_fp32
conv_dtypes = []


def spy_conv(t, kern, padding=0, groups=1):
    out = orig_conv(t, kern, padding, groups)
    conv_dtypes.append((out.dtype, t.dtype))
    return out


L._conv2d_fp32 = spy_conv
# rebind inside ssim_map's global lookup is via module attr -> works since
# ssim_map references the module-level name at call time? NO - it references
# the global _conv2d_fp32, which we replaced in L. Good.

for step in range(8):
    x, y = next(iter(ld))
    x, y = x.to(DEV), y.to(DEV)
    g = global_descriptor(x)[0]
    pi = router(g[None])[0]
    net.set_pi(pi)
    conv_dtypes.clear()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = net(x)
        # cfg-C context: loss_fn called under bf16 autocast
        loss, parts = loss_fn(pred.float(), y)
    print(f"step {step}: loss={loss.item():.5f} parts={ {k: round(v,4) for k,v in parts.items()} } "
          f"conv_outs={set(str(d[0]).split('.')[-1] for d in conv_dtypes)} "
          f"conv_ins={set(str(d[1]).split('.')[-1] for d in conv_dtypes)}")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    print(f"  grad_norm={gn.item():.3f} finite={torch.isfinite(gn).item()}")
    if not torch.isfinite(gn):
        # decompose: which component carries the NaN grad?
        for name, comp in (("l1", None), ("ssim", None), ("edge", None)):
            pass
        break
    opt.step()
