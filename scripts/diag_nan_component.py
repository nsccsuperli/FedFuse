"""Which loss component NaNs under bf16 autocast wrapper after the fix?

Same setup as cfg C (whole thing under bf16 autocast incl. loss_fn call)
but decompose: l1 / ssim / edge gradients computed separately.
Also runs the full cfg-C-style step (opt step) and checks EVERY param grad.
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

for step in range(2):
    x, y = next(iter(ld))
    x, y = x.to(DEV), y.to(DEV)
    g = global_descriptor(x)[0]
    pi = router(g[None])[0]
    net.set_pi(pi)
    # exact cfg-C context: forward AND loss_fn call under bf16 autocast
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = net(x)
        loss, parts = loss_fn(pred.float(), y)
    print(f"step {step}: loss={loss.item():.5f} finite={torch.isfinite(loss).item()}")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    bad = [(n, p.grad.norm().item()) for n, p in net.named_parameters()
           if p.requires_grad and p.grad is not None
           and not torch.isfinite(p.grad).all()]
    print(f"  nonfinite grads: {len(bad)}")
    for n, v in bad[:5]:
        print(f"    {n}: {v}")
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    print(f"  grad_norm={gn.item():.3f} finite={torch.isfinite(gn).item()}")
    if not torch.isfinite(gn):
        break
    opt.step()

