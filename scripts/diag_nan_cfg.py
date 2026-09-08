"""Pinpoint which sub-path under bf16 autocast breaks backward (v2, fixed).

Configs (2 steps each, same data order per config):
  A: vit runs fp32 (autocast disabled around the whole DINOv3 call);
     CNN noise branch / fusions / decoder stay under outer bf16 autocast
  C: everything bf16 (autocast as shipped)                  [expect NAN]
  D: everything fp32 (no autocast anywhere)                 [expect CLEAN]
If A is CLEAN  -> NaN originates inside the ViT's own bf16 backward.
If A is still NAN -> NaN originates in the CNN/fusion/decoder bf16 path.
"""
import contextlib
import sys
import torch
from torch.utils.data import DataLoader

from fedfuse.data import MayoLDCTDataset, TRAIN_PATIENTS
from fedfuse.losses import CompositeLoss
from fedfuse.models import FedFuseNet, LoRARouter
from fedfuse.physics import global_descriptor

cfg = sys.argv[1] if len(sys.argv) > 1 else "C"
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

# ---- cfg A: force DINOv3 forward into fp32 by patching backbone.forward ---
orig_bkb_fwd = net.backbone.forward.__func__  # unbound
import types


def bkb_fwd_fp32(self, x):
    """Same as DinoV3Backbone.forward but DINOv3 runs under autocast off."""
    x3 = x.repeat(1, 3, 1, 1)
    x3 = (x3 - self.mean) / self.std
    with torch.autocast("cuda", enabled=False):
        out = self.vit(pixel_values=x3, output_hidden_states=True)
    hs = out.hidden_states
    B, _, H, W = x.shape
    h, w = H // self.patch, W // self.patch
    feats = []
    for l in self.tap_layers:
        t = hs[l + 1]
        t = t[:, -h * w:, :]
        feats.append(t.transpose(1, 2).reshape(B, self.embed_dim, h, w))
    return feats


if cfg == "A":
    net.backbone.forward = types.MethodType(bkb_fwd_fp32, net.backbone)
elif cfg == "E":
    # CNN noise branch / fusions / decoder fp32; backbone stays under outer
    # bf16 autocast (true mixed mode, unlike the buggy first version)
    import torch.nn.functional as F
    from fedfuse.physics import noise_residual, local_variance_map

    def net_fwd_mixed(self, x):
        with torch.autocast("cuda", enabled=False):
            r = noise_residual(x)
            v = local_variance_map(r)
            f_img = self.noise_branch(x, r, v)
        f_sem = self.backbone(x)               # outer bf16 autocast applies
        with torch.autocast("cuda", enabled=False):
            fused = [self.fusions[i](f_img[i], f_sem[i]) for i in range(4)]
            out = x + self.decoder(fused)
        return out

    net.forward = types.MethodType(net_fwd_mixed, net)
elif cfg == "F":
    # noise branch fp32; fusions + decoder stay bf16
    import torch.nn.functional as F
    from fedfuse.physics import noise_residual, local_variance_map

    def net_fwd_fp32(self, x):
        with torch.autocast("cuda", enabled=False):
            r = noise_residual(x)
            v = local_variance_map(r)
            f_img = self.noise_branch(x, r, v)
        f_sem = self.backbone(x)
        fused = [self.fusions[i](f_img[i], f_sem[i]) for i in range(4)]
        out = x + self.decoder(fused)
        return out

    net.forward = types.MethodType(net_fwd_fp32, net)

print(f"cfg={cfg}")
ok = True
for step in range(8):
    x, y = next(iter(ld))
    x, y = x.to(DEV), y.to(DEV)
    g = global_descriptor(x)[0]
    pi = router(g[None])[0]
    net.set_pi(pi)
    with torch.autocast("cuda", dtype=torch.bfloat16) if cfg == "A" or cfg == "C" \
            else contextlib.nullcontext():
        pred = net(x)
        loss, parts = loss_fn(pred.float(), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    print(f"  step {step}: loss={loss.item():.4f} grad_norm={gn.item():.3f} "
          f"finite={torch.isfinite(gn).item()}")
    if not torch.isfinite(gn):
        ok = False
        break
    opt.step()

print(f"RESULT[{cfg}]: {'CLEAN' if ok else 'NAN'}")
