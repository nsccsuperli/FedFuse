"""Smoke: DinoV3Backbone on the REAL DINOv3 ViT-L (after layer-location fix).

Verifies (F2):
  1. layer-list location works (24 blocks injected, 48 MoLoRA q/v modules)
  2. tap_layers stay valid for a 24-layer net and reshape to (B,1024,h,w)
  3. one forward+backward step: loss finite, grads only on B (A frozen)
  4. register-token handling drops cls+registers, keeps h*w patch tokens
"""
import torch
from fedfuse.models import DinoV3Backbone

MODEL_DIR = "/mnt/d/icassp/FedFuse_code/models/dinov3"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

m = DinoV3Backbone(MODEL_DIR, lora_r=8, n_experts=3).to(DEV)
vit = m.vit
layers = (vit.model.layer if hasattr(vit, "model") and hasattr(vit.model, "layer")
          else vit.encoder.layer)
n_blk = len(layers)
n_lora = len(m.lora_modules())
n_B = sum(1 for p in m.parameters() if p.requires_grad)
print(f"blocks={n_blk}  injected q/v modules={n_lora}  "
      f"trainable tensors={n_B}  dev={DEV}")
assert n_lora == 2 * n_blk, n_lora
assert m.tap_layers == [5, 11, 17, 23], m.tap_layers

# grad sink: every tap feature must receive gradient -> sum-pool to scalar
feats = m(torch.rand(1, 1, 224, 224, device=DEV))
print("tap shapes:", [tuple(f.shape) for f in feats])
assert all(f.shape == (1, 1024, 14, 14) for f in feats)

loss = sum(f.pow(2).mean() for f in feats)
loss.backward()

n_grad_B = sum(1 for mm in m.lora_modules() if mm.B.grad is not None)
n_grad_A = sum(1 for mm in m.lora_modules() if mm.A.grad is not None)
grad_norm_B = sum(mm.B.grad.norm().item() ** 2 for mm in m.lora_modules()
                  if mm.B.grad is not None) ** 0.5
grad_norm_A = sum(mm.A.grad.norm().item() ** 2 for mm in m.lora_modules()
                  if mm.A.grad is not None) ** 0.5
# new paper design: A AND B are both trainable; anchors are frozen buffers
n_anchor_grad = sum(1 for mm in m.lora_modules()
                    if mm.anchor_C.grad is not None or mm.anchor_D.grad is not None)
print(f"loss={loss.item():.4f}  B grads={n_grad_B}/{n_lora}  "
      f"A grads={n_grad_A}/{n_lora}  anchor grads={n_anchor_grad} (want 0)  "
      f"||grad_B||={grad_norm_B:.4f} ||grad_A||={grad_norm_A:.4f}")
assert torch.isfinite(loss)
assert n_grad_B == n_lora
assert n_grad_A == n_lora
assert n_anchor_grad == 0
print("SMOKE F2 PASS")
