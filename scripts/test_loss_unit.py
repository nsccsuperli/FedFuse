"""Unit-test the rewritten CompositeLoss in isolation (fp32 + bf16 inputs)."""
import torch
from fedfuse.losses import CompositeLoss, ssim, sobel

torch.manual_seed(0)
loss_fn = CompositeLoss()

# random smooth-ish tensors like actual pred/gt
x = torch.rand(1, 1, 64, 64)
y = torch.rand(1, 1, 64, 64)

for dtype, name in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
    a, b = x.to(dtype), y.to(dtype)
    with torch.autocast("cuda", dtype=torch.bfloat16):   # hostile context
        loss, parts = loss_fn(a.float(), b)
        s = ssim(a.float(), b)
        e = sobel(a.float())
    print(f"{name}: loss={loss.item():.5f} finite={torch.isfinite(loss).item()} "
          f"parts={parts} ssim={s.item():.4f} "
          f"sobel_finite={bool(torch.isfinite(e).all().item())}")
    loss.backward()
    print(f"  backward ok")

# constant-image edge case: sxx=syy=0 -> division by ~c2 only
c = torch.full((1, 1, 64, 64), 0.5)
loss, parts = loss_fn(c, c)
print(f"const-const: loss={loss.item():.5f} finite={torch.isfinite(loss).item()}")
