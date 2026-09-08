"""Check nested autocast semantics: does autocast(enabled=False) inside an
outer autocast(bf16) really disable downcasting for conv ops?"""
import torch

x = torch.randn(2, 3, 32, 32, device="cuda")
w = torch.randn(8, 3, 3, 3, device="cuda")

with torch.autocast("cuda", dtype=torch.bfloat16):
    out_outer = torch.nn.functional.conv2d(x, w)          # expect bf16
    with torch.autocast("cuda", enabled=False):
        out_inner = torch.nn.functional.conv2d(x, w)      # expect fp32?

print(f"outer bf16 ctx          : {out_outer.dtype}")
print(f"inner enabled=False ctx : {out_inner.dtype}  (want float32)")

# also: enabled=False nested inside fp16 vs explicit disable via float()
with torch.autocast("cuda", dtype=torch.bfloat16):
    out_f = torch.nn.functional.conv2d(x.float(), w.float())
print(f"outer ctx but .float()  : {out_f.dtype}  (want float32)")
