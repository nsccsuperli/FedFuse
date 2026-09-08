"""Verify dtype chain after CNN-fp32 patch: backbone feat dtype under bf16
outer autocast, whole-net pred finiteness."""
import torch
from fedfuse.models import FedFuseNet

net = FedFuseNet("models/dinov3", lora_r=8, n_experts=3).cuda().eval()
x = torch.rand(1, 1, 256, 256).cuda()
with torch.autocast("cuda", dtype=torch.bfloat16):
    feats = net.backbone(x)
    print("backbone feat dtypes:", sorted({str(f.dtype).split(".")[-1] for f in feats}))
    pred = net(x)
    print("net pred dtype:", str(pred.dtype).split(".")[-1],
          " finite:", bool(torch.isfinite(pred).all().item()))
