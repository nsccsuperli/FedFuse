"""Analyze the 20-round full federal run (router-fixed, data-size stacking)."""
import sys
import json
import torch
sys.path.insert(0, "/mnt/d/icassp/FedFuse_code")
from fedfuse.models.lora import LoRARouter
from fedfuse.data import MayoLDCTDataset, client8_patients
from fedfuse.physics import global_descriptor

ck = torch.load("/mnt/d/icassp/FedFuse_code/runs/fedfuse_stack_full_v2/fedfuse_latest.pt",
                map_location="cpu", weights_only=False)
hist = ck["history"]
print(f"rounds logged: {len(hist)}")
pats = client8_patients()

# per-round aggregate
print("\nround | L310 | L506 | per-client PSNR")
for r in hist:
    ps = [f"{r[f'c{i}_psnr']:.1f}" for i in range(8) if f"c{i}_psnr" in r]
    print(f"  {r['round']:>3}  | {r['L310_psnr_mean']:5.2f} | {r['L506_psnr_mean']:5.2f} | {' '.join(ps)}")

# trend: first, mid, last
def avg_psnr(r):
    return torch.tensor([r[f"c{i}_psnr"] for i in range(8) if f"c{i}_psnr" in r]).mean().item()
print("\nmean per-client PSNR:", [f"{avg_psnr(r):.2f}" for r in [hist[0], hist[5], hist[10], hist[15], hist[-1]]])

# per-client progression (round 0 -> last)
print("\nper-client PSNR r0 -> r19 (delta):")
r0, rl = hist[0], hist[-1]
for i in range(8):
    if f"c{i}_psnr" in r0:
        d = rl[f"c{i}_psnr"] - r0[f"c{i}_psnr"]
        print(f"  c{i} ({pats[i]}): {r0[f'c{i}_psnr']:.2f} -> {rl[f'c{i}_psnr']:.2f} ({d:+.2f})")

# router specialization check on each client's own descriptor
print("\nrouter pi on own g_k (final):")
routers = ck["router"]
for cid in sorted(routers, key=int):
    ds = MayoLDCTDataset("/mnt/d/icassp/datas/mayo_2016_npy", [pats[int(cid)]], max_slices=6)
    gs = []
    for i in range(min(3, len(ds))):
        x, _ = ds[i]
        gs.append(global_descriptor(x[None])[0])
    g = torch.stack(gs).mean(0)
    router = LoRARouter(d_in=288, n_experts=3)
    router.load_state_dict(routers[cid])
    router.eval()
    with torch.no_grad():
        pi = torch.softmax(router.net(g) / router.tau, dim=-1)
    print(f"  c{cid} ({pats[int(cid)]}): pi={pi.numpy().round(3)}")

# anchor magnitude
C, D = ck["C"], ck["D"]
print(f"\nanchor: C absmax={C.abs().max().item():.4f} D absmax={D.abs().max().item():.4f}")
