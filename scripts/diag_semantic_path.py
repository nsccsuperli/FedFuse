"""Diagnose whether the DINOv3 semantic path + LoRA actually contributed.

Checks from the 20-round checkpoint:
  1. Fusion gate gamma_s: init 1e-3. If still ~1e-3 after training, the
     semantic path is NOT being absorbed (Eq.5 gate stayed closed).
  2. LoRA adaptation magnitude ||B A|| per module vs frozen base scale:
     if ~0, adapters are not doing anything.
  3. Per-client A/B drift from init (A init ~ N(0,1/sqrt(r)), B init 0).
  4. How much of the final output delta comes from the anchor vs personal.
"""
import sys
import numpy as np
import torch
sys.path.insert(0, "/mnt/d/icassp/FedFuse_code")
from fedfuse.data import client8_patients

ck = torch.load("/mnt/d/icassp/FedFuse_code/runs/fedfuse_stack_full_v2/fedfuse_latest.pt",
                map_location="cpu", weights_only=False)
pats = client8_patients()

# ---- 1. fusion gates gamma (stored in per-client local state) ------------
print("=== 1. fusion gate gamma_s (init = 1e-3) ===")
for cid in range(8):
    loc = ck["local"].get(cid)
    if loc:
        gammas = [loc[k].item() for k in sorted(loc) if k.startswith("fusions") and k.endswith("gamma")]
        print(f"  c{cid} ({pats[cid]:6s}): gamma = {[f'{g:.4f}' for g in gammas]}")
    else:
        print(f"  c{cid}: no local state")

# ---- 2/3. LoRA magnitude & drift for each client -------------------------
print("\n=== 2. LoRA ||B@A|| (adaptation strength) per client ===")
for cid in range(8):
    pers = ck["personal"][cid]
    norms = []
    for li in range(3):                # sample 3 modules
        A, B = pers[f"lora{li}"]["A"], pers[f"lora{li}"]["B"]
        BA = torch.matmul(B, A)               # (E,d,d)
        norms.append(BA.norm(dim=(1, 2)).mean().item())
    print(f"  c{cid} ({pats[cid]:6s}): mean ||B@A||_F (per module, 3 exp avg) = "
          f"{[f'{n:.4f}' for n in norms]}")

# B drift from zero init (B started at exactly 0)
print("\n=== 3. B magnitude (init=0) -- did adapters move? ===")
for cid in [0, 5]:
    pers = ck["personal"][cid]
    A, B = pers["lora0"]["A"], pers["lora0"]["B"]
    print(f"  c{cid}: B absmax={B.abs().max().item():.5f}  A absmax={A.abs().max().item():.5f}")

# ---- 4. anchor vs personal contribution ----------------------------------
print("\n=== 4. global anchor magnitude (D rows ~ A stack) ===")
C, D = ck["C"], ck["D"]
print(f"  anchor per-module: ||C@D||_F mean over modules = "
      f"{(torch.matmul(C, D)).norm(dim=(1,2)).mean().item():.4f}")
print(f"  personal ||B@A|| ~ 1e-3-1e-2 vs anchor ||CD|| above -> relative share")
