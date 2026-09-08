"""Minimal repro: does NoiseBranch alone NaN under bf16 autocast backward?

Feeds real physics inputs [x, r, V(x)] and optimizes a trivial L1 loss to the
input (identity-ish target), 4 steps, once fp32 once bf16.
"""
import contextlib
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fedfuse.data import MayoLDCTDataset, TRAIN_PATIENTS
from fedfuse.models.fedfuse import NoiseBranch
from fedfuse.physics import noise_residual, local_variance_map

mode = sys.argv[1] if len(sys.argv) > 1 else "bf16"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

br = NoiseBranch().to(DEV)
opt = torch.optim.AdamW(br.parameters(), lr=1e-4, weight_decay=1e-5)

ds = MayoLDCTDataset("/mnt/d/icassp/datas/mayo_2016_npy",
                     TRAIN_PATIENTS[:1], max_slices=4, patch_size=256)
ld = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

print(f"mode={mode}")
ok = True
for step in range(4):
    x, _ = next(iter(ld))
    x = x.to(DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16) if mode == "bf16" \
            else contextlib.nullcontext():
        r = noise_residual(x)
        v = local_variance_map(r)
        feats = br(x, r, v)
        loss = sum(f.pow(2).mean() for f in feats)   # L2 sink, all channels
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(br.parameters(), 1.0)
    print(f"  step {step}: loss={loss.item():.4f} grad_norm={gn.item():.3f} "
          f"finite={torch.isfinite(gn).item()}")
    if not torch.isfinite(gn):
        ok = False
        break
    opt.step()

print(f"RESULT[{mode}]: {'CLEAN' if ok else 'NAN'}")
