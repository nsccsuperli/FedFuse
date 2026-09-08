"""Quantify each of the 8 clients: patient, data size, and ACTUAL injected
noise/blur of its degradation spec (measured, not just the spec name)."""
import sys
import numpy as np
import torch
sys.path.insert(0, "/mnt/d/icassp/FedFuse_code")
from fedfuse.data import (MayoLDCTDataset, client8_patients, client8_specs,
                          DoseDegrader, TRAIN_PATIENTS)

pats = client8_patients()
specs = client8_specs()

print(f"{'c':>2} {'patient':6} {'n_slices':>8}  spec-name        "
      f"dose_scale kernel  injected_std_HU  measured_noise_HU(25->deg)")
for cid, (pat, spec) in enumerate(zip(pats, specs)):
    ds = MayoLDCTDataset("/mnt/d/icassp/datas/mayo_2016_npy", [pat],
                         max_slices=None, preload=False)
    # native 25% slice -> measure native noise
    x_native_u = np.load(ds.items[0][0]).astype(np.float32)
    deg = DoseDegrader(spec)
    # degrade the SAME native slice
    xt = torch.from_numpy((x_native_u.astype(np.float32) - 1024.0 + 1024.0) / 4096.0)[None, None]
    # NOTE: MayoLDCTDataset normalizes (hu+1024)/4096; replicate: (raw-1024+1024)/4096
    xd = deg(xt)[0, 0].numpy()
    x_n = (x_native_u.astype(np.float32) - 1024.0 + 1024.0) / 4096.0
    # residual noise std on body region (hu>0.5 ~ -500HU... use raw>0.5 approx)
    body = x_n > 0.5
    if body.sum() < 100:
        body = x_n > 0.2
    n_native = float(np.std((x_n[body] - xd[body])))
    n_deg = float(np.std(xd[body] - x_n[body]))
    print(f"{cid:>2} {pat:6} {len(ds):>8}  {spec.name:14s} "
          f"{spec.dose_scale:8.2f} {spec.kernel_sigma:6.1f} "
          f"{deg.inject_std_hu:14.2f}  {n_deg:.4f} (01-scale)")

# total dataset picture
print("\ntotal train patients (all):", TRAIN_PATIENTS)
print("total train pairs across all 8 clients:",
      sum(len(MayoLDCTDataset('/mnt/d/icassp/datas/mayo_2016_npy', [p],
                              preload=False)) for p in pats))
