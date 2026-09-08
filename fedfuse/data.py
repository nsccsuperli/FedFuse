"""Mayo-2016 LDCT dataset, client partitioning and degradation heterogeneity.

Storage convention: <root>/<Lxxx>_<25|target>_<slice>_img.npy, uint16 512x512,
HU = raw - 1024. All tensors are normalized to [0, 1] over the wide window
[-1024, 3072] HU (standard for LDCT enhancement training).

Degradation heterogeneity (Sec. III-A of the manuscript): the native Mayo
release contains a single low-dose level (25%). To emulate multi-center
degradation heterogeneity we inject, per client, a signal-dependent noise
field calibrated to a target dose scale:

    sigma(u,v) = sqrt( alpha * mu_local(u,v) + beta ) ,  mu_local = box(mu)

where mu is the linearized attenuation (proportional to HU+1024). The noise
std of LDCT scales as 1/sqrt(dose); the injection strength is solved so that
the resulting slice matches the noise level of the target dose scale. An
optional client-specific smoothing kernel emulates reconstruction-kernel
diversity.
"""
import os
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

FNAME_RE = re.compile(r"^(L\d+)_(25|target)_(\d+)_img\.npy$")
HU_OFFSET = 1024.0
NORM_LO, NORM_HI = -1024.0, 3072.0          # wide window -> [0, 1]
REF_NOISE_HU = 42.0                          # measured median native 25% noise
REF_DOSE = 0.25

TRAIN_PATIENTS = ["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L333"]
TEST_PATIENTS = ["L310", "L506"]


def to_hu(arr_u16: np.ndarray) -> np.ndarray:
    return arr_u16.astype(np.float32) - HU_OFFSET


def normalize_hu(x_hu: np.ndarray) -> np.ndarray:
    return np.clip((x_hu - NORM_LO) / (NORM_HI - NORM_LO), 0.0, 1.0)


def denormalize_hu(x01: np.ndarray) -> np.ndarray:
    return x01 * (NORM_HI - NORM_LO) + NORM_LO


@dataclass
class DegradationSpec:
    """Per-client degradation configuration."""
    dose_scale: float = 1.0      # relative to native 25% (1.0 = keep native)
    kernel_sigma: float = 0.0    # Gaussian blur sigma (px), 0 = none
    name: str = "native25"


class DoseDegrader:
    """Signal-dependent noise injection calibrated to a dose scale."""

    def __init__(self, spec: DegradationSpec):
        self.spec = spec
        # solve injected std so that sqrt(ref^2 + inj^2) = ref * sqrt(1/dose_scale)
        target_std = REF_NOISE_HU * np.sqrt(1.0 / max(spec.dose_scale, 1e-6))
        self.inject_std_hu = float(np.sqrt(max(target_std**2 - REF_NOISE_HU**2, 0.0)))

    def __call__(self, x01: torch.Tensor) -> torch.Tensor:
        """x01: (B,1,H,W) in [0,1] -> degraded copy (signal-dependent noise)."""
        if self.inject_std_hu <= 0 and self.spec.kernel_sigma <= 0:
            return x01
        hu = x01 * (NORM_HI - NORM_LO) + NORM_LO
        mu = (hu + HU_OFFSET) / HU_OFFSET                       # linearized attenuation
        mu_local = F.avg_pool2d(mu, kernel_size=17, stride=1, padding=8)
        # 80% quantum (signal-dependent) + 20% electronic (uniform) noise energy
        var = 0.8 * (self.inject_std_hu ** 2) * mu_local + 0.2 * (self.inject_std_hu ** 2)
        noisy_hu = hu + torch.randn_like(hu) * torch.sqrt(var.clamp_min(0))
        out = (noisy_hu - NORM_LO) / (NORM_HI - NORM_LO)
        out = out.clamp(0, 1)
        if self.spec.kernel_sigma > 0:
            k = int(2 * round(3 * self.spec.kernel_sigma) + 1)
            g = torch.exp(-0.5 * ((torch.arange(k, device=out.device) - k // 2)
                                  / self.spec.kernel_sigma) ** 2)
            g = (g / g.sum())
            kern = (g[:, None] * g[None, :]).expand(1, 1, k, k)
            out = F.conv2d(out, kern, padding=k // 2)
        return out


class MayoLDCTDataset(Dataset):
    """Paired LDCT/NDCT slices of selected patients, with optional degradation.

    preload=True caches raw uint16 slices in RAM (whole Mayo train split needs
    ~5 GB) — worth it because /mnt/* IO dominates epoch time otherwise.
    """

    def __init__(self, root, patients, degrader: DoseDegrader | None = None,
                 slice_range=None, max_slices=None, preload=True, patch_size=None):
        self.root = root
        self.degrader = degrader
        self.patch_size = patch_size           # random crop for training; None = full slice
        self.items = []                       # (ld_path, nd_path)
        for pat in patients:
            ld, nd = {}, {}
            for f in sorted(os.listdir(root)):
                m = FNAME_RE.match(f)
                if not m or m.group(1) != pat:
                    continue
                (ld if m.group(2) == "25" else nd)[int(m.group(3))] = f
            idxs = sorted(set(ld) & set(nd))
            if slice_range is not None:
                lo, hi = slice_range
                idxs = [i for i in idxs if lo <= i <= hi]
            for i in idxs:
                self.items.append((os.path.join(root, ld[i]), os.path.join(root, nd[i])))
        if max_slices is not None:
            self.items = self.items[:max_slices]
        self._cache = None
        if preload and self.items:
            self._cache = [(np.load(a), np.load(b)) for a, b in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self._cache is not None:
            ld_u, nd_u = self._cache[i]
        else:
            ld_u, nd_u = np.load(self.items[i][0]), np.load(self.items[i][1])
        ld = normalize_hu(to_hu(ld_u))
        nd = normalize_hu(to_hu(nd_u))
        x = torch.from_numpy(ld)[None]        # (1,H,W)
        y = torch.from_numpy(nd)[None]
        if self.degrader is not None:
            x = self.degrader(x[None])[0]
        if self.patch_size is not None:
            p = self.patch_size
            H, W = x.shape[-2:]
            i = int(np.random.randint(0, H - p + 1))
            j = int(np.random.randint(0, W - p + 1))
            x, y = x[..., i:i + p, j:j + p], y[..., i:i + p, j:j + p]
        return x, y


def default_client_specs(n_clients=4):
    """Dose-scale ladder emulating {25%, 15%, 10%, 5%} + kernel diversity."""
    ladder = [
        DegradationSpec(1.00, 0.0, "d25-native"),
        DegradationSpec(0.60, 0.0, "d15-inj"),      # ~15% dose
        DegradationSpec(0.40, 0.6, "d10-inj-soft"),  # ~10% dose + soft kernel
        DegradationSpec(0.20, 0.0, "d05-inj"),      # ~5% dose
    ]
    return ladder[:n_clients]


def client8_specs():
    """8 degradation specs, one per federated client (paper: 8 clients, K=8).

    Image-domain dose ladder (no projection/CTLib): dose_scale is relative to
    the native 25% acquisition; smaller scale -> heavier injected noise.
    Includes kernel diversity to emulate reconstruction-kernel heterogeneity.
    Order aligned with client8_patients() (L067 ... L333).
    """
    return [
        DegradationSpec(1.00, 0.0, "d25-native"),     # L067  mildest
        DegradationSpec(0.60, 0.0, "d15-inj"),        # L096
        DegradationSpec(0.45, 0.6, "d10-inj-soft"),   # L109  ~10% + soft kernel
        DegradationSpec(0.35, 0.0, "d075-inj"),       # L143
        DegradationSpec(0.25, 0.0, "d06-inj"),        # L192
        DegradationSpec(0.18, 1.2, "d04-inj-blur"),   # L286  heavy dose + blur
        DegradationSpec(0.12, 0.0, "d03-inj"),        # L291
        DegradationSpec(0.08, 0.8, "d02-inj-soft"),   # L333  heaviest
    ]


def client8_patients():
    """One distinct train patient per client (K=8): 8 patients, 8 clients.

    Each client owns exactly one patient's slices, so client identity carries
    a real anatomical prior (per-patient), while the paired degradation spec
    (client8_specs) adds dose/kernel heterogeneity -- the image-domain
    emulation of multi-center protocol diversity (no CTLib projection sim).
    """
    return sorted(TRAIN_PATIENTS)   # L067..L333 -> 8 entries


def build_client_datasets(root, n_clients=4, train_patients=TRAIN_PATIENTS,
                          specs=None, max_slices_per_client=None, seed=0,
                          patch_size=None, patients_per_client=None):
    """Partition training patients across clients.

    Default: deterministic round-robin split of the patient list (one patient
    may appear on several clients when n_clients < len(patients)).
    With patients_per_client (len == n_clients, e.g. client8_patients for
    K=8) each client gets exactly the listed patient(s) + its degradation
    spec -- the one-patient-per-client layout used for the 8-client paper
    protocol.
    """
    if patients_per_client is not None:
        assert len(patients_per_client) == n_clients, \
            f"patients_per_client {len(patients_per_client)} != n_clients {n_clients}"
        specs = specs or client8_specs()
        assert len(specs) == n_clients
        clients = []
        for cid, (pat, spec) in enumerate(zip(patients_per_client, specs)):
            ds = MayoLDCTDataset(root, [pat], DoseDegrader(spec),
                                 max_slices=max_slices_per_client,
                                 patch_size=patch_size)
            clients.append(dict(cid=cid, patients=[pat], spec=spec, dataset=ds))
        return clients
    specs = specs or default_client_specs(n_clients)
    assert len(specs) == n_clients
    rng = np.random.default_rng(seed)
    pats = list(train_patients)
    rng.shuffle(pats)
    groups = [sorted(pats[i::n_clients]) for i in range(n_clients)]
    clients = []
    for cid, (grp, spec) in enumerate(zip(groups, specs)):
        ds = MayoLDCTDataset(root, grp, DoseDegrader(spec),
                             max_slices=max_slices_per_client,
                             patch_size=patch_size)
        clients.append(dict(cid=cid, patients=grp, spec=spec, dataset=ds))
    return clients


def build_test_dataset(root, patients=TEST_PATIENTS, max_slices=None):
    return MayoLDCTDataset(root, patients, degrader=None, max_slices=max_slices,
                           patch_size=None)
