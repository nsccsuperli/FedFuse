"""Protocol-matched re-evaluation of the 20-round federal run.

Two protocols per client:
  SHARED: test L310/L506 at native 25% (current train loop protocol)
  MATCHED: test L310/L506 degraded with THIS client's spec (its own domain)
          -> the per-client protocol used by ProFed/SCAN-PhysFed tables

Reports per-client PSNR/SSIM under both, plus input-LDCT baseline under the
matched spec so the +delta over input is visible (L310/L506 native input
baseline differs from degraded input baseline).
"""
import argparse
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, "/mnt/d/icassp/FedFuse_code")
from fedfuse.data import (MayoLDCTDataset, TEST_PATIENTS, client8_patients,
                          client8_specs, DoseDegrader, TRAIN_PATIENTS)
from fedfuse.metrics import psnr_ssim_rmse
from fedfuse.models import FedFuseNet, LoRARouter
from fedfuse.physics import global_descriptor

DEV = "cuda"

def make_loader(pat, degrader=None, n=40):
    ds = MayoLDCTDataset("/mnt/d/icassp/datas/mayo_2016_npy", [pat],
                         degrader=degrader, max_slices=n)
    return DataLoader(ds, batch_size=1, num_workers=2)

def eval_slices(model, router, g_k, loader):
    model.eval()
    rows = []
    with torch.no_grad():
        pi = router(g_k[None].to(DEV))[0]
        model.set_pi(pi)
        for x, y in loader:
            x = x.to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x)
            ph = (pred[0, 0].float().cpu().numpy() * 4096.0) - 1024.0
            th = (y[0, 0].numpy() * 4096.0) - 1024.0
            rows.append(psnr_ssim_rmse(ph, th, th > -500))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

def input_baseline(pat, degrader, n=40):
    """PSNR/SSIM of the (degraded) input itself vs its target."""
    ds = MayoLDCTDataset("/mnt/d/icassp/datas/mayo_2016_npy", [pat],
                         degrader=degrader, max_slices=n)
    rows = []
    for x, y in DataLoader(ds, batch_size=1, num_workers=2):
        xh = (x[0, 0].numpy() * 4096.0) - 1024.0
        th = (y[0, 0].numpy() * 4096.0) - 1024.0
        rows.append(psnr_ssim_rmse(xh, th, th > -500))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

def main():
    ck = torch.load("/mnt/d/icassp/FedFuse_code/runs/fedfuse_stack_full_v2/fedfuse_latest.pt",
                    map_location="cpu", weights_only=False)
    pats = client8_patients()
    specs = client8_specs()
    model = FedFuseNet("models/dinov3", lora_r=8, n_experts=3,
                       n_clients=8).to(DEV)
    router = LoRARouter(d_in=288, n_experts=3).to(DEV)

    print(f"{'client':6} {'patient':6} {'spec':14} | {'SHARED PSNR/SSIM':>20} | "
          f"{'MATCHED PSNR/SSIM':>20} | {'matched input base':>18}")
    for cid in range(8):
        spec = specs[cid]
        # load this client's personalized model
        st = ck["personal"][cid]
        with torch.no_grad():
            for i, mod in enumerate(model.backbone.lora_modules()):
                ls = st.get(f"lora{i}")
                if ls is not None:
                    mod.A.copy_(ls["A"]); mod.B.copy_(ls["B"])
        if cid in ck.get("router", {}):
            router.load_state_dict(ck["router"][cid])
        loc = ck.get("local", {}).get(cid)
        if loc:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in loc:
                        p.copy_(loc[n])
        # g_k from client's own (degraded) data
        ds_own = MayoLDCTDataset("/mnt/d/icassp/datas/mayo_2016_npy",
                                 [pats[cid]], DoseDegrader(spec), max_slices=6)
        gs = []
        for i in range(min(4, len(ds_own))):
            x, _ = ds_own[i]
            gs.append(global_descriptor(x[None].to(DEV))[0].cpu())
        g_k = torch.stack(gs).mean(0).to(DEV)

        # SHARED (native) eval on both patients
        sh = []
        for p in TEST_PATIENTS:
            sh.append(eval_slices(model, router, g_k, make_loader(p, None)))
        sh_p = np.mean([s["psnr"] for s in sh]); sh_s = np.mean([s["ssim"] for s in sh])

        # MATCHED eval: test patients degraded with this client's spec
        deg = DoseDegrader(spec)
        mt = []
        for p in TEST_PATIENTS:
            mt.append(eval_slices(model, router, g_k, make_loader(p, deg)))
        mt_p = np.mean([s["psnr"] for s in mt]); mt_s = np.mean([s["ssim"] for s in mt])

        # matched input baseline (input degraded, vs target)
        ib = [input_baseline(p, deg) for p in TEST_PATIENTS]
        ib_p = np.mean([b["psnr"] for b in ib]); ib_s = np.mean([b["ssim"] for b in ib])

        print(f"c{cid:<5} {pats[cid]:6} {spec.name:14} | "
              f"{sh_p:6.2f}/{sh_s:.3f}          | {mt_p:6.2f}/{mt_s:.3f}          | "
              f"{ib_p:6.2f}/{ib_s:.3f}")

if __name__ == "__main__":
    main()
