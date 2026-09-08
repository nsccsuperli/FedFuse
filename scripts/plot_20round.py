"""Visualize the 20-round federal run: convergence curves + per-client bars."""
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/d/icassp/FedFuse_code")
from fedfuse.data import client8_patients, client8_specs

ck = torch.load("/mnt/d/icassp/FedFuse_code/runs/fedfuse_stack_full_v2/fedfuse_latest.pt",
                map_location="cpu", weights_only=False)
hist = ck["history"]
pats = client8_patients()
specs = [s.name for s in client8_specs()]

# per-client PSNR per round from history
rounds = [r["round"] for r in hist]
L310 = [r["L310_psnr_mean"] for r in hist]
L506 = [r["L506_psnr_mean"] for r in hist]
per_client = {i: [r[f"c{i}_psnr"] for r in hist if f"c{i}_psnr" in r]
              for i in range(8)}

# protocol-matched re-eval results (parsed from eval output; hardcode from run)
matched = {
    0: dict(shared=(41.08, 0.974), matched=(41.08, 0.974), ib=(40.90, 0.955)),
    1: dict(shared=(43.78, 0.974), matched=(43.53, 0.975), ib=(38.31, 0.927)),
    2: dict(shared=(42.73, 0.966), matched=(41.20, 0.968), ib=(40.20, 0.956)),
    3: dict(shared=(43.78, 0.972), matched=(43.05, 0.976), ib=(35.77, 0.883)),
    4: dict(shared=(43.37, 0.981), matched=(42.75, 0.977), ib=(34.23, 0.847)),
    5: dict(shared=(34.88, 0.960), matched=(41.70, 0.973), ib=(36.53, 0.946)),
    6: dict(shared=(43.57, 0.973), matched=(42.43, 0.976), ib=(30.95, 0.744)),
    7: dict(shared=(41.78, 0.962), matched=(41.36, 0.973), ib=(36.34, 0.902)),
}

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# (1) convergence: L310/L506 + mean per-client PSNR per round
ax = axes[0, 0]
for i in range(8):
    ax.plot(rounds[:len(per_client[i])], per_client[i], lw=0.8, alpha=0.55,
            label=f"c{i} {pats[i]} {specs[i]}")
mean_pc = [np.mean([per_client[i][j] for i in range(8)]) for j in range(len(rounds))]
ax.plot(rounds, mean_pc, "k-", lw=2, label="mean")
ax.set_xlabel("communication round"); ax.set_ylabel("PSNR (dB)")
ax.set_title("Per-client PSNR across 20 FL rounds (full-range, shared protocol)")
ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)

# (2) test-patient aggregate convergence
ax = axes[0, 1]
ax.plot(rounds, L310, "o-", label="L310"); ax.plot(rounds, L506, "s-", label="L506")
ax.set_xlabel("round"); ax.set_ylabel("PSNR (dB)")
ax.set_title("Test-patient PSNR (mean over clients)"); ax.legend(); ax.grid(alpha=0.3)

# (3) per-client final: SHARED vs MATCHED vs input baseline
ax = axes[1, 0]
idx = np.arange(8)
w = 0.27
sh = [matched[i]["shared"][0] for i in range(8)]
mt = [matched[i]["matched"][0] for i in range(8)]
ib = [matched[i]["ib"][0] for i in range(8)]
ax.bar(idx - w, sh, w, label="SHARED (native 25% test)")
ax.bar(idx, mt, w, label="MATCHED (own protocol test)")
ax.bar(idx + w, ib, w, label="input LDCT baseline", alpha=0.6)
ax.set_xticks(idx); ax.set_xticklabels([f"c{i}\n{pats[i]}" for i in range(8)], fontsize=8)
ax.set_ylabel("PSNR (dB)"); ax.set_title("Final per-client PSNR: eval protocol vs input baseline")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

# (4) SSIM matched + enhancement gain (matched - input)
ax = axes[1, 1]
ss = [matched[i]["matched"][1] for i in range(8)]
gain = [matched[i]["matched"][0] - matched[i]["ib"][0] for i in range(8)]
ax.bar(idx - w/2, ss, w, label="SSIM (matched)", color="tab:green")
ax.set_xticks(idx); ax.set_xticklabels([f"c{i}\n{pats[i]}" for i in range(8)], fontsize=8)
ax.set_ylabel("SSIM"); ax.set_ylim(0.9, 1.0)
ax2 = ax.twinx()
ax2.bar(idx + w/2, gain, w, label="PSNR gain vs input", color="tab:orange", alpha=0.7)
ax2.set_ylabel("PSNR gain (dB)")
ax.set_title("Final matched SSIM & enhancement gain over input")
ax.legend(loc="lower left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("/mnt/d/icassp/FedFuse_code/runs/fedfuse_stack_full_v2/fedfuse_20round_analysis.png", dpi=150)
print("saved: runs/fedfuse_stack_full_v2/fedfuse_20round_analysis.png")

# text summary
print("\nfinal per-client (SHARED / MATCHED / input base):")
for i in range(8):
    print(f"  c{i} {pats[i]:6s} {specs[i]:14s} "
          f"sh={matched[i]['shared'][0]:5.2f}/{matched[i]['shared'][1]:.3f}  "
          f"mt={matched[i]['matched'][0]:5.2f}/{matched[i]['matched'][1]:.3f}  "
          f"in={matched[i]['ib'][0]:5.2f}")
ms = np.mean([matched[i]['matched'][0] for i in range(8)])
print(f"\nMATCHED mean PSNR: {ms:.2f}")
