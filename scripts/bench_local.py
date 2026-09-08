"""Benchmark: per-step training time (batch 1/2/4) + full-slice eval time,
so we can size the 64-slice verification run correctly."""
import time
import torch
from torch.utils.data import DataLoader

from fedfuse.data import MayoLDCTDataset, TRAIN_PATIENTS
from fedfuse.losses import CompositeLoss
from fedfuse.models import FedFuseNet

DEV = "cuda"
torch.manual_seed(0)

net = FedFuseNet("models/dinov3", lora_r=8, n_experts=1).to(DEV)
trainable = ([m.B for m in net.backbone.lora_modules()]
             + [p for n, p in net.named_parameters()
                 if n.startswith(("noise_branch", "fusions", "decoder"))])
opt = torch.optim.AdamW(trainable, lr=1e-4)
loss_fn = CompositeLoss()

ds = MayoLDCTDataset("/mnt/d/icassp/datas/mayo_2016_npy",
                     TRAIN_PATIENTS[:1], max_slices=8, patch_size=256)

def run_steps(batch, n=3):
    ld = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2, drop_last=True)
    net.train()
    t0 = time.time()
    for i, (x, y) in enumerate(ld):
        if i >= n:
            break
        x, y = x.to(DEV), y.to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(x)
            loss, _ = loss_fn(pred.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    dt = (time.time() - t0) / n
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"batch={batch}: {dt:.1f} s/step  peak_mem={mem:.2f} GB")
    return dt

for b in (1, 2, 4):
    try:
        run_steps(b)
    except torch.cuda.OutOfMemoryError:
        print(f"batch={b}: OOM")
        break

# full-slice (512) eval timing, no grad
from fedfuse.data import build_test_dataset
net.eval()
te = build_test_dataset("/mnt/d/icassp/datas/mayo_2016_npy", max_slices=2)
tld = DataLoader(te, batch_size=1, num_workers=2)
t0 = time.time()
with torch.no_grad():
    for x, y in tld:
        x = x.to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = net(x)
print(f"full-512 eval: {(time.time()-t0)/2:.1f} s/slice")
