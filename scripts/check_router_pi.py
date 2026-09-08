"""Post-run router check: did pi actually specialize after the set_pi fix?"""
import sys
import torch
sys.path.insert(0, "/mnt/d/icassp/FedFuse_code")
from fedfuse.models.lora import LoRARouter
from fedfuse.data import MayoLDCTDataset
from fedfuse.physics import global_descriptor
from fedfuse.data import client8_patients

ckpt = torch.load("/mnt/d/icassp/FedFuse_code/runs/fedfuse_routerfix_smoke/fedfuse_latest.pt",
                  map_location="cpu", weights_only=False)
routers = ckpt["router"]
print("clients with router state:", list(routers.keys()))

# per-client TRUE descriptors from their own data (same as training EMA start)
pats = client8_patients()
for cid in sorted(routers, key=int):
    ds = MayoLDCTDataset("/mnt/d/icassp/datas/mayo_2016_npy", [pats[int(cid)]],
                         max_slices=8)
    gs = []
    for i in range(min(4, len(ds))):
        x, _ = ds[i]
        gs.append(global_descriptor(x[None])[0])
    g_true = torch.stack(gs).mean(0)
    router = LoRARouter(d_in=288, n_experts=3)
    router.load_state_dict(routers[cid])
    router.eval()
    with torch.no_grad():
        pi_true = torch.softmax(router.net(g_true) / router.tau, dim=-1)
    print(f"  c{cid} (patient {pats[int(cid)]}): pi on own g = "
          f"{pi_true.numpy().round(3)}")
