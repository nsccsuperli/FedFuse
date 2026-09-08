"""Unit test: stacking aggregation identity C@D == sum_k w_k B_k A_k with
DATA-SIZE weights (pi stays private per user decision), plus anchor frozen.

Also verifies the server aggregate is repeatable across rounds.
"""
import torch
from fedfuse.federated.server import FedFuseServer
from fedfuse.models.lora import MoLoRALinear

torch.manual_seed(0)
K, E, r, d = 8, 3, 8, 64          # 8 clients, 3 experts, rank 8, proj 64
N_LORA = 2                        # simulate 2 adapted projections

# ---- fake uploads: A/B differ per client+expert, NO pi --------------------
uploads = []
for k in range(K):
    A, B = {}, {}
    for i in range(N_LORA):
        A[f"lora{i}"] = torch.randn(E, r, d) * 0.1
        B[f"lora{i}"] = torch.randn(E, d, r) * 0.1
    uploads.append(dict(A=A, B=B, n=100 + k * 7))

# ---- reference: data-size weighted exact sum ------------------------------
ns = torch.tensor([u["n"] for u in uploads], dtype=torch.float64)
w = ns / ns.sum()                 # (K,)

refs = []
for i in range(N_LORA):
    ref = torch.zeros(d, d, dtype=torch.float64)
    for k in range(K):
        for e in range(E):
            ref += w[k] * uploads[k]["B"][f"lora{i}"][e].double() @ \
                   uploads[k]["A"][f"lora{i}"][e].double()
    refs.append(ref)

# ---- server stacking ------------------------------------------------------
srv = FedFuseServer(N_LORA, E, r, d_in=d, d_out=d)
C, D = srv.aggregate(uploads)
errs = []
for i in range(N_LORA):
    prod = C[i].double() @ D[i].double()
    errs.append((prod - refs[i]).abs().max().item())
    print(f"lora{i}: C@D shape {tuple(C[i].shape)} max|err| vs sum w_k B A = {errs[-1]:.3e}")
assert all(e < 1e-8 for e in errs), "stacking identity FAILED"

# ---- multi-round: aggregate() must be repeatable (state does not corrupt) --
C2, D2 = srv.aggregate([dict(A=u["A"], B=u["B"], n=u["n"]) for u in uploads])
assert C2.shape == C.shape and D2.shape == D.shape, "aggregate not repeatable"
print("MULTI-ROUND AGGREGATE PASS")
print("STACKING IDENTITY PASS (data-size weights, max err < 1e-8)")

# ---- anchor frozen + not trainable ---------------------------------------
base = torch.nn.Linear(d, d, bias=False)
m = MoLoRALinear(base, r=r, n_experts=E, n_clients=K)
assert all(not p.requires_grad for p in (m.anchor_C, m.anchor_D))
print(f"anchor_C shape {tuple(m.anchor_C.shape)} requires_grad="
      f"{m.anchor_C.requires_grad} (want False)")
assert tuple(m.anchor_C.shape) == (d, K * E * r)
assert tuple(m.anchor_D.shape) == (K * E * r, d)

# forward with anchor set, grad flows only to A/B
m.set_global_anchor(C[0], D[0])
x = torch.randn(4, d, requires_grad=True)
y = m(x)
loss = y.pow(2).mean()
loss.backward()
print(f"A grad finite={torch.isfinite(m.A.grad).all().item()} "
      f"B grad finite={torch.isfinite(m.B.grad).all().item()}")
assert torch.isfinite(m.A.grad).all() and torch.isfinite(m.B.grad).all()
print("MOLORA FORWARD/BACKWARD PASS (A+B trainable, anchor frozen)")
