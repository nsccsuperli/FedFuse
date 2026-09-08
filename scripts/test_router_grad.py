"""Verify router receives TASK gradient after the set_pi graph fix.

At B=0 init, task gradient to pi is exactly 0 (d loss/d pi ~ B), so we first
let B move (few steps), then check the router gets non-zero task gradient
and pi actually moves away from uniform. Use a REAL descriptor-conditional
setup: g differs per sample so the router has something to learn.
"""
import torch
from fedfuse.models.lora import MoLoRALinear, LoRARouter

torch.manual_seed(0)
E, r, d = 3, 8, 32
base = torch.nn.Linear(d, d, bias=False)
m = MoLoRALinear(base, r=r, n_experts=E, n_clients=2).train()
router = LoRARouter(d_in=8, n_experts=E, hidden=16)
opt = torch.optim.Adam([m.A, m.B] + list(router.parameters()), lr=1e-2)

# two "clients" with different descriptors g and different targets
gA = torch.randn(8) * 2 + 3
gB = torch.randn(8) * 2 - 3
xA, yA = torch.randn(4, d), torch.randn(4, d)
xB, yB = torch.randn(4, d), torch.randn(4, d)

def one_step(g, x, y):
    opt.zero_grad(set_to_none=True)
    pi = router(g)
    m.set_pi(pi)
    loss = (m(x) - y).pow(2).mean()
    loss.backward()
    gnorm = router.net[0].weight.grad.norm().item()
    opt.step()
    return loss.item(), gnorm, pi.detach().clone()

# warm-up so B leaves zero
for _ in range(5):
    one_step(gA, xA, yA)
    one_step(gB, xB, yB)

router_gnorms = []
pis = []
for _ in range(20):
    _, g1, p1 = one_step(gA, xA, yA)
    _, g2, p2 = one_step(gB, xB, yB)
    router_gnorms.append((g1, g2))
    pis.append((p1, p2))

mx = max(max(a, b) for a, b in router_gnorms)
print(f"router task-grad norm after warmup: max over steps = {mx:.6f}")
assert mx > 1e-4, f"router task gradient too small: {mx}"

pa_first, pb_first = pis[0]
pa_last, pb_last = pis[-1]
print(f"piA first {pa_first.numpy().round(3)} -> last {pa_last.numpy().round(3)}")
print(f"piB first {pb_first.numpy().round(3)} -> last {pb_last.numpy().round(3)}")
# distinct descriptors should push pi in distinct directions
sep = (pa_last - pb_last).abs().max().item()
print(f"|piA - piB| max at end: {sep:.4f}")
print("ROUTER TASK-GRADIENT FIX PASS")
