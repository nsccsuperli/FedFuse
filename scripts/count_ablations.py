"""Count trainable params per ablation branch on the REAL DINOv3 ViT-L/16.

Branches for the comparison table:
  1. FULL DINOv3 fine-tune : every weight of the ViT backbone
                             (what full fine-tuning would train)
  2. PEFT DINOv3 only      : project MoLoRA q/v, r=8, E=3, A+B both trainable
                             (current paper method applied to the backbone)
  3. Lightweight CNN only  : noise_branch + 4 gated fusions + decoder
                             (image-domain path without any backbone adapter)
  4. FedFuse TOTAL         : 2 + LoRARouter + 3  (dual-path + PEFT)

Cross-check: requires_grad parameters of the assembled FedFuseNet must equal
A/B + CNN exactly (the ViT body is frozen, anchors are buffers, router is a
separate module). Loads the model once, CPU only, no training / no writes.
"""
import torch
import transformers

from fedfuse.models import FedFuseNet, LoRARouter

MODEL_DIR = "/mnt/d/icassp/FedFuse_code/models/dinov3"
R, E, K = 8, 3, 8          # project defaults: rank 8, E=3 experts, K=8 clients


def numel(params):
    return sum(p.numel() for p in params)


def grp(named_params, pred):
    """numel of named params whose name passes `pred` (prefix match on '.'-parts)."""
    n = 0
    for name, p in named_params:
        if p.requires_grad and pred(name):
            n += p.numel()
    return n


def main():
    print(f"transformers {transformers.__version__}  torch {torch.__version__}")
    net = FedFuseNet(MODEL_DIR, lora_r=R, n_experts=E, n_clients=K)  # CPU
    net.eval()
    vit = net.backbone.vit

    # ---- 2. PEFT: MoLoRA A+B on q/v of every block -------------------------
    # (computed first: section 1 subtracts it to report the pure ViT count)
    lora_params = [p for m in net.backbone.lora_modules() for p in m.parameters()
                   if p.requires_grad]
    n_lora = numel(lora_params)
    n_lora_mod = len(net.backbone.lora_modules())
    n_A = sum(1 for m in net.backbone.lora_modules() if m.A.requires_grad)
    n_B = sum(1 for m in net.backbone.lora_modules() if m.B.requires_grad)

    # ---- 1. full DINOv3 backbone (everything in the ViT) ------------------
    # NOTE: q_proj/v_proj are now MoLoRALinear wrappers whose .base still holds
    # the ORIGINAL frozen Linear weights, so vit.parameters() includes the
    # injected A/B factors too -- subtract n_lora to report the pure ViT-L/16
    # weight count (the number a full fine-tune would actually train).
    n_full_vit = numel(vit.parameters()) - n_lora
    cfg = vit.config
    print(f"\n== DINOv3 {cfg.model_type}: hidden={cfg.hidden_size} "
          f"layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads} "
          f"patch={cfg.patch_size} registers={cfg.num_register_tokens}")

    # ---- 3. lightweight CNN local modules ----------------------------------
    n_cnn = grp(net.named_parameters(), lambda nm: nm.startswith(
        ("noise_branch.", "fusions.", "decoder.")))
    # per-submodule split for the paper table
    n_nb = grp(net.named_parameters(), lambda nm: nm.startswith("noise_branch."))
    n_fu = grp(net.named_parameters(), lambda nm: nm.startswith("fusions."))
    n_dec = grp(net.named_parameters(), lambda nm: nm.startswith("decoder."))

    # ---- router (Eq.9): descriptor 288 -> experts ---------------------------
    router = LoRARouter(d_in=288, n_experts=E)
    n_router = numel(router.parameters())

    # ---- FedFuse total ------------------------------------------------------
    n_trainable_net = numel(p for p in net.parameters() if p.requires_grad)
    n_total_ff = n_trainable_net + n_router
    n_total_all = numel(net.parameters()) + numel(router.parameters())

    # ---- cross-checks -------------------------------------------------------
    assert n_A == n_B == n_lora_mod, (n_A, n_B, n_lora_mod)
    assert n_lora == E * R * 2 * cfg.hidden_size * n_lora_mod, n_lora
    assert n_trainable_net == n_lora + n_cnn, (n_trainable_net, n_lora, n_cnn)
    # sanity: the ViT body (incl. MoLoRA base weights) is truly frozen --
    # the only requires_grad tensors under backbone.vit.* are the injected
    # A/B factors (named within the vit namespace), so the count must equal
    # n_lora exactly; anything above that = accidentally unfrozen base weight.
    n_vit_train = grp(net.named_parameters(), lambda nm: nm.startswith("backbone.vit."))
    assert n_vit_train == n_lora, (n_vit_train, n_lora)

    print(f"\n================= ABLATION PARAM COUNT (r={R}, E={E}, K={K}) =================")
    rows = [
        ("DINOv3 full fine-tune (entire ViT)",      n_full_vit),
        ("  PEFT / MoLoRA q/v only (A+B, 48 mods)", n_lora),
        ("  LoRARouter (288->64->E)",               n_router),
        ("Lightweight CNN only (noise+fusion+dec)", n_cnn),
        ("    noise_branch",                         n_nb),
        ("    fusions x4 (incl. 1024->c_img proj)", n_fu),
        ("    decoder",                              n_dec),
        ("FedFuse TOTAL trainable (dual-path+PEFT)", n_total_ff),
    ]
    for label, n in rows:
        pct_vit = 100.0 * n / n_full_vit
        pct_ff = 100.0 * n / n_total_ff
        print(f"  {label:45s} {n:>12,}  {n/1e6:8.3f} M"
              f"  [{pct_vit:6.2f}% of ViT | {pct_ff:5.2f}% of FedFuse]")
    print(f"\n  whole FedFuseNet incl. frozen ViT + router: {n_total_all:,} "
          f"({n_total_all/1e6:.2f} M)  <- frozen body dominates, not trainable")
    print(f"\n  verification: FedFuse trainable == A/B ({n_lora:,}) "
          f"+ CNN ({n_cnn:,}) + router ({n_router:,}) "
          f"== {n_total_ff:,}  OK")
    print("  per-module sanity: d=1024, E=3, r=8 -> "
          f"{E*R*2*cfg.hidden_size:,} params x {n_lora_mod} modules "
          f"= {n_lora:,}  OK")


if __name__ == "__main__":
    main()
