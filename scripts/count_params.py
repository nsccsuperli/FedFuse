"""Count DINOv3 ViT-L/16 parameters and LoRA (r=8) trainable-parameter budgets.

Reads the locally downloaded model (no training, no weight writes).
Reports:
  1. total model parameters
  2. standard LoRA r=8 adapted onto q/v  -> trainable count
  3. standard LoRA r=8 adapted onto q/k/v/o -> trainable count
  4. (reference) project MoLoRA r=8, E=3, q/v, only-B trained -> per-round comms
"""
import torch
import transformers

MODEL_DIR = "/mnt/d/icassp/FedFuse_code/models/dinov3"
R = 8
E = 3  # MoLoRA experts (project default)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    print(f"transformers {transformers.__version__}  torch {torch.__version__}")
    model = transformers.AutoModel.from_pretrained(
        MODEL_DIR, low_cpu_mem_usage=True)
    model.eval()

    total = count_params(model)
    cfg = model.config
    print(f"\n== model: {cfg.model_type}  hidden={cfg.hidden_size} "
          f"layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads} "
          f"patch={cfg.patch_size} registers={cfg.num_register_tokens}")

    # --- locate attention projections inside every block -------------------
    # locate transformer block list (DINOv3 nests encoder under .model)
    if hasattr(model, "model") and hasattr(model.model, "layer"):
        layers = model.model.layer
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        layers = model.encoder.layer
    else:
        raise AttributeError("cannot locate transformer layer list")
    attn = layers[0].attention if hasattr(layers[0], "attention") else layers[0]
    print(f"\nattention module type: {type(attn).__name__}")
    print("linear projections found in block 0:")
    for name, m in attn.named_modules():
        if isinstance(m, torch.nn.Linear):
            print(f"  {name:28s} {m.in_features:>5d} -> {m.out_features:<5d}")

    def proj_sel(kind):
        """kind in {'q','v','k','o'} -> list of (name, in_f, out_f) per block."""
        sel = []
        for blk in layers:
            a = blk.attention if hasattr(blk, "attention") else blk
            for name, m in a.named_modules():
                if not isinstance(m, torch.nn.Linear):
                    continue
                nm = name.lower()
                if kind == "q" and ("q_proj" in nm or nm.endswith("query")
                                    or "query" in nm and "key" not in nm
                                    and "value" not in nm and "out" not in nm):
                    if m.in_features == m.out_features:
                        sel.append(m)
                elif kind == "v" and ("v_proj" in nm or nm.endswith("value")
                                      or "value" in nm):
                    if m.in_features == m.out_features:
                        sel.append(m)
                elif kind == "k" and ("k_proj" in nm or nm.endswith("key")
                                      or "key" in nm):
                    if m.in_features == m.out_features:
                        sel.append(m)
                elif kind == "o" and ("o_proj" in nm or "out_proj" in nm
                                      or nm.endswith("proj") and "q" not in nm
                                      and "k" not in nm and "v" not in nm):
                    if m.in_features == m.out_features:
                        sel.append(m)
        return sel

    qs, vs, ks, os_ = (proj_sel(k) for k in ("q", "v", "k", "o"))
    print(f"\nmatched per block: q={len(qs)} v={len(vs)} k={len(ks)} "
          f"o={len(os_)}  (expect {cfg.num_hidden_layers} each)")

    # --- standard LoRA: A (d_in x r) + B (r x d_out), both trainable -------
    def lora_trainable(proj_list):
        return sum(m.in_features * R + R * m.out_features
                   for m in proj_list)

    t_qv = lora_trainable(qs + vs)
    t_qkvo = lora_trainable(qs + ks + vs + os_)

    # --- reference: project MoLoRA, only B trained -------------------------
    t_molora_b = sum(E * m.out_features * R for m in qs + vs)  # B only

    print("\n================ RESULTS ================")
    print(f"total params (ViT-L/16 backbone): {total:,}")
    print(f"  = {total / 1e6:.2f} M")
    print(f"\nstandard LoRA r={R} trainable params:")
    print(f"  q/v  adapted : {t_qv:>10,}  ({t_qv / 1e6:.3f} M, "
          f"{100 * t_qv / total:.2f}% of backbone)")
    print(f"  q/k/v/o adapted: {t_qkvo:>10,}  ({t_qkvo / 1e6:.3f} M, "
          f"{100 * t_qkvo / total:.2f}% of backbone)")
    print(f"\nreference (project MoLoRA r={R} E={E} q/v, B only, frozen A):")
    print(f"  trainable (B, per-client comms/round): {t_molora_b:,} "
          f"({t_molora_b / 1e6:.3f} M, {100 * t_molora_b / total:.2f}%)")


if __name__ == "__main__":
    main()
