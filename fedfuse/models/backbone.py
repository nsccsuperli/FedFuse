"""Frozen DINOv3 ViT backbone with MoLoRA-injected attention (Sec. III-B).

Loads DINOv3 from a LOCAL directory (gated HF repo — download weights yourself
and point `dinov3_path` in the config at the folder containing config.json and
model weights). Falls back to any HF model id if a string id is given.

We aggregate patch tokens from L=4 intermediate layers and reshape them into
multi-scale feature maps. The backbone stays frozen; gradients flow only into
the LoRA up-projections.
"""
import torch
import torch.nn as nn

from .lora import MoLoRALinear

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def _find_attn_qv(module: nn.Module):
    """Recursively locate the submodule holding query/value projections
    (handles Dinov2Attention->Dinov2SelfAttention nesting and q_proj variants)."""
    for q_name, v_name in (("query", "value"), ("q_proj", "v_proj")):
        if hasattr(module, q_name) and hasattr(module, v_name):
            return module, q_name, v_name
    for child in module.children():
        try:
            return _find_attn_qv(child)
        except AttributeError:
            continue
    raise AttributeError(f"no q/v projections found under {type(module).__name__}")


class DinoV3Backbone(nn.Module):
    def __init__(self, path: str, n_layers: int = 24, lora_r: int = 8,
                 n_experts: int = 3, n_clients: int = 1,
                 lora_alpha: float = 16.0,
                 tap_layers=(5, 11, 17, 23), seed: int = 1234):
        super().__init__()
        from transformers import AutoModel
        self.vit = AutoModel.from_pretrained(path)
        for p in self.vit.parameters():
            p.requires_grad_(False)
        self.embed_dim = self.vit.config.hidden_size
        self.tap_layers = list(tap_layers)
        self.patch = getattr(self.vit.config, "patch_size", 16)

        # inject MoLoRA into q/v of every attention block.
        # Layer-list location differs across HF releases:
        #   DINOv3 (transformers>=5.x):  vit.model.layer  (encoder nested under .model)
        #   DINOv2 / older ViT:          vit.encoder.layer
        #   legacy layouts:              vit.layer
        if hasattr(self.vit, "model") and hasattr(self.vit.model, "layer"):
            layers = self.vit.model.layer
        elif hasattr(self.vit, "encoder") and hasattr(self.vit.encoder, "layer"):
            layers = self.vit.encoder.layer
        elif hasattr(self.vit, "layer"):
            layers = self.vit.layer
        else:
            raise AttributeError(
                f"cannot locate transformer layer list in {type(self.vit).__name__}")
        n_layers = len(layers)
        if max(self.tap_layers) >= n_layers:
            # rescale taps proportionally (e.g. 24-layer taps on a 12-layer dev model)
            span = max(self.tap_layers) + 1
            self.tap_layers = sorted({min(int(t * n_layers / span), n_layers - 1)
                                      for t in self.tap_layers})
        self._lora_modules = nn.ModuleList()
        for blk in layers:
            attn = blk.attention if hasattr(blk, "attention") else blk
            host, q_name, v_name = _find_attn_qv(attn)
            for proj_name in (q_name, v_name):
                base = getattr(host, proj_name)
                lora = MoLoRALinear(base, r=lora_r, n_experts=n_experts,
                                    n_clients=n_clients,
                                    alpha=lora_alpha, seed=seed)
                setattr(host, proj_name, lora)
                self._lora_modules.append(lora)

        mean = IMAGENET_MEAN.view(1, 3, 1, 1)
        std = IMAGENET_STD.view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def lora_modules(self):
        return list(self._lora_modules)

    def set_global_anchor(self, C: torch.Tensor, D: torch.Tensor):
        """Broadcast server-stacked (C, D) to every injected q/v module."""
        for m in self._lora_modules:
            m.set_global_anchor(C, D)

    def zero_anchors(self):
        """Round 0: global branch is identically zero until first aggregation."""
        with torch.no_grad():
            for m in self._lora_modules:
                m.anchor_C.zero_()
                m.anchor_D.zero_()

    def set_pi(self, pi):
        for m in self._lora_modules:
            m.set_pi(pi)

    def forward(self, x01: torch.Tensor,
                pi: torch.Tensor | None = None) -> list:
        """x01: (B,1,H,W) in [0,1] -> list of token maps (B,C,h,w) at tap layers.
        pi: (E,) routing weights with graph (training) or None -> buffer (eval)."""
        x = x01.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        out = self.vit(pixel_values=x, output_hidden_states=True)
        hs = out.hidden_states                     # tuple of (B, 1+N, C) incl. embeddings
        B, _, H, W = x01.shape
        h, w = H // self.patch, W // self.patch
        feats = []
        for l in self.tap_layers:
            t = hs[l + 1]                          # hidden_states[0] = embeddings
            t = t[:, -h * w:, :]                   # drop cls/register tokens
            feats.append(t.transpose(1, 2).reshape(B, self.embed_dim, h, w))
        return feats
