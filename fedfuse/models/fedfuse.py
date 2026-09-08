"""FedFuse dual-path fusion network (Sec. III-B).

Semantic structure path : frozen DINOv3 ViT + MoLoRA -> multi-scale F_sem^s
Image-domain noise path : lightweight CNN on [x, r, V(x)] -> F_img^s
Fine-grained fusion     : gated feature modulation (Eq. 5)
Decoder                 : lightweight, skip-connected, predicts residual R(x),
                          y_hat = x + R(x)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DinoV3Backbone
from ..physics import noise_residual, local_variance_map


def conv_bn(cin, cout, k=3, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2),
                         nn.GroupNorm(8, cout), nn.GELU())


def _cnn_ctx():
    """CNN-side modules compute in fp32: their conv backward is numerically
    unstable under bf16 autocast (NaN reproduced in multi-step training even
    after the loss was forced fp32; the ViT backbone stays bf16 for speed).
    bf16 speedup belongs to the 300M-param backbone, not these small convs."""
    return torch.autocast("cuda", enabled=False)


class NoiseBranch(nn.Module):
    """Shallow conv encoder with skip connections over [x, r, V(x)]."""

    def __init__(self, widths=(32, 64, 96, 128)):
        super().__init__()
        self.stem = conv_bn(3, widths[0])                       # 512
        self.enc1 = conv_bn(widths[0], widths[1], s=2)          # 256
        self.enc2 = conv_bn(widths[1], widths[2], s=2)          # 128
        self.enc3 = conv_bn(widths[2], widths[3], s=2)          # 64
        self.widths = widths

    def forward(self, x, r, v):
        with _cnn_ctx():
            t = torch.cat([x, r, v], dim=1)
            s1 = self.stem(t)        # 512^2
            s2 = self.enc1(s1)       # 256^2
            s3 = self.enc2(s2)       # 128^2
            s4 = self.enc3(s3)       # 64^2
        return [s1, s2, s3, s4]  # fine -> coarse


class GatedFusion(nn.Module):
    """F~ = F_img + gamma * sigmoid(conv([F_img, F_sem])) * F_sem  (Eq. 5)."""

    def __init__(self, c_img, c_sem):
        super().__init__()
        self.proj_sem = nn.Conv2d(c_sem, c_img, 1)
        self.gate = nn.Conv2d(c_img * 2, c_img, 3, 1, 1)
        self.gamma = nn.Parameter(torch.full((1, c_img, 1, 1), 1e-3))

    def forward(self, f_img, f_sem):
        with _cnn_ctx():
            f_sem = F.interpolate(f_sem, size=f_img.shape[-2:], mode="bilinear",
                                  align_corners=False)
            f_sem = self.proj_sem(f_sem)
            g = torch.sigmoid(self.gate(torch.cat([f_img, f_sem], dim=1)))
            return f_img + self.gamma * g * f_sem


class Decoder(nn.Module):
    def __init__(self, widths=(32, 64, 96, 128)):
        super().__init__()
        w = widths
        self.up3 = conv_bn(w[3] + w[2], w[2])      # 128^2
        self.up2 = conv_bn(w[2] + w[1], w[1])      # 256^2
        self.up1 = conv_bn(w[1] + w[0], w[0])      # 512^2
        self.head = nn.Conv2d(w[0], 1, 3, 1, 1)

    def forward(self, fused):
        with _cnn_ctx():
            s1, s2, s3, s4 = fused
            u3 = F.interpolate(s4, scale_factor=2, mode="bilinear", align_corners=False)
            u3 = self.up3(torch.cat([u3, s3], 1))
            u2 = F.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False)
            u2 = self.up2(torch.cat([u2, s2], 1))
            u1 = F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)
            u1 = self.up1(torch.cat([u1, s1], 1))
            return self.head(u1)


class FedFuseNet(nn.Module):
    def __init__(self, dinov3_path: str, widths=(32, 64, 96, 128),
                 lora_r=8, n_experts=3, n_clients=1, lora_alpha=16.0,
                 seed=1234, tap_layers=(5, 11, 17, 23)):
        super().__init__()
        self.backbone = DinoV3Backbone(dinov3_path, lora_r=lora_r,
                                       n_experts=n_experts,
                                       n_clients=n_clients,
                                       lora_alpha=lora_alpha,
                                       seed=seed, tap_layers=tap_layers)
        self.noise_branch = NoiseBranch(widths)
        self.fusions = nn.ModuleList([
            GatedFusion(widths[i], self.backbone.embed_dim) for i in range(4)])
        self.decoder = Decoder(widths)

    def set_pi(self, pi):
        self.backbone.set_pi(pi)

    def forward(self, x):
        r = noise_residual(x)
        v = local_variance_map(r)
        f_img = self.noise_branch(x, r, v)
        f_sem = self.backbone(x)
        fused = [self.fusions[i](f_img[i], f_sem[i]) for i in range(4)]
        return x + self.decoder(fused)

    # ------------------------------------------------------------------ #
    # parameter bookkeeping for federated exchange
    # ------------------------------------------------------------------ #
    def shared_parameters(self):
        """Communicated per round: LoRA factors A and B of every q/v module
        (global anchors never leave the server)."""
        return {f"lora{i}": {"A": m.A, "B": m.B}
                for i, m in enumerate(self.backbone.lora_modules())}

    def set_global_anchor(self, C, D):
        self.backbone.set_global_anchor(C, D)

    def zero_anchors(self):
        self.backbone.zero_anchors()

    def local_parameters(self):
        """Stay on the client: noise branch, fusion gates, decoder."""
        keys = []
        for n, _ in (list(self.noise_branch.named_parameters())
                     + list(self.fusions.named_parameters())
                     + list(self.decoder.named_parameters())):
            keys.append(n)
        return keys
