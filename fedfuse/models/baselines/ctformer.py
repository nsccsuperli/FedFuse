"""CTformer (Wang et al., 2022) — convolution-free Token2Token dilated ViT
for LDCT denoising. Reproduction at patch scale (128x128 patches, overlapped
tokenization), full-image inference by tiling."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Token2Token(nn.Module):
    """Progressive overlapped re-tokenization: unfold -> reshape -> tokens."""

    def __init__(self, img_size=128, in_ch=1, embed=256, first=7, second=3):
        super().__init__()
        # stage 1: unfold k=7 s=4 -> tokens, project to embed
        p1, s1 = first, 4
        t1 = (img_size - p1) // s1 + 1
        self.proj1 = nn.Linear(p1 * p1 * in_ch, embed)
        # stage 2: re-fold to t1 x t1, unfold k=3 s=2
        p2, s2 = second, 2
        self.t1 = t1
        self.t2 = (t1 - p2) // s2 + 1
        self.proj2 = nn.Linear(p2 * p2 * embed, embed)
        self.embed = embed
        self.p1, self.s1, self.p2, self.s2 = p1, s1, p2, s2

    def forward(self, x):
        B = x.shape[0]
        t = F.unfold(x, self.p1, stride=self.s1)              # B, C*p1^2, L1
        t = rearrange(t, "b c l -> b l c")
        t = self.proj1(t)                                     # B, L1, E
        t = rearrange(t, "b (h w) e -> b e h w", h=self.t1)
        t = F.unfold(t, self.p2, stride=self.s2)              # B, E*p2^2, L2
        t = rearrange(t, "b c l -> b l c")
        t = self.proj2(t)                                     # B, L2, E
        return t


class DilatedBlock(nn.Module):
    """MHSA with token-dilation (CTformer uses dilated attention to enlarge
    the receptive field without conv) + FFN."""

    def __init__(self, dim, heads=8, dilation=1, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))
        self.dilation = dilation

    def forward(self, x, grid):
        B, L, C = x.shape
        h = w = grid
        y = self.norm1(x)
        if self.dilation > 1 and h % self.dilation == 0:
            # dilated attention: subsample tokens on a strided grid
            ys = rearrange(y, "b (h w) c -> b h w c", h=h)
            ys = ys[:, :: self.dilation, :: self.dilation, :]
            ys = rearrange(ys, "b h w c -> b (h w) c")
            k, v = ys, ys
        else:
            k, v = y, y
        a, _ = self.attn(y, k, v, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class CTformer(nn.Module):
    def __init__(self, patch=128, embed=256, depth=6, heads=8, dilations=(1, 2, 1, 2, 1, 1)):
        super().__init__()
        self.patch = patch
        self.t2t = Token2Token(img_size=patch, embed=embed)
        self.grid = self.t2t.t2
        n_tok = self.grid * self.grid
        self.pos = nn.Parameter(torch.zeros(1, n_tok, embed))
        nn.init.trunc_normal_(self.pos, std=0.02)
        dil = list(dilations) + [1] * max(0, depth - len(dilations))
        self.blocks = nn.ModuleList([DilatedBlock(embed, heads, dil[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(embed)
        self.head = nn.Linear(embed, embed)
        # token2image: fold tokens back via transposed-linear projection
        self.out = nn.Conv2d(embed, 1, 1)

    def forward_tokens(self, x):
        t = self.t2t(x) + self.pos
        for b in self.blocks:
            t = b(t, self.grid)
        t = self.norm(t)
        return rearrange(self.out(rearrange(self.head(t), "b (h w) c -> b c h w",
                                            h=self.grid)),
                         "b c h w -> b c h w")

    def forward(self, x):
        """x: (B,1,H,W) with H=W=self.patch during training."""
        B, _, H, W = x.shape
        assert H == self.patch and W == self.patch
        low = F.interpolate(self.forward_tokens(x), size=(H, W), mode="bilinear",
                            align_corners=False)
        return x + low

    @torch.no_grad()
    def forward_full(self, x, overlap=32):
        """Tiled inference for arbitrary 512x512 images."""
        B, _, H, W = x.shape
        p, s = self.patch, self.patch - overlap
        out = torch.zeros_like(x)
        wsum = torch.zeros_like(x)
        for yy in range(0, H - p + 1, s):
            for xx in range(0, W - p + 1, s):
                tile = x[..., yy:yy + p, xx:xx + p]
                out[..., yy:yy + p, xx:xx + p] += self.forward(tile)
                wsum[..., yy:yy + p, xx:xx + p] += 1
        return out / wsum.clamp_min(1)
