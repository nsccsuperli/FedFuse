"""DUGAN (Fan et al., 2019) — U-Net generator with dual (image + gradient)
domain discriminators and gradient-domain loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _grad(t):
    kx = torch.tensor([[-1., 0., 1.]], device=t.device).view(1, 1, 1, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(t, kx, padding=(0, 1))   # (B,1,H,W)
    gy = F.conv2d(t, ky, padding=(1, 0))   # (B,1,H,W)
    return gx, gy


class ConvBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True),
                               nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True))

    def forward(self, x):
        return self.b(x)


class UNetG(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.c1, self.c2, self.c3, self.c4 = (ConvBlock(1, ch), ConvBlock(ch, ch * 2),
                                              ConvBlock(ch * 2, ch * 4), ConvBlock(ch * 4, ch * 8))
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, 2)
        self.c5 = ConvBlock(ch * 8, ch * 4)
        self.u2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, 2)
        self.c6 = ConvBlock(ch * 4, ch * 2)
        self.u1 = nn.ConvTranspose2d(ch * 2, ch, 2, 2)
        self.c7 = ConvBlock(ch * 2, ch)
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(self.pool(c1))
        c3 = self.c3(self.pool(c2))
        c4 = self.c4(self.pool(c3))
        u3 = self.c5(torch.cat([self.u3(c4), c3], 1))
        u2 = self.c6(torch.cat([self.u2(u3), c2], 1))
        u1 = self.c7(torch.cat([self.u1(u2), c1], 1))
        return x + self.out(u1)


class Discriminator(nn.Module):
    def __init__(self, cin=1, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, ch, 3, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.BatchNorm2d(ch * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch * 2, ch * 4, 3, 2, 1), nn.BatchNorm2d(ch * 4), nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch * 4, 1))

    def forward(self, x):
        return self.net(x)


class DUGAN(nn.Module):
    """Container: generator + dual discriminators + gradient loss helpers."""

    def __init__(self, ch=64):
        super().__init__()
        self.G = UNetG(ch)
        self.D_img = Discriminator(1)
        self.D_grad = Discriminator(2)

    def forward(self, x):
        return self.G(x)

    @staticmethod
    def grad_pair(t):
        gx, gy = _grad(t)
        return torch.cat([gx, gy], 1)

    @staticmethod
    def gradient_loss(pred, tgt):
        pg = DUGAN.grad_pair(pred)
        tg = DUGAN.grad_pair(tgt)
        return F.mse_loss(pg, tg)
