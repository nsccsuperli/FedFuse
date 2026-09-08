"""WGAN-VGG (Yang et al., 2018) — WGAN-GP with VGG19 perceptual loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetGen(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        def cb(i, o, k=4, s=2):
            return nn.Sequential(nn.Conv2d(i, o, k, s, 1), nn.InstanceNorm2d(o),
                                 nn.LeakyReLU(0.2, True))
        self.e1 = nn.Sequential(nn.Conv2d(1, ch, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.e2, self.e3, self.e4 = cb(ch, ch * 2), cb(ch * 2, ch * 4), cb(ch * 4, ch * 8)
        def db(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1),
                                 nn.InstanceNorm2d(o), nn.ReLU(True))
        self.d3, self.d2, self.d1 = db(ch * 8, ch * 4), db(ch * 8, ch * 2), db(ch * 4, ch)
        self.out = nn.ConvTranspose2d(ch * 2, 1, 4, 2, 1)

    def forward(self, x):
        e1, e2, e3, e4 = self.e1(x), self.e2(self.e1(x)), None, None
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        d3 = self.d3(e4)
        d2 = self.d2(torch.cat([d3, e3], 1))
        d1 = self.d1(torch.cat([d2, e2], 1))
        return x + self.out(torch.cat([d1, e1], 1))


class PatchCritic(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.InstanceNorm2d(ch * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.InstanceNorm2d(ch * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1), nn.InstanceNorm2d(ch * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch * 8, 1, 3, 1, 1))

    def forward(self, x):
        return self.net(x).mean(dim=(1, 2, 3))


class VGGPerceptual(nn.Module):
    """VGG19 feature loss on relu1_1/2_1/3_1 (WGAN-VGG paper setting)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        feats = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in feats.parameters():
            p.requires_grad_(False)
        self.feats = feats
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        y = (y.repeat(1, 3, 1, 1) - self.mean) / self.std
        loss = 0.0
        for i, layer in enumerate(self.feats):
            x, y = layer(x), layer(y)
            if i in (0, 3, 7, 12):
                loss = loss + F.l1_loss(x, y)
        return loss


def gradient_penalty(critic, real, fake):
    B = real.shape[0]
    eps = torch.rand(B, 1, 1, 1, device=real.device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    out = critic(interp)
    grad = torch.autograd.grad(out.sum(), interp, create_graph=True)[0]
    return ((grad.flatten(1).norm(2, dim=1) - 1) ** 2).mean()
