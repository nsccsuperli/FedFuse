"""RED-CNN (Chen et al., 2017) — residual encoder-decoder CNN, LDCT classic."""
import torch
import torch.nn as nn


class REDCNN(nn.Module):
    def __init__(self, ch=96, n_layers=5):
        super().__init__()
        enc, dec = [], []
        enc.append(nn.Sequential(nn.Conv2d(1, ch, 5, 1, 2), nn.ReLU(inplace=True)))
        for _ in range(n_layers - 1):
            enc.append(nn.Sequential(nn.Conv2d(ch, ch, 5, 1, 2), nn.ReLU(inplace=True)))
        for _ in range(n_layers - 1):
            dec.append(nn.Sequential(nn.ConvTranspose2d(ch, ch, 5, 1, 2),
                                     nn.ReLU(inplace=True)))
        dec.append(nn.ConvTranspose2d(ch, 1, 5, 1, 2))
        self.enc, self.dec = nn.ModuleList(enc), nn.ModuleList(dec)

    def forward(self, x):
        skips = []
        h = x
        for e in self.enc:
            h = e(h)
            skips.append(h)
        # mirror skips: dec_i (i=0..3) receives enc output of its mirror layer
        for i, d in enumerate(self.dec):
            h = d(h)
            if i < len(self.dec) - 1:
                h = h + skips[len(self.dec) - 2 - i]
        return x + h
