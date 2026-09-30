"""Small CNN over stacked frames (3F channels) with BatchNorm, binary blockage-within-horizon head."""
import torch.nn as nn


def blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))


class BlockNet(nn.Module):
    def __init__(self, in_ch=24, d=128):
        super().__init__()
        self.net = nn.Sequential(blk(in_ch, 32), blk(32, 64), blk(64, 128), blk(128, d),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(d, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 2))

    def features(self, b):
        """Input of the last linear layer (used by prototype methods)."""
        return self.head[:-1](self.net(b["x"]))

    def forward(self, b):
        return self.head[-1](self.features(b))
