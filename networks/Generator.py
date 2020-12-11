import torch
import torch.nn as nn


class SmallGeneratorNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SmallGeneratorNet, self).__init__()
        self.seq = nn.Sequential(
            self.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            self.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            self.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            self.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.seq(x)


class GeneratorNet(nn.Module):
    pass
