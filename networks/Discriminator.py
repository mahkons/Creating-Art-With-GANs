import torch
import torch.nn as nn
from utils.nets import define_D


class SmallDiscriminatorNet(nn.Module):
    def __init__(self, in_channels):
        super(SmallDiscriminatorNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(start_dim=1),
            nn.Linear(2048, 1)
        )

    def forward(self, x):
        return self.seq(x)


class DiscriminatorNet(nn.Module):
    def __init__(self, in_channels):
        super(DiscriminatorNet, self).__init__()
        self.seq = define_D(in_channels, ndf=64, netD='basic')

    def forward(self, x):
        return self.seq(x)
