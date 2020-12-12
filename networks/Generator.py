import torch
import torch.nn as nn
from utils.nets import define_G


class SmallGeneratorNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SmallGeneratorNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.seq(x)


class GeneratorNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GeneratorNet, self).__init__()
        self.seq = define_G(in_channels, out_channels, ngf=64, netG='resnet_9blocks')

    def forward(self, x):
        return self.seq(x)
