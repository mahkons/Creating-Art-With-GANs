import torch
import torchvision
import matplotlib.pyplot as plt

from model.CycleGAN import CycleGAN
from addColour import colouredMNIST

if __name__ == "__main__":
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None).data.float().unsqueeze(1) / 255.
    coloured_train = colouredMNIST(mnist_train, "train.torch")

    model = CycleGAN()
    model.train(mnist_train, coloured_train)
