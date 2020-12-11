import torch
import torchvision
import matplotlib.pyplot as plt

from model.CycleGAN import CycleGAN
from addColour import colouredMNIST


if __name__ == "__main__":
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None).data.float().unsqueeze(1)
    coloured_train = colouredMNIST(mnist_train, "train.torch")

    model = CycleGAN()
    model.train(mnist_train, coloured_train)

    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None).data.float().unsqueeze(1)
    coloured_test = colouredMNIST(mnist_test, "test.torch")
