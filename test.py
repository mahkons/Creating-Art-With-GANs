import os
import torch
import torchvision
import matplotlib.pyplot as plt

from model.CycleGAN import CycleGAN
from addColour import colouredMNIST


if __name__ == "__main__":
    model = CycleGAN().to("cpu")
    model.load_state_dict(torch.load(os.path.join("generated", "CycleGAN"), map_location="cpu"))

    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None).data.float().unsqueeze(1) / 255.
    coloured_test = colouredMNIST(mnist_test, "test.torch")


    perm = torch.randperm(mnist_test.shape[0])
    samples = mnist_test[perm[:5]]
    transformed_samples = model.translate_XY(samples)
    coloured_samples = coloured_test[perm[:5]]
    transformed_coloured_samples = model.translate_YX(coloured_samples)

    grid_img = torchvision.utils.make_grid(
            torch.cat([samples.expand(5, 3, 28, 28), transformed_samples, coloured_samples, transformed_coloured_samples.expand(5, 3, 28, 28)]),
        nrow=5)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
        


