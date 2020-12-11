import os
import torch
import torchvision
import torchvision.transforms as T
import PIL

def colouredMNIST(data, filename=None, rewrite=False):
    if filename is not None:
        path = os.path.join("data", "ColouredMNIST", filename)
        if os.path.isfile(path):
            return torch.load(path)

    batch_size = data.shape[0]
    lena = PIL.Image.open("data/Lenna.png")

    data = data.expand(batch_size, 3, 28, 28) > 0.5
    coloured_data = torch.zeros(data.shape, dtype=torch.float)

    transform = T.Compose([
        T.Resize((128, 128)),
        T.RandomCrop((28, 28)),
        T.ToTensor(),
    ])

    for i in range(batch_size):
        coloured_data[i] = transform(lena)
        coloured_data[i, data[i]] = 1 - coloured_data[i, data[i]] 

    if filename is not None:
        os.makedirs(os.path.join("data", "ColouredMNIST"), exist_ok=True)
        torch.save(coloured_data, path)

    return coloured_data

