import torch
import torch.nn as nn
import torch.nn.functional as F

def mathGANLoss(X_true, X_false):
    return (F.logsigmoid(X_true) + F.logsigmoid(-X_false)).mean()

def classicGANLoss(X_true, X_false):
    return (F.logsigmoid(X_true) - F.logsigmoid(X_false)).mean()

# as in paper
def MSEGanLoss(X_true, X_false):
    return ((X_true - 1) ** 2 + X_false ** 2).mean()


# TODO try something WGAN-GP like?


# mean or sum?
def reconstructionLoss(X, X_recon):
    return torch.abs(X - X_recon).mean(dim=(1, 2, 3)).mean(dim=0)
