import torch
import torch.nn as nn
import itertools

from networks import SmallDiscriminatorNet, SmallGeneratorNet
from model.Losses import mathGANLoss, classicGANLoss, MSEGanLoss, reconstructionLoss

device = torch.device("cpu")
BATCH_SIZE = 256
LR = 1e-4
epochs = 2

ADAM_BETA = (0.0, 0.9)
recon_lam = 10

class CycleGAN():
    def __init__(self):

        self.discriminator_X = SmallDiscriminatorNet(1).to(device)
        self.discriminator_Y = SmallGeneratorNet(3).to(device)

        self.generator_X = SmallGeneratorNet(3, 1).to(device)
        self.generator_Y = SmallGeneratorNet(1, 3).to(device)

        self.optimizer_discriminators = torch.optim.Adam(
                itertools.chain(self.discriminator_X.parameters(), self.discriminator_Y.parameters()),
                lr=LR,
                betas=ADAM_BETA
        )

        self.optimizer_generators = torch.optim.Adam(
                itertools.chain(self.generator_X.parameters(), self.generator_Y.parameters()),
                lr=LR,
                betas=ADAM_BETA
        )
        # TODO LR schedulers?


    def optimize(self, X, Y):
        X_generated = self.generator_X(Y)
        Y_generated = self.generator_Y(X)

        X_true, X_false = self.discriminator_X(X), self.discriminator_X(X_generated)
        Y_true, Y_false = self.discriminator_Y(Y), self.discriminator_Y(Y_generated)

        X_reconstracted = self.generator_X(Y_generated)
        Y_reconstracted = self.generator_Y(X_generated)

        disc_loss = -(MSEGanLoss(X_true, X_false) + MSEGanLoss(Y_true, Y_false))
        gen_loss = MSEGanLoss(X_false, X_true) + MSEGanLoss(Y_false, Y_true) + \
                recon_lam * (reconstructionLoss(X, X_reconstracted) + reconstructionLoss(Y, Y_reconstracted))

        self.optimizer_discriminators.zero_grad()
        disc_loss.backward()
        self.optimizer_discriminators.step()

        self.optimizer_generators.zero_grad()
        gen_loss.backward()
        self.optimizer_generators.zero_grad()


    def train(self, X_data, Y_data):
        X_loader = torch.utils.data.DataLoader(X_data.to(device), batch_size=batch_size, shuffle=True) 
        Y_loader = torch.utils.data.DataLoader(Y_data.to(device), batch_size=batch_size, shuffle=True) 

        for epoch in range(epochs):
            for X, Y in zip(X_loader, Y_loader):
                self.optimize(X, Y)

                # TODO logging
                # TODO optimize generator not every iteration?
