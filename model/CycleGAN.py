import os
import torch
import torch.nn as nn
import itertools

from networks import SmallDiscriminatorNet, SmallGeneratorNet
from model.Losses import mathGANLoss, classicGANLoss, MSEGanLoss, reconstructionLoss

device = torch.device("cuda")
BATCH_SIZE = 64 # more better ?
LR = 2e-4
epochs = 2
n_disc = 1 # try == 1?

ADAM_BETA = (0.0, 0.9)
recon_lam = 10

class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()

        self.discriminator_X = SmallDiscriminatorNet(1).to(device)
        self.discriminator_Y = SmallDiscriminatorNet(3).to(device)

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
    
    
    def translate_XY(self, X):
        with torch.no_grad():
            return self.generator_Y(X * 2 - 1) * 0.5 + 0.5

    def translate_YX(self, Y):
        with torch.no_grad():
            return self.generator_X(Y * 2 - 1) * 0.5 + 0.5

    def disc_loss(self, X, Y):
        X_generated = self.generator_X(Y)
        Y_generated = self.generator_Y(X)

        X_true, X_false = self.discriminator_X(X), self.discriminator_X(X_generated)
        Y_true, Y_false = self.discriminator_Y(Y), self.discriminator_Y(Y_generated)

        disc_loss = MSEGanLoss(X_true, X_false) + MSEGanLoss(Y_true, Y_false)
        return disc_loss


    def gen_loss(self, X, Y):
        X_generated = self.generator_X(Y)
        Y_generated = self.generator_Y(X)

        X_true, X_false = self.discriminator_X(X), self.discriminator_X(X_generated)
        Y_true, Y_false = self.discriminator_Y(Y), self.discriminator_Y(Y_generated)

        X_reconstructed = self.generator_X(Y_generated)
        Y_reconstructed = self.generator_Y(X_generated)

        gen_loss = MSEGanLoss(X_false, X_true) + MSEGanLoss(Y_false, Y_true) + \
                recon_lam * (reconstructionLoss(X, X_reconstructed) + reconstructionLoss(Y, Y_reconstructed))

        return gen_loss


    def train(self, X_data, Y_data):
        X_data = 2 * X_data - 1
        Y_data = 2 * Y_data - 1

        X_loader = torch.utils.data.DataLoader(X_data.to(device), batch_size=BATCH_SIZE, shuffle=True) 
        Y_loader = torch.utils.data.DataLoader(Y_data.to(device), batch_size=BATCH_SIZE, shuffle=True) 

        for epoch in range(epochs):
            for i, (X, Y) in enumerate(zip(X_loader, Y_loader)):
                disc_loss = self.disc_loss(X, Y)

                self.optimizer_discriminators.zero_grad()
                disc_loss.backward()
                self.optimizer_discriminators.step()

                if i % n_disc == 0:
                    gen_loss = self.gen_loss(X, Y)
                    self.optimizer_generators.zero_grad()
                    gen_loss.backward()
                    self.optimizer_generators.zero_grad()
                else:
                    gen_loss = 0

                print(disc_loss, gen_loss)
                # TODO logging
                # TODO optimize generator not every iteration?
            torch.save(self.state_dict(), os.path.join("generated", "CycleGAN"))
