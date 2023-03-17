__author__ = 'clcp'
__version__ = '0.9'

'''
Adapted from https://colab.research.google.com/github/YannDubs/lossyless/blob/main/notebooks/minimal_code.ipynb
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck
import numpy as np


class ArrayCompressor(nn.Module):
    """Compressor for any vectors, by using an entropy bottleneck and MSE distortion."""

    def __init__(self, z_dim):
        super().__init__()
        self.bottleneck = EntropyBottleneck(z_dim)
        self.scaling = nn.Parameter(torch.ones(z_dim))
        self.biasing = nn.Parameter(torch.zeros(z_dim))
        self.is_updated = False

    def forward(self, z, y):
        z = (z + self.biasing) * self.scaling.exp()
        z_hat, q_z = self.bottleneck(z.unsqueeze(-1).unsqueeze(-1))
        z_hat = z_hat.squeeze() / self.scaling.exp() - self.biasing
        return z_hat, q_z.squeeze(), y.squeeze()

    def compress(self, z):
        if not self.is_updated:
            self.bottleneck.update(force=True)
            self.is_updated = True
        z = (z + self.biasing) * self.scaling.exp()
        return self.bottleneck.compress(z.unsqueeze(-1).unsqueeze(-1))

    def decompress(self, z_bytes):
        z_hat = self.bottleneck.decompress(z_bytes, [1, 1]).squeeze()
        return (z_hat / self.scaling.exp()) - self.biasing
    
    def set_hparams(self, params):
        self.hparams = params.__dict__


class CompressionTrainWrapper:
    def __init__(self, compressor, hparams):
        self.hparams = hparams
        self.module = compressor
        self.module.set_hparams(hparams)
        
    def loss(self, z, y):
        z_hat, q_z, _ = self.module(z, y)
        rate = -torch.log(q_z).sum(-1).mean()
        distortion = torch.norm(z - z_hat, p=1, dim=-1).mean()
        return distortion, rate # compression loss
    
    def coding_loss(self):
        return self.module.bottleneck.loss()
    
    def step(self, z, y, train=True):
        if train:
            self.optimizer.zero_grad()
            self.optimizer_coder.zero_grad()

        distortion, rate = self.loss(z, y)
        loss = distortion + self.hparams.lmbda * rate
        coder_loss = self.coding_loss()
        
        if train:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            coder_loss.backward()
            self.optimizer_coder.step()
            self.scheduler_coder.step()
        return loss.item(), rate.item(), distortion.item(), coder_loss.item()
    
    def configure_optimizers(self):
        param = [p for n, p in self.module.named_parameters() if not n.endswith(".quantiles")]
        quantile_param = [p for n, p in self.module.named_parameters() if n.endswith(".quantiles")]
        self.optimizer = torch.optim.Adam(param, lr=self.hparams.lr)
        self.optimizer_coder = torch.optim.Adam(quantile_param, lr=self.hparams.lr)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.hparams.lr_step)
        self.scheduler_coder = torch.optim.lr_scheduler.StepLR(self.optimizer_coder, self.hparams.lr_step)
    
    def to(self, device):
        self.module.to(device)