import time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from models  import *

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

class GMVAE(nn.Module):
    def __init__(self,
                 frame_size,
                 latent_size,
                 num_condition_frames,
                 num_future_predictions,
                 normalization,
                 num_experts):

        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self.mode = normalization.get("mode")
        self.data_max = normalization.get("max")
        self.data_min = normalization.get("min")
        self.data_avg = normalization.get("avg")
        self.data_std = normalization.get("std")

        hidden_size = 256
        args = (
            frame_size,
            latent_size,
            hidden_size,
            num_condition_frames,
            num_future_predictions,
        )

        self.encoder = Encoder(*args)

        self.decoder = MixedDecoder(*args, num_experts)

    def normalize(self, t):
        if self.mode == "minmax":
            return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t):
        if self.mode == "minmax":
            return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
        elif self.mode == "zscore":
            return t * self.data_std + self.data_avg
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def encode(self, x, c):
        _, mu, logvar = self.encoder(x, c)
        return mu, logvar

    def forward(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        return self.decoder(z, c), mu, logvar

    def sample(self, z, c, deterministic=False):
        return self.decoder(z, c)
