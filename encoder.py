import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Encoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, latent_dim: int = 128):
        super().__init__(observation_space, features_dim=latent_dim)
        in_dim = int(np.prod(observation_space.shape))  # flat vector length

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.view(obs.shape[0], -1)
        return self.net(obs)