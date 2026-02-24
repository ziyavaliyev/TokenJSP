import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from graph_features import clb


class FrozenGAEObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, encoder, latent_dim, n_jobs, n_machines, device="cpu", include_duration_in_x=False, pool="mean"):
        super().__init__(env)
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.n_machines = int(n_machines)
        self.T = int(n_jobs) * self.n_machines
        self.include_duration_in_x = bool(include_duration_in_x)
        self.pool = pool

        self.is_scheduled = np.zeros((self.T, 1), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.is_scheduled[...] = 0.0
        return self.observation(obs), info

    def step(self, action):
        a = int(action)
        if 0 <= a < self.T:
            self.is_scheduled[a, 0] = 1.0
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        A = obs[:, :self.T]
        base = obs[:, self.T:self.T + self.n_machines + 1]
        machine_oh = base[:, :self.n_machines]
        #duration = base[:, self.n_machines:self.n_machines + 1]

        clb_data = clb(A, base).astype(np.float32, copy=False)
        
        x_np = np.concatenate([machine_oh, self.is_scheduled, clb_data], axis=1)

        src, dst = np.nonzero(A > 0)
        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long, device=self.device) if src.size else \
            torch.empty((2, 0), dtype=torch.long, device=self.device)

        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            z_nodes = self.encoder(x, edge_index)
            if self.pool == "sum":
                z = z_nodes.sum(dim=0)
            elif self.pool == "max":
                z = z_nodes.max(dim=0).values
            else:
                z = z_nodes.mean(dim=0)

        return z.cpu().numpy().astype(np.float32)