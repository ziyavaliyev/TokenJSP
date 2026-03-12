import sys, os
sys.path.append(os.path.abspath("."))
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from graph_features import clb
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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
        #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32) # TODO: FOR TAKING THE MEAN
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.T, latent_dim), dtype=np.float32)

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
            z=z_nodes
            """if self.pool == "sum":
                z = z_nodes.sum(dim=0)
            elif self.pool == "max":
                z = z_nodes.max(dim=0).values
            else:
                z = z_nodes.mean(dim=0)"""#TODO: TO TAKE THE MEAN

        return z.cpu().numpy().astype(np.float32)
    
class GAEFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        encoder,
        n_jobs,
        n_machines,
        features_dim=32,
        pool="mean",
    ):
        super().__init__(observation_space, features_dim)
        self.encoder = encoder
        self.n_jobs = int(n_jobs)
        self.n_machines = int(n_machines)
        self.T = self.n_jobs * self.n_machines
        self.pool = pool

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: [B, T, T + n_machines + 1]
        device = observations.device
        batch_size = observations.shape[0]
        outs = []

        for b in range(batch_size):
            obs = observations[b]  # [T, T + n_machines + 1]

            A = obs[:, :self.T]  # [T, T]
            base = obs[:, self.T:self.T + self.n_machines + 1]  # [T, n_machines+1]
            machine_oh = base[:, :self.n_machines]  # [T, n_machines]

            # clb currently works with numpy, so convert just this sample
            A_np = A.detach().cpu().numpy().astype(np.float32)
            base_np = base.detach().cpu().numpy().astype(np.float32)
            clb_data = clb(A_np, base_np).astype(np.float32, copy=False)

            clb_tensor = torch.tensor(clb_data, dtype=torch.float32, device=device)

            # "is_scheduled" is not tracked here yet, so use zeros for now
            is_scheduled = torch.zeros((self.T, 1), dtype=torch.float32, device=device)

            x = torch.cat([machine_oh, is_scheduled, clb_tensor], dim=1)

            src, dst = torch.nonzero(A > 0, as_tuple=True)
            if src.numel() > 0:
                edge_index = torch.stack([src, dst], dim=0).long()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

            z_nodes = self.encoder(x, edge_index)

            if self.pool == "sum":
                z = z_nodes.sum(dim=0)
            elif self.pool == "max":
                z = z_nodes.max(dim=0).values
            else:
                z = z_nodes.mean(dim=0)

            outs.append(z)

        return torch.stack(outs, dim=0)  # [B, latent_dim]

class ScheduleFlagWrapper(gym.Wrapper):
    def __init__(self, env, n_jobs: int, n_machines: int):
        super().__init__(env)
        self.T = int(n_jobs) * int(n_machines)
        self.is_scheduled = np.zeros((self.T, 1), dtype=np.float32)

        old_space = env.observation_space
        assert len(old_space.shape) == 2, "Expected matrix observation before GAE wrapping"
        h, w = old_space.shape

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(h, w + 1),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.is_scheduled[...] = 0.0
        return self._augment(obs), info

    def step(self, action):
        a = int(action)
        if 0 <= a < self.T:
            self.is_scheduled[a, 0] = 1.0

        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info

    def _augment(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return np.concatenate([obs, self.is_scheduled], axis=1).astype(np.float32)