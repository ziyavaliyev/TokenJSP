import os
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from gae_wrapper import FrozenGAEObsWrapper, GAEFeatureExtractor, ScheduleFlagWrapper


def mask_fn(env: gym.Env):
    return env.unwrapped.valid_action_mask()


def build_single_eval_env(jsp, encoder, latent_dim, n_jobs, n_machines, device, frozen):
    env = DisjunctiveGraphJspEnv(
        jps_instance=jsp,
        perform_left_shift_if_possible=True,
        normalize_observation_space=True,
        flat_observation_space=False,
        action_mode="task",
    )
    env = Monitor(env)
    env = ActionMasker(env, mask_fn)

    if frozen:
        env = FrozenGAEObsWrapper(
            env,
            encoder=encoder,
            latent_dim=latent_dim,
            n_jobs=n_jobs,
            n_machines=n_machines,
            device=device,
        )
    else:
        env = ScheduleFlagWrapper(env, n_jobs=n_jobs, n_machines=n_machines)

    return env


def validate_model(model, val_instances, encoder, latent_dim, n_jobs, n_machines, device, frozen):
    makespans = []
    
    for jsp in val_instances:
        env = build_single_eval_env(
            jsp=jsp,
            encoder=encoder,
            latent_dim=latent_dim,
            n_jobs=n_jobs,
            n_machines=n_machines,
            device=device,
            frozen=frozen,
        )

        obs, info = env.reset()
        done = truncated = False

        while not (done or truncated):
            action_masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, done, truncated, info = env.step(int(action))

        makespans.append(info["makespan"])

    return float(np.mean(makespans))