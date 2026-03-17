import argparse
import wandb
import time
import os
import json
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import sb3_contrib
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from encoder import Encoder
from gae_wrapper import FrozenGAEObsWrapper, GAEFeatureExtractor, ScheduleFlagWrapper
from instance_wrapper import InstanceSampler
from jsp_instance_utils.instances import abz5, ft06, ft10
from callback import LogValCallback

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()

def build_env(instances, encoder, latent_dim, n_jobs, n_machines, device, frozen):
    env = DisjunctiveGraphJspEnv(
        jps_instance=instances[0],
        perform_left_shift_if_possible=True,
        normalize_observation_space=True,
        flat_observation_space=False,
        action_mode="task",
    )
    env = InstanceSampler(env, instances)
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
        env=ScheduleFlagWrapper(env, n_jobs=n_jobs, n_machines=n_machines)
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--encoder_weights", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    default_config = {
        "total_steps": 2_000_000,
        "n_steps": 2000,
        "batch_size": 64,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "hidden": 64,
        "latent": 32,
        "in_channels": 12,
        "n_jobs": 10,
        "n_machines": 10,
        "eval_freq": 50000,
        "frozen_encoder": True
    }

    if args.use_wandb:
        wandb.init(project="jsp-rl", name=args.run_name, config=default_config)
        config = wandb.config
    else:
        config = default_config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_instances = np.load(args.train_data, allow_pickle=False)["data"]

    encoder = Encoder(config["in_channels"], config["hidden"], config["latent"], gnn_type="egc").to(device)
    encoder.load_state_dict(torch.load(args.encoder_weights, map_location=device))
    if config["frozen_encoder"]:
        encoder.eval()
    else:
        encoder.train()

    env = build_env(
        instances=train_instances,
        encoder=encoder,
        latent_dim=config["latent"],
        n_jobs=config["n_jobs"],
        n_machines=config["n_machines"],
        device=device,
        frozen=config["frozen_encoder"]
    )

    if config["frozen_encoder"]:
        model = sb3_contrib.MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            device=device,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            gamma=config["gamma"],
            ent_coef=config["ent_coef"],
            verbose=1,
        )
    else:
        policy_kwargs = dict(
            features_extractor_class=GAEFeatureExtractor,
            features_extractor_kwargs=dict(
                encoder=encoder,
                n_jobs=config["n_jobs"],
                n_machines=config["n_machines"],
                features_dim=config["latent"],
            ),
        )
        model = sb3_contrib.MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        device=device,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        ent_coef=config["ent_coef"],
        verbose=1,
        policy_kwargs=policy_kwargs,
        )

    out_dir = os.path.join("runs", args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(dict(config), f, indent=2)

    if args.use_wandb:
        val_instances = np.load(args.val_data, allow_pickle=False)["data"]
        callback = LogValCallback(
                val_instances=val_instances,
                encoder=encoder,
                latent_dim=config["latent"],
                n_jobs=config["n_jobs"],
                n_machines=config["n_machines"],
                device=device,
                frozen=config["frozen_encoder"],
                eval_freq=config["eval_freq"],
                save_dir=os.path.join(out_dir, "checkpoints"),
                verbose=1,
            )
    else:
        callback = None

    model.learn(
        total_timesteps=config["total_steps"],
        progress_bar=True,
        callback=callback,
    )

    save_path = os.path.join(out_dir, "model.zip")
    model.save(save_path)

    if args.use_wandb:
        wandb.log({"final/model_path_saved": 1})
        wandb.finish()

if __name__ == "__main__":
    main()