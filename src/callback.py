import os
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from validation import validate_model
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from gae_wrapper import FrozenGAEObsWrapper, GAEFeatureExtractor, ScheduleFlagWrapper
import wandb

class LogValCallback(BaseCallback):
    def __init__(
        self,
        val_instances,
        encoder,
        latent_dim,
        n_jobs,
        n_machines,
        device,
        frozen,
        eval_freq,
        save_dir,
        verbose=1,
    ):
        super().__init__(verbose)
        self.val_instances = val_instances
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.device = device
        self.frozen = frozen
        self.eval_freq = eval_freq
        self.save_dir = save_dir
        self.best_val_makespan = float("inf")

        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        mean_val_makespan = validate_model(
            model=self.model,
            val_instances=self.val_instances,
            encoder=self.encoder,
            latent_dim=self.latent_dim,
            n_jobs=self.n_jobs,
            n_machines=self.n_machines,
            device=self.device,
            frozen=self.frozen,
        )

        if self.verbose:
            print(f"\n[Validation] step={self.num_timesteps}, mean makespan={mean_val_makespan:.2f}")

        ckpt_path = os.path.join(self.save_dir, f"model_step_{self.num_timesteps}.zip")
        self.model.save(ckpt_path)

        # save best checkpoint
        if mean_val_makespan < self.best_val_makespan:
            self.best_val_makespan = mean_val_makespan
            best_path = os.path.join(self.save_dir, "best_model.zip")
            self.model.save(best_path)
            if self.verbose:
                print(f"[Validation] New best model saved: {best_path}")
        wandb.log({"val/mean_makespan": mean_val_makespan, "val/timestep": self.num_timesteps}, step=self.num_timesteps)

        return True

    def _on_rollout_end(self):
        logger_dict = self.model.logger.name_to_value

        wandb.log({
            "train/loss": logger_dict.get("train/loss"),
            "train/value_loss": logger_dict.get("train/value_loss"),
            "train/policy_gradient_loss": logger_dict.get("train/policy_gradient_loss"),
            "train/entropy_loss": logger_dict.get("train/entropy_loss"),
            "train/approx_kl": logger_dict.get("train/approx_kl"),
            "train/clip_fraction": logger_dict.get("train/clip_fraction"),
            "train/explained_variance": logger_dict.get("train/explained_variance"),
        }, step=self.num_timesteps)