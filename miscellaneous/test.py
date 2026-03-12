import os
import json
import torch
import numpy as np
import gymnasium as gym
import sb3_contrib

from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from encoder import Encoder
from gae_wrapper import FrozenGAEObsWrapper, GAEFeatureExtractor, ScheduleFlagWrapper


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Paths
# ----------------------------
DATASET_PATH = "miscellaneous/L2D_10x10.npz"#"datasets/rl_dataset_test_jsp_10x10.npz"
ENCODER_CKPT = "weights/encoder_10x10_baseline.pt"

#FROZEN_MODEL_PATH = "runs/rl_1772757162/model.zip"
FROZEN2_MODEL_PATH = "runs/test_full_latent_representation/checkpoints/model_step_1150000.zip"
#NONFROZEN_MODEL_PATH = "runs/rl_1772794331/model.zip"

# ----------------------------
# Model / encoder parameters
# ----------------------------
HIDDEN = 64
LATENT = 32
IN_CHANNELS = 12

N_JOBS = 10
N_MACHINES = 10


def mask_fn(env: gym.Env):
    return env.unwrapped.valid_action_mask()


def build_base_env(jsp_instance: np.ndarray):
    env = DisjunctiveGraphJspEnv(
        jps_instance=jsp_instance,
        perform_left_shift_if_possible=True,
        normalize_observation_space=True,
        flat_observation_space=False,
        action_mode="task",
    )
    env = Monitor(env)
    env = ActionMasker(env, mask_fn)
    return env


def load_encoder():
    encoder = Encoder(IN_CHANNELS, HIDDEN, LATENT).to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location=DEVICE))
    return encoder


def load_model(model_path: str, frozen: bool, jsp_instance: np.ndarray):
    encoder = load_encoder()
    env = build_base_env(jsp_instance)

    if frozen:
        encoder.eval()
        env = FrozenGAEObsWrapper(
            env,
            encoder=encoder,
            latent_dim=LATENT,
            n_jobs=N_JOBS,
            n_machines=N_MACHINES,
            device=DEVICE,
        )
        model = sb3_contrib.MaskablePPO.load(
            model_path,
            env=env,
            device=DEVICE,
        )
    else:
        encoder.train()
        env = ScheduleFlagWrapper(
            env,
            n_jobs=N_JOBS,
            n_machines=N_MACHINES,
        )

        policy_kwargs = dict(
            features_extractor_class=GAEFeatureExtractor,
            features_extractor_kwargs=dict(
                encoder=encoder,
                n_jobs=N_JOBS,
                n_machines=N_MACHINES,
                features_dim=LATENT,
            ),
        )

        model = sb3_contrib.MaskablePPO.load(
            model_path,
            env=env,
            device=DEVICE,
            custom_objects={"policy_kwargs": policy_kwargs},
        )

    return model, env


def run_single_instance(model, env):
    obs, info = env.reset()
    done = truncated = False
    total_reward = 0.0

    while not (done or truncated):
        action_masks = get_action_masks(env)
        action, _ = model.predict(
            obs,
            deterministic=True,
            action_masks=action_masks,
        )
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += float(reward)

    return {
        "makespan": float(info["makespan"]),
        "reward": total_reward,
    }


def evaluate_model(model_path: str, frozen: bool, dataset: np.ndarray, name: str):
    print(f"\n=== Evaluating {name} ===")
    print(f"Model: {model_path}")
    print(f"Frozen: {frozen}")

    makespans = []
    rewards = []

    # build once with first instance, then reload instance each loop through unwrapped base env
    model, env = load_model(model_path, frozen, dataset[0])

    for idx, jsp in enumerate(dataset):
        # reload underlying JSP instance
        env.unwrapped.load_instance(jsp)

        result = run_single_instance(model, env)
        makespans.append(result["makespan"])
        rewards.append(result["reward"])

        if (idx + 1) % 10 == 0 or (idx + 1) == len(dataset):
            print(f"[{idx + 1}/{len(dataset)}] current mean makespan: {np.mean(makespans):.2f}")

    makespans = np.array(makespans, dtype=np.float64)
    rewards = np.array(rewards, dtype=np.float64)

    summary = {
        "name": name,
        "frozen": frozen,
        "n_instances": int(len(dataset)),
        "mean_makespan": float(np.mean(makespans)),
        "std_makespan": float(np.std(makespans)),
        "min_makespan": float(np.min(makespans)),
        "max_makespan": float(np.max(makespans)),
        "mean_reward": float(np.mean(rewards)),
    }
    return summary


def main():
    data = np.load("miscellaneous/L2D_10x10.npy")[:,[1,0],:,:]
    data[:,0] -= 1

    print(f"Loaded dataset: {DATASET_PATH}")
    print(f"Number of test instances: {len(data)}")

    """frozen_summary = evaluate_model(
        model_path=FROZEN_MODEL_PATH,
        frozen=True,
        dataset=data,
        name="Frozen encoder",
    )"""

    frozen2_summary = evaluate_model(
        model_path=FROZEN2_MODEL_PATH,
        frozen=True,
        dataset=data,
        name="Frozen encoder 2",
    )

    """nonfrozen_summary = evaluate_model(
        model_path=NONFROZEN_MODEL_PATH,
        frozen=False,
        dataset=data,
        name="Non-frozen encoder",
    )"""

    print("\n=== Comparison ===")
    print(json.dumps({
        #"frozen": frozen_summary,
        "frozen2": frozen2_summary,
        #"non_frozen": nonfrozen_summary,
    }, indent=2))


if __name__ == "__main__":
    main()