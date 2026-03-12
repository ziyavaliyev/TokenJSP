import torch
import numpy as np
import gymnasium as gym
import sb3_contrib
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from encoder import Encoder
from gae_wrapper import FrozenGAEObsWrapper, GAEFeatureExtractor, ScheduleFlagWrapper
from jsp_instance_utils.instances import abz6, abz6_makespan


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Mode
# ----------------------------
FROZEN = False   # True -> old frozen wrapper mode, False -> finetuned encoder mode

# ----------------------------
# Model / encoder parameters
# ----------------------------
HIDDEN = 64
LATENT = 32
IN_CHANNELS = 12

ENCODER_CKPT = "weights/encoder_10x10_baseline.pt"
MODEL_PATH = "runs/rl_1772794331/model.zip"

N_JOBS = 10
N_MACHINES = 10
T = N_JOBS * N_MACHINES

# ----------------------------
# Example JSP instance
# ----------------------------
jsp = abz6

# ----------------------------
# Build environment
# ----------------------------
env = DisjunctiveGraphJspEnv(
    jps_instance=jsp,
    perform_left_shift_if_possible=True,
    normalize_observation_space=True,
    flat_observation_space=False,
    action_mode="task",
)

env = Monitor(env)


def mask_fn(env: gym.Env):
    return env.unwrapped.valid_action_mask()


env = ActionMasker(env, mask_fn)

# ----------------------------
# Load encoder
# ----------------------------
encoder = Encoder(IN_CHANNELS, HIDDEN, LATENT).to(DEVICE)
encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location=DEVICE))

if FROZEN:
    encoder.eval()
    print("Loaded frozen encoder:", ENCODER_CKPT)

    env = FrozenGAEObsWrapper(
        env,
        encoder=encoder,
        latent_dim=LATENT,
        n_jobs=N_JOBS,
        n_machines=N_MACHINES,
        device=DEVICE,
    )

    model = sb3_contrib.MaskablePPO.load(
        MODEL_PATH,
        env=env,
        device=DEVICE,
    )

else:
    encoder.train()
    print("Loaded finetunable encoder:", ENCODER_CKPT)

    # In the finetuned setup, the env must stay in raw-graph form
    # and only append the scheduling flag.
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
        MODEL_PATH,
        env=env,
        device=DEVICE,
        custom_objects={
            "policy_kwargs": policy_kwargs,
        },
    )

print("Loaded RL model:", MODEL_PATH)

# ----------------------------
# Run inference
# ----------------------------
print("\n=== Running Inference ===")

obs, info = env.reset()
done = truncated = False
total_reward = 0

while not (done or truncated):
    action_masks = get_action_masks(env)

    action, _ = model.predict(
        obs,
        deterministic=True,
        action_masks=action_masks,
    )

    obs, reward, done, truncated, info = env.step(int(action))
    total_reward += reward

print("\nEpisode finished")
print("Makespan:", info["makespan"])
print(f"Optimal makespan is: {abz6_makespan}")