import torch
import numpy as np
import gymnasium as gym
import sb3_contrib
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from encoder import Encoder
from gae_wrapper import FrozenGAEObsWrapper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN = 64
LATENT = 32
ENCODER_CKPT = "checkpoints/encoder.pt"

N_JOBS = 6
N_MACHINES = 6
T = N_JOBS * N_MACHINES

IN_CHANNELS = 8
jsp=np.array([[
        [2, 0, 1, 3, 5, 4],
        [1, 2, 4, 5, 0, 3],
        [2, 3, 5, 0, 1, 4],
        [1, 0, 2, 3, 4, 5],
        [2, 1, 4, 5, 0, 3],
        [1, 3, 5, 0, 4, 2],
    ], [
        [1, 3, 6, 7, 3, 6],
        [8, 5, 10, 10, 10, 4],
        [5, 4, 8, 9, 1, 7],
        [5, 5, 5, 3, 8, 9],
        [9, 3, 5, 4, 3, 1],
        [3, 3, 9, 10, 4, 1],
    ]])

env = DisjunctiveGraphJspEnv(
    jps_instance=jsp,
    perform_left_shift_if_possible=True,
    normalize_observation_space=True,
    flat_observation_space=False,
    action_mode="task",
)
env = Monitor(env)

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()

env = ActionMasker(env, mask_fn)

encoder = Encoder(IN_CHANNELS, HIDDEN, LATENT).to(DEVICE)
encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location=DEVICE))
encoder.eval()
print("Loaded encoder weights:", ENCODER_CKPT)

env = FrozenGAEObsWrapper(
    env,
    encoder=encoder,
    latent_dim=LATENT,
    n_jobs=N_JOBS,
    n_machines=N_MACHINES,
    device=DEVICE,
)
#MaskableActorCriticPolicy
model = sb3_contrib.MaskablePPO.load(
    "gae_rl_jsp.zip",
    env,
    device=DEVICE,
)

#model.learn(total_timesteps=100_000)
#model.save("gae_rl_jsp")
#print("Model saved.")

print("\n=== Running Inference ===")

obs, info = env.reset()
done = truncated = False
total_reward = 0

while not (done or truncated):
    action_masks = get_action_masks(env)
    action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
    obs, reward, done, truncated, info = env.step(int(action))
    #env.render()
    total_reward += reward

print("Episode finished.")
print("Makespan:", env.unwrapped.get_makespan())