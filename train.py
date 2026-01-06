import gymnasium as gym
import sb3_contrib
import numpy as np
from stable_baselines3.common.monitor import Monitor
from encoder import Encoder

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

jsp = np.array([
    [[1, 2, 0], [0, 2, 1]],
    [[17, 12, 19], [8, 6, 2]]
])

env = DisjunctiveGraphJspEnv(
    jps_instance=jsp,
    perform_left_shift_if_possible=True,
    normalize_observation_space=True,
    flat_observation_space=True,
    action_mode="task",
)
env = Monitor(env)

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()

env = ActionMasker(env, mask_fn)

policy_kwargs = dict(
    features_extractor_class=Encoder,
    features_extractor_kwargs=dict(latent_dim=128),
)

model = sb3_contrib.MaskablePPO(
    MaskableActorCriticPolicy,
    env,
    verbose=1,
    policy_kwargs=policy_kwargs,
)

model.learn(total_timesteps=10_000)