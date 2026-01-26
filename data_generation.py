from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from utils import generate_jsp_instance
import numpy as np

n_jobs = 6
n_machines = 6
T = n_jobs * n_machines  # number of tasks/nodes (without dummies)

def random_rollout_collect_states(jsp, max_steps=10_000) -> list[tuple[np.ndarray, np.ndarray]]:
    env = DisjunctiveGraphJspEnv(
        jps_instance=jsp,
        perform_left_shift_if_possible=True,
        normalize_observation_space=True,
        flat_observation_space=False,
        action_mode="task",
    )

    samples = []

    obs, info = env.reset()
    done = truncated = False
    steps = 0

    while not (done or truncated) and steps < max_steps:
        # obs shape: (T, T + M + 1)
        A = obs[:, :T].copy()                 # (T, T)
        X = obs[:, T:T + n_machines + 1].copy()  # (T, M+1)
        samples.append((A, X))

        valid_actions = env.valid_action_list()
        if not valid_actions:
            break

        action = np.random.choice(valid_actions)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1

    return samples


all_samples = []

for _ in range(1_000):
    jsp = generate_jsp_instance(n_jobs=n_jobs, n_machines=n_machines)
    samples = random_rollout_collect_states(jsp)
    all_samples.extend(samples)
    print(f"collected {len(samples)} samples (total={len(all_samples)})")

# Example access:
A0, X0 = all_samples[0]
print("A0:", A0.shape, "X0:", X0.shape)

A_arr = np.stack([A for (A, X) in all_samples], axis=0).astype(np.float32)  # (N, T, T)
X_arr = np.stack([X for (A, X) in all_samples], axis=0).astype(np.float32)  # (N, T, M+1)

np.savez_compressed(
    "gae_dataset_jsp_6x6.npz",
    A=A_arr,
    X=X_arr,
    n_jobs=np.array([n_jobs], dtype=np.int32),
    n_machines=np.array([n_machines], dtype=np.int32),
)

print("Saved:", A_arr.shape, X_arr.shape)