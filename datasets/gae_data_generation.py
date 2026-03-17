from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from utils import generate_jsp_instance
from graph_features import clb
import numpy as np

N_JOBS = 10
N_MACHINES = 10
NUM_INSTANCES = 10

T = N_JOBS * N_MACHINES

def random_rollout_collect_states(jsp: np.ndarray, max_steps: int = 10_000) -> list[tuple[np.ndarray, np.ndarray]]:
    env = DisjunctiveGraphJspEnv(
        jps_instance=jsp,
        perform_left_shift_if_possible=True,
        normalize_observation_space=True,
        flat_observation_space=False,
        action_mode="task",
    )

    obs, info = env.reset()
    done = truncated = False
    steps = 0
    samples = []

    is_scheduled = np.zeros((T, 1), dtype=np.float32)

    while not (done or truncated) and steps < max_steps:
        A = obs[:, :T].copy()

        # base features from the environment (one-hot machine + duration): (T, M+1)
        base = obs[:, T:T + N_MACHINES + 1].copy()

        #CLB
        clb_data = clb(A, base)

        X = np.concatenate([is_scheduled, clb_data], axis=1).astype(np.float32)  # (T, F)

        samples.append((A.astype(np.float32), X, base[:,:-1]))

        valid_actions = env.valid_action_list()
        if not valid_actions:
            break

        action = int(np.random.choice(valid_actions))
        is_scheduled[action, 0] = 1.0

        obs, reward, done, truncated, info = env.step(action)
        steps += 1

    return samples


def main():
    all_samples: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(NUM_INSTANCES):
        jsp = generate_jsp_instance(n_jobs=N_JOBS, n_machines=N_MACHINES)
        samples = random_rollout_collect_states(jsp, max_steps=10_000)
        all_samples.extend(samples)
        print(f"[{i+1}/{NUM_INSTANCES}] collected {len(samples)} samples (total={len(all_samples)})")

    A0, X0, base0 = all_samples[0]
    print("A0:", A0.shape, "X0:", X0.shape, "base0:", base0.shape)

    A_arr = np.stack([A for (A, X, base) in all_samples], axis=0).astype(np.float32)  # (N, T, T)
    X_arr = np.stack([X for (A, X, base) in all_samples], axis=0).astype(np.float32)  # (N, T, F)
    base_arr = np.stack([base for (A, X, base) in all_samples], axis=0).astype(np.float32)  # (N, T, M)

    out_path = f"datasets/gae_dataset_jsp_{N_JOBS}x{N_MACHINES}.npz"
    
    np.savez_compressed(
        out_path,
        A=A_arr,
        X=X_arr,
        M=base_arr,
        n_jobs=np.array([N_JOBS], dtype=np.int32),
        n_machines=np.array([N_MACHINES], dtype=np.int32),
    )

    print("Saved:", out_path, "A:", A_arr.shape, "X:", X_arr.shape, "Machine correspondence:")


if __name__ == "__main__":
    main()