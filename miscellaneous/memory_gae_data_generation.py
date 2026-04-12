from pathlib import Path
import numpy as np

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from utils import generate_jsp_instance
from graph_features import clb

N_JOBS = 10
N_MACHINES = 10
NUM_INSTANCES = 1000
MAX_STEPS = 10_000

T = N_JOBS * N_MACHINES
F = 2  # is_scheduled + clb
M = N_MACHINES  # machine one-hot only

OUT_PATH = Path(f"datasets/gae_dataset_jsp_{N_JOBS}x{N_MACHINES}_250.npz")
TMP_DIR = Path("datasets/tmp_memmap")
TMP_DIR.mkdir(parents=True, exist_ok=True)


def rollout_generator(jsp: np.ndarray, max_steps: int = 10_000):
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

    is_scheduled = np.zeros((T, 1), dtype=np.float32)

    while not (done or truncated) and steps < max_steps:
        A = obs[:, :T].astype(np.float32, copy=True)
        base = obs[:, T:T + N_MACHINES + 1].astype(np.float32, copy=True)

        clb_data = clb(A, base)  # expected shape (T, 1)
        X = np.concatenate([is_scheduled, clb_data], axis=1).astype(np.float32)

        # machine correspondence only (drop duration)
        machine_onehot = base[:, :-1].astype(np.float32)

        yield A, X, machine_onehot

        valid_actions = env.valid_action_list()
        if not valid_actions:
            break

        action = int(np.random.choice(valid_actions))
        is_scheduled[action, 0] = 1.0

        obs, reward, done, truncated, info = env.step(action)
        steps += 1


def count_total_samples() -> int:
    total = 0
    for i in range(NUM_INSTANCES):
        jsp = generate_jsp_instance(n_jobs=N_JOBS, n_machines=N_MACHINES)
        n = sum(1 for _ in rollout_generator(jsp, max_steps=MAX_STEPS))
        total += n
        print(f"[count {i+1}/{NUM_INSTANCES}] samples={n}, total={total}")
    return total


def main():
    print("Pass 1/2: counting total number of samples...")
    total_samples = count_total_samples()
    print(f"Total samples: {total_samples}")

    A_path = TMP_DIR / "A.dat"
    X_path = TMP_DIR / "X.dat"
    M_path = TMP_DIR / "M.dat"

    A_mm = np.memmap(A_path, dtype=np.float32, mode="w+", shape=(total_samples, T, T))
    X_mm = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(total_samples, T, F))
    M_mm = np.memmap(M_path, dtype=np.float32, mode="w+", shape=(total_samples, T, M))

    print("Pass 2/2: generating and writing samples to disk-backed arrays...")
    idx = 0
    for i in range(NUM_INSTANCES):
        jsp = generate_jsp_instance(n_jobs=N_JOBS, n_machines=N_MACHINES)

        local_count = 0
        for A, X, machine_onehot in rollout_generator(jsp, max_steps=MAX_STEPS):
            A_mm[idx] = A
            X_mm[idx] = X
            M_mm[idx] = machine_onehot
            idx += 1
            local_count += 1

        print(f"[write {i+1}/{NUM_INSTANCES}] wrote {local_count} samples, total={idx}")

        # flush regularly
        A_mm.flush()
        X_mm.flush()
        M_mm.flush()

    print("Saving final .npz...")
    np.savez_compressed(
        OUT_PATH,
        A=np.asarray(A_mm),
        X=np.asarray(X_mm),
        M=np.asarray(M_mm),
        n_jobs=np.array([N_JOBS], dtype=np.int32),
        n_machines=np.array([N_MACHINES], dtype=np.int32),
    )

    print(f"Saved: {OUT_PATH}")
    print("A:", A_mm.shape, "X:", X_mm.shape, "M:", M_mm.shape)

    # optional cleanup of temp files
    del A_mm, X_mm, M_mm
    # A_path.unlink(missing_ok=True)
    # X_path.unlink(missing_ok=True)
    # M_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()