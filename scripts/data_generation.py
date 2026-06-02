"""
Generate JSP graph-state datasets for GAE/VGAE experiments.

For each random JSP instance, the script performs a random rollout in the
disjunctive graph environment and stores all visited graph states.
"""

import argparse
import os

import numpy as np
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

from tokenjsp.utils import clb, generate_jsp_instance


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_jobs", type=int, default=20)
    parser.add_argument("--n_machines", type=int, default=20)
    parser.add_argument("--num_instances", type=int, default=25)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--out_dir", type=str, default="datasets")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def random_rollout_collect_states(
    jsp: np.ndarray,
    n_jobs: int,
    n_machines: int,
    max_steps: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Collects graph states from one random rollout."""

    num_tasks = n_jobs * n_machines

    env = DisjunctiveGraphJspEnv(
        jps_instance=jsp,
        perform_left_shift_if_possible=True,
        normalize_observation_space=True,
        flat_observation_space=False,
        action_mode="task",
    )

    obs, _ = env.reset()

    done = False
    truncated = False
    steps = 0

    samples = []
    is_scheduled = np.zeros((num_tasks, 1), dtype=np.float32)

    while not (done or truncated) and steps < max_steps:
        adjacency = obs[:, :num_tasks].copy()

        # Environment features: machine one-hot + duration.
        base_features = obs[:, num_tasks:num_tasks + n_machines + 1].copy()

        clb_features = clb(adjacency, base_features)

        node_features = np.concatenate(
            [is_scheduled, clb_features],
            axis=1,
        ).astype(np.float32)

        machine_one_hot = base_features[:, :-1]

        samples.append(
            (
                adjacency.astype(np.float32),
                node_features,
                machine_one_hot.astype(np.float32),
            )
        )

        valid_actions = env.valid_action_list()

        if not valid_actions:
            break

        action = int(np.random.choice(valid_actions))
        is_scheduled[action, 0] = 1.0

        obs, _, done, truncated, _ = env.step(action)
        steps += 1

    return samples


def save_samples(
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_path: str,
    n_jobs: int,
    n_machines: int,
) -> None:
    """Stores collected graph states in compressed .npz format."""

    adjacency_arr = np.stack(
        [adjacency for adjacency, _, _ in samples],
        axis=0,
    ).astype(np.float32)

    feature_arr = np.stack(
        [features for _, features, _ in samples],
        axis=0,
    ).astype(np.float32)

    machine_arr = np.stack(
        [machine for _, _, machine in samples],
        axis=0,
    ).astype(np.float32)

    np.savez_compressed(
        out_path,
        A=adjacency_arr,
        X=feature_arr,
        M=machine_arr,
        n_jobs=np.array([n_jobs], dtype=np.int32),
        n_machines=np.array([n_machines], dtype=np.int32),
    )

    print(
        f"Saved: {out_path} | "
        f"A={adjacency_arr.shape}, "
        f"X={feature_arr.shape}, "
        f"M={machine_arr.shape}"
    )


if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    # Generate random JSP instances and collect rollout states.
    all_samples = []

    for i in range(args.num_instances):
        jsp = generate_jsp_instance(
            n_jobs=args.n_jobs,
            n_machines=args.n_machines,
        )

        samples = random_rollout_collect_states(
            jsp=jsp,
            n_jobs=args.n_jobs,
            n_machines=args.n_machines,
            max_steps=args.max_steps,
        )

        all_samples.extend(samples)

        print(
            f"[{i + 1}/{args.num_instances}] "
            f"collected {len(samples)} samples "
            f"(total={len(all_samples)})"
        )

    if not all_samples:
        raise RuntimeError("No graph states were collected.")

    # Save all collected states as one compressed dataset.
    out_path = os.path.join(
        args.out_dir,
        f"gae_dataset_jsp_{args.n_jobs}x{args.n_machines}.npz",
    )

    save_samples(
        samples=all_samples,
        out_path=out_path,
        n_jobs=args.n_jobs,
        n_machines=args.n_machines,
    )