import numpy as np
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from jsp_instance_utils.instances import ft10

DATASET_PATH = "L2D_10x10.npy"

def get_task_info(env, action: int):
    """
    action is 0-based, task_id in env is action + 1
    """
    task_id = action + 1
    node = env.G.nodes[task_id]
    job = int(node["job"])
    duration = int(node["duration"])
    return task_id, job, duration


def heuristic_action(env, rule: str):
    valid_actions = env.valid_action_list()

    if len(valid_actions) == 1:
        return valid_actions[0]

    candidates = []
    for action in valid_actions:
        task_id, job, duration = get_task_info(env, action)

        # remaining operations in this job including current one
        remaining_ops = 0
        start_task = job * env.n_machines + 1
        end_task = start_task + env.n_machines
        for tid in range(start_task, end_task):
            if not env.G.nodes[tid]["scheduled"]:
                remaining_ops += 1

        candidates.append({
            "action": action,
            "duration": duration,
            "remaining_ops": remaining_ops,
        })

    if rule == "SPT":
        return min(candidates, key=lambda x: x["duration"])["action"]

    elif rule == "LPT":
        return max(candidates, key=lambda x: x["duration"])["action"]

    elif rule == "MOR":
        return max(candidates, key=lambda x: x["remaining_ops"])["action"]

    elif rule == "RANDOM":
        return np.random.choice(valid_actions)

    else:
        raise ValueError(f"Unknown rule: {rule}")


def run_heuristic_on_instance(jsp, rule: str):
    env = DisjunctiveGraphJspEnv(
        jps_instance=jsp,
        perform_left_shift_if_possible=True,
        normalize_observation_space=True,
        flat_observation_space=False,
        action_mode="task",
    )

    obs, info = env.reset()
    done = truncated = False

    while not (done or truncated):
        action = heuristic_action(env, rule)
        obs, reward, done, truncated, info = env.step(action)

    return info["makespan"]


def evaluate_dataset(dataset, rule: str):
    makespans = []

    for i, jsp in enumerate(dataset, start=1):
        ms = run_heuristic_on_instance(jsp, rule)
        makespans.append(ms)

        if i % 10 == 0 or i == len(dataset):
            print(f"[{rule}] {i}/{len(dataset)} done, current mean makespan: {np.mean(makespans):.2f}")

    makespans = np.array(makespans, dtype=np.float64)
    return {
        "rule": rule,
        "mean_makespan": float(np.mean(makespans)),
        "std_makespan": float(np.std(makespans)),
        "min_makespan": float(np.min(makespans)),
        "max_makespan": float(np.max(makespans)),
    }


def main():
    data = np.load(DATASET_PATH, allow_pickle=False)
    data = data[:, [1, 0], :, :]
    data[:, 0] -= 1
    data = x = np.expand_dims(ft10, axis=0)
    print(f"Loaded dataset: {DATASET_PATH}")
    print(f"Dataset shape: {data.shape}")
    print(f"Number of instances: {len(data)}")

    rules = ["SPT", "LPT", "MOR", "RANDOM"]
    results = []
    np.savez(f"L2D_10x10.npz", data=data)
    for rule in rules:
        print(f"\n=== Evaluating {rule} ===")
        result = evaluate_dataset(data, rule)
        results.append(result)

    print("\n=== Final Results ===")
    for r in results:
        print(
            f"{r['rule']:>6} | "
            f"mean={r['mean_makespan']:.2f} | "
            f"std={r['std_makespan']:.2f} | "
            f"min={r['min_makespan']:.2f} | "
            f"max={r['max_makespan']:.2f}"
        )
    np.save("L2D_10x10.npz", data)


if __name__ == "__main__":
    main()