import json
import time
import numpy as np
from ortools.sat.python import cp_model


DATASET_PATH = "datasets/rl_dataset_test_jsp_10x10.npz"
TIME_LIMIT_PER_INSTANCE_SEC = 300.0   # increase if you want stronger optimality guarantees
NUM_WORKERS = 8


def solve_jsp_ortools(jsp_instance: np.ndarray,
                      time_limit_sec: float = 300.0,
                      num_workers: int = 8) -> dict:
    """
    Solve one JSP instance with OR-Tools CP-SAT.

    Expected jsp_instance shape: (2, n_jobs, n_machines)
      jsp_instance[0] = machine order
      jsp_instance[1] = processing times
    """
    machine_order = np.asarray(jsp_instance[0], dtype=int)
    processing_times = np.asarray(jsp_instance[1], dtype=int)

    n_jobs, n_machines = machine_order.shape
    assert processing_times.shape == (n_jobs, n_machines)

    model = cp_model.CpModel()

    # Standard horizon upper bound = sum of all durations.
    horizon = int(processing_times.sum())

    # Variables
    starts = {}
    ends = {}
    intervals = {}
    machine_to_intervals = {m: [] for m in range(n_machines)}

    for j in range(n_jobs):
        for t in range(n_machines):
            m = int(machine_order[j, t])
            d = int(processing_times[j, t])

            start = model.new_int_var(0, horizon, f"start_{j}_{t}")
            end = model.new_int_var(0, horizon, f"end_{j}_{t}")
            interval = model.new_interval_var(start, d, end, f"interval_{j}_{t}")

            starts[(j, t)] = start
            ends[(j, t)] = end
            intervals[(j, t)] = interval
            machine_to_intervals[m].append(interval)

    # Job precedence constraints
    for j in range(n_jobs):
        for t in range(n_machines - 1):
            model.add(starts[(j, t + 1)] >= ends[(j, t)])

    # Machine capacity constraints
    for m in range(n_machines):
        model.add_no_overlap(machine_to_intervals[m])

    # Makespan objective
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, [ends[(j, n_machines - 1)] for j in range(n_jobs)])
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = int(num_workers)

    t0 = time.time()
    status = solver.solve(model)
    elapsed = time.time() - t0

    status_name = solver.status_name(status)

    result = {
        "status": status_name,
        "solve_time_sec": elapsed,
        "objective": None,
        "schedule": [],
    }

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result["objective"] = int(solver.value(makespan))

        # Optional: extract operation schedule
        schedule = []
        for j in range(n_jobs):
            for t in range(n_machines):
                schedule.append({
                    "job": int(j),
                    "op": int(t),
                    "machine": int(machine_order[j, t]),
                    "duration": int(processing_times[j, t]),
                    "start": int(solver.value(starts[(j, t)])),
                    "end": int(solver.value(ends[(j, t)])),
                })
        result["schedule"] = schedule

    return result


def main():
    data = np.load(DATASET_PATH, allow_pickle=False)["data"]
    print(f"Loaded dataset: {DATASET_PATH}")
    print(f"Number of instances: {len(data)}")

    results = []
    makespans_optimal = []
    makespans_feasible = []

    n_optimal = 0
    n_feasible = 0
    n_unknown = 0
    n_infeasible = 0

    for idx, jsp in enumerate(data, start=1):
        res = solve_jsp_ortools(
            jsp,
            time_limit_sec=TIME_LIMIT_PER_INSTANCE_SEC,
            num_workers=NUM_WORKERS,
        )
        results.append(res)

        if res["status"] == "OPTIMAL":
            n_optimal += 1
            makespans_optimal.append(res["objective"])
        elif res["status"] == "FEASIBLE":
            n_feasible += 1
            makespans_feasible.append(res["objective"])
        elif res["status"] == "UNKNOWN":
            n_unknown += 1
        elif res["status"] == "INFEASIBLE":
            n_infeasible += 1

        print(
            f"[{idx}/{len(data)}] "
            f"status={res['status']:<10} "
            f"makespan={res['objective']} "
            f"time={res['solve_time_sec']:.2f}s"
        )

    all_found = [x for x in (makespans_optimal + makespans_feasible) if x is not None]

    summary = {
        "dataset": DATASET_PATH,
        "n_instances": len(data),
        "time_limit_per_instance_sec": TIME_LIMIT_PER_INSTANCE_SEC,
        "num_workers": NUM_WORKERS,
        "n_optimal": n_optimal,
        "n_feasible_only": n_feasible,
        "n_unknown": n_unknown,
        "n_infeasible": n_infeasible,
        "mean_makespan_all_found": float(np.mean(all_found)) if all_found else None,
        "mean_makespan_optimal_only": float(np.mean(makespans_optimal)) if makespans_optimal else None,
    }

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    with open("ortools_jsp_results.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\nSaved detailed results to ortools_jsp_results.json")


if __name__ == "__main__":
    main()