import argparse
import csv
import os
import random
import sys
import tracemalloc
from pathlib import Path
from time import perf_counter

from agents import AgentC, AgentE
from baseline import ReactTTC_variant, TTC
from eval import total_rank_improvement
from weakTTC import run_weak_ttc


# What it does:
# Reads data/combined_data_jd200_1.csv.
# Ignores type=d.
# Treats type=c as EVs.
# Treats type=f as charging points.
# Selects subsets of real EVs and charging points for four real-data experiments:
# varying total EV with fixed EV-to-CP ratio, varying total charging points with
# fixed EV count, varying charging-point capacity, and varying profile size
# (charging points per preference class).
# Builds EV preferences from real EV-to-charging-point distances in
# data/distance_matrix_jd200_1.csv.
# Uses compact internal IDs for the algorithms.
# Default initial assignment is random feasible, because nearest-feasible can
# make real-distance instances already stable with zero cycles. You can still
# run nearest with --initial-assignment nearest.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


EXPERIMENT_LABELS = {
    "varying_ev": "Total EV",
    "varying_cp": "Total Charging Points",
    "varying_capacity": "Capacity",
    "varying_profile_size": "Profile Size",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run WeakTTC experiments on real EV and charging-point data."
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=REPO_ROOT / "data" / "combined_data_jd200_1.csv",
        help="CSV containing EV and charging-point nodes.",
    )
    parser.add_argument(
        "--distance-matrix",
        type=Path,
        default=REPO_ROOT / "data" / "distance_matrix_jd200_1.csv",
        help="CSV distance matrix keyed by original node IDs.",
    )
    parser.add_argument(
        "--experiment",
        choices=("all", "varying_ev", "varying_cp", "varying_capacity", "varying_profile_size"),
        default="all",
        help="Which real-data experiment to run.",
    )
    parser.add_argument("--iterations", type=int, default=10, help="Iterations per setting.")
    parser.add_argument("--base-seed", type=int, default=123, help="Base random seed.")
    parser.add_argument(
        "--initial-assignment",
        choices=("random", "nearest"),
        default="random",
        help="Initial EV-to-charging-point assignment scheme.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "results_real_ev_charging_experiments.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=REPO_ROOT / "results" / "real_ev_charging_experiment_plots",
        help="Directory where plots will be saved.",
    )
    return parser.parse_args()


def read_real_nodes(nodes_path):
    """
    @brief Read EV and charging-point rows, ignoring depot rows.
    @param nodes_path CSV path with ID,type,lng,lat,... columns.
    @return Tuple of EV rows and charging-point rows.
    """
    ev_rows = []
    cp_rows = []
    with open(nodes_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            node_type = row["type"].strip()
            if node_type == "c":
                ev_rows.append(row)
            elif node_type == "f":
                cp_rows.append(row)

    ev_rows.sort(key=lambda row: int(row["ID"]))
    cp_rows.sort(key=lambda row: int(row["ID"]))
    return ev_rows, cp_rows


def read_distance_matrix(distance_matrix_path):
    """
    @brief Read a square distance matrix keyed by original node IDs.
    @param distance_matrix_path CSV distance matrix path.
    @return Mapping from original source ID to original target ID distance.
    """
    distances = {}
    with open(distance_matrix_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        target_ids = [int(value) for value in header[1:]]

        for row in reader:
            if not row:
                continue
            source_id = int(row[0])
            distances[source_id] = {
                target_id: float(value)
                for target_id, value in zip(target_ids, row[1:])
            }

    return distances


def selected_experiments(experiment_name):
    if experiment_name == "all":
        return ["varying_ev", "varying_cp", "varying_capacity", "varying_profile_size"]
    return [experiment_name]


def experiment_settings(experiment_name):
    """
    @brief Return parameter settings for one real-data experiment.
    @param experiment_name Experiment key.
    @return List of setting dictionaries.
    """
    if experiment_name == "varying_ev":
        return [
            {
                "n_e": n_e,
                "n_c": int(n_e * 0.5),
                "capacity": 2,
                "profile_size": 2,
                "x_value": n_e,
            }
            for n_e in range(20, 201, 20)
        ]

    if experiment_name == "varying_cp":
        return [
            {
                "n_e": 100,
                "n_c": n_c,
                "capacity": 2,
                "profile_size": 2,
                "x_value": n_c,
            }
            for n_c in range(50, 101, 10)
        ]

    if experiment_name == "varying_capacity":
        return [
            {
                "n_e": 100,
                "n_c": 100,
                "capacity": capacity,
                "profile_size": 2,
                "x_value": capacity,
            }
            for capacity in [1, 2, 5]
        ]

    if experiment_name == "varying_profile_size":
        return [
            {
                "n_e": 100,
                "n_c": 100,
                "capacity": 2,
                "profile_size": profile_size,
                "x_value": profile_size,
            }
            for profile_size in [2, 5, 10]
        ]

    raise ValueError(f"Unknown experiment: {experiment_name}")


def select_rows(rows, count, rng):
    """
    @brief Select a deterministic random subset and return it sorted by original ID.
    @param rows Available real-data rows.
    @param count Number of rows to select.
    @param rng Random number generator.
    @return Selected rows sorted by original node ID.
    """
    if count > len(rows):
        raise ValueError(f"Cannot select {count} rows from only {len(rows)} rows.")
    if count == len(rows):
        return list(rows)
    return sorted(rng.sample(rows, count), key=lambda row: int(row["ID"]))


def build_real_instance(
    ev_rows,
    cp_rows,
    distances,
    capacity,
    profile_size,
    initial_assignment_scheme,
    seed,
):
    """
    @brief Build AgentE/AgentC objects from selected real EV and charging-point data.
    @param ev_rows Selected real EV rows.
    @param cp_rows Selected real charging-point rows.
    @param distances Original-ID distance matrix.
    @param capacity Capacity assigned to each charging point.
    @param profile_size Number of charging points tied in each preference class.
    @param initial_assignment_scheme Either "random" or "nearest".
    @param seed Seed for random initial assignment.
    @return Tuple of agents, initial endowment, and preferences.
    """
    if capacity < 1:
        raise ValueError("capacity must be at least 1.")
    if profile_size < 1:
        raise ValueError("profile_size must be at least 1.")
    if len(cp_rows) * capacity < len(ev_rows):
        raise ValueError("Total charging-point capacity is smaller than the number of EVs.")

    ev_id_map = {int(row["ID"]): index for index, row in enumerate(ev_rows)}
    cp_id_map = {int(row["ID"]): index for index, row in enumerate(cp_rows)}

    agents_e = [AgentE(agent_id=index) for index in range(len(ev_rows))]
    agents_c = [AgentC(agent_id=index, capacity=capacity) for index in range(len(cp_rows))]

    cp_original_ids = [int(row["ID"]) for row in cp_rows]
    preferences = {}
    for row in ev_rows:
        original_ev_id = int(row["ID"])
        internal_ev_id = ev_id_map[original_ev_id]
        sorted_cps = sorted(
            cp_original_ids,
            key=lambda cp_id: (distances[original_ev_id][cp_id], cp_id),
        )

        pref = {}
        rank = 1
        for start_index in range(0, len(sorted_cps), profile_size):
            tied_original_cp_ids = sorted_cps[start_index : start_index + profile_size]
            pref[rank] = {cp_id_map[cp_id] for cp_id in tied_original_cp_ids}
            rank = rank + 1

        agents_e[internal_ev_id].pref = pref
        preferences[internal_ev_id] = pref

    if initial_assignment_scheme == "nearest":
        initial_endowment = assign_nearest_available_cp(
            ev_rows,
            cp_rows,
            distances,
            capacity,
            ev_id_map,
            cp_id_map,
            agents_e,
            agents_c,
        )
    elif initial_assignment_scheme == "random":
        initial_endowment = assign_random_available_cp(
            ev_rows,
            cp_rows,
            capacity,
            ev_id_map,
            cp_id_map,
            agents_e,
            agents_c,
            seed,
        )
    else:
        raise ValueError('initial_assignment_scheme must be either "random" or "nearest".')

    return agents_e, agents_c, initial_endowment, preferences


def assign_nearest_available_cp(
    ev_rows,
    cp_rows,
    distances,
    capacity,
    ev_id_map,
    cp_id_map,
    agents_e,
    agents_c,
):
    """
    @brief Assign each EV to the nearest charging point with remaining capacity.
    @return Initial AgentE-to-AgentC assignment map using internal IDs.
    """
    cp_original_ids = [int(row["ID"]) for row in cp_rows]
    remaining_capacity = {cp_id_map[int(row["ID"])]: capacity for row in cp_rows}
    initial_endowment = {}

    for row in ev_rows:
        original_ev_id = int(row["ID"])
        internal_ev_id = ev_id_map[original_ev_id]
        nearest_available_original_cp_id = min(
            (
                cp_id
                for cp_id in cp_original_ids
                if remaining_capacity[cp_id_map[cp_id]] > 0
            ),
            key=lambda cp_id: (distances[original_ev_id][cp_id], cp_id),
        )
        internal_cp_id = cp_id_map[nearest_available_original_cp_id]
        remaining_capacity[internal_cp_id] = remaining_capacity[internal_cp_id] - 1

        agents_e[internal_ev_id].initial_assignment = internal_cp_id
        agents_c[internal_cp_id].assigned.append(internal_ev_id)
        initial_endowment[internal_ev_id] = internal_cp_id

    return initial_endowment


def assign_random_available_cp(
    ev_rows,
    cp_rows,
    capacity,
    ev_id_map,
    cp_id_map,
    agents_e,
    agents_c,
    seed,
):
    """
    @brief Randomly assign each EV to a charging point while respecting capacity.
    @return Initial AgentE-to-AgentC assignment map using internal IDs.
    """
    available_cp_ids = []
    for row in cp_rows:
        internal_cp_id = cp_id_map[int(row["ID"])]
        available_cp_ids.extend([internal_cp_id] * capacity)

    rng = random.Random(seed)
    rng.shuffle(available_cp_ids)

    initial_endowment = {}
    for row, internal_cp_id in zip(ev_rows, available_cp_ids):
        internal_ev_id = ev_id_map[int(row["ID"])]
        agents_e[internal_ev_id].initial_assignment = internal_cp_id
        agents_c[internal_cp_id].assigned.append(internal_ev_id)
        initial_endowment[internal_ev_id] = internal_cp_id

    return initial_endowment


def run_algorithm(run_fn, initial_endowment, preferences):
    """
    @brief Run one matching algorithm and measure improvement, runtime, and memory.
    @return Tuple of improvement, elapsed seconds, peak MB, and cycle count.
    """
    tracemalloc.start()
    start_time = perf_counter()
    result = run_fn()
    elapsed_time = perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if len(result) == 3:
        final_match_e, _, cycle_count = result
    else:
        final_match_e, _ = result
        cycle_count = None

    improvement = total_rank_improvement(initial_endowment, final_match_e, preferences)
    peak_memory_mb = peak_memory / (1024 * 1024)
    return improvement, elapsed_time, peak_memory_mb, cycle_count


def algorithm_calls(agents_e, agents_c):
    return [
        (
            "WeakTTC",
            lambda: run_weak_ttc(
                agents_e,
                agents_c,
                cycle_sort_scheme="rank_diff_sum",
                return_cycle_count=True,
            ),
        ),
        (
            "ReactTTC",
            lambda: ReactTTC_variant(
                agents_e,
                agents_c,
                return_cycle_count=True,
            ),
        ),
        (
            "TTC",
            lambda: TTC(
                agents_e,
                agents_c,
                return_cycle_count=True,
            ),
        ),
    ]


def result_fieldnames():
    return [
        "dataset",
        "experiment",
        "x_label",
        "x_value",
        "n_e",
        "n_c",
        "capacity",
        "profile_size",
        "initial_assignment",
        "iteration_number",
        "seed",
        "method",
        "total_rank_improvement",
        "execution_time",
        "peak_memory_mb",
        "cycle_count",
    ]


def main():
    args = parse_args()
    ev_rows, cp_rows = read_real_nodes(args.nodes)
    distances = read_distance_matrix(args.distance_matrix)
    experiments = selected_experiments(args.experiment)

    rows = []
    for experiment_name in experiments:
        x_label = EXPERIMENT_LABELS[experiment_name]
        for setting in experiment_settings(experiment_name):
            if setting["n_c"] * setting["capacity"] < setting["n_e"]:
                raise ValueError(
                    f"Infeasible setting for {experiment_name}: "
                    f"n_e={setting['n_e']}, n_c={setting['n_c']}, "
                    f"capacity={setting['capacity']}"
                )

            for iteration_number in range(1, args.iterations + 1):
                seed = (
                    args.base_seed
                    + 100000 * experiments.index(experiment_name)
                    + 1000 * int(setting["x_value"])
                    + iteration_number
                )
                rng = random.Random(seed)
                selected_ev_rows = select_rows(ev_rows, setting["n_e"], rng)
                selected_cp_rows = select_rows(cp_rows, setting["n_c"], rng)
                agents_e, agents_c, initial_endowment, preferences = build_real_instance(
                    selected_ev_rows,
                    selected_cp_rows,
                    distances,
                    setting["capacity"],
                    setting["profile_size"],
                    args.initial_assignment,
                    seed,
                )

                print(
                    f"Running {experiment_name}: {x_label}={setting['x_value']}, "
                    f"n_e={setting['n_e']}, n_c={setting['n_c']}, "
                    f"capacity={setting['capacity']}, profile_size={setting['profile_size']}, "
                    f"iteration={iteration_number}, seed={seed}",
                    flush=True,
                )

                for method, run_fn in algorithm_calls(agents_e, agents_c):
                    print(f"  {method}", flush=True)
                    improvement, elapsed_time, peak_memory_mb, cycle_count = run_algorithm(
                        run_fn,
                        initial_endowment,
                        preferences,
                    )
                    rows.append(
                        {
                            "dataset": args.nodes.name,
                            "experiment": experiment_name,
                            "x_label": x_label,
                            "x_value": setting["x_value"],
                            "n_e": setting["n_e"],
                            "n_c": setting["n_c"],
                            "capacity": setting["capacity"],
                            "profile_size": setting["profile_size"],
                            "initial_assignment": args.initial_assignment,
                            "iteration_number": iteration_number,
                            "seed": seed,
                            "method": method,
                            "total_rank_improvement": improvement,
                            "execution_time": elapsed_time,
                            "peak_memory_mb": peak_memory_mb,
                            "cycle_count": cycle_count,
                        }
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=result_fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results written to {args.output}", flush=True)

    from tools.plot_real_ev_charging import plot_results

    plot_paths = plot_results(args.output, args.plot_dir)
    print("Plots written:", flush=True)
    for plot_path in plot_paths:
        print(f"  {plot_path}", flush=True)


if __name__ == "__main__":
    main()
