import argparse
import csv
import os
import random
import sys
import tracemalloc
from pathlib import Path
from time import perf_counter

from agents import initialize_agents, initialize_preferences, set_initial_endowment
from baseline import ReactTTC_variant, TTC
from eval import total_rank_improvement
from weakTTC import run_weak_ttc


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run WeakTTC baselines with fixed n_e and varying n_c / n_e ratio."
    )
    parser.add_argument("--n-e", type=int, default=100, help="Fixed number of AgentE agents.")
    parser.add_argument("--iterations", type=int, default=10, help="Iterations per ratio.")
    parser.add_argument("--base-seed", type=int, default=123, help="Base random seed.")
    parser.add_argument("--capacity", type=int, default=2, help="Capacity for each AgentC.")
    parser.add_argument(
        "--capacity-type",
        choices=("strict", "loose"),
        default="strict",
        help="Capacity assignment mode.",
    )
    parser.add_argument(
        "--max-rank-size",
        type=int,
        default=2,
        help="Maximum number of AgentC IDs per weak-preference rank.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "results_varying_resource_ratio.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=REPO_ROOT / "results" / "resource_ratio_plots",
        help="Directory where plots will be saved.",
    )
    return parser.parse_args()


def run_algorithm(run_fn, initial_endowment, preferences):
    """
    @brief Run one matching algorithm and measure improvement, runtime, and memory.
    @param run_fn Callable that returns final AgentE and AgentC match maps.
    @param initial_endowment Initial AgentE-to-AgentC assignment map.
    @param preferences Mapping from AgentE IDs to weak preference dictionaries.
    @return Tuple of total rank improvement, elapsed seconds, peak memory in MB,
        and total cycle count.
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


def main():
    """
    @brief Run experiments with n_e fixed and n_c varied by ratio.

    Keeps n_e fixed at 100 by default, sweeps ratio from 0.5 to 1.0 in steps of
    0.1, sets n_c = int(n_e * ratio), and writes algorithm metrics to CSV.
    """
    args = parse_args()
    cycle_sort_scheme = "rank_diff_sum"
    preferences_output_path = os.path.join(
        "/tmp", "weakttc_varying_resource_ratio_preferences.json"
    )

    fieldnames = [
        "n_e",
        "n_c",
        "resource_ratio",
        "capacity",
        "capacity_type",
        "max_rank_size",
        "iteration_number",
        "weak_ttc_total_rank_improvement",
        "weak_ttc_execution_time",
        "weak_ttc_peak_memory_mb",
        "weak_ttc_cycle_count",
        "react_ttc_total_rank_improvement",
        "react_ttc_execution_time",
        "react_ttc_peak_memory_mb",
        "react_ttc_cycle_count",
        "ttc_total_rank_improvement",
        "ttc_execution_time",
        "ttc_peak_memory_mb",
        "ttc_cycle_count",
    ]

    os.makedirs(args.output.parent, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for ratio_step in range(5, 11):
            resource_ratio = ratio_step / 10
            n_c = int(args.n_e * resource_ratio)
            rng = random.Random(args.base_seed + ratio_step)

            for iteration_number in range(1, args.iterations + 1):
                seed = rng.randrange(1_000_000_000)
                print(
                    f"Running n_e={args.n_e}, n_c={n_c}, ratio={resource_ratio:.1f}, "
                    f"iteration={iteration_number}, seed={seed}",
                    flush=True,
                )
                agents_e, agents_c = initialize_agents(
                    n_e=args.n_e,
                    n_c=n_c,
                    capacity=args.capacity,
                    capacity_type=args.capacity_type,
                    seed=seed,
                )
                initial_endowment = set_initial_endowment(agents_e, agents_c, seed=seed)
                preferences = initialize_preferences(
                    agents_e,
                    agents_c,
                    max_rank_size=args.max_rank_size,
                    output_path=preferences_output_path,
                    seed=seed,
                )

                print("  WeakTTC", flush=True)
                (
                    weak_improvement,
                    weak_elapsed,
                    weak_peak_memory_mb,
                    weak_cycle_count,
                ) = run_algorithm(
                    lambda: run_weak_ttc(
                        agents_e,
                        agents_c,
                        cycle_sort_scheme=cycle_sort_scheme,
                        return_cycle_count=True,
                    ),
                    initial_endowment,
                    preferences,
                )
                print("  ReACT-TTC", flush=True)
                (
                    react_improvement,
                    react_elapsed,
                    react_peak_memory_mb,
                    react_cycle_count,
                ) = run_algorithm(
                    lambda: ReactTTC_variant(
                        agents_e,
                        agents_c,
                        return_cycle_count=True,
                    ),
                    initial_endowment,
                    preferences,
                )
                print("  TTC", flush=True)
                (
                    ttc_improvement,
                    ttc_elapsed,
                    ttc_peak_memory_mb,
                    ttc_cycle_count,
                ) = run_algorithm(
                    lambda: TTC(
                        agents_e,
                        agents_c,
                        return_cycle_count=True,
                    ),
                    initial_endowment,
                    preferences,
                )

                writer.writerow(
                    {
                        "n_e": args.n_e,
                        "n_c": n_c,
                        "resource_ratio": resource_ratio,
                        "capacity": args.capacity,
                        "capacity_type": args.capacity_type,
                        "max_rank_size": args.max_rank_size,
                        "iteration_number": iteration_number,
                        "weak_ttc_total_rank_improvement": weak_improvement,
                        "weak_ttc_execution_time": weak_elapsed,
                        "weak_ttc_peak_memory_mb": weak_peak_memory_mb,
                        "weak_ttc_cycle_count": weak_cycle_count,
                        "react_ttc_total_rank_improvement": react_improvement,
                        "react_ttc_execution_time": react_elapsed,
                        "react_ttc_peak_memory_mb": react_peak_memory_mb,
                        "react_ttc_cycle_count": react_cycle_count,
                        "ttc_total_rank_improvement": ttc_improvement,
                        "ttc_execution_time": ttc_elapsed,
                        "ttc_peak_memory_mb": ttc_peak_memory_mb,
                        "ttc_cycle_count": ttc_cycle_count,
                    }
                )
                csv_file.flush()
                print(
                    f"Finished ratio={resource_ratio:.1f}, iteration={iteration_number}",
                    flush=True,
                )

    print(f"Results written to {args.output}", flush=True)
    from tools.plot_varying_resource_ratio import plot_results

    plot_paths = plot_results(args.output, args.plot_dir)
    print("Plots written:", flush=True)
    for plot_path in plot_paths:
        print(f"  {plot_path}", flush=True)


if __name__ == "__main__":
    main()
