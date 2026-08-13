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
    @brief Run experiments varying the number of AgentE agents.

    Sweeps n_e from 10 to 40 by 10, keeps n_c and capacity fixed, and runs
    10 reproducible random iterations for each size. Results are written to CSV.
    Memory is reported as peak traced Python allocation in MB for each algorithm
    call. Cycle count is the total number of cycles enumerated across all rounds
    of an algorithm call.
    """
    
    max_rank_size = 2
    iterations = 10
    base_seed = 123
    cycle_sort_scheme = "rank_diff_sum"
    output_path = REPO_ROOT / "results" / "results_varying_agentsize.csv"
    preferences_output_path = os.path.join("/tmp", "weakttc_varying_agentsize_preferences.json")

    # rng = random.Random(base_seed)
    fieldnames = [
        "n_e",
        "n_c",
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

    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for n_e in range(10, 101, 10):
            rng = random.Random(base_seed)
            n_c = int(n_e * 0.7)
            capacity = 2
            capacity_type = "strict"
            for iteration_number in range(1, iterations + 1):
                seed = rng.randrange(1_000_000_000)
                print(
                    f"Running n_e={n_e}, iteration={iteration_number}, seed={seed}",
                    flush=True,
                )
                agents_e, agents_c = initialize_agents(
                    n_e=n_e,
                    n_c=n_c,
                    capacity=capacity,
                    capacity_type=capacity_type,
                    seed=seed,
                )
                initial_endowment = set_initial_endowment(agents_e, agents_c, seed=seed)
                preferences = initialize_preferences(
                    agents_e,
                    agents_c,
                    max_rank_size=max_rank_size,
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
                        "n_e": n_e,
                        "n_c": n_c,
                        "capacity": capacity,
                        "capacity_type": capacity_type,
                        "max_rank_size": max_rank_size,
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
                print(f"Finished n_e={n_e}, iteration={iteration_number}", flush=True)

    print(f"Results written to {output_path}", flush=True)
    from tools.plot_varying_agentsize import plot_results

    plot_paths = plot_results(
        output_path,
        REPO_ROOT / "results" / "agentsize_plots",
    )
    print("Plots written:", flush=True)
    for plot_path in plot_paths:
        print(f"  {plot_path}", flush=True)


if __name__ == "__main__":
    main()
