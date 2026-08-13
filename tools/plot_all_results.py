import argparse
from pathlib import Path

from plot_real_ev_charging import plot_results as plot_real_ev_charging
from plot_varying_agentsize import plot_results as plot_varying_agentsize
from plot_varying_agentsize_fixed_resource import (
    plot_results as plot_varying_agentsize_fixed_resource,
)
from plot_varying_capacity import plot_results as plot_varying_capacity
from plot_varying_max_rank_size import plot_results as plot_varying_max_rank_size
from plot_varying_resource_ratio import plot_results as plot_varying_resource_ratio


REPO_ROOT = Path(__file__).resolve().parents[1]


PLOT_JOBS = [
    {
        "name": "synthetic varying total agents",
        "input": REPO_ROOT / "results" / "results_varying_agentsize.csv",
        "out_dir": REPO_ROOT / "results" / "agentsize_plots",
        "plotter": plot_varying_agentsize,
    },
    {
        "name": "synthetic varying total agents with fixed resources",
        "input": REPO_ROOT / "results" / "results_varying_agentsize_fixed_resource.csv",
        "out_dir": REPO_ROOT / "results" / "agentsize_fixed_resource_plots",
        "plotter": plot_varying_agentsize_fixed_resource,
    },
    {
        "name": "synthetic varying resource ratio",
        "input": REPO_ROOT / "results" / "results_varying_resource_ratio.csv",
        "out_dir": REPO_ROOT / "results" / "resource_ratio_plots",
        "plotter": plot_varying_resource_ratio,
    },
    {
        "name": "synthetic varying profile size",
        "input": REPO_ROOT / "results" / "results_varying_max_rank_size.csv",
        "out_dir": REPO_ROOT / "results" / "max_rank_size_plots",
        "plotter": plot_varying_max_rank_size,
    },
    {
        "name": "synthetic varying capacity",
        "input": REPO_ROOT / "results" / "results_varying_capacity.csv",
        "out_dir": REPO_ROOT / "results" / "capacity_plots",
        "plotter": plot_varying_capacity,
    },
    {
        "name": "real EV charging experiments",
        "input": REPO_ROOT / "results" / "results_real_ev_charging_experiments.csv",
        "out_dir": REPO_ROOT / "results" / "real_ev_charging_experiment_plots",
        "plotter": plot_real_ev_charging,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate all plots from existing results CSV files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory containing results CSVs and receiving plot folders.",
    )
    parser.add_argument(
        "--format",
        choices=("pdf", "png"),
        default="pdf",
        help="Output plot format.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected CSV file is missing.",
    )
    return parser.parse_args()


def rebase_job(job, results_dir):
    input_path = results_dir / job["input"].name
    out_dir = results_dir / job["out_dir"].name
    return input_path, out_dir


def main():
    args = parse_args()
    generated = []
    skipped = []

    for job in PLOT_JOBS:
        input_path, out_dir = rebase_job(job, args.results_dir)
        if not input_path.exists():
            message = f"Skipping {job['name']}: missing {input_path}"
            if args.strict:
                raise FileNotFoundError(message)
            print(message, flush=True)
            skipped.append(input_path)
            continue

        print(f"Plotting {job['name']} from {input_path}", flush=True)
        plot_paths = job["plotter"](input_path, out_dir, args.format)
        generated.extend(plot_paths)
        for plot_path in plot_paths:
            print(f"  {plot_path}", flush=True)

    print(
        f"Generated {len(generated)} plot files; skipped {len(skipped)} missing CSV files.",
        flush=True,
    )


if __name__ == "__main__":
    main()
