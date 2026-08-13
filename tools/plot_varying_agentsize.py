import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/weakttc_mplconfig")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams


# Avoid type 3 fonts
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["text.usetex"] = False


METHOD_NAMES = {
    "weak_ttc": "W-TTC",
    "react_ttc": "ReACT-TTC",
    "ttc": "TTC",
}

METRIC_NAMES = {
    "total_rank_improvement": "Total rank improvement",
    "execution_time": "Execution time (s)",
    "peak_memory_mb": "Peak memory (MB)",
    "cycle_count": "Cycle count",
}

def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Plot results_varying_agentsize.csv with seaborn."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "results" / "results_varying_agentsize.csv",
        help="Path to results_varying_agentsize.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "results" / "agentsize_plots",
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--format",
        choices=("pdf", "png"),
        default="pdf",
        help="Output plot format.",
    )
    return parser.parse_args()


def to_long_format(df):
    id_vars = [
        "n_e",
        "n_c",
        "capacity",
        "capacity_type",
        "max_rank_size",
        "iteration_number",
    ]
    value_vars = [
        f"{method}_{metric}"
        for method in METHOD_NAMES
        for metric in METRIC_NAMES
    ]

    df_long = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="method_metric",
        value_name="Value",
    )
    df_long[["method_key", "metric_key"]] = df_long["method_metric"].str.extract(
        r"^(weak_ttc|react_ttc|ttc)_(.*)$"
    )
    df_long["Method"] = df_long["method_key"].map(METHOD_NAMES)
    df_long["Metric"] = df_long["metric_key"].map(METRIC_NAMES)
    df_long["Total Agents"] = df_long["n_e"].astype(int)
    return df_long


def save_plot(out_dir, out_file_name, plot_format):
    out_path = out_dir / f"{out_file_name}.{plot_format}"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, format=plot_format)
    plt.close()
    return out_path


def apply_common_labels(xlabel, ylabel):
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(title_fontsize=18, fontsize=18)
    plt.tight_layout()


def total_agents_order(df_long):
    return [str(value) for value in sorted(df_long["Total Agents"].astype(int).unique())]


def plot_lines(df_long, out_dir, plot_format):
    generated = []

    for metric_key, metric_label in METRIC_NAMES.items():
        df_metric = df_long[df_long["metric_key"] == metric_key]
        if df_metric.empty:
            continue
        df_metric = (
            df_metric.groupby(["Total Agents", "Method"], as_index=False)["Value"]
            .mean()
            .sort_values("Total Agents")
        )
        plt.figure(figsize=(8, 6))
        sns.lineplot(
            data=df_metric,
            x="Total Agents",
            y="Value",
            hue="Method",
            marker="o",
            estimator=None,
            errorbar=None,
        )

        apply_common_labels("Total Agents", metric_label)
        generated.append(
            save_plot(out_dir, f"{metric_key}_vs_agentsize", plot_format)
        )

    return generated


def plot_bars(df_long, out_dir, plot_format):
    generated = []
    df_plot = df_long.copy()
    order = total_agents_order(df_plot)
    df_plot["Total Agents"] = df_plot["Total Agents"].astype(int).astype(str)
    df_plot["Total Agents"] = pd.Categorical(df_plot["Total Agents"], categories=order, ordered=True)

    for metric_key, metric_label in METRIC_NAMES.items():
        df_metric = df_plot[df_plot["metric_key"] == metric_key]
        if df_metric.empty:
            continue
        plt.figure(figsize=(8, 6))
        sns.barplot(
            data=df_metric,
            x="Total Agents",
            y="Value",
            hue="Method",
            order=order,
            errorbar="sd",
            dodge=True,
        )

        plt.xlabel("Total Agents", fontsize=20)
        plt.ylabel(metric_label, fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(title="Method", title_fontsize=18, fontsize=18)
        plt.tight_layout()
        generated.append(
            save_plot(out_dir, f"{metric_key}_bar_vs_agentsize", plot_format)
        )

    return generated


def plot_results(input_path, out_dir, plot_format="pdf"):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    df_long = to_long_format(df)

    # Match the existing plotting scripts.
    sns.set(style="whitegrid", palette="Set1")

    generated = []
    generated.extend(
        plot_bars(
            df_long[df_long["metric_key"] == "total_rank_improvement"],
            out_dir,
            plot_format,
        )
    )
    generated.extend(
        plot_lines(
            df_long[df_long["metric_key"] == "execution_time"],
            out_dir,
            plot_format,
        )
    )
    return generated


def main():
    args = parse_args()
    generated = plot_results(args.input, args.out_dir, args.format)

    print(f"Saved {len(generated)} plots to {args.out_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
