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


METRIC_NAMES = {
    "total_rank_improvement": "Total rank improvement",
    "execution_time": "Execution time (s)",
}

REAL_PALETTE = "Dark2"


EXPERIMENT_LABELS = {
    "varying_ev": "Total EV",
    "varying_cp": "Total Charging Points",
    "varying_capacity": "Capacity",
    "varying_profile_size": "Profile Size",
}


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Plot real EV charging-point experiment results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "results" / "results_real_ev_charging_experiments.csv",
        help="Path to results_real_ev_charging_experiments.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "results" / "real_ev_charging_experiment_plots",
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--format",
        choices=("pdf", "png"),
        default="pdf",
        help="Output plot format.",
    )
    return parser.parse_args()


def save_plot(out_dir, out_file_name, plot_format):
    out_path = out_dir / f"{out_file_name}.{plot_format}"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, format=plot_format)
    plt.close()
    return out_path


def safe_label(label):
    return label.lower().replace(" ", "_")


def plot_execution_time_line(
    df_experiment,
    experiment_name,
    metric_label,
    out_dir,
    plot_format,
):
    x_label = EXPERIMENT_LABELS.get(experiment_name, df_experiment["x_label"].iloc[0])
    df_metric = (
        df_experiment.groupby(["x_value", "method"], as_index=False)["execution_time"]
        .mean()
        .sort_values("x_value")
    )

    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df_metric,
        x="x_value",
        y="execution_time",
        hue="method",
        marker="o",
        estimator=None,
        errorbar=None,
        palette=REAL_PALETTE,
    )
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(metric_label, fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(title="Method", title_fontsize=18, fontsize=18)
    plt.tight_layout()

    return save_plot(
        out_dir,
        f"execution_time_vs_{safe_label(x_label)}_real_ev_charging",
        plot_format,
    )


def plot_rank_improvement_bar(
    df_experiment,
    experiment_name,
    metric_label,
    out_dir,
    plot_format,
):
    x_label = EXPERIMENT_LABELS.get(experiment_name, df_experiment["x_label"].iloc[0])
    df_metric = df_experiment.copy()
    order = [str(value) for value in sorted(df_metric["x_value"].unique())]
    df_metric["x_value_label"] = df_metric["x_value"].astype(int).astype(str)
    df_metric["x_value_label"] = pd.Categorical(
        df_metric["x_value_label"], categories=order, ordered=True
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=df_metric,
        x="x_value_label",
        y="total_rank_improvement",
        hue="method",
        order=order,
        errorbar="sd",
        dodge=True,
        palette=REAL_PALETTE,
    )
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(metric_label, fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(title="Method", title_fontsize=18, fontsize=18)
    plt.tight_layout()

    return save_plot(
        out_dir,
        f"total_rank_improvement_bar_vs_{safe_label(x_label)}_real_ev_charging",
        plot_format,
    )


def plot_results(input_path, out_dir, plot_format="pdf"):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    df["x_value"] = pd.to_numeric(df["x_value"])

    sns.set(style="whitegrid", palette=REAL_PALETTE)

    generated = []
    for experiment_name in sorted(df["experiment"].unique()):
        df_experiment = df[df["experiment"] == experiment_name]
        generated.append(
            plot_rank_improvement_bar(
                df_experiment,
                experiment_name,
                METRIC_NAMES["total_rank_improvement"],
                out_dir,
                plot_format,
            )
        )
        generated.append(
            plot_execution_time_line(
                df_experiment,
                experiment_name,
                METRIC_NAMES["execution_time"],
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
