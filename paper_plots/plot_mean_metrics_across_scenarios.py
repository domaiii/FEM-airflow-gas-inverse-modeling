from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUMMARY_PATHS = [
    Path("/app/scenarios/10x6_appartment/eval/metrics_summary.csv"),
    Path("/app/scenarios/10x6_labyrinth/eval/metrics_summary.csv"),
    Path("/app/scenarios/10x6_multiple_obstacles/eval/metrics_summary.csv"),
]
OUTPUT_PATH = Path("/app/noisy_plot.pdf")


def load_summaries() -> pd.DataFrame:
    frames = []
    for path in SUMMARY_PATHS:
        df = pd.read_csv(path)
        df["scenario"] = path.parents[1].name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def mean_over_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["method", "sample_size", "metric"], as_index=False)
        .agg(mean=("mean", "mean"), n_scenarios=("scenario", "nunique"))
    )


def method_colors(methods: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("Set1")
    colors = [cmap.colors[i] for i in [0, 1, 2, 4, 5]]
    return {method: colors[i % len(colors)] for i, method in enumerate(methods)}


def metric_values(summary: pd.DataFrame, metric: str, method: str) -> pd.DataFrame:
    values = summary[(summary["metric"] == metric) & (summary["method"] == method)]
    return values.sort_values("sample_size")


def main() -> None:
    raw = load_summaries()
    summary = mean_over_scenarios(raw)
    methods = sorted(summary["method"].unique())
    colors = method_colors(methods)

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })

    fig, ax_ang = plt.subplots(figsize=(4.6, 3.7), dpi=300)
    ax_mag = ax_ang.twinx()

    angular_handles = []
    angular_labels = []
    magnitude_handles = []
    magnitude_labels = []

    for method in methods:
        values = metric_values(summary, "angular_error_mean_deg", method)
        if values.empty:
            continue
        line, = ax_ang.plot(
            values["sample_size"].to_numpy(dtype=float),
            values["mean"].to_numpy(dtype=float),
            color=colors[method],
            marker="D",
            markersize=3.2,
            linewidth=1.25,
            linestyle="--",
            label=method,
        )
        angular_handles.append(line)
        angular_labels.append(method)

    for method in methods:
        values = metric_values(summary, "magnitude_mae_m_per_s", method)
        if values.empty:
            continue
        line, = ax_mag.plot(
            values["sample_size"].to_numpy(dtype=float),
            values["mean"].to_numpy(dtype=float),
            color=colors[method],
            marker="o",
            markersize=3.2,
            linewidth=1.25,
            linestyle="-",
            label=method,
        )
        magnitude_handles.append(line)
        magnitude_labels.append(method)

    ax_ang.set_ylim(10.0, 100.0)
    ax_ang.set_yticks(np.arange(10.0, 101.0, 10.0))

    ax_mag.set_ylim(0.1, 1.0)
    ax_mag.set_yticks(np.arange(0.1, 1.01, 0.1))

    ax_ang.set_xlabel("Number of Measurements")
    ax_ang.set_ylabel("Mean Angular Error [deg]")
    ax_mag.set_ylabel("Mean Magnitude Error [m/s]")
    ax_ang.grid(axis="y", alpha=0.28, linewidth=0.5)
    ax_ang.tick_params(axis="both", labelsize=7)
    ax_mag.tick_params(axis="y", labelsize=7)

    if angular_handles:
        angular_legend = ax_ang.legend(
            angular_handles,
            angular_labels,
            title="Angular Errors",
            loc="upper right",
            bbox_to_anchor=(0.45, 0.99),
            fontsize=7,
            title_fontsize=7,
            handlelength=2.6,
            frameon=True,
        )
        ax_ang.add_artist(angular_legend)
    if magnitude_handles:
        ax_mag.legend(
            magnitude_handles,
            magnitude_labels,
            title="Magnitude Errors",
            loc="upper right",
            bbox_to_anchor=(0.82, 0.99),
            fontsize=7,
            title_fontsize=7,
            handlelength=2.6,
            frameon=True,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", pad_inches=0.01)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
