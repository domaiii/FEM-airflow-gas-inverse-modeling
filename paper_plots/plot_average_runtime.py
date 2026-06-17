from pathlib import Path

import numpy as np
import pandas as pd

results = ['/app/scenarios/10x6_multiple_obstacles/eval/metrics_summary.csv',
           '/app/scenarios/10x6_labyrinth/eval/metrics_summary.csv',
           '/app/scenarios/10x6_appartment/eval/metrics_summary.csv']

methods = ["GMRF", "GPR", "NSRM", "WFNS"]
sample_sizes = [50, 100, 200]


def pooled_mean_std(rows: pd.DataFrame) -> tuple[float, float, int]:
    counts = rows["count"].to_numpy(dtype=int)
    means = rows["mean"].to_numpy(dtype=float)
    variances = rows["var"].to_numpy(dtype=float)
    total_count = int(counts.sum())

    pooled_mean = float(np.sum(counts * means) / total_count)
    pooled_ss = np.sum(
        (counts - 1) * variances
        + counts * (means - pooled_mean) ** 2
    )
    pooled_std = float(np.sqrt(pooled_ss / (total_count - 1)))
    return pooled_mean, pooled_std, total_count


all_results = pd.concat(
    [pd.read_csv(path) for path in results],
    ignore_index=True,
)
runtime = all_results.loc[
    (all_results["metric"] == "estimation_runtime_sec")
    & all_results["sample_size"].isin(sample_sizes)
]

for method in methods:
    for sample_size in sample_sizes:
        rows = runtime.loc[
            (runtime["method"] == method)
            & (runtime["sample_size"] == sample_size)
        ]
        if rows.empty:
            print(f"{method}, K={sample_size}: no runtime data")
            continue

        mean, std, count = pooled_mean_std(rows)
        print(f"{method}, K={sample_size}: {mean:.6f} +/- {std:.6f} s (n={count})")
