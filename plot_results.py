from __future__ import annotations

import csv
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "experiment_results.csv"
OUT = ROOT / "results" / "quality_runtime.png"


def main() -> None:
    with RESULTS.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))

    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    algorithms = list(dict.fromkeys(row["algorithm"] for row in rows))
    colors = {"EgoNet-APM": "#2f6f73", "Louvain": "#b15d3a"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "runtime_sec", "Runtime"),
        (axes[1], "modularity", "Modularity"),
        (axes[2], "auc", "Link AUC"),
    ]:
        width = 0.36
        xs = range(len(datasets))
        for offset, algorithm in enumerate(algorithms):
            values = [
                float(next(row[metric] for row in rows if row["dataset"] == dataset and row["algorithm"] == algorithm))
                for dataset in datasets
            ]
            shifted = [x + (offset - 0.5) * width for x in xs]
            ax.bar(shifted, values, width=width, label=algorithm, color=colors.get(algorithm))
        ax.set_title(title)
        ax.set_xticks(list(xs), datasets, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("seconds")
    axes[1].set_ylabel("score")
    axes[2].set_ylabel("score")
    axes[2].legend(frameon=False, loc="best")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
