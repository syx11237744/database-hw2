from __future__ import annotations

import csv
import pathlib
import statistics

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "experiment_results.csv"
OUT = ROOT / "results" / "quality_runtime.png"


def main() -> None:
    with RESULTS.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))

    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    algorithms = list(dict.fromkeys(row["algorithm"] for row in rows))
    apm_colors = {
        "W1": "#b7d3ec",
        "W2": "#8dbfe2",
        "W3": "#3f88c5",
        "W4": "#155a9a",
    }
    cc_colors = {
        "W1": "#b8d9b1",
        "W2": "#96c98f",
        "W3": "#4f9d69",
        "W4": "#1f6f43",
    }
    colors: dict[str, str] = {}
    for algorithm in algorithms:
        if algorithm.startswith("EgoNet-APM-"):
            colors[algorithm] = apm_colors[algorithm.rsplit("-", 1)[1]]
        elif algorithm.startswith("EgoNet-CC-"):
            colors[algorithm] = cc_colors[algorithm.rsplit("-", 1)[1]]
        elif algorithm == "Louvain":
            colors[algorithm] = "#b15d3a"
        else:
            colors[algorithm] = "#777777"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "runtime_sec", "Runtime"),
        (axes[1], "auc", "Link AUC"),
        (axes[2], "average_precision", "Link AP"),
    ]:
        width = min(0.8 / max(len(algorithms), 1), 0.12)
        xs = range(len(datasets))
        for offset, algorithm in enumerate(algorithms):
            values = [
                statistics.mean(
                    float(row[metric])
                    for row in rows
                    if row["dataset"] == dataset and row["algorithm"] == algorithm and row[metric] != ""
                )
                for dataset in datasets
            ]
            shifted = [x + (offset - (len(algorithms) - 1) / 2) * width for x in xs]
            ax.bar(shifted, values, width=width, label=algorithm, color=colors.get(algorithm))
        ax.set_title(title)
        ax.set_xticks(list(xs), datasets, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("seconds")
    axes[1].set_ylabel("score")
    axes[2].set_ylabel("score")
    axes[2].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
