from __future__ import annotations

import csv
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "dynamic_experiment_results.csv"
OUT = ROOT / "results" / "dynamic_speedup.png"


def main() -> None:
    with RESULTS.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))

    labels = [f"{row['dataset']}-{int(float(row['batch']))}" for row in rows]
    dynamic = [float(row["dynamic_update_sec"]) for row in rows]
    full = [float(row["full_egonet_sec"]) for row in rows]
    speedups = [float(row["speedup_vs_full"]) for row in rows]
    xs = range(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    width = 0.36
    axes[0].bar([x - width / 2 for x in xs], dynamic, width=width, label="Incremental", color="#2f6f73")
    axes[0].bar([x + width / 2 for x in xs], full, width=width, label="Full rebuild", color="#b15d3a")
    axes[0].set_title("Dynamic Update Runtime")
    axes[0].set_ylabel("seconds")
    axes[0].set_xticks(list(xs), labels, rotation=25, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].bar(labels, speedups, color="#6b7d2a")
    axes[1].set_title("Speedup vs Full Rebuild")
    axes[1].set_ylabel("x")
    axes[1].set_xticks(list(xs), labels, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.25)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
