from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from egonet_mining.algorithms import (
    ego_feature_scores,
    louvain_pair_scores,
    louvain_partition,
)
from egonet_mining.data import load_graph_csv
from egonet_mining.evaluation import (
    evaluate_link_prediction,
    evaluate_partition,
    sample_negative_edges,
    split_train_test_edges,
)


DATASETS = ["facebook", "enron", "astro_ph"]


def timed(callable_obj, *args, **kwargs):
    start = time.perf_counter()
    result = callable_obj(*args, **kwargs)
    return result, time.perf_counter() - start


def run_dataset(name: str, args: argparse.Namespace) -> list[dict[str, float | str]]:
    graph = load_graph_csv(ROOT / "data" / f"{name}.csv")
    train, positives = split_train_test_edges(
        graph,
        test_ratio=args.test_ratio,
        max_test_edges=args.max_test_edges,
        seed=args.seed,
    )
    negatives = sample_negative_edges(graph, len(positives), seed=args.seed + 100)
    eval_pairs = [tuple(edge) for edge in positives] + [tuple(edge) for edge in negatives]

    rows: list[dict[str, float | str]] = []
    base = {
        "dataset": name,
        "nodes": float(graph.number_of_nodes()),
        "edges": float(graph.number_of_edges()),
        "train_edges": float(train.number_of_edges()),
        "heldout_edges": float(len(positives)),
    }

    for clusterer in args.clusterers:
        ego_result, ego_runtime = timed(
            ego_feature_scores,
            train,
            clusterer=clusterer,
            features=tuple(args.features),
            candidate_pairs=eval_pairs,
            alpha=args.alpha,
            beta=args.beta,
            max_iter=args.max_iter,
            min_cluster_size=args.min_cluster_size,
            seed=args.seed,
            construction=args.construction,
        )
        clusterer_name = clusterer.upper()
        for feature in args.features:
            ego_lp = evaluate_link_prediction(ego_result.scores[feature], positives, negatives)
            rows.append(
                {
                    **base,
                    "algorithm": f"EgoNet-{clusterer_name}-{feature}",
                    "clusterer": clusterer_name,
                    "feature": feature,
                    "runtime_sec": ego_runtime,
                    "community_count": "",
                    "mean_size": "",
                    "modularity": "",
                    "coverage": "",
                    "mean_density": "",
                    "mean_conductance": "",
                    **ego_lp,
                    **ego_result.stats,
                }
            )

    louvain_comms, louvain_runtime = timed(louvain_partition, train, seed=args.seed)
    louvain_scores = louvain_pair_scores(train, louvain_comms, eval_pairs)
    louvain_quality = evaluate_partition(train, louvain_comms)
    louvain_lp = evaluate_link_prediction(louvain_scores, positives, negatives)

    rows.append(
        {
            **base,
            "algorithm": "Louvain",
            "clusterer": "global",
            "feature": "community_common_neighbors",
            "runtime_sec": louvain_runtime,
            **louvain_quality,
            **louvain_lp,
            "ego_cluster_count": 0.0,
            "ego_cluster_mean_size": 0.0,
            "ego_cluster_mean_density": 0.0,
            "ego_cluster_mean_conductance": 0.0,
        },
    )
    return rows


def write_results(rows: list[dict[str, float | str]], output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "algorithm",
        "clusterer",
        "feature",
        "nodes",
        "edges",
        "train_edges",
        "heldout_edges",
        "runtime_sec",
        "community_count",
        "mean_size",
        "modularity",
        "coverage",
        "mean_density",
        "mean_conductance",
        "auc",
        "average_precision",
        "ego_cluster_count",
        "ego_cluster_mean_size",
        "ego_cluster_mean_density",
        "ego_cluster_mean_conductance",
    ]
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, float | str]]) -> None:
    def format_optional(value: float | str, width: int = 8) -> str:
        if value == "":
            return f"{'n/a':>{width}}"
        return f"{float(value):>{width}.4f}"

    print("\nExperiment summary")
    print("=" * 118)
    print(
        f"{'dataset':<12} {'algorithm':<18} {'time(s)':>9} {'mod':>8} "
        f"{'local_den':>10} {'AUC':>8} {'AP':>8}"
    )
    for row in rows:
        print(
            f"{row['dataset']:<12} {row['algorithm']:<18} "
            f"{float(row['runtime_sec']):>9.5f} "
            f"{format_optional(row['modularity'])} "
            f"{float(row['ego_cluster_mean_density']):>10.4f} "
            f"{float(row['auc']):>8.4f} "
            f"{float(row['average_precision']):>8.4f}"
        )
    ego_times = {
        (row["dataset"], row["clusterer"]): float(row["runtime_sec"])
        for row in rows
        if str(row["algorithm"]).startswith("EgoNet-") and row["feature"] == "W1"
    }
    louvain_times = [float(row["runtime_sec"]) for row in rows if row["algorithm"] == "Louvain"]
    print("=" * 118)
    print(f"mean EgoNet runtime:    {statistics.mean(ego_times.values()):.5f}s")
    print(f"mean Louvain runtime:   {statistics.mean(louvain_times):.5f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Project 2 topic 1 experiments.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--output", default="results/experiment_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--max-test-edges", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=1.0)
    parser.add_argument("--construction", choices=["triangle", "neighbor"], default="triangle")
    parser.add_argument("--clusterers", nargs="+", choices=["apm", "cc"], default=["apm", "cc"])
    parser.add_argument("--features", nargs="+", choices=["W1", "W2", "W3", "W4"], default=["W1", "W2", "W3", "W4"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Avoid charging one-time NetworkX Louvain setup to the first dataset.
    _ = louvain_partition(load_graph_csv(ROOT / "data" / f"{args.datasets[0]}.csv"), seed=args.seed)
    rows: list[dict[str, float | str]] = []
    for dataset in args.datasets:
        rows.extend(run_dataset(dataset, args))
    output = ROOT / args.output
    write_results(rows, output)
    print_summary(rows)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
