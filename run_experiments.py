from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import sys
import time
from collections import defaultdict

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
    precision_recall_curve_at_k,
    sample_negative_edges,
    split_train_test_edges,
)


DATASETS = ["facebook", "enron", "astro_ph"]


def timed(callable_obj, *args, **kwargs):
    start = time.perf_counter()
    result = callable_obj(*args, **kwargs)
    return result, time.perf_counter() - start


def circle_reconstruction_fields() -> dict[str, float | str]:
    return {
        "circle_available": "false",
        "circle_precision": "",
        "circle_recall": "",
        "circle_f1": "",
    }


def empty_ego_stats() -> dict[str, float]:
    return {
        "ego_cluster_count": 0.0,
        "ego_cluster_mean_size": 0.0,
        "ego_cluster_mean_density": 0.0,
        "ego_cluster_density_p10": 0.0,
        "ego_cluster_density_p25": 0.0,
        "ego_cluster_density_p50": 0.0,
        "ego_cluster_density_p75": 0.0,
        "ego_cluster_density_p90": 0.0,
        "ego_cluster_mean_conductance": 0.0,
        "ego_cluster_conductance_p10": 0.0,
        "ego_cluster_conductance_p25": 0.0,
        "ego_cluster_conductance_p50": 0.0,
        "ego_cluster_conductance_p75": 0.0,
        "ego_cluster_conductance_p90": 0.0,
    }


def run_dataset_seed(
    name: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    graph = load_graph_csv(ROOT / "data" / f"{name}.csv")
    train, positives = split_train_test_edges(
        graph,
        test_ratio=args.test_ratio,
        max_test_edges=args.max_test_edges,
        seed=seed,
    )
    negative_seed = seed + args.negative_seed_offset
    negatives = sample_negative_edges(graph, len(positives), seed=negative_seed)
    eval_pairs = [tuple(edge) for edge in positives] + [tuple(edge) for edge in negatives]

    rows: list[dict[str, float | str]] = []
    topk_rows: list[dict[str, float | str]] = []
    base = {
        "dataset": name,
        "seed": float(seed),
        "split_seed": float(seed),
        "negative_seed": float(negative_seed),
        "apm_seed": float(seed),
        "apm_seed_strategy": "base_seed_plus_sorted_ego_offset",
        "construction": args.construction,
        "test_ratio": float(args.test_ratio),
        "max_test_edges": float(args.max_test_edges),
        "nodes": float(graph.number_of_nodes()),
        "edges": float(graph.number_of_edges()),
        "train_edges": float(train.number_of_edges()),
        "heldout_edges": float(len(positives)),
        "negative_edges": float(len(negatives)),
    }
    circle_fields = circle_reconstruction_fields()

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
            seed=seed,
            construction=args.construction,
        )
        clusterer_name = clusterer.upper()
        for feature in args.features:
            ego_lp = evaluate_link_prediction(ego_result.scores[feature], positives, negatives)
            algorithm = f"EgoNet-{clusterer_name}-{feature}"
            rows.append(
                {
                    **base,
                    "algorithm": algorithm,
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
                    **circle_fields,
                }
            )
            for point in precision_recall_curve_at_k(ego_result.scores[feature], positives, negatives, args.top_k):
                topk_rows.append(
                    {
                        "dataset": name,
                        "seed": float(seed),
                        "algorithm": algorithm,
                        "clusterer": clusterer_name,
                        "feature": feature,
                        **point,
                    }
                )

    louvain_comms, louvain_runtime = timed(louvain_partition, train, seed=seed)
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
            **empty_ego_stats(),
            **circle_fields,
        },
    )
    for point in precision_recall_curve_at_k(louvain_scores, positives, negatives, args.top_k):
        topk_rows.append(
            {
                "dataset": name,
                "seed": float(seed),
                "algorithm": "Louvain",
                "clusterer": "global",
                "feature": "community_common_neighbors",
                **point,
            }
        )
    return rows, topk_rows


def write_results(rows: list[dict[str, float | str]], output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "algorithm",
        "clusterer",
        "feature",
        "seed",
        "split_seed",
        "negative_seed",
        "apm_seed",
        "apm_seed_strategy",
        "construction",
        "test_ratio",
        "max_test_edges",
        "nodes",
        "edges",
        "train_edges",
        "heldout_edges",
        "negative_edges",
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
        "ego_cluster_density_p10",
        "ego_cluster_density_p25",
        "ego_cluster_density_p50",
        "ego_cluster_density_p75",
        "ego_cluster_density_p90",
        "ego_cluster_mean_conductance",
        "ego_cluster_conductance_p10",
        "ego_cluster_conductance_p25",
        "ego_cluster_conductance_p50",
        "ego_cluster_conductance_p75",
        "ego_cluster_conductance_p90",
        "circle_available",
        "circle_precision",
        "circle_recall",
        "circle_f1",
    ]
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_topk(rows: list[dict[str, float | str]], output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "seed",
        "algorithm",
        "clusterer",
        "feature",
        "k",
        "evaluated_k",
        "precision",
        "recall",
    ]
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_topk(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    grouped: dict[tuple[str, str, str, str, float], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["dataset"]),
                str(row["algorithm"]),
                str(row["clusterer"]),
                str(row["feature"]),
                float(row["k"]),
            )
        ].append(row)

    summary: list[dict[str, float | str]] = []
    for (dataset, algorithm, clusterer, feature, k), group in sorted(grouped.items()):
        seeds = sorted({int(float(row["seed"])) for row in group})
        for metric in ["precision", "recall"]:
            values = [float(row[metric]) for row in group]
            summary.append(
                {
                    "dataset": dataset,
                    "algorithm": algorithm,
                    "clusterer": clusterer,
                    "feature": feature,
                    "k": k,
                    "metric": metric,
                    "seeds": " ".join(str(seed) for seed in seeds),
                    "n": float(len(values)),
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                }
            )
    return summary


def write_topk_summary(rows: list[dict[str, float | str]], output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "algorithm", "clusterer", "feature", "k", "metric", "seeds", "n", "mean", "std"]
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric_value(value: float | str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_results(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    metrics = [
        "runtime_sec",
        "modularity",
        "coverage",
        "mean_density",
        "mean_conductance",
        "auc",
        "average_precision",
        "ego_cluster_count",
        "ego_cluster_mean_size",
        "ego_cluster_mean_density",
        "ego_cluster_density_p10",
        "ego_cluster_density_p25",
        "ego_cluster_density_p50",
        "ego_cluster_density_p75",
        "ego_cluster_density_p90",
        "ego_cluster_mean_conductance",
        "ego_cluster_conductance_p10",
        "ego_cluster_conductance_p25",
        "ego_cluster_conductance_p50",
        "ego_cluster_conductance_p75",
        "ego_cluster_conductance_p90",
        "circle_precision",
        "circle_recall",
        "circle_f1",
    ]
    grouped: dict[tuple[str, str, str, str], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["algorithm"]), str(row["clusterer"]), str(row["feature"]))].append(row)

    summary: list[dict[str, float | str]] = []
    for (dataset, algorithm, clusterer, feature), group in sorted(grouped.items()):
        seeds = sorted({int(float(row["seed"])) for row in group})
        for metric in metrics:
            values = [value for row in group if (value := numeric_value(row.get(metric, ""))) is not None]
            if not values:
                continue
            summary.append(
                {
                    "dataset": dataset,
                    "algorithm": algorithm,
                    "clusterer": clusterer,
                    "feature": feature,
                    "metric": metric,
                    "seeds": " ".join(str(seed) for seed in seeds),
                    "n": float(len(values)),
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                }
            )
    return summary


def write_summary(rows: list[dict[str, float | str]], output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "algorithm", "clusterer", "feature", "metric", "seeds", "n", "mean", "std"]
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, float | str]]) -> None:
    def format_optional(value: float | str, width: int = 8) -> str:
        if value == "":
            return f"{'n/a':>{width}}"
        return f"{float(value):>{width}.4f}"

    display_rows: list[dict[str, float | str]] = []
    grouped: dict[tuple[str, str], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["algorithm"]))].append(row)
    for (_, _), group in sorted(grouped.items()):
        first = group[0]
        display_rows.append(
            {
                "dataset": first["dataset"],
                "algorithm": first["algorithm"],
                "feature": first["feature"],
                "clusterer": first["clusterer"],
                "runtime_sec": statistics.mean(float(row["runtime_sec"]) for row in group),
                "modularity": ""
                if first["modularity"] == ""
                else statistics.mean(float(row["modularity"]) for row in group),
                "ego_cluster_mean_density": statistics.mean(float(row["ego_cluster_mean_density"]) for row in group),
                "auc": statistics.mean(float(row["auc"]) for row in group),
                "average_precision": statistics.mean(float(row["average_precision"]) for row in group),
            }
        )

    print("\nExperiment summary (mean over seeds)")
    print("=" * 118)
    print(
        f"{'dataset':<12} {'algorithm':<18} {'time(s)':>9} {'mod':>8} "
        f"{'local_den':>10} {'AUC':>8} {'AP':>8}"
    )
    for row in display_rows:
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
        for row in display_rows
        if str(row["algorithm"]).startswith("EgoNet-") and row["feature"] == "W1"
    }
    louvain_times = [float(row["runtime_sec"]) for row in display_rows if row["algorithm"] == "Louvain"]
    print("=" * 118)
    print(f"mean EgoNet runtime:    {statistics.mean(ego_times.values()):.5f}s")
    print(f"mean Louvain runtime:   {statistics.mean(louvain_times):.5f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Project 2 topic 1 experiments.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--output", default="results/experiment_results.csv")
    parser.add_argument("--summary-output", default="results/experiment_summary.csv")
    parser.add_argument("--topk-output", default="results/topk_precision_recall.csv")
    parser.add_argument("--topk-summary-output", default="results/topk_precision_recall_summary.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--negative-seed-offset", type=int, default=100)
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
    parser.add_argument("--top-k", nargs="+", type=int, default=[10, 20, 50, 100, 200, 500, 1000])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else [args.seed]
    # Avoid charging one-time NetworkX Louvain setup to the first dataset.
    _ = louvain_partition(load_graph_csv(ROOT / "data" / f"{args.datasets[0]}.csv"), seed=seeds[0])
    rows: list[dict[str, float | str]] = []
    topk_rows: list[dict[str, float | str]] = []
    for dataset in args.datasets:
        for seed in seeds:
            seed_rows, seed_topk_rows = run_dataset_seed(dataset, seed, args)
            rows.extend(seed_rows)
            topk_rows.extend(seed_topk_rows)
    output = ROOT / args.output
    write_results(rows, output)
    summary = aggregate_results(rows)
    summary_output = ROOT / args.summary_output
    write_summary(summary, summary_output)
    topk_output = ROOT / args.topk_output
    write_topk(topk_rows, topk_output)
    topk_summary = aggregate_topk(topk_rows)
    topk_summary_output = ROOT / args.topk_summary_output
    write_topk_summary(topk_summary, topk_summary_output)
    print_summary(rows)
    print(f"\nWrote {output}")
    print(f"Wrote {summary_output}")
    print(f"Wrote {topk_output}")
    print(f"Wrote {topk_summary_output}")


if __name__ == "__main__":
    main()
