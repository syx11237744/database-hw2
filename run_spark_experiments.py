from __future__ import annotations

import argparse
import csv
import os
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
SPARK_DEPS = ROOT / ".pyspark_deps"
sys.path.insert(0, str(SRC))
if SPARK_DEPS.exists():
    sys.path.insert(0, str(SPARK_DEPS))
pythonpath_entries = [str(SRC)]
if SPARK_DEPS.exists():
    pythonpath_entries.append(str(SPARK_DEPS))
pythonpath_entries.append(os.environ.get("PYTHONPATH", ""))
os.environ["PYTHONPATH"] = os.pathsep.join(entry for entry in pythonpath_entries if entry)

from egonet_mining.algorithms import (
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
from egonet_mining.spark_algorithms import spark_ego_friendship_scores

DATASETS = ["facebook", "enron", "astro_ph"]


def timed(callable_obj, *args, **kwargs):
    start = time.perf_counter()
    result = callable_obj(*args, **kwargs)
    return result, time.perf_counter() - start


def create_spark(master: str, shuffle_partitions: int):
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise SystemExit(
            "PySpark is not installed. Install it with: "
            "python3 -m pip install --target .pyspark_deps pyspark"
        ) from exc

    spark = (
        SparkSession.builder
        .appName("database-hw2-egonet-mapreduce")
        .master(master)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.executorEnv.PYTHONPATH", os.environ["PYTHONPATH"])
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def run_dataset(name: str, sc, args: argparse.Namespace) -> list[dict[str, float | str]]:
    graph = load_graph_csv(ROOT / "data" / f"{name}.csv")
    train, positives = split_train_test_edges(
        graph,
        test_ratio=args.test_ratio,
        max_test_edges=args.max_test_edges,
        seed=args.seed,
    )
    negatives = sample_negative_edges(graph, len(positives), seed=args.seed + 100)
    eval_pairs = [tuple(edge) for edge in positives] + [tuple(edge) for edge in negatives]

    spark_result = spark_ego_friendship_scores(
        train,
        sc,
        rho=args.rho,
        alpha=args.alpha,
        beta=args.beta,
        max_iter=args.max_iter,
        min_cluster_size=args.min_cluster_size,
        features=tuple(args.features),
        candidate_pairs=eval_pairs,
        seed=args.seed,
        rdd_partitions=args.rdd_partitions,
    )

    louvain_comms, louvain_runtime = timed(louvain_partition, train, seed=args.seed)
    louvain_scores = louvain_pair_scores(train, louvain_comms, eval_pairs)
    louvain_quality = evaluate_partition(train, louvain_comms)
    louvain_lp = evaluate_link_prediction(louvain_scores, positives, negatives)

    base = {
        "dataset": name,
        "nodes": float(graph.number_of_nodes()),
        "edges": float(graph.number_of_edges()),
        "train_edges": float(train.number_of_edges()),
        "heldout_edges": float(len(positives)),
        "rho": float(args.rho),
    }
    zero_stats = {
        "spark_input_edges": 0.0,
        "spark_replicated_edges": 0.0,
        "spark_replication_factor": 0.0,
        "spark_partition_triples": 0.0,
        "spark_ego_edges": 0.0,
        "spark_ego_count": 0.0,
    }
    rows: list[dict[str, float | str]] = []
    for feature in args.features:
        ego_lp = evaluate_link_prediction(spark_result.scores[feature], positives, negatives)
        rows.append(
            {
                **base,
                "algorithm": f"Spark-EgoNet-APM-{feature}",
                "clusterer": "APM",
                "feature": feature,
                "runtime_sec": spark_result.stats["spark_runtime_sec"],
                "community_count": "",
                "mean_size": "",
                "modularity": "",
                "coverage": "",
                "mean_density": "",
                "mean_conductance": "",
                **ego_lp,
                "ego_cluster_count": spark_result.stats["ego_cluster_count"],
                "ego_cluster_mean_size": spark_result.stats["ego_cluster_mean_size"],
                "ego_cluster_mean_density": spark_result.stats["ego_cluster_mean_density"],
                "ego_cluster_mean_conductance": spark_result.stats["ego_cluster_mean_conductance"],
                **spark_result.stats,
            }
        )

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
            **zero_stats,
            "spark_runtime_sec": 0.0,
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
        "rho",
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
        "spark_input_edges",
        "spark_replicated_edges",
        "spark_replication_factor",
        "spark_partition_triples",
        "spark_ego_edges",
        "spark_ego_count",
        "spark_runtime_sec",
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

    print("\nSpark experiment summary")
    print("=" * 120)
    print(
        f"{'dataset':<12} {'algorithm':<21} {'rho':>4} {'time(s)':>9} "
        f"{'mod':>8} {'local_den':>10} {'AUC':>8} {'rep':>6}"
    )
    for row in rows:
        print(
            f"{row['dataset']:<12} {row['algorithm']:<21} "
            f"{int(float(row['rho'])):>4} "
            f"{float(row['runtime_sec']):>9.4f} "
            f"{format_optional(row['modularity'])} "
            f"{float(row['ego_cluster_mean_density']):>10.4f} "
            f"{float(row['auc']):>8.4f} "
            f"{float(row['spark_replication_factor']):>6.2f}"
        )

    spark_times = {
        row["dataset"]: float(row["runtime_sec"])
        for row in rows
        if str(row["algorithm"]).startswith("Spark-EgoNet-APM-") and row["feature"] == "W1"
    }
    louvain_times = [float(row["runtime_sec"]) for row in rows if row["algorithm"] == "Louvain"]
    if spark_times and louvain_times:
        print("=" * 120)
        print(f"mean Spark-EgoNet runtime: {statistics.mean(spark_times.values()):.4f}s")
        print(f"mean Louvain runtime:      {statistics.mean(louvain_times):.4f}s")
        print(f"mean runtime ratio:        {statistics.mean(spark_times.values()) / statistics.mean(louvain_times):.2f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local PySpark MapReduce-style ego-net experiments.")
    parser.add_argument("--datasets", nargs="+", default=["facebook"], choices=DATASETS)
    parser.add_argument("--output", default="results/spark_experiment_results.csv")
    parser.add_argument("--spark-master", default="local[*]")
    parser.add_argument("--rho", type=int, default=4)
    parser.add_argument("--rdd-partitions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--max-test-edges", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--features", nargs="+", choices=["W1", "W2", "W3", "W4"], default=["W1", "W2", "W3", "W4"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rho < 3:
        raise SystemExit("--rho must be at least 3")

    spark = create_spark(args.spark_master, args.rdd_partitions or args.rho)
    try:
        rows: list[dict[str, float | str]] = []
        for dataset in args.datasets:
            rows.extend(run_dataset(dataset, spark.sparkContext, args))
        output = ROOT / args.output
        write_results(rows, output)
        print_summary(rows)
        print(f"\nWrote {output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
