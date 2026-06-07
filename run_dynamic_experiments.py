from __future__ import annotations

import argparse
import csv
import pathlib
import random
import statistics
import sys
import time

import networkx as nx

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from egonet_mining.algorithms import louvain_partition
from egonet_mining.data import load_graph_csv
from egonet_mining.dynamic import DynamicEgoNetMiner, EdgeEvent
from egonet_mining.evaluation import evaluate_partition

DATASETS = ["facebook", "enron", "astro_ph"]


def timed(callable_obj, *args, **kwargs):
    start = time.perf_counter()
    result = callable_obj(*args, **kwargs)
    return result, time.perf_counter() - start


def make_stream(
    graph: nx.Graph,
    *,
    batches: int,
    batch_size: int,
    seed: int,
) -> tuple[nx.Graph, list[list[EdgeEvent]]]:
    rng = random.Random(seed)
    total = batches * batch_size
    edges = [tuple(sorted(edge)) for edge in graph.edges()]
    rng.shuffle(edges)

    future_additions = edges[:total]
    initial = graph.copy()
    initial.remove_edges_from(future_additions)

    deletion_pool = [tuple(sorted(edge)) for edge in initial.edges()]
    rng.shuffle(deletion_pool)
    future_deletions = deletion_pool[:total]

    stream: list[list[EdgeEvent]] = []
    for batch in range(batches):
        additions = future_additions[batch * batch_size : (batch + 1) * batch_size]
        deletions = future_deletions[batch * batch_size : (batch + 1) * batch_size]
        events: list[EdgeEvent] = []
        events.extend(("add", source, target) for source, target in additions)
        events.extend(("remove", source, target) for source, target in deletions)
        rng.shuffle(events)
        stream.append(events)
    return initial, stream


def full_egonet_partition(graph: nx.Graph, args: argparse.Namespace) -> list[set[str]]:
    miner = DynamicEgoNetMiner(
        graph,
        alpha=args.alpha,
        beta=args.beta,
        max_iter=args.max_iter,
        min_score=args.min_score,
        seed=args.seed,
        clusterer=args.clusterer,
        reuse_labels=args.reuse_labels,
    )
    miner.initialize()
    return miner.partition()


def run_dataset(name: str, args: argparse.Namespace) -> list[dict[str, float | str]]:
    graph = load_graph_csv(ROOT / "data" / f"{name}.csv")
    initial, stream = make_stream(
        graph,
        batches=args.batches,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    miner = DynamicEgoNetMiner(
        initial,
        alpha=args.alpha,
        beta=args.beta,
        max_iter=args.max_iter,
        min_score=args.min_score,
        seed=args.seed,
        clusterer=args.clusterer,
        reuse_labels=args.reuse_labels,
    )
    init_time = miner.initialize()
    print(f"{name}: initialized dynamic miner in {init_time:.3f}s")

    rows: list[dict[str, float | str]] = []
    for batch_id, events in enumerate(stream, start=1):
        update_stats = miner.apply_events(events)
        dynamic_partition = miner.partition()
        dynamic_quality = evaluate_partition(miner.graph, dynamic_partition)

        full_partition, full_runtime = timed(full_egonet_partition, miner.graph, args)
        full_quality = evaluate_partition(miner.graph, full_partition)

        louvain_comms, louvain_runtime = timed(louvain_partition, miner.graph, seed=args.seed)
        louvain_quality = evaluate_partition(miner.graph, louvain_comms)

        speedup = full_runtime / update_stats["runtime_sec"] if update_stats["runtime_sec"] else 0.0
        row = {
            "dataset": name,
            "batch": float(batch_id),
            "nodes": float(miner.graph.number_of_nodes()),
            "edges": float(miner.graph.number_of_edges()),
            "events": float(len(events)),
            "affected_egos": update_stats["affected_egos"],
            "ego_edge_updates": update_stats["ego_edge_updates"],
            "score_delta_pairs": update_stats["score_delta_pairs"],
            "init_runtime_sec": init_time if batch_id == 1 else 0.0,
            "dynamic_update_sec": update_stats["runtime_sec"],
            "full_egonet_sec": full_runtime,
            "louvain_sec": louvain_runtime,
            "speedup_vs_full": speedup,
            "dynamic_modularity": dynamic_quality["modularity"],
            "full_modularity": full_quality["modularity"],
            "louvain_modularity": louvain_quality["modularity"],
            "dynamic_density": dynamic_quality["mean_density"],
            "full_density": full_quality["mean_density"],
            "louvain_density": louvain_quality["mean_density"],
            "dynamic_conductance": dynamic_quality["mean_conductance"],
            "full_conductance": full_quality["mean_conductance"],
            "louvain_conductance": louvain_quality["mean_conductance"],
            "score_edges": float(len(miner.scores)),
        }
        rows.append(row)
        print(
            f"{name} batch {batch_id}: dynamic={update_stats['runtime_sec']:.3f}s, "
            f"full={full_runtime:.3f}s, speedup={speedup:.2f}x, "
            f"dyn_mod={dynamic_quality['modularity']:.4f}"
        )
    return rows


def write_results(rows: list[dict[str, float | str]], output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "batch",
        "nodes",
        "edges",
        "events",
        "affected_egos",
        "ego_edge_updates",
        "score_delta_pairs",
        "init_runtime_sec",
        "dynamic_update_sec",
        "full_egonet_sec",
        "louvain_sec",
        "speedup_vs_full",
        "dynamic_modularity",
        "full_modularity",
        "louvain_modularity",
        "dynamic_density",
        "full_density",
        "louvain_density",
        "dynamic_conductance",
        "full_conductance",
        "louvain_conductance",
        "score_edges",
    ]
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, float | str]]) -> None:
    print("\nDynamic experiment summary")
    print("=" * 126)
    print(
        f"{'dataset':<10} {'batch':>5} {'events':>7} {'affected':>9} "
        f"{'ego_edges':>9} {'score_delta':>11} {'dyn(s)':>9} {'full(s)':>9} "
        f"{'speedup':>8} {'dyn_mod':>9} {'full_mod':>9}"
    )
    for row in rows:
        print(
            f"{row['dataset']:<10} {int(float(row['batch'])):>5} "
            f"{int(float(row['events'])):>7} {int(float(row['affected_egos'])):>9} "
            f"{int(float(row['ego_edge_updates'])):>9} "
            f"{int(float(row['score_delta_pairs'])):>11} "
            f"{float(row['dynamic_update_sec']):>9.4f} "
            f"{float(row['full_egonet_sec']):>9.4f} "
            f"{float(row['speedup_vs_full']):>8.2f} "
            f"{float(row['dynamic_modularity']):>9.4f} "
            f"{float(row['full_modularity']):>9.4f}"
        )
    print("=" * 126)
    print(f"mean dynamic update: {statistics.mean(float(r['dynamic_update_sec']) for r in rows):.4f}s")
    print(f"mean full recompute: {statistics.mean(float(r['full_egonet_sec']) for r in rows):.4f}s")
    print(f"mean speedup:        {statistics.mean(float(r['speedup_vs_full']) for r in rows):.2f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optional dynamic graph experiments.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--output", default="results/dynamic_experiment_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=1.0)
    parser.add_argument("--clusterer", choices=["apm", "cc"], default="apm")
    parser.add_argument("--reuse-labels", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, float | str]] = []
    for dataset in args.datasets:
        rows.extend(run_dataset(dataset, args))
    output = ROOT / args.output
    write_results(rows, output)
    print_summary(rows)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
