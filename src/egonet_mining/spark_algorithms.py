from __future__ import annotations

import hashlib
import itertools
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx

from .algorithms import (
    apm_label_propagation,
    ego_graph_from_edges,
    triangle_ego_net_edges,
)

if TYPE_CHECKING:
    from pyspark import SparkContext


Edge = tuple[str, str]
PartitionTriple = tuple[int, int, int]


@dataclass(frozen=True)
class SparkEgoNetResult:
    scores: dict[Edge, float]
    ego_clusters: list[set[str]]
    stats: dict[str, float]


def stable_partition(node: str, rho: int, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{node}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") % rho


def partition_triples_for_edge(edge: Edge, rho: int, seed: int) -> list[tuple[PartitionTriple, Edge]]:
    """Map one edge to Algorithm-3-style partition-triple keys."""
    if rho < 3:
        raise ValueError("rho must be at least 3 for partition triples")
    source, target = edge
    left = stable_partition(source, rho, seed)
    right = stable_partition(target, rho, seed)

    outputs: list[tuple[PartitionTriple, Edge]] = []
    if left == right:
        others = [part for part in range(rho) if part != left]
        for first, second in itertools.combinations(others, 2):
            outputs.append((tuple(sorted((left, first, second))), edge))
    else:
        for third in range(rho):
            if third not in {left, right}:
                outputs.append((tuple(sorted((left, right, third))), edge))
    return outputs


def _normalize_edge(source: str, target: str) -> Edge:
    return (source, target) if source <= target else (target, source)


def _construct_ego_edges_for_partition(item: tuple[PartitionTriple, Any]) -> list[tuple[str, Edge]]:
    _, edge_values = item
    graph = nx.Graph()
    graph.add_edges_from(edge_values)
    ego_edges = triangle_ego_net_edges(graph)
    return [
        (ego, edge)
        for ego, edges in ego_edges.items()
        for edge in edges
    ]


def _cluster_ego_edges(item: tuple[str, Any], params: dict[str, Any]) -> tuple[list[tuple[Edge, float]], list[set[str]]]:
    ego, edge_values = item
    edges = set(edge_values)
    if not edges:
        return [], []

    ego_graph = ego_graph_from_edges(edges)
    clusters = apm_label_propagation(
        ego_graph,
        alpha=params["alpha"],
        beta=params["beta"],
        max_iter=params["max_iter"],
        seed=params["seed"] + params["node_offsets"][ego],
    )

    edge_set: set[Edge] = params["edge_set"]
    min_cluster_size = params["min_cluster_size"]
    score_items: list[tuple[Edge, float]] = []
    kept_clusters: list[set[str]] = []
    for cluster in clusters:
        if len(cluster) < min_cluster_size:
            continue
        kept_clusters.append(cluster)
        for source, target in itertools.combinations(sorted(cluster), 2):
            pair = _normalize_edge(source, target)
            if pair not in edge_set:
                score_items.append((pair, 1.0))
    return score_items, kept_clusters


def spark_ego_friendship_scores(
    graph: nx.Graph,
    sc: SparkContext,
    *,
    rho: int = 4,
    alpha: float = 0.3,
    beta: float = 0.01,
    max_iter: int = 20,
    min_cluster_size: int = 2,
    seed: int = 42,
    rdd_partitions: int | None = None,
) -> SparkEgoNetResult:
    """Compute ego friendship scores through local PySpark MapReduce stages."""
    start = time.perf_counter()
    edges = [_normalize_edge(source, target) for source, target in graph.edges()]
    sorted_nodes = sorted(graph.nodes())
    node_offsets = {node: offset for offset, node in enumerate(sorted_nodes)}
    partitions = rdd_partitions or max(rho, 1)

    edge_rdd = sc.parallelize(edges, partitions).cache()
    input_edges = edge_rdd.count()

    partitioned_edges = edge_rdd.flatMap(
        lambda edge: partition_triples_for_edge(edge, rho, seed)
    ).cache()
    replicated_edges = partitioned_edges.count()
    partition_triples = partitioned_edges.keys().distinct().count()

    ego_edge_rdd = (
        partitioned_edges
        .groupByKey()
        .flatMap(_construct_ego_edges_for_partition)
        .distinct()
        .cache()
    )
    ego_edges = ego_edge_rdd.count()
    ego_count = ego_edge_rdd.keys().distinct().count()

    params = sc.broadcast({
        "alpha": alpha,
        "beta": beta,
        "max_iter": max_iter,
        "min_cluster_size": min_cluster_size,
        "seed": seed,
        "node_offsets": node_offsets,
        "edge_set": set(edges),
    })
    clustered = (
        ego_edge_rdd
        .groupByKey()
        .map(lambda item: _cluster_ego_edges(item, params.value))
        .cache()
    )

    score_items = clustered.flatMap(lambda result: result[0])
    scores = dict(score_items.reduceByKey(lambda left, right: left + right).collect())
    ego_clusters = clustered.flatMap(lambda result: result[1]).collect()

    return SparkEgoNetResult(
        scores=scores,
        ego_clusters=ego_clusters,
        stats={
            "spark_input_edges": float(input_edges),
            "spark_replicated_edges": float(replicated_edges),
            "spark_replication_factor": float(replicated_edges / input_edges) if input_edges else 0.0,
            "spark_partition_triples": float(partition_triples),
            "spark_ego_edges": float(ego_edges),
            "spark_ego_count": float(ego_count),
            "spark_runtime_sec": time.perf_counter() - start,
        },
    )
