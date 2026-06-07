from __future__ import annotations

import hashlib
import itertools
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx

from .algorithms import (
    EGO_FEATURES,
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
    scores: dict[str, dict[Edge, float]]
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


def _cluster_density(graph: nx.Graph, cluster: set[str]) -> float:
    if len(cluster) < 2:
        return 0.0
    possible = len(cluster) * (len(cluster) - 1) / 2
    return graph.subgraph(cluster).number_of_edges() / possible if possible else 0.0


def _cluster_conductance(graph: nx.Graph, cluster: set[str]) -> float:
    if not cluster or len(cluster) == graph.number_of_nodes():
        return 0.0
    cut_edges = nx.cut_size(graph, cluster)
    volume = sum(dict(graph.degree(cluster)).values())
    return cut_edges / volume if volume else 0.0


def _candidate_index(candidate_pairs: list[Edge]) -> dict[str, set[str]]:
    by_node: dict[str, set[str]] = defaultdict(set)
    for source, target in candidate_pairs:
        by_node[source].add(target)
        by_node[target].add(source)
    return by_node


def _cluster_candidate_pairs(
    nodes: list[str],
    cluster_set: set[str],
    candidate_by_node: dict[str, set[str]],
) -> list[Edge]:
    pairs: list[Edge] = []
    for source in nodes:
        for target in candidate_by_node.get(source, set()).intersection(cluster_set):
            if source < target:
                pairs.append((source, target))
    return pairs


def _cluster_ego_edges(item: tuple[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    ego, edge_values = item
    edges = set(edge_values)
    if not edges:
        return {
            "scores": {feature: [] for feature in params["features"]},
            "clusters": [],
            "densities": [],
            "conductances": [],
            "cluster_sizes": [],
        }

    ego_graph = ego_graph_from_edges(edges)
    clusters = apm_label_propagation(
        ego_graph,
        alpha=params["alpha"],
        beta=params["beta"],
        max_iter=params["max_iter"],
        seed=params["seed"] + params["node_offsets"][ego],
    )

    min_cluster_size = params["min_cluster_size"]
    candidate_by_node: dict[str, set[str]] = params["candidate_by_node"]
    features: tuple[str, ...] = tuple(params["features"])
    score_items: dict[str, list[tuple[Edge, float]]] = {feature: [] for feature in features}
    kept_clusters: list[set[str]] = []
    densities: list[float] = []
    conductances: list[float] = []
    cluster_sizes: list[int] = []

    for cluster in clusters:
        if len(cluster) < min_cluster_size:
            continue
        cluster_set = set(cluster)
        nodes = sorted(cluster_set)
        pairs = _cluster_candidate_pairs(nodes, cluster_set, candidate_by_node)
        density = _cluster_density(ego_graph, cluster_set)
        conductance = _cluster_conductance(ego_graph, cluster_set)
        kept_clusters.append(cluster_set)
        densities.append(density)
        conductances.append(conductance)
        cluster_sizes.append(len(cluster_set))
        if not pairs:
            continue

        neighbors_in_cluster = {
            node: set(ego_graph.neighbors(node)).intersection(cluster_set)
            for node in nodes
        }
        for source, target in pairs:
            if "W1" in score_items:
                score_items["W1"].append(((source, target), 1.0))
            if "W2" in score_items:
                score_items["W2"].append(((source, target), density))
            if "W3" in score_items:
                score_items["W3"].append(
                    ((source, target), float(len(neighbors_in_cluster[source].intersection(neighbors_in_cluster[target]))))
                )
            if "W4" in score_items:
                score_items["W4"].append(
                    (
                        (source, target),
                        min(len(neighbors_in_cluster[source]), len(neighbors_in_cluster[target])) / len(cluster_set),
                    )
                )

    return {
        "scores": score_items,
        "clusters": kept_clusters,
        "densities": densities,
        "conductances": conductances,
        "cluster_sizes": cluster_sizes,
    }


def spark_ego_friendship_scores(
    graph: nx.Graph,
    sc: SparkContext,
    *,
    rho: int = 4,
    alpha: float = 0.3,
    beta: float = 0.01,
    max_iter: int = 20,
    min_cluster_size: int = 2,
    features: tuple[str, ...] = EGO_FEATURES,
    candidate_pairs: list[Edge] | None = None,
    seed: int = 42,
    rdd_partitions: int | None = None,
) -> SparkEgoNetResult:
    """Compute ego friendship scores through local PySpark MapReduce stages."""
    unsupported = set(features) - set(EGO_FEATURES)
    if unsupported:
        raise ValueError(f"Unsupported ego-net features: {sorted(unsupported)}")

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
        "features": features,
        "candidate_by_node": _candidate_index(
            [_normalize_edge(source, target) for source, target in candidate_pairs or []]
        ),
    })
    clustered = (
        ego_edge_rdd
        .groupByKey()
        .map(lambda item: _cluster_ego_edges(item, params.value))
        .cache()
    )

    scores = {
        feature: dict(
            clustered
            .flatMap(lambda result, feature=feature: result["scores"][feature])
            .reduceByKey(lambda left, right: left + right)
            .collect()
        )
        for feature in features
    }
    ego_clusters = clustered.flatMap(lambda result: result["clusters"]).collect()
    cluster_sizes = clustered.flatMap(lambda result: result["cluster_sizes"]).collect()
    densities = clustered.flatMap(lambda result: result["densities"]).collect()
    conductances = clustered.flatMap(lambda result: result["conductances"]).collect()

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
            "ego_cluster_count": float(len(cluster_sizes)),
            "ego_cluster_mean_size": float(sum(cluster_sizes) / len(cluster_sizes)) if cluster_sizes else 0.0,
            "ego_cluster_mean_density": float(sum(densities) / len(densities)) if densities else 0.0,
            "ego_cluster_mean_conductance": float(sum(conductances) / len(conductances)) if conductances else 0.0,
        },
    )
