from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import networkx as nx

from .metrics import coverage, mean_conductance, mean_density, modularity_safe


def split_train_test_edges(
    graph: nx.Graph,
    *,
    test_ratio: float = 0.15,
    max_test_edges: int | None = None,
    seed: int = 42,
) -> tuple[nx.Graph, list[tuple[str, str]]]:
    """Remove random non-tree edges for link-prediction evaluation."""
    rng = random.Random(seed)
    train = graph.copy()
    target = max(1, round(graph.number_of_edges() * test_ratio))
    if max_test_edges is not None:
        target = min(target, max_test_edges)

    tree_edges = {tuple(sorted(edge)) for edge in nx.minimum_spanning_edges(graph, data=False)}
    candidates = [
        tuple(sorted(edge))
        for edge in graph.edges()
        if tuple(sorted(edge)) not in tree_edges
    ]
    rng.shuffle(candidates)
    held_out = candidates[:target]
    train.remove_edges_from(held_out)
    return train, held_out


def sample_negative_edges(
    graph: nx.Graph,
    count: int,
    *,
    seed: int = 42,
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    negatives: set[tuple[str, str]] = set()
    max_attempts = max(1000, count * 100)
    attempts = 0
    while len(negatives) < count and attempts < max_attempts:
        source, target = rng.sample(nodes, 2)
        edge = tuple(sorted((source, target)))
        if not graph.has_edge(*edge):
            negatives.add(edge)
        attempts += 1
    if len(negatives) < count:
        for edge in nx.non_edges(graph):
            negatives.add(tuple(sorted(edge)))
            if len(negatives) >= count:
                break
    return list(negatives)


def roc_auc(labels: Sequence[int], values: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, values) if label == 1]
    negatives = [score for label, score in zip(labels, values) if label == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def average_precision(labels: Sequence[int], values: Sequence[float]) -> float:
    pairs = sorted(zip(values, labels), key=lambda pair: pair[0], reverse=True)
    positives = sum(labels)
    if positives == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    rank = 1
    while rank <= len(pairs):
        score = pairs[rank - 1][0]
        end = rank
        group_positive = 0
        while end <= len(pairs) and pairs[end - 1][0] == score:
            group_positive += pairs[end - 1][1]
            end += 1
        if group_positive:
            precision_sum += group_positive * (hits + group_positive) / (end - 1)
            hits += group_positive
        rank = end
    return precision_sum / positives


def evaluate_link_prediction(
    scores: Mapping[tuple[str, str], float],
    positive_edges: Sequence[tuple[str, str]],
    negative_edges: Sequence[tuple[str, str]],
) -> dict[str, float]:
    labels = [1] * len(positive_edges) + [0] * len(negative_edges)
    values = [
        scores.get(tuple(sorted(edge)), 0.0)
        for edge in list(positive_edges) + list(negative_edges)
    ]
    return {
        "auc": roc_auc(labels, values),
        "average_precision": average_precision(labels, values),
    }


def evaluate_partition(graph: nx.Graph, communities: Sequence[set[str]]) -> dict[str, float]:
    return {
        "community_count": float(len(communities)),
        "mean_size": sum(len(comm) for comm in communities) / len(communities)
        if communities
        else 0.0,
        "modularity": modularity_safe(graph, communities),
        "coverage": coverage(graph, communities),
        "mean_density": mean_density(graph, communities),
        "mean_conductance": mean_conductance(graph, communities),
    }
