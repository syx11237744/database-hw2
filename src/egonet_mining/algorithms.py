from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping

import networkx as nx

from .metrics import mean_density


def apm_label_propagation(
    graph: nx.Graph,
    *,
    alpha: float = 0.3,
    beta: float = 0.01,
    max_iter: int = 20,
    seed: int = 42,
) -> list[set[str]]:
    """Absolute-Potts-style label propagation from Epasto et al. appendix.

    The update score is C_u(l) - alpha * (T(l) - C_u(l)), where C_u(l) is
    the number of neighbors with label l and T(l) is the current label size.
    """
    if graph.number_of_nodes() == 0:
        return []
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    labels = {node: node for node in nodes}
    label_sizes = Counter(labels.values())

    for _ in range(max_iter):
        changed = 0
        order = nodes[:]
        rng.shuffle(order)
        for node in order:
            neighbor_labels = Counter(labels[neighbor] for neighbor in graph.neighbors(node))
            if not neighbor_labels:
                continue
            best_score = -math.inf
            best_labels: list[str] = []
            for label, count in neighbor_labels.items():
                score = count - alpha * (label_sizes[label] - count)
                if score > best_score:
                    best_score = score
                    best_labels = [label]
                elif score == best_score:
                    best_labels.append(label)
            new_label = rng.choice(best_labels)
            old_label = labels[node]
            if new_label != old_label:
                labels[node] = new_label
                label_sizes[old_label] -= 1
                label_sizes[new_label] += 1
                changed += 1
        if changed / len(nodes) < beta:
            break

    communities: dict[str, set[str]] = defaultdict(set)
    for node, label in labels.items():
        communities[label].add(node)
    return list(communities.values())


def ego_friendship_scores(
    graph: nx.Graph,
    *,
    alpha: float = 0.3,
    beta: float = 0.01,
    max_iter: int = 20,
    min_cluster_size: int = 2,
    seed: int = 42,
) -> tuple[dict[tuple[str, str], float], list[set[str]]]:
    """Compute W(v,w): number of ego-net communities containing v and w."""
    scores: dict[tuple[str, str], float] = defaultdict(float)
    all_ego_clusters: list[set[str]] = []

    for offset, ego in enumerate(sorted(graph.nodes())):
        neighbors = sorted(graph.neighbors(ego))
        if len(neighbors) < min_cluster_size:
            continue
        ego_graph = graph.subgraph(neighbors).copy()
        if ego_graph.number_of_edges() == 0:
            continue
        clusters = apm_label_propagation(
            ego_graph,
            alpha=alpha,
            beta=beta,
            max_iter=max_iter,
            seed=seed + offset,
        )
        for cluster in clusters:
            if len(cluster) < min_cluster_size:
                continue
            all_ego_clusters.append(cluster)
            for source, target in itertools.combinations(sorted(cluster), 2):
                if not graph.has_edge(source, target):
                    scores[(source, target)] += 1.0
    return dict(scores), all_ego_clusters


def ego_score_partition(
    graph: nx.Graph,
    scores: Mapping[tuple[str, str], float],
    *,
    min_score: float = 1.0,
) -> list[set[str]]:
    """Aggregate ego-net pair scores into communities via score graph components."""
    score_graph = nx.Graph()
    score_graph.add_nodes_from(graph.nodes())
    for (source, target), score in scores.items():
        if score >= min_score:
            score_graph.add_edge(source, target, weight=score)
    return [set(component) for component in nx.connected_components(score_graph)]


def louvain_partition(graph: nx.Graph, *, seed: int = 42) -> list[set[str]]:
    if graph.number_of_edges() == 0:
        return [{node} for node in graph.nodes()]
    return [
        set(community)
        for community in nx.algorithms.community.louvain_communities(
            graph,
            seed=seed,
        )
    ]


def louvain_pair_scores(
    graph: nx.Graph,
    communities: list[set[str]],
    pairs: list[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], float]:
    """Score non-edges by shared Louvain community plus common-neighbor support."""
    node_to_comm: dict[str, int] = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_comm[node] = idx

    scores: dict[tuple[str, str], float] = {}
    degrees = dict(graph.degree())
    candidate_pairs = pairs if pairs is not None else list(itertools.combinations(sorted(graph.nodes()), 2))
    for source, target in candidate_pairs:
        same = 1.0 if node_to_comm.get(source) == node_to_comm.get(target) else 0.0
        common = len(set(graph.neighbors(source)).intersection(graph.neighbors(target)))
        denom = math.sqrt(max(degrees[source], 1) * max(degrees[target], 1))
        scores[tuple(sorted((source, target)))] = same + common / denom
    return scores


def describe_ego_clusters(graph: nx.Graph, clusters: list[set[str]]) -> dict[str, float]:
    sizes = [len(cluster) for cluster in clusters]
    return {
        "ego_cluster_count": float(len(sizes)),
        "ego_cluster_mean_size": sum(sizes) / len(sizes) if sizes else 0.0,
        "ego_cluster_mean_density": mean_density(graph, clusters),
    }
