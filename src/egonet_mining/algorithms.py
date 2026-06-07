from __future__ import annotations

import itertools
import heapq
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping

import networkx as nx

from .metrics import mean_density


EgoEdges = dict[str, set[tuple[str, str]]]


def _sorted_pair(source: str, target: str) -> tuple[str, str]:
    return (source, target) if source <= target else (target, source)


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
    nodes = sorted(graph.nodes())
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


def triangle_ego_net_edges(graph: nx.Graph) -> EgoEdges:
    """Construct all ego-net-without-ego edge sets by Schank-style triangles.

    Every edge in Z_u corresponds to a triangle (u, v, w) in the original
    graph. The implementation follows the paper's Algorithm 2 idea: repeatedly
    process a current minimum-degree node in the residual graph, enumerate the
    triangles incident to it, and add each triangle edge to the three ego-nets.
    """
    residual_adj: dict[str, set[str]] = {
        node: set(graph.neighbors(node))
        for node in graph.nodes()
    }
    node_order = {node: idx for idx, node in enumerate(sorted(residual_adj))}
    heap = [
        (len(neighbors), node_order[node], node)
        for node, neighbors in residual_adj.items()
    ]
    heapq.heapify(heap)

    ego_edges: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    while heap:
        degree, _, ego = heapq.heappop(heap)
        neighbors = residual_adj.get(ego)
        if neighbors is None or degree != len(neighbors):
            continue

        ordered_neighbors = sorted(neighbors)
        for left_idx, source in enumerate(ordered_neighbors):
            source_adj = residual_adj.get(source)
            if source_adj is None:
                continue
            for target in ordered_neighbors[left_idx + 1 :]:
                if target not in source_adj:
                    continue
                ego_edges[ego].add(_sorted_pair(source, target))
                ego_edges[source].add(_sorted_pair(ego, target))
                ego_edges[target].add(_sorted_pair(ego, source))

        del residual_adj[ego]
        for neighbor in ordered_neighbors:
            neighbor_adj = residual_adj.get(neighbor)
            if neighbor_adj is None:
                continue
            neighbor_adj.discard(ego)
            heapq.heappush(heap, (len(neighbor_adj), node_order[neighbor], neighbor))

    return dict(ego_edges)


def ego_net_edges_for_node(graph: nx.Graph, ego: str) -> set[tuple[str, str]]:
    """Build one Z_ego edge set by intersecting the ego's neighborhood."""
    if ego not in graph:
        return set()
    neighbors = set(graph.neighbors(ego))
    edges: set[tuple[str, str]] = set()
    for source in sorted(neighbors):
        for target in graph.neighbors(source):
            if target in neighbors and source < target:
                edges.add((source, target))
    return edges


def ego_graph_from_edges(edges: set[tuple[str, str]]) -> nx.Graph:
    ego_graph = nx.Graph()
    ego_graph.add_edges_from(sorted(edges))
    return ego_graph


def ego_friendship_scores(
    graph: nx.Graph,
    *,
    alpha: float = 0.3,
    beta: float = 0.01,
    max_iter: int = 20,
    min_cluster_size: int = 2,
    seed: int = 42,
    construction: str = "triangle",
) -> tuple[dict[tuple[str, str], float], list[set[str]]]:
    """Compute W(v,w): number of ego-net communities containing v and w."""
    scores: dict[tuple[str, str], float] = defaultdict(float)
    all_ego_clusters: list[set[str]] = []
    sorted_nodes = sorted(graph.nodes())
    node_offsets = {node: offset for offset, node in enumerate(sorted_nodes)}

    if construction == "triangle":
        ego_edges_by_node = triangle_ego_net_edges(graph)
    elif construction == "neighbor":
        ego_edges_by_node = {
            ego: ego_net_edges_for_node(graph, ego)
            for ego in sorted_nodes
        }
    else:
        raise ValueError(f"Unsupported ego-net construction mode: {construction}")

    for ego in sorted_nodes:
        ego_edges = ego_edges_by_node.get(ego, set())
        if not ego_edges:
            continue
        ego_graph = ego_graph_from_edges(ego_edges)
        clusters = apm_label_propagation(
            ego_graph,
            alpha=alpha,
            beta=beta,
            max_iter=max_iter,
            seed=seed + node_offsets[ego],
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
