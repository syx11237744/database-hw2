from __future__ import annotations

import itertools
import heapq
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from collections.abc import Mapping

import networkx as nx

from .metrics import mean_density


EgoEdges = dict[str, set[tuple[str, str]]]
EgoFeatureScores = dict[str, dict[tuple[str, str], float]]
EGO_FEATURES = ("W1", "W2", "W3", "W4")


@dataclass(frozen=True)
class EgoFeatureResult:
    """Paper-aligned ego-net friend-suggestion features."""

    scores: EgoFeatureScores
    clusters: list[set[str]]
    stats: dict[str, float]


@dataclass(frozen=True)
class LocalCSRGraph:
    """Small undirected ego-net graph stored as CSR arrays."""

    nodes: list[str]
    node_to_idx: dict[str, int]
    indptr: list[int]
    indices: list[int]
    degrees: list[int]
    edge_count: int

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def neighbors(self, idx: int) -> list[int]:
        return self.indices[self.indptr[idx] : self.indptr[idx + 1]]


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


def csr_graph_from_edges(edges: set[tuple[str, str]]) -> LocalCSRGraph:
    nodes = sorted({node for edge in edges for node in edge})
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    adj: list[list[int]] = [[] for _ in nodes]
    for source, target in edges:
        source_idx = node_to_idx[source]
        target_idx = node_to_idx[target]
        adj[source_idx].append(target_idx)
        adj[target_idx].append(source_idx)

    indptr = [0]
    indices: list[int] = []
    degrees: list[int] = []
    for neighbors in adj:
        neighbors.sort()
        degrees.append(len(neighbors))
        indices.extend(neighbors)
        indptr.append(len(indices))
    return LocalCSRGraph(
        nodes=nodes,
        node_to_idx=node_to_idx,
        indptr=indptr,
        indices=indices,
        degrees=degrees,
        edge_count=len(edges),
    )


def ego_connected_components(graph: nx.Graph) -> list[set[str]]:
    """Connected-components baseline inside one ego-net Z_u."""
    return [set(component) for component in nx.connected_components(graph)]


def csr_connected_components(graph: LocalCSRGraph) -> list[list[int]]:
    visited = [False] * graph.node_count
    components: list[list[int]] = []
    for start in range(graph.node_count):
        if visited[start]:
            continue
        visited[start] = True
        component: list[int] = []
        stack = [start]
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in graph.neighbors(node):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        component.sort()
        components.append(component)
    return components


def csr_apm_label_propagation(
    graph: LocalCSRGraph,
    *,
    alpha: float = 0.3,
    beta: float = 0.01,
    max_iter: int = 20,
    seed: int = 42,
) -> list[list[int]]:
    if graph.node_count == 0:
        return []
    rng = random.Random(seed)
    labels = list(range(graph.node_count))
    label_sizes = [1] * graph.node_count
    nodes = list(range(graph.node_count))

    for _ in range(max_iter):
        changed = 0
        order = nodes[:]
        rng.shuffle(order)
        for node in order:
            neighbor_labels: dict[int, int] = {}
            for neighbor in graph.neighbors(node):
                label = labels[neighbor]
                neighbor_labels[label] = neighbor_labels.get(label, 0) + 1
            if not neighbor_labels:
                continue
            best_score = -math.inf
            best_labels: list[int] = []
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
        if changed / graph.node_count < beta:
            break

    communities: dict[int, list[int]] = defaultdict(list)
    for node, label in enumerate(labels):
        communities[label].append(node)
    return [nodes for nodes in communities.values()]


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


def _csr_cluster_profile(
    graph: LocalCSRGraph,
    cluster: list[int],
) -> tuple[float, float, dict[int, set[int]]]:
    cluster_set = set(cluster)
    possible = len(cluster) * (len(cluster) - 1) / 2
    internal_twice = 0
    volume = 0
    neighbors_in_cluster: dict[int, set[int]] = {}
    for node in cluster:
        local_neighbors = {neighbor for neighbor in graph.neighbors(node) if neighbor in cluster_set}
        neighbors_in_cluster[node] = local_neighbors
        internal_twice += len(local_neighbors)
        volume += graph.degrees[node]
    internal_edges = internal_twice / 2
    density = internal_edges / possible if possible else 0.0
    if not cluster or len(cluster) == graph.node_count:
        conductance = 0.0
    else:
        cut_edges = volume - internal_twice
        conductance = cut_edges / volume if volume else 0.0
    return density, conductance, neighbors_in_cluster


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _quantile_stats(prefix: str, values: list[float]) -> dict[str, float]:
    return {
        f"{prefix}_p10": _quantile(values, 0.10),
        f"{prefix}_p25": _quantile(values, 0.25),
        f"{prefix}_p50": _quantile(values, 0.50),
        f"{prefix}_p75": _quantile(values, 0.75),
        f"{prefix}_p90": _quantile(values, 0.90),
    }


def _candidate_index(
    candidate_pairs: list[tuple[str, str]] | None,
) -> tuple[set[tuple[str, str]] | None, dict[str, set[str]]]:
    if candidate_pairs is None:
        return None, {}
    normalized = {_sorted_pair(source, target) for source, target in candidate_pairs}
    by_node: dict[str, set[str]] = defaultdict(set)
    for source, target in normalized:
        by_node[source].add(target)
        by_node[target].add(source)
    return normalized, by_node


def _cluster_candidate_pairs(
    nodes: list[str],
    cluster_set: set[str],
    candidate_by_node: dict[str, set[str]],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for source in nodes:
        for target in candidate_by_node.get(source, set()).intersection(cluster_set):
            if source < target:
                pairs.append((source, target))
    return pairs


def ego_feature_scores(
    graph: nx.Graph,
    *,
    clusterer: str = "apm",
    features: tuple[str, ...] = EGO_FEATURES,
    candidate_pairs: list[tuple[str, str]] | None = None,
    alpha: float = 0.3,
    beta: float = 0.01,
    max_iter: int = 20,
    min_cluster_size: int = 2,
    seed: int = 42,
    construction: str = "triangle",
) -> EgoFeatureResult:
    """Compute paper W1-W4 ego-network friendship features.

    W1 counts co-clustering in ego-net communities. W2 weights W1 by local
    cluster density. W3 counts common neighbors inside the cluster. W4 weights
    by the smaller endpoint degree inside the cluster divided by cluster size.
    """
    unsupported = set(features) - set(EGO_FEATURES)
    if unsupported:
        raise ValueError(f"Unsupported ego-net features: {sorted(unsupported)}")

    sorted_nodes = sorted(graph.nodes())
    node_offsets = {node: offset for offset, node in enumerate(sorted_nodes)}
    graph_edges = {_sorted_pair(source, target) for source, target in graph.edges()}
    candidate_set, candidate_by_node = _candidate_index(candidate_pairs)

    if construction == "triangle":
        ego_edges_by_node = triangle_ego_net_edges(graph)
    elif construction == "neighbor":
        ego_edges_by_node = {
            ego: ego_net_edges_for_node(graph, ego)
            for ego in sorted_nodes
        }
    else:
        raise ValueError(f"Unsupported ego-net construction mode: {construction}")

    score_tables: EgoFeatureScores = {feature: defaultdict(float) for feature in features}
    all_ego_clusters: list[set[str]] = []
    densities: list[float] = []
    conductances: list[float] = []

    for ego in sorted_nodes:
        ego_edges = ego_edges_by_node.get(ego, set())
        if not ego_edges:
            continue
        ego_graph = csr_graph_from_edges(ego_edges)
        if clusterer == "apm":
            clusters = csr_apm_label_propagation(
                ego_graph,
                alpha=alpha,
                beta=beta,
                max_iter=max_iter,
                seed=seed + node_offsets[ego],
            )
        elif clusterer == "cc":
            clusters = csr_connected_components(ego_graph)
        else:
            raise ValueError(f"Unsupported ego-net clusterer: {clusterer}")

        for cluster in clusters:
            if len(cluster) < min_cluster_size:
                continue
            nodes = [ego_graph.nodes[idx] for idx in cluster]
            cluster_set = set(nodes)
            density, conductance, neighbors_in_cluster = _csr_cluster_profile(ego_graph, cluster)
            all_ego_clusters.append(cluster_set)
            densities.append(density)
            conductances.append(conductance)

            if candidate_set is None:
                pairs = [
                    (ego_graph.node_to_idx[source], ego_graph.node_to_idx[target], source, target)
                    for source, target in itertools.combinations(nodes, 2)
                    if (source, target) not in graph_edges
                ]
            else:
                pairs = [
                    (ego_graph.node_to_idx[source], ego_graph.node_to_idx[target], source, target)
                    for source, target in _cluster_candidate_pairs(nodes, cluster_set, candidate_by_node)
                ]
            if not pairs:
                continue

            for source_idx, target_idx, source, target in pairs:
                if "W1" in score_tables:
                    score_tables["W1"][(source, target)] += 1.0
                if "W2" in score_tables:
                    score_tables["W2"][(source, target)] += density
                if "W3" in score_tables:
                    score_tables["W3"][(source, target)] += len(
                        neighbors_in_cluster[source_idx].intersection(neighbors_in_cluster[target_idx])
                    )
                if "W4" in score_tables:
                    score_tables["W4"][(source, target)] += (
                        min(len(neighbors_in_cluster[source_idx]), len(neighbors_in_cluster[target_idx]))
                        / len(cluster_set)
                    )

    cluster_sizes = [len(cluster) for cluster in all_ego_clusters]
    stats = {
        "ego_cluster_count": float(len(all_ego_clusters)),
        "ego_cluster_mean_size": sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0.0,
        "ego_cluster_mean_density": sum(densities) / len(densities) if densities else 0.0,
        "ego_cluster_mean_conductance": sum(conductances) / len(conductances) if conductances else 0.0,
    }
    stats.update(_quantile_stats("ego_cluster_density", densities))
    stats.update(_quantile_stats("ego_cluster_conductance", conductances))
    return EgoFeatureResult(
        scores={feature: dict(scores) for feature, scores in score_tables.items()},
        clusters=all_ego_clusters,
        stats=stats,
    )


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
    """Backward-compatible wrapper for the paper's W1 feature using APM."""
    result = ego_feature_scores(
        graph,
        clusterer="apm",
        features=("W1",),
        alpha=alpha,
        beta=beta,
        max_iter=max_iter,
        min_cluster_size=min_cluster_size,
        seed=seed,
        construction=construction,
    )
    return result.scores["W1"], result.clusters


def ego_score_partition(
    graph: nx.Graph,
    scores: Mapping[tuple[str, str], float],
    *,
    min_score: float = 1.0,
) -> list[set[str]]:
    """Legacy global score-graph component heuristic.

    This is not the connected-components baseline from Epasto et al.; the paper
    uses connected components inside each ego-net Z_u before computing friend
    suggestion scores. New experiments should use ``ego_feature_scores`` with
    ``clusterer="cc"`` for that paper-aligned baseline.
    """
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
