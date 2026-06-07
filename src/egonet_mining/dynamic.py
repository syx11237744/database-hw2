from __future__ import annotations

import hashlib
import math
import random
import time
from collections import Counter
from collections.abc import Iterable

import networkx as nx

from .algorithms import (
    LocalCSRGraph,
    csr_connected_components,
    csr_graph_from_edges,
    ego_score_partition,
    triangle_ego_net_edges,
)

EdgeEvent = tuple[str, str, str]
Edge = tuple[str, str]


def stable_seed(value: str, base_seed: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=4).digest()
    return base_seed + int.from_bytes(digest, "big")


def sorted_edge(source: str, target: str) -> Edge:
    return (source, target) if source <= target else (target, source)


def build_adjacency(graph: nx.Graph) -> dict[str, set[str]]:
    adjacency = {node: set() for node in graph.nodes()}
    for source, target in graph.edges():
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    return adjacency


def csr_apm_label_propagation_with_labels(
    graph: LocalCSRGraph,
    *,
    initial_labels: dict[str, str] | None = None,
    alpha: float = 0.3,
    beta: float = 0.01,
    max_iter: int = 20,
    seed: int = 42,
) -> tuple[list[list[int]], dict[str, str]]:
    """Run CSR APM with optional warm-start labels.

    When ``initial_labels`` is not supplied, this is equivalent to the static
    CSR APM path: every node starts with its own label. When labels are
    supplied, nodes that carried the same previous label start together while
    newly appearing nodes start as singletons.
    """
    if graph.node_count == 0:
        return [], {}

    label_to_idx: dict[tuple[str, str], int] = {}
    label_sizes: list[int] = []
    labels: list[int] = []
    for node in graph.nodes:
        if initial_labels is not None and node in initial_labels:
            label_key = ("old", initial_labels[node])
        else:
            label_key = ("new", node)
        label_idx = label_to_idx.get(label_key)
        if label_idx is None:
            label_idx = len(label_to_idx)
            label_to_idx[label_key] = label_idx
            label_sizes.append(0)
        labels.append(label_idx)
        label_sizes[label_idx] += 1

    rng = random.Random(seed)
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

    communities_by_label: list[list[int]] = [[] for _ in label_sizes]
    seen_labels: list[int] = []
    for node, label in enumerate(labels):
        if not communities_by_label[label]:
            seen_labels.append(label)
        communities_by_label[label].append(node)

    communities = [communities_by_label[label] for label in seen_labels]
    label_snapshot: dict[str, str] = {}
    for community in communities:
        representative = min(graph.nodes[node] for node in community)
        for node in community:
            label_snapshot[graph.nodes[node]] = representative
    return communities, label_snapshot


class DynamicEgoNetMiner:
    """Incrementally maintain ego-net pair scores under edge updates."""

    def __init__(
        self,
        graph: nx.Graph,
        *,
        alpha: float = 0.3,
        beta: float = 0.01,
        max_iter: int = 20,
        min_cluster_size: int = 2,
        min_score: float = 1.0,
        seed: int = 42,
        clusterer: str = "apm",
        reuse_labels: bool = True,
    ) -> None:
        if clusterer not in {"apm", "cc"}:
            raise ValueError(f"Unsupported dynamic ego-net clusterer: {clusterer}")
        self.graph = graph.copy()
        self.adjacency = build_adjacency(self.graph)
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.min_cluster_size = min_cluster_size
        self.min_score = min_score
        self.seed = seed
        self.clusterer = clusterer
        self.reuse_labels = reuse_labels
        self.scores: Counter[tuple[str, str]] = Counter()
        self.ego_contributions: dict[str, Counter[tuple[str, str]]] = {}
        self.ego_clusters: dict[str, list[set[str]]] = {}
        self.ego_labels: dict[str, dict[str, str]] = {}
        self.ego_edges_by_node: dict[str, set[Edge]] = {}

    def initialize(self) -> float:
        start = time.perf_counter()
        self.adjacency = build_adjacency(self.graph)
        self.ego_edges_by_node = {
            ego: set(edges)
            for ego, edges in triangle_ego_net_edges(self.graph).items()
        }
        for node in self.graph.nodes():
            self.ego_edges_by_node.setdefault(node, set())
        self.scores.clear()
        self.ego_contributions.clear()
        self.ego_clusters.clear()
        self.ego_labels.clear()
        for ego in sorted(self.graph.nodes()):
            self._replace_ego(ego)
        return time.perf_counter() - start

    def apply_events(self, events: Iterable[EdgeEvent]) -> dict[str, float]:
        events = list(events)
        start = time.perf_counter()
        affected: set[str] = set()
        ego_edge_updates = 0
        for action, source, target in events:
            if source == target:
                continue
            if action == "add":
                event_affected, edge_updates = self._add_graph_edge(source, target)
                affected.update(event_affected)
                ego_edge_updates += edge_updates
            elif action == "remove":
                event_affected, edge_updates = self._remove_graph_edge(source, target)
                affected.update(event_affected)
                ego_edge_updates += edge_updates
            else:
                raise ValueError(f"Unsupported edge event action: {action}")

        score_delta_pairs = 0
        for ego in sorted(affected):
            score_delta_pairs += self._replace_ego(ego)

        return {
            "affected_egos": float(len(affected)),
            "ego_edge_updates": float(ego_edge_updates),
            "score_delta_pairs": float(score_delta_pairs),
            "runtime_sec": time.perf_counter() - start,
        }

    def partition(self) -> list[set[str]]:
        return ego_score_partition(self.graph, self.scores, min_score=self.min_score)

    def _add_graph_edge(self, source: str, target: str) -> tuple[set[str], int]:
        if self.graph.has_edge(source, target):
            return set(), 0
        source_neighbors = self.adjacency.setdefault(source, set())
        target_neighbors = self.adjacency.setdefault(target, set())
        common = set(source_neighbors).intersection(target_neighbors)
        affected = {source, target, *common}
        edge_updates = self._add_triangle_ego_edges(source, target, common)
        self.graph.add_edge(source, target)
        self.ego_edges_by_node.setdefault(source, set())
        self.ego_edges_by_node.setdefault(target, set())
        source_neighbors.add(target)
        target_neighbors.add(source)
        return affected, edge_updates

    def _remove_graph_edge(self, source: str, target: str) -> tuple[set[str], int]:
        if not self.graph.has_edge(source, target):
            raise ValueError(f"Cannot remove missing edge: {(source, target)}")
        common = self.adjacency[source].intersection(self.adjacency[target])
        affected = {source, target, *common}
        edge_updates = self._remove_triangle_ego_edges(source, target, common)
        self.graph.remove_edge(source, target)
        self.adjacency[source].discard(target)
        self.adjacency[target].discard(source)
        return affected, edge_updates

    def _add_triangle_ego_edges(self, source: str, target: str, common: set[str]) -> int:
        updates = 0
        source_edges = self.ego_edges_by_node.setdefault(source, set())
        target_edges = self.ego_edges_by_node.setdefault(target, set())
        for neighbor in common:
            neighbor_edges = self.ego_edges_by_node.setdefault(neighbor, set())
            updates += self._add_ego_edge(source_edges, sorted_edge(target, neighbor))
            updates += self._add_ego_edge(target_edges, sorted_edge(source, neighbor))
            updates += self._add_ego_edge(neighbor_edges, sorted_edge(source, target))
        return updates

    def _remove_triangle_ego_edges(self, source: str, target: str, common: set[str]) -> int:
        updates = 0
        source_edges = self.ego_edges_by_node.setdefault(source, set())
        target_edges = self.ego_edges_by_node.setdefault(target, set())
        for neighbor in common:
            neighbor_edges = self.ego_edges_by_node.setdefault(neighbor, set())
            updates += self._discard_ego_edge(source_edges, sorted_edge(target, neighbor))
            updates += self._discard_ego_edge(target_edges, sorted_edge(source, neighbor))
            updates += self._discard_ego_edge(neighbor_edges, sorted_edge(source, target))
        return updates

    @staticmethod
    def _add_ego_edge(edges: set[Edge], edge: Edge) -> int:
        if edge in edges:
            return 0
        edges.add(edge)
        return 1

    @staticmethod
    def _discard_ego_edge(edges: set[Edge], edge: Edge) -> int:
        if edge not in edges:
            return 0
        edges.remove(edge)
        return 1

    def _replace_ego(self, ego: str) -> int:
        old_contribution = self.ego_contributions.get(ego, Counter())
        old_labels = self.ego_labels.get(ego) if self.reuse_labels else None
        contribution, clusters, labels = self._compute_ego(ego, old_labels)
        changed_pairs = self._apply_score_delta(old_contribution, contribution)
        if contribution:
            self.ego_contributions[ego] = contribution
        else:
            self.ego_contributions.pop(ego, None)
        if clusters:
            self.ego_clusters[ego] = clusters
        else:
            self.ego_clusters.pop(ego, None)
        if labels:
            self.ego_labels[ego] = labels
        else:
            self.ego_labels.pop(ego, None)
        return changed_pairs

    def _apply_score_delta(
        self,
        old: Counter[tuple[str, str]],
        new: Counter[tuple[str, str]],
    ) -> int:
        changed_pairs = 0
        for pair in set(old).union(new):
            delta = new.get(pair, 0.0) - old.get(pair, 0.0)
            if delta == 0:
                continue
            self.scores[pair] += delta
            if self.scores[pair] <= 0:
                del self.scores[pair]
            changed_pairs += 1
        return changed_pairs

    def _compute_ego(
        self,
        ego: str,
        initial_labels: dict[str, str] | None,
    ) -> tuple[Counter[tuple[str, str]], list[set[str]], dict[str, str]]:
        contribution: Counter[tuple[str, str]] = Counter()
        if ego not in self.graph:
            return contribution, [], {}

        ego_edges = self.ego_edges_by_node.get(ego, set())
        if not ego_edges:
            return contribution, [], {}
        ego_graph = csr_graph_from_edges(ego_edges)

        if self.clusterer == "apm":
            clusters, labels = csr_apm_label_propagation_with_labels(
                ego_graph,
                initial_labels=initial_labels,
                alpha=self.alpha,
                beta=self.beta,
                max_iter=self.max_iter,
                seed=stable_seed(ego, self.seed),
            )
        else:
            clusters = csr_connected_components(ego_graph)
            labels = {}
            for cluster in clusters:
                representative = min(ego_graph.nodes[node] for node in cluster)
                for node in cluster:
                    labels[ego_graph.nodes[node]] = representative

        kept_clusters: list[set[str]] = []
        for cluster in clusters:
            if len(cluster) < self.min_cluster_size:
                continue
            nodes = [ego_graph.nodes[node] for node in cluster]
            cluster_set = set(nodes)
            kept_clusters.append(cluster_set)
            nodes.sort()
            for idx, source in enumerate(nodes):
                for target in nodes[idx + 1 :]:
                    if target not in self.adjacency.get(source, set()):
                        contribution[(source, target)] += 1.0
        return contribution, kept_clusters, labels
