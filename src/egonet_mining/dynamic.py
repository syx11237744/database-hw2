from __future__ import annotations

import hashlib
import time
from collections import Counter
from collections.abc import Iterable

import networkx as nx

from .algorithms import apm_label_propagation, ego_score_partition

EdgeEvent = tuple[str, str, str]


def stable_seed(value: str, base_seed: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=4).digest()
    return base_seed + int.from_bytes(digest, "big")


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
    ) -> None:
        self.graph = graph.copy()
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.min_cluster_size = min_cluster_size
        self.min_score = min_score
        self.seed = seed
        self.scores: Counter[tuple[str, str]] = Counter()
        self.ego_contributions: dict[str, Counter[tuple[str, str]]] = {}
        self.ego_clusters: dict[str, list[set[str]]] = {}

    def initialize(self) -> float:
        start = time.perf_counter()
        for ego in sorted(self.graph.nodes()):
            self._replace_ego(ego)
        return time.perf_counter() - start

    def apply_events(self, events: Iterable[EdgeEvent]) -> dict[str, float]:
        events = list(events)
        affected = self._affected_egos(events)
        start = time.perf_counter()

        for ego in affected:
            self._remove_ego(ego)

        for action, source, target in events:
            if source == target:
                continue
            if action == "add":
                self.graph.add_edge(source, target)
            elif action == "remove" and self.graph.has_edge(source, target):
                self.graph.remove_edge(source, target)
            else:
                raise ValueError(f"Unsupported edge event action: {action}")

        for ego in sorted(affected):
            if ego in self.graph:
                self._replace_ego(ego)

        return {
            "affected_egos": float(len(affected)),
            "runtime_sec": time.perf_counter() - start,
        }

    def partition(self) -> list[set[str]]:
        return ego_score_partition(self.graph, self.scores, min_score=self.min_score)

    def _affected_egos(self, events: Iterable[EdgeEvent]) -> set[str]:
        affected: set[str] = set()
        for _, source, target in events:
            affected.update([source, target])
            if source in self.graph and target in self.graph:
                affected.update(nx.common_neighbors(self.graph, source, target))
        return affected

    def _replace_ego(self, ego: str) -> None:
        contribution, clusters = self._compute_ego(ego)
        self.ego_contributions[ego] = contribution
        self.ego_clusters[ego] = clusters
        self.scores.update(contribution)

    def _remove_ego(self, ego: str) -> None:
        old = self.ego_contributions.pop(ego, Counter())
        self.ego_clusters.pop(ego, None)
        for pair, value in old.items():
            self.scores[pair] -= value
            if self.scores[pair] <= 0:
                del self.scores[pair]

    def _compute_ego(self, ego: str) -> tuple[Counter[tuple[str, str]], list[set[str]]]:
        contribution: Counter[tuple[str, str]] = Counter()
        if ego not in self.graph:
            return contribution, []

        neighbors = sorted(self.graph.neighbors(ego))
        if len(neighbors) < self.min_cluster_size:
            return contribution, []

        ego_graph = self.graph.subgraph(neighbors).copy()
        if ego_graph.number_of_edges() == 0:
            return contribution, []

        clusters = apm_label_propagation(
            ego_graph,
            alpha=self.alpha,
            beta=self.beta,
            max_iter=self.max_iter,
            seed=stable_seed(ego, self.seed),
        )
        kept_clusters: list[set[str]] = []
        for cluster in clusters:
            if len(cluster) < self.min_cluster_size:
                continue
            kept_clusters.append(cluster)
            nodes = sorted(cluster)
            for idx, source in enumerate(nodes):
                for target in nodes[idx + 1 :]:
                    if not self.graph.has_edge(source, target):
                        contribution[(source, target)] += 1.0
        return contribution, kept_clusters
