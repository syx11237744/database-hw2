from __future__ import annotations

import pathlib
import sys
import unittest

import networkx as nx

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egonet_mining.algorithms import ego_friendship_scores, triangle_ego_net_edges
from egonet_mining.dynamic import DynamicEgoNetMiner


def naive_ego_edges(graph: nx.Graph) -> dict[str, set[tuple[str, str]]]:
    result: dict[str, set[tuple[str, str]]] = {}
    for ego in graph.nodes():
        neighbors = set(graph.neighbors(ego))
        edges: set[tuple[str, str]] = set()
        for source, target in graph.edges():
            if source in neighbors and target in neighbors:
                edges.add((source, target) if source <= target else (target, source))
        if edges:
            result[ego] = edges
    return result


class TriangleConstructionTest(unittest.TestCase):
    def test_triangle_ego_edges_match_naive_induced_edges(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from(
            [
                ("u", "a"),
                ("u", "b"),
                ("u", "c"),
                ("a", "b"),
                ("b", "c"),
                ("x", "y"),
                ("y", "z"),
                ("z", "x"),
                ("a", "x"),
            ]
        )

        self.assertEqual(triangle_ego_net_edges(graph), naive_ego_edges(graph))

    def test_triangle_and_neighbor_construction_produce_same_scores(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from(
            [
                ("u", "a"),
                ("u", "b"),
                ("u", "c"),
                ("a", "b"),
                ("b", "c"),
                ("v", "a"),
                ("v", "b"),
                ("v", "c"),
                ("u", "v"),
            ]
        )

        triangle_scores, triangle_clusters = ego_friendship_scores(
            graph,
            seed=7,
            construction="triangle",
        )
        neighbor_scores, neighbor_clusters = ego_friendship_scores(
            graph,
            seed=7,
            construction="neighbor",
        )

        self.assertEqual(triangle_scores, neighbor_scores)
        self.assertEqual(
            sorted(sorted(cluster) for cluster in triangle_clusters),
            sorted(sorted(cluster) for cluster in neighbor_clusters),
        )

    def test_dynamic_update_matches_fresh_rebuild(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from(
            [
                ("u", "a"),
                ("u", "b"),
                ("u", "c"),
                ("a", "b"),
                ("b", "c"),
                ("v", "a"),
                ("v", "b"),
                ("v", "c"),
            ]
        )

        miner = DynamicEgoNetMiner(graph, seed=11)
        miner.initialize()
        miner.apply_events([("add", "u", "v"), ("remove", "a", "b")])

        fresh = DynamicEgoNetMiner(miner.graph, seed=11)
        fresh.initialize()

        self.assertEqual(dict(miner.scores), dict(fresh.scores))
        self.assertEqual(
            sorted(sorted(cluster) for cluster in miner.partition()),
            sorted(sorted(cluster) for cluster in fresh.partition()),
        )


if __name__ == "__main__":
    unittest.main()
