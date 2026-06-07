from __future__ import annotations

import pathlib
import sys
import unittest

import networkx as nx

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egonet_mining.algorithms import (
    apm_label_propagation,
    csr_apm_label_propagation,
    csr_graph_from_edges,
    ego_feature_scores,
    ego_friendship_scores,
    ego_graph_from_edges,
    triangle_ego_net_edges,
)
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

    def test_dynamic_triangle_index_tracks_edge_events(self) -> None:
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
        self.assertEqual(
            {ego: edges for ego, edges in miner.ego_edges_by_node.items() if edges},
            naive_ego_edges(miner.graph),
        )
        miner.apply_events([("add", "u", "v"), ("remove", "a", "b")])
        self.assertEqual(
            {ego: edges for ego, edges in miner.ego_edges_by_node.items() if edges},
            naive_ego_edges(miner.graph),
        )

    def test_dynamic_update_matches_fresh_rebuild_without_label_reuse(self) -> None:
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

        miner = DynamicEgoNetMiner(graph, seed=11, reuse_labels=False)
        miner.initialize()
        miner.apply_events([("add", "u", "v"), ("remove", "a", "b")])

        fresh = DynamicEgoNetMiner(miner.graph, seed=11, reuse_labels=False)
        fresh.initialize()

        self.assertEqual(dict(miner.scores), dict(fresh.scores))
        self.assertEqual(
            sorted(sorted(cluster) for cluster in miner.partition()),
            sorted(sorted(cluster) for cluster in fresh.partition()),
        )

    def test_cc_clusterer_scores_paper_w1_to_w4_inside_ego_nets(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from(
            [
                ("u", "a"),
                ("u", "b"),
                ("u", "c"),
                ("u", "d"),
                ("a", "c"),
                ("b", "c"),
                ("b", "d"),
            ]
        )

        result = ego_feature_scores(
            graph,
            clusterer="cc",
            candidate_pairs=[("b", "a")],
            construction="triangle",
        )
        full_result = ego_feature_scores(
            graph,
            clusterer="cc",
            construction="triangle",
        )

        pair = ("a", "b")
        self.assertEqual(result.scores["W1"][pair], 2.0)
        self.assertAlmostEqual(result.scores["W2"][pair], 0.5 + 2 / 3)
        self.assertEqual(result.scores["W3"][pair], 2.0)
        self.assertAlmostEqual(result.scores["W4"][pair], 1 / 4 + 1 / 3)
        for feature in ("W1", "W2", "W3", "W4"):
            self.assertAlmostEqual(result.scores[feature][pair], full_result.scores[feature][pair])

    def test_csr_apm_matches_networkx_apm_on_same_ego_graph(self) -> None:
        edges = {
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
            ("c", "d"),
            ("d", "e"),
            ("e", "f"),
            ("d", "f"),
        }

        nx_clusters = apm_label_propagation(ego_graph_from_edges(edges), seed=13)
        csr_graph = csr_graph_from_edges(edges)
        csr_clusters = [
            {csr_graph.nodes[idx] for idx in cluster}
            for cluster in csr_apm_label_propagation(csr_graph, seed=13)
        ]

        self.assertEqual(
            sorted(sorted(cluster) for cluster in nx_clusters),
            sorted(sorted(cluster) for cluster in csr_clusters),
        )

    def test_parallel_candidate_scoring_matches_serial(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from(
            [
                ("u", "a"),
                ("u", "b"),
                ("u", "c"),
                ("u", "d"),
                ("a", "c"),
                ("b", "c"),
                ("b", "d"),
                ("v", "a"),
                ("v", "b"),
                ("v", "d"),
            ]
        )
        candidates = [("a", "b"), ("c", "d")]

        serial = ego_feature_scores(
            graph,
            clusterer="apm",
            candidate_pairs=candidates,
            construction="triangle",
            seed=17,
            ego_workers=1,
        )
        try:
            parallel = ego_feature_scores(
                graph,
                clusterer="apm",
                candidate_pairs=candidates,
                construction="triangle",
                seed=17,
                ego_workers=2,
            )
        except PermissionError as exc:
            self.skipTest(f"process pools are unavailable in this sandbox: {exc}")

        self.assertEqual(serial.scores, parallel.scores)
        self.assertEqual(serial.stats, parallel.stats)
        self.assertEqual(
            sorted(sorted(cluster) for cluster in serial.clusters),
            sorted(sorted(cluster) for cluster in parallel.clusters),
        )


if __name__ == "__main__":
    unittest.main()
