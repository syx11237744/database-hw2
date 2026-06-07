from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egonet_mining.spark_algorithms import partition_triples_for_edge, stable_partition


class SparkPartitioningTest(unittest.TestCase):
    def test_cross_partition_edge_goes_to_rho_minus_two_triples(self) -> None:
        rho = 6
        seed = 3
        edge = ("source", "target")
        left = stable_partition(edge[0], rho, seed)
        right = stable_partition(edge[1], rho, seed)
        if left == right:
            self.skipTest("chosen fixture hashed to the same partition")

        triples = partition_triples_for_edge(edge, rho, seed)

        self.assertEqual(len(triples), rho - 2)
        self.assertEqual(len({key for key, _ in triples}), rho - 2)
        for key, value in triples:
            self.assertEqual(value, edge)
            self.assertIn(left, key)
            self.assertIn(right, key)

    def test_same_partition_edge_goes_to_other_partition_pairs(self) -> None:
        rho = 5
        seed = 11
        edge = None
        nodes = [f"n{idx}" for idx in range(100)]
        for left in nodes:
            for right in nodes:
                if left >= right:
                    continue
                if stable_partition(left, rho, seed) == stable_partition(right, rho, seed):
                    edge = (left, right)
                    break
            if edge is not None:
                break
        self.assertIsNotNone(edge)

        triples = partition_triples_for_edge(edge, rho, seed)

        self.assertEqual(len(triples), (rho - 1) * (rho - 2) // 2)
        self.assertEqual(len({key for key, _ in triples}), len(triples))


if __name__ == "__main__":
    unittest.main()
