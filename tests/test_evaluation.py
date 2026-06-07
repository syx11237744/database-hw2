from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egonet_mining.evaluation import precision_recall_curve_at_k


class EvaluationTest(unittest.TestCase):
    def test_precision_recall_curve_at_k_is_deterministic_for_score_ties(self) -> None:
        positives = [("a", "b"), ("c", "d")]
        negatives = [("a", "c"), ("b", "d")]
        scores = {
            ("a", "c"): 0.9,
            ("a", "b"): 0.8,
            ("b", "d"): 0.8,
            ("c", "d"): 0.1,
        }

        curve = precision_recall_curve_at_k(scores, positives, negatives, [1, 2, 10])

        self.assertEqual(curve[0]["k"], 1.0)
        self.assertEqual(curve[0]["evaluated_k"], 1.0)
        self.assertEqual(curve[0]["precision"], 0.0)
        self.assertEqual(curve[0]["recall"], 0.0)
        self.assertEqual(curve[1]["k"], 2.0)
        self.assertEqual(curve[1]["evaluated_k"], 2.0)
        self.assertEqual(curve[1]["precision"], 0.5)
        self.assertEqual(curve[1]["recall"], 0.5)
        self.assertEqual(curve[2]["k"], 10.0)
        self.assertEqual(curve[2]["evaluated_k"], 4.0)
        self.assertEqual(curve[2]["precision"], 0.5)
        self.assertEqual(curve[2]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
