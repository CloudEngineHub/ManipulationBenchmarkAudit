import math
import unittest
from decimal import Decimal, InvalidOperation
from statistics import NormalDist

import generate_significance_categories as categories
import significance_cutoffs as reference


class SignificanceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simpler = next(p for p in categories.PARAMS if p.benchmark == "SimplerEnv")
        cls.libero = next(p for p in categories.PARAMS if p.track == "goal")
        cls.z = NormalDist().inv_cdf(0.95)
        cls.qmax = categories.compute_binary_qmax_all(4, 24)

    def test_four_task_counterexamples(self):
        self.assertEqual((self.simpler.T, self.simpler.S), (4, 24))
        # Feasible (A successes, B successes, discordances) in each task.
        cases = [
            (41, 57, [(10, 14, 24)] * 3 + [(11, 15, 22)]),
            (29, 43, [(7, 10, 17)] * 2 + [(7, 11, 18), (8, 12, 20)]),
        ]
        for A, B, tasks in cases:
            with self.subTest(A=A, B=B):
                self.assertEqual(sum(a for a, b, m in tasks), A)
                self.assertEqual(sum(b for a, b, m in tasks), B)
                for a, b, m in tasks:
                    self.assertLessEqual(abs(b - a), m)
                    self.assertLessEqual(m, min(a + b, 48 - a - b))
                    self.assertEqual((a + b - m) % 2, 0)
                Q = sum(m - (b - a) ** 2 / 24 for a, b, m in tasks)
                self.assertLess((B - A) / math.sqrt(24 / 23 * Q), self.z)
                self.assertAlmostEqual(self.qmax[A, B], Q)
                result, _, _ = categories.category_for(
                    Decimal(B - A), A, B, self.simpler, self.z, self.qmax
                )
                self.assertEqual(result, categories.INDETERMINATE)

    def test_decimal_halfway_ties(self):
        for score, count in [("94.9", 474), ("94.3", 472), ("99.9", 500)]:
            with self.subTest(score=score):
                parsed = categories.parse_score(score)
                self.assertEqual(categories.score_to_count(parsed, self.libero), count)
                mu_a = float(Decimal(score) / 100)
                cutoff = reference.compute_cutoffs(mu_a, 1, 500, 1, 0.05)
                self.assertEqual(cutoff["P_A"], count / 500)

    def test_zero_and_saturated_gaps(self):
        cases = [
            (Decimal("0"), 96, 96, categories.NO_IMPROVEMENT),
            (Decimal("0.01"), 96, 96, categories.PROVABLY_NOT_SIGNIFICANT),
            (Decimal("1"), 95, 96, categories.PROVABLY_NOT_SIGNIFICANT),
        ]
        for delta, A, B, expected in cases:
            with self.subTest(delta=delta, A=A, B=B):
                result, _, _ = categories.category_for(
                    delta, A, B, self.simpler, self.z, self.qmax
                )
                self.assertEqual(result, expected)

    def test_missing_and_invalid_scores(self):
        self.assertIsNone(categories.parse_score(" "))
        with self.assertRaises(InvalidOperation):
            categories.parse_score("unknown")
        with self.assertRaises(ValueError):
            categories.score_to_count(Decimal("100.1"), self.libero)


if __name__ == "__main__":
    unittest.main()
