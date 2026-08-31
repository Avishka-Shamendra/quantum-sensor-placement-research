"""Tests for randomized near-uniform and random allocation baselines."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from algorithms.near_uniform import (  # noqa: E402
    near_uniform_global_discrimination,
)
from algorithms.random_allocation import (  # noqa: E402
    random_global_discrimination,
)
from objective import pgm_error_probability  # noqa: E402


DELTAS = {
    (0, 1): [-1.0, 0.6, 0.4, -0.2, 0.8],
    (0, 2): [0.5, -0.8, -0.3, 0.7, -0.1],
    (1, 2): [1.5, -1.4, -0.7, 0.9, -0.9],
}


class BaselineAlgorithmTests(unittest.TestCase):
    def test_near_uniform_distributes_remainder_randomly(self) -> None:
        first = near_uniform_global_discrimination(
            3, 5, 12, DELTAS, "product", seed=19
        )
        second = near_uniform_global_discrimination(
            3, 5, 12, DELTAS, "product", seed=19
        )

        self.assertEqual(first.allocation, second.allocation)
        self.assertEqual(sum(first.allocation), 12)
        self.assertLessEqual(max(first.allocation) - min(first.allocation), 1)
        self.assertEqual(sorted(first.allocation), [2, 2, 2, 3, 3])

    def test_random_allocation_is_reproducible_and_feasible(self) -> None:
        first = random_global_discrimination(
            3, 5, 12, DELTAS, "local_ghz", seed=23
        )
        second = random_global_discrimination(
            3, 5, 12, DELTAS, "local_ghz", seed=23
        )

        self.assertEqual(first.allocation, second.allocation)
        self.assertEqual(sum(first.allocation), 12)
        self.assertTrue(all(count >= 0 for count in first.allocation))

    def test_baselines_report_recomputed_actual_pgm_error(self) -> None:
        results = (
            near_uniform_global_discrimination(
                3, 5, 7, DELTAS, "global_ghz", seed=29
            ),
            random_global_discrimination(
                3, 5, 7, DELTAS, "global_ghz", seed=31
            ),
        )
        for result in results:
            recomputed = pgm_error_probability(
                3, 5, result.allocation, DELTAS, "global_ghz"
            )
            self.assertAlmostEqual(result.error_probability, recomputed)


if __name__ == "__main__":
    unittest.main()
