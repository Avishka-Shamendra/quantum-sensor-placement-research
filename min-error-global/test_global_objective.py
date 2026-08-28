"""Tests for unwrapped global inputs and actual PGM-error optimization."""

from __future__ import annotations

import sys
import tempfile
import unittest
from math import pi, sqrt
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from algorithms.greedy import greedy_global_discrimination  # noqa: E402
from algorithms.optimal import optimal_global_discrimination  # noqa: E402
from algorithms.vnd import vnd_global_discrimination  # noqa: E402
from input_data import (  # noqa: E402
    load_delta_input,
    validate_delta_consistency,
    write_delta_input,
)
from objective import (  # noqa: E402
    STATE_MODELS,
    output_state_gram_matrix,
    pgm_error_probability,
)


CONSISTENT_DELTAS = {
    (0, 1): [-1.0, 0.6],
    (0, 2): [0.5, -0.8],
    (1, 2): [1.5, -1.4],
}


class GlobalObjectiveTests(unittest.TestCase):
    def test_two_state_product_pgm_matches_closed_form(self) -> None:
        deltas = {(0, 1): [pi / 2.0]}
        actual = pgm_error_probability(
            2,
            1,
            (1,),
            deltas,
            "product",
        )
        expected = 0.5 * (1.0 - 1.0 / sqrt(2.0))

        self.assertAlmostEqual(actual, expected, places=12)

    def test_unwrapped_deltas_above_pi_form_psd_gram_matrices(self) -> None:
        deltas = {
            (0, 1): [-4.0],
            (0, 2): [2.0],
            (1, 2): [6.0],
        }
        validate_delta_consistency(deltas, K=3, M=1)

        for state_model in STATE_MODELS:
            gram = output_state_gram_matrix(
                3,
                1,
                (3,),
                deltas,
                state_model,
            )
            self.assertGreaterEqual(float(np.linalg.eigvalsh(gram)[0]), -1e-10)

    def test_inconsistent_delta_triangle_is_rejected(self) -> None:
        inconsistent = {
            (0, 1): [1.0],
            (0, 2): [3.0],
            (1, 2): [1.0],
        }

        with self.assertRaisesRegex(ValueError, "not mutually consistent"):
            validate_delta_consistency(inconsistent, K=3, M=1)

    def test_delta_csv_round_trip(self) -> None:
        rows = [
            (i, j, m, delta)
            for (i, j), values in CONSISTENT_DELTAS.items()
            for m, delta in enumerate(values)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.csv"
            write_delta_input(path, rows)
            K, M, loaded = load_delta_input(path)

        self.assertEqual((K, M), (3, 2))
        self.assertEqual(loaded, CONSISTENT_DELTAS)

    def test_all_algorithms_optimize_and_report_actual_error(self) -> None:
        for state_model in STATE_MODELS:
            optimal = optimal_global_discrimination(
                3,
                2,
                4,
                CONSISTENT_DELTAS,
                state_model,
            )
            greedy = greedy_global_discrimination(
                3,
                2,
                4,
                CONSISTENT_DELTAS,
                state_model,
            )
            vnd = vnd_global_discrimination(
                3,
                2,
                4,
                CONSISTENT_DELTAS,
                state_model,
            )

            self.assertLessEqual(
                optimal.error_probability,
                greedy.error_probability + 1e-12,
            )
            self.assertLessEqual(
                optimal.error_probability,
                vnd.error_probability + 1e-12,
            )
            for result in (optimal, greedy, vnd):
                recomputed = pgm_error_probability(
                    3,
                    2,
                    result.allocation,
                    CONSISTENT_DELTAS,
                    state_model,
                )
                self.assertAlmostEqual(
                    result.error_probability,
                    recomputed,
                    places=12,
                )
                self.assertGreaterEqual(result.error_upper_bound, 0.0)


if __name__ == "__main__":
    unittest.main()
