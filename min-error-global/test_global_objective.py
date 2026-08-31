"""Tests for unwrapped global inputs and actual PGM-error optimization."""

from __future__ import annotations

import sys
import tempfile
import unittest
from math import pi, sqrt
from pathlib import Path
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from algorithms.greedy import greedy_global_discrimination  # noqa: E402
from algorithms.near_uniform import (  # noqa: E402
    near_uniform_global_discrimination,
)
from algorithms.optimal import optimal_global_discrimination  # noqa: E402
from algorithms.random_allocation import (  # noqa: E402
    random_global_discrimination,
)
import generate_inputs  # noqa: E402
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
    def test_global_benchmark_phase_scale_is_pi(self) -> None:
        self.assertAlmostEqual(generate_inputs.GLOBAL_BENCHMARK_PHASE_SCALE, pi)

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

    def test_generator_does_not_wrap_delta_values(self) -> None:
        def fixed_fields(K, locations, rng):
            del K, locations, rng
            return (
                [[4.5, 5.5], [0.2, 0.4], [-4.4, -5.2]],
                [{"fixed": True}] * 3,
            )

        with patch.dict(
            generate_inputs.FIELD_MODELS,
            {"unwrapped_test": fixed_fields},
        ):
            rows, _ = generate_inputs.generate_delta_case(
                "unwrapped_test",
                dimension=1,
                K=3,
                M=2,
                seed=1,
                phase_scale=1.0,
            )

        self.assertTrue(all(abs(delta) > pi for *_, delta in rows))
        self.assertAlmostEqual(rows[0][3], 4.3)

    def test_generator_preserves_zero_differences_without_resampling(self) -> None:
        def equal_fields(K, locations, rng):
            del K, rng
            values = [0.5] * len(locations)
            return ([values, values.copy()], [{"fixed": True}] * 2)

        with patch.dict(generate_inputs.FIELD_MODELS, {"equal_test": equal_fields}):
            rows, metadata = generate_inputs.generate_delta_case(
                "equal_test",
                dimension=1,
                K=2,
                M=2,
                seed=71,
                phase_scale=pi,
            )

        self.assertEqual([delta for *_, delta in rows], [0.0, 0.0])
        self.assertEqual(metadata["generation_seed"], 71)
        self.assertNotIn("generation_attempt", metadata)

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
            near_uniform = near_uniform_global_discrimination(
                3,
                2,
                4,
                CONSISTENT_DELTAS,
                state_model,
                seed=11,
            )
            random = random_global_discrimination(
                3,
                2,
                4,
                CONSISTENT_DELTAS,
                state_model,
                seed=17,
            )

            for result in (greedy, near_uniform, random):
                self.assertLessEqual(
                    optimal.error_probability,
                    result.error_probability + 1e-12,
                )
            for result in (optimal, greedy, near_uniform, random):
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


if __name__ == "__main__":
    unittest.main()
