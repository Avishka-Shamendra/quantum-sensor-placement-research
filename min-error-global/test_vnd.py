"""Focused tests for the deterministic VND sensor-allocation heuristic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from algorithms.vnd import vnd_sensor_allocation  # noqa: E402


class VNDSensorAllocationTests(unittest.TestCase):
    def test_near_uniform_start_is_returned_without_improvement(self) -> None:
        result = vnd_sensor_allocation(lambda allocation: 1.0, M=4, N=6)

        self.assertEqual(result.allocation, (2, 2, 1, 1))
        self.assertEqual(sum(result.allocation), 6)
        self.assertEqual(result.accepted_v1_moves, 0)
        self.assertEqual(result.accepted_v2_moves, 0)
        self.assertGreater(result.cache_hits, 0)

    def test_v1_uses_first_improvement_and_preserves_budget(self) -> None:
        target = (0, 2, 1)

        def objective(allocation: tuple[int, ...]) -> float:
            return float(sum(abs(a - b) for a, b in zip(allocation, target)))

        first = vnd_sensor_allocation(objective, M=3, N=3)
        second = vnd_sensor_allocation(objective, M=3, N=3)

        self.assertEqual(first.allocation, second.allocation)
        self.assertEqual(first.objective_value, second.objective_value)
        self.assertEqual(first.objective_evaluations, second.objective_evaluations)
        self.assertEqual(first.cache_hits, second.cache_hits)
        self.assertEqual(first.accepted_v1_moves, second.accepted_v1_moves)
        self.assertEqual(first.accepted_v2_moves, second.accepted_v2_moves)
        self.assertEqual(first.allocation, target)
        self.assertEqual(sum(first.allocation), 3)
        self.assertGreater(first.accepted_v1_moves, 0)

    def test_v2_can_improve_when_every_pair_is_locally_blocked(self) -> None:
        values = {
            (1, 1, 1): 3.0,
            (0, 0, 3): 1.0,
        }

        def objective(allocation: tuple[int, ...]) -> float:
            return values.get(allocation, 4.0)

        result = vnd_sensor_allocation(objective, M=3, N=3)

        self.assertEqual(result.allocation, (0, 0, 3))
        self.assertEqual(result.accepted_v1_moves, 0)
        self.assertEqual(result.accepted_v2_moves, 1)

    def test_improvement_threshold_is_enforced(self) -> None:
        def objective(allocation: tuple[int, ...]) -> float:
            return 1.0 if allocation == (1, 1) else 0.9995

        result = vnd_sensor_allocation(
            objective,
            M=2,
            N=2,
            improvement_threshold=1e-3,
        )

        self.assertEqual(result.allocation, (1, 1))
        self.assertEqual(result.accepted_v1_moves, 0)


if __name__ == "__main__":
    unittest.main()
