"""Greedy allocation for minimum-error global discrimination."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


PROBLEM_ROOT = Path(__file__).resolve().parent.parent
if str(PROBLEM_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBLEM_ROOT))

from objective import (  # noqa: E402
    DeltaTable,
    StateModel,
    pgm_error_probability,
    pgm_error_upper_bound,
    validate_problem,
)


@dataclass(frozen=True)
class GreedyResult:
    """The allocation and objective value produced by the greedy method."""

    allocation: tuple[int, ...]
    error_probability: float
    error_upper_bound: float
    state_model: StateModel


def greedy_global_discrimination(
    K: int,
    M: int,
    N: int,
    delta_ijm: DeltaTable,
    state_model: str,
) -> GreedyResult:
    """Allocate the fixed sensor budget one sensor at a time.

    At each of ``N`` steps, the method tentatively adds one sensor at every
    location and selects the location producing the smallest actual PGM error
    from the corresponding output-state Gram matrix.

    Args:
        K: Number of candidate fields, indexed ``0`` through ``K - 1``.
            Exactly one of these fields is assumed to be active.
        M: Number of candidate sensing locations, indexed ``0`` through
            ``M - 1``.
        N: Fixed total number of one-qubit sensors. The algorithm places one
            sensor per iteration and therefore uses the entire budget.
        delta_ijm: Unwrapped pairwise phase differences in radians. This mapping must contain
            every canonical field pair ``(i, j)`` with ``0 <= i < j < K``.
            ``delta_ijm[(i, j)]`` contains ``c*(f_i-f_j)`` at each location.
        state_model: Fixed input-state model. Use ``"product"`` for independent
            ``|+>`` sensors, ``"local_ghz"`` for location-wise GHZ states, or
            ``"global_ghz"`` for one GHZ state across all ``N`` sensors.

    Returns:
        A :class:`GreedyResult`. ``allocation[m]`` is the final number of
        sensors assigned to location ``m`` and ``error_probability`` is its
        actual PGM error. ``error_upper_bound`` is reported secondarily.

    Raises:
        ValueError: If a dimension, budget, phase, pair key, or state model is
            invalid.

    Notes:
        Objective ties are resolved in favor of the lowest-indexed location.
    """
    pairs, checked_model = validate_problem(
        K, M, N, delta_ijm, state_model
    )
    allocation = [0] * M

    # Place exactly one sensor in each iteration of Algorithm 7.
    for _ in range(N):
        best_location = 0
        best_error = float("inf")

        for location in range(M):
            trial = allocation.copy()
            trial[location] += 1
            error = pgm_error_probability(
                K, M, trial, delta_ijm, checked_model, pairs
            )

            # Strict comparison gives deterministic lowest-index tie-breaking.
            if error < best_error:
                best_location = location
                best_error = error

        allocation[best_location] += 1

    final_allocation = tuple(allocation)
    final_error = pgm_error_probability(
        K, M, final_allocation, delta_ijm, checked_model, pairs
    )
    return GreedyResult(
        allocation=final_allocation,
        error_probability=final_error,
        error_upper_bound=pgm_error_upper_bound(
            K,
            M,
            final_allocation,
            delta_ijm,
            checked_model,
            pairs,
        ),
        state_model=checked_model,
    )
