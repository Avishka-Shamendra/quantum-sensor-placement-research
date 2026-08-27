"""Greedy allocation for minimum-error global discrimination."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


PROBLEM_ROOT = Path(__file__).resolve().parent.parent
if str(PROBLEM_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBLEM_ROOT))

from objective import (  # noqa: E402
    PhaseTable,
    StateModel,
    pgm_error_upper_bound,
    validate_problem,
)


@dataclass(frozen=True)
class GreedyResult:
    """The allocation and objective value produced by the greedy method."""

    allocation: tuple[int, ...]
    error_upper_bound: float
    state_model: StateModel


def greedy_global_discrimination(
    K: int,
    M: int,
    N: int,
    rijm: PhaseTable,
    state_model: str,
) -> GreedyResult:
    """Allocate the fixed sensor budget one sensor at a time.

    This is Algorithm 7 in Section V of the paper. At each of ``N`` steps, it
    tentatively adds one sensor at every location and permanently selects the
    location producing the smallest PGM error upper bound.

    Args:
        K: Number of candidate fields, indexed ``0`` through ``K - 1``.
            Exactly one of these fields is assumed to be active.
        M: Number of candidate sensing locations, indexed ``0`` through
            ``M - 1``.
        N: Fixed total number of one-qubit sensors. The algorithm places one
            sensor per iteration and therefore uses the entire budget.
        rijm: Pairwise phase differences in radians. This mapping must contain
            every canonical field pair ``(i, j)`` with ``0 <= i < j < K``.
            ``rijm[(i, j)]`` is a length-``M`` sequence, and entry ``m`` is
            ``r_ijm`` at location ``m`` in the principal interval ``(-pi, pi]``.
        state_model: Fixed input-state model. Use ``"product"`` for independent
            ``|+>`` sensors, ``"local_ghz"`` for location-wise GHZ states, or
            ``"global_ghz"`` for one GHZ state across all ``N`` sensors.

    Returns:
        A :class:`GreedyResult`. ``allocation[m]`` is the final number of
        sensors assigned to location ``m`` and ``error_upper_bound`` is its
        Pretty Good Measurement (PGM) upper bound from Section V.

    Raises:
        ValueError: If a dimension, budget, phase, pair key, or state model is
            invalid.

    Notes:
        The paper gives complexity O(N * M^2 * K^2). Objective ties are
        resolved in favor of the lowest-indexed location.
    """
    pairs, checked_model = validate_problem(K, M, N, rijm, state_model)
    allocation = [0] * M

    # Place exactly one sensor in each iteration of Algorithm 7.
    for _ in range(N):
        best_location = 0
        best_bound = float("inf")

        for location in range(M):
            trial = allocation.copy()
            trial[location] += 1
            bound = pgm_error_upper_bound(
                K, M, trial, rijm, checked_model, pairs
            )

            # Strict comparison gives deterministic lowest-index tie-breaking.
            if bound < best_bound:
                best_location = location
                best_bound = bound

        allocation[best_location] += 1

    final_allocation = tuple(allocation)
    final_bound = pgm_error_upper_bound(
        K, M, final_allocation, rijm, checked_model, pairs
    )
    return GreedyResult(
        allocation=final_allocation,
        error_upper_bound=final_bound,
        state_model=checked_model,
    )
