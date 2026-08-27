"""Exact exhaustive search for minimum-error global discrimination."""

from __future__ import annotations

from collections.abc import Iterator
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
class OptimalResult:
    """The best allocation found by exhaustive search."""

    allocation: tuple[int, ...]
    error_upper_bound: float
    state_model: StateModel


def optimal_global_discrimination(
    K: int,
    M: int,
    N: int,
    rijm: PhaseTable,
    state_model: str,
) -> OptimalResult:
    """Find the globally optimal fixed-budget sensor allocation.

    This is Algorithm 6 in Section V of the paper. It enumerates every
    nonnegative integer allocation whose entries sum to ``N`` and minimizes
    the PGM error upper bound for the selected input-state model.

    Args:
        K: Number of candidate fields, indexed ``0`` through ``K - 1``.
            Exactly one of these fields is assumed to be active.
        M: Number of candidate sensing locations, indexed ``0`` through
            ``M - 1``.
        N: Fixed total number of one-qubit sensors. Every returned allocation
            uses exactly this budget.
        rijm: Pairwise phase differences in radians. This mapping must contain
            every canonical field pair ``(i, j)`` with ``0 <= i < j < K``.
            ``rijm[(i, j)]`` is a length-``M`` sequence, and entry ``m`` is
            ``r_ijm`` at location ``m`` in the principal interval ``(-pi, pi]``.
        state_model: Fixed input-state model. Use ``"product"`` for independent
            ``|+>`` sensors, ``"local_ghz"`` for location-wise GHZ states, or
            ``"global_ghz"`` for one GHZ state across all ``N`` sensors.

    Returns:
        An :class:`OptimalResult`. ``allocation[m]`` is the number of sensors
        assigned to location ``m`` and ``error_upper_bound`` is the minimized
        Pretty Good Measurement (PGM) upper bound from Section V.

    Raises:
        ValueError: If a dimension, budget, phase, pair key, or state model is
            invalid.

    Notes:
        There are ``binomial(N + M - 1, M - 1)`` feasible allocations. The
        paper gives total complexity O(binomial(N+M-1, M-1) * M * K^2).
        Equal objective values are resolved by the first allocation generated.
    """
    pairs, checked_model = validate_problem(K, M, N, rijm, state_model)

    best_allocation: tuple[int, ...] | None = None
    best_bound = float("inf")

    # Exhaust all weak compositions of N into M location counts.
    for allocation in _budget_allocations(N, M):
        bound = pgm_error_upper_bound(
            K, M, allocation, rijm, checked_model, pairs
        )
        if bound < best_bound:
            best_allocation = allocation
            best_bound = bound

    assert best_allocation is not None
    return OptimalResult(
        allocation=best_allocation,
        error_upper_bound=best_bound,
        state_model=checked_model,
    )


def _budget_allocations(total: int, locations: int) -> Iterator[tuple[int, ...]]:
    """Yield every nonnegative allocation with entries summing to ``total``."""
    if locations == 1:
        yield (total,)
        return

    for first in range(total + 1):
        for remainder in _budget_allocations(total - first, locations - 1):
            yield (first, *remainder)
