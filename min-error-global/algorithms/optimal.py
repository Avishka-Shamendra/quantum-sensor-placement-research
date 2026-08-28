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
    DeltaTable,
    StateModel,
    pgm_error_probability,
    pgm_error_upper_bound,
    validate_problem,
)


@dataclass(frozen=True)
class OptimalResult:
    """The best allocation found by exhaustive search."""

    allocation: tuple[int, ...]
    error_probability: float
    error_upper_bound: float
    state_model: StateModel


def optimal_global_discrimination(
    K: int,
    M: int,
    N: int,
    delta_ijm: DeltaTable,
    state_model: str,
) -> OptimalResult:
    """Find the globally optimal fixed-budget sensor allocation.

    It enumerates every nonnegative integer allocation whose entries sum to
    ``N`` and minimizes the actual equal-prior PGM error obtained from the
    output-state Gram matrix for the selected input-state model.

    Args:
        K: Number of candidate fields, indexed ``0`` through ``K - 1``.
            Exactly one of these fields is assumed to be active.
        M: Number of candidate sensing locations, indexed ``0`` through
            ``M - 1``.
        N: Fixed total number of one-qubit sensors. Every returned allocation
            uses exactly this budget.
        delta_ijm: Unwrapped pairwise phase differences in radians. This mapping must contain
            every canonical field pair ``(i, j)`` with ``0 <= i < j < K``.
            ``delta_ijm[(i, j)]`` is a length-``M`` sequence, and entry ``m``
            equals ``c*(f_i(x_m)-f_j(x_m))`` without phase wrapping.
        state_model: Fixed input-state model. Use ``"product"`` for independent
            ``|+>`` sensors, ``"local_ghz"`` for location-wise GHZ states, or
            ``"global_ghz"`` for one GHZ state across all ``N`` sensors.

    Returns:
        An :class:`OptimalResult`. ``allocation[m]`` is the number of sensors
        assigned to location ``m`` and ``error_probability`` is the minimized
        actual Pretty Good Measurement (PGM) error. ``error_upper_bound`` is
        also returned as a secondary reporting metric.

    Raises:
        ValueError: If a dimension, budget, phase, pair key, or state model is
            invalid.

    Notes:
        There are ``binomial(N + M - 1, M - 1)`` feasible allocations. The
        Each objective evaluation includes an eigendecomposition of a ``K`` by
        ``K`` Gram matrix. Equal values use the first generated allocation.
    """
    pairs, checked_model = validate_problem(
        K, M, N, delta_ijm, state_model
    )

    best_allocation: tuple[int, ...] | None = None
    best_error = float("inf")

    # Exhaust all weak compositions of N into M location counts.
    for allocation in _budget_allocations(N, M):
        error = pgm_error_probability(
            K, M, allocation, delta_ijm, checked_model, pairs
        )
        if error < best_error:
            best_allocation = allocation
            best_error = error

    assert best_allocation is not None
    return OptimalResult(
        allocation=best_allocation,
        error_probability=best_error,
        error_upper_bound=pgm_error_upper_bound(
            K,
            M,
            best_allocation,
            delta_ijm,
            checked_model,
            pairs,
        ),
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
