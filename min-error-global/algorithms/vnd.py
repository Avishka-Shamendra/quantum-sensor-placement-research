"""Deterministic variable-neighborhood descent for sensor allocation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from numbers import Real
from pathlib import Path
import sys
from time import perf_counter


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


Allocation = tuple[int, ...]
Objective = Callable[[Allocation], float]
DEFAULT_IMPROVEMENT_THRESHOLD = 1e-3


@dataclass(frozen=True)
class VNDSearchResult:
    """Generic VND solution and search diagnostics."""

    allocation: Allocation
    objective_value: float
    objective_evaluations: int
    cache_hits: int
    accepted_v1_moves: int
    accepted_v2_moves: int
    runtime_seconds: float


@dataclass(frozen=True)
class VNDResult:
    """Global allocation, actual error, bound, and VND diagnostics."""

    allocation: Allocation
    error_probability: float
    error_upper_bound: float
    state_model: StateModel
    objective_evaluations: int
    cache_hits: int
    accepted_v1_moves: int
    accepted_v2_moves: int
    runtime_seconds: float


def vnd_global_discrimination(
    K: int,
    M: int,
    N: int,
    delta_ijm: DeltaTable,
    state_model: str,
    *,
    improvement_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD,
) -> VNDResult:
    """Minimize actual Gram-matrix PGM error using deterministic VND.

    Args:
        K: Number of candidate fields, indexed ``0`` through ``K - 1``.
        M: Number of candidate sensing locations.
        N: Fixed total number of one-qubit sensors.
        delta_ijm: Canonical unwrapped phase differences. Each pair stores
            ``c*(f_i-f_j)`` at all ``M`` sensing locations.
        state_model: ``"product"``, ``"local_ghz"``, or ``"global_ghz"``.
        improvement_threshold: Minimum objective reduction required to accept
            a move. The supplied VND plan specifies the default ``0.001``.

    Returns:
        The final allocation, its actual PGM error, analytical bound, and search
        diagnostics. The search starts near-uniformly and uses exact V1/V2
        redistributions with deterministic first improvement.
    """
    pairs, checked_model = validate_problem(
        K, M, N, delta_ijm, state_model
    )

    def objective(allocation: Allocation) -> float:
        return pgm_error_probability(
            K, M, allocation, delta_ijm, checked_model, pairs
        )

    search = vnd_sensor_allocation(
        objective,
        M,
        N,
        improvement_threshold=improvement_threshold,
    )
    return VNDResult(
        allocation=search.allocation,
        error_probability=search.objective_value,
        error_upper_bound=pgm_error_upper_bound(
            K,
            M,
            search.allocation,
            delta_ijm,
            checked_model,
            pairs,
        ),
        state_model=checked_model,
        objective_evaluations=search.objective_evaluations,
        cache_hits=search.cache_hits,
        accepted_v1_moves=search.accepted_v1_moves,
        accepted_v2_moves=search.accepted_v2_moves,
        runtime_seconds=search.runtime_seconds,
    )


def vnd_sensor_allocation(
    objective: Objective,
    M: int,
    N: int,
    *,
    improvement_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD,
) -> VNDSearchResult:
    """Run generic two- and three-location variable-neighborhood descent.

    The initial allocation is near-uniform, with remainder sensors assigned to
    the lowest-indexed locations. V1 and V2 visit location subsets
    lexicographically. Within each subset, redistributions are ordered by local
    L1 distance and then lexicographically. The first move meeting the threshold
    is accepted, after which the search restarts from V1.
    """
    _validate_search_inputs(objective, M, N, improvement_threshold)
    started = perf_counter()
    allocation = _near_uniform_allocation(M, N)
    cache: dict[Allocation, float] = {}
    objective_evaluations = 0
    cache_hits = 0

    # Cache full allocations because the same candidate can occur in multiple
    # location subsets and again after a later accepted move.
    def evaluate(candidate: Allocation) -> float:
        nonlocal objective_evaluations, cache_hits
        if candidate in cache:
            cache_hits += 1
            return cache[candidate]
        raw_value = objective(candidate)
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, Real)
            or not isfinite(float(raw_value))
        ):
            raise ValueError("objective must return a finite real number")
        value = float(raw_value)
        cache[candidate] = value
        objective_evaluations += 1
        return value

    current_value = evaluate(allocation)
    accepted_v1_moves = 0
    accepted_v2_moves = 0

    while True:
        move = _first_improving_move(
            allocation,
            current_value,
            combinations(range(M), 2),
            improvement_threshold,
            evaluate,
        )
        if move is not None:
            allocation, current_value = move
            accepted_v1_moves += 1
            continue

        move = _first_improving_move(
            allocation,
            current_value,
            combinations(range(M), 3),
            improvement_threshold,
            evaluate,
        )
        if move is not None:
            allocation, current_value = move
            accepted_v2_moves += 1
            continue
        break

    return VNDSearchResult(
        allocation=allocation,
        objective_value=current_value,
        objective_evaluations=objective_evaluations,
        cache_hits=cache_hits,
        accepted_v1_moves=accepted_v1_moves,
        accepted_v2_moves=accepted_v2_moves,
        runtime_seconds=perf_counter() - started,
    )


def _first_improving_move(
    allocation: Allocation,
    current_value: float,
    location_subsets: Iterator[tuple[int, ...]],
    improvement_threshold: float,
    evaluate: Callable[[Allocation], float],
) -> tuple[Allocation, float] | None:
    """Return the first threshold-improving move in deterministic order."""
    for locations in location_subsets:
        current_local = tuple(allocation[m] for m in locations)
        local_total = sum(current_local)
        redistributions = sorted(
            (
                candidate
                for candidate in _weak_compositions(
                    local_total, len(locations)
                )
                if candidate != current_local
            ),
            key=lambda candidate: (
                sum(
                    abs(candidate[index] - current_local[index])
                    for index in range(len(locations))
                ),
                candidate,
            ),
        )
        for local_candidate in redistributions:
            candidate = list(allocation)
            for location, count in zip(locations, local_candidate):
                candidate[location] = count
            candidate_allocation = tuple(candidate)
            candidate_value = evaluate(candidate_allocation)
            if candidate_value <= current_value - improvement_threshold:
                return candidate_allocation, candidate_value
    return None


def _weak_compositions(total: int, parts: int) -> Iterator[Allocation]:
    """Yield nonnegative ``parts``-tuples summing to ``total`` lexicographically."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for remainder in _weak_compositions(total - first, parts - 1):
            yield (first, *remainder)


def _near_uniform_allocation(M: int, N: int) -> Allocation:
    """Return the deterministic near-uniform initial allocation."""
    base, remainder = divmod(N, M)
    return tuple(base + int(location < remainder) for location in range(M))


def _validate_search_inputs(
    objective: Objective,
    M: int,
    N: int,
    improvement_threshold: float,
) -> None:
    """Validate generic VND dimensions, callable, and threshold."""
    if not callable(objective):
        raise ValueError("objective must be callable")
    if isinstance(M, bool) or not isinstance(M, int) or M < 1:
        raise ValueError("M must be a positive integer")
    if isinstance(N, bool) or not isinstance(N, int) or N < 0:
        raise ValueError("N must be a nonnegative integer")
    if (
        isinstance(improvement_threshold, bool)
        or not isinstance(improvement_threshold, Real)
        or not isfinite(float(improvement_threshold))
        or improvement_threshold <= 0.0
    ):
        raise ValueError("improvement_threshold must be positive and finite")
