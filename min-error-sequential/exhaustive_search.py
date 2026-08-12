"""Exhaustive optimization for fixed-budget sequential discrimination."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from numbers import Real


Pair = tuple[int, int]
PairErrorTable = Mapping[Pair, Sequence[Mapping[int, float]]]


@dataclass(frozen=True)
class DecisionLeaf:
    """A leaf declaring the remaining candidate field."""

    declared_field: int


@dataclass(frozen=True)
class DecisionNode:
    """A pairwise test with one subtree for each discarded field."""

    pair: Pair
    location: int
    pair_error: float
    discard_i: DecisionTree
    discard_j: DecisionTree


DecisionTree = DecisionLeaf | DecisionNode


@dataclass(frozen=True)
class ExhaustiveSearchResult:
    """The optimal sequential discrimination strategy."""

    allocation: tuple[int, ...]
    pair_locations: dict[Pair, int]
    decision_tree: DecisionTree
    minimum_error: float


def exhaustive_search(
    K: int,
    M: int,
    N: int,
    pijmn: PairErrorTable,
) -> ExhaustiveSearchResult:
    """Find the optimal allocation and decision tree for a sensor budget."""
    pairs = _validate_inputs(K, M, N, pijmn)

    best_error = float("inf")
    best_allocation: tuple[int, ...] | None = None
    best_pair_locations: dict[Pair, int] | None = None
    best_tree: DecisionTree | None = None

    for allocation in _budget_allocations(N, M):
        pair_locations, pair_errors = _compute_pair_errors(
            pairs, M, allocation, pijmn
        )
        initial_weights = tuple(1.0 / K for _ in range(K))
        error, tree = _best_decision_tree(
            tuple(range(K)), initial_weights, pair_locations, pair_errors
        )

        if error < best_error:
            best_error = error
            best_allocation = allocation
            best_pair_locations = pair_locations
            best_tree = tree

    assert best_allocation is not None
    assert best_pair_locations is not None
    assert best_tree is not None
    return ExhaustiveSearchResult(
        allocation=best_allocation,
        pair_locations=best_pair_locations,
        decision_tree=best_tree,
        minimum_error=best_error,
    )


def _validate_inputs(
    K: int,
    M: int,
    N: int,
    pijmn: PairErrorTable,
) -> tuple[Pair, ...]:
    """Validate the problem dimensions, budget, and pairwise errors."""
    if isinstance(K, bool) or not isinstance(K, int) or K < 2:
        raise ValueError("K must be an integer of at least 2.")
    if isinstance(M, bool) or not isinstance(M, int) or M < 1:
        raise ValueError("M must be a positive integer.")
    if isinstance(N, bool) or not isinstance(N, int) or N < 0:
        raise ValueError("N must be a nonnegative integer sensor budget.")
    if not isinstance(pijmn, Mapping):
        raise ValueError("pijmn must be a mapping keyed by field-pair tuples.")

    pairs = tuple((i, j) for i in range(K) for j in range(i + 1, K))
    if set(pijmn) != set(pairs):
        raise ValueError(
            "pijmn must contain exactly every pair (i, j) with "
            "0 <= i < j < K."
        )

    for pair in pairs:
        location_tables = pijmn[pair]
        if len(location_tables) != M:
            raise ValueError(
                f"pijmn[{pair}] must contain exactly M={M} location tables."
            )

        has_feasible_error_value = False
        for m, error_table in enumerate(location_tables):
            if not isinstance(error_table, Mapping):
                raise ValueError(f"pijmn[{pair}][{m}] must be a mapping.")

            for sensor_count, error in error_table.items():
                if (
                    isinstance(sensor_count, bool)
                    or not isinstance(sensor_count, int)
                    or sensor_count < 0
                ):
                    raise ValueError(
                        "Every supplied sensor count must be a nonnegative integer."
                    )
                if sensor_count <= N:
                    has_feasible_error_value = True
                if (
                    isinstance(error, bool)
                    or not isinstance(error, Real)
                    or not isfinite(error)
                    or not 0.0 <= float(error) <= 0.5
                ):
                    raise ValueError(
                        "Every pairwise error probability must lie in [0, 0.5]."
                    )

        if not has_feasible_error_value:
            raise ValueError(
                f"Pair {pair} must have an error value for at least one sensor "
                f"count no greater than the budget N={N}."
            )

    return pairs


def _budget_allocations(total: int, locations: int) -> Iterator[tuple[int, ...]]:
    """Yield all allocations whose entries sum to ``total``."""
    if locations == 1:
        yield (total,)
        return

    for first in range(total + 1):
        for remainder in _budget_allocations(total - first, locations - 1):
            yield (first, *remainder)


def _compute_pair_errors(
    pairs: tuple[Pair, ...],
    M: int,
    allocation: tuple[int, ...],
    pijmn: PairErrorTable,
) -> tuple[dict[Pair, int], dict[Pair, float]]:
    """Determine the best sensing location and error for each pair."""
    pair_locations: dict[Pair, int] = {}
    pair_errors: dict[Pair, float] = {}

    for pair in pairs:
        best_location = 0
        best_error = 0.5

        for m in range(M):
            available_counts = (
                count for count in pijmn[pair][m] if count <= allocation[m]
            )
            largest_available = max(available_counts, default=None)
            location_error = (
                0.5
                if largest_available is None
                else float(pijmn[pair][m][largest_available])
            )

            if location_error < best_error:
                best_location = m
                best_error = location_error

        pair_locations[pair] = best_location
        pair_errors[pair] = best_error

    return pair_locations, pair_errors


def _best_decision_tree(
    remaining: tuple[int, ...],
    weights: tuple[float, ...],
    pair_locations: Mapping[Pair, int],
    pair_errors: Mapping[Pair, float],
) -> tuple[float, DecisionTree]:
    """Find the best decision tree for the fields that are still left."""
    # If only one field is left, choose it as the active field.  The result is
    # wrong whenever any other field is actually active.
    if len(remaining) == 1:
        declared = remaining[0]
        leaf_error = sum(
            weight for field, weight in enumerate(weights) if field != declared
        )
        return leaf_error, DecisionLeaf(declared_field=declared)

    best_error = float("inf")
    best_tree: DecisionTree | None = None

    # Try every pair of fields as the next test.
    for i, j in combinations(remaining, 2):
        pair = (i, j)
        error_probability = pair_errors[pair]

        # Find the chance of reaching the branch where field i is removed.
        discard_i_weights = tuple(
            weight * _branch_probability(
                active_field=field,
                pair=pair,
                discarded_field=i,
                pair_error=error_probability,
            )
            for field, weight in enumerate(weights)
        )
        # Find the chance of reaching the branch where field j is removed.
        discard_j_weights = tuple(
            weight * _branch_probability(
                active_field=field,
                pair=pair,
                discarded_field=j,
                pair_error=error_probability,
            )
            for field, weight in enumerate(weights)
        )

        # Find the best next tests for both possible outcomes.
        error_if_i_discarded, tree_if_i_discarded = _best_decision_tree(
            tuple(field for field in remaining if field != i),
            discard_i_weights,
            pair_locations,
            pair_errors,
        )
        error_if_j_discarded, tree_if_j_discarded = _best_decision_tree(
            tuple(field for field in remaining if field != j),
            discard_j_weights,
            pair_locations,
            pair_errors,
        )
        # Add the errors from the two possible outcomes.
        total_error = error_if_i_discarded + error_if_j_discarded

        # Keep this test if it gives the smallest error found so far.
        if total_error < best_error:
            best_error = total_error
            best_tree = DecisionNode(
                pair=pair,
                location=pair_locations[pair],
                pair_error=error_probability,
                discard_i=tree_if_i_discarded,
                discard_j=tree_if_j_discarded,
            )

    assert best_tree is not None
    return best_error, best_tree


def _branch_probability(
    active_field: int,
    pair: Pair,
    discarded_field: int,
    pair_error: float,
) -> float:
    """Return the branch probability conditioned on the active field."""
    i, j = pair
    if active_field not in pair:
        return 0.5
    if active_field == discarded_field:
        return pair_error
    return 1.0 - pair_error
