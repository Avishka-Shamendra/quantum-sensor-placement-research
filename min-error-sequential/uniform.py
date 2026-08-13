"""Uniform-allocation baseline for sequential discrimination."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
class UniformResult:
    """The strategy obtained from a uniform sensor allocation."""

    allocation: tuple[int, ...]
    pair_locations: dict[Pair, int]
    decision_tree: DecisionTree
    error_probability: float


def uniform(
    K: int,
    M: int,
    N: int,
    pijmn: PairErrorTable,
) -> UniformResult:
    """Distribute sensors evenly and build a sequential decision tree."""
    pairs = _validate_inputs(K, M, N, pijmn)

    # Spread the remainder across the first locations after equal division.
    sensors_per_location, remainder = divmod(N, M)
    allocation = tuple(
        sensors_per_location + (1 if m < remainder else 0)
        for m in range(M)
    )

    pair_locations, pair_errors = _compute_pair_errors(
        pairs, M, allocation, pijmn
    )
    tree = _build_tree(tuple(range(K)), pair_locations, pair_errors)
    error = _tree_error(tree, K)

    return UniformResult(
        allocation=allocation,
        pair_locations=pair_locations,
        decision_tree=tree,
        error_probability=error,
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
            # Use the largest stored count supported by this allocation.
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


def _build_tree(
    remaining: tuple[int, ...],
    pair_locations: Mapping[Pair, int],
    pair_errors: Mapping[Pair, float],
) -> DecisionTree:
    """Build a tree by testing the lowest-error remaining pair."""
    if len(remaining) == 1:
        return DecisionLeaf(declared_field=remaining[0])

    # Test the remaining pair with the smallest available error.
    pair = min(combinations(remaining, 2), key=lambda item: pair_errors[item])
    i, j = pair
    return DecisionNode(
        pair=pair,
        location=pair_locations[pair],
        pair_error=pair_errors[pair],
        discard_i=_build_tree(
            tuple(field for field in remaining if field != i),
            pair_locations,
            pair_errors,
        ),
        discard_j=_build_tree(
            tuple(field for field in remaining if field != j),
            pair_locations,
            pair_errors,
        ),
    )


def _tree_error(tree: DecisionTree, K: int) -> float:
    """Find the tree error by treating every field as equally likely."""
    return _subtree_error(tree, tuple(1.0 / K for _ in range(K)))


def _subtree_error(
    tree: DecisionTree,
    weights: tuple[float, ...],
) -> float:
    """Find the error below one tree node."""
    if isinstance(tree, DecisionLeaf):
        return sum(
            weight
            for field, weight in enumerate(weights)
            if field != tree.declared_field
        )

    i, j = tree.pair

    # Pass the probability mass for each possible active field to both branches.
    discard_i_weights = tuple(
        weight
        * _branch_probability(
            field, tree.pair, discarded_field=i, pair_error=tree.pair_error
        )
        for field, weight in enumerate(weights)
    )
    discard_j_weights = tuple(
        weight
        * _branch_probability(
            field, tree.pair, discarded_field=j, pair_error=tree.pair_error
        )
        for field, weight in enumerate(weights)
    )
    return _subtree_error(tree.discard_i, discard_i_weights) + _subtree_error(
        tree.discard_j, discard_j_weights
    )


def _branch_probability(
    active_field: int,
    pair: Pair,
    discarded_field: int,
    pair_error: float,
) -> float:
    """Return the branch probability for one possible active field."""
    if active_field not in pair:
        return 0.5
    if active_field == discarded_field:
        return pair_error
    return 1.0 - pair_error
