"""Allocation + tree heuristic for sequential discrimination."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from numbers import Real
from random import Random

from strategy import OutcomeModel, StrategyResult
from input_data import pair_error_probability

Pair = tuple[int, int]
DEFAULT_MOVE_SIZES = (1, 2, 3)


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
class _PendingNode:
    """A decision-tree position that has not yet selected a pair test."""

    remaining: tuple[int, ...]


@dataclass(frozen=True)
class _PartialNode:
    """A selected pair test whose children may still be incomplete."""

    pair: Pair
    location: int
    pair_error: float
    discard_i: _PartialTree
    discard_j: _PartialTree


_PartialTree = DecisionLeaf | _PendingNode | _PartialNode


def heuristic(
    K: int,
    M: int,
    N: int,
    phase_gaps,
    *,
    time_limit=None,
    beam_width: int = 3,
    max_iterations: int | None = 30,
    improvement_threshold: float = 0.0,
    initialization: str = "near_uniform",
    random_seed: int | None = 0,
    include_greedy_start: bool = False,
    move_sizes: tuple[int, ...] = DEFAULT_MOVE_SIZES,
) -> StrategyResult:
    """Use pairwise scores for allocation/tree construction; report actual error."""
    model = OutcomeModel(K, M, N, phase_gaps, time_limit=time_limit)
    pairs = model.pairs
    _validate_settings(
        beam_width,
        max_iterations,
        improvement_threshold,
        initialization,
        random_seed,
        include_greedy_start,
        move_sizes,
    )

    current_allocation = _initial_allocation(M, N, initialization, random_seed)
    _, current_pair_errors = _compute_pair_errors(
        pairs, M, current_allocation, model.phases
    )
    current_score = _allocation_score(current_pair_errors)

    iterations_run = 0
    accepted_moves = 0
    stopping_reason = "vnd_converged"
    neighborhood_index = 0

    # Variable Neighborhood Descent. At each level L, select the best feasible
    # L-sensor transfer. Accept any strict AS improvement and return to L=1.
    while neighborhood_index < len(move_sizes):
        model.check_time()
        if max_iterations is not None and iterations_run >= max_iterations:
            stopping_reason = "maximum_iterations"
            break
        iterations_run += 1
        move_size = move_sizes[neighborhood_index]
        candidate = _best_allocation_neighbor(
            current_allocation,
            pairs,
            M,
            model.phases,
            current_score,
            move_size,
            improvement_threshold,
        )
        if candidate is None:
            neighborhood_index += 1
            continue

        current_allocation, current_score = candidate
        accepted_moves += 1
        neighborhood_index = 0

    pair_locations, pair_errors = _compute_pair_errors(
        pairs,
        M,
        current_allocation,
        model.phases,
    )
    pair_outcomes = {
        pair: model.probabilities(
            pair,
            pair_locations[pair],
            current_allocation[pair_locations[pair]],
        )
        for pair in pairs
    }
    tree = _build_tree_with_beam_search(
        K,
        pair_locations,
        pair_errors,
        pair_outcomes,
        beam_width,
        check_time=model.check_time,
    )
    return model.result(current_allocation, tree)




def _validate_settings(
    beam_width: int,
    max_iterations: int | None,
    improvement_threshold: float,
    initialization: str,
    random_seed: int | None,
    include_greedy_start: bool,
    move_sizes: tuple[int, ...],
) -> None:
    """Validate the controls used by the heuristic search."""
    if isinstance(beam_width, bool) or not isinstance(beam_width, int):
        raise ValueError("beam_width must be a positive integer.")
    if beam_width < 1:
        raise ValueError("beam_width must be a positive integer.")
    if max_iterations is not None:
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise ValueError("max_iterations must be a nonnegative integer or None.")
        if max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative integer or None.")
    if (
        isinstance(improvement_threshold, bool)
        or not isinstance(improvement_threshold, Real)
        or not isfinite(improvement_threshold)
        or improvement_threshold < 0.0
    ):
        raise ValueError("improvement_threshold must be finite and nonnegative.")
    if initialization not in {"near_uniform", "random"}:
        raise ValueError(
            "initialization must be 'near_uniform' or 'random'."
        )
    if random_seed is not None and (
        isinstance(random_seed, bool) or not isinstance(random_seed, int)
    ):
        raise ValueError("random_seed must be an integer or None.")
    if not isinstance(include_greedy_start, bool):
        raise ValueError("include_greedy_start must be a boolean.")
    if include_greedy_start:
        raise ValueError(
            "include_greedy_start is not supported by the allocation + tree heuristic."
        )
    if not move_sizes:
        raise ValueError("move_sizes must not be empty.")
    if any(
        isinstance(move_size, bool)
        or not isinstance(move_size, int)
        or move_size < 1
        for move_size in move_sizes
    ):
        raise ValueError("move_sizes must contain positive integers.")
    if len(set(move_sizes)) != len(move_sizes):
        raise ValueError("move_sizes must not contain duplicate values.")


def _allocation_score(pair_errors: Mapping[Pair, float]) -> float:
    """Average the best pairwise errors induced by an allocation."""
    return sum(pair_errors.values()) / len(pair_errors)


def _best_allocation_neighbor(
    allocation: tuple[int, ...],
    pairs: tuple[Pair, ...],
    M: int,
    phase_gaps: Mapping[Pair, tuple[float, ...]],
    current_score: float,
    move_size: int,
    improvement_threshold: float,
) -> tuple[tuple[int, ...], float] | None:
    """Return the best improving d-sensor allocation move."""
    best_allocation: tuple[int, ...] | None = None
    best_score = current_score

    for source in range(M):
        if allocation[source] < move_size:
            continue
        for destination in range(M):
            if destination == source:
                continue
            candidate = list(allocation)
            candidate[source] -= move_size
            candidate[destination] += move_size
            candidate_allocation = tuple(candidate)
            _, candidate_errors = _compute_pair_errors(
                pairs,
                M,
                candidate_allocation,
                phase_gaps,
            )
            candidate_score = _allocation_score(candidate_errors)
            if candidate_score < best_score - improvement_threshold:
                best_allocation = candidate_allocation
                best_score = candidate_score

    if best_allocation is None:
        return None
    return best_allocation, best_score


def _initial_allocation(
    M: int,
    N: int,
    initialization: str,
    random_seed: int | None,
) -> tuple[int, ...]:
    """Create a seeded starting allocation that uses every sensor."""
    generator = Random(random_seed)

    if initialization == "random":
        allocation = [0 for _ in range(M)]
        for _ in range(N):
            allocation[generator.randrange(M)] += 1
        return tuple(allocation)

    sensors_per_location, remainder = divmod(N, M)
    allocation = [sensors_per_location for _ in range(M)]
    remainder_locations = generator.sample(range(M), remainder)
    for location in remainder_locations:
        allocation[location] += 1
    return tuple(allocation)




def _compute_pair_errors(
    pairs: tuple[Pair, ...],
    M: int,
    allocation: tuple[int, ...],
    phase_gaps: Mapping[Pair, tuple[float, ...]],
) -> tuple[dict[Pair, int], dict[Pair, float]]:
    """Find the best available location and error for every field pair."""
    pair_locations: dict[Pair, int] = {}
    pair_errors: dict[Pair, float] = {}

    for pair in pairs:
        best_location = 0
        best_error = 0.5

        for location in range(M):
            location_error = pair_error_probability(
                phase_gaps[pair][location], allocation[location]
            )

            if location_error < best_error:
                best_location = location
                best_error = location_error

        pair_locations[pair] = best_location
        pair_errors[pair] = best_error

    return pair_locations, pair_errors


def _build_tree_with_beam_search(
    K: int,
    pair_locations: Mapping[Pair, int],
    pair_errors: Mapping[Pair, float],
    pair_outcomes: Mapping[Pair, tuple[float, ...]],
    beam_width: int,
    *, check_time=lambda: None,
) -> DecisionTree:
    """Build a decision tree while retaining only the best partial trees."""
    # The root starts with every field still under consideration.
    beam: list[_PartialTree] = [_PendingNode(remaining=tuple(range(K)))]

    while True:
        check_time()
        expanded: list[_PartialTree] = []

        # Expand one unfinished node in each retained partial tree.
        for partial_tree in beam:
            alternatives, changed = _expand_first_pending(
                partial_tree, pair_locations, pair_errors
            )
            if changed:
                expanded.extend(alternatives)

        # No unfinished nodes means every tree in the beam is complete.
        if not expanded:
            break

        # Score only tests already selected in each partial tree.
        ranked = sorted(
            (
                (
                    _partial_tree_score(partial_tree, K, pair_outcomes),
                    _partial_tree_signature(partial_tree),
                    partial_tree,
                )
                for partial_tree in expanded
            ),
            key=lambda item: (item[0], item[1]),
        )
        # Keep only the B lowest-score partial trees from this expansion.
        beam = [item[2] for item in ranked[:beam_width]]

    # Convert the retained complete partial trees to result trees.
    completed = [_finalize_complete_tree(partial_tree) for partial_tree in beam]
    return min(
        completed,
        key=lambda candidate: _tree_score(candidate, K, pair_outcomes),
    )


def _expand_first_pending(
    tree: _PartialTree,
    pair_locations: Mapping[Pair, int],
    pair_errors: Mapping[Pair, float],
) -> tuple[list[_PartialTree], bool]:
    """Expand the first unfinished position using every available pair test."""
    if isinstance(tree, DecisionLeaf):
        return [], False

    if isinstance(tree, _PendingNode):
        alternatives: list[_PartialTree] = []

        # Try every pair of fields that remains at this unfinished node.
        for pair in combinations(tree.remaining, 2):
            i, j = pair
            error = pair_errors[pair]
            alternatives.append(
                _PartialNode(
                    pair=pair,
                    location=pair_locations[pair],
                    pair_error=error,
                    # This child is reached when the test eliminates field i.
                    discard_i=_next_partial_child(
                        tuple(field for field in tree.remaining if field != i)
                    ),
                    # This child is reached when the test eliminates field j.
                    discard_j=_next_partial_child(
                        tuple(field for field in tree.remaining if field != j)
                    ),
                )
            )
        return alternatives, True

    # Search the left child first to make expansion order deterministic.
    left_alternatives, left_changed = _expand_first_pending(
        tree.discard_i, pair_locations, pair_errors
    )
    if left_changed:
        return [
            _PartialNode(
                pair=tree.pair,
                location=tree.location,
                pair_error=tree.pair_error,
                discard_i=alternative,
                discard_j=tree.discard_j,
            )
            for alternative in left_alternatives
        ], True

    # Expand the right child after the left subtree is complete.
    right_alternatives, right_changed = _expand_first_pending(
        tree.discard_j, pair_locations, pair_errors
    )
    if right_changed:
        return [
            _PartialNode(
                pair=tree.pair,
                location=tree.location,
                pair_error=tree.pair_error,
                discard_i=tree.discard_i,
                discard_j=alternative,
            )
            for alternative in right_alternatives
        ], True

    return [], False


def _next_partial_child(
    remaining: tuple[int, ...],
) -> _PartialTree:
    """Create a finished leaf or another unfinished tree position."""
    if len(remaining) == 1:
        return DecisionLeaf(declared_field=remaining[0])
    return _PendingNode(remaining=remaining)


def _finalize_complete_tree(tree: _PartialTree) -> DecisionTree:
    """Convert a fully expanded partial tree into a decision tree."""
    if isinstance(tree, DecisionLeaf):
        return tree
    if isinstance(tree, _PendingNode):
        raise RuntimeError("cannot finalize an unfinished decision tree")
    return DecisionNode(
        pair=tree.pair,
        location=tree.location,
        pair_error=tree.pair_error,
        discard_i=_finalize_complete_tree(tree.discard_i),
        discard_j=_finalize_complete_tree(tree.discard_j),
    )


def _partial_tree_score(
    tree: _PartialTree,
    K: int,
    pair_outcomes: Mapping[Pair, tuple[float, ...]],
) -> float:
    """Score selected tests using their exact P/Q branch probabilities."""
    return _partial_subtree_score(
        tree,
        tuple(1.0 / K for _ in range(K)),
        pair_outcomes,
    )


def _partial_subtree_score(
    tree: _PartialTree,
    mass: tuple[float, ...],
    pair_outcomes: Mapping[Pair, tuple[float, ...]],
) -> float:
    """Find the accumulated weighted-error score below one partial node."""
    if isinstance(tree, (DecisionLeaf, _PendingNode)):
        return 0.0

    i, j = tree.pair
    keep_i = pair_outcomes[tree.pair]
    node_score = mass[i] * (1.0 - keep_i[i]) + mass[j] * keep_i[j]

    discard_i_mass = tuple(
        0.0 if field == i else value * (1.0 - keep_i[field])
        for field, value in enumerate(mass)
    )
    discard_j_mass = tuple(
        0.0 if field == j else value * keep_i[field]
        for field, value in enumerate(mass)
    )
    return (
        node_score
        + _partial_subtree_score(tree.discard_i, discard_i_mass, pair_outcomes)
        + _partial_subtree_score(tree.discard_j, discard_j_mass, pair_outcomes)
    )


def _partial_tree_signature(tree: _PartialTree) -> tuple[object, ...]:
    """Return a stable ordering key for deterministic beam selection."""
    if isinstance(tree, DecisionLeaf):
        return ("leaf", tree.declared_field)
    if isinstance(tree, _PendingNode):
        return ("pending", tree.remaining)
    return (
        "node",
        tree.pair,
        _partial_tree_signature(tree.discard_i),
        _partial_tree_signature(tree.discard_j),
    )


def _tree_score(
    tree: DecisionTree,
    K: int,
    pair_outcomes: Mapping[Pair, tuple[float, ...]],
) -> float:
    """Find the exact P/Q error when all fields have equal prior probability."""
    return _subtree_score(
        tree,
        tuple(1.0 / K for _ in range(K)),
        pair_outcomes,
    )


def _subtree_score(
    tree: DecisionTree,
    weights: tuple[float, ...],
    pair_outcomes: Mapping[Pair, tuple[float, ...]],
) -> float:
    """Find the error below one decision-tree node."""
    if isinstance(tree, DecisionLeaf):
        return sum(
            weight
            for field, weight in enumerate(weights)
            if field != tree.declared_field
        )

    keep_i = pair_outcomes[tree.pair]
    discard_i_weights = tuple(
        weight * (1.0 - probability)
        for weight, probability in zip(weights, keep_i)
    )
    discard_j_weights = tuple(
        weight * probability
        for weight, probability in zip(weights, keep_i)
    )
    return _subtree_score(
        tree.discard_i, discard_i_weights, pair_outcomes
    ) + _subtree_score(
        tree.discard_j, discard_j_weights, pair_outcomes
    )
