"""Tree-aware alternating heuristic for sequential discrimination."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from numbers import Real
from random import Random


Pair = tuple[int, int]
PairErrorTable = Mapping[Pair, Sequence[Mapping[int, float]]]
# Stores an allocation and compact pair locations and errors in pair order.
_SensorMove = tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...]]


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
class HeuristicResult:
    """The allocation and tree returned by the alternating heuristic."""

    allocation: tuple[int, ...]
    pair_locations: dict[Pair, int]
    decision_tree: DecisionTree
    error_probability: float
    iterations_run: int
    accepted_moves: int
    stopping_reason: str
    max_iterations: int


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
    pijmn: PairErrorTable,
    *,
    beam_width: int = 8,
    max_iterations: int = 25,
    improvement_threshold: float = 1e-3,
    initialization: str = "near_uniform",
    random_seed: int | None = 0,
) -> HeuristicResult:
    """Alternate between beam-search trees and one-sensor allocation moves."""
    pairs = _validate_inputs(K, M, N, pijmn)
    _validate_settings(
        beam_width,
        max_iterations,
        improvement_threshold,
        initialization,
        random_seed,
    )

    # Step 1: Start from a feasible allocation that uses all N sensors.
    allocation = _initial_allocation(
        M,
        N,
        initialization,
        random_seed,
    )

    # Step 2: Find the lowest available error and its location for each pair.
    pair_locations, pair_errors = _compute_pair_errors(
        pairs, M, allocation, pijmn
    )

    # Step 3: Build an adaptive pairwise decision tree using beam search.
    tree = _build_tree_with_beam_search(
        K, pair_locations, pair_errors, beam_width
    )
    error = _tree_error(tree, K)
    move_generator = Random(random_seed)
    iterations_run = 0
    accepted_moves = 0
    stopping_reason = "maximum_iterations"

    # Steps 4-14: Alternate between tree analysis and sensor-allocation moves.
    for _ in range(max_iterations):
        iterations_run += 1

        # Steps 4-6: Propagate mass through the tree and combine node weights.
        pair_weights = _compute_pair_weights(tree, K)

        # Steps 7-9: Keep moves that reduce J, then try them randomly.
        candidates = _improving_sensor_moves(
            allocation,
            pairs,
            M,
            pijmn,
            pair_weights,
            pair_errors,
        )
        move_generator.shuffle(candidates)

        accepted = False
        for (
            candidate_allocation,
            candidate_location_values,
            candidate_error_values,
        ) in candidates:
            # Step 10: Rebuild the full tree before judging a proposed move.
            candidate_locations = dict(zip(pairs, candidate_location_values))
            candidate_errors = dict(zip(pairs, candidate_error_values))
            candidate_tree = _build_tree_with_beam_search(
                K,
                candidate_locations,
                candidate_errors,
                beam_width,
            )
            candidate_error = _tree_error(candidate_tree, K)

            # Step 11: Require the actual error reduction to exceed the threshold.
            if candidate_error < error - improvement_threshold:
                # Step 12: Store the new tree so its weights are recomputed next.
                allocation = candidate_allocation
                pair_locations = candidate_locations
                pair_errors = candidate_errors
                tree = candidate_tree
                error = candidate_error
                accepted = True
                accepted_moves += 1
                break

        # Step 14: Stop when no J-reducing move improves the actual error.
        if not accepted:
            stopping_reason = "no_accepted_move"
            break

        # Step 13: Repeat with the accepted allocation and rebuilt tree.

    # Step 15: Return the final allocation, tree, locations, and error.
    return HeuristicResult(
        allocation=allocation,
        pair_locations=pair_locations,
        decision_tree=tree,
        error_probability=error,
        iterations_run=iterations_run,
        accepted_moves=accepted_moves,
        stopping_reason=stopping_reason,
        max_iterations=max_iterations,
    )


def _validate_settings(
    beam_width: int,
    max_iterations: int,
    improvement_threshold: float,
    initialization: str,
    random_seed: int | None,
) -> None:
    """Validate the controls used by the heuristic search."""
    if isinstance(beam_width, bool) or not isinstance(beam_width, int):
        raise ValueError("beam_width must be a positive integer.")
    if beam_width < 1:
        raise ValueError("beam_width must be a positive integer.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise ValueError("max_iterations must be a nonnegative integer.")
    if max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer.")
    if (
        isinstance(improvement_threshold, bool)
        or not isinstance(improvement_threshold, Real)
        or not isfinite(improvement_threshold)
        or improvement_threshold < 0.0
    ):
        raise ValueError("improvement_threshold must be finite and nonnegative.")
    if initialization not in {"near_uniform", "random"}:
        raise ValueError("initialization must be 'near_uniform' or 'random'.")
    if random_seed is not None and (
        isinstance(random_seed, bool) or not isinstance(random_seed, int)
    ):
        raise ValueError("random_seed must be an integer or None.")


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
        for location, error_table in enumerate(location_tables):
            if not isinstance(error_table, Mapping):
                raise ValueError(f"pijmn[{pair}][{location}] must be a mapping.")

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
    """Find the best available location and error for every field pair."""
    pair_locations: dict[Pair, int] = {}
    pair_errors: dict[Pair, float] = {}

    for pair in pairs:
        best_location = 0
        best_error = 0.5

        for location in range(M):
            available_counts = (
                count
                for count in pijmn[pair][location]
                if count <= allocation[location]
            )
            largest_available = max(available_counts, default=None)
            location_error = (
                0.5
                if largest_available is None
                else float(pijmn[pair][location][largest_available])
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
    beam_width: int,
) -> DecisionTree:
    """Build a decision tree while retaining only the best partial trees."""
    # The root starts with every field still under consideration.
    beam: list[_PartialTree] = [_PendingNode(remaining=tuple(range(K)))]

    while True:
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
                    _partial_tree_score(partial_tree, K),
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
    return min(completed, key=lambda candidate: _tree_error(candidate, K))


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


def _partial_tree_score(tree: _PartialTree, K: int) -> float:
    """Sum weighted errors for tests already selected in a partial tree."""
    return _partial_subtree_score(tree, tuple(1.0 / K for _ in range(K)))


def _partial_subtree_score(
    tree: _PartialTree,
    mass: tuple[float, ...],
) -> float:
    """Find the accumulated weighted-error score below one partial node."""
    if isinstance(tree, (DecisionLeaf, _PendingNode)):
        return 0.0

    i, j = tree.pair
    node_score = (mass[i] + mass[j]) * tree.pair_error

    # The partial-tree score uses the same q-mass rule as pair weighting.
    discard_i_mass = tuple(
        0.0
        if field == i
        else (
            value * (1.0 - tree.pair_error)
            if field == j
            else value * 0.5
        )
        for field, value in enumerate(mass)
    )
    discard_j_mass = tuple(
        0.0
        if field == j
        else (
            value * (1.0 - tree.pair_error)
            if field == i
            else value * 0.5
        )
        for field, value in enumerate(mass)
    )
    return (
        node_score
        + _partial_subtree_score(tree.discard_i, discard_i_mass)
        + _partial_subtree_score(tree.discard_j, discard_j_mass)
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


def _improving_sensor_moves(
    allocation: tuple[int, ...],
    pairs: tuple[Pair, ...],
    M: int,
    pijmn: PairErrorTable,
    pair_weights: Mapping[Pair, float],
    current_pair_errors: Mapping[Pair, float],
) -> list[_SensorMove]:
    """Return all one-sensor moves that reduce the fixed objective J."""
    improving: list[_SensorMove] = []
    current_objective = sum(
        pair_weights.get(pair, 0.0) * current_pair_errors[pair]
        for pair in pairs
    )

    # Step 8: Move one sensor from each nonempty source to every destination.
    for source in range(M):
        if allocation[source] == 0:
            continue
        for destination in range(M):
            if destination == source:
                continue

            candidate = list(allocation)
            candidate[source] -= 1
            candidate[destination] += 1
            candidate_allocation = tuple(candidate)
            candidate_locations, candidate_errors = _compute_pair_errors(
                pairs, M, candidate_allocation, pijmn
            )
            # Step 7: Hold the weights fixed and calculate J for this move.
            objective = sum(
                pair_weights.get(pair, 0.0) * candidate_errors[pair]
                for pair in pairs
            )
            if objective < current_objective:
                improving.append(
                    (
                        candidate_allocation,
                        tuple(candidate_locations[pair] for pair in pairs),
                        tuple(candidate_errors[pair] for pair in pairs),
                    )
                )

    return improving


def _compute_pair_weights(tree: DecisionTree, K: int) -> dict[Pair, float]:
    """Combine the probability-based importance of repeated pair tests."""
    pair_weights: defaultdict[Pair, float] = defaultdict(float)

    # Step 4: At the root, every field has probability mass 1/K.
    initial_mass = tuple(1.0 / K for _ in range(K))

    def visit(node: DecisionTree, mass: tuple[float, ...]) -> None:
        if isinstance(node, DecisionLeaf):
            return

        i, j = node.pair

        # Step 5: The node weight is the current mass of fields i and j.
        # Step 6: Adding to the same key combines repeated tests of this pair.
        pair_weights[node.pair] += mass[i] + mass[j]

        # For discard_i, field i receives zero mass, field j survives with
        # factor (1 - p_ij), and every field outside the pair receives 1/2.
        discard_i_mass = tuple(
            0.0
            if field == i
            else (
                value * (1.0 - node.pair_error)
                if field == j
                else value * 0.5
            )
            for field, value in enumerate(mass)
        )
        # The discard_j branch applies the same rule with i and j reversed.
        discard_j_mass = tuple(
            0.0
            if field == j
            else (
                value * (1.0 - node.pair_error)
                if field == i
                else value * 0.5
            )
            for field, value in enumerate(mass)
        )
        # Continue until every branch reaches a one-field leaf.
        visit(node.discard_i, discard_i_mass)
        visit(node.discard_j, discard_j_mass)

    visit(tree, initial_mass)
    return dict(pair_weights)


def _tree_error(tree: DecisionTree, K: int) -> float:
    """Find the final error when all fields have equal prior probability."""
    return _subtree_error(tree, tuple(1.0 / K for _ in range(K)))


def _subtree_error(
    tree: DecisionTree,
    weights: tuple[float, ...],
) -> float:
    """Find the error below one decision-tree node."""
    if isinstance(tree, DecisionLeaf):
        return sum(
            weight
            for field, weight in enumerate(weights)
            if field != tree.declared_field
        )

    discard_i_weights = _branch_weights(
        weights,
        tree.pair,
        discarded_field=tree.pair[0],
        pair_error=tree.pair_error,
    )
    discard_j_weights = _branch_weights(
        weights,
        tree.pair,
        discarded_field=tree.pair[1],
        pair_error=tree.pair_error,
    )
    return _subtree_error(tree.discard_i, discard_i_weights) + _subtree_error(
        tree.discard_j, discard_j_weights
    )


def _branch_weights(
    weights: tuple[float, ...],
    pair: Pair,
    *,
    discarded_field: int,
    pair_error: float,
) -> tuple[float, ...]:
    """Pass active-field probability masses into one test branch."""
    return tuple(
        weight
        * _branch_probability(
            field,
            pair,
            discarded_field=discarded_field,
            pair_error=pair_error,
        )
        for field, weight in enumerate(weights)
    )


def _branch_probability(
    active_field: int,
    pair: Pair,
    *,
    discarded_field: int,
    pair_error: float,
) -> float:
    """Return the branch probability conditioned on the active field."""
    if active_field not in pair:
        return 0.5
    if active_field == discarded_field:
        return pair_error
    return 1.0 - pair_error
