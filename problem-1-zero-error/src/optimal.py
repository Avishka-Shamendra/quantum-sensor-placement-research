"""Exact sequential allocation by dynamic programming over covered pairs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional


Threshold = Optional[int]


def optimal_perfect_pairwise_sensor_placement(
    F: int,
    M: int,
    Nijm: Mapping[tuple[int, int], Sequence[Threshold]],
) -> tuple[int, list[int], list[list[int | None]]]:
    """Return (minimum_total, allocation, pair_locations).

    Nijm[(i, j)][m] is a positive sensor threshold, or None for an unusable
    location. Indices start at zero; each pair must have one usable location.
    The symmetric pair_locations matrix identifies a covering location and
    has None on its diagonal. Invalid or infeasible inputs raise ValueError.
    For P=F(F-1)/2 pairs, worst-case space is O(2^P) and time O(M*P*2^P)."""
    if isinstance(F, bool) or not isinstance(F, int) or F < 2:
        raise ValueError("F must be an integer of at least 2.")
    if isinstance(M, bool) or not isinstance(M, int) or M < 1:
        raise ValueError("M must be a positive integer.")
    if not isinstance(Nijm, Mapping):
        raise ValueError("Nijm must be a mapping keyed by function-pair tuples.")

    pairs = tuple((i, j) for i in range(F) for j in range(i + 1, F))
    required_keys = set(pairs)
    if set(Nijm) != required_keys:
        raise ValueError(
            "Nijm must contain exactly one (i, j) key with 0 <= i < j < F "
            "for every function pair."
        )

    for pair in pairs:
        thresholds = Nijm[pair]
        if len(thresholds) != M:
            raise ValueError("Every Nijm[(i, j)] sequence must have length M.")
        for threshold in thresholds:
            if threshold is None:
                continue
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, int)
                or threshold <= 0
            ):
                raise ValueError(
                    "Each Nijm threshold must be a positive integer."
                )
        if all(threshold is None for threshold in thresholds):
            raise ValueError(
                f"Pair {pair} cannot be distinguished at any sensing location."
            )

    pair_bits = {pair: 1 << index for index, pair in enumerate(pairs)}
    all_pairs_mask = (1 << len(pairs)) - 1

    # Dynamic program state counts: Counts_m = {0} union {N_ijm : (i, j) in P}.
    # Counts between thresholds have identical coverage and are unnecessary.
    location_options: list[list[tuple[int, int]]] = []
    for m in range(M):
        counts = sorted(
            {0}
            | {
                threshold
                for pair in pairs
                if (threshold := Nijm[pair][m]) is not None
            }
        )
        options = []
        for count in counts:
            covered_mask = 0
            for pair in pairs:
                if Nijm[pair][m] is not None and count >= Nijm[pair][m]:
                    covered_mask |= pair_bits[pair]
            options.append((count, covered_mask))
        location_options.append(options)

    # Map each covered-pair mask to its lowest cost after each location.
    costs = {0: 0}
    parent_layers: list[dict[int, tuple[int, int]]] = []

    for options in location_options:
        next_costs: dict[int, int] = {}
        parents: dict[int, tuple[int, int]] = {}
        for covered_mask, cost in costs.items():
            for count, local_mask in options:
                new_mask = covered_mask | local_mask
                new_cost = cost + count
                if new_mask not in next_costs or new_cost < next_costs[new_mask]:
                    next_costs[new_mask] = new_cost
                    parents[new_mask] = (covered_mask, count)
        costs = next_costs
        parent_layers.append(parents)

    if all_pairs_mask not in costs:
        raise ValueError("No feasible allocation covers every function pair.")

    allocation = [0] * M
    mask = all_pairs_mask
    for m in range(M - 1, -1, -1):
        previous_mask, count = parent_layers[m][mask]
        allocation[m] = count
        mask = previous_mask

    pair_locations: list[list[int | None]] = [
        [None for _ in range(F)] for _ in range(F)
    ]
    for i, j in pairs:
        location = next(
            m
            for m in range(M)
            if Nijm[(i, j)][m] is not None
            and allocation[m] >= Nijm[(i, j)][m]
        )
        pair_locations[i][j] = location
        pair_locations[j][i] = location

    return costs[all_pairs_mask], allocation, pair_locations
