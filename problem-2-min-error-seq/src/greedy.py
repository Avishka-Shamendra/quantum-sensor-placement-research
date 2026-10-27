"""Simple P-only greedy allocation and tree construction."""

from itertools import combinations

from strategy import DecisionLeaf, DecisionNode, OutcomeModel


def location_gain(model, allocation, location):
    """Return the mean pair-error reduction from one sensor at a location."""
    model.check_time()
    count = allocation[location]
    return sum(
        model.pair_error(pair, location, count)
        - model.pair_error(pair, location, count + 1)
        for pair in model.pairs
    ) / len(model.pairs)


def greedy_allocation(model):
    """Allocate sensors by the largest mean pair-error gain."""
    allocation = (0,) * model.M
    for _ in range(model.N):
        location = max(
            range(model.M),
            key=lambda candidate: (
                location_gain(model, allocation, candidate), -candidate
            ),
        )
        allocation = tuple(
            count + int(index == location)
            for index, count in enumerate(allocation)
        )
    return allocation


def build_greedy_tree(model, allocation):
    """Build a tree using only the smallest current pairwise error at each node."""

    if not any(allocation):
        return DecisionLeaf(0)

    def build(remaining):
        model.check_time()
        if len(remaining) == 1:
            return DecisionLeaf(remaining[0])
        candidates = []
        for pair in combinations(remaining, 2):
            for location, count in enumerate(allocation):
                if count <= 0:
                    continue
                candidates.append(
                    (model.pair_error(pair, location, count), pair, location)
                )

        # Tuple ordering implements deterministic ties: P, then pair, then location.
        pair_error, pair, location = min(candidates)
        i, j = pair
        discard_i = build(tuple(field for field in remaining if field != i))
        discard_j = build(tuple(field for field in remaining if field != j))
        return DecisionNode(pair, location, pair_error, discard_i, discard_j)

    return build(tuple(range(model.K)))


def greedy(K, M, N, phase_gaps, *, time_limit=None):
    """Allocate by the largest mean P gain, then build one minimum-P tree."""
    model = OutcomeModel(K, M, N, phase_gaps, time_limit=time_limit)
    allocation = greedy_allocation(model)
    tree = build_greedy_tree(model, allocation)
    return model.result(allocation, tree)
