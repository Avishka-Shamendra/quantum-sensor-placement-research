"""Simple P-only greedy allocation and tree construction."""

from itertools import combinations

from strategy import DecisionLeaf, DecisionNode, OutcomeModel


def allocation_score(model, allocation):
    """Average each field pair's smallest P over the sensing locations."""
    model.check_time()
    return sum(
        min(
            model.pair_error(pair, location, allocation[location])
            for location in range(model.M)
        )
        for pair in model.pairs
    ) / len(model.pairs)


def build_greedy_tree(model, allocation):
    """Build a tree using only the smallest current pairwise error at each node."""

    def build(remaining):
        model.check_time()
        if len(remaining) == 1:
            return DecisionLeaf(remaining[0])
        candidates = []
        for pair in combinations(remaining, 2):
            for location, count in enumerate(allocation):
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
    """Allocate by mean minimum P, then build one local minimum-P tree."""
    model = OutcomeModel(K, M, N, phase_gaps, time_limit=time_limit)
    allocation = (0,) * M
    for _ in range(N):
        candidates = []
        for location in range(M):
            trial = tuple(
                count + int(index == location)
                for index, count in enumerate(allocation)
            )
            candidates.append((allocation_score(model, trial), location, trial))
        _, _, allocation = min(candidates)
    tree = build_greedy_tree(model, allocation)
    return model.result(allocation, tree)
