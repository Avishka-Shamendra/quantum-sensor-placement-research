"""Fixed-allocation tree builders for comparison methods."""

from itertools import combinations

from strategy import DecisionLeaf, DecisionNode


def pair_score_tree(model, allocation):
    if not any(allocation):
        return DecisionLeaf(0)

    def build(remaining):
        model.check_time()
        if len(remaining) == 1:
            return DecisionLeaf(remaining[0])
        candidates = []
        for pair in combinations(remaining, 2):
            error, location = min((model.pair_error(pair, m, n), m)
                                  for m, n in enumerate(allocation) if n > 0)
            candidates.append((error, pair, location))
        error, pair, location = min(candidates)
        i, j = pair
        return DecisionNode(pair, location, error,
                            build(tuple(f for f in remaining if f != i)),
                            build(tuple(f for f in remaining if f != j)))
    return build(tuple(range(model.K)))


def random_tree(model, allocation, rng):
    """Choose every node uniformly from all valid pair-location tests."""
    if not any(allocation):
        return DecisionLeaf(0)

    def build(remaining):
        model.check_time()
        if len(remaining) == 1:
            return DecisionLeaf(remaining[0])
        pair, location = rng.choice([
            (pair, location)
            for pair in combinations(remaining, 2)
            for location, count in enumerate(allocation) if count > 0
        ])
        error = model.pair_error(pair, location, allocation[location])
        i, j = pair
        return DecisionNode(pair, location, error,
                            build(tuple(f for f in remaining if f != i)),
                            build(tuple(f for f in remaining if f != j)))
    return build(tuple(range(model.K)))
