"""Fixed-allocation tree builders for comparison methods."""

from itertools import combinations

from strategy import DecisionLeaf, DecisionNode


def pair_score_tree(model, allocation):
    def build(remaining):
        model.check_time()
        if len(remaining) == 1:
            return DecisionLeaf(remaining[0])
        candidates = []
        for pair in combinations(remaining, 2):
            error, location = min((model.pair_error(pair, m, n), m)
                                  for m, n in enumerate(allocation))
            candidates.append((error, pair, location))
        error, pair, location = min(candidates)
        i, j = pair
        return DecisionNode(pair, location, error,
                            build(tuple(f for f in remaining if f != i)),
                            build(tuple(f for f in remaining if f != j)))
    return build(tuple(range(model.K)))


def random_tree(model, allocation, rng):
    """Choose every node uniformly from all valid pair-location tests."""
    def build(remaining):
        model.check_time()
        if len(remaining) == 1:
            return DecisionLeaf(remaining[0])
        pair, location = rng.choice([
            (pair, location)
            for pair in combinations(remaining, 2)
            for location in range(model.M)
        ])
        error = model.pair_error(pair, location, allocation[location])
        i, j = pair
        return DecisionNode(pair, location, error,
                            build(tuple(f for f in remaining if f != i)),
                            build(tuple(f for f in remaining if f != j)))
    return build(tuple(range(model.K)))
