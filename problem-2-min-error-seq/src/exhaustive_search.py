"""Exhaustive actual-error optimization over allocations and adaptive trees."""

from functools import lru_cache
from itertools import combinations

from strategy import DecisionLeaf, DecisionNode, OutcomeModel, branch_weights, budget_allocations


def best_tree(model, allocation):
    """Optimize every node-specific pair and location for a fixed allocation."""
    if not any(allocation):
        return 1.0 - 1.0 / model.K, DecisionLeaf(0)

    @lru_cache(maxsize=50_000)
    def solve(remaining, weights):
        model.check_time()
        if len(remaining) == 1:
            field = remaining[0]
            return weights[field], DecisionLeaf(field)
        best_success, best = -1.0, None
        for pair in combinations(remaining, 2):
            i, j = pair
            for location, count in enumerate(allocation):
                if count <= 0:
                    continue
                probabilities = model.probabilities(pair, location, count)
                wi = branch_weights(weights, probabilities, i, keep_i=False)
                wj = branch_weights(weights, probabilities, j, keep_i=True)
                si, ti = solve(tuple(f for f in remaining if f != i), wi)
                sj, tj = solve(tuple(f for f in remaining if f != j), wj)
                # Strict comparison retains the first deterministic tie:
                # lexicographically smaller pair, then smaller location.
                if si + sj > best_success:
                    best_success = si + sj
                    best = DecisionNode(pair, location, model.pair_error(pair, location, count), ti, tj)
        return best_success, best

    try:
        success, tree = solve(tuple(range(model.K)), (1.0 / model.K,) * model.K)
        return 1.0 - success, tree
    finally:
        solve.cache_clear()


def exhaustive_search(K, M, N, phase_gaps, *, time_limit=None):
    """Optimize exact end-to-end error over every allocation and adaptive tree."""
    model = OutcomeModel(K, M, N, phase_gaps, time_limit=time_limit)
    best_error, best_allocation, best = float("inf"), None, None
    for allocation in budget_allocations(N, M):
        error, tree = best_tree(model, allocation)
        if error < best_error:
            best_error, best_allocation, best = error, allocation, tree
    result = model.result(best_allocation, best)
    if abs(result.actual_error - best_error) > 1e-9:
        raise RuntimeError("Search and independent tree evaluation disagree")
    return result
