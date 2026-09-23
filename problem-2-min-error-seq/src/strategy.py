"""Shared strategy types, input checks, and cached physical outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from time import monotonic

from actual_error import actual_node_probabilities, actual_tree_error
from input_data import pair_error_probability, principal_phase


@dataclass(frozen=True)
class DecisionLeaf:
    declared_field: int


@dataclass(frozen=True)
class DecisionNode:
    pair: tuple[int, int]
    location: int
    pair_error: float
    discard_i: DecisionTree
    discard_j: DecisionTree


DecisionTree = DecisionLeaf | DecisionNode


@dataclass(frozen=True)
class StrategyResult:
    allocation: tuple[int, ...]
    decision_tree: DecisionTree
    actual_error: float


def tree_to_dict(tree):
    if hasattr(tree, "declared_field"):
        return {"declared_field": tree.declared_field}
    return {"pair": list(tree.pair), "location": tree.location, "pair_error": tree.pair_error,
            "discard_i": tree_to_dict(tree.discard_i), "discard_j": tree_to_dict(tree.discard_j)}


class SearchTimeout(TimeoutError):
    """The configured search time was exceeded; no optimum is reported."""


class OutcomeModel:
    def __init__(self, K, M, N, phase_gaps, *, time_limit=None):
        for name, value, minimum in (("K", K, 2), ("M", M, 1), ("N", N, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if time_limit is not None and (not isfinite(time_limit) or time_limit <= 0):
            raise ValueError("time_limit must be positive and finite")
        self.K, self.M, self.N = K, M, N
        self.pairs = tuple(combinations(range(K), 2))
        if set(phase_gaps) != set(self.pairs):
            raise ValueError("Provide every canonical field pair")
        self.phases = {p: tuple(principal_phase(v) for v in phase_gaps[p]) for p in self.pairs}
        if any(len(values) != M for values in self.phases.values()):
            raise ValueError("Each phase pair must contain M locations")
        for m in range(M):
            phases = [0.0] + [-self.phases[(0, i)][m] for i in range(1, K)]
            for i, j in self.pairs:
                if abs(principal_phase(phases[i] - phases[j] - self.phases[(i, j)][m])) > 1e-9:
                    raise ValueError("Pair gaps must come from a consistent set of field phases")
        self.deadline = None if time_limit is None else monotonic() + time_limit
        self._outcomes = {}

    def check_time(self):
        if self.deadline is not None and monotonic() > self.deadline:
            raise SearchTimeout("Search time limit exceeded")

    def probabilities(self, pair, location, count):
        """P(keep i | true field) for every field, cached per test."""
        self.check_time()
        key = (pair, location, count)
        if key not in self._outcomes:
            self._outcomes[key] = tuple(
                actual_node_probabilities(pair, location, count, field, self.phases).keep_i
                for field in range(self.K)
            )
        return self._outcomes[key]

    def pair_error(self, pair, location, count):
        return pair_error_probability(self.phases[pair][location], count)

    def result(self, allocation, tree):
        if len(allocation) != self.M or sum(allocation) != self.N or any(n < 0 for n in allocation):
            raise ValueError("Allocation must use the given budget")
        error = actual_tree_error(tree, self.K, self.phases, allocation)
        self.check_time()
        return StrategyResult(tuple(allocation), tree, min(1.0, max(0.0, error)))


def budget_allocations(total, locations):
    if locations == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for tail in budget_allocations(total - first, locations - 1):
                yield (first, *tail)


def branch_weights(weights, probabilities, discarded, *, keep_i):
    """Unnormalized surviving-field masses; discarded truth is already an error."""
    return tuple(0.0 if field == discarded else weight * (p if keep_i else 1 - p)
                 for field, (weight, p) in enumerate(zip(weights, probabilities)))
