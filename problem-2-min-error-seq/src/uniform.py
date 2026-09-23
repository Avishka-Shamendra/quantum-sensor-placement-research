"""Near-uniform allocation with a seeded random decision tree."""

from random import Random

from baselines import random_tree
from strategy import OutcomeModel


def uniform(K, M, N, phase_gaps, *, random_seed=0, time_limit=None):
    model = OutcomeModel(K, M, N, phase_gaps, time_limit=time_limit)
    rng = Random(random_seed)
    quotient, remainder = divmod(N, M)
    allocation = [quotient] * M
    for m in rng.sample(range(M), remainder):
        allocation[m] += 1
    return model.result(allocation, random_tree(model, allocation, rng))
