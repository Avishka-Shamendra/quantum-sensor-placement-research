"""Seeded random allocation and decision tree, evaluated exactly."""

from random import Random

from baselines import random_tree
from strategy import OutcomeModel


def random_baseline(K, M, N, phase_gaps, *, random_seed=0, time_limit=None):
    model = OutcomeModel(K, M, N, phase_gaps, time_limit=time_limit)
    rng = Random(random_seed)
    allocation = [0] * M
    for _ in range(N):
        allocation[rng.randrange(M)] += 1
    return model.result(allocation, random_tree(model, allocation, rng))
