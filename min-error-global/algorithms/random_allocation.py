"""Uniform-random fixed-budget allocation for global discrimination."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
import sys


PROBLEM_ROOT = Path(__file__).resolve().parent.parent
if str(PROBLEM_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBLEM_ROOT))

from objective import (  # noqa: E402
    DeltaTable,
    StateModel,
    pgm_error_probability,
    validate_problem,
)


@dataclass(frozen=True)
class RandomAllocationResult:
    """A uniformly sampled allocation and its actual PGM error."""

    allocation: tuple[int, ...]
    error_probability: float
    state_model: StateModel


def random_global_discrimination(
    K: int,
    M: int,
    N: int,
    delta_ijm: DeltaTable,
    state_model: str,
    *,
    seed: int,
) -> RandomAllocationResult:
    """Sample uniformly from all fixed-budget sensor allocations.

    Args:
        K: Number of candidate fields.
        M: Number of candidate sensing locations.
        N: Fixed total sensor budget.
        delta_ijm: Canonical, unwrapped pairwise phase differences in radians.
        state_model: ``"product"``, ``"local_ghz"``, or ``"global_ghz"``.
        seed: Reproducibility seed for the sampled weak composition.

    Returns:
        One uniformly sampled weak composition of ``N`` into ``M`` location
        counts, together with its actual equal-prior PGM error.
    """
    pairs, checked_model = validate_problem(
        K, M, N, delta_ijm, state_model
    )
    allocation = _uniform_weak_composition(N, M, Random(seed))
    return RandomAllocationResult(
        allocation=allocation,
        error_probability=pgm_error_probability(
            K,
            M,
            allocation,
            delta_ijm,
            checked_model,
            pairs,
        ),
        state_model=checked_model,
    )


def _uniform_weak_composition(
    total: int,
    parts: int,
    rng: Random,
) -> tuple[int, ...]:
    """Sample one weak composition using the stars-and-bars bijection."""
    separators = sorted(rng.sample(range(total + parts - 1), parts - 1))
    boundaries = (-1, *separators, total + parts - 1)
    return tuple(
        right - left - 1
        for left, right in zip(boundaries, boundaries[1:])
    )
