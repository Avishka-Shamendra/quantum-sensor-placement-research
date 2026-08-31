"""Randomized near-uniform allocation for global discrimination."""

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
class NearUniformResult:
    """A randomized near-uniform allocation and its actual PGM error."""

    allocation: tuple[int, ...]
    error_probability: float
    state_model: StateModel


def near_uniform_global_discrimination(
    K: int,
    M: int,
    N: int,
    delta_ijm: DeltaTable,
    state_model: str,
    *,
    seed: int,
) -> NearUniformResult:
    """Distribute sensors as evenly as possible with random remainder sites.

    Args:
        K: Number of candidate fields.
        M: Number of candidate sensing locations.
        N: Fixed total sensor budget.
        delta_ijm: Canonical, unwrapped pairwise phase differences in radians.
        state_model: ``"product"``, ``"local_ghz"``, or ``"global_ghz"``.
        seed: Reproducibility seed for selecting the remainder locations.

    Returns:
        An allocation in which every location receives ``N // M`` sensors and
        ``N % M`` distinct random locations receive one additional sensor,
        together with its actual equal-prior PGM error.
    """
    pairs, checked_model = validate_problem(
        K, M, N, delta_ijm, state_model
    )
    base_count, remainder = divmod(N, M)
    allocation = [base_count] * M
    for location in Random(seed).sample(range(M), remainder):
        allocation[location] += 1

    final_allocation = tuple(allocation)
    return NearUniformResult(
        allocation=final_allocation,
        error_probability=pgm_error_probability(
            K,
            M,
            final_allocation,
            delta_ijm,
            checked_model,
            pairs,
        ),
        state_model=checked_model,
    )
