"""Gram-matrix objective functions for minimum-error global discrimination."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from math import cos, isfinite, prod
from numbers import Real
from typing import Literal

import numpy as np

from input_data import validate_delta_consistency


Pair = tuple[int, int]
DeltaTable = Mapping[Pair, Sequence[Real]]
StateModel = Literal["product", "local_ghz", "global_ghz"]

STATE_MODELS: tuple[StateModel, ...] = (
    "product",
    "local_ghz",
    "global_ghz",
)


def validate_problem(
    K: int,
    M: int,
    N: int,
    delta_ijm: DeltaTable,
    state_model: str,
) -> tuple[tuple[Pair, ...], StateModel]:
    """Validate dimensions, unwrapped phase differences, and state model."""
    if isinstance(K, bool) or not isinstance(K, int) or K < 2:
        raise ValueError("K must be an integer of at least 2")
    if isinstance(M, bool) or not isinstance(M, int) or M < 1:
        raise ValueError("M must be a positive integer")
    if isinstance(N, bool) or not isinstance(N, int) or N < 0:
        raise ValueError("N must be a nonnegative integer sensor budget")
    if not isinstance(delta_ijm, Mapping):
        raise ValueError("delta_ijm must be keyed by field-pair tuples")
    if state_model not in STATE_MODELS:
        choices = ", ".join(repr(model) for model in STATE_MODELS)
        raise ValueError(f"state_model must be one of: {choices}")

    pairs = tuple(combinations(range(K), 2))
    if set(delta_ijm) != set(pairs):
        raise ValueError(
            "delta_ijm must contain every canonical pair (i, j) with "
            "0 <= i < j < K"
        )
    for pair in pairs:
        deltas = delta_ijm[pair]
        if isinstance(deltas, (str, bytes)):
            raise ValueError(
                f"delta_ijm[{pair}] must be a sequence of exactly M={M} values"
            )
        try:
            delta_count = len(deltas)
        except TypeError as error:
            raise ValueError(
                f"delta_ijm[{pair}] must be an indexable sequence"
            ) from error
        if delta_count != M:
            raise ValueError(
                f"delta_ijm[{pair}] must contain exactly M={M} values"
            )
        if any(
            isinstance(delta, bool)
            or not isinstance(delta, Real)
            or not isfinite(float(delta))
            for delta in deltas
        ):
            raise ValueError("every delta_ijm value must be a finite real number")

    validate_delta_consistency(delta_ijm, K, M)
    return pairs, state_model  # type: ignore[return-value]


def output_state_gram_matrix(
    K: int,
    M: int,
    allocation: Sequence[int],
    delta_ijm: DeltaTable,
    state_model: StateModel,
    pairs: Sequence[Pair] | None = None,
) -> np.ndarray:
    """Construct the signed Gram matrix of the K final pure states.

    Unwrapped, mutually consistent ``delta_ijm`` values preserve the signs of
    all pairwise overlaps. This is required for one positive-semidefinite Gram
    matrix; independently wrapping pairwise differences can lose that global
    consistency.
    """
    _validate_allocation(allocation, M)
    if pairs is None:
        pairs = tuple(combinations(range(K), 2))

    gram = np.eye(K, dtype=float)
    for i, j in pairs:
        overlap = _state_overlap(
            M,
            allocation,
            delta_ijm[(i, j)],
            state_model,
        )
        gram[i, j] = overlap
        gram[j, i] = overlap
    return gram


def pgm_error_probability(
    K: int,
    M: int,
    allocation: Sequence[int],
    delta_ijm: DeltaTable,
    state_model: StateModel,
    pairs: Sequence[Pair] | None = None,
) -> float:
    """Return the actual equal-prior PGM error for one allocation.

    For output-state Gram matrix ``G``, the equal-prior PGM success probability
    is ``sum_i |sqrt(G)[i, i]|^2 / K``. The returned objective is therefore a
    probability in ``[0, 1]``.
    """
    gram = output_state_gram_matrix(
        K,
        M,
        allocation,
        delta_ijm,
        state_model,
        pairs,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if float(eigenvalues[0]) < -1e-9:
        raise ValueError("delta_ijm values produced a non-PSD Gram matrix")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    gram_square_root = (
        eigenvectors * np.sqrt(eigenvalues)
    ) @ eigenvectors.T
    success_probability = float(
        np.sum(np.diag(gram_square_root) ** 2) / K
    )
    error_probability = 1.0 - success_probability
    if not -1e-10 <= error_probability <= 1.0 + 1e-10:
        raise RuntimeError("computed PGM error lies outside [0, 1]")
    return min(1.0, max(0.0, error_probability))


def _state_overlap(
    M: int,
    allocation: Sequence[int],
    deltas: Sequence[Real],
    state_model: StateModel,
) -> float:
    """Return the signed output-state overlap for an input-state model."""
    if state_model == "product":
        return prod(
            cos(float(deltas[m]) / 2.0) ** allocation[m]
            for m in range(M)
        )
    if state_model == "local_ghz":
        return prod(
            cos(allocation[m] * float(deltas[m]) / 2.0)
            for m in range(M)
        )
    return cos(
        sum(allocation[m] * float(deltas[m]) for m in range(M)) / 2.0
    )


def _validate_allocation(allocation: Sequence[int], M: int) -> None:
    """Validate nonnegative integer sensor counts for M locations."""
    if len(allocation) != M or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in allocation
    ):
        raise ValueError("allocation must contain M nonnegative integers")
