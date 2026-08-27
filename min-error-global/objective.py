"""Shared validation and objective equations for the global problem."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import cos, isfinite, pi, prod
from numbers import Real
from typing import Literal


Pair = tuple[int, int]
PhaseTable = Mapping[Pair, Sequence[Real]]
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
    rijm: PhaseTable,
    state_model: str,
) -> tuple[tuple[Pair, ...], StateModel]:
    """Validate dimensions, the phase table, and the input-state model."""
    if isinstance(K, bool) or not isinstance(K, int) or K < 2:
        raise ValueError("K must be an integer of at least 2.")
    if isinstance(M, bool) or not isinstance(M, int) or M < 1:
        raise ValueError("M must be a positive integer.")
    if isinstance(N, bool) or not isinstance(N, int) or N < 0:
        raise ValueError("N must be a nonnegative integer sensor budget.")
    if not isinstance(rijm, Mapping):
        raise ValueError("rijm must be a mapping keyed by field-pair tuples.")
    if state_model not in STATE_MODELS:
        choices = ", ".join(repr(model) for model in STATE_MODELS)
        raise ValueError(f"state_model must be one of: {choices}.")

    pairs = tuple((i, j) for i in range(K) for j in range(i + 1, K))
    if set(rijm) != set(pairs):
        raise ValueError(
            "rijm must contain exactly every canonical pair (i, j) with "
            "0 <= i < j < K."
        )

    for pair in pairs:
        phases = rijm[pair]
        if isinstance(phases, (str, bytes)):
            raise ValueError(f"rijm[{pair}] must be a sequence of M phases.")
        try:
            phase_count = len(phases)
        except TypeError as error:
            raise ValueError(
                f"rijm[{pair}] must be an indexable sequence of M phases."
            ) from error
        if phase_count != M:
            raise ValueError(f"rijm[{pair}] must contain exactly M={M} phases.")

        for phase in phases:
            if (
                isinstance(phase, bool)
                or not isinstance(phase, Real)
                or not isfinite(float(phase))
                or not -pi < float(phase) <= pi
            ):
                raise ValueError(
                    "Every r_ijm phase must be a finite real number in "
                    "the paper's interval (-pi, pi]."
                )

    # The membership check above narrows this value to the supported literals.
    return pairs, state_model  # type: ignore[return-value]


def pgm_error_upper_bound(
    K: int,
    M: int,
    allocation: Sequence[int],
    rijm: PhaseTable,
    state_model: StateModel,
    pairs: Sequence[Pair] | None = None,
) -> float:
    """Evaluate Section V's Pretty Good Measurement error upper bound.

    ``allocation[m]`` is the number of sensors at location ``m``. The three
    supported models implement Equations (18), (19), and (20) of the paper.
    Inputs are expected to have been checked by :func:`validate_problem`.
    """
    if pairs is None:
        pairs = tuple((i, j) for i in range(K) for j in range(i + 1, K))

    overlap_sum = 0.0
    for pair in pairs:
        phases = rijm[pair]

        if state_model == "product":
            # Independent |+> sensors: multiply one-sensor overlaps.
            overlap = prod(
                abs(cos(float(phases[m]) / 2.0)) ** allocation[m]
                for m in range(M)
            )
        elif state_model == "local_ghz":
            # Sensors share a GHZ state only with sensors at the same location.
            overlap = prod(
                abs(cos(allocation[m] * float(phases[m]) / 2.0))
                for m in range(M)
            )
        else:
            # One GHZ state spans all sensors and accumulates all local phases.
            accumulated_phase = sum(
                allocation[m] * float(phases[m]) for m in range(M)
            )
            overlap = abs(cos(accumulated_phase / 2.0))

        overlap_sum += overlap

    # Sum over unordered pairs; this equals (1/K) times the ordered-pair sum.
    # This is the paper's analytical bound, so values above one are not clipped.
    return (2.0 / K) * overlap_sum
