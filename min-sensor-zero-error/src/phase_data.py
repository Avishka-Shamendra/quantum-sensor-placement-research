"""Load the paper's phase-gap input and derive sensor thresholds.

The CSV representation stores one row for every canonical field pair ``i < j``
and sensing location ``m``. Values obey

    r_ijm = Arg(exp(i c (f_i(x_m) - f_j(x_m)))) in (-pi, pi].

The reverse orientation is not stored because
``r_jim = principal_phase(-r_ijm)``. At the branch cut this identity is modulo
``2 pi``: both ``pi`` and ``-pi`` have the canonical representation ``pi``.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from math import ceil, isfinite, pi
from numbers import Real
from pathlib import Path
from typing import Optional


PhaseGaps = dict[tuple[int, int], list[float]]
Threshold = Optional[int]
Thresholds = dict[tuple[int, int], list[Threshold]]


def principal_phase(value: Real) -> float:
    """Wrap a finite phase to the paper's principal interval ``(-pi, pi]``."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError("phase must be a finite real number")
    wrapped = (float(value) + pi) % (2 * pi) - pi
    return pi if wrapped <= -pi else wrapped


def reverse_phase(r_ijm: Real) -> float:
    """Return the canonical reverse phase ``r_jim`` from ``r_ijm``."""
    return principal_phase(-float(r_ijm))


def get_phase_gap(
    phase_gaps: Mapping[tuple[int, int], Sequence[float]],
    i: int,
    j: int,
    m: int,
) -> float:
    """Return an oriented phase gap while storing only canonical pairs."""
    if i == j:
        return 0.0
    if i < j:
        return float(phase_gaps[(i, j)][m])
    return reverse_phase(phase_gaps[(j, i)][m])


def minimum_sensors_for_phase_gap(r_ijm: Real) -> Threshold:
    """Return ``ceil(pi / abs(r_ijm))`` or ``None`` for a zero gap."""
    phase = principal_phase(r_ijm)
    if phase == 0.0:
        return None
    return ceil(pi / abs(phase))


def compute_nijm(
    K: int,
    M: int,
    phase_gaps: Mapping[tuple[int, int], Sequence[Real]],
) -> Thresholds:
    """Convert canonical phase gaps to the paper's ``N_ijm`` thresholds."""
    _validate_dimensions(K, M)
    pairs = tuple((i, j) for i in range(K) for j in range(i + 1, K))
    if set(phase_gaps) != set(pairs):
        raise ValueError(
            "phase_gaps must contain exactly every canonical pair 0 <= i < j < K"
        )

    thresholds: Thresholds = {}
    for pair in pairs:
        values = phase_gaps[pair]
        if len(values) != M:
            raise ValueError("Every phase-gap sequence must have length M")
        thresholds[pair] = [
            minimum_sensors_for_phase_gap(value) for value in values
        ]
        if all(threshold is None for threshold in thresholds[pair]):
            raise ValueError(
                f"Pair {pair} has zero phase gap at every sensing location"
            )
    return thresholds


def load_phase_csv(csv_path: str | Path) -> tuple[int, int, PhaseGaps]:
    """Load and validate one ``i,j,m,r_ijm`` input file."""
    path = Path(csv_path)
    try:
        input_file = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        raise ValueError(f"Cannot read CSV file: {path}") from error

    with input_file:
        reader = csv.DictReader(input_file)
        required_columns = {"i", "j", "m", "r_ijm"}
        if reader.fieldnames is None or set(reader.fieldnames) != required_columns:
            raise ValueError(
                f"{path} must have exactly these columns: i,j,m,r_ijm"
            )

        raw: dict[tuple[int, int], dict[int, float]] = {}
        for row_number, row in enumerate(reader, start=2):
            try:
                i = int(row["i"])
                j = int(row["j"])
                m = int(row["m"])
                phase = float(row["r_ijm"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid value on row {row_number} of {path}") from error

            if i < 0 or j < 0 or m < 0 or i >= j:
                raise ValueError(
                    f"Row {row_number} of {path} must satisfy 0 <= i < j and m >= 0"
                )
            if not isfinite(phase) or not -pi < phase <= pi:
                raise ValueError(
                    f"r_ijm on row {row_number} must lie in (-pi, pi]"
                )

            pair = (i, j)
            if m in raw.setdefault(pair, {}):
                raise ValueError(
                    f"Duplicate entry for pair {pair} at location {m} in {path}"
                )
            raw[pair][m] = phase

    if not raw:
        raise ValueError(f"CSV file contains no phase data: {path}")

    K = max(j for _, j in raw) + 1
    M = max(m for values in raw.values() for m in values) + 1
    _validate_dimensions(K, M)
    pairs = tuple((i, j) for i in range(K) for j in range(i + 1, K))
    if set(raw) != set(pairs):
        raise ValueError(f"{path} must contain every canonical pair for K={K}")

    phase_gaps: PhaseGaps = {}
    for pair in pairs:
        if set(raw[pair]) != set(range(M)):
            raise ValueError(
                f"{path} must contain locations 0 through {M - 1} for pair {pair}"
            )
        phase_gaps[pair] = [raw[pair][m] for m in range(M)]

    compute_nijm(K, M, phase_gaps)
    return K, M, phase_gaps


# Check the number of fields and sensing locations.
def _validate_dimensions(K: int, M: int) -> None:
    if isinstance(K, bool) or not isinstance(K, int) or K < 2:
        raise ValueError("K must be an integer of at least 2")
    if isinstance(M, bool) or not isinstance(M, int) or M < 1:
        raise ValueError("M must be a positive integer")


# Backwards-compatible name for notebooks written against the original layout.
load_rijm_csv = load_phase_csv
