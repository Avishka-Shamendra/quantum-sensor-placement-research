"""Phase-gap input utilities for sequential discrimination."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from math import isfinite, pi, sin
from numbers import Real
from pathlib import Path


Pair = tuple[int, int]
PhaseGaps = dict[Pair, list[float]]
PhaseRow = tuple[int, int, int, float]


def principal_phase(value: Real) -> float:
    """Wrap a phase to the interval ``(-pi, pi]``."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError("phase must be a finite real number")
    wrapped = (float(value) + pi) % (2 * pi) - pi
    return pi if wrapped <= -pi else wrapped


def reverse_phase(phase: Real) -> float:
    """Return ``r_jim`` from a stored canonical phase ``r_ijm``."""
    return principal_phase(-float(phase))


def pair_error_probability(phase: Real, sensor_count: int) -> float:
    """Compute the tested-pair error directly from a phase gap."""
    if isinstance(sensor_count, bool) or not isinstance(sensor_count, int):
        raise ValueError("sensor_count must be a nonnegative integer")
    if sensor_count < 0:
        raise ValueError("sensor_count must be a nonnegative integer")

    gap = abs(principal_phase(phase))
    if gap == 0.0 or sensor_count == 0:
        return 0.5
    if sensor_count * gap >= pi:
        return 0.0
    return 0.5 * (1.0 - sin(sensor_count * gap / 2.0))


def write_phase_input(csv_path: str | Path, rows: Sequence[PhaseRow]) -> None:
    """Write one canonical phase gap per field pair and sensing location."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(("i", "j", "m", "r_ijm"))
        for i, j, m, phase in rows:
            writer.writerow((i, j, m, format(phase, ".17g")))


def load_phase_input(csv_path: str | Path) -> tuple[int, int, PhaseGaps]:
    """Load canonical phase gaps from a four-column CSV file."""
    path = Path(csv_path)
    try:
        input_file = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        raise ValueError(f"Cannot read CSV file: {path}") from error

    with input_file:
        reader = csv.DictReader(input_file)
        required = {"i", "j", "m", "r_ijm"}
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(f"{path} must have exactly these columns: i,j,m,r_ijm")

        raw_phases: dict[Pair, dict[int, float]] = {}
        for row_number, row in enumerate(reader, start=2):
            try:
                i = int(row["i"])
                j = int(row["j"])
                m = int(row["m"])
                phase = float(row["r_ijm"])
            except (TypeError, ValueError) as parse_error:
                raise ValueError(f"Invalid value on row {row_number} of {path}") from parse_error

            if i < 0 or i >= j or m < 0:
                raise ValueError(
                    f"Row {row_number} must satisfy 0 <= i < j and m >= 0"
                )
            if not isfinite(phase) or not -pi < phase <= pi:
                raise ValueError(f"r_ijm on row {row_number} must lie in (-pi, pi]")

            pair = (i, j)
            locations = raw_phases.setdefault(pair, {})
            if m in locations:
                raise ValueError(f"Duplicate r_ijm for pair {pair}, location {m}")
            locations[m] = phase

    if not raw_phases:
        raise ValueError(f"CSV file contains no data: {path}")

    K = max(j for _, j in raw_phases) + 1
    M = max(m for locations in raw_phases.values() for m in locations) + 1
    pairs = tuple((i, j) for i in range(K) for j in range(i + 1, K))
    if set(raw_phases) != set(pairs):
        raise ValueError(f"{path} must contain every pair for K={K}")

    phase_gaps: PhaseGaps = {}
    for pair in pairs:
        if set(raw_phases[pair]) != set(range(M)):
            raise ValueError(
                f"Pair {pair} must contain locations 0 through {M - 1}"
            )
        phase_gaps[pair] = [raw_phases[pair][m] for m in range(M)]

    return K, M, phase_gaps


def validate_reverse_phases(phase_gaps: Mapping[Pair, Sequence[float]]) -> None:
    """Check that reverse phases satisfy ``r_jim = -r_ijm`` modulo ``2 pi``."""
    for values in phase_gaps.values():
        for phase in values:
            reverse = reverse_phase(phase)
            if abs(principal_phase(phase + reverse)) > 1e-12:
                raise ValueError("phase gaps do not satisfy reverse-pair symmetry")
