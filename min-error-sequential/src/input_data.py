"""Utilities for sequential-discrimination input data."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from math import isfinite, pi, sin
from numbers import Real
from pathlib import Path


Pair = tuple[int, int]
PhaseGaps = dict[Pair, list[float]]
PairErrorData = dict[Pair, list[dict[int, float]]]
SparseRow = tuple[int, int, int, float, int, float]


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
    """Compute the minimum pairwise error for a phase gap and sensor count."""
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


def write_sparse_input(csv_path: str | Path, rows: Sequence[SparseRow]) -> None:
    """Write stored ``r_ijm`` and ``P_ijmn`` values to a CSV file."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(("i", "j", "m", "r_ijm", "n", "p_ijmn"))
        for i, j, m, phase, sensor_count, error in rows:
            writer.writerow(
                (
                    i,
                    j,
                    m,
                    format(phase, ".17g"),
                    sensor_count,
                    format(error, ".17g"),
                )
            )


def load_sparse_input(
    csv_path: str | Path,
) -> tuple[int, int, PhaseGaps, PairErrorData]:
    """Load one input file and fill missing integer sensor counts exactly."""
    path = Path(csv_path)
    try:
        input_file = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        raise ValueError(f"Cannot read CSV file: {path}") from error

    with input_file:
        reader = csv.DictReader(input_file)
        required = {"i", "j", "m", "r_ijm", "n", "p_ijmn"}
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(
                f"{path} must have exactly these columns: "
                "i,j,m,r_ijm,n,p_ijmn"
            )

        raw_phases: dict[Pair, dict[int, float]] = {}
        raw_errors: dict[Pair, dict[int, dict[int, float]]] = {}
        maximum_sensor_count = 0

        # Read canonical pairs only; reverse phases are reconstructed when needed.
        for row_number, row in enumerate(reader, start=2):
            try:
                i = int(row["i"])
                j = int(row["j"])
                m = int(row["m"])
                phase = float(row["r_ijm"])
                sensor_count = int(row["n"])
                error = float(row["p_ijmn"])
            except (TypeError, ValueError) as parse_error:
                raise ValueError(f"Invalid value on row {row_number} of {path}") from parse_error

            if i < 0 or i >= j or m < 0 or sensor_count < 0:
                raise ValueError(
                    f"Row {row_number} must satisfy 0 <= i < j, m >= 0, n >= 0"
                )
            maximum_sensor_count = max(maximum_sensor_count, sensor_count)
            if not isfinite(phase) or not -pi < phase <= pi:
                raise ValueError(f"r_ijm on row {row_number} must lie in (-pi, pi]")
            if not isfinite(error) or not 0.0 <= error <= 0.5:
                raise ValueError(f"p_ijmn on row {row_number} must lie in [0, 0.5]")

            pair = (i, j)
            known_phase = raw_phases.setdefault(pair, {}).get(m)
            if known_phase is not None and phase != known_phase:
                raise ValueError(f"Inconsistent r_ijm for pair {pair}, location {m}")
            raw_phases[pair][m] = phase

            location_errors = raw_errors.setdefault(pair, {}).setdefault(m, {})
            if sensor_count in location_errors:
                raise ValueError(
                    f"Duplicate P_ijmn for pair {pair}, location {m}, n={sensor_count}"
                )
            expected = pair_error_probability(phase, sensor_count)
            if abs(error - expected) > 1e-12:
                raise ValueError(
                    f"P_ijmn on row {row_number} does not match the model equation"
                )
            location_errors[sensor_count] = error

    if not raw_errors:
        raise ValueError(f"CSV file contains no data: {path}")

    K = max(j for _, j in raw_errors) + 1
    M = max(m for locations in raw_errors.values() for m in locations) + 1
    pairs = tuple((i, j) for i in range(K) for j in range(i + 1, K))
    if set(raw_errors) != set(pairs):
        raise ValueError(f"{path} must contain every pair for K={K}")

    phase_gaps: PhaseGaps = {}
    pair_errors: PairErrorData = {}

    # Convert sparse row dictionaries into the structure used by the algorithms.
    for pair in pairs:
        if set(raw_errors[pair]) != set(range(M)):
            raise ValueError(
                f"Pair {pair} must contain locations 0 through {M - 1}"
            )
        phase_gaps[pair] = [raw_phases[pair][m] for m in range(M)]
        for m in range(M):
            phase = raw_phases[pair][m]
            location_errors = raw_errors[pair][m]
            for sensor_count in range(maximum_sensor_count + 1):
                location_errors.setdefault(
                    sensor_count,
                    pair_error_probability(phase, sensor_count),
                )
        pair_errors[pair] = [raw_errors[pair][m] for m in range(M)]

    return K, M, phase_gaps, pair_errors


def validate_reverse_phases(phase_gaps: Mapping[Pair, Sequence[float]]) -> None:
    """Check that reverse phases satisfy ``r_jim = -r_ijm`` modulo ``2 pi``."""
    for values in phase_gaps.values():
        for phase in values:
            reverse = reverse_phase(phase)
            if abs(principal_phase(phase + reverse)) > 1e-12:
                raise ValueError("phase gaps do not satisfy reverse-pair symmetry")
