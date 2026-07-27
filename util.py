"""Input and threshold utilities for perfect pairwise discrimination.

CSV case files use the columns ``i,j,m,r_ijm``.  Function and location
indices are zero-based throughout this repository.
"""

import csv
from collections.abc import Mapping, Sequence
from math import ceil, isfinite, pi
from numbers import Real
from pathlib import Path


def minimum_sensors_for_phase_gap(r_ijm: Real) -> int:
    """Compute the Section III threshold ``ceil(pi / abs(r_ijm))``.

    ``r_ijm`` must be a non-zero signed principal phase gap satisfying
    ``-pi < r_ijm <= pi``.  A zero gap has no finite perfect-discrimination
    sensor threshold at that location.
    """
    if isinstance(r_ijm, bool) or not isinstance(r_ijm, Real):
        raise ValueError("r_ijm must be a real number.")
    if not isfinite(r_ijm):
        raise ValueError("r_ijm must be finite.")
    if not -pi < r_ijm <= pi:
        raise ValueError("r_ijm must satisfy -pi < r_ijm <= pi.")
    if r_ijm == 0:
        raise ValueError(
            "r_ijm is zero, so this pair has no finite perfect-discrimination "
            "sensor threshold at this location."
        )
    return ceil(pi / abs(r_ijm))


def compute_nijm(
    F: int,
    M: int,
    rijm: Mapping[tuple[int, int], Sequence[Real]],
) -> dict[tuple[int, int], list[int]]:
    """Convert all Section III phase gaps into ``Nijm`` thresholds."""
    if isinstance(F, bool) or not isinstance(F, int) or F < 2:
        raise ValueError("F must be an integer of at least 2.")
    if isinstance(M, bool) or not isinstance(M, int) or M < 1:
        raise ValueError("M must be a positive integer.")
    if not isinstance(rijm, Mapping):
        raise ValueError("rijm must be a mapping keyed by function-pair tuples.")

    pairs = tuple((i, j) for i in range(F) for j in range(i + 1, F))
    if set(rijm) != set(pairs):
        raise ValueError(
            "rijm must contain exactly one (i, j) key with 0 <= i < j < F "
            "for every function pair."
        )

    Nijm: dict[tuple[int, int], list[int]] = {}
    for pair in pairs:
        phase_gaps = rijm[pair]
        if len(phase_gaps) != M:
            raise ValueError("Every rijm[(i, j)] sequence must have length M.")
        Nijm[pair] = [
            minimum_sensors_for_phase_gap(phase_gap)
            for phase_gap in phase_gaps
        ]
    return Nijm


def load_rijm_csv(
    csv_path: str | Path,
) -> tuple[int, int, dict[tuple[int, int], list[float]]]:
    """Read one ``i,j,m,r_ijm`` case CSV and infer ``F`` and ``M``.

    Every unordered function pair and every location index must occur exactly
    once.  Thus, the largest function index plus one determines ``F`` and the
    largest location index plus one determines ``M``.
    """
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
                f"{path} must have exactly these columns: i,j,m,r_ijm."
            )

        values: dict[tuple[int, int], dict[int, float]] = {}
        for row_number, row in enumerate(reader, start=2):
            try:
                i = int(row["i"])
                j = int(row["j"])
                m = int(row["m"])
                phase_gap = float(row["r_ijm"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid value on row {row_number} of {path}.") from error

            if i < 0 or j < 0 or m < 0 or i >= j:
                raise ValueError(
                    f"Row {row_number} of {path} must satisfy 0 <= i < j and m >= 0."
                )
            if not isfinite(phase_gap) or not -pi < phase_gap <= pi:
                raise ValueError(
                    f"r_ijm on row {row_number} of {path} must satisfy -pi < r_ijm <= pi."
                )

            pair = (i, j)
            if m in values.setdefault(pair, {}):
                raise ValueError(
                    f"Duplicate entry for pair {pair} at location {m} in {path}."
                )
            values[pair][m] = phase_gap

    if not values:
        raise ValueError(f"CSV file contains no case data: {path}")

    F = max(j for _, j in values) + 1
    M = max(m for pair_values in values.values() for m in pair_values) + 1
    if F <= 2:
        raise ValueError(
            f"{path} must contain more than two candidate fields (F > 2)."
        )
    pairs = tuple((i, j) for i in range(F) for j in range(i + 1, F))
    if set(values) != set(pairs):
        raise ValueError(
            f"{path} must contain every pair (i, j) for 0 <= i < j < {F}."
        )

    rijm: dict[tuple[int, int], list[float]] = {}
    for pair in pairs:
        if set(values[pair]) != set(range(M)):
            raise ValueError(
                f"{path} must contain every location m from 0 through {M - 1} "
                f"for pair {pair}."
            )
        rijm[pair] = [values[pair][m] for m in range(M)]

    return F, M, rijm


def load_rijm_cases(
    inputs_directory: str | Path = "inputs",
) -> dict[str, tuple[int, int, dict[tuple[int, int], list[float]]]]:
    """Load every CSV case in ``inputs_directory``, keyed by filename stem."""
    directory = Path(inputs_directory)
    if not directory.is_dir():
        raise ValueError(f"Inputs directory does not exist: {directory}")

    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV case files found in: {directory}")

    return {file_path.stem: load_rijm_csv(file_path) for file_path in csv_files}
