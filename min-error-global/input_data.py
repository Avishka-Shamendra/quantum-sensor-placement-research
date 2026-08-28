"""Input utilities for unwrapped global-discrimination phase differences."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from itertools import combinations
from math import isclose, isfinite
from numbers import Real
from pathlib import Path


Pair = tuple[int, int]
DeltaTable = dict[Pair, list[float]]
DeltaRow = tuple[int, int, int, float]


def write_delta_input(
    csv_path: str | Path,
    rows: Sequence[DeltaRow],
) -> None:
    """Write canonical unwrapped ``delta_ijm`` values to a CSV file."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(("i", "j", "m", "delta_ijm"))
        for i, j, m, delta in rows:
            writer.writerow((i, j, m, format(delta, ".17g")))


def load_delta_input(csv_path: str | Path) -> tuple[int, int, DeltaTable]:
    """Load and validate one global-discrimination input file."""
    path = Path(csv_path)
    try:
        input_file = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        raise ValueError(f"Cannot read CSV file: {path}") from error

    with input_file:
        reader = csv.DictReader(input_file)
        required = {"i", "j", "m", "delta_ijm"}
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(
                f"{path} must have exactly these columns: i,j,m,delta_ijm"
            )

        raw: dict[Pair, dict[int, float]] = {}
        for row_number, row in enumerate(reader, start=2):
            try:
                i = int(row["i"])
                j = int(row["j"])
                m = int(row["m"])
                delta = float(row["delta_ijm"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid value on row {row_number} of {path}"
                ) from error

            if i < 0 or i >= j or m < 0:
                raise ValueError(
                    f"Row {row_number} must satisfy 0 <= i < j and m >= 0"
                )
            if not isfinite(delta):
                raise ValueError(
                    f"delta_ijm on row {row_number} must be finite"
                )

            pair = (i, j)
            locations = raw.setdefault(pair, {})
            if m in locations:
                raise ValueError(
                    f"Duplicate delta_ijm for pair {pair}, location {m}"
                )
            locations[m] = delta

    if not raw:
        raise ValueError(f"CSV file contains no data: {path}")

    K = max(j for _, j in raw) + 1
    M = max(m for locations in raw.values() for m in locations) + 1
    pairs = tuple(combinations(range(K), 2))
    if set(raw) != set(pairs):
        raise ValueError(f"{path} must contain every canonical pair for K={K}")

    deltas: DeltaTable = {}
    for pair in pairs:
        if set(raw[pair]) != set(range(M)):
            raise ValueError(
                f"Pair {pair} must contain locations 0 through {M - 1}"
            )
        deltas[pair] = [raw[pair][m] for m in range(M)]

    validate_delta_consistency(deltas, K, M)
    return K, M, deltas


def validate_delta_consistency(
    delta_ijm: Mapping[Pair, Sequence[Real]],
    K: int,
    M: int,
    *,
    tolerance: float = 1e-10,
) -> None:
    """Check that canonical deltas can come from one phase per field."""
    pairs = set(combinations(range(K), 2))
    if set(delta_ijm) != pairs:
        raise ValueError("delta_ijm must contain every canonical field pair")
    if any(len(delta_ijm[pair]) != M for pair in pairs):
        raise ValueError("every delta_ijm sequence must contain M values")

    # For delta_ijm = theta_im - theta_jm, every field triple must satisfy
    # delta_ijm + delta_jkm = delta_ikm at each sensing location.
    for i, j, k in combinations(range(K), 3):
        for m in range(M):
            left = float(delta_ijm[(i, j)][m]) + float(
                delta_ijm[(j, k)][m]
            )
            right = float(delta_ijm[(i, k)][m])
            if not isclose(
                left,
                right,
                rel_tol=tolerance,
                abs_tol=tolerance,
            ):
                raise ValueError(
                    "delta_ijm values are not mutually consistent at "
                    f"location {m} for fields {(i, j, k)}"
                )
