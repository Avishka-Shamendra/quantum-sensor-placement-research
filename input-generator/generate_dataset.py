"""Create one configurable Section III CSV dataset below ``inputs/``.

Example from the repository root:

    python3 input-generator/generate_dataset.py \
        --dataset gaussian-study --family gaussian \
        --k-min 3 --k-max 10 --m-min 3 --m-max 10 --cases 20

The generator distributes cases as evenly as possible across the requested
K/M grid, records every seed and parameter in ``manifest.json``, and never
chooses cases based on an algorithm result.
"""

import argparse
import csv
import json
import re
import sys
from math import isfinite
from pathlib import Path
from random import Random

from field_models import FIELD_FORMULAS, FIELD_MODELS, CsvRow, generate_case


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
ROOT_DIRECTORY = SCRIPT_DIRECTORY.parent
INPUT_ROOT = ROOT_DIRECTORY / "inputs"
DEFAULT_SEED = 20260728
DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive finite number") from error
    if not isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a balanced synthetic field dataset under inputs/."
    )
    parser.add_argument("--dataset", required=True, help="New folder name under inputs/.")
    parser.add_argument("--family", required=True, choices=tuple(sorted(FIELD_MODELS)))
    parser.add_argument("--k-min", type=_positive_int, required=True)
    parser.add_argument("--k-max", type=_positive_int, required=True)
    parser.add_argument("--m-min", type=_positive_int, required=True)
    parser.add_argument("--m-max", type=_positive_int, required=True)
    parser.add_argument("--cases", type=_positive_int, required=True)
    parser.add_argument("--c", type=_positive_float, default=1.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated CSV files and manifest in this dataset folder.",
    )
    arguments = parser.parse_args()
    if arguments.k_min > arguments.k_max:
        parser.error("--k-min must not exceed --k-max")
    if arguments.k_min < 3:
        parser.error("--k-min must be at least 3 for this study")
    if arguments.m_min > arguments.m_max:
        parser.error("--m-min must not exceed --m-max")
    if arguments.m_min < 2:
        parser.error("--m-min must be at least 2")
    if not DATASET_NAME.fullmatch(arguments.dataset):
        parser.error("--dataset may contain only letters, digits, underscores, and hyphens")
    return arguments


def _balanced_configurations(
    k_min: int, k_max: int, m_min: int, m_max: int, cases: int, seed: int
) -> list[tuple[int, int]]:
    """Cycle through shuffled K/M grids so every configuration is represented."""
    grid = [(K, M) for K in range(k_min, k_max + 1) for M in range(m_min, m_max + 1)]
    rng = Random(seed)
    configurations: list[tuple[int, int]] = []
    while len(configurations) < cases:
        batch = grid.copy()
        rng.shuffle(batch)
        configurations.extend(batch)
    return configurations[:cases]


def _write_case(path: Path, rows: list[CsvRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(("i", "j", "m", "r_ijm"))
        for i, j, m, phase_gap in rows:
            writer.writerow((i, j, m, f"{phase_gap:.4f}"))


def _prepare_dataset_directory(directory: Path, overwrite: bool) -> None:
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise ValueError(
            f"Dataset directory already contains files: {directory}. Use --overwrite to replace generated data."
        )
    directory.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for csv_file in directory.glob("*.csv"):
            csv_file.unlink()
        (directory / "manifest.json").unlink(missing_ok=True)


def main() -> int:
    arguments = _parse_arguments()
    target = INPUT_ROOT / arguments.dataset
    try:
        _prepare_dataset_directory(target, arguments.overwrite)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    configurations = _balanced_configurations(
        arguments.k_min,
        arguments.k_max,
        arguments.m_min,
        arguments.m_max,
        arguments.cases,
        arguments.seed,
    )
    cases: list[dict[str, object]] = []
    for number, (K, M) in enumerate(configurations, start=1):
        case_seed = arguments.seed + number * 1_000_003
        rows, metadata = generate_case(
            arguments.family, K, M, seed=case_seed, c=arguments.c
        )
        filename = f"case_{number:03d}_k{K:02d}_m{M:02d}.csv"
        _write_case(target / filename, rows)
        cases.append({"file": filename, **metadata})

    manifest = {
        "dataset": arguments.dataset,
        "field_family": arguments.family,
        "field_formula": FIELD_FORMULAS[arguments.family],
        "phase_scale_c": arguments.c,
        "phase_definition": "r_ijm = principal_value(-c(f_i(x_m) - f_j(x_m)))",
        "selection": "balanced deterministic coverage of the requested K/M grid",
        "seed": arguments.seed,
        "requested_ranges": {
            "K": [arguments.k_min, arguments.k_max],
            "M": [arguments.m_min, arguments.m_max],
        },
        "case_count": arguments.cases,
        "cases": cases,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated {arguments.cases} {arguments.family} cases in {target} (c={arguments.c:g}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
