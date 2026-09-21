"""Generate reproducible phase-gap datasets for exact and greedy allocation.

Examples, run from the repository root:

    python3 problem-1-zero-error/scripts/generate_inputs.py custom_1d
    python3 problem-1-zero-error/scripts/generate_inputs.py custom_2d --dimension 2 --cases 40

Dataset names describe experiments, not field models. By default, cases cycle
through every available model and the requested K/M grid. The model used for a
case is recorded in ``manifest.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from math import isfinite
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve().parents[1]
SOURCE_ROOT = HERE / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from field_models import (
    DEFAULT_MIN_BEST_PAIR_PHASE_GAP,
    FIELD_FORMULAS,
    FIELD_MODELS,
    FIELD_PARAMETER_RANGES,
    CsvRow,
    generate_case,
)


INPUT_ROOT = HERE / "inputs"
DEFAULT_SEED = 20260728
DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


# Parse a command-line value that must be a positive integer.
def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


# Parse a command-line value that must be positive and finite.
def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive finite number") from error
    if not isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


# Define and validate the custom dataset parameters.
def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate canonical phase-gap inputs for both allocation algorithms."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="custom_1d",
        help="Neutral experiment name below inputs/ (default: custom_1d).",
    )
    parser.add_argument("--dimension", type=int, choices=(1, 2), default=1)
    parser.add_argument("--cases", type=_positive_int, default=20)
    parser.add_argument("--k-min", type=_positive_int, default=3)
    parser.add_argument("--k-max", type=_positive_int, default=8)
    parser.add_argument("--m-min", type=_positive_int, default=3)
    parser.add_argument("--m-max", type=_positive_int, default=8)
    parser.add_argument("--c", type=_positive_float, default=1.0)
    parser.add_argument(
        "--min-best-pair-phase-gap",
        type=_positive_float,
        default=DEFAULT_MIN_BEST_PAIR_PHASE_GAP,
        help=(
            "Reject a generated case unless every field pair has at least one "
            "location with |r_ijm| at or above this value (default: 1e-2)."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(sorted(FIELD_MODELS)),
        default=tuple(sorted(FIELD_MODELS)),
        help="Field models to balance across cases (default: all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace CSV files and manifest in an existing dataset directory.",
    )
    arguments = parser.parse_args()

    if not DATASET_NAME.fullmatch(arguments.dataset):
        parser.error("dataset may contain only letters, digits, underscores, and hyphens")
    if arguments.k_min < 2 or arguments.k_min > arguments.k_max:
        parser.error("require 2 <= --k-min <= --k-max")
    if arguments.m_min < 1 or arguments.m_min > arguments.m_max:
        parser.error("require 1 <= --m-min <= --m-max")
    return arguments


def balanced_configurations(
    models: list[str],
    k_min: int,
    k_max: int,
    m_min: int,
    m_max: int,
    cases: int,
    seed: int,
) -> list[tuple[str, int, int]]:
    """Balance every requested dimension independently to within one case."""
    rng = Random(seed)

    # Repeat and shuffle each value list until it contains the requested cases.
    def balanced_values(values: list[object]) -> list[object]:
        result: list[object] = []
        while len(result) < cases:
            batch = values.copy()
            rng.shuffle(batch)
            result.extend(batch)
        return result[:cases]

    model_values = balanced_values(list(models))
    k_values = balanced_values(list(range(k_min, k_max + 1)))
    m_values = balanced_values(list(range(m_min, m_max + 1)))
    return [
        (str(model), int(K), int(M))
        for model, K, M in zip(model_values, k_values, m_values)
    ]


# Write one phase-gap case using the standard CSV schema.
def _write_case(path: Path, rows: list[CsvRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(("i", "j", "m", "r_ijm"))
        for i, j, m, phase_gap in rows:
            writer.writerow((i, j, m, format(phase_gap, ".17g")))


# Prepare an empty dataset directory without touching unrelated files.
def _prepare_directory(directory: Path, overwrite: bool) -> None:
    existing_generated_files = list(directory.glob("case_*.csv")) if directory.exists() else []
    manifest = directory / "manifest.json"
    if (existing_generated_files or manifest.exists()) and not overwrite:
        raise ValueError(f"Dataset already exists: {directory}. Use --overwrite to replace it.")
    directory.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing_generated_files:
            path.unlink()
        manifest.unlink(missing_ok=True)


def generate_factorial_dataset(
    dataset: str,
    *,
    dimension: int,
    models: list[str],
    k_values: list[int],
    m_values: list[int],
    c_values: list[float],
    replicates: int,
    seed: int,
    description: str,
    overwrite: bool = False,
    min_best_pair_phase_gap: float = DEFAULT_MIN_BEST_PAIR_PHASE_GAP,
) -> Path:
    """Generate a complete model x K x M x c factorial experiment."""
    if not DATASET_NAME.fullmatch(dataset):
        raise ValueError("Invalid dataset name")
    if dimension not in (1, 2):
        raise ValueError("dimension must be 1 or 2")
    selected_models = list(dict.fromkeys(models))
    if not selected_models or any(model not in FIELD_MODELS for model in selected_models):
        raise ValueError("models must contain known field-model names")
    if not k_values or any(
        isinstance(K, bool) or not isinstance(K, int) or K < 2
        for K in k_values
    ):
        raise ValueError("Every K must be an integer of at least 2")
    if not m_values or any(
        isinstance(M, bool) or not isinstance(M, int) or M < 1
        for M in m_values
    ):
        raise ValueError("Every M must be an integer of at least 1")
    if not c_values or any(
        isinstance(c, bool)
        or not isinstance(c, (int, float))
        or not isfinite(c)
        or c <= 0
        for c in c_values
    ):
        raise ValueError("Every c must be a positive finite number")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if (
        isinstance(min_best_pair_phase_gap, bool)
        or not isinstance(min_best_pair_phase_gap, (int, float))
        or not isfinite(min_best_pair_phase_gap)
        or min_best_pair_phase_gap <= 0
    ):
        raise ValueError("min_best_pair_phase_gap must be a positive finite number")

    selected_k = sorted(set(k_values))
    selected_m = sorted(set(m_values))
    selected_c = sorted(set(float(c) for c in c_values))
    target = INPUT_ROOT / dataset
    _prepare_directory(target, overwrite)

    cases: list[dict[str, object]] = []
    case_number = 0
    realization_number = 0

    # Generate every model, K, M, replicate, and phase-scale combination.
    for model in selected_models:
        for K in selected_k:
            for M in selected_m:
                for replicate in range(1, replicates + 1):
                    realization_number += 1
                    case_seed = seed + realization_number * 1_000_003
                    for c in selected_c:
                        case_number += 1
                        rows, metadata = generate_case(
                            model,
                            dimension,
                            K,
                            M,
                            seed=case_seed,
                            c=c,
                            min_best_pair_phase_gap=min_best_pair_phase_gap,
                        )
                        filename = f"case_{case_number:05d}.csv"
                        _write_case(target / filename, rows)
                        cases.append({
                            "file": filename,
                            "replicate": replicate,
                            **metadata,
                        })
                        if case_number % 1_000 == 0:
                            print(f"  {dataset}: {case_number:,} cases generated", flush=True)

    manifest = {
        "schema_version": 2,
        "dataset": dataset,
        "description": description,
        "phase_data": {
            "interval": "(-pi, pi]",
            "definition": "r_ijm = Arg(exp(i*c*(f_i(x_m)-f_j(x_m))))",
            "reverse_pair": "r_jim = principal_phase(-r_ijm)",
            "stored_pairs": "i < j only",
            "sensor_threshold": "N_ijm = ceil(pi / abs(r_ijm)); undefined when r_ijm = 0",
            "nondegeneracy_filter": (
                "Each generated case must satisfy max_m abs(r_ijm) >= "
                "min_best_pair_phase_gap for every pair (i, j)."
            ),
        },
        "design": {
            "type": "complete factorial",
            "spatial_dimension": dimension,
            "sensing_region": "[0, 1]" if dimension == 1 else "[0, 1]^2",
            "location_sampling": "independent uniform",
            "models": selected_models,
            "K": selected_k,
            "M": selected_m,
            "phase_scale_c": selected_c,
            "replicates_per_cell": replicates,
            "independent_realizations": len(selected_models)
            * len(selected_k)
            * len(selected_m)
            * replicates,
            "case_count": case_number,
            "paired_across_c": len(selected_c) > 1,
            "min_best_pair_phase_gap": float(min_best_pair_phase_gap),
        },
        "seed": seed,
        "models": {
            model: {
                "formula": FIELD_FORMULAS[model][dimension],
                "parameter_ranges": FIELD_PARAMETER_RANGES[model][dimension],
            }
            for model in selected_models
        },
        "cases": cases,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return target


# Generate a custom balanced dataset from command-line parameters.
def main() -> int:
    arguments = _parse_arguments()
    target = INPUT_ROOT / arguments.dataset
    try:
        _prepare_directory(target, arguments.overwrite)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    configurations = balanced_configurations(
        list(dict.fromkeys(arguments.models)),
        arguments.k_min,
        arguments.k_max,
        arguments.m_min,
        arguments.m_max,
        arguments.cases,
        arguments.seed,
    )

    cases: list[dict[str, object]] = []
    for number, (model, K, M) in enumerate(configurations, start=1):
        case_seed = arguments.seed + number * 1_000_003
        rows, metadata = generate_case(
            model,
            arguments.dimension,
            K,
            M,
            seed=case_seed,
            c=arguments.c,
            min_best_pair_phase_gap=arguments.min_best_pair_phase_gap,
        )
        filename = f"case_{number:03d}.csv"
        _write_case(target / filename, rows)
        cases.append({"file": filename, **metadata})

    selected_models = list(dict.fromkeys(arguments.models))
    manifest = {
        "schema_version": 1,
        "dataset": arguments.dataset,
        "phase_data": {
            "interval": "(-pi, pi]",
            "definition": "r_ijm = Arg(exp(i*c*(f_i(x_m)-f_j(x_m))))",
            "reverse_pair": "r_jim = principal_phase(-r_ijm)",
            "stored_pairs": "i < j only",
            "sensor_threshold": "N_ijm = ceil(pi / abs(r_ijm)); undefined when r_ijm = 0",
            "phase_scale_c": arguments.c,
            "nondegeneracy_filter": (
                "Each generated case must satisfy max_m abs(r_ijm) >= "
                "min_best_pair_phase_gap for every pair (i, j)."
            ),
            "min_best_pair_phase_gap": arguments.min_best_pair_phase_gap,
        },
        "sampling": "deterministic; model, K, and M marginal counts differ by at most one",
        "spatial_dimension": arguments.dimension,
        "sensing_region": "[0, 1]" if arguments.dimension == 1 else "[0, 1]^2",
        "location_sampling": "independent uniform",
        "seed": arguments.seed,
        "models": {
            model: {
                "formula": FIELD_FORMULAS[model][arguments.dimension],
                "parameter_ranges": FIELD_PARAMETER_RANGES[model][arguments.dimension],
            }
            for model in selected_models
        },
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
    print(f"Generated {arguments.cases} cases in {target}")
    print(f"Models: {', '.join(selected_models)}; c={arguments.c:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
