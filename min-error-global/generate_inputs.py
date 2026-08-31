"""Generate independent unwrapped inputs for global discrimination."""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations, product
from math import isfinite, pi
from numbers import Real
from pathlib import Path
from random import Random

from input_data import DeltaRow, write_delta_input


HERE = Path(__file__).resolve().parent
INPUT_ROOT = HERE / "inputs"
ZERO_ERROR_ROOT = HERE.parent / "min-sensor-zero-error"
DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
GLOBAL_BENCHMARK_PHASE_SCALE = pi

# Reuse exactly the paper field families and parameter ranges.
if str(ZERO_ERROR_ROOT) not in sys.path:
    sys.path.insert(0, str(ZERO_ERROR_ROOT))

from field_models import (  # noqa: E402
    FIELD_FORMULAS,
    FIELD_MODELS,
    FIELD_PARAMETER_RANGES,
    sensing_locations,
)


def generate_delta_case(
    model: str,
    dimension: int,
    K: int,
    M: int,
    seed: int,
    phase_scale: float,
) -> tuple[list[DeltaRow], dict[str, object]]:
    """Generate one unfiltered case with unwrapped pairwise differences."""
    if model not in FIELD_MODELS:
        raise ValueError(f"unknown field model: {model}")

    rng = Random(seed)
    locations = sensing_locations(dimension, M, rng)
    fields, parameters = FIELD_MODELS[model](K, locations, rng)
    rows = [
        (
            i,
            j,
            m,
            phase_scale * (fields[i][m] - fields[j][m]),
        )
        for i, j in combinations(range(K), 2)
        for m in range(M)
    ]
    if not all(isfinite(delta) for *_, delta in rows):
        raise RuntimeError("field model produced a nonfinite phase difference")
    return rows, {
        "field_model": model,
        "spatial_dimension": dimension,
        "K": K,
        "M": M,
        "phase_scale_c": phase_scale,
        "generation_seed": seed,
        "sensor_locations": [
            location[0] if dimension == 1 else list(location)
            for location in locations
        ],
        "field_parameters": parameters,
    }


def generate_factorial_dataset(
    dataset: str,
    *,
    dimension: int,
    models: list[str],
    k_values: list[int],
    m_values: list[int],
    replicates: int,
    phase_scale: float,
    seed: int,
    description: str,
    overwrite: bool = False,
) -> Path:
    """Generate a complete field-family by K by M global dataset."""
    _validate_design(
        dataset,
        dimension,
        models,
        k_values,
        m_values,
        replicates,
        phase_scale,
    )
    target = INPUT_ROOT / dataset
    _prepare_directory(target, overwrite)
    selected_models = list(dict.fromkeys(models))
    selected_k = sorted(set(k_values))
    selected_m = sorted(set(m_values))

    cases: list[dict[str, object]] = []
    case_number = 0
    for model, K, M in product(selected_models, selected_k, selected_m):
        for replicate in range(1, replicates + 1):
            case_number += 1
            case_seed = seed + case_number * 1_000_003
            rows, metadata = generate_delta_case(
                model,
                dimension,
                K,
                M,
                case_seed,
                phase_scale,
            )
            filename = f"case_{case_number:05d}.csv"
            write_delta_input(target / filename, rows)
            cases.append(
                {
                    "file": filename,
                    "replicate": replicate,
                    **metadata,
                }
            )

    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "description": description,
        "input_format": {
            "columns": ["i", "j", "m", "delta_ijm"],
            "stored_pairs": "i < j only",
            "definition": "delta_ijm = c*(f_i(x_m)-f_j(x_m))",
            "phase_wrapping": "none",
            "reverse_difference": "delta_jim = -delta_ijm",
            "theoretical_field_difference_interval": [-3.0, 3.0],
            "theoretical_delta_interval": [
                -3.0 * phase_scale,
                3.0 * phase_scale,
            ],
        },
        "design": {
            "type": "complete factorial",
            "spatial_dimension": dimension,
            "sensing_region": "[0, 1]" if dimension == 1 else "[0, 1]^2",
            "field_models": selected_models,
            "K": selected_k,
            "M": selected_m,
            "replicates_per_cell": replicates,
            "cases_per_field_model": (
                len(selected_k) * len(selected_m) * replicates
            ),
            "phase_scale_c": phase_scale,
            "case_count": case_number,
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
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def _validate_design(
    dataset: str,
    dimension: int,
    models: list[str],
    k_values: list[int],
    m_values: list[int],
    replicates: int,
    phase_scale: float,
) -> None:
    """Validate a requested global benchmark design."""
    if not DATASET_NAME.fullmatch(dataset):
        raise ValueError("dataset must contain only letters, digits, _ and -")
    if dimension not in (1, 2):
        raise ValueError("dimension must be 1 or 2")
    if not models or any(model not in FIELD_MODELS for model in models):
        raise ValueError("models must contain supported field-family names")
    if not k_values or any(
        isinstance(K, bool) or not isinstance(K, int) or K < 2
        for K in k_values
    ):
        raise ValueError("K values must be at least 2")
    if not m_values or any(
        isinstance(M, bool) or not isinstance(M, int) or M < 2
        for M in m_values
    ):
        raise ValueError("M values must be at least 2")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 1
    ):
        raise ValueError("replicates must be a positive integer")
    if (
        isinstance(phase_scale, bool)
        or not isinstance(phase_scale, Real)
        or not isfinite(float(phase_scale))
        or phase_scale <= 0.0
    ):
        raise ValueError("phase_scale must be positive and finite")


def _prepare_directory(target: Path, overwrite: bool) -> None:
    """Create an empty output directory without silently deleting data."""
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"dataset already exists: {target}")
        for path in sorted(target.iterdir()):
            if path.is_file():
                path.unlink()
            else:
                raise ValueError(f"unexpected subdirectory in dataset: {path}")
    else:
        target.mkdir(parents=True)


def _parse_arguments() -> argparse.Namespace:
    """Parse custom global-dataset settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--dimension", type=int, choices=(1, 2), required=True)
    parser.add_argument("--models", nargs="+", default=list(FIELD_MODELS))
    parser.add_argument("--k-values", nargs="+", type=int, required=True)
    parser.add_argument("--m-values", nargs="+", type=int, required=True)
    parser.add_argument("--replicates", type=int, required=True)
    parser.add_argument("--phase-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Generate one custom global dataset."""
    arguments = _parse_arguments()
    target = generate_factorial_dataset(
        arguments.dataset,
        dimension=arguments.dimension,
        models=arguments.models,
        k_values=arguments.k_values,
        m_values=arguments.m_values,
        replicates=arguments.replicates,
        phase_scale=arguments.phase_scale,
        seed=arguments.seed,
        description="Custom minimum-error global-discrimination benchmark.",
        overwrite=arguments.overwrite,
    )
    print(f"Generated {arguments.dataset} in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
