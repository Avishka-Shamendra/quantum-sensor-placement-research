"""Generate inputs for minimum-error sequential discrimination."""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations, product
from math import ceil, isfinite
from pathlib import Path
from zlib import crc32

HERE = Path(__file__).resolve().parents[1]
SOURCE_ROOT = HERE / "src"
INPUT_ROOT = HERE / "inputs"
ZERO_ERROR_ROOT = HERE.parent / "problem-1-zero-error"
ZERO_ERROR_SOURCE_ROOT = ZERO_ERROR_ROOT / "src"
DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from input_data import write_phase_input

# Reuse the field generators and zero-error solver without modifying that project.
if str(ZERO_ERROR_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(ZERO_ERROR_SOURCE_ROOT))

from field_models import (  # noqa: E402
    DEFAULT_MIN_BEST_PAIR_PHASE_GAP,
    FIELD_FORMULAS,
    FIELD_MODELS,
    FIELD_PARAMETER_RANGES,
    generate_case,
)
from optimal import optimal_perfect_pairwise_sensor_placement  # noqa: E402
from phase_data import compute_nijm  # noqa: E402


def select_budgets(
    minimum_sensors: int,
    fractions: list[float],
    budget_cap: int | None,
) -> list[int]:
    """Select distinct budgets below the perfect-discrimination minimum."""
    largest_budget = minimum_sensors - 1
    if budget_cap is not None:
        largest_budget = min(largest_budget, budget_cap)
    if largest_budget <= 0:
        return [0]

    budgets = {
        min(largest_budget, max(1, ceil(fraction * largest_budget)))
        for fraction in fractions
    }
    return sorted(budgets)


def _draw_case(model, dimension, K, M, seed, phase_scale):
    return generate_case(
        model,
        dimension,
        K,
        M,
        seed=seed,
        c=phase_scale,
        min_best_pair_phase_gap=DEFAULT_MIN_BEST_PAIR_PHASE_GAP,
    )


def generate_factorial_dataset(
    dataset: str,
    *,
    dimension: int,
    models: list[str],
    k_values: list[int],
    m_values: list[int],
    replicates: int | None,
    cases_per_model: int | None = None,
    phase_scale: float,
    budget_fractions: list[float],
    budget_cap: int | None,
    seed: int,
    description: str,
    overwrite: bool = False,
) -> Path:
    """Generate a balanced field-family by K by M input dataset."""
    _validate_design(
        dataset,
        dimension,
        models,
        k_values,
        m_values,
        replicates,
        cases_per_model,
        phase_scale,
        budget_fractions,
        budget_cap,
    )

    target = INPUT_ROOT / dataset
    _prepare_directory(target, overwrite)
    selected_models = list(dict.fromkeys(models))
    selected_k = sorted(set(k_values))
    selected_m = sorted(set(m_values))
    factorial_cells = list(product(selected_k, selected_m))

    # Give each K and M cell the same count, with at most one extra case.
    if cases_per_model is None:
        assert replicates is not None
        cell_replicates = [replicates] * len(factorial_cells)
    else:
        base_replicates, extra_replicates = divmod(
            cases_per_model, len(factorial_cells)
        )
        cell_replicates = [
            base_replicates + int(cell_index < extra_replicates)
            for cell_index in range(len(factorial_cells))
        ]
    cases: list[dict[str, object]] = []
    cell_statistics = []
    case_number = 0
    dataset_code = crc32(dataset.encode())

    # Separate replicate seeds so rejection retries cannot overlap.
    for model in selected_models:
        for (K, M), replicate_count in zip(factorial_cells, cell_replicates):
            cell_code = crc32(f"{model}:{dimension}:{K}:{M}".encode())
            data_seed = ((seed + dataset_code) << 32) + cell_code
            for replicate in range(1, replicate_count + 1):
                phase_rows, metadata = _draw_case(
                    model,
                    dimension,
                    K,
                    M,
                    data_seed + (replicate - 1) * 1_000_003,
                    phase_scale,
                )

                case_number += 1
                phase_lookup = {
                    (i, j, location): phase
                    for i, j, location, phase in phase_rows
                }
                phase_gaps = {
                    pair: [phase_lookup[(*pair, m)] for m in range(M)]
                    for pair in combinations(range(K), 2)
                }
                thresholds = compute_nijm(K, M, phase_gaps)
                minimum_sensors, _, _ = optimal_perfect_pairwise_sensor_placement(
                    K, M, thresholds
                )
                budgets = select_budgets(
                    minimum_sensors, budget_fractions, budget_cap
                )
                filename = f"case_{case_number:05d}.csv"
                write_phase_input(target / filename, phase_rows)
                cases.append({
                    "file": filename,
                    "replicate": replicate,
                    "minimum_sensors_for_zero_error": minimum_sensors,
                    "sensor_budgets": budgets,
                    **metadata,
                })
            cell_statistics.append({
                "field_model": model,
                "spatial_dimension": dimension,
                "K": K,
                "M": M,
                "generated_cases": replicate_count,
            })
            print(f"{dataset}: {model} K={K} M={M}, saved {replicate_count} draws")

    manifest = {
        "schema_version": 5,
        "dataset": dataset,
        "description": description,
        "input_format": {
            "columns": ["i", "j", "m", "r_ijm"],
            "stored_pairs": "i < j only",
            "phase_interval": "(-pi, pi]",
            "reverse_phase": "r_jim = principal_phase(-r_ijm)",
            "probabilities": (
                "All tested-pair and off-pair probabilities are computed from "
                "the phase gaps when the algorithms run."
            ),
        },
        "design": {
            "type": (
                "complete factorial"
                if cases_per_model is None
                else "balanced factorial sample"
            ),
            "spatial_dimension": dimension,
            "sensing_region": "[0, 1]" if dimension == 1 else "[0, 1]^2",
            "field_models": selected_models,
            "K": selected_k,
            "M": selected_m,
            "replicates_per_cell": (
                replicates if cases_per_model is None else None
            ),
            "replicates_per_cell_range": [
                min(cell_replicates),
                max(cell_replicates),
            ],
            "cases_per_field_model": (
                len(factorial_cells) * replicates
                if cases_per_model is None and replicates is not None
                else cases_per_model
            ),
            "phase_scale_c": phase_scale,
            "budget_fractions_of_capped_range": budget_fractions,
            "budget_cap": budget_cap,
            "sampling": "random field draws conditioned on the pairwise phase-gap threshold",
            "min_best_pair_phase_gap": DEFAULT_MIN_BEST_PAIR_PHASE_GAP,
            "validity_rule": "For every pair (i, j), max_m abs(r_ijm) >= min_best_pair_phase_gap",
            "replicate_seed_stride": 1_000_003,
            "budget_cap_constraint": (
                "The cap limits exhaustive-search runtime and is not a "
                "constraint of the sensor-allocation problem."
                if budget_cap is not None
                else "No computational budget cap is applied."
            ),
            "case_count": case_number,
        },
        "cell_statistics": cell_statistics,
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


def _validate_design(
    dataset: str,
    dimension: int,
    models: list[str],
    k_values: list[int],
    m_values: list[int],
    replicates: int | None,
    cases_per_model: int | None,
    phase_scale: float,
    budget_fractions: list[float],
    budget_cap: int | None,
) -> None:
    """Validate a requested dataset design."""
    if not DATASET_NAME.fullmatch(dataset):
        raise ValueError("dataset contains unsupported characters")
    if dimension not in (1, 2):
        raise ValueError("dimension must be 1 or 2")
    if not models or any(model not in FIELD_MODELS for model in models):
        raise ValueError("models must contain known field-family names")
    if not k_values or any(K < 3 for K in k_values):
        raise ValueError("every K must be at least 3")
    if not m_values or any(M < 2 for M in m_values):
        raise ValueError("every M must be at least 2")
    if (replicates is None) == (cases_per_model is None):
        raise ValueError(
            "provide exactly one of replicates or cases_per_model"
        )
    if replicates is not None and replicates < 1:
        raise ValueError("replicates must be positive")
    factorial_cell_count = len(set(k_values)) * len(set(m_values))
    if cases_per_model is not None and cases_per_model < factorial_cell_count:
        raise ValueError(
            "cases_per_model must cover every K and M factorial cell"
        )
    if not isfinite(phase_scale) or phase_scale <= 0:
        raise ValueError("phase_scale must be positive and finite")
    if not budget_fractions or any(
        not isfinite(fraction) or not 0 < fraction <= 1
        for fraction in budget_fractions
    ):
        raise ValueError("budget fractions must lie in (0, 1]")
    if budget_cap is not None and budget_cap < 1:
        raise ValueError("budget_cap must be positive or None")


def _prepare_directory(directory: Path, overwrite: bool) -> None:
    """Create an empty target directory without removing unrelated files."""
    generated_files = list(directory.glob("case_*.csv")) if directory.exists() else []
    manifest = directory / "manifest.json"
    if (generated_files or manifest.exists()) and not overwrite:
        raise ValueError(f"Dataset already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in generated_files:
            path.unlink()
        manifest.unlink(missing_ok=True)


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line options for a custom dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--dimension", type=int, choices=(1, 2), default=1)
    parser.add_argument("--models", nargs="+", choices=sorted(FIELD_MODELS), default=sorted(FIELD_MODELS))
    parser.add_argument("--k-values", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--m-values", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--phase-scale", type=float, default=1.0)
    parser.add_argument("--budget-fractions", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--budget-cap", type=int, default=12)
    parser.add_argument("--no-budget-cap", action="store_true")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Generate a custom input dataset."""
    arguments = _parse_arguments()
    target = generate_factorial_dataset(
        arguments.dataset,
        dimension=arguments.dimension,
        models=arguments.models,
        k_values=arguments.k_values,
        m_values=arguments.m_values,
        replicates=arguments.replicates,
        cases_per_model=None,
        phase_scale=arguments.phase_scale,
        budget_fractions=arguments.budget_fractions,
        budget_cap=None if arguments.no_budget_cap else arguments.budget_cap,
        seed=arguments.seed,
        description="Custom minimum-error sequential benchmark.",
        overwrite=arguments.overwrite,
    )
    print(f"Generated dataset: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
