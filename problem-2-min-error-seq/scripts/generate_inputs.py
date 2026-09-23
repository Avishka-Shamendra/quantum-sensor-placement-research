"""Generate inputs for minimum-error sequential discrimination."""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations, product
from math import ceil, isfinite
from pathlib import Path
from statistics import fmean
from zlib import crc32

import numpy as np

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
    FIELD_FORMULAS,
    FIELD_MODELS,
    FIELD_PARAMETER_RANGES,
    generate_case,
)
from optimal import optimal_perfect_pairwise_sensor_placement  # noqa: E402
from phase_data import compute_nijm  # noqa: E402


R_MIN = 1e-2
MIN_DISTINCT_BEST_LOCATIONS = 2
SECOND_BEST_RATIO = 0.5
COMPETITIVE_PAIR_FRACTION = 0.15
ALT_LOCAL_RATIO = 0.8
CONFLICT_QUANTILE = 0.70
CALIBRATION_SAMPLES = 1_000
EPSILON = 1e-12


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


def _phase_structure(rows, K, M):
    gaps = {pair: [0.0] * M for pair in combinations(range(K), 2)}
    for i, j, location, value in rows:
        gaps[(i, j)][location] = abs(float(value))
    return gaps


def _hard_quartile_mean(fields, best_gaps):
    values = sorted(
        best_gaps[tuple(sorted(pair))]
        for pair in combinations(sorted(fields), 2)
    )
    count = max(1, ceil(0.25 * len(values)))
    return fmean(values[:count])


def structural_metrics(rows, K, M):
    """Return preliminary checks and the H-based conflict diagnostics."""
    gaps = _phase_structure(rows, K, M)
    best_gaps = {pair: max(values) for pair, values in gaps.items()}
    best_locations = {
        pair: max(range(M), key=lambda location: (values[location], -location))
        for pair, values in gaps.items()
    }
    second_ratios = {}
    for pair, values in gaps.items():
        ordered = sorted(values, reverse=True)
        second_ratios[pair] = ordered[1] / ordered[0] if ordered[0] else 1.0

    preliminary = (
        all(value >= R_MIN for value in best_gaps.values())
        and len(set(best_locations.values())) >= MIN_DISTINCT_BEST_LOCATIONS
        and sum(value >= SECOND_BEST_RATIO for value in second_ratios.values())
        / len(second_ratios)
        >= COMPETITIVE_PAIR_FRACTION
    )

    fields = set(range(K))

    def future(pair):
        i, j = pair
        return 0.5 * (
            _hard_quartile_mean(fields - {i}, best_gaps)
            + _hard_quartile_mean(fields - {j}, best_gaps)
        )

    tests = sorted(
        (-values[location], pair, location)
        for pair, values in gaps.items()
        for location in range(M)
    )
    negative_local_g, pair_g, location_g = tests[0]
    local_g = -negative_local_g
    future_g = future(pair_g)
    alternatives = []
    for negative_local, pair, location in tests[1:]:
        local = -negative_local
        if local + EPSILON >= ALT_LOCAL_RATIO * local_g:
            alternatives.append((future(pair), local, pair, location))

    alternative = None
    conflict_ratio = 1.0
    if alternatives:
        future_a, local_a, pair_a, location_a = min(
            alternatives,
            key=lambda item: (-item[0], -item[1], item[2], item[3]),
        )
        conflict_ratio = future_a / max(future_g, EPSILON)
        alternative = {
            "pair": list(pair_a),
            "location": location_a,
            "local_score": local_a,
            "future_score": future_a,
        }

    return preliminary, {
        "best_locations": [
            {"pair": list(pair), "location": best_locations[pair]}
            for pair in sorted(best_locations)
        ],
        "second_best_ratios": [
            {"pair": list(pair), "ratio": second_ratios[pair]}
            for pair in sorted(second_ratios)
        ],
        "H_full": _hard_quartile_mean(fields, best_gaps),
        "local_test": {
            "pair": list(pair_g),
            "location": location_g,
            "local_score": local_g,
            "future_score": future_g,
        },
        "competitive_alternative": alternative,
        "conflict_ratio": conflict_ratio,
    }


def _draw_case(model, dimension, K, M, seed, phase_scale):
    return generate_case(
        model,
        dimension,
        K,
        M,
        seed=seed,
        c=phase_scale,
        max_attempts=1,
        min_best_pair_phase_gap=0.0,
    )


def calibrate_conflict_threshold(
    model, dimension, K, M, seed, phase_scale, sample_count
):
    ratios = []
    generated = 0
    while len(ratios) < sample_count:
        rows, _ = _draw_case(
            model, dimension, K, M, seed + generated, phase_scale
        )
        generated += 1
        preliminary, metrics = structural_metrics(rows, K, M)
        if preliminary and metrics["competitive_alternative"] is not None:
            ratios.append(metrics["conflict_ratio"])
    return float(np.quantile(ratios, CONFLICT_QUANTILE)), generated


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
    calibration_samples: int = CALIBRATION_SAMPLES,
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
        calibration_samples,
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

    # Calibrate and then sample each design cell with independent seeds.
    for model in selected_models:
        for (K, M), replicate_count in zip(factorial_cells, cell_replicates):
            cell_code = crc32(f"{model}:{dimension}:{K}:{M}".encode())
            calibration_seed = (seed << 32) + cell_code
            threshold, calibration_generated = calibrate_conflict_threshold(
                model,
                dimension,
                K,
                M,
                calibration_seed,
                phase_scale,
                calibration_samples,
            )
            data_seed = ((seed + dataset_code) << 32) + cell_code
            generated = 0
            for replicate in range(1, replicate_count + 1):
                while True:
                    phase_rows, metadata = _draw_case(
                        model,
                        dimension,
                        K,
                        M,
                        data_seed + generated,
                        phase_scale,
                    )
                    generated += 1
                    preliminary, metrics = structural_metrics(phase_rows, K, M)
                    if (
                        preliminary
                        and metrics["competitive_alternative"] is not None
                        and metrics["conflict_ratio"] + EPSILON >= threshold
                    ):
                        break

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
                    "structural_metrics": {
                        **metrics,
                        "conflict_threshold": threshold,
                    },
                    **metadata,
                })
            cell_statistics.append({
                "field_model": model,
                "spatial_dimension": dimension,
                "K": K,
                "M": M,
                "calibration_samples": calibration_samples,
                "calibration_candidates": calibration_generated,
                "conflict_threshold": threshold,
                "accepted_cases": replicate_count,
                "generated_candidates": generated,
                "acceptance_rate": replicate_count / generated,
            })
            print(
                f"{dataset}: {model} K={K} M={M}, "
                f"threshold={threshold:.6f}, accepted {replicate_count}/{generated}"
            )

    manifest = {
        "schema_version": 4,
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
            "structural_filter": {
                "r_min": R_MIN,
                "min_distinct_best_locations": MIN_DISTINCT_BEST_LOCATIONS,
                "second_best_ratio": SECOND_BEST_RATIO,
                "competitive_pair_fraction": COMPETITIVE_PAIR_FRACTION,
                "alt_local_ratio": ALT_LOCAL_RATIO,
                "conflict_quantile": CONFLICT_QUANTILE,
                "calibration_samples_per_cell": calibration_samples,
            },
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
    calibration_samples: int,
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
    if (
        isinstance(calibration_samples, bool)
        or not isinstance(calibration_samples, int)
        or calibration_samples < 10
    ):
        raise ValueError("calibration_samples must be an integer >= 10")


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
    parser.add_argument("--calibration-samples", type=int, default=CALIBRATION_SAMPLES)
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
        calibration_samples=arguments.calibration_samples,
    )
    print(f"Generated dataset: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
