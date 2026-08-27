"""Generate sparse inputs for minimum-error sequential discrimination."""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations, product
from math import ceil, isfinite
from pathlib import Path
from random import Random

from input_data import SparseRow, pair_error_probability, write_sparse_input


HERE = Path(__file__).resolve().parent
INPUT_ROOT = HERE / "inputs"
ZERO_ERROR_ROOT = HERE.parent / "min-sensor-zero-error"
DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

# Reuse the field generators and zero-error solver without modifying that project.
if str(ZERO_ERROR_ROOT) not in sys.path:
    sys.path.insert(0, str(ZERO_ERROR_ROOT))

from field_models import (  # noqa: E402
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


def sample_sensor_counts(
    maximum_budget: int,
    samples_per_location: int,
    rng: Random,
) -> set[int]:
    """Select a small set of sensor counts for one pair and location."""
    sample_size = min(maximum_budget, samples_per_location)
    return set(rng.sample(range(1, maximum_budget + 1), sample_size))


def build_sparse_rows(
    K: int,
    M: int,
    phase_rows: list[tuple[int, int, int, float]],
    budgets: list[int],
    samples_per_location: int,
    seed: int,
) -> list[SparseRow]:
    """Create sparse ``P_ijmn`` rows from canonical phase gaps."""
    rng = Random(seed)
    phases = {(i, j, m): phase for i, j, m, phase in phase_rows}
    maximum_budget = max(budgets)
    if maximum_budget < 1:
        raise ValueError("at least one positive sensor budget is required")
    positive_budgets = [budget for budget in budgets if budget > 0]
    smallest_positive_budget = min(positive_budgets, default=0)
    rows: list[SparseRow] = []

    for pair in combinations(range(K), 2):
        counts_by_location = [
            sample_sensor_counts(maximum_budget, samples_per_location, rng)
            for _ in range(M)
        ]

        # Give every pair at least one positive error value that can be used
        # with the smallest positive budget.
        if smallest_positive_budget > 0:
            location = rng.randrange(M)
            counts_by_location[location].add(
                rng.randint(1, smallest_positive_budget)
            )

        for m in range(M):
            phase = phases[(*pair, m)]
            for sensor_count in sorted(counts_by_location[m]):
                rows.append(
                    (
                        pair[0],
                        pair[1],
                        m,
                        phase,
                        sensor_count,
                        pair_error_probability(phase, sensor_count),
                    )
                )

    return rows


def generate_nonzero_phase_case(
    model: str,
    dimension: int,
    K: int,
    M: int,
    seed: int,
    phase_scale: float,
) -> tuple[list[tuple[int, int, int, float]], dict[str, object]]:
    """Generate a case with a nonzero phase gap at every stored location."""
    generation_seed = seed
    while True:
        rows, metadata = generate_case(
            model,
            dimension,
            K,
            M,
            seed=generation_seed,
            c=phase_scale,
        )
        if all(phase != 0.0 for _, _, _, phase in rows):
            return rows, metadata
        generation_seed += 1


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
    samples_per_location: int,
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
        samples_per_location,
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
    case_number = 0

    # Generate every requested family and its balanced K and M cells.
    for model in selected_models:
        for (K, M), replicate_count in zip(
            factorial_cells, cell_replicates
        ):
            for replicate in range(1, replicate_count + 1):
                case_number += 1
                case_seed = seed + case_number * 1_000_003
                phase_rows, metadata = generate_nonzero_phase_case(
                    model,
                    dimension,
                    K,
                    M,
                    case_seed,
                    phase_scale,
                )
                phase_lookup = {
                    (i, j, location): phase
                    for i, j, location, phase in phase_rows
                }
                phase_gaps = {
                    pair: [
                        phase_lookup[(*pair, m)]
                        for m in range(M)
                    ]
                    for pair in combinations(range(K), 2)
                }
                thresholds = compute_nijm(K, M, phase_gaps)

                # Use the zero-error optimum to keep every tested budget below N_min.
                minimum_sensors, _, _ = (
                    optimal_perfect_pairwise_sensor_placement(
                        K, M, thresholds
                    )
                )
                budgets = select_budgets(
                    minimum_sensors, budget_fractions, budget_cap
                )
                sparse_rows = build_sparse_rows(
                    K,
                    M,
                    phase_rows,
                    budgets,
                    samples_per_location,
                    seed=case_seed + 500_009,
                )
                filename = f"case_{case_number:05d}.csv"
                write_sparse_input(target / filename, sparse_rows)
                cases.append(
                    {
                        "file": filename,
                        "replicate": replicate,
                        "minimum_sensors_for_zero_error": minimum_sensors,
                        "sensor_budgets": budgets,
                        **metadata,
                    }
                )

    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "description": description,
        "input_format": {
            "columns": ["i", "j", "m", "r_ijm", "n", "p_ijmn"],
            "stored_pairs": "i < j only",
            "phase_interval": "(-pi, pi]",
            "reverse_phase": "r_jim = principal_phase(-r_ijm)",
            "pair_error": (
                "P_ijmn = 0.5*(1-sin(n*abs(r_ijm)/2)) for "
                "n*abs(r_ijm) < pi; otherwise 0"
            ),
            "missing_count_rule": (
                "use the largest available sensor count not exceeding n; "
                "use 0.5 if none exists"
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
            "budget_cap_constraint": (
                "The cap limits exhaustive-search runtime and is not a "
                "constraint of the sensor-allocation problem."
                if budget_cap is not None
                else "No computational budget cap is applied."
            ),
            "samples_per_pair_location": samples_per_location,
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
    samples_per_location: int,
) -> None:
    """Validate a requested dataset design."""
    if not DATASET_NAME.fullmatch(dataset):
        raise ValueError("dataset contains unsupported characters")
    if dimension not in (1, 2):
        raise ValueError("dimension must be 1 or 2")
    if not models or any(model not in FIELD_MODELS for model in models):
        raise ValueError("models must contain known field-family names")
    if not k_values or any(K < 2 for K in k_values):
        raise ValueError("every K must be at least 2")
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
    if samples_per_location < 1:
        raise ValueError("samples_per_location must be positive")


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
    parser.add_argument("--samples-per-location", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Generate a custom sparse input dataset."""
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
        samples_per_location=arguments.samples_per_location,
        seed=arguments.seed,
        description="Custom sparse minimum-error sequential benchmark.",
        overwrite=arguments.overwrite,
    )
    print(f"Generated dataset: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
