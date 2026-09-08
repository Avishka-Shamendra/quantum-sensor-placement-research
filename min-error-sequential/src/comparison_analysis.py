"""Paired evaluation and plotting helpers for allocation algorithms."""

from __future__ import annotations

import csv
import json
from math import isclose, sqrt
from pathlib import Path
from statistics import fmean, median, stdev
from time import perf_counter
from zlib import crc32

import matplotlib.pyplot as plt
import numpy as np

from exhaustive_search import exhaustive_search
from greedy import greedy
import heuristic as heuristic_module
from plot_style import configure_plot_style
from random_baseline import random_baseline
from uniform import uniform


HERE = Path(__file__).resolve().parents[1]
from input_data import load_sparse_input, validate_reverse_phases

INPUT_ROOT = HERE / "inputs"
DEFAULT_RESULTS_ROOT = HERE / "results" / "algorithm_comparison_compact"

# Use the same algorithm colors in every comparison figure.
ALGORITHM_COLORS = {
    "Optimal": "#1f77b4",
    "Greedy": "#ff7f0e",
    "Heuristic": "#d62728",
    "Uniform": "#7b3294",
    "Random": "#006400",
}
FIELD_FAMILY_ORDER = ("gaussian", "sinusoidal", "quadratic")

configure_plot_style()


def run_paired_dataset(
    dataset: str,
    replicates_per_cell: int | None = 1,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    optimal_parameters: dict[str, object] | None = None,
    greedy_parameters: dict[str, object] | None = None,
    tree_aware_parameters: dict[str, object] | None = None,
    uniform_parameters: dict[str, object] | None = None,
    random_parameters: dict[str, object] | None = None,
    progress_every: int = 25,
) -> list[dict[str, object]]:
    """Run all algorithms on the same cases and budgets."""
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    optimal_parameters = dict(optimal_parameters or {})
    greedy_parameters = dict(greedy_parameters or {})
    tree_aware_parameters = dict(tree_aware_parameters or {})
    uniform_parameters = dict(uniform_parameters or {})
    random_parameters = dict(random_parameters or {})
    dataset_directory = INPUT_ROOT / dataset
    manifest = json.loads(
        (dataset_directory / "manifest.json").read_text(encoding="utf-8")
    )
    cases = manifest["cases"]

    # A value of None runs every case; a number limits replicates per K and M cell.
    if replicates_per_cell is not None:
        cases = [
            case
            for case in cases
            if int(case["replicate"]) <= replicates_per_cell
        ]

    records: list[dict[str, object]] = []

    # Give every algorithm the same input case and sensor budget.
    for case_number, case in enumerate(cases, start=1):
        K, M, phase_gaps, pair_errors = load_sparse_input(
            dataset_directory / str(case["file"])
        )
        validate_reverse_phases(phase_gaps)
        budgets = list(case["sensor_budgets"])
        minimum_sensors = int(case["minimum_sensors_for_zero_error"])

        for budget_level, budget_value in enumerate(budgets, start=1):
            budget = int(budget_value)

            started = perf_counter()
            optimal = exhaustive_search(
                K, M, budget, pair_errors, **optimal_parameters
            )
            optimal_seconds = perf_counter() - started

            started = perf_counter()
            greedy_result = greedy(K, M, budget, pair_errors, **greedy_parameters)
            greedy_seconds = perf_counter() - started

            started = perf_counter()
            tree_aware_result = heuristic_module.heuristic(
                K, M, budget, pair_errors, **tree_aware_parameters
            )
            tree_aware_seconds = perf_counter() - started

            started = perf_counter()
            baseline = uniform(K, M, budget, pair_errors, **uniform_parameters)
            uniform_seconds = perf_counter() - started

            started = perf_counter()
            random_result = random_baseline(
                K,
                M,
                budget,
                pair_errors,
                **_case_random_parameters(
                    random_parameters, dataset, case, budget
                ),
            )
            random_seconds = perf_counter() - started

            _validate_strategy(
                optimal.allocation,
                optimal.pair_locations,
                optimal.minimum_error,
                K,
                M,
                budget,
            )
            _validate_strategy(
                greedy_result.allocation,
                greedy_result.pair_locations,
                greedy_result.error_probability,
                K,
                M,
                budget,
            )
            _validate_strategy(
                tree_aware_result.allocation,
                tree_aware_result.pair_locations,
                tree_aware_result.error_probability,
                K,
                M,
                budget,
            )
            _validate_strategy(
                baseline.allocation,
                baseline.pair_locations,
                baseline.error_probability,
                K,
                M,
                budget,
            )
            _validate_strategy(
                random_result.allocation,
                random_result.pair_locations,
                random_result.error_probability,
                K,
                M,
                budget,
            )
            if optimal.minimum_error > greedy_result.error_probability + 1e-12:
                raise RuntimeError("greedy error is below the exact optimum")
            if optimal.minimum_error > tree_aware_result.error_probability + 1e-12:
                raise RuntimeError("heuristic error is below the exact optimum")
            if optimal.minimum_error > baseline.error_probability + 1e-12:
                raise RuntimeError("uniform error is below the exact optimum")
            if optimal.minimum_error > random_result.error_probability + 1e-12:
                raise RuntimeError("random error is below the exact optimum")

            # Measure quality and allocation differences from the exact optimum.
            greedy_error_gap = (
                greedy_result.error_probability - optimal.minimum_error
            )
            tree_aware_error_gap = (
                tree_aware_result.error_probability - optimal.minimum_error
            )
            uniform_error_gap = baseline.error_probability - optimal.minimum_error
            random_error_gap = (
                random_result.error_probability - optimal.minimum_error
            )
            greedy_allocation_distance = sum(
                abs(greedy_count - optimal_count)
                for greedy_count, optimal_count in zip(
                    greedy_result.allocation, optimal.allocation
                )
            )
            tree_aware_allocation_distance = sum(
                abs(tree_aware_count - optimal_count)
                for tree_aware_count, optimal_count in zip(
                    tree_aware_result.allocation, optimal.allocation
                )
            )
            uniform_allocation_distance = sum(
                abs(uniform_count - optimal_count)
                for uniform_count, optimal_count in zip(
                    baseline.allocation, optimal.allocation
                )
            )
            random_allocation_distance = sum(
                abs(random_count - optimal_count)
                for random_count, optimal_count in zip(
                    random_result.allocation, optimal.allocation
                )
            )
            records.append(
                {
                    "dataset": dataset,
                    "case": Path(str(case["file"])).stem,
                    "field_model": case["field_model"],
                    "spatial_dimension": int(case["spatial_dimension"]),
                    "replicate": int(case["replicate"]),
                    "K": K,
                    "M": M,
                    "minimum_sensors_for_zero_error": minimum_sensors,
                    "sensor_budget": budget,
                    "budget_level": budget_level,
                    "number_of_budget_levels": len(budgets),
                    "budget_fraction_of_N_min": budget / minimum_sensors,
                    "optimal_error": optimal.minimum_error,
                    "greedy_error": greedy_result.error_probability,
                    "tree_aware_error": tree_aware_result.error_probability,
                    "uniform_error": baseline.error_probability,
                    "random_error": random_result.error_probability,
                    "greedy_error_gap": greedy_error_gap,
                    "tree_aware_error_gap": tree_aware_error_gap,
                    "uniform_error_gap": uniform_error_gap,
                    "random_error_gap": random_error_gap,
                    "greedy_exact_error_match": isclose(
                        greedy_error_gap, 0.0, abs_tol=1e-12
                    ),
                    "tree_aware_exact_error_match": isclose(
                        tree_aware_error_gap, 0.0, abs_tol=1e-12
                    ),
                    "uniform_exact_error_match": isclose(
                        uniform_error_gap, 0.0, abs_tol=1e-12
                    ),
                    "random_exact_error_match": isclose(
                        random_error_gap, 0.0, abs_tol=1e-12
                    ),
                    "optimal_seconds": optimal_seconds,
                    "greedy_seconds": greedy_seconds,
                    "tree_aware_seconds": tree_aware_seconds,
                    "uniform_seconds": uniform_seconds,
                    "random_seconds": random_seconds,
                    "greedy_runtime_speedup": optimal_seconds / greedy_seconds,
                    "tree_aware_runtime_speedup": (
                        optimal_seconds / tree_aware_seconds
                    ),
                    "uniform_runtime_speedup": optimal_seconds / uniform_seconds,
                    "random_runtime_speedup": optimal_seconds / random_seconds,
                    "optimal_allocation": list(optimal.allocation),
                    "greedy_allocation": list(greedy_result.allocation),
                    "tree_aware_allocation": list(tree_aware_result.allocation),
                    "uniform_allocation": list(baseline.allocation),
                    "random_allocation": list(random_result.allocation),
                    "greedy_allocation_l1_distance": greedy_allocation_distance,
                    "tree_aware_allocation_l1_distance": (
                        tree_aware_allocation_distance
                    ),
                    "uniform_allocation_l1_distance": uniform_allocation_distance,
                    "random_allocation_l1_distance": random_allocation_distance,
                    "greedy_allocation_difference_fraction": (
                        greedy_allocation_distance / (2 * budget)
                        if budget > 0
                        else 0.0
                    ),
                    "tree_aware_allocation_difference_fraction": (
                        tree_aware_allocation_distance / (2 * budget)
                        if budget > 0
                        else 0.0
                    ),
                    "uniform_allocation_difference_fraction": (
                        uniform_allocation_distance / (2 * budget)
                        if budget > 0
                        else 0.0
                    ),
                    "random_allocation_difference_fraction": (
                        random_allocation_distance / (2 * budget)
                        if budget > 0
                        else 0.0
                    ),
                    "optimal_pair_locations": _pair_locations_to_dict(
                        optimal.pair_locations
                    ),
                    "greedy_pair_locations": _pair_locations_to_dict(
                        greedy_result.pair_locations
                    ),
                    "tree_aware_pair_locations": _pair_locations_to_dict(
                        tree_aware_result.pair_locations
                    ),
                    "uniform_pair_locations": _pair_locations_to_dict(
                        baseline.pair_locations
                    ),
                    "random_pair_locations": _pair_locations_to_dict(
                        random_result.pair_locations
                    ),
                    "optimal_decision_tree": _tree_to_dict(optimal.decision_tree),
                    "greedy_decision_tree": _tree_to_dict(
                        greedy_result.decision_tree
                    ),
                    "tree_aware_decision_tree": _tree_to_dict(
                        tree_aware_result.decision_tree
                    ),
                    "uniform_decision_tree": _tree_to_dict(
                        baseline.decision_tree
                    ),
                    "random_decision_tree": _tree_to_dict(
                        random_result.decision_tree
                    ),
                }
            )

        if (
            case_number == 1
            or case_number % progress_every == 0
            or case_number == len(cases)
        ):
            print(
                f"{dataset}: completed case {case_number}/{len(cases)} "
                f"({case['field_model']}, K={K}, M={M})",
                flush=True,
            )

    run_scope = "full" if replicates_per_cell is None else "test"
    algorithm_parameters = {
        "optimal": optimal_parameters,
        "greedy": greedy_parameters,
        "tree_aware": tree_aware_parameters,
        "uniform": uniform_parameters,
        "random": random_parameters,
    }
    save_paired_results(
        dataset,
        records,
        results_root,
        run_scope=run_scope,
        algorithm_parameters=algorithm_parameters,
    )
    return records


def save_paired_results(
    dataset: str,
    records: list[dict[str, object]],
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    run_scope: str = "test",
    algorithm_parameters: dict[str, dict[str, object]] | None = None,
) -> None:
    """Save detailed and compact paired comparison results."""
    tables_directory = _dataset_results_directory(
        results_root, dataset, run_scope
    ) / "tables"
    tables_directory.mkdir(parents=True, exist_ok=True)

    # JSON keeps complete strategies; CSV keeps the comparison measures.
    (tables_directory / "results.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "result_count": len(records),
                "algorithm_parameters": algorithm_parameters or {},
                "results": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    columns = [
        "dataset",
        "case",
        "field_model",
        "spatial_dimension",
        "replicate",
        "K",
        "M",
        "minimum_sensors_for_zero_error",
        "sensor_budget",
        "budget_level",
        "budget_fraction_of_N_min",
        "optimal_error",
        "greedy_error",
        "tree_aware_error",
        "uniform_error",
        "random_error",
        "greedy_error_gap",
        "tree_aware_error_gap",
        "uniform_error_gap",
        "random_error_gap",
        "greedy_exact_error_match",
        "tree_aware_exact_error_match",
        "uniform_exact_error_match",
        "random_exact_error_match",
        "optimal_seconds",
        "greedy_seconds",
        "tree_aware_seconds",
        "uniform_seconds",
        "random_seconds",
        "greedy_runtime_speedup",
        "tree_aware_runtime_speedup",
        "uniform_runtime_speedup",
        "random_runtime_speedup",
        "greedy_allocation_l1_distance",
        "tree_aware_allocation_l1_distance",
        "uniform_allocation_l1_distance",
        "random_allocation_l1_distance",
        "greedy_allocation_difference_fraction",
        "tree_aware_allocation_difference_fraction",
        "uniform_allocation_difference_fraction",
        "random_allocation_difference_fraction",
    ]
    with (tables_directory / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({column: record[column] for column in columns})


def load_paired_results(
    dataset: str,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    run_scope: str = "test",
) -> list[dict[str, object]]:
    """Load previously saved paired records without rerunning algorithms."""
    if run_scope not in {"test", "full"}:
        raise ValueError("run_scope must be 'test' or 'full'")
    results_path = (
        _dataset_results_directory(results_root, dataset, run_scope)
        / "tables"
        / "results.json"
    )
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    return list(payload["results"])


def make_comparison_plots(
    dataset: str,
    records: list[dict[str, object]],
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    run_scope: str = "test",
) -> list[tuple[str, plt.Figure]]:
    """Create and save six candidate comparison plots."""
    figures_directory = _dataset_results_directory(
        results_root, dataset, run_scope
    ) / "figures"
    figures_directory.mkdir(parents=True, exist_ok=True)
    dimension = int(records[0]["spatial_dimension"])
    figures = [
        (
            "Mean surrogate-error curves",
            _plot_mean_error_curves(records, dimension),
            "mean_error_curves",
        ),
        (
            "Heuristic surrogate error versus Exhaustive surrogate error",
            _plot_error_scatter(records, dimension),
            "greedy_vs_optimal_scatter",
        ),
        (
            "Additive surrogate-error gap",
            _plot_error_gap(records, dimension),
            "additive_error_gap",
        ),
        (
            "Exact-error match rate",
            _plot_match_heatmap(records, dimension),
            "exact_match_heatmap",
        ),
        (
            "Runtime speedup",
            _plot_runtime_speedup(records, dimension),
            "runtime_speedup",
        ),
        (
            "Surrogate-error gap by field family",
            _plot_gap_by_family(records, dimension),
            "error_gap_by_field_family",
        ),
    ]

    result: list[tuple[str, plt.Figure]] = []
    for title, figure, filename in figures:
        figure.savefig(figures_directory / f"{filename}.png", dpi=300)
        figure.savefig(figures_directory / f"{filename}.pdf")
        result.append((title, figure))
    return result


def make_combined_comparison_plots(
    records_by_dataset: dict[str, list[dict[str, object]]],
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    run_scope: str = "test",
) -> list[tuple[str, plt.Figure]]:
    """Create plots with 1D and 2D results shown side by side."""
    if run_scope not in {"test", "full"}:
        raise ValueError("run_scope must be 'test' or 'full'")
    dataset_names = "_".join(records_by_dataset)
    summary_name = f"results_summary_{dataset_names}_{run_scope}"
    figures_directory = Path(results_root) / summary_name / "figures"
    figures_directory.mkdir(parents=True, exist_ok=True)
    dimension_records = sorted(
        records_by_dataset.values(),
        key=lambda records: int(records[0]["spatial_dimension"]),
    )
    if [int(records[0]["spatial_dimension"]) for records in dimension_records] != [1, 2]:
        raise ValueError("combined plots require one 1D and one 2D dataset")

    # Keep the detailed curves, an overall summary, and the two distribution plots.
    figures = [
        (
            "Mean surrogate-error curves",
            _plot_combined_mean_error_curves(dimension_records),
            "mean_error_curves",
        ),
        (
            "Overall mean surrogate-error curves",
            _plot_overall_mean_error_curves(dimension_records),
            "overall_mean_error_curves",
        ),
        (
            "Runtime speedup",
            _plot_combined_runtime_speedup(dimension_records),
            "runtime_speedup",
        ),
        (
            "Surrogate-error gap by field family",
            _plot_combined_gap_by_family(dimension_records),
            "error_gap_by_field_family",
        ),
    ]

    result: list[tuple[str, plt.Figure]] = []
    for title, figure, filename in figures:
        figure.savefig(figures_directory / f"{filename}.png", dpi=300)
        figure.savefig(figures_directory / f"{filename}.pdf")
        result.append((title, figure))
    return result


def _dataset_results_directory(
    results_root: str | Path,
    dataset: str,
    run_scope: str,
) -> Path:
    """Build the dataset-specific directory for a test or full run."""
    if run_scope not in {"test", "full"}:
        raise ValueError("run_scope must be 'test' or 'full'")
    return Path(results_root) / f"{dataset}_results_{run_scope}"


def paired_summary(records: list[dict[str, object]]) -> dict[str, float | int]:
    """Return the main quality and runtime comparison values."""
    greedy_gaps = [float(record["greedy_error_gap"]) for record in records]
    tree_aware_gaps = [
        float(record["tree_aware_error_gap"]) for record in records
    ]
    uniform_gaps = [float(record["uniform_error_gap"]) for record in records]
    random_gaps = [float(record["random_error_gap"]) for record in records]
    return {
        "cases": len({record["case"] for record in records}),
        "runs": len(records),
        "greedy_mean_gap": fmean(greedy_gaps),
        "tree_aware_mean_gap": fmean(tree_aware_gaps),
        "uniform_mean_gap": fmean(uniform_gaps),
        "random_mean_gap": fmean(random_gaps),
        "greedy_maximum_gap": max(greedy_gaps),
        "tree_aware_maximum_gap": max(tree_aware_gaps),
        "uniform_maximum_gap": max(uniform_gaps),
        "random_maximum_gap": max(random_gaps),
        "greedy_exact_match_rate": fmean(
            float(bool(record["greedy_exact_error_match"])) for record in records
        ),
        "tree_aware_exact_match_rate": fmean(
            float(bool(record["tree_aware_exact_error_match"]))
            for record in records
        ),
        "uniform_exact_match_rate": fmean(
            float(bool(record["uniform_exact_error_match"])) for record in records
        ),
        "random_exact_match_rate": fmean(
            float(bool(record["random_exact_error_match"])) for record in records
        ),
        "greedy_median_speedup": median(
            float(record["greedy_runtime_speedup"]) for record in records
        ),
        "tree_aware_median_speedup": median(
            float(record["tree_aware_runtime_speedup"]) for record in records
        ),
        "uniform_median_speedup": median(
            float(record["uniform_runtime_speedup"]) for record in records
        ),
        "random_median_speedup": median(
            float(record["random_runtime_speedup"]) for record in records
        ),
    }


def overall_error_curve_summary(
    records: list[dict[str, object]],
) -> list[dict[str, float | int | str]]:
    """Summarize overall surrogate-error curves."""
    dimension = f"{int(records[0]['spatial_dimension'])}D"
    rows: list[dict[str, float | int | str]] = []
    levels = sorted({int(record["budget_level"]) for record in records})
    for level in levels:
        selected = [
            record for record in records if int(record["budget_level"]) == level
        ]
        optimal_error = fmean(float(record["optimal_error"]) for record in selected)
        greedy_error = fmean(float(record["greedy_error"]) for record in selected)
        heuristic_error = fmean(
            float(record["tree_aware_error"]) for record in selected
        )
        uniform_error = fmean(
            float(record["uniform_error"]) for record in selected
        )
        random_error = fmean(
            float(record["random_error"]) for record in selected
        )
        optimal_from_greedy = greedy_error - optimal_error
        optimal_from_heuristic = heuristic_error - optimal_error
        heuristic_from_greedy = greedy_error - heuristic_error
        rows.append(
            {
                "dimension": dimension,
                "budget_level": level,
                "mean_budget_fraction": fmean(
                    float(record["budget_fraction_of_N_min"])
                    for record in selected
                ),
                "optimal_mean_error": optimal_error,
                "greedy_mean_error": greedy_error,
                "heuristic_mean_error": heuristic_error,
                "uniform_mean_error": uniform_error,
                "random_mean_error": random_error,
                "optimal_from_greedy": optimal_from_greedy,
                "optimal_from_greedy_percent": (
                    100.0 * optimal_from_greedy / greedy_error
                    if greedy_error > 0.0
                    else 0.0
                ),
                "optimal_from_heuristic": optimal_from_heuristic,
                "optimal_from_heuristic_percent": (
                    100.0 * optimal_from_heuristic / heuristic_error
                    if heuristic_error > 0.0
                    else 0.0
                ),
                "heuristic_from_greedy": heuristic_from_greedy,
                "heuristic_from_greedy_percent": (
                    100.0 * heuristic_from_greedy / greedy_error
                    if greedy_error > 0.0
                    else 0.0
                ),
            }
        )
    return rows


def runtime_by_k_summary(
    records: list[dict[str, object]],
) -> list[dict[str, float | int | str]]:
    """Summarize runtime and optimal-search speedup for each K value."""
    dimension = f"{int(records[0]['spatial_dimension'])}D"
    rows: list[dict[str, float | int | str]] = []
    for number_of_fields in sorted({int(record["K"]) for record in records}):
        selected = [
            record
            for record in records
            if int(record["K"]) == number_of_fields
        ]
        rows.append(
            {
                "dimension": dimension,
                "K": number_of_fields,
                "optimal_median_seconds": median(
                    float(record["optimal_seconds"]) for record in selected
                ),
                "greedy_median_seconds": median(
                    float(record["greedy_seconds"]) for record in selected
                ),
                "heuristic_median_seconds": median(
                    float(record["tree_aware_seconds"]) for record in selected
                ),
                "uniform_median_seconds": median(
                    float(record["uniform_seconds"]) for record in selected
                ),
                "random_median_seconds": median(
                    float(record["random_seconds"]) for record in selected
                ),
                "greedy_median_speedup": median(
                    float(record["greedy_runtime_speedup"])
                    for record in selected
                ),
                "heuristic_median_speedup": median(
                    float(record["tree_aware_runtime_speedup"])
                    for record in selected
                ),
                "uniform_median_speedup": median(
                    float(record["uniform_runtime_speedup"])
                    for record in selected
                ),
                "random_median_speedup": median(
                    float(record["random_runtime_speedup"])
                    for record in selected
                ),
            }
        )
    return rows


def field_family_summary(
    records: list[dict[str, object]],
) -> list[dict[str, float | int | str]]:
    """Summarize the mean surrogate-error gap for each field family."""
    dimension = f"{int(records[0]['spatial_dimension'])}D"
    present = {str(record["field_model"]) for record in records}
    families = [family for family in FIELD_FAMILY_ORDER if family in present]
    families.extend(sorted(present - set(FIELD_FAMILY_ORDER)))
    rows: list[dict[str, float | int | str]] = []
    for family in families:
        selected = [
            record for record in records if str(record["field_model"]) == family
        ]
        greedy_gap = fmean(
            float(record["greedy_error_gap"]) for record in selected
        )
        heuristic_gap = fmean(
            float(record["tree_aware_error_gap"]) for record in selected
        )
        reduction = greedy_gap - heuristic_gap
        rows.append(
            {
                "dimension": dimension,
                "field_family": family,
                "greedy_mean_gap": greedy_gap,
                "heuristic_mean_gap": heuristic_gap,
                "uniform_mean_gap": fmean(
                    float(record["uniform_error_gap"]) for record in selected
                ),
                "random_mean_gap": fmean(
                    float(record["random_error_gap"]) for record in selected
                ),
                "heuristic_gap_reduction": reduction,
                "heuristic_gap_reduction_percent": (
                    100.0 * reduction / greedy_gap if greedy_gap > 0.0 else 0.0
                ),
            }
        )
    return rows


def _validate_strategy(
    allocation: tuple[int, ...],
    pair_locations: dict[tuple[int, int], int],
    error: float,
    K: int,
    M: int,
    budget: int,
) -> None:
    """Check one algorithm result before recording it."""
    used_sensors = sum(allocation)
    if (
        len(allocation) != M
        or any(count < 0 for count in allocation)
        or used_sensors != budget
    ):
        raise RuntimeError("algorithm returned an invalid allocation")
    pairs = {(i, j) for i in range(K) for j in range(i + 1, K)}
    if set(pair_locations) != pairs:
        raise RuntimeError("algorithm returned invalid pair locations")
    if any(not 0 <= location < M for location in pair_locations.values()):
        raise RuntimeError("algorithm returned an out-of-range location")
    if not 0.0 <= error <= 1.0:
        raise RuntimeError("algorithm returned an invalid error probability")


def _tree_to_dict(tree: object) -> dict[str, object]:
    """Convert either algorithm's decision tree to JSON data."""
    if hasattr(tree, "declared_field"):
        return {"declared_field": int(getattr(tree, "declared_field"))}
    return {
        "pair": list(getattr(tree, "pair")),
        "location": int(getattr(tree, "location")),
        "pair_error": float(getattr(tree, "pair_error")),
        "discard_i": _tree_to_dict(getattr(tree, "discard_i")),
        "discard_j": _tree_to_dict(getattr(tree, "discard_j")),
    }


def _pair_locations_to_dict(
    locations: dict[tuple[int, int], int],
) -> dict[str, int]:
    """Convert pair-location tuple keys to JSON string keys."""
    return {f"{i},{j}": location for (i, j), location in locations.items()}


def _case_random_parameters(
    parameters: dict[str, object],
    dataset: str,
    case: dict[str, object],
    budget: int,
) -> dict[str, object]:
    """Create a stable random seed for one case and sensor budget."""
    case_parameters = dict(parameters)
    base_seed = case_parameters.get("random_seed", 0)
    if isinstance(base_seed, int) and not isinstance(base_seed, bool):
        seed_text = f"{base_seed}|{dataset}|{case['file']}|{budget}"
        case_parameters["random_seed"] = crc32(seed_text.encode("utf-8"))
    return case_parameters


def _mean_and_sem(values: list[float]) -> tuple[float, float]:
    """Return the sample mean and standard error."""
    if len(values) < 2:
        return fmean(values), 0.0
    return fmean(values), stdev(values) / sqrt(len(values))


def _plot_mean_error_curves(
    records: list[dict[str, object]], dimension: int
) -> plt.Figure:
    """Compare mean error curves for every algorithm at each K."""
    K_values = sorted({int(record["K"]) for record in records})
    figure, axes = plt.subplots(
        1, len(K_values), figsize=(5.2 * len(K_values), 4.4), sharey=True
    )
    axes = np.atleast_1d(axes)
    for axis, K in zip(axes, K_values):
        K_records = [record for record in records if record["K"] == K]
        levels = sorted({int(record["budget_level"]) for record in K_records})
        for algorithm, error_key, line_style in (
            ("Optimal", "optimal_error", "-"),
            ("Greedy", "greedy_error", "-"),
            ("Heuristic", "tree_aware_error", "-"),
            ("Uniform", "uniform_error", "-"),
            ("Random", "random_error", "-"),
        ):
            x_values = []
            means = []
            for level in levels:
                selected = [
                    record
                    for record in K_records
                    if record["budget_level"] == level
                ]
                x_values.append(
                    fmean(
                        float(record["budget_fraction_of_N_min"])
                        for record in selected
                    )
                )
                means.append(
                    fmean(float(record[error_key]) for record in selected)
                )
            axis.plot(
                x_values,
                means,
                marker="o",
                linestyle=line_style,
                color=ALGORITHM_COLORS[algorithm],
                label=algorithm,
            )
        axis.set_title(f"K={K}")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean surrogate final error")
    axes[-1].legend()
    figure.supxlabel(
        r"Budget ratio $\rho=N/N_{\min}^{opt}$"
    )
    figure.tight_layout()
    return figure


def _plot_error_scatter(
    records: list[dict[str, object]], dimension: int
) -> plt.Figure:
    """Plot greedy error against optimal error with an equality line."""
    figure, axis = plt.subplots(figsize=(6.2, 5.5))
    for K in sorted({int(record["K"]) for record in records}):
        selected = [record for record in records if record["K"] == K]
        axis.scatter(
            [float(record["optimal_error"]) for record in selected],
            [float(record["greedy_error"]) for record in selected],
            alpha=0.75,
            label=f"K={K}",
        )
    limit = max(
        max(float(record["optimal_error"]) for record in records),
        max(float(record["greedy_error"]) for record in records),
    )
    axis.plot([0, limit], [0, limit], color="black", linestyle="--", label="Equal error")
    axis.set(
        xlabel="Optimal error",
        ylabel="Greedy error",
        xlim=(0, limit * 1.03),
        ylim=(0, limit * 1.03),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def _plot_error_gap(
    records: list[dict[str, object]], dimension: int
) -> plt.Figure:
    """Plot the mean greedy-minus-optimal surrogate-error gap by budget level."""
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for K in sorted({int(record["K"]) for record in records}):
        selected = [record for record in records if record["K"] == K]
        levels = sorted({int(record["budget_level"]) for record in selected})
        means = []
        errors = []
        for level in levels:
            values = [
                float(record["greedy_error_gap"])
                for record in selected
                if record["budget_level"] == level
            ]
            mean, sem = _mean_and_sem(values)
            means.append(mean)
            errors.append(sem)
        axis.errorbar(levels, means, yerr=errors, marker="o", capsize=3, label=f"K={K}")
    axis.axhline(0, color="black", linewidth=1)
    axis.set(
        xlabel="Budget level (smallest to largest)",
        ylabel=r"Mean surrogate-error gap $P_{greedy}-P_{optimal}$",
        xticks=sorted({int(record["budget_level"]) for record in records}),
        ylim=(0, None),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def _plot_match_heatmap(
    records: list[dict[str, object]], dimension: int
) -> plt.Figure:
    """Show the exact-error match rate for every K and M combination."""
    K_values = sorted({int(record["K"]) for record in records})
    M_values = sorted({int(record["M"]) for record in records})
    matrix = np.empty((len(K_values), len(M_values)))
    for row, K in enumerate(K_values):
        for column, M in enumerate(M_values):
            matrix[row, column] = fmean(
                float(bool(record["greedy_exact_error_match"]))
                for record in records
                if record["K"] == K and record["M"] == M
            )

    figure, axis = plt.subplots(figsize=(6.2, 4.5))
    image = axis.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axis.set_xticks(range(len(M_values)), M_values)
    axis.set_yticks(range(len(K_values)), K_values)
    axis.set(
        xlabel="Number of locations M",
        ylabel="Number of fields K",
    )
    for row in range(len(K_values)):
        for column in range(len(M_values)):
            color = "white" if matrix[row, column] > 0.55 else "black"
            axis.text(
                column,
                row,
                f"{100 * matrix[row, column]:.0f}%",
                ha="center",
                va="center",
                color=color,
            )
    figure.colorbar(image, ax=axis, label="Exact-error match rate")
    figure.tight_layout()
    return figure


def _plot_runtime_speedup(
    records: list[dict[str, object]], dimension: int
) -> plt.Figure:
    """Compare optimal-to-greedy runtime ratios across K values."""
    K_values = sorted({int(record["K"]) for record in records})
    values = [
        [
            float(record["greedy_runtime_speedup"])
            for record in records
            if record["K"] == K
        ]
        for K in K_values
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.boxplot(values, tick_labels=[f"K={K}" for K in K_values], showmeans=True)
    axis.axhline(1, color="black", linestyle="--", linewidth=1)
    axis.set_yscale("log")
    axis.set(
        xlabel="Number of fields",
        ylabel="Runtime speedup (optimal / greedy, log scale)",
    )
    axis.grid(axis="y", alpha=0.25, which="both")
    figure.tight_layout()
    return figure


def _plot_gap_by_family(
    records: list[dict[str, object]], dimension: int
) -> plt.Figure:
    """Compare additive surrogate-error gaps across field families."""
    present_families = {str(record["field_model"]) for record in records}
    families = [
        family for family in FIELD_FAMILY_ORDER if family in present_families
    ]
    families.extend(sorted(present_families - set(FIELD_FAMILY_ORDER)))
    values = [
        [
            float(record["greedy_error_gap"])
            for record in records
            if record["field_model"] == family
        ]
        for family in families
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.boxplot(values, tick_labels=families, showmeans=True)
    axis.set(
        xlabel="Field family",
        ylabel=r"Surrogate-error gap $P_{greedy}-P_{optimal}$",
        ylim=(0, None),
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def _plot_combined_mean_error_curves(
    dimension_records: list[list[dict[str, object]]],
) -> plt.Figure:
    """Compare all algorithms with dimensions in columns and K in rows."""
    K_values = sorted(
        {int(record["K"]) for records in dimension_records for record in records}
    )
    figure, axes = plt.subplots(
        len(K_values),
        2,
        figsize=(12.5, 3.7 * len(K_values)),
        sharey=True,
        constrained_layout=True,
    )
    for column, records in enumerate(dimension_records):
        dimension = int(records[0]["spatial_dimension"])
        for row, K in enumerate(K_values):
            axis = axes[row, column]
            selected_K = [record for record in records if record["K"] == K]
            levels = sorted(
                {int(record["budget_level"]) for record in selected_K}
            )
            for algorithm, error_key, line_style in (
                ("Optimal", "optimal_error", "-"),
                ("Greedy", "greedy_error", "-"),
                ("Heuristic", "tree_aware_error", "-"),
                ("Uniform", "uniform_error", "-"),
                ("Random", "random_error", "-"),
            ):
                x_values = []
                means = []
                for level in levels:
                    selected = [
                        record
                        for record in selected_K
                        if record["budget_level"] == level
                    ]
                    x_values.append(
                        fmean(
                            float(record["budget_fraction_of_N_min"])
                            for record in selected
                        )
                    )
                    means.append(
                        fmean(float(record[error_key]) for record in selected)
                    )
                axis.plot(
                    x_values,
                    means,
                    marker="o",
                    linestyle=line_style,
                    color=ALGORITHM_COLORS[algorithm],
                    label=algorithm,
                )
            axis.set_title(f"{dimension}D, K={K}")
            axis.grid(alpha=0.25)
            if column == 0:
                axis.set_ylabel("Mean surrogate final error")
    axes[0, 1].legend()
    figure.supxlabel(
        r"Budget ratio $\rho=N/N_{\min}^{opt}$"
    )
    return figure


def _plot_overall_mean_error_curves(
    dimension_records: list[list[dict[str, object]]],
) -> plt.Figure:
    """Plot overall mean error with one panel for each dimension."""
    figure, axes = plt.subplots(
        1, 2, figsize=(12.8, 4.8), sharey=True, constrained_layout=True
    )
    for axis, records in zip(axes, dimension_records):
        dimension = int(records[0]["spatial_dimension"])
        levels = sorted({int(record["budget_level"]) for record in records})
        for algorithm, error_key, line_style in (
            ("Optimal", "optimal_error", "-"),
            ("Greedy", "greedy_error", "-"),
            ("Heuristic", "tree_aware_error", "-"),
            ("Uniform", "uniform_error", "-"),
            ("Random", "random_error", "-"),
        ):
            x_values: list[float] = []
            means: list[float] = []
            for level in levels:
                selected = [
                    record
                    for record in records
                    if int(record["budget_level"]) == level
                ]
                x_values.append(
                    fmean(
                        float(record["budget_fraction_of_N_min"])
                        for record in selected
                    )
                )
                means.append(
                    fmean(float(record[error_key]) for record in selected)
                )
            axis.plot(
                x_values,
                means,
                marker="o",
                linestyle=line_style,
                color=ALGORITHM_COLORS[algorithm],
                label=algorithm,
            )
        axis.set_title(f"{dimension}D")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean surrogate final error")
    axes[1].legend()
    figure.supxlabel(
        r"Budget ratio $\rho=N/N_{\min}^{opt}$"
    )
    return figure


def _plot_combined_error_scatter(
    dimension_records: list[list[dict[str, object]]],
) -> plt.Figure:
    """Plot paired errors with 1D and 2D in side-by-side panels."""
    limit = max(
        float(record[key])
        for records in dimension_records
        for record in records
        for key in ("optimal_error", "greedy_error", "uniform_error")
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(12.2, 5.3), sharex=True, sharey=True, constrained_layout=True
    )
    for axis, records in zip(axes, dimension_records):
        dimension = int(records[0]["spatial_dimension"])
        K_colors = {3: "#1f77b4", 4: "#7b3294", 5: "#ff7f0e"}
        for K in sorted({int(record["K"]) for record in records}):
            selected = [record for record in records if record["K"] == K]
            axis.scatter(
                [float(record["optimal_error"]) for record in selected],
                [float(record["greedy_error"]) for record in selected],
                alpha=0.75,
                color=K_colors[K],
                marker="o",
                label=f"Greedy, K={K}",
            )
            axis.scatter(
                [float(record["optimal_error"]) for record in selected],
                [float(record["uniform_error"]) for record in selected],
                alpha=0.75,
                color=K_colors[K],
                marker="x",
                label=f"Uniform, K={K}",
            )
        axis.plot(
            [0, limit],
            [0, limit],
            color="black",
            linestyle="--",
            label="Equal error",
        )
        axis.set(
            xlabel="Optimal error",
            title=f"{dimension}D",
            xlim=(0, limit * 1.03),
            ylim=(0, limit * 1.03),
        )
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Heuristic surrogate final error")
    axes[1].legend()
    return figure


def _plot_combined_error_gap(
    dimension_records: list[list[dict[str, object]]],
) -> plt.Figure:
    """Plot heuristic surrogate-error gaps with dimensions in columns and K in rows."""
    K_values = sorted(
        {int(record["K"]) for records in dimension_records for record in records}
    )
    figure, axes = plt.subplots(
        len(K_values),
        2,
        figsize=(12.5, 4.0 * len(K_values)),
        sharey=True,
        constrained_layout=True,
    )
    maximum_gap = max(
        float(record[key])
        for records in dimension_records
        for record in records
        for key in ("greedy_error_gap", "uniform_error_gap")
    )
    for column, records in enumerate(dimension_records):
        dimension = int(records[0]["spatial_dimension"])
        for row, K in enumerate(K_values):
            axis = axes[row, column]
            selected = [record for record in records if record["K"] == K]
            levels = sorted(
                {int(record["budget_level"]) for record in selected}
            )
            for algorithm, gap_key, line_style in (
                ("Greedy", "greedy_error_gap", "--"),
                ("Uniform", "uniform_error_gap", ":"),
            ):
                means = []
                errors = []
                for level in levels:
                    values = [
                        float(record[gap_key])
                        for record in selected
                        if record["budget_level"] == level
                    ]
                    mean, sem = _mean_and_sem(values)
                    means.append(mean)
                    errors.append(sem)
                axis.errorbar(
                    levels,
                    means,
                    yerr=errors,
                    marker="o",
                    linestyle=line_style,
                    capsize=3,
                    color=ALGORITHM_COLORS[algorithm],
                    label=algorithm,
                )
            axis.axhline(0, color="black", linewidth=1)
            axis.set(
                title=f"{dimension}D, K={K}",
                xticks=levels,
                ylim=(0, maximum_gap * 1.08),
            )
            if column == 0:
                axis.set_ylabel(r"Mean surrogate-error gap $P_{method}-P_{optimal}$")
            axis.grid(alpha=0.25)
    axes[0, 1].legend()
    figure.supxlabel("Budget level (smallest to largest)")
    return figure


def _plot_combined_match_heatmap(
    dimension_records: list[list[dict[str, object]]],
) -> plt.Figure:
    """Show exact-error match heatmaps by method and dimension."""
    figure, axes = plt.subplots(
        2, 2, figsize=(11.5, 8.2), sharex=True, sharey=True, constrained_layout=True
    )
    image = None
    for column, records in enumerate(dimension_records):
        dimension = int(records[0]["spatial_dimension"])
        K_values = sorted({int(record["K"]) for record in records})
        M_values = sorted({int(record["M"]) for record in records})
        for row, (algorithm, match_key) in enumerate(
            (
                ("Greedy", "greedy_exact_error_match"),
                ("Uniform", "uniform_exact_error_match"),
            )
        ):
            axis = axes[row, column]
            matrix = np.empty((len(K_values), len(M_values)))
            for K_index, K in enumerate(K_values):
                for M_index, M in enumerate(M_values):
                    matrix[K_index, M_index] = fmean(
                        float(bool(record[match_key]))
                        for record in records
                        if record["K"] == K and record["M"] == M
                    )
            image = axis.imshow(
                matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1
            )
            axis.set_xticks(range(len(M_values)), M_values)
            axis.set_yticks(range(len(K_values)), K_values)
            axis.set_title(f"{dimension}D, {algorithm}")
            if row == 1:
                axis.set_xlabel("Number of locations M")
            if column == 0:
                axis.set_ylabel("Number of fields K")
            for K_index in range(len(K_values)):
                for M_index in range(len(M_values)):
                    color = (
                        "white" if matrix[K_index, M_index] > 0.55 else "black"
                    )
                    axis.text(
                        M_index,
                        K_index,
                        f"{100 * matrix[K_index, M_index]:.0f}%",
                        ha="center",
                        va="center",
                        color=color,
                    )
    assert image is not None
    figure.colorbar(image, ax=axes, label="Exact-error match rate")
    return figure


def _plot_combined_runtime_speedup(
    dimension_records: list[list[dict[str, object]]],
) -> plt.Figure:
    """Show runtime-speedup distributions side by side."""
    figure, axes = plt.subplots(
        1, 2, figsize=(13.2, 4.8), sharey=True, constrained_layout=True
    )
    for axis, records in zip(axes, dimension_records):
        dimension = int(records[0]["spatial_dimension"])
        K_values = sorted({int(record["K"]) for record in records})
        positions = np.arange(1, len(K_values) + 1)
        for algorithm, speedup_key, offset in (
            ("Greedy", "greedy_runtime_speedup", -0.3),
            ("Heuristic", "tree_aware_runtime_speedup", -0.1),
            ("Uniform", "uniform_runtime_speedup", 0.1),
            ("Random", "random_runtime_speedup", 0.3),
        ):
            values = [
                [
                    float(record[speedup_key])
                    for record in records
                    if record["K"] == K
                ]
                for K in K_values
            ]
            boxplot = axis.boxplot(
                values,
                positions=positions + offset,
                widths=0.18,
                patch_artist=True,
                showmeans=True,
                meanprops={"markerfacecolor": "black", "markeredgecolor": "black"},
                medianprops={"color": "black"},
            )
            for box in boxplot["boxes"]:
                box.set_facecolor(ALGORITHM_COLORS[algorithm])
                box.set_alpha(0.45)
            boxplot["boxes"][0].set_label(algorithm)
        axis.axhline(1, color="black", linestyle="--", linewidth=1)
        axis.set_yscale("log")
        axis.set(
            xlabel="Number of fields",
            title=f"{dimension}D",
            xticks=positions,
            xticklabels=[f"K={K}" for K in K_values],
        )
        axis.grid(axis="y", alpha=0.25, which="both")
    axes[0].set_ylabel("Runtime speedup (optimal / method, log scale)")
    axes[1].legend()
    return figure


def _plot_combined_gap_by_family(
    dimension_records: list[list[dict[str, object]]],
) -> plt.Figure:
    """Compare field-family surrogate-error gaps in side-by-side panels."""
    figure, axes = plt.subplots(
        1, 2, figsize=(13.2, 4.8), sharey=True, constrained_layout=True
    )
    for axis, records in zip(axes, dimension_records):
        dimension = int(records[0]["spatial_dimension"])
        present_families = {str(record["field_model"]) for record in records}
        families = [
            family for family in FIELD_FAMILY_ORDER if family in present_families
        ]
        families.extend(sorted(present_families - set(FIELD_FAMILY_ORDER)))
        positions = np.arange(1, len(families) + 1)
        for algorithm, gap_key, offset in (
            ("Greedy", "greedy_error_gap", -0.3),
            ("Heuristic", "tree_aware_error_gap", -0.1),
            ("Uniform", "uniform_error_gap", 0.1),
            ("Random", "random_error_gap", 0.3),
        ):
            values = [
                [
                    float(record[gap_key])
                    for record in records
                    if record["field_model"] == family
                ]
                for family in families
            ]
            boxplot = axis.boxplot(
                values,
                positions=positions + offset,
                widths=0.18,
                patch_artist=True,
                showmeans=True,
                meanprops={"markerfacecolor": "black", "markeredgecolor": "black"},
                medianprops={"color": "black"},
            )
            for box in boxplot["boxes"]:
                box.set_facecolor(ALGORITHM_COLORS[algorithm])
                box.set_alpha(0.45)
            boxplot["boxes"][0].set_label(algorithm)
        axis.set(
            xlabel="Field family",
            title=f"{dimension}D",
            ylim=(0, None),
            xticks=positions,
            xticklabels=families,
        )
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(r"Surrogate-error gap $P_{method}-P_{optimal}$")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(labels),
    )
    return figure
