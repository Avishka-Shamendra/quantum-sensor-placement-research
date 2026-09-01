"""Actual quantum error evaluation for returned sequential strategies."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, cos, isfinite, pi, sin, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import greedy as greedy_module
import heuristic as heuristic_module
from approximate_comparison_analysis import (
    DEFAULT_RESULTS_ROOT as DEFAULT_EXPANDED_RESULTS_ROOT,
)
from approximate_comparison_analysis import load_approximate_results
from comparison_analysis import DEFAULT_RESULTS_ROOT, load_paired_results
from input_data import PhaseGaps, principal_phase
from plot_style import configure_plot_style


HERE = Path(__file__).resolve().parent
DEFAULT_ACTUAL_RESULTS_ROOT = HERE / "results" / "actual_error_evaluation"

METHODS = {
    "optimal": "Exhaustive",
    "greedy": "Greedy",
    "tree_aware": "Heuristic",
}
EXPANDED_METHODS = {
    "greedy": "Greedy",
    "heuristic": "Heuristic",
}
METHOD_COLORS = {
    "Exhaustive": "#1f77b4",
    "Greedy": "#ff7f0e",
    "Heuristic": "#d62728",
}

configure_plot_style()


@dataclass(frozen=True)
class NodeProbabilities:
    """Two outcome probabilities for one true field at one tree node."""

    keep_i: float
    keep_j: float


def actual_node_probabilities(
    pair: tuple[int, int],
    location: int,
    sensor_count: int,
    true_field: int,
    phase_gaps: PhaseGaps,
) -> NodeProbabilities:
    """Return actual Helstrom outcome probabilities for one pairwise node.

    ``keep_i`` is the probability of the outcome favoring field ``i``; in the
    elimination trees used here, that probability flows to ``discard_j``.
    """
    if sensor_count < 0:
        raise ValueError("sensor_count must be nonnegative")

    i, j = pair
    phase = _signed_phase_difference(i, j, location, phase_gaps)
    gap = abs(phase)
    if sensor_count == 0 or gap == 0.0:
        return NodeProbabilities(keep_i=0.5, keep_j=0.5)

    test_sensor_count = _sensors_used_for_test(gap, sensor_count)
    components = _probe_components(gap, test_sensor_count)
    field_phases = _field_phases(location, phase_gaps)
    state_i = _output_state(i, location, components, field_phases)
    state_j = _output_state(j, location, components, field_phases)
    state_k = _output_state(true_field, location, components, field_phases)

    rho_i = np.outer(state_i, state_i.conjugate())
    rho_j = np.outer(state_j, state_j.conjugate())
    eigenvalues, eigenvectors = np.linalg.eigh(rho_i - rho_j)

    positive = eigenvalues > 1e-12
    zero = np.abs(eigenvalues) <= 1e-12
    projected_positive = eigenvectors[:, positive].conjugate().T @ state_k
    projected_zero = eigenvectors[:, zero].conjugate().T @ state_k
    keep_i = float(
        np.vdot(projected_positive, projected_positive).real
        + 0.5 * np.vdot(projected_zero, projected_zero).real
    )
    keep_i = min(1.0, max(0.0, keep_i))
    return NodeProbabilities(keep_i=keep_i, keep_j=1.0 - keep_i)


def actual_tree_error(
    tree: Mapping[str, object] | object,
    K: int,
    phase_gaps: PhaseGaps,
    allocation: Sequence[int],
) -> float:
    """Evaluate a returned tree using physical third-field probabilities."""
    if K < 2:
        raise ValueError("K must be at least 2")
    if len(allocation) == 0:
        raise ValueError("allocation must contain at least one location")

    success = 0.0
    for true_field in range(K):
        success += _leaf_success_probability(
            tree, true_field, phase_gaps, allocation
        )
    return 1.0 - success / K


def surrogate_tree_error(tree: Mapping[str, object] | object, K: int) -> float:
    """Re-evaluate a tree using the current 0.5/0.5 outside-pair rule."""
    error = 0.0
    for true_field in range(K):
        error += 1.0 - _surrogate_leaf_success_probability(tree, true_field)
    return error / K


def evaluate_actual_records(
    dataset: str,
    records: list[dict[str, object]],
    *,
    input_root: str | Path = HERE / "inputs",
    progress_every: int | None = None,
    methods: Mapping[str, str] = METHODS,
) -> list[dict[str, object]]:
    """Attach actual errors and evaluation gaps to paired comparison records."""
    dataset_directory = Path(input_root) / dataset
    input_cache: dict[str, tuple[int, int, PhaseGaps, object]] = {}
    evaluated: list[dict[str, object]] = []

    for record_number, record in enumerate(records, start=1):
        case_name = str(record["case"])
        input_data = input_cache.get(case_name)
        if input_data is None:
            from input_data import load_sparse_input

            input_data = load_sparse_input(
                dataset_directory / f"{case_name}.csv"
            )
            input_cache[case_name] = input_data
        _, _, phase_gaps, pair_error_table = input_data

        row = dict(record)
        for prefix in methods:
            allocation = [int(value) for value in row[f"{prefix}_allocation"]]
            tree = _record_tree(
                row,
                prefix,
                int(row["K"]),
                int(row["M"]),
                allocation,
                pair_error_table,
            )
            surrogate_error = float(row[f"{prefix}_error"])
            actual_error = actual_tree_error(
                tree, int(row["K"]), phase_gaps, allocation
            )
            recomputed_surrogate = surrogate_tree_error(tree, int(row["K"]))
            row[f"{prefix}_actual_error"] = actual_error
            row[f"{prefix}_actual_eval_gap"] = actual_error - surrogate_error
            row[f"{prefix}_surrogate_recomputed_error"] = recomputed_surrogate
            row[f"{prefix}_surrogate_recompute_delta"] = (
                recomputed_surrogate - surrogate_error
            )

        row["tree_aware_actual_gap_vs_optimal"] = (
            float(row["tree_aware_actual_error"]) - float(row["optimal_actual_error"])
            if "tree_aware_actual_error" in row and "optimal_actual_error" in row
            else None
        )
        row["greedy_actual_gap_vs_optimal"] = (
            float(row["greedy_actual_error"]) - float(row["optimal_actual_error"])
            if "greedy_actual_error" in row and "optimal_actual_error" in row
            else None
        )
        row["tree_aware_surrogate_gap_vs_optimal"] = (
            float(row["tree_aware_error"]) - float(row["optimal_error"])
            if "tree_aware_error" in row and "optimal_error" in row
            else None
        )
        row["greedy_surrogate_gap_vs_optimal"] = (
            float(row["greedy_error"]) - float(row["optimal_error"])
            if "greedy_error" in row and "optimal_error" in row
            else None
        )
        greedy_actual_error = float(row["greedy_actual_error"])
        heuristic_prefix = (
            "tree_aware" if "tree_aware_actual_error" in row else "heuristic"
        )
        tree_aware_actual_error = float(row[f"{heuristic_prefix}_actual_error"])
        row[f"{heuristic_prefix}_actual_reduction_vs_greedy"] = (
            greedy_actual_error - tree_aware_actual_error
        )
        row[f"{heuristic_prefix}_actual_reduction_vs_greedy_percent"] = (
            100.0
            * (greedy_actual_error - tree_aware_actual_error)
            / greedy_actual_error
            if greedy_actual_error > 0.0
            else 0.0
        )
        evaluated.append(row)
        if (
            progress_every is not None
            and progress_every > 0
            and (
                record_number == 1
                or record_number % progress_every == 0
                or record_number == len(records)
            )
        ):
            print(
                f"{dataset}: evaluated {record_number}/{len(records)} records",
                flush=True,
            )

    return evaluated


def run_actual_error_evaluation(
    datasets: Sequence[str] = ("compact_benchmark_1d", "compact_benchmark_2d"),
    *,
    comparison_results_root: str | Path = DEFAULT_RESULTS_ROOT,
    results_root: str | Path = DEFAULT_ACTUAL_RESULTS_ROOT,
    run_scope: str = "test",
    progress_every: int | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Load paired results, evaluate actual errors, and save tables."""
    records_by_dataset: dict[str, list[dict[str, object]]] = {}
    for dataset_number, dataset in enumerate(datasets, start=1):
        print(
            f"[{dataset_number}/{len(datasets)}] Loading {dataset} {run_scope} records",
            flush=True,
        )
        records = load_paired_results(
            dataset,
            comparison_results_root,
            run_scope=run_scope,
        )
        print(
            f"[{dataset_number}/{len(datasets)}] Evaluating {len(records)} records",
            flush=True,
        )
        evaluated = evaluate_actual_records(
            dataset,
            records,
            progress_every=progress_every,
        )
        records_by_dataset[dataset] = evaluated
        save_actual_results(dataset, evaluated, results_root, run_scope=run_scope)
        print(
            f"[{dataset_number}/{len(datasets)}] Saved {dataset} {run_scope} tables",
            flush=True,
        )
    return records_by_dataset


def run_expanded_actual_error_evaluation(
    datasets: Sequence[str] = ("benchmark_1d", "benchmark_2d"),
    *,
    comparison_results_root: str | Path = DEFAULT_EXPANDED_RESULTS_ROOT,
    results_root: str | Path = DEFAULT_ACTUAL_RESULTS_ROOT,
    run_scope: str = "full",
    progress_every: int | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Evaluate saved expanded benchmark Greedy and Heuristic strategies."""
    records_by_dataset: dict[str, list[dict[str, object]]] = {}
    for dataset_number, dataset in enumerate(datasets, start=1):
        print(
            f"[{dataset_number}/{len(datasets)}] Loading {dataset} {run_scope} records",
            flush=True,
        )
        records = load_approximate_results(
            dataset,
            comparison_results_root,
            run_scope=run_scope,
        )
        print(
            f"[{dataset_number}/{len(datasets)}] Evaluating {len(records)} records",
            flush=True,
        )
        evaluated = evaluate_actual_records(
            dataset,
            records,
            progress_every=progress_every,
            methods=EXPANDED_METHODS,
        )
        records_by_dataset[dataset] = evaluated
        save_actual_results(
            dataset,
            evaluated,
            results_root,
            run_scope=f"expanded_{run_scope}",
            methods=EXPANDED_METHODS,
        )
        print(f"[{dataset_number}/{len(datasets)}] Saved {dataset} tables", flush=True)
    return records_by_dataset


def save_actual_results(
    dataset: str,
    records: list[dict[str, object]],
    results_root: str | Path = DEFAULT_ACTUAL_RESULTS_ROOT,
    *,
    run_scope: str = "test",
    methods: Mapping[str, str] = METHODS,
) -> None:
    """Save actual-error records and compact CSV summaries."""
    tables_directory = _dataset_results_directory(
        results_root, dataset, run_scope
    ) / "tables"
    tables_directory.mkdir(parents=True, exist_ok=True)
    (tables_directory / "results.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "result_count": len(records),
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
        "sensor_budget",
        "budget_level",
        "budget_fraction_of_N_min",
        *(f"{prefix}_error" for prefix in methods),
        *(f"{prefix}_actual_error" for prefix in methods),
        *(f"{prefix}_actual_eval_gap" for prefix in methods),
        *(f"{prefix}_surrogate_recompute_delta" for prefix in methods),
    ]
    optional_columns = [
        "greedy_surrogate_gap_vs_optimal",
        "greedy_actual_gap_vs_optimal",
        "tree_aware_surrogate_gap_vs_optimal",
        "tree_aware_actual_gap_vs_optimal",
        "tree_aware_actual_reduction_vs_greedy",
        "tree_aware_actual_reduction_vs_greedy_percent",
        "heuristic_actual_reduction_vs_greedy",
        "heuristic_actual_reduction_vs_greedy_percent",
    ]
    columns.extend(
        column
        for column in optional_columns
        if any(column in record for record in records)
    )
    with (tables_directory / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in columns})


def actual_error_summary(
    records: list[dict[str, object]],
    methods: Mapping[str, str] = METHODS,
) -> list[dict[str, float | int | str]]:
    """Return actual-vs-surrogate validation rows for each method."""
    rows: list[dict[str, float | int | str]] = []
    for method_key, method_name in methods.items():
        eval_gaps = [
            float(record[f"{method_key}_actual_eval_gap"]) for record in records
        ]
        rows.append(
            {
                "method": method_name,
                "runs": len(records),
                "mean_surrogate_error": float(
                    np.mean(
                        [
                            float(record[f"{method_key}_error"])
                            for record in records
                        ]
                    )
                ),
                "mean_actual_error": float(
                    np.mean(
                        [
                            float(record[f"{method_key}_actual_error"])
                            for record in records
                        ]
                    )
                ),
                "mean_eval_gap": float(np.mean(eval_gaps)),
                "mean_absolute_eval_gap": float(
                    np.mean([abs(value) for value in eval_gaps])
                ),
                "maximum_absolute_eval_gap": max(abs(value) for value in eval_gaps),
            }
        )
    return rows


def actual_method_comparison_summary(
    records: list[dict[str, object]],
) -> dict[str, float | int]:
    """Summarize method quality using actual final error."""
    greedy_actual_errors = [
        float(record["greedy_actual_error"]) for record in records
    ]
    heuristic_actual_errors = [
        float(record["tree_aware_actual_error"]) for record in records
    ]
    optimal_actual_errors = [
        float(record["optimal_actual_error"]) for record in records
    ]
    mean_greedy_actual = float(np.mean(greedy_actual_errors))
    mean_heuristic_actual = float(np.mean(heuristic_actual_errors))
    mean_reduction = mean_greedy_actual - mean_heuristic_actual
    return {
        "runs": len(records),
        "optimal_mean_actual_error": float(np.mean(optimal_actual_errors)),
        "greedy_mean_actual_error": mean_greedy_actual,
        "tree_aware_mean_actual_error": mean_heuristic_actual,
        "tree_aware_actual_reduction_vs_greedy": mean_reduction,
        "tree_aware_actual_reduction_vs_greedy_percent": (
            100.0 * mean_reduction / mean_greedy_actual
            if mean_greedy_actual > 0.0
            else 0.0
        ),
        "tree_aware_actual_gap_vs_optimal": float(
            np.mean(heuristic_actual_errors) - np.mean(optimal_actual_errors)
        ),
    }


def expanded_actual_method_comparison_summary(
    records: list[dict[str, object]],
) -> dict[str, float | int]:
    """Summarize expanded benchmark methods using actual final error."""
    greedy_actual_errors = [
        float(record["greedy_actual_error"]) for record in records
    ]
    heuristic_actual_errors = [
        float(record["heuristic_actual_error"]) for record in records
    ]
    mean_greedy_actual = float(np.mean(greedy_actual_errors))
    mean_heuristic_actual = float(np.mean(heuristic_actual_errors))
    mean_reduction = mean_greedy_actual - mean_heuristic_actual
    return {
        "runs": len(records),
        "greedy_mean_actual_error": mean_greedy_actual,
        "heuristic_mean_actual_error": mean_heuristic_actual,
        "heuristic_actual_reduction_vs_greedy": mean_reduction,
        "heuristic_actual_reduction_vs_greedy_percent": (
            100.0 * mean_reduction / mean_greedy_actual
            if mean_greedy_actual > 0.0
            else 0.0
        ),
    }


def surrogate_quality_summary(
    records_by_dataset: Mapping[str, list[dict[str, object]]],
    *,
    methods: Mapping[str, str] = METHODS,
) -> list[dict[str, float | str]]:
    """Correlate raw per-run surrogate and actual errors by dimension."""
    dimension_records = {
        int(records[0]["spatial_dimension"]): records
        for records in records_by_dataset.values()
    }
    rows: list[dict[str, float | str]] = []
    for method_key, method_name in methods.items():
        row: dict[str, float | str] = {"method": method_name}
        for dimension in (1, 2):
            records = dimension_records[dimension]
            surrogate = [float(record[f"{method_key}_error"]) for record in records]
            actual = [
                float(record[f"{method_key}_actual_error"]) for record in records
            ]
            row[f"{dimension}d_pearson"] = _pearson_correlation(surrogate, actual)
            row[f"{dimension}d_spearman"] = _spearman_correlation(surrogate, actual)
        rows.append(row)

    pooled_row: dict[str, float | str] = {"method": "All methods"}
    for dimension in (1, 2):
        records = dimension_records[dimension]
        surrogate = [
            float(record[f"{method_key}_error"])
            for record in records
            for method_key in methods
        ]
        actual = [
            float(record[f"{method_key}_actual_error"])
            for record in records
            for method_key in methods
        ]
        pooled_row[f"{dimension}d_pearson"] = _pearson_correlation(surrogate, actual)
        pooled_row[f"{dimension}d_spearman"] = _spearman_correlation(
            surrogate,
            actual,
        )
    rows.append(pooled_row)
    return rows


def bias_mae_summary(
    records_by_dataset: Mapping[str, list[dict[str, object]]],
    *,
    methods: Mapping[str, str] = METHODS,
) -> list[dict[str, float | str]]:
    """Summarize residual bias and MAE by method and dimension."""
    rows: list[dict[str, float | str]] = []
    for records in _records_by_dimension(records_by_dataset).values():
        dimension = int(records[0]["spatial_dimension"])
        for method_key, method_name in methods.items():
            residuals = [
                float(record[f"{method_key}_actual_error"])
                - float(record[f"{method_key}_error"])
                for record in records
            ]
            rows.append(
                {
                    "dimension": f"{dimension}D",
                    "method": method_name,
                    "bias": float(np.mean(residuals)),
                    "mae": float(np.mean([abs(value) for value in residuals])),
                }
            )
    return rows


def within_budget_correlation_summary(
    records_by_dataset: Mapping[str, list[dict[str, object]]],
    *,
    methods: Mapping[str, str] = METHODS,
) -> list[dict[str, float | int | str]]:
    """Correlate raw errors within each existing budget level."""
    rows: list[dict[str, float | int | str]] = []
    for records in _records_by_dimension(records_by_dataset).values():
        dimension = int(records[0]["spatial_dimension"])
        for budget_level in sorted({int(record["budget_level"]) for record in records}):
            selected = [
                record
                for record in records
                if int(record["budget_level"]) == budget_level
            ]
            surrogate = [
                float(record[f"{method_key}_error"])
                for record in selected
                for method_key in methods
            ]
            actual = [
                float(record[f"{method_key}_actual_error"])
                for record in selected
                for method_key in methods
            ]
            rows.append(
                {
                    "dimension": f"{dimension}D",
                    "budget_level": budget_level,
                    "mean_budget_ratio": float(
                        np.mean(
                            [
                                float(record["budget_fraction_of_N_min"])
                                for record in selected
                            ]
                        )
                    ),
                    "pearson": _pearson_correlation(surrogate, actual),
                    "spearman": _spearman_correlation(surrogate, actual),
                }
            )
    return rows


def method_ranking_agreement_summary(
    records_by_dataset: Mapping[str, list[dict[str, object]]],
    *,
    methods: Mapping[str, str] = METHODS,
    tie_tolerance: float = 1e-12,
) -> list[dict[str, float | int | str]]:
    """Compare surrogate-best and actual-best method rankings per run."""
    rows: list[dict[str, float | int | str]] = []
    for records in _records_by_dimension(records_by_dataset).values():
        dimension = int(records[0]["spatial_dimension"])
        rows.append(
            _ranking_agreement_row(
                f"{dimension}D",
                "All",
                records,
                methods,
                tie_tolerance,
            )
        )
        for budget_level in sorted({int(record["budget_level"]) for record in records}):
            selected = [
                record
                for record in records
                if int(record["budget_level"]) == budget_level
            ]
            rows.append(
                _ranking_agreement_row(
                    f"{dimension}D",
                    str(budget_level),
                    selected,
                    methods,
                    tie_tolerance,
                )
            )
    return rows


def residual_by_k_summary(
    records_by_dataset: Mapping[str, list[dict[str, object]]],
    *,
    methods: Mapping[str, str] = METHODS,
) -> list[dict[str, float | int | str]]:
    """Summarize mean residual by dimension, K, and method."""
    rows: list[dict[str, float | int | str]] = []
    for records in _records_by_dimension(records_by_dataset).values():
        dimension = int(records[0]["spatial_dimension"])
        for K in sorted({int(record["K"]) for record in records}):
            selected = [record for record in records if int(record["K"]) == K]
            for method_key, method_name in methods.items():
                residuals = [
                    float(record[f"{method_key}_actual_error"])
                    - float(record[f"{method_key}_error"])
                    for record in selected
                ]
                rows.append(
                    {
                        "dimension": f"{dimension}D",
                        "K": K,
                        "method": method_name,
                        "mean_residual": float(np.mean(residuals)),
                    }
                )
    return rows


def make_combined_actual_error_plots(
    records_by_dataset: dict[str, list[dict[str, object]]],
    results_root: str | Path = DEFAULT_ACTUAL_RESULTS_ROOT,
    *,
    run_scope: str = "test",
    methods: Mapping[str, str] = METHODS,
) -> list[tuple[str, plt.Figure]]:
    """Create side-by-side 1D/2D actual-error plots."""
    summary_name = f"results_summary_{'_'.join(records_by_dataset)}_{run_scope}"
    figures_directory = Path(results_root) / summary_name / "figures"
    figures_directory.mkdir(parents=True, exist_ok=True)
    dimension_records = sorted(
        records_by_dataset.values(),
        key=lambda rows: int(rows[0]["spatial_dimension"]),
    )
    figures = [
        (
            "Figure 1: Surrogate validation",
            _plot_combined_actual_vs_surrogate(dimension_records, methods),
            "figure_1_surrogate_validation",
        ),
        (
            "Figure 2: Algorithm performance using actual error",
            _plot_combined_actual_error_performance(dimension_records, methods),
            "figure_2_actual_error_performance",
        ),
    ]
    result: list[tuple[str, plt.Figure]] = []
    for title, figure, filename in figures:
        figure.savefig(figures_directory / f"{filename}.png", dpi=300)
        figure.savefig(figures_directory / f"{filename}.pdf")
        result.append((title, figure))
    return result


def _record_tree(
    record: Mapping[str, object],
    prefix: str,
    K: int,
    M: int,
    allocation: Sequence[int],
    pair_error_table: object,
) -> Mapping[str, object] | object:
    tree_key = f"{prefix}_decision_tree"
    if tree_key in record:
        return record[tree_key]

    pairs = tuple((i, j) for i in range(K) for j in range(i + 1, K))
    _, pair_errors = heuristic_module._compute_pair_errors(
        pairs,
        M,
        tuple(allocation),
        pair_error_table,
    )
    if prefix == "greedy":
        tree = greedy_module._build_greedy_tree(
            tuple(range(K)),
            _record_pair_locations(record, prefix),
            pair_errors,
        )
        rebuilt_error = greedy_module._tree_error(tree, K)
    elif prefix == "heuristic":
        tree = heuristic_module._build_tree_with_beam_search(
            K,
            _record_pair_locations(record, prefix),
            pair_errors,
            beam_width=3,
        )
        rebuilt_error = heuristic_module._tree_error(tree, K)
    else:
        raise ValueError(f"Cannot rebuild a saved tree for method {prefix!r}")

    saved_error = float(record[f"{prefix}_error"])
    if abs(rebuilt_error - saved_error) > 1e-10:
        raise RuntimeError(
            f"Rebuilt {prefix} tree error {rebuilt_error:.12g} does not "
            f"match saved error {saved_error:.12g}"
        )
    return tree


def _record_pair_locations(
    record: Mapping[str, object],
    prefix: str,
) -> dict[tuple[int, int], int]:
    locations = record[f"{prefix}_pair_locations"]
    if not isinstance(locations, Mapping):
        raise ValueError(f"{prefix}_pair_locations must be a mapping")

    result: dict[tuple[int, int], int] = {}
    for key, value in locations.items():
        i_text, j_text = str(key).split(",", maxsplit=1)
        result[(int(i_text), int(j_text))] = int(value)
    return result


def _records_by_dimension(
    records_by_dataset: Mapping[str, list[dict[str, object]]],
) -> dict[int, list[dict[str, object]]]:
    return {
        int(records[0]["spatial_dimension"]): records
        for records in records_by_dataset.values()
    }


def _ranking_agreement_row(
    dimension: str,
    budget_level: str,
    records: Sequence[Mapping[str, object]],
    methods: Mapping[str, str],
    tie_tolerance: float,
) -> dict[str, float | int | str]:
    agreements = 0
    for record in records:
        surrogate_values = {
            method: float(record[f"{method}_error"]) for method in methods
        }
        actual_values = {
            method: float(record[f"{method}_actual_error"]) for method in methods
        }
        surrogate_best = _best_methods(surrogate_values, tie_tolerance)
        actual_best = _best_methods(actual_values, tie_tolerance)
        if surrogate_best & actual_best:
            agreements += 1

    total = len(records)
    return {
        "dimension": dimension,
        "budget_level": budget_level,
        "runs": total,
        "agreement_rate": agreements / total if total else 0.0,
    }


def _best_methods(
    values: Mapping[str, float],
    tie_tolerance: float,
) -> set[str]:
    best_value = min(values.values())
    return {
        method
        for method, value in values.items()
        if value <= best_value + tie_tolerance
    }


def _pearson_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("correlation inputs must have matching length at least two")
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    first_centered = first_values - float(np.mean(first_values))
    second_centered = second_values - float(np.mean(second_values))
    denominator = float(
        np.sqrt(np.sum(first_centered**2) * np.sum(second_centered**2))
    )
    if denominator == 0.0:
        return 0.0
    return float(np.sum(first_centered * second_centered) / denominator)


def _spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    return _pearson_correlation(_average_ranks(first), _average_ranks(second))


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        rank = 0.5 * (position + 1 + end)
        for original_index, _ in indexed[position:end]:
            ranks[original_index] = rank
        position = end
    return ranks


def _signed_phase_difference(
    first: int,
    second: int,
    location: int,
    phase_gaps: PhaseGaps,
) -> float:
    if first == second:
        return 0.0
    if first < second:
        return float(phase_gaps[(first, second)][location])
    return principal_phase(-float(phase_gaps[(second, first)][location]))


def _field_phases(location: int, phase_gaps: PhaseGaps) -> tuple[float, ...]:
    K = max(j for _, j in phase_gaps) + 1
    phases = [0.0]
    for field in range(1, K):
        phases.append(
            principal_phase(-float(phase_gaps[(0, field)][location]))
        )
    return tuple(phases)


def _sensors_used_for_test(gap: float, available_sensor_count: int) -> int:
    threshold = ceil(pi / gap)
    return min(available_sensor_count, threshold)


def _probe_components(
    gap: float,
    sensor_count: int,
) -> list[tuple[int, int, complex]]:
    theta = gap / 2.0
    if sensor_count * gap < pi:
        amplitude = 1.0 / sqrt(2.0)
        return [
            (sensor_count, 0, amplitude),
            (0, sensor_count, amplitude),
        ]

    q = ceil(pi / gap)
    alpha = q * theta
    if q == 1:
        amplitude = 1.0 / sqrt(2.0)
        return [(1, 0, amplitude), (0, 1, amplitude)]

    if q % 2 == 0:
        p = 1.0 / (1.0 - cos(alpha))
        return [
            (q, 0, sqrt(p / 2.0)),
            (0, q, sqrt(p / 2.0)),
            (q // 2, q // 2, sqrt(max(0.0, 1.0 - p))),
        ]

    p = cos(theta) / (cos(theta) - cos(alpha))
    return [
        (q, 0, sqrt(p / 2.0)),
        (0, q, sqrt(p / 2.0)),
        ((q + 1) // 2, (q - 1) // 2, sqrt(max(0.0, (1.0 - p) / 2.0))),
        ((q - 1) // 2, (q + 1) // 2, sqrt(max(0.0, (1.0 - p) / 2.0))),
    ]


def _output_state(
    field: int,
    location: int,
    components: Sequence[tuple[int, int, complex]],
    field_phases: Sequence[float],
) -> np.ndarray:
    del location
    phase = field_phases[field]
    return np.array(
        [
            amplitude
            * np.exp(-0.5j * phase * (plus_count - minus_count))
            for plus_count, minus_count, amplitude in components
        ],
        dtype=complex,
    )


def _leaf_success_probability(
    tree: Mapping[str, object] | object,
    true_field: int,
    phase_gaps: PhaseGaps,
    allocation: Sequence[int],
) -> float:
    declared = _declared_field(tree)
    if declared is not None:
        return 1.0 if declared == true_field else 0.0

    pair = _tree_pair(tree)
    location = _tree_location(tree)
    probabilities = actual_node_probabilities(
        pair,
        location,
        int(allocation[location]),
        true_field,
        phase_gaps,
    )
    return (
        probabilities.keep_j
        * _leaf_success_probability(
            _tree_child(tree, "discard_i"), true_field, phase_gaps, allocation
        )
        + probabilities.keep_i
        * _leaf_success_probability(
            _tree_child(tree, "discard_j"), true_field, phase_gaps, allocation
        )
    )


def _surrogate_leaf_success_probability(
    tree: Mapping[str, object] | object,
    true_field: int,
) -> float:
    declared = _declared_field(tree)
    if declared is not None:
        return 1.0 if declared == true_field else 0.0

    pair = _tree_pair(tree)
    pair_error = _tree_pair_error(tree)
    if true_field == pair[0]:
        keep_i = 1.0 - pair_error
    elif true_field == pair[1]:
        keep_i = pair_error
    else:
        keep_i = 0.5
    keep_j = 1.0 - keep_i
    return (
        keep_j
        * _surrogate_leaf_success_probability(
            _tree_child(tree, "discard_i"), true_field
        )
        + keep_i
        * _surrogate_leaf_success_probability(
            _tree_child(tree, "discard_j"), true_field
        )
    )


def _declared_field(tree: Mapping[str, object] | object) -> int | None:
    if isinstance(tree, Mapping):
        value = tree.get("declared_field")
    else:
        value = getattr(tree, "declared_field", None)
    return None if value is None else int(value)


def _tree_pair(tree: Mapping[str, object] | object) -> tuple[int, int]:
    value = tree["pair"] if isinstance(tree, Mapping) else getattr(tree, "pair")
    i, j = value
    return int(i), int(j)


def _tree_location(tree: Mapping[str, object] | object) -> int:
    value = tree["location"] if isinstance(tree, Mapping) else getattr(tree, "location")
    return int(value)


def _tree_pair_error(tree: Mapping[str, object] | object) -> float:
    value = (
        tree["pair_error"]
        if isinstance(tree, Mapping)
        else getattr(tree, "pair_error")
    )
    if not isfinite(float(value)):
        raise ValueError("tree pair_error must be finite")
    return float(value)


def _tree_child(
    tree: Mapping[str, object] | object,
    name: str,
) -> Mapping[str, object] | object:
    return tree[name] if isinstance(tree, Mapping) else getattr(tree, name)


def _dataset_results_directory(
    results_root: str | Path,
    dataset: str,
    run_scope: str,
) -> Path:
    if not run_scope:
        raise ValueError("run_scope must be nonempty")
    return Path(results_root) / f"{dataset}_results_{run_scope}"


def _plot_combined_actual_vs_surrogate(
    dimension_records: list[list[dict[str, object]]],
    methods: Mapping[str, str],
) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.3), sharex=True, sharey=True)
    limit = 0.0
    for records in dimension_records:
        for method_key in methods:
            limit = max(
                limit,
                max(float(record[f"{method_key}_error"]) for record in records),
                max(float(record[f"{method_key}_actual_error"]) for record in records),
            )
    for axis, records in zip(axes, dimension_records):
        dimension = int(records[0]["spatial_dimension"])
        for method_key, method_name in methods.items():
            axis.scatter(
                [float(record[f"{method_key}_error"]) for record in records],
                [float(record[f"{method_key}_actual_error"]) for record in records],
                alpha=0.55,
                s=15,
                color=METHOD_COLORS[method_name],
                label=method_name,
            )
        axis.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
        axis.set_title(f"({chr(96 + dimension)}) {dimension}D")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Actual final error")
    for axis in axes:
        axis.set_xlabel("Surrogate final error")
        axis.set_xlim(0, limit * 1.03)
        axis.set_ylim(0, limit * 1.03)
    axes[-1].legend()
    figure.tight_layout()
    return figure


def _plot_combined_actual_error_performance(
    dimension_records: list[list[dict[str, object]]],
    methods: Mapping[str, str],
) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.3), sharey=True)
    for axis, records in zip(axes, dimension_records):
        dimension = int(records[0]["spatial_dimension"])
        budget_levels = sorted({int(record["budget_level"]) for record in records})
        for method_key, method_name in methods.items():
            x_values = []
            means = []
            for budget_level in budget_levels:
                selected = [
                    record
                    for record in records
                    if int(record["budget_level"]) == budget_level
                ]
                x_values.append(
                    float(
                        np.mean(
                            [
                                float(record["budget_fraction_of_N_min"])
                                for record in selected
                            ]
                        )
                    )
                )
                means.append(
                    float(
                        np.mean(
                            [
                                float(record[f"{method_key}_actual_error"])
                                for record in selected
                            ]
                        )
                    )
                )
            axis.plot(
                x_values,
                means,
                marker="o",
                color=METHOD_COLORS[method_name],
                label=method_name,
            )
        axis.set_title(f"({chr(96 + dimension)}) {dimension}D")
        axis.set_xlabel(r"Budget ratio $\rho=N/N_{\min}^{opt}$")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean actual end-to-end error")
    axes[-1].legend()
    figure.tight_layout()
    return figure
