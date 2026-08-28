"""Evaluation and plotting helpers for the exhaustive-search notebook."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import fmean, median, stdev
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from evaluate_benchmarks import tree_to_dict, validate_result
from exhaustive_search import exhaustive_search
from plot_style import configure_plot_style


HERE = Path(__file__).resolve().parent
from input_data import load_sparse_input, validate_reverse_phases

INPUT_ROOT = HERE / "inputs"
DEFAULT_RESULTS_ROOT = HERE / "results" / "optimal_evaluation"

configure_plot_style()


def run_dataset(
    dataset: str,
    replicates_per_cell: int | None = 1,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
) -> list[dict[str, object]]:
    """Run the exact algorithm on selected cases from one dataset."""
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

    # Run the exhaustive solver at every selected budget for each case.
    for case_number, case in enumerate(cases, start=1):
        K, M, phase_gaps, pair_errors = load_sparse_input(
            dataset_directory / str(case["file"])
        )
        validate_reverse_phases(phase_gaps)
        budgets = list(case["sensor_budgets"])
        minimum_sensors = int(case["minimum_sensors_for_zero_error"])

        for budget_level, budget in enumerate(budgets, start=1):
            started = perf_counter()
            result = exhaustive_search(K, M, int(budget), pair_errors)
            seconds = perf_counter() - started
            validate_result(result, K, M, int(budget))
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
                    "sensor_budget": int(budget),
                    "budget_level": budget_level,
                    "number_of_budget_levels": len(budgets),
                    "budget_fraction_of_N_min": int(budget) / minimum_sensors,
                    "minimum_error": result.minimum_error,
                    "seconds": seconds,
                    "allocation": list(result.allocation),
                    "pair_locations": {
                        f"{i},{j}": location
                        for (i, j), location in result.pair_locations.items()
                    },
                    "decision_tree": tree_to_dict(result.decision_tree),
                }
            )

        print(
            f"{dataset}: completed case {case_number}/{len(cases)} "
            f"({case['field_model']}, K={K}, M={M})"
        )

    run_scope = "full" if replicates_per_cell is None else "test"
    save_results(dataset, records, results_root, run_scope=run_scope)
    return records


def save_results(
    dataset: str,
    records: list[dict[str, object]],
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    run_scope: str = "test",
) -> None:
    """Save detailed JSON and a compact CSV table for one dataset."""
    result_directory = _dataset_results_directory(results_root, dataset, run_scope)
    tables_directory = result_directory / "tables"
    tables_directory.mkdir(parents=True, exist_ok=True)

    # JSON keeps complete strategies; CSV keeps compact numerical columns.
    (tables_directory / "results.json").write_text(
        json.dumps(
            {"dataset": dataset, "result_count": len(records), "results": records},
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
        "number_of_budget_levels",
        "budget_fraction_of_N_min",
        "minimum_error",
        "seconds",
    ]
    with (tables_directory / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({column: record[column] for column in columns})


def make_candidate_plots(
    dataset: str,
    records: list[dict[str, object]],
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    run_scope: str = "test",
) -> list[tuple[str, plt.Figure]]:
    """Create and save five candidate plots for one dataset."""
    result_directory = _dataset_results_directory(results_root, dataset, run_scope)
    figures_directory = result_directory / "figures"
    figures_directory.mkdir(parents=True, exist_ok=True)
    dimension = int(records[0]["spatial_dimension"])

    # Build each candidate figure before saving it in both output formats.
    figures = [
        (
            "Error versus sensor budget",
            _plot_error_vs_budget(records, dimension),
            "error_vs_budget",
        ),
        (
            "Error by field family",
            _plot_error_by_family(records, dimension),
            "error_by_field_family",
        ),
        (
            "Error heatmap over K and M",
            _plot_error_heatmap(records, dimension),
            "error_heatmap_k_m",
        ),
        (
            "Runtime scaling",
            _plot_runtime(records, dimension),
            "runtime_scaling",
        ),
        (
            "Representative case curves",
            _plot_representative_cases(records, dimension),
            "representative_case_curves",
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


def run_summary(records: list[dict[str, object]]) -> dict[str, object]:
    """Return the main counts and timing values for a notebook run."""
    return {
        "cases": len({record["case"] for record in records}),
        "runs": len(records),
        "K values": sorted({int(record["K"]) for record in records}),
        "M values": sorted({int(record["M"]) for record in records}),
        "mean error": fmean(float(record["minimum_error"]) for record in records),
        "total seconds": sum(float(record["seconds"]) for record in records),
        "maximum seconds": max(float(record["seconds"]) for record in records),
    }


def _highest_budget_records(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Select the largest evaluated budget for each case."""
    return [
        record
        for record in records
        if record["budget_level"] == record["number_of_budget_levels"]
    ]


def _mean_and_sem(values: list[float]) -> tuple[float, float]:
    """Return the sample mean and standard error."""
    if len(values) < 2:
        return fmean(values), 0.0
    return fmean(values), stdev(values) / sqrt(len(values))


def _plot_error_vs_budget(
    records: list[dict[str, object]], dimension: int
) -> plt.Figure:
    """Plot mean error against the fraction of the zero-error minimum."""
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for K in sorted({int(record["K"]) for record in records}):
        K_records = [record for record in records if record["K"] == K]
        points = []
        for level in sorted({int(record["budget_level"]) for record in K_records}):
            level_records = [
                record for record in K_records if record["budget_level"] == level
            ]
            x = fmean(
                float(record["budget_fraction_of_N_min"])
                for record in level_records
            )
            mean_error, error_sem = _mean_and_sem(
                [float(record["minimum_error"]) for record in level_records]
            )
            points.append((x, mean_error, error_sem))
        axis.errorbar(
            [point[0] for point in points],
            [point[1] for point in points],
            yerr=[point[2] for point in points],
            marker="o",
            capsize=3,
            label=f"K={K}",
        )
    axis.set(
        xlabel=r"Sensor budget $N/N_{min}$",
        ylabel="Mean optimal error",
        ylim=(0, None),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def _plot_error_by_family(
    records: list[dict[str, object]], dimension: int
) -> plt.Figure:
    """Compare highest-budget errors across field families."""
    selected = _highest_budget_records(records)
    families = sorted({str(record["field_model"]) for record in selected})
    values = [
        [
            float(record["minimum_error"])
            for record in selected
            if record["field_model"] == family
        ]
        for family in families
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.boxplot(values, tick_labels=families, showmeans=True)
    axis.set(
        xlabel="Field family",
        ylabel="Optimal error at largest tested budget",
        ylim=(0, None),
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def _plot_error_heatmap(
    records: list[dict[str, object]], dimension: int
) -> plt.Figure:
    """Show mean highest-budget error for each K and M combination."""
    selected = _highest_budget_records(records)
    K_values = sorted({int(record["K"]) for record in selected})
    M_values = sorted({int(record["M"]) for record in selected})
    matrix = np.empty((len(K_values), len(M_values)))
    for row, K in enumerate(K_values):
        for column, M in enumerate(M_values):
            matrix[row, column] = fmean(
                float(record["minimum_error"])
                for record in selected
                if record["K"] == K and record["M"] == M
            )

    figure, axis = plt.subplots(figsize=(6.2, 4.5))
    image = axis.imshow(matrix, cmap="viridis", aspect="auto", vmin=0)
    axis.set_xticks(range(len(M_values)), M_values)
    axis.set_yticks(range(len(K_values)), K_values)
    axis.set(
        xlabel="Number of locations M",
        ylabel="Number of fields K",
    )
    for row in range(len(K_values)):
        for column in range(len(M_values)):
            color = "white" if matrix[row, column] > matrix.max() * 0.55 else "black"
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                color=color,
            )
    figure.colorbar(image, ax=axis, label="Mean optimal error")
    figure.tight_layout()
    return figure


def _plot_runtime(
    records: list[dict[str, object]], dimension: int
) -> plt.Figure:
    """Plot median runtime by K and relative budget level."""
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for K in sorted({int(record["K"]) for record in records}):
        K_records = [record for record in records if record["K"] == K]
        levels = sorted({int(record["budget_level"]) for record in K_records})
        runtimes = [
            median(
                float(record["seconds"])
                for record in K_records
                if record["budget_level"] == level
            )
            for level in levels
        ]
        axis.plot(levels, runtimes, marker="o", label=f"K={K}")
    axis.set_yscale("log")
    axis.set(
        xlabel="Budget level (smallest to largest)",
        ylabel="Median runtime (seconds, log scale)",
        xticks=sorted({int(record["budget_level"]) for record in records}),
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    figure.tight_layout()
    return figure


def _plot_representative_cases(
    records: list[dict[str, object]], dimension: int
) -> plt.Figure:
    """Plot one maximum-K, maximum-M case from each field family."""
    maximum_K = max(int(record["K"]) for record in records)
    maximum_M = max(int(record["M"]) for record in records)
    families = sorted({str(record["field_model"]) for record in records})
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for family in families:
        candidates = [
            record
            for record in records
            if record["field_model"] == family
            and record["K"] == maximum_K
            and record["M"] == maximum_M
        ]
        case = sorted({str(record["case"]) for record in candidates})[0]
        case_records = sorted(
            [record for record in candidates if record["case"] == case],
            key=lambda record: int(record["sensor_budget"]),
        )
        axis.plot(
            [float(record["budget_fraction_of_N_min"]) for record in case_records],
            [float(record["minimum_error"]) for record in case_records],
            marker="o",
            label=family,
        )
    axis.set(
        xlabel=r"Sensor budget $N/N_{min}$",
        ylabel="Optimal error",
        ylim=(0, None),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure
