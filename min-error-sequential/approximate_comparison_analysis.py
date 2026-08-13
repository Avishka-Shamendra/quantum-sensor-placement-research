"""Evaluate and plot the scalable allocation algorithms."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean, median
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from greedy import greedy
from heuristic import heuristic
from input_data import load_sparse_input, validate_reverse_phases
from uniform import uniform


HERE = Path(__file__).resolve().parent
INPUT_ROOT = HERE / "inputs"
DEFAULT_RESULTS_ROOT = HERE / "results" / "algorithm_comparison"
FIELD_FAMILY_ORDER = ("gaussian", "sinusoidal", "quadratic")
ALGORITHM_STYLES = {
    "Greedy": ("greedy", "#ff7f0e", "-"),
    "Heuristic": ("heuristic", "#d62728", "-"),
    "Uniform": ("uniform", "#2ca02c", "-"),
}
SENSOR_BUDGET_LABEL = (
    r"$N/N_{\min}$ "
    "(Sensor Budget / Required Min Sensors for Perfect Discrimination)"
)


def run_approximate_dataset(
    dataset: str,
    replicates_per_cell: int | None = 4,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    greedy_parameters: dict[str, object] | None = None,
    heuristic_parameters: dict[str, object] | None = None,
    uniform_parameters: dict[str, object] | None = None,
    progress_every: int = 25,
    k_values: tuple[int, ...] | None = None,
    m_values: tuple[int, ...] | None = None,
) -> list[dict[str, object]]:
    """Run Greedy, Heuristic, and Uniform on the same benchmark cases."""
    if replicates_per_cell is not None and replicates_per_cell < 1:
        raise ValueError("replicates_per_cell must be positive or None")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")

    greedy_parameters = dict(greedy_parameters or {})
    heuristic_parameters = dict(heuristic_parameters or {})
    uniform_parameters = dict(uniform_parameters or {})
    dataset_directory = INPUT_ROOT / dataset
    manifest = json.loads(
        (dataset_directory / "manifest.json").read_text(encoding="utf-8")
    )
    cases = _select_cases(
        list(manifest["cases"]),
        replicates_per_cell,
        k_values,
        m_values,
    )

    records: list[dict[str, object]] = []
    for case_number, case in enumerate(cases, start=1):
        K, M, phase_gaps, pair_errors = load_sparse_input(
            dataset_directory / str(case["file"])
        )
        validate_reverse_phases(phase_gaps)
        budgets = [int(value) for value in case["sensor_budgets"]]
        minimum_sensors = int(case["minimum_sensors_for_zero_error"])

        for budget_level, budget in enumerate(budgets, start=1):
            started = perf_counter()
            greedy_result = greedy(K, M, budget, pair_errors, **greedy_parameters)
            greedy_seconds = perf_counter() - started

            started = perf_counter()
            heuristic_result = heuristic(
                K, M, budget, pair_errors, **heuristic_parameters
            )
            heuristic_seconds = perf_counter() - started

            started = perf_counter()
            uniform_result = uniform(
                K, M, budget, pair_errors, **uniform_parameters
            )
            uniform_seconds = perf_counter() - started

            for result in (greedy_result, heuristic_result, uniform_result):
                _validate_strategy(
                    result.allocation,
                    result.pair_locations,
                    result.error_probability,
                    K,
                    M,
                    budget,
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
                    "greedy_error": greedy_result.error_probability,
                    "heuristic_error": heuristic_result.error_probability,
                    "uniform_error": uniform_result.error_probability,
                    "greedy_seconds": greedy_seconds,
                    "heuristic_seconds": heuristic_seconds,
                    "uniform_seconds": uniform_seconds,
                    "greedy_allocation": list(greedy_result.allocation),
                    "heuristic_allocation": list(heuristic_result.allocation),
                    "uniform_allocation": list(uniform_result.allocation),
                    "greedy_pair_locations": _pair_locations_to_dict(
                        greedy_result.pair_locations
                    ),
                    "heuristic_pair_locations": _pair_locations_to_dict(
                        heuristic_result.pair_locations
                    ),
                    "uniform_pair_locations": _pair_locations_to_dict(
                        uniform_result.pair_locations
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
                f"({case['field_model']}, K={K}, M={M})"
            )

    run_scope = (
        "full"
        if replicates_per_cell is None and k_values is None and m_values is None
        else "test"
    )
    save_approximate_results(
        dataset,
        records,
        results_root,
        run_scope=run_scope,
        algorithm_parameters={
            "greedy": greedy_parameters,
            "heuristic": heuristic_parameters,
            "uniform": uniform_parameters,
        },
    )
    return records


def _select_cases(
    cases: list[dict[str, object]],
    replicates_per_cell: int | None,
    k_values: tuple[int, ...] | None,
    m_values: tuple[int, ...] | None,
) -> list[dict[str, object]]:
    """Apply the requested replicate, K, and M filters."""
    selected = cases
    if replicates_per_cell is not None:
        selected = [
            case
            for case in selected
            if int(case["replicate"]) <= replicates_per_cell
        ]
    if k_values is not None:
        allowed_k = set(k_values)
        selected = [case for case in selected if int(case["K"]) in allowed_k]
    if m_values is not None:
        allowed_m = set(m_values)
        selected = [case for case in selected if int(case["M"]) in allowed_m]
    if not selected:
        raise ValueError("the selected settings contain no benchmark cases")
    return selected


def save_approximate_results(
    dataset: str,
    records: list[dict[str, object]],
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    run_scope: str = "test",
    algorithm_parameters: dict[str, dict[str, object]] | None = None,
) -> None:
    """Save complete JSON results and a compact CSV table."""
    tables_directory = (
        Path(results_root) / f"{dataset}_results_{run_scope}" / "tables"
    )
    tables_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "result_count": len(records),
        "algorithm_parameters": algorithm_parameters or {},
        "results": records,
    }
    (tables_directory / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
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
        "greedy_error",
        "heuristic_error",
        "uniform_error",
        "greedy_seconds",
        "heuristic_seconds",
        "uniform_seconds",
    ]
    with (tables_directory / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({column: record[column] for column in columns})


def load_approximate_results(
    dataset: str,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    run_scope: str = "test",
) -> list[dict[str, object]]:
    """Load previously saved records without rerunning the algorithms."""
    if run_scope not in {"test", "full"}:
        raise ValueError("run_scope must be 'test' or 'full'")
    results_path = (
        Path(results_root)
        / f"{dataset}_results_{run_scope}"
        / "tables"
        / "results.json"
    )
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    return list(payload["results"])


def approximate_summary(
    records: list[dict[str, object]],
) -> list[dict[str, float | int | str]]:
    """Summarize error and runtime for each approximate algorithm."""
    rows: list[dict[str, float | int | str]] = []
    case_count = len({str(record["case"]) for record in records})
    for algorithm, (prefix, _, _) in ALGORITHM_STYLES.items():
        errors = [float(record[f"{prefix}_error"]) for record in records]
        runtimes = [float(record[f"{prefix}_seconds"]) for record in records]
        rows.append(
            {
                "algorithm": algorithm,
                "cases": case_count,
                "runs": len(records),
                "mean_error": fmean(errors),
                "maximum_error": max(errors),
                "mean_seconds": fmean(runtimes),
                "median_seconds": median(runtimes),
            }
        )
    return rows


def make_combined_approximate_plots(
    records_by_dataset: dict[str, list[dict[str, object]]],
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    run_scope: str = "test",
) -> list[tuple[str, plt.Figure]]:
    """Create and save the three-algorithm 1D and 2D comparison plots."""
    if run_scope not in {"test", "full"}:
        raise ValueError("run_scope must be 'test' or 'full'")
    dimension_records = sorted(
        records_by_dataset.values(),
        key=lambda records: int(records[0]["spatial_dimension"]),
    )
    if [int(records[0]["spatial_dimension"]) for records in dimension_records] != [
        1,
        2,
    ]:
        raise ValueError("combined plots require one 1D and one 2D dataset")

    dataset_names = "_".join(records_by_dataset)
    figures_directory = (
        Path(results_root)
        / f"results_summary_{dataset_names}_{run_scope}"
        / "figures"
    )
    figures_directory.mkdir(parents=True, exist_ok=True)
    figure_specs = [
        (
            "Mean error curves",
            _plot_mean_error_curves(dimension_records),
            "mean_error_curves",
        ),
        (
            "Overall mean error curves",
            _plot_overall_mean_error_curves(dimension_records),
            "overall_mean_error_curves",
        ),
        (
            "Runtime distribution by K",
            _plot_runtime_by_k(dimension_records),
            "runtime_distribution_by_k",
        ),
        (
            "Error distribution by field family",
            _plot_error_by_family(dimension_records),
            "error_distribution_by_field_family",
        ),
    ]
    for _, figure, filename in figure_specs:
        figure.savefig(figures_directory / f"{filename}.png", dpi=300)
        figure.savefig(figures_directory / f"{filename}.pdf")
    return [(title, figure) for title, figure, _ in figure_specs]


def _plot_mean_error_curves(
    dimension_records: list[list[dict[str, object]]],
) -> plt.Figure:
    """Plot mean final error against normalized sensor budget."""
    k_values = sorted(
        {int(record["K"]) for records in dimension_records for record in records}
    )
    figure, axes = plt.subplots(
        len(k_values),
        2,
        figsize=(12.5, 3.7 * len(k_values)),
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    for column, records in enumerate(dimension_records):
        dimension = int(records[0]["spatial_dimension"])
        for row, K in enumerate(k_values):
            axis = axes[row, column]
            selected_k = [record for record in records if int(record["K"]) == K]
            levels = sorted(
                {int(record["budget_level"]) for record in selected_k}
            )
            for algorithm, (prefix, color, line_style) in ALGORITHM_STYLES.items():
                x_values: list[float] = []
                means: list[float] = []
                for level in levels:
                    selected = [
                        record
                        for record in selected_k
                        if int(record["budget_level"]) == level
                    ]
                    x_values.append(
                        fmean(
                            float(record["budget_fraction_of_N_min"])
                            for record in selected
                        )
                    )
                    means.append(
                        fmean(float(record[f"{prefix}_error"]) for record in selected)
                    )
                axis.plot(
                    x_values,
                    means,
                    marker="o",
                    linestyle=line_style,
                    color=color,
                    label=algorithm,
                )
            axis.set_title(f"{dimension}D, K={K}")
            axis.grid(alpha=0.25)
            if column == 0:
                axis.set_ylabel("Mean final error")
    axes[0, 1].legend()
    figure.supxlabel(SENSOR_BUDGET_LABEL)
    figure.suptitle("Greedy, heuristic, and uniform error versus sensor budget")
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
        for algorithm, (prefix, color, line_style) in ALGORITHM_STYLES.items():
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
                    fmean(float(record[f"{prefix}_error"]) for record in selected)
                )
            axis.plot(
                x_values,
                means,
                marker="o",
                linestyle=line_style,
                color=color,
                label=algorithm,
            )
        axis.set_title(f"{dimension}D")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean final error")
    axes[1].legend()
    figure.supxlabel(SENSOR_BUDGET_LABEL)
    figure.suptitle(
        "Overall algorithm error versus sensor budget across all K and M values"
    )
    return figure


def _plot_runtime_by_k(
    dimension_records: list[list[dict[str, object]]],
) -> plt.Figure:
    """Plot grouped runtime distributions for each number of fields."""
    figure, axes = plt.subplots(
        1, 2, figsize=(13.2, 4.8), sharey=True, constrained_layout=True
    )
    for axis, records in zip(axes, dimension_records):
        dimension = int(records[0]["spatial_dimension"])
        k_values = sorted({int(record["K"]) for record in records})
        positions = np.arange(1, len(k_values) + 1)
        for offset, (algorithm, (prefix, color, _)) in zip(
            (-0.24, 0.0, 0.24), ALGORITHM_STYLES.items()
        ):
            values = [
                [
                    float(record[f"{prefix}_seconds"])
                    for record in records
                    if int(record["K"]) == K
                ]
                for K in k_values
            ]
            boxplot = axis.boxplot(
                values,
                positions=positions + offset,
                widths=0.22,
                patch_artist=True,
                showmeans=True,
                meanprops={"markerfacecolor": "black", "markeredgecolor": "black"},
                medianprops={"color": "black"},
            )
            for box in boxplot["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.45)
            boxplot["boxes"][0].set_label(algorithm)
        axis.set_yscale("log")
        axis.set(
            title=f"{dimension}D",
            xlabel="Number of fields",
            xticks=positions,
            xticklabels=[f"K={K}" for K in k_values],
        )
        axis.grid(axis="y", alpha=0.25, which="both")
    axes[0].set_ylabel("Runtime (seconds, log scale)")
    axes[1].legend()
    figure.suptitle("Runtime distribution by number of fields")
    return figure


def _plot_error_by_family(
    dimension_records: list[list[dict[str, object]]],
) -> plt.Figure:
    """Plot grouped final-error distributions for every field family."""
    figure, axes = plt.subplots(
        1, 2, figsize=(13.2, 4.8), sharey=True, constrained_layout=True
    )
    for axis, records in zip(axes, dimension_records):
        dimension = int(records[0]["spatial_dimension"])
        present = {str(record["field_model"]) for record in records}
        families = [family for family in FIELD_FAMILY_ORDER if family in present]
        families.extend(sorted(present - set(FIELD_FAMILY_ORDER)))
        positions = np.arange(1, len(families) + 1)
        for offset, (algorithm, (prefix, color, _)) in zip(
            (-0.24, 0.0, 0.24), ALGORITHM_STYLES.items()
        ):
            values = [
                [
                    float(record[f"{prefix}_error"])
                    for record in records
                    if record["field_model"] == family
                ]
                for family in families
            ]
            boxplot = axis.boxplot(
                values,
                positions=positions + offset,
                widths=0.22,
                patch_artist=True,
                showmeans=True,
                meanprops={"markerfacecolor": "black", "markeredgecolor": "black"},
                medianprops={"color": "black"},
            )
            for box in boxplot["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.45)
            boxplot["boxes"][0].set_label(algorithm)
        axis.set(
            title=f"{dimension}D",
            xlabel="Field family",
            ylim=(0, None),
            xticks=positions,
            xticklabels=families,
        )
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Final error")
    axes[1].legend()
    figure.suptitle("Final-error distribution by field family")
    return figure


def _validate_strategy(
    allocation: tuple[int, ...],
    pair_locations: dict[tuple[int, int], int],
    error: float,
    K: int,
    M: int,
    budget: int,
) -> None:
    """Check one algorithm result before recording it."""
    if (
        len(allocation) != M
        or any(count < 0 for count in allocation)
        or sum(allocation) != budget
    ):
        raise RuntimeError("algorithm returned an invalid allocation")
    pairs = {(i, j) for i in range(K) for j in range(i + 1, K)}
    if set(pair_locations) != pairs:
        raise RuntimeError("algorithm returned invalid pair locations")
    if any(not 0 <= location < M for location in pair_locations.values()):
        raise RuntimeError("algorithm returned an out-of-range location")
    if not 0.0 <= error <= 1.0:
        raise RuntimeError("algorithm returned an invalid error probability")


def _pair_locations_to_dict(
    locations: dict[tuple[int, int], int],
) -> dict[str, int]:
    """Convert pair-location tuple keys to JSON string keys."""
    return {f"{i},{j}": location for (i, j), location in locations.items()}
