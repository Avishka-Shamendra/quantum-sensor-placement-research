"""Evaluate four allocation methods using the actual global PGM error."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from math import isfinite
from pathlib import Path
from random import Random
from statistics import fmean
from time import perf_counter
from zlib import crc32

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from algorithms.greedy import greedy_global_discrimination  # noqa: E402
from algorithms.near_uniform import (  # noqa: E402
    near_uniform_global_discrimination,
)
from algorithms.optimal import optimal_global_discrimination  # noqa: E402
from algorithms.random_allocation import (  # noqa: E402
    random_global_discrimination,
)
from input_data import load_delta_input  # noqa: E402
from objective import STATE_MODELS, pgm_error_probability  # noqa: E402


INPUT_ROOT = HERE / "inputs"
DEFAULT_RESULTS_ROOT = HERE / "results" / "actual_pgm_four_methods_c_pi"
EXACT_ERROR_TOLERANCE = 1e-12
STATE_MODEL_LABELS = {
    "product": "Product",
    "local_ghz": "Local GHZ",
    "global_ghz": "Global GHZ",
}
FIELD_FAMILY_ORDER = ("gaussian", "sinusoidal", "quadratic")
FIELD_FAMILY_LABELS = {
    "gaussian": "Gaussian",
    "sinusoidal": "Sinusoidal",
    "quadratic": "Quadratic",
}
METHOD_KEYS = ("optimal", "greedy", "near_uniform", "random")
SCALABLE_METHOD_KEYS = ("greedy", "near_uniform", "random")
METHOD_LABELS = {
    "optimal": "Optimal",
    "greedy": "Greedy",
    "near_uniform": "Near-uniform",
    "random": "Random",
}
METHOD_STYLES = {
    "Optimal": {
        "color": "#2171B5",
        "linestyle": "-",
        "marker": "o",
    },
    "Greedy": {
        "color": "#D95F02",
        "linestyle": "-",
        "marker": "s",
    },
    "Near-uniform": {
        "color": "#238B45",
        "linestyle": "-",
        "marker": "^",
    },
    "Random": {
        "color": "#756BB1",
        "linestyle": "-",
        "marker": "D",
    },
}
STATE_COMPARISON_STYLES = {
    "product": {
        "color": "#A23B72",
        "linestyle": "-",
        "marker": "o",
    },
    "local_ghz": {
        "color": "#2A9D8F",
        "linestyle": "-",
        "marker": "s",
    },
    "global_ghz": {
        "color": "#D4A017",
        "linestyle": "-",
        "marker": "^",
    },
}


def run_paired_dataset(
    dataset: str,
    *,
    sensor_budgets: tuple[int, ...] = tuple(range(3, 13)),
    run_mode: str = "test",
    test_case_count: int = 100,
    sample_seed: int = 20260827,
    progress_every: int = 10,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
) -> list[dict[str, object]]:
    """Evaluate all four methods for one benchmark dataset.

    ``test`` mode samples cases evenly across field-family, K, and M cells;
    ``full`` mode evaluates every case. Random baseline seeds depend only on
    dataset, case, and budget, so each state model receives the same baseline
    allocation for a paired comparison.
    """
    if run_mode not in {"test", "full"}:
        raise ValueError("run_mode must be 'test' or 'full'")
    if (
        isinstance(test_case_count, bool)
        or not isinstance(test_case_count, int)
        or test_case_count < 1
    ):
        raise ValueError("test_case_count must be a positive integer")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    budgets = _validate_sensor_budgets(sensor_budgets)

    dataset_directory = INPUT_ROOT / dataset
    manifest = json.loads(
        (dataset_directory / "manifest.json").read_text(encoding="utf-8")
    )
    all_cases = list(manifest["cases"])
    cases = (
        all_cases
        if run_mode == "full"
        else _stratified_case_sample(
            all_cases,
            test_case_count,
            sample_seed + crc32(dataset.encode("utf-8")),
        )
    )

    records: list[dict[str, object]] = []
    for case_number, case in enumerate(cases, start=1):
        case_file = str(case["file"])
        K, M, delta_ijm = load_delta_input(dataset_directory / case_file)
        delta_values = [
            float(delta)
            for pair_values in delta_ijm.values()
            for delta in pair_values
        ]
        delta_min = min(delta_values)
        delta_max = max(delta_values)
        delta_abs_min = min(abs(delta) for delta in delta_values)
        delta_abs_max = max(abs(delta) for delta in delta_values)

        for budget in budgets:
            near_uniform_seed = _allocation_seed(
                dataset, case_file, budget, "near_uniform", sample_seed
            )
            random_seed = _allocation_seed(
                dataset, case_file, budget, "random", sample_seed
            )
            for state_model in STATE_MODELS:
                method_results: dict[str, object] = {}
                method_seconds: dict[str, float] = {}
                calls = {
                    "optimal": lambda: optimal_global_discrimination(
                        K, M, budget, delta_ijm, state_model
                    ),
                    "greedy": lambda: greedy_global_discrimination(
                        K, M, budget, delta_ijm, state_model
                    ),
                    "near_uniform": lambda: near_uniform_global_discrimination(
                        K,
                        M,
                        budget,
                        delta_ijm,
                        state_model,
                        seed=near_uniform_seed,
                    ),
                    "random": lambda: random_global_discrimination(
                        K,
                        M,
                        budget,
                        delta_ijm,
                        state_model,
                        seed=random_seed,
                    ),
                }
                for method, call in calls.items():
                    started = perf_counter()
                    method_results[method] = call()
                    method_seconds[method] = perf_counter() - started

                errors: dict[str, float] = {}
                allocations: dict[str, tuple[int, ...]] = {}
                for method in METHOD_KEYS:
                    result = method_results[method]
                    allocation = tuple(result.allocation)  # type: ignore[attr-defined]
                    reported = float(result.error_probability)  # type: ignore[attr-defined]
                    _validate_result(allocation, reported, M, budget)
                    recomputed = pgm_error_probability(
                        K, M, allocation, delta_ijm, state_model
                    )
                    if abs(reported - recomputed) > EXACT_ERROR_TOLERANCE:
                        raise RuntimeError(
                            f"{METHOD_LABELS[method]} returned an inconsistent "
                            "PGM error"
                        )
                    allocations[method] = allocation
                    errors[method] = recomputed

                optimal_error = errors["optimal"]
                for method in METHOD_KEYS[1:]:
                    if optimal_error > errors[method] + EXACT_ERROR_TOLERANCE:
                        raise RuntimeError(
                            f"{METHOD_LABELS[method]} PGM error is below the "
                            "exact optimum"
                        )

                record: dict[str, object] = {
                    "dataset": dataset,
                    "case": case_file,
                    "field_model": case["field_model"],
                    "spatial_dimension": int(case["spatial_dimension"]),
                    "K": K,
                    "M": M,
                    "sensor_budget": budget,
                    "state_model": state_model,
                    "near_uniform_seed": near_uniform_seed,
                    "random_seed": random_seed,
                    "delta_min": delta_min,
                    "delta_max": delta_max,
                    "delta_abs_min": delta_abs_min,
                    "delta_abs_max": delta_abs_max,
                    "n_delta_min": budget * delta_min,
                    "n_delta_max": budget * delta_max,
                    "n_delta_abs_min": budget * delta_abs_min,
                    "n_delta_abs_max": budget * delta_abs_max,
                }
                for method in METHOD_KEYS:
                    record[f"{method}_allocation"] = list(allocations[method])
                    record[f"{method}_pgm_error"] = errors[method]
                    record[f"{method}_seconds"] = method_seconds[method]
                for method in METHOD_KEYS[1:]:
                    gap = max(0.0, errors[method] - optimal_error)
                    record[f"{method}_pgm_error_gap"] = gap
                    record[f"{method}_relative_pgm_error_gap"] = (
                        gap / optimal_error if optimal_error > 0.0 else 0.0
                    )
                    record[f"{method}_same_allocation_as_optimal"] = (
                        allocations[method] == allocations["optimal"]
                    )
                records.append(record)

        if case_number % progress_every == 0 or case_number == len(cases):
            print(f"{dataset}: completed {case_number}/{len(cases)} cases")

    _save_results(
        records,
        dataset=dataset,
        run_mode=run_mode,
        sensor_budgets=budgets,
        selected_case_count=len(cases),
        total_case_count=len(all_cases),
        sample_seed=sample_seed,
        results_root=Path(results_root),
    )
    return records


def run_scalable_dataset(
    dataset: str,
    *,
    sensor_budgets: tuple[int, ...] = tuple(range(3, 13)),
    run_mode: str = "test",
    test_case_count: int = 100,
    sample_seed: int = 20260827,
    progress_every: int = 10,
    results_root: str | Path,
) -> list[dict[str, object]]:
    """Evaluate scalable methods without exhaustive Optimal search.

    This runner is intended for the extended benchmarks, where the complete
    ``K``/``M`` design makes exhaustive allocation enumeration unsuitable as a
    full-dataset reference.
    """
    if run_mode not in {"test", "full"}:
        raise ValueError("run_mode must be 'test' or 'full'")
    if (
        isinstance(test_case_count, bool)
        or not isinstance(test_case_count, int)
        or test_case_count < 1
    ):
        raise ValueError("test_case_count must be a positive integer")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    budgets = _validate_sensor_budgets(sensor_budgets)

    dataset_directory = INPUT_ROOT / dataset
    manifest = json.loads(
        (dataset_directory / "manifest.json").read_text(encoding="utf-8")
    )
    all_cases = list(manifest["cases"])
    cases = (
        all_cases
        if run_mode == "full"
        else _stratified_case_sample(
            all_cases,
            test_case_count,
            sample_seed + crc32(dataset.encode("utf-8")),
        )
    )

    records: list[dict[str, object]] = []
    for case_number, case in enumerate(cases, start=1):
        case_file = str(case["file"])
        K, M, delta_ijm = load_delta_input(dataset_directory / case_file)
        delta_values = [
            float(delta)
            for pair_values in delta_ijm.values()
            for delta in pair_values
        ]
        delta_min = min(delta_values)
        delta_max = max(delta_values)
        delta_abs_min = min(abs(delta) for delta in delta_values)
        delta_abs_max = max(abs(delta) for delta in delta_values)

        for budget in budgets:
            near_uniform_seed = _allocation_seed(
                dataset, case_file, budget, "near_uniform", sample_seed
            )
            random_seed = _allocation_seed(
                dataset, case_file, budget, "random", sample_seed
            )
            for state_model in STATE_MODELS:
                calls = {
                    "greedy": lambda: greedy_global_discrimination(
                        K, M, budget, delta_ijm, state_model
                    ),
                    "near_uniform": lambda: near_uniform_global_discrimination(
                        K,
                        M,
                        budget,
                        delta_ijm,
                        state_model,
                        seed=near_uniform_seed,
                    ),
                    "random": lambda: random_global_discrimination(
                        K,
                        M,
                        budget,
                        delta_ijm,
                        state_model,
                        seed=random_seed,
                    ),
                }
                errors: dict[str, float] = {}
                allocations: dict[str, tuple[int, ...]] = {}
                method_seconds: dict[str, float] = {}
                for method, call in calls.items():
                    started = perf_counter()
                    result = call()
                    method_seconds[method] = perf_counter() - started
                    allocation = tuple(result.allocation)
                    reported = float(result.error_probability)
                    _validate_result(allocation, reported, M, budget)
                    recomputed = pgm_error_probability(
                        K, M, allocation, delta_ijm, state_model
                    )
                    if abs(reported - recomputed) > EXACT_ERROR_TOLERANCE:
                        raise RuntimeError(
                            f"{METHOD_LABELS[method]} returned an inconsistent "
                            "PGM error"
                        )
                    allocations[method] = allocation
                    errors[method] = recomputed

                record: dict[str, object] = {
                    "dataset": dataset,
                    "case": case_file,
                    "field_model": case["field_model"],
                    "spatial_dimension": int(case["spatial_dimension"]),
                    "K": K,
                    "M": M,
                    "sensor_budget": budget,
                    "state_model": state_model,
                    "near_uniform_seed": near_uniform_seed,
                    "random_seed": random_seed,
                    "delta_min": delta_min,
                    "delta_max": delta_max,
                    "delta_abs_min": delta_abs_min,
                    "delta_abs_max": delta_abs_max,
                    "n_delta_min": budget * delta_min,
                    "n_delta_max": budget * delta_max,
                    "n_delta_abs_min": budget * delta_abs_min,
                    "n_delta_abs_max": budget * delta_abs_max,
                }
                for method in SCALABLE_METHOD_KEYS:
                    record[f"{method}_allocation"] = list(allocations[method])
                    record[f"{method}_pgm_error"] = errors[method]
                    record[f"{method}_seconds"] = method_seconds[method]
                records.append(record)

        if case_number % progress_every == 0 or case_number == len(cases):
            print(f"{dataset}: completed {case_number}/{len(cases)} cases")

    _save_results(
        records,
        dataset=dataset,
        run_mode=run_mode,
        sensor_budgets=budgets,
        selected_case_count=len(cases),
        total_case_count=len(all_cases),
        sample_seed=sample_seed,
        results_root=Path(results_root),
        method_keys=SCALABLE_METHOD_KEYS,
    )
    return records


def load_paired_results(
    dataset: str,
    *,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
) -> list[dict[str, object]]:
    """Load saved JSON records for one dataset."""
    path = Path(results_root) / dataset / "paired_results.json"
    return list(json.loads(path.read_text(encoding="utf-8"))["records"])


def error_curve_summary(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate actual PGM errors and gaps by dimension, state, and budget."""
    groups: dict[tuple[int, str, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        key = (
            int(record["spatial_dimension"]),
            str(record["state_model"]),
            int(record["sensor_budget"]),
        )
        groups[key].append(record)

    rows: list[dict[str, object]] = []
    for (dimension, state_model, budget), group in sorted(groups.items()):
        row: dict[str, object] = {
            "spatial_dimension": dimension,
            "state_model": state_model,
            "sensor_budget": budget,
            "instances": len(group),
        }
        for method in METHOD_KEYS:
            row[f"{method}_mean_pgm_error"] = fmean(
                float(record[f"{method}_pgm_error"])
                for record in group
            )
        for method in METHOD_KEYS[1:]:
            row[f"{method}_mean_pgm_error_gap"] = fmean(
                float(record[f"{method}_pgm_error_gap"])
                for record in group
            )
        rows.append(row)
    return rows


def scalable_error_curve_summary(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate actual PGM errors for the three scalable methods."""
    groups: dict[tuple[int, str, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        key = (
            int(record["spatial_dimension"]),
            str(record["state_model"]),
            int(record["sensor_budget"]),
        )
        groups[key].append(record)

    rows: list[dict[str, object]] = []
    for (dimension, state_model, budget), group in sorted(groups.items()):
        row: dict[str, object] = {
            "spatial_dimension": dimension,
            "state_model": state_model,
            "sensor_budget": budget,
            "instances": len(group),
        }
        for method in SCALABLE_METHOD_KEYS:
            row[f"{method}_mean_pgm_error"] = fmean(
                float(record[f"{method}_pgm_error"])
                for record in group
            )
        rows.append(row)
    return rows


def diagnostic_gap_summary(
    records: list[dict[str, object]],
    *,
    exact_tolerance: float = EXACT_ERROR_TOLERANCE,
) -> list[dict[str, object]]:
    """Summarize absolute/relative gaps and exact-optimal fractions."""
    groups: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[
            int(record["spatial_dimension"]), str(record["state_model"])
        ].append(record)

    rows: list[dict[str, object]] = []
    for (dimension, state_model), group in sorted(groups.items()):
        for method in METHOD_KEYS[1:]:
            absolute = sorted(
                float(record[f"{method}_pgm_error_gap"])
                for record in group
            )
            relative = sorted(
                float(record[f"{method}_relative_pgm_error_gap"])
                for record in group
            )
            rows.append(
                {
                    "spatial_dimension": dimension,
                    "state_model": state_model,
                    "method": METHOD_LABELS[method],
                    "runs": len(group),
                    "mean_absolute_gap": fmean(absolute),
                    "p90_absolute_gap": _percentile(absolute, 0.90),
                    "max_absolute_gap": max(absolute),
                    "mean_relative_gap": fmean(relative),
                    "p90_relative_gap": _percentile(relative, 0.90),
                    "max_relative_gap": max(relative),
                    "fraction_exactly_optimal": sum(
                        gap <= exact_tolerance for gap in absolute
                    )
                    / len(group),
                }
            )
    return rows


def phase_range_summary(records: list[dict[str, object]]) -> dict[str, object]:
    """Return observed signed and absolute ranges of delta and N*delta."""
    unique_cases = {
        (str(record["dataset"]), str(record["case"])) for record in records
    }
    return {
        "cases": len(unique_cases),
        "delta_min": min(float(record["delta_min"]) for record in records),
        "delta_max": max(float(record["delta_max"]) for record in records),
        "delta_abs_min": min(
            float(record["delta_abs_min"]) for record in records
        ),
        "delta_abs_max": max(
            float(record["delta_abs_max"]) for record in records
        ),
        "n_delta_min": min(
            float(record["n_delta_min"]) for record in records
        ),
        "n_delta_max": max(
            float(record["n_delta_max"]) for record in records
        ),
        "n_delta_abs_min": min(
            float(record["n_delta_abs_min"]) for record in records
        ),
        "n_delta_abs_max": max(
            float(record["n_delta_abs_max"]) for record in records
        ),
    }


def make_combined_comparison_plots(
    records: list[dict[str, object]],
    *,
    output_directory: str | Path | None = None,
) -> list[tuple[str, plt.Figure]]:
    """Create error, gap, state-comparison, and runtime figures."""
    if not records:
        raise ValueError("records must not be empty")
    _configure_plot_style()
    summary = error_curve_summary(records)
    output = Path(output_directory) if output_directory is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    figures = [
        (
            "Mean actual PGM errors across input states",
            _make_method_grid(summary, gap=False),
            "mean_actual_pgm_errors",
        ),
        (
            "Mean actual PGM-error gaps",
            _make_method_grid(summary, gap=True),
            "mean_actual_pgm_error_gaps",
        ),
        (
            "PGM-error gap by field family",
            _make_field_family_boxplots(
                records,
                methods=SCALABLE_METHOD_KEYS,
                gap=True,
            ),
            "pgm_error_gap_by_field_family",
        ),
        (
            "Greedy input-state comparison",
            _make_state_comparison(summary, "greedy"),
            "greedy_input_state_comparison",
        ),
        (
            "Optimal input-state comparison",
            _make_state_comparison(summary, "optimal"),
            "optimal_input_state_comparison",
        ),
        (
            "Runtime speedup relative to Optimal",
            _make_runtime_speedup_boxplots(records),
            "runtime_boxplots",
        ),
    ]
    for _, figure, filename in figures:
        if output is not None:
            figure.savefig(output / f"{filename}.png", dpi=300)
            figure.savefig(output / f"{filename}.pdf")
    return [(title, figure) for title, figure, _ in figures]


def make_extended_comparison_plots(
    records: list[dict[str, object]],
    *,
    output_directory: str | Path | None = None,
) -> list[tuple[str, plt.Figure]]:
    """Create the concise error and runtime figures for extended benchmarks."""
    if not records:
        raise ValueError("records must not be empty")
    _configure_plot_style()
    summary = scalable_error_curve_summary(records)
    output = Path(output_directory) if output_directory is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    figures = [
        (
            "Extended-benchmark mean actual PGM errors",
            _make_method_grid(
                summary,
                gap=False,
                methods=SCALABLE_METHOD_KEYS,
            ),
            "extended_mean_actual_pgm_errors",
        ),
        (
            "Extended-benchmark PGM error by field family",
            _make_field_family_boxplots(
                records,
                methods=SCALABLE_METHOD_KEYS,
                gap=False,
            ),
            "extended_pgm_error_by_field_family",
        ),
        (
            "Extended-benchmark runtimes",
            _make_raw_runtime_boxplots(records, SCALABLE_METHOD_KEYS),
            "extended_runtime_boxplots",
        ),
    ]
    for _, figure, filename in figures:
        if output is not None:
            figure.savefig(output / f"{filename}.png", dpi=300)
            figure.savefig(output / f"{filename}.pdf")
    return [(title, figure) for title, figure, _ in figures]


def _make_method_grid(
    summary: list[dict[str, object]],
    *,
    gap: bool,
    methods: tuple[str, ...] | None = None,
) -> plt.Figure:
    """Plot state-model rows and dimension columns for errors or gaps."""
    figure, axes = plt.subplots(
        3, 2, figsize=(12, 12), sharex=True, sharey="row", constrained_layout=True
    )
    selected_methods = methods or (METHOD_KEYS[1:] if gap else METHOD_KEYS)
    budgets = sorted({int(row["sensor_budget"]) for row in summary})
    for row_index, state_model in enumerate(STATE_MODELS):
        row_values: list[float] = []
        for column_index, dimension in enumerate((1, 2)):
            axis = axes[row_index, column_index]
            subset = sorted(
                (
                    row
                    for row in summary
                    if int(row["spatial_dimension"]) == dimension
                    and row["state_model"] == state_model
                ),
                key=lambda row: int(row["sensor_budget"]),
            )
            for method in selected_methods:
                label = METHOD_LABELS[method]
                key = (
                    f"{method}_mean_pgm_error_gap"
                    if gap
                    else f"{method}_mean_pgm_error"
                )
                values = [float(row[key]) for row in subset]
                row_values.extend(values)
                axis.plot(
                    [int(row["sensor_budget"]) for row in subset],
                    values,
                    label=label,
                    linewidth=1.8,
                    markersize=5,
                    **METHOD_STYLES[label],
                )
            axis.set_title(f"{dimension}D")
            axis.set_xticks(budgets)
            axis.grid(alpha=0.25)
            if column_index == 0:
                measure = "mean PGM-error gap" if gap else "mean PGM error"
                axis.set_ylabel(
                    f"{STATE_MODEL_LABELS[state_model]}\n{measure}"
                )
            if row_index == 2:
                axis.set_xlabel(r"Sensor budget $N$")
        if gap:
            upper = max(row_values, default=0.0)
            axes[row_index, 0].set_ylim(0.0, upper * 1.08 or 1.0)
        else:
            _set_padded_probability_limits(axes[row_index, 0], row_values)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(selected_methods),
    )
    return figure


def _make_field_family_boxplots(
    records: list[dict[str, object]],
    *,
    methods: tuple[str, ...],
    gap: bool,
) -> plt.Figure:
    """Compare method distributions across field families and input states."""
    figure, axes = plt.subplots(
        3, 2, figsize=(12, 12), sharey="row", constrained_layout=True
    )
    present = {str(record["field_model"]) for record in records}
    families = [family for family in FIELD_FAMILY_ORDER if family in present]
    families.extend(sorted(present - set(FIELD_FAMILY_ORDER)))
    positions = list(range(1, len(families) + 1))
    box_width = 0.22

    for row_index, state_model in enumerate(STATE_MODELS):
        for column_index, dimension in enumerate((1, 2)):
            axis = axes[row_index, column_index]
            selected = [
                record
                for record in records
                if int(record["spatial_dimension"]) == dimension
                and record["state_model"] == state_model
            ]
            for method_index, method in enumerate(methods):
                value_key = (
                    f"{method}_pgm_error_gap"
                    if gap
                    else f"{method}_pgm_error"
                )
                values = [
                    [
                        float(record[value_key])
                        for record in selected
                        if record["field_model"] == family
                    ]
                    for family in families
                ]
                offset = (method_index - (len(methods) - 1) / 2) * box_width
                boxplot = axis.boxplot(
                    values,
                    positions=[position + offset for position in positions],
                    widths=box_width * 0.85,
                    patch_artist=True,
                    showfliers=False,
                    showmeans=True,
                    meanprops={
                        "marker": "^",
                        "markerfacecolor": "#FFFFFF",
                        "markeredgecolor": "#111111",
                        "markersize": 4,
                    },
                    medianprops={"color": "#111111", "linewidth": 1.2},
                )
                label = METHOD_LABELS[method]
                for box in boxplot["boxes"]:
                    box.set_facecolor(METHOD_STYLES[label]["color"])
                    box.set_alpha(0.65)
                boxplot["boxes"][0].set_label(label)

            axis.set_title(f"{dimension}D")
            axis.set_xticks(
                positions,
                [FIELD_FAMILY_LABELS.get(family, family.title()) for family in families],
            )
            axis.grid(axis="y", alpha=0.25)
            if column_index == 0:
                measure = "PGM-error gap" if gap else "PGM error"
                axis.set_ylabel(
                    f"{STATE_MODEL_LABELS[state_model]}\n{measure}"
                )
            if row_index == 2:
                axis.set_xlabel("Field family")
        if gap:
            axes[row_index, 0].set_ylim(bottom=0.0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(methods),
    )
    return figure


def _make_state_comparison(
    summary: list[dict[str, object]],
    method: str,
) -> plt.Figure:
    """Compare the three state models for one allocation method."""
    figure, axes = plt.subplots(
        1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True
    )
    budgets = sorted({int(row["sensor_budget"]) for row in summary})
    plotted_values: list[float] = []
    for axis, dimension in zip(axes, (1, 2)):
        for state_model in STATE_MODELS:
            subset = sorted(
                (
                    row
                    for row in summary
                    if int(row["spatial_dimension"]) == dimension
                    and row["state_model"] == state_model
                ),
                key=lambda row: int(row["sensor_budget"]),
            )
            values = [float(row[f"{method}_mean_pgm_error"]) for row in subset]
            plotted_values.extend(values)
            axis.plot(
                [int(row["sensor_budget"]) for row in subset],
                values,
                label=STATE_MODEL_LABELS[state_model],
                linewidth=1.8,
                markersize=5,
                **STATE_COMPARISON_STYLES[state_model],
            )
        axis.set_title(f"{dimension}D")
        axis.set_xlabel(r"Sensor budget $N$")
        axis.set_xticks(budgets)
        axis.grid(alpha=0.25)
    _set_padded_probability_limits(axes[0], plotted_values)
    axes[0].set_ylabel("mean PGM error")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3)
    return figure


def _make_raw_runtime_boxplots(
    records: list[dict[str, object]],
    methods: tuple[str, ...],
) -> plt.Figure:
    """Plot raw runtime distributions when no exact reference is available."""
    figure, axes = plt.subplots(
        1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True
    )
    labels = [METHOD_LABELS[method] for method in methods]
    colors = [METHOD_STYLES[label]["color"] for label in labels]
    for axis, dimension in zip(axes, (1, 2)):
        dimension_records = [
            record
            for record in records
            if int(record["spatial_dimension"]) == dimension
        ]
        distributions = [
            [
                max(float(record[f"{method}_seconds"]), 1e-12)
                for record in dimension_records
            ]
            for method in methods
        ]
        if any(not values for values in distributions):
            raise ValueError(f"runtime records are missing for {dimension}D")
        boxplot = axis.boxplot(
            distributions,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
            showmeans=True,
            widths=0.6,
            medianprops={"color": "#111111", "linewidth": 1.5},
            meanprops={
                "marker": "^",
                "markerfacecolor": "#FFFFFF",
                "markeredgecolor": "#111111",
                "markersize": 5,
            },
        )
        for box, color in zip(boxplot["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.65)
        axis.set_title(f"{dimension}D")
        axis.set_xlabel("Allocation method")
        axis.set_yscale("log")
        axis.grid(axis="y", which="both", alpha=0.25)
    axes[0].set_ylabel("runtime per problem run (seconds, log scale)")
    return figure


def _make_runtime_speedup_boxplots(
    records: list[dict[str, object]],
) -> plt.Figure:
    """Plot paired runtime speedups relative to Optimal in 1D and 2D."""
    figure, axes = plt.subplots(
        1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True
    )
    labels = [METHOD_LABELS[method] for method in METHOD_KEYS]
    colors = [METHOD_STYLES[label]["color"] for label in labels]
    for axis, dimension in zip(axes, (1, 2)):
        dimension_records = [
            record
            for record in records
            if int(record["spatial_dimension"]) == dimension
        ]
        distributions = [
            [
                max(float(record["optimal_seconds"]), 1e-12)
                / max(float(record[f"{method}_seconds"]), 1e-12)
                for record in dimension_records
            ]
            for method in METHOD_KEYS
        ]
        if any(not values for values in distributions):
            raise ValueError(f"runtime records are missing for {dimension}D")
        boxplot = axis.boxplot(
            distributions,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
            showmeans=True,
            widths=0.6,
            medianprops={"color": "#111111", "linewidth": 1.5},
            meanprops={
                "marker": "^",
                "markerfacecolor": "#FFFFFF",
                "markeredgecolor": "#111111",
                "markersize": 5,
            },
        )
        for box, color in zip(boxplot["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.65)
        axis.set_title(f"{dimension}D")
        axis.set_xlabel("Allocation method")
        axis.set_yscale("log")
        axis.axhline(1.0, color="#555555", linewidth=1.0, linestyle="-")
        axis.grid(axis="y", which="both", alpha=0.25)
    axes[0].set_ylabel("runtime speedup (Optimal / method, log scale)")
    return figure


def _set_padded_probability_limits(
    axis: plt.Axes,
    values: list[float],
) -> None:
    """Set a readable probability range while preserving shared axes."""
    if not values:
        return
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum
    padding = 0.05 * span if span > 0.0 else max(0.01, 0.03 * maximum)
    lower = max(0.0, minimum - padding)
    upper = min(1.0, maximum + padding)
    if upper <= lower:
        upper = min(1.0, lower + 0.01)
    axis.set_ylim(lower, upper)


def _configure_plot_style() -> None:
    """Apply the requested publication typography to every output figure."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


def _allocation_seed(
    dataset: str,
    case_file: str,
    budget: int,
    method: str,
    sample_seed: int,
) -> int:
    """Derive a stable allocation seed from the paired run identifiers."""
    token = f"{dataset}|{case_file}|{budget}|{method}".encode("utf-8")
    return crc32(token, sample_seed) & 0xFFFFFFFF


def _validate_sensor_budgets(values: tuple[int, ...]) -> tuple[int, ...]:
    """Return unique sorted positive sensor budgets."""
    if not values or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in values
    ):
        raise ValueError("sensor_budgets must contain positive integers")
    return tuple(sorted(set(values)))


def _validate_result(
    allocation: tuple[int, ...],
    error_probability: float,
    M: int,
    budget: int,
) -> None:
    """Validate one algorithm's fixed-budget allocation and probability."""
    if len(allocation) != M or sum(allocation) != budget:
        raise RuntimeError("algorithm returned an invalid fixed-budget allocation")
    if any(count < 0 for count in allocation):
        raise RuntimeError("algorithm returned a negative sensor count")
    _validate_probability(error_probability)


def _validate_probability(value: float) -> None:
    """Reject nonfinite values and values outside the probability interval."""
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError("algorithm returned an invalid PGM-error probability")


def _stratified_case_sample(
    cases: list[dict[str, object]],
    requested_count: int,
    seed: int,
) -> list[dict[str, object]]:
    """Sample cases evenly from field-family, K, and M design cells."""
    if requested_count >= len(cases):
        return list(cases)
    groups: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for case in cases:
        key = (str(case["field_model"]), int(case["K"]), int(case["M"]))
        groups[key].append(case)
    if requested_count < len(groups):
        raise ValueError(
            f"test_case_count must be at least {len(groups)} to cover every "
            "field-family/K/M cell"
        )

    rng = Random(seed)
    shuffled = {key: rng.sample(group, len(group)) for key, group in groups.items()}
    selected: list[dict[str, object]] = []
    ordered_keys = sorted(groups)
    while len(selected) < requested_count:
        for key in ordered_keys:
            if shuffled[key] and len(selected) < requested_count:
                selected.append(shuffled[key].pop())
    return selected


def _save_results(
    records: list[dict[str, object]],
    *,
    dataset: str,
    run_mode: str,
    sensor_budgets: tuple[int, ...],
    selected_case_count: int,
    total_case_count: int,
    sample_seed: int,
    results_root: Path,
    method_keys: tuple[str, ...] = METHOD_KEYS,
) -> None:
    """Write machine-readable paired records and run metadata."""
    output = results_root / dataset
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "objective": "actual equal-prior PGM error from the Gram matrix",
        "methods": [METHOD_LABELS[key] for key in method_keys],
        "input_phase": "unwrapped delta_ijm with c = pi",
        "run_mode": run_mode,
        "sensor_budgets": list(sensor_budgets),
        "selected_case_count": selected_case_count,
        "total_case_count": total_case_count,
        "sample_seed": sample_seed,
        "records": records,
    }
    (output / "paired_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    if records:
        with (output / "paired_results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def _percentile(sorted_values: list[float], probability: float) -> float:
    """Return a linearly interpolated percentile from sorted values."""
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )
