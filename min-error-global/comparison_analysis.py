"""Evaluate Optimal, Greedy, and VND global-discrimination allocations."""

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
from algorithms.optimal import optimal_global_discrimination  # noqa: E402
from algorithms.vnd import (  # noqa: E402
    DEFAULT_IMPROVEMENT_THRESHOLD,
    vnd_global_discrimination,
)
from input_data import load_delta_input  # noqa: E402
from objective import STATE_MODELS, pgm_error_probability  # noqa: E402


INPUT_ROOT = HERE / "inputs"
DEFAULT_RESULTS_ROOT = HERE / "results" / "actual_pgm_optimal_greedy_vnd"
STATE_MODEL_LABELS = {
    "product": "Product",
    "local_ghz": "Local GHZ",
    "global_ghz": "Global GHZ",
}
METHOD_STYLES = {
    "Optimal": {
        "color": "#2171B5",
        "linestyle": "-",
        "marker": "o",
        "markersize": 6,
        "zorder": 3,
    },
    "Greedy": {
        "color": "#D95F02",
        "linestyle": "--",
        "marker": "s",
        "markersize": 5,
        "zorder": 2,
    },
    # The x marker remains visible when VND and Optimal have identical means.
    "VND": {
        "color": "#287271",
        "linestyle": ":",
        "marker": "x",
        "markersize": 7,
        "markeredgewidth": 1.5,
        "zorder": 4,
    },
}


def run_paired_dataset(
    dataset: str,
    *,
    sensor_budgets: tuple[int, ...] = tuple(range(1, 11)),
    vnd_improvement_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD,
    run_mode: str = "test",
    test_case_count: int = 100,
    sample_seed: int = 20260827,
    progress_every: int = 10,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
) -> list[dict[str, object]]:
    """Run Optimal, Greedy, and VND on one compact benchmark.

    In ``test`` mode, exactly ``test_case_count`` cases are sampled as evenly
    as possible across field-family, K, and M cells. ``full`` mode evaluates
    every case. Each selected case uses the requested budgets and all three
    Section V input-state models. ``sensor_budgets`` is an absolute-N schedule
    defined by this global-discrimination experiment; manifest budgets and
    sequential zero-error thresholds are deliberately ignored.
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
    manifest_path = dataset_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
        K, M, delta_ijm = load_delta_input(
            dataset_directory / str(case["file"])
        )
        for budget in budgets:
            for state_model in STATE_MODELS:
                started = perf_counter()
                optimal = optimal_global_discrimination(
                    K, M, budget, delta_ijm, state_model
                )
                optimal_seconds = perf_counter() - started

                started = perf_counter()
                greedy = greedy_global_discrimination(
                    K, M, budget, delta_ijm, state_model
                )
                greedy_seconds = perf_counter() - started

                started = perf_counter()
                vnd = vnd_global_discrimination(
                    K,
                    M,
                    budget,
                    delta_ijm,
                    state_model,
                    improvement_threshold=vnd_improvement_threshold,
                )
                vnd_seconds = perf_counter() - started

                _validate_result(
                    optimal.allocation,
                    optimal.error_probability,
                    optimal.error_upper_bound,
                    M,
                    budget,
                    K,
                )
                _validate_result(
                    greedy.allocation,
                    greedy.error_probability,
                    greedy.error_upper_bound,
                    M,
                    budget,
                    K,
                )
                _validate_result(
                    vnd.allocation,
                    vnd.error_probability,
                    vnd.error_upper_bound,
                    M,
                    budget,
                    K,
                )
                if optimal.error_probability > greedy.error_probability + 1e-12:
                    raise RuntimeError("greedy PGM error is below the exact optimum")
                if optimal.error_probability > vnd.error_probability + 1e-12:
                    raise RuntimeError("VND PGM error is below the exact optimum")

                optimal_pgm_error = pgm_error_probability(
                    K,
                    M,
                    optimal.allocation,
                    delta_ijm,
                    state_model,
                )
                greedy_pgm_error = pgm_error_probability(
                    K,
                    M,
                    greedy.allocation,
                    delta_ijm,
                    state_model,
                )
                vnd_pgm_error = pgm_error_probability(
                    K,
                    M,
                    vnd.allocation,
                    delta_ijm,
                    state_model,
                )
                _validate_probability(optimal_pgm_error)
                _validate_probability(greedy_pgm_error)
                _validate_probability(vnd_pgm_error)
                for algorithm_name, reported, recomputed in (
                    ("Optimal", optimal.error_probability, optimal_pgm_error),
                    ("Greedy", greedy.error_probability, greedy_pgm_error),
                    ("VND", vnd.error_probability, vnd_pgm_error),
                ):
                    if abs(reported - recomputed) > 1e-12:
                        raise RuntimeError(
                            f"{algorithm_name} returned an inconsistent PGM error"
                        )

                gap = greedy.error_upper_bound - optimal.error_upper_bound
                pgm_error_gap = max(
                    0.0, greedy_pgm_error - optimal_pgm_error
                )
                vnd_gap = vnd.error_upper_bound - optimal.error_upper_bound
                vnd_pgm_error_gap = max(
                    0.0, vnd_pgm_error - optimal_pgm_error
                )
                records.append(
                    {
                        "dataset": dataset,
                        "case": Path(str(case["file"])).stem,
                        "field_model": str(case["field_model"]),
                        "spatial_dimension": int(case["spatial_dimension"]),
                        "replicate": int(case["replicate"]),
                        "K": K,
                        "M": M,
                        "sensor_budget": budget,
                        "state_model": state_model,
                        "optimal_upper_bound": optimal.error_upper_bound,
                        "greedy_upper_bound": greedy.error_upper_bound,
                        "vnd_upper_bound": vnd.error_upper_bound,
                        "optimal_pgm_error": optimal_pgm_error,
                        "greedy_pgm_error": greedy_pgm_error,
                        "vnd_pgm_error": vnd_pgm_error,
                        "greedy_upper_bound_difference": gap,
                        "greedy_pgm_error_gap": pgm_error_gap,
                        "vnd_upper_bound_difference": vnd_gap,
                        "vnd_pgm_error_gap": vnd_pgm_error_gap,
                        "greedy_relative_pgm_error_gap": (
                            pgm_error_gap / optimal_pgm_error
                            if optimal_pgm_error > 1e-15
                            else (0.0 if pgm_error_gap <= 1e-15 else None)
                        ),
                        "vnd_relative_pgm_error_gap": (
                            vnd_pgm_error_gap / optimal_pgm_error
                            if optimal_pgm_error > 1e-15
                            else (0.0 if vnd_pgm_error_gap <= 1e-15 else None)
                        ),
                        "greedy_same_allocation": (
                            optimal.allocation == greedy.allocation
                        ),
                        "vnd_same_allocation": optimal.allocation == vnd.allocation,
                        "optimal_seconds": optimal_seconds,
                        "greedy_seconds": greedy_seconds,
                        "vnd_seconds": vnd_seconds,
                        "greedy_runtime_speedup": (
                            optimal_seconds / greedy_seconds
                            if greedy_seconds > 0.0
                            else float("inf")
                        ),
                        "vnd_runtime_speedup": (
                            optimal_seconds / vnd_seconds
                            if vnd_seconds > 0.0
                            else float("inf")
                        ),
                        "vnd_improvement_threshold": vnd_improvement_threshold,
                        "vnd_objective_evaluations": vnd.objective_evaluations,
                        "vnd_cache_hits": vnd.cache_hits,
                        "vnd_accepted_v1_moves": vnd.accepted_v1_moves,
                        "vnd_accepted_v2_moves": vnd.accepted_v2_moves,
                        "optimal_allocation": list(optimal.allocation),
                        "greedy_allocation": list(greedy.allocation),
                        "vnd_allocation": list(vnd.allocation),
                    }
                )

        if case_number % progress_every == 0 or case_number == len(cases):
            print(
                f"{dataset}: completed {case_number}/{len(cases)} cases "
                f"({case['field_model']}, K={K}, M={M})"
            )

    save_paired_results(
        dataset,
        records,
        results_root,
        run_mode=run_mode,
        selected_case_count=len(cases),
        available_case_count=len(all_cases),
        sample_seed=sample_seed,
    )
    return records


def _validate_sensor_budgets(sensor_budgets: tuple[int, ...]) -> tuple[int, ...]:
    """Validate and canonicalize the global experiment's absolute budgets."""
    if not sensor_budgets:
        raise ValueError("sensor_budgets must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in sensor_budgets
    ):
        raise ValueError("sensor_budgets must contain positive integers")
    if len(set(sensor_budgets)) != len(sensor_budgets):
        raise ValueError("sensor_budgets must not contain duplicates")
    return tuple(sorted(sensor_budgets))


def _stratified_case_sample(
    cases: list[dict[str, object]],
    sample_size: int,
    seed: int,
) -> list[dict[str, object]]:
    """Sample exactly ``sample_size`` cases nearly evenly across design cells."""
    if sample_size > len(cases):
        raise ValueError("test_case_count exceeds the available case count")

    grouped: defaultdict[tuple[str, int, int], list[dict[str, object]]] = (
        defaultdict(list)
    )
    for case in cases:
        key = (str(case["field_model"]), int(case["K"]), int(case["M"]))
        grouped[key].append(case)

    cells = sorted(grouped)
    if sample_size < len(cells):
        raise ValueError(
            "test_case_count must be at least the number of field-family/K/M cells"
        )
    base_count, extra_count = divmod(sample_size, len(cells))
    rng = Random(seed)
    extra_cells = set(rng.sample(cells, extra_count))
    if any(
        len(grouped[cell]) < base_count + int(cell in extra_cells)
        for cell in cells
    ):
        raise ValueError("a design cell does not contain enough cases")
    selected: list[dict[str, object]] = []
    for cell in cells:
        count = base_count + int(cell in extra_cells)
        selected.extend(rng.sample(grouped[cell], count))

    # Restore source order so progress output and saved records remain stable.
    source_order = {str(case["file"]): index for index, case in enumerate(cases)}
    selected.sort(key=lambda case: source_order[str(case["file"])])
    return selected


def _validate_result(
    allocation: tuple[int, ...],
    error_probability: float,
    upper_bound: float,
    M: int,
    budget: int,
    K: int,
) -> None:
    """Validate an allocation and both returned PGM metrics."""
    if len(allocation) != M or sum(allocation) != budget:
        raise RuntimeError("algorithm returned an invalid sensor allocation")
    if any(count < 0 for count in allocation):
        raise RuntimeError("algorithm returned a negative sensor count")
    if not isfinite(upper_bound) or not -1e-12 <= upper_bound <= K - 1 + 1e-12:
        raise RuntimeError("algorithm returned an invalid PGM upper bound")
    _validate_probability(error_probability)


def _validate_probability(value: float) -> None:
    """Validate a computed PGM error probability."""
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError("computed PGM error is not a probability")


def save_paired_results(
    dataset: str,
    records: list[dict[str, object]],
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    run_mode: str,
    selected_case_count: int,
    available_case_count: int,
    sample_seed: int,
) -> None:
    """Save complete JSON records and a compact CSV table."""
    tables_directory = _dataset_results_directory(
        results_root, dataset, run_mode
    ) / "tables"
    tables_directory.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": dataset,
        "run_mode": run_mode,
        "selected_case_count": selected_case_count,
        "available_case_count": available_case_count,
        "sample_seed": sample_seed,
        "result_count": len(records),
        "sensor_budgets": sorted(
            {int(record["sensor_budget"]) for record in records}
        ),
        "state_models": list(STATE_MODELS),
        "optimization_objective": "actual_equal_prior_pgm_error",
        "input_phase_definition": "unwrapped_delta_ijm",
        "vnd_improvement_threshold": (
            records[0]["vnd_improvement_threshold"] if records else None
        ),
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
        "sensor_budget",
        "state_model",
        "optimal_upper_bound",
        "greedy_upper_bound",
        "vnd_upper_bound",
        "optimal_pgm_error",
        "greedy_pgm_error",
        "vnd_pgm_error",
        "greedy_upper_bound_difference",
        "greedy_pgm_error_gap",
        "vnd_upper_bound_difference",
        "vnd_pgm_error_gap",
        "greedy_relative_pgm_error_gap",
        "vnd_relative_pgm_error_gap",
        "greedy_same_allocation",
        "vnd_same_allocation",
        "optimal_seconds",
        "greedy_seconds",
        "vnd_seconds",
        "greedy_runtime_speedup",
        "vnd_runtime_speedup",
        "vnd_improvement_threshold",
        "vnd_objective_evaluations",
        "vnd_cache_hits",
        "vnd_accepted_v1_moves",
        "vnd_accepted_v2_moves",
        "optimal_allocation",
        "greedy_allocation",
        "vnd_allocation",
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
    run_mode: str = "test",
) -> list[dict[str, object]]:
    """Load saved records without rerunning the allocation algorithms."""
    results_path = (
        _dataset_results_directory(results_root, dataset, run_mode)
        / "tables"
        / "results.json"
    )
    return list(json.loads(results_path.read_text(encoding="utf-8"))["results"])


def paired_summary(
    records: list[dict[str, object]],
) -> list[dict[str, float | int | str]]:
    """Summarize actual-error-optimal Greedy and VND gaps by state model."""
    rows: list[dict[str, float | int | str]] = []
    dimension = f"{int(records[0]['spatial_dimension'])}D"
    for state_model in STATE_MODELS:
        selected = [
            record for record in records if record["state_model"] == state_model
        ]
        rows.append(
            {
                "dimension": dimension,
                "state_model": state_model,
                "cases": len({record["case"] for record in selected}),
                "runs": len(selected),
                "optimal_mean_bound": fmean(
                    float(record["optimal_upper_bound"]) for record in selected
                ),
                "greedy_mean_bound": fmean(
                    float(record["greedy_upper_bound"]) for record in selected
                ),
                "vnd_mean_bound": fmean(
                    float(record["vnd_upper_bound"]) for record in selected
                ),
                "optimal_mean_pgm_error": fmean(
                    float(record["optimal_pgm_error"]) for record in selected
                ),
                "greedy_mean_pgm_error": fmean(
                    float(record["greedy_pgm_error"]) for record in selected
                ),
                "vnd_mean_pgm_error": fmean(
                    float(record["vnd_pgm_error"]) for record in selected
                ),
                "mean_greedy_upper_bound_difference": fmean(
                    float(record["greedy_upper_bound_difference"])
                    for record in selected
                ),
                "mean_greedy_pgm_error_gap": fmean(
                    float(record["greedy_pgm_error_gap"])
                    for record in selected
                ),
                "mean_vnd_upper_bound_difference": fmean(
                    float(record["vnd_upper_bound_difference"])
                    for record in selected
                ),
                "mean_vnd_pgm_error_gap": fmean(
                    float(record["vnd_pgm_error_gap"])
                    for record in selected
                ),
            }
        )
    return rows


def bound_curve_summary(
    records: list[dict[str, object]],
) -> list[dict[str, float | int | str]]:
    """Summarize both PGM metrics at each absolute sensor budget."""
    rows: list[dict[str, float | int | str]] = []
    dimension = f"{int(records[0]['spatial_dimension'])}D"
    for state_model in STATE_MODELS:
        model_records = [
            record for record in records if record["state_model"] == state_model
        ]
        for budget in sorted(
            {int(record["sensor_budget"]) for record in model_records}
        ):
            selected = [
                record
                for record in model_records
                if int(record["sensor_budget"]) == budget
            ]
            rows.append(
                {
                    "dimension": dimension,
                    "state_model": state_model,
                    "sensor_budget": budget,
                    "optimal_mean_bound": fmean(
                        float(record["optimal_upper_bound"]) for record in selected
                    ),
                    "greedy_mean_bound": fmean(
                        float(record["greedy_upper_bound"]) for record in selected
                    ),
                    "vnd_mean_bound": fmean(
                        float(record["vnd_upper_bound"]) for record in selected
                    ),
                    "optimal_mean_pgm_error": fmean(
                        float(record["optimal_pgm_error"]) for record in selected
                    ),
                    "greedy_mean_pgm_error": fmean(
                        float(record["greedy_pgm_error"]) for record in selected
                    ),
                    "vnd_mean_pgm_error": fmean(
                        float(record["vnd_pgm_error"]) for record in selected
                    ),
                    "mean_greedy_upper_bound_difference": fmean(
                        float(record["greedy_upper_bound_difference"])
                        for record in selected
                    ),
                    "mean_greedy_pgm_error_gap": fmean(
                        float(record["greedy_pgm_error_gap"])
                        for record in selected
                    ),
                    "mean_vnd_upper_bound_difference": fmean(
                        float(record["vnd_upper_bound_difference"])
                        for record in selected
                    ),
                    "mean_vnd_pgm_error_gap": fmean(
                        float(record["vnd_pgm_error_gap"])
                        for record in selected
                    ),
                    "runs": len(selected),
                }
            )
    return rows


def make_combined_comparison_plots(
    records_by_dataset: dict[str, list[dict[str, object]]],
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    run_mode: str = "test",
) -> list[tuple[str, plt.Figure]]:
    """Create absolute three-method curves and Optimal-relative gap curves."""
    if run_mode not in {"test", "full"}:
        raise ValueError("run_mode must be 'test' or 'full'")
    dimension_records = sorted(
        records_by_dataset.values(),
        key=lambda records: int(records[0]["spatial_dimension"]),
    )
    if [int(records[0]["spatial_dimension"]) for records in dimension_records] != [1, 2]:
        raise ValueError("combined plots require one 1D and one 2D dataset")

    dataset_names = "_".join(records_by_dataset)
    figures_directory = (
        Path(results_root)
        / f"results_summary_{dataset_names}_{run_mode}"
        / "figures"
    )
    figures_directory.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()

    for obsolete_name in (
        "upper_bound_gap_curves",
        "upper_bound_gap_by_field_family",
        "greedy_runtime_speedup",
    ):
        for suffix in ("png", "pdf"):
            (figures_directory / f"{obsolete_name}.{suffix}").unlink(
                missing_ok=True
            )

    figures = [
        (
            "Mean PGM upper bounds (all methods)",
            _plot_mean_metric_curves(
                dimension_records,
                summary_keys={
                    "Optimal": "optimal_mean_bound",
                    "Greedy": "greedy_mean_bound",
                    "VND": "vnd_mean_bound",
                },
                ylabel="mean PGM upper bound",
            ),
            "mean_upper_bound_curves",
        ),
        (
            "Mean actual PGM errors (all methods)",
            _plot_mean_metric_curves(
                dimension_records,
                summary_keys={
                    "Optimal": "optimal_mean_pgm_error",
                    "Greedy": "greedy_mean_pgm_error",
                    "VND": "vnd_mean_pgm_error",
                },
                ylabel="mean actual PGM error",
            ),
            "mean_pgm_error_curves",
        ),
        (
            "Mean PGM upper-bound differences (Method - actual-error Optimal)",
            _plot_mean_gap_curves(
                dimension_records,
                summary_keys={
                    "Greedy": "mean_greedy_upper_bound_difference",
                    "VND": "mean_vnd_upper_bound_difference",
                },
                ylabel="mean upper-bound difference",
            ),
            "mean_upper_bound_gap_curves",
        ),
        (
            "Mean actual PGM-error gaps (Method - Optimal)",
            _plot_mean_gap_curves(
                dimension_records,
                summary_keys={
                    "Greedy": "mean_greedy_pgm_error_gap",
                    "VND": "mean_vnd_pgm_error_gap",
                },
                ylabel="mean PGM-error gap",
                nonnegative=True,
            ),
            "mean_pgm_error_gap_curves",
        ),
    ]

    output: list[tuple[str, plt.Figure]] = []
    for title, figure, filename in figures:
        figure.savefig(figures_directory / f"{filename}.png", dpi=300)
        figure.savefig(figures_directory / f"{filename}.pdf")
        output.append((title, figure))
    return output


def _plot_mean_metric_curves(
    dimension_records: list[list[dict[str, object]]],
    *,
    summary_keys: dict[str, str],
    ylabel: str,
) -> plt.Figure:
    """Plot absolute mean metrics for Optimal, Greedy, and VND."""
    figure, axes = plt.subplots(
        3, 2, figsize=(10.8, 11.0), sharex="col", sharey="row"
    )
    for row, state_model in enumerate(STATE_MODELS):
        for column, records in enumerate(dimension_records):
            axis = axes[row, column]
            summaries = [
                item
                for item in bound_curve_summary(records)
                if item["state_model"] == state_model
            ]
            x = [int(item["sensor_budget"]) for item in summaries]
            for method, summary_key in summary_keys.items():
                axis.plot(
                    x,
                    [float(item[summary_key]) for item in summaries],
                    label=method,
                    **METHOD_STYLES[method],
                )
            if row == 0:
                axis.set_title(f"{int(records[0]['spatial_dimension'])}D")
            if column == 0:
                axis.set_ylabel(
                    f"{STATE_MODEL_LABELS[state_model]}\n{ylabel}"
                )
            if row == len(STATE_MODELS) - 1:
                axis.set_xlabel(r"Sensor budget $N$")
                axis.set_xticks(x)
            axis.margins(y=0.1)
            axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def _plot_mean_gap_curves(
    dimension_records: list[list[dict[str, object]]],
    *,
    summary_keys: dict[str, str],
    ylabel: str,
    nonnegative: bool = False,
) -> plt.Figure:
    """Plot method-minus-actual-error-Optimal differences by state model."""
    figure, axes = plt.subplots(
        3, 2, figsize=(10.8, 11.0), sharex="col", sharey="row"
    )
    for row, state_model in enumerate(STATE_MODELS):
        for column, records in enumerate(dimension_records):
            axis = axes[row, column]
            summaries = [
                item
                for item in bound_curve_summary(records)
                if item["state_model"] == state_model
            ]
            x = [int(item["sensor_budget"]) for item in summaries]
            for method, summary_key in summary_keys.items():
                axis.plot(
                    x,
                    [float(item[summary_key]) for item in summaries],
                    label=method,
                    **METHOD_STYLES[method],
                )
            axis.axhline(0.0, color="#555555", linewidth=0.9)
            if row == 0:
                axis.set_title(f"{int(records[0]['spatial_dimension'])}D")
            if column == 0:
                axis.set_ylabel(
                    f"{STATE_MODEL_LABELS[state_model]}\n{ylabel}"
                )
            if row == len(STATE_MODELS) - 1:
                axis.set_xlabel(r"Sensor budget $N$")
                axis.set_xticks(x)
            axis.margins(y=0.15)
            if nonnegative:
                axis.set_ylim(bottom=0.0)
            axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
    )
    figure.suptitle("Method - actual-error Optimal", y=0.955)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure


def _dataset_results_directory(
    results_root: str | Path,
    dataset: str,
    run_mode: str,
) -> Path:
    """Return the test- or full-mode directory for one dataset."""
    if run_mode not in {"test", "full"}:
        raise ValueError("run_mode must be 'test' or 'full'")
    return Path(results_root) / f"{dataset}_results_{run_mode}"


def _configure_plot_style() -> None:
    """Apply a restrained publication-oriented Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
