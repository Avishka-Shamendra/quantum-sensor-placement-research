"""Evaluate tree-aware heuristic performance across beam widths."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from random import Random
from statistics import fmean
from time import perf_counter

import matplotlib.pyplot as plt

from heuristic import heuristic
from input_data import load_sparse_input, validate_reverse_phases


HERE = Path(__file__).resolve().parent
INPUT_ROOT = HERE / "inputs"
DEFAULT_RESULTS_ROOT = HERE / "results" / "heuristic_beam_width"
FIELD_FAMILY_ORDER = ("gaussian", "sinusoidal", "quadratic")
BEAM_WIDTH_COLORS = ("#4C78A8", "#F28E2B", "#59A14F", "#E15759", "#B279A2")
LINE_STYLES = ("-", "--", "-.", ":")


def run_beam_width_study(
    dataset: str,
    beam_widths: tuple[int, ...] = (1, 3, 5),
    replicates_per_cell: int | None = 1,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    max_iterations: int = 50,
    improvement_threshold: float = 1e-4,
    initialization: str = "near_uniform",
    random_seed: int | None = 0,
    progress_every: int = 25,
    k_values: tuple[int, ...] | None = None,
    m_values: tuple[int, ...] | None = None,
    random_cases_per_k_m: int | None = None,
    case_sample_seed: int = 20260813,
) -> list[dict[str, object]]:
    """Run the heuristic with several beam widths on the same cases."""
    widths = _validate_beam_widths(beam_widths)
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    dataset_directory = INPUT_ROOT / dataset
    manifest = json.loads(
        (dataset_directory / "manifest.json").read_text(encoding="utf-8")
    )
    cases = list(manifest["cases"])
    if replicates_per_cell is not None:
        cases = [
            case
            for case in cases
            if int(case["replicate"]) <= replicates_per_cell
        ]
    if k_values is not None:
        selected_k = set(k_values)
        cases = [case for case in cases if int(case["K"]) in selected_k]
    if m_values is not None:
        selected_m = set(m_values)
        cases = [case for case in cases if int(case["M"]) in selected_m]
    if random_cases_per_k_m is not None:
        if random_cases_per_k_m < 1:
            raise ValueError("random_cases_per_k_m must be positive")
        generator = Random(case_sample_seed)
        cases_by_k_m: defaultdict[tuple[int, int], list[dict[str, object]]] = (
            defaultdict(list)
        )
        for case in cases:
            cases_by_k_m[(int(case["K"]), int(case["M"]))].append(case)
        cases = []
        for cell in sorted(cases_by_k_m):
            cell_cases = cases_by_k_m[cell]
            if len(cell_cases) < random_cases_per_k_m:
                raise ValueError(f"not enough cases available for K={cell[0]}, M={cell[1]}")
            cases.extend(generator.sample(cell_cases, random_cases_per_k_m))
    if not cases:
        raise ValueError("the selected test settings contain no cases")

    records: list[dict[str, object]] = []
    for case_number, case in enumerate(cases, start=1):
        K, M, phase_gaps, pair_errors = load_sparse_input(
            dataset_directory / str(case["file"])
        )
        validate_reverse_phases(phase_gaps)
        minimum_sensors = int(case["minimum_sensors_for_zero_error"])
        budgets = [int(value) for value in case["sensor_budgets"]]

        for budget_level, budget in enumerate(budgets, start=1):
            for beam_width in widths:
                started = perf_counter()
                result = heuristic(
                    K,
                    M,
                    budget,
                    pair_errors,
                    beam_width=beam_width,
                    max_iterations=max_iterations,
                    improvement_threshold=improvement_threshold,
                    initialization=initialization,
                    random_seed=random_seed,
                )
                elapsed = perf_counter() - started

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
                        "budget_fraction_of_N_min": budget / minimum_sensors,
                        "beam_width": beam_width,
                        "heuristic_error": result.error_probability,
                        "heuristic_seconds": elapsed,
                        "allocation": list(result.allocation),
                        "pair_locations": {
                            f"{i},{j}": location
                            for (i, j), location in result.pair_locations.items()
                        },
                    }
                )

        if case_number == 1 or case_number % progress_every == 0 or case_number == len(cases):
            print(
                f"{dataset}: completed case {case_number}/{len(cases)} "
                f"({case['field_model']}, K={K}, M={M})"
            )

    run_scope = (
        "full"
        if replicates_per_cell is None and random_cases_per_k_m is None
        else "test"
    )
    parameters = {
        "beam_widths": list(widths),
        "max_iterations": max_iterations,
        "improvement_threshold": improvement_threshold,
        "initialization": initialization,
        "random_seed": random_seed,
        "progress_every": progress_every,
        "k_values": list(k_values) if k_values is not None else None,
        "m_values": list(m_values) if m_values is not None else None,
        "random_cases_per_k_m": random_cases_per_k_m,
        "case_sample_seed": case_sample_seed,
    }
    save_beam_width_results(
        dataset,
        records,
        results_root,
        run_scope=run_scope,
        parameters=parameters,
    )
    return records


def save_beam_width_results(
    dataset: str,
    records: list[dict[str, object]],
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    run_scope: str = "test",
    parameters: dict[str, object] | None = None,
) -> None:
    """Save detailed beam-width results and a compact CSV table."""
    tables_directory = (
        Path(results_root) / f"{dataset}_results_{run_scope}" / "tables"
    )
    tables_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "result_count": len(records),
        "parameters": parameters or {},
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
        "budget_level",
        "budget_fraction_of_N_min",
        "beam_width",
        "heuristic_error",
        "heuristic_seconds",
    ]
    with (tables_directory / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({column: record[column] for column in columns})


def make_beam_width_plot(
    records_by_dataset: dict[str, list[dict[str, object]]],
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    run_scope: str = "test",
) -> plt.Figure:
    """Plot final error by budget with one panel for each dimension."""
    dimension_records = sorted(
        records_by_dataset.values(),
        key=lambda records: int(records[0]["spatial_dimension"]),
    )
    if [int(records[0]["spatial_dimension"]) for records in dimension_records] != [1, 2]:
        raise ValueError("the plot requires one 1D and one 2D dataset")

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 4.8),
        sharey=True,
        constrained_layout=True,
    )
    for axis, records in zip(axes, dimension_records):
        dimension = int(records[0]["spatial_dimension"])
        widths = sorted({int(record["beam_width"]) for record in records})
        levels = sorted({int(record["budget_level"]) for record in records})

        # Average over all field families, K values, and M values.
        for width_index, beam_width in enumerate(widths):
            x_values: list[float] = []
            means: list[float] = []
            for level in levels:
                selected = [
                    record
                    for record in records
                    if int(record["beam_width"]) == beam_width
                    and int(record["budget_level"]) == level
                ]
                x_values.append(
                    fmean(
                        float(record["budget_fraction_of_N_min"])
                        for record in selected
                    )
                )
                errors = [float(record["heuristic_error"]) for record in selected]
                means.append(fmean(errors))

            axis.plot(
                x_values,
                means,
                marker="o",
                linestyle=LINE_STYLES[width_index % len(LINE_STYLES)],
                linewidth=1.7,
                color=BEAM_WIDTH_COLORS[width_index % len(BEAM_WIDTH_COLORS)],
                label=f"Beam width {beam_width}",
            )

        axis.set(
            title=f"{dimension}D",
        )
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Mean final error")
    axes[1].legend()
    figure.supxlabel(
        r"$N/N_{\min}$ (Sensor Budget / Required Min Sensors for Perfect Discrimination)"
    )
    figure.suptitle("Tree-aware heuristic error versus sensor budget by beam width")

    figures_directory = (
        Path(results_root)
        / f"results_summary_heuristic_beam_width_{run_scope}"
        / "figures"
    )
    figures_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        figures_directory / "mean_error_curves_by_beam_width.png", dpi=300
    )
    figure.savefig(figures_directory / "mean_error_curves_by_beam_width.pdf")
    return figure


def beam_width_summary(
    records: list[dict[str, object]],
) -> list[dict[str, float | int]]:
    """Summarize error and runtime for every tested beam width."""
    rows: list[dict[str, float | int]] = []
    for beam_width in sorted({int(record["beam_width"]) for record in records}):
        selected = [
            record for record in records if int(record["beam_width"]) == beam_width
        ]
        rows.append(
            {
                "beam_width": beam_width,
                "runs": len(selected),
                "mean_final_error": fmean(
                    float(record["heuristic_error"]) for record in selected
                ),
                "maximum_final_error": max(
                    float(record["heuristic_error"]) for record in selected
                ),
                "mean_seconds": fmean(
                    float(record["heuristic_seconds"]) for record in selected
                ),
            }
        )
    return rows


def _validate_beam_widths(beam_widths: tuple[int, ...]) -> tuple[int, ...]:
    """Validate and preserve the requested beam-width order."""
    if not beam_widths:
        raise ValueError("beam_widths must not be empty")
    if any(
        isinstance(width, bool) or not isinstance(width, int) or width < 1
        for width in beam_widths
    ):
        raise ValueError("every beam width must be a positive integer")
    if len(set(beam_widths)) != len(beam_widths):
        raise ValueError("beam widths must be unique")
    return tuple(beam_widths)
