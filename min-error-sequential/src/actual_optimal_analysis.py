"""Actual-error optimality validation on compact benchmark samples."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from time import perf_counter

import matplotlib.pyplot as plt

from actual_error_analysis import actual_node_probabilities, actual_tree_error
from comparison_analysis import load_paired_results
from exhaustive_search import DecisionLeaf, DecisionNode, _budget_allocations
from input_data import PhaseGaps, load_sparse_input
from plot_style import (
    METHOD_COLORS,
    METHOD_LINESTYLES,
    METHOD_MARKERS,
    REFERENCE_LINE_COLOR,
    configure_plot_style,
)


HERE = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_RESULTS_ROOT = HERE / "results" / "cmp_compact"
DEFAULT_RESULTS_ROOT = HERE / "results" / "actual_optimal"
DEFAULT_INPUT_ROOT = HERE / "inputs"
COMPACT_DATASETS = ("compact_benchmark_1d", "compact_benchmark_2d")
FIELD_FAMILY_ORDER = ("gaussian", "sinusoidal", "quadratic")
DEFAULT_K_VALUES = (3, 4)
DEFAULT_M_VALUES = (3, 4)

configure_plot_style()


@dataclass(frozen=True)
class ActualOptimalResult:
    """The allocation and tree with minimum actual error."""

    allocation: tuple[int, ...]
    decision_tree: DecisionLeaf | DecisionNode
    minimum_actual_error: float


def run_actual_optimal_validation(
    datasets: Sequence[str] = COMPACT_DATASETS,
    *,
    comparison_results_root: str | Path = DEFAULT_COMPARISON_RESULTS_ROOT,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    comparison_run_scope: str = "test",
    replicates_per_cell: int = 1,
    k_values: Sequence[int] | None = DEFAULT_K_VALUES,
    m_values: Sequence[int] | None = DEFAULT_M_VALUES,
    sample_scope: str | None = None,
    progress_every: int = 1,
) -> dict[str, list[dict[str, object]]]:
    """Compare surrogate-optimal actual error with true actual-optimal error."""
    records_by_dataset: dict[str, list[dict[str, object]]] = {}
    output_scope = sample_scope or sample_scope_name(
        replicates_per_cell,
        comparison_run_scope=comparison_run_scope,
        k_values=k_values,
        m_values=m_values,
    )
    for dataset in datasets:
        saved_records = load_paired_results(
            dataset,
            results_root=comparison_results_root,
            run_scope=comparison_run_scope,
        )
        selected_records = select_balanced_compact_sample(
            saved_records,
            replicates_per_cell=replicates_per_cell,
            k_values=k_values,
            m_values=m_values,
        )
        print(
            f"{dataset}: selected {len(selected_records)} records "
            f"({len({record['case'] for record in selected_records})} cases, "
            f"K={sorted({int(record['K']) for record in selected_records})}, "
            f"M={sorted({int(record['M']) for record in selected_records})})",
            flush=True,
        )
        records_by_dataset[dataset] = evaluate_actual_optimal_records(
            dataset,
            selected_records,
            input_root=input_root,
            progress_every=progress_every,
        )
        save_actual_optimal_results(
            dataset,
            records_by_dataset[dataset],
            results_root=results_root,
            sample_scope=output_scope,
            parameters={
                "comparison_run_scope": comparison_run_scope,
                "replicates_per_cell": replicates_per_cell,
                "k_values": list(k_values) if k_values is not None else None,
                "m_values": list(m_values) if m_values is not None else None,
            },
        )
    return records_by_dataset


def select_balanced_compact_sample(
    records: Sequence[Mapping[str, object]],
    *,
    replicates_per_cell: int = 1,
    k_values: Sequence[int] | None = DEFAULT_K_VALUES,
    m_values: Sequence[int] | None = DEFAULT_M_VALUES,
) -> list[dict[str, object]]:
    """Keep a symmetric set of compact records across field families, K, and M."""
    if replicates_per_cell < 1:
        raise ValueError("replicates_per_cell must be positive")
    selected_k = None if k_values is None else {int(value) for value in k_values}
    selected_m = None if m_values is None else {int(value) for value in m_values}
    selected = [
        dict(record)
        for record in records
        if int(record["replicate"]) <= replicates_per_cell
        and (selected_k is None or int(record["K"]) in selected_k)
        and (selected_m is None or int(record["M"]) in selected_m)
    ]
    return sorted(
        selected,
        key=lambda record: (
            int(record["spatial_dimension"]),
            FIELD_FAMILY_ORDER.index(str(record["field_model"]))
            if str(record["field_model"]) in FIELD_FAMILY_ORDER
            else len(FIELD_FAMILY_ORDER),
            str(record["field_model"]),
            int(record["K"]),
            int(record["M"]),
            int(record["replicate"]),
            str(record["case"]),
            int(record["budget_level"]),
        ),
    )


def evaluate_actual_optimal_records(
    dataset: str,
    records: Sequence[Mapping[str, object]],
    *,
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    progress_every: int = 1,
) -> list[dict[str, object]]:
    """Evaluate the actual-optimal gap for selected saved surrogate records."""
    input_directory = Path(input_root) / dataset
    input_cache: dict[str, tuple[int, int, PhaseGaps, object]] = {}
    evaluated: list[dict[str, object]] = []

    for index, record in enumerate(records, start=1):
        case_name = str(record["case"])
        if case_name not in input_cache:
            input_cache[case_name] = load_sparse_input(input_directory / f"{case_name}.csv")
        K, M, phase_gaps, pair_error_table = input_cache[case_name]
        if K != int(record["K"]) or M != int(record["M"]):
            raise AssertionError(
                "saved comparison record does not match loaded input case: "
                f"record K={record['K']}, input K={K}, "
                f"record M={record['M']}, input M={M}, case={case_name}"
            )
        budget = int(record["sensor_budget"])

        started = perf_counter()
        actual_optimal = actual_exhaustive_search(
            K,
            M,
            budget,
            phase_gaps,
            pair_error_table,
        )
        actual_optimal_seconds = perf_counter() - started

        surrogate_allocation = tuple(int(value) for value in record["optimal_allocation"])
        surrogate_actual_error = actual_tree_error(
            record["optimal_decision_tree"],
            K,
            phase_gaps,
            surrogate_allocation,
        )
        actual_optimal_error = actual_optimal.minimum_actual_error
        actual_optimal_recomputed_error = actual_tree_error(
            actual_optimal.decision_tree,
            K,
            phase_gaps,
            actual_optimal.allocation,
        )
        if abs(actual_optimal_recomputed_error - actual_optimal_error) > 1e-9:
            raise AssertionError(
                "actual-optimal recursion and evaluator disagree: "
                f"recursive={actual_optimal_error:.12g}, "
                f"recomputed={actual_optimal_recomputed_error:.12g}, "
                f"case={case_name}, K={K}, M={M}, N={budget}"
            )
        gap = surrogate_actual_error - actual_optimal_error
        if gap < -1e-9:
            raise AssertionError(
                "actual-optimal search returned a solution worse than the "
                "surrogate-optimal tree under actual evaluation: "
                f"gap={gap:.12g}, case={case_name}, K={K}, M={M}, N={budget}"
            )

        evaluated.append(
            {
                "dataset": dataset,
                "case": case_name,
                "field_model": str(record["field_model"]),
                "spatial_dimension": int(record["spatial_dimension"]),
                "replicate": int(record["replicate"]),
                "K": K,
                "M": M,
                "sensor_budget": budget,
                "budget_level": int(record["budget_level"]),
                "budget_fraction_of_N_min": float(record["budget_fraction_of_N_min"]),
                "surrogate_optimal_surrogate_error": float(record["optimal_error"]),
                "surrogate_optimal_actual_error": surrogate_actual_error,
                "actual_optimal_actual_error": actual_optimal_error,
                "actual_optimal_recomputed_actual_error": actual_optimal_recomputed_error,
                "actual_optimality_gap": gap,
                "actual_optimal_seconds": actual_optimal_seconds,
                "surrogate_optimal_allocation": list(surrogate_allocation),
                "actual_optimal_allocation": list(actual_optimal.allocation),
            }
        )
        if (
            progress_every > 0
            and (index == 1 or index % progress_every == 0 or index == len(records))
        ):
            print(
                f"{dataset}: evaluated {index}/{len(records)} "
                f"(case={case_name}, field={record['field_model']}, "
                f"K={K}, M={M}, budget_level={record['budget_level']}, "
                f"N={budget})",
                flush=True,
            )

    return evaluated


def actual_exhaustive_search(
    K: int,
    M: int,
    N: int,
    phase_gaps: PhaseGaps,
    pair_error_table: object,
) -> ActualOptimalResult:
    """Find the allocation and adaptive tree minimizing actual Helstrom error."""
    best: ActualOptimalResult | None = None
    for allocation in _budget_allocations(N, M):
        error, tree = _best_actual_tree(
            tuple(range(K)),
            tuple(1.0 / K for _ in range(K)),
            allocation,
            phase_gaps,
            pair_error_table,
        )
        if best is None or error < best.minimum_actual_error:
            best = ActualOptimalResult(
                allocation=allocation,
                decision_tree=tree,
                minimum_actual_error=error,
            )
    assert best is not None
    return best


def _best_actual_tree(
    remaining: tuple[int, ...],
    weights: tuple[float, ...],
    allocation: tuple[int, ...],
    phase_gaps: PhaseGaps,
    pair_error_table: object,
) -> tuple[float, DecisionLeaf | DecisionNode]:
    """Return the best actual-error tree below one candidate set."""
    if len(remaining) == 1:
        declared = remaining[0]
        error = sum(weight for field, weight in enumerate(weights) if field != declared)
        return error, DecisionLeaf(declared_field=declared)

    best_error = float("inf")
    best_tree: DecisionLeaf | DecisionNode | None = None
    for i_index, i in enumerate(remaining):
        for j in remaining[i_index + 1 :]:
            pair = (i, j)
            for location, sensor_count in enumerate(allocation):
                active_fields = [
                    field for field, weight in enumerate(weights) if weight > 0.0
                ]
                probabilities_by_field = {
                    field: actual_node_probabilities(
                        pair,
                        location,
                        sensor_count,
                        field,
                        phase_gaps,
                    )
                    for field in active_fields
                }
                discard_i_weights = tuple(
                    weight * probabilities_by_field[field].keep_j
                    if field in probabilities_by_field
                    else 0.0
                    for field, weight in enumerate(weights)
                )
                discard_j_weights = tuple(
                    weight * probabilities_by_field[field].keep_i
                    if field in probabilities_by_field
                    else 0.0
                    for field, weight in enumerate(weights)
                )
                discard_i_error, discard_i_tree = _best_actual_tree(
                    tuple(field for field in remaining if field != i),
                    discard_i_weights,
                    allocation,
                    phase_gaps,
                    pair_error_table,
                )
                discard_j_error, discard_j_tree = _best_actual_tree(
                    tuple(field for field in remaining if field != j),
                    discard_j_weights,
                    allocation,
                    phase_gaps,
                    pair_error_table,
                )
                total_error = discard_i_error + discard_j_error
                if total_error < best_error:
                    pair_error = _pair_error_at_location(pair_error_table, pair, location, sensor_count)
                    best_error = total_error
                    best_tree = DecisionNode(
                        pair=pair,
                        location=location,
                        pair_error=pair_error,
                        discard_i=discard_i_tree,
                        discard_j=discard_j_tree,
                    )

    assert best_tree is not None
    return best_error, best_tree


def _pair_error_at_location(
    pair_error_table: object,
    pair: tuple[int, int],
    location: int,
    sensor_count: int,
) -> float:
    location_table = pair_error_table[pair][location]
    available_counts = (count for count in location_table if count <= sensor_count)
    largest_available = max(available_counts, default=None)
    return 0.5 if largest_available is None else float(location_table[largest_available])


def summarize_actual_optimal_gaps(
    records_by_dataset: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    """Summarize actual-optimal gaps by dimension."""
    rows: list[dict[str, object]] = []
    for dataset, records in records_by_dataset.items():
        gaps = [float(record["actual_optimality_gap"]) for record in records]
        rows.append(
            {
                "dataset": dataset,
                "dimension": f"{int(records[0]['spatial_dimension'])}D",
                "runs": len(records),
                "mean_surrogate_optimal_actual_error": fmean(
                    float(record["surrogate_optimal_actual_error"]) for record in records
                ),
                "mean_actual_optimal_actual_error": fmean(
                    float(record["actual_optimal_actual_error"]) for record in records
                ),
                "mean_actual_optimality_gap": fmean(gaps),
                "max_actual_optimality_gap": max(gaps),
                "mean_actual_optimal_seconds": fmean(
                    float(record["actual_optimal_seconds"]) for record in records
                ),
            }
        )
    return rows


def gap_rows_by_budget(
    records_by_dataset: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    """Return mean actual-optimal gaps by dimension and budget level."""
    rows: list[dict[str, object]] = []
    for dataset, records in records_by_dataset.items():
        for budget_level in sorted({int(record["budget_level"]) for record in records}):
            selected = [
                record
                for record in records
                if int(record["budget_level"]) == budget_level
            ]
            rows.append(
                {
                    "dataset": dataset,
                    "dimension": f"{int(selected[0]['spatial_dimension'])}D",
                    "budget_level": budget_level,
                    "budget_ratio": fmean(
                        float(record["budget_fraction_of_N_min"]) for record in selected
                    ),
                    "runs": len(selected),
                    "surrogate_optimal_actual_error": fmean(
                        float(record["surrogate_optimal_actual_error"])
                        for record in selected
                    ),
                    "actual_optimal_actual_error": fmean(
                        float(record["actual_optimal_actual_error"]) for record in selected
                    ),
                    "actual_optimality_gap": fmean(
                        float(record["actual_optimality_gap"]) for record in selected
                    ),
                }
            )
    return rows


def gap_rows_by_field_family(
    records_by_dataset: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    """Return mean actual-optimal gaps by dimension and field family."""
    rows: list[dict[str, object]] = []
    for dataset, records in records_by_dataset.items():
        field_families = sorted(
            {str(record["field_model"]) for record in records},
            key=lambda value: (
                FIELD_FAMILY_ORDER.index(value)
                if value in FIELD_FAMILY_ORDER
                else len(FIELD_FAMILY_ORDER),
                value,
            ),
        )
        for field_family in field_families:
            selected = [
                record
                for record in records
                if str(record["field_model"]) == field_family
            ]
            gaps = [float(record["actual_optimality_gap"]) for record in selected]
            rows.append(
                {
                    "dataset": dataset,
                    "dimension": f"{int(selected[0]['spatial_dimension'])}D",
                    "field_family": field_family,
                    "runs": len(selected),
                    "mean_surrogate_optimal_actual_error": fmean(
                        float(record["surrogate_optimal_actual_error"])
                        for record in selected
                    ),
                    "mean_actual_optimal_actual_error": fmean(
                        float(record["actual_optimal_actual_error"])
                        for record in selected
                    ),
                    "mean_actual_optimality_gap": fmean(gaps),
                    "max_actual_optimality_gap": max(gaps),
                    "mean_actual_optimal_seconds": fmean(
                        float(record["actual_optimal_seconds"]) for record in selected
                    ),
                }
            )
    return rows


def make_actual_optimal_gap_plot(
    records_by_dataset: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    sample_scope: str = "replicate_1",
    figure_stem: str = "actual_optimality_gap",
    save: bool = True,
) -> plt.Figure:
    """Plot surrogate-optimal and actual-optimal actual errors by budget."""
    rows = gap_rows_by_budget(records_by_dataset)
    dimensions = sorted({row["dimension"] for row in rows})
    figure, axes = plt.subplots(1, len(dimensions), figsize=(10.8, 4.0), sharey=True)
    if len(dimensions) == 1:
        axes = [axes]

    styles = {
        "surrogate": {
            "label": r"Surrogate-optimal, evaluated by $P_{actual}$",
            "color": METHOD_COLORS["Optimal"],
            "linestyle": METHOD_LINESTYLES["Optimal"],
            "marker": METHOD_MARKERS["Optimal"],
        },
        "actual": {
            "label": r"Actual-optimal, evaluated by $P_{actual}$",
            "color": METHOD_COLORS["Heuristic"],
            "linestyle": METHOD_LINESTYLES["Heuristic"],
            "marker": METHOD_MARKERS["Heuristic"],
        },
    }
    for axis, dimension in zip(axes, dimensions):
        selected = [row for row in rows if row["dimension"] == dimension]
        axis.plot(
            [row["budget_ratio"] for row in selected],
            [row["surrogate_optimal_actual_error"] for row in selected],
            **styles["surrogate"],
        )
        axis.plot(
            [row["budget_ratio"] for row in selected],
            [row["actual_optimal_actual_error"] for row in selected],
            **styles["actual"],
        )
        axis.axhline(0.0, color=REFERENCE_LINE_COLOR, linewidth=1.0)
        axis.set_title(f"{dimension} compact sample")
        axis.set_xlabel(r"Budget ratio $\rho=N/N_{\min}$")
        axis.grid(alpha=0.22, linewidth=0.8)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Mean actual misclassification probability")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=len(labels))
    figure.tight_layout(rect=(0, 0, 1, 0.88))

    if save:
        figures_directory = Path(results_root) / f"summary_{sample_scope}" / "figures"
        figures_directory.mkdir(parents=True, exist_ok=True)
        figure.savefig(figures_directory / f"{figure_stem}.png", dpi=600, bbox_inches="tight")
        figure.savefig(figures_directory / f"{figure_stem}.pdf", bbox_inches="tight")
    return figure


def save_actual_optimal_results(
    dataset: str,
    records: Sequence[Mapping[str, object]],
    *,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    sample_scope: str = "replicate_1",
    parameters: Mapping[str, object] | None = None,
) -> None:
    """Save detailed actual-optimal validation records."""
    tables_directory = (
        Path(results_root) / f"{_result_dataset_name(dataset)}_{sample_scope}" / "tables"
    )
    tables_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "sample_scope": sample_scope,
        "result_count": len(records),
        "parameters": dict(parameters or {}),
        "results": list(records),
    }
    (tables_directory / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n",
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
        "surrogate_optimal_surrogate_error",
        "surrogate_optimal_actual_error",
        "actual_optimal_actual_error",
        "actual_optimal_recomputed_actual_error",
        "actual_optimality_gap",
        "actual_optimal_seconds",
    ]
    with (tables_directory / "summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({column: record[column] for column in columns})


def load_actual_optimal_results(
    dataset: str,
    *,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    sample_scope: str = "replicate_1",
) -> list[dict[str, object]]:
    """Load saved actual-optimal validation records."""
    path = (
        Path(results_root)
        / f"{_result_dataset_name(dataset)}_{sample_scope}"
        / "tables"
        / "results.json"
    )
    return list(json.loads(path.read_text(encoding="utf-8"))["results"])


def _result_dataset_name(dataset: str) -> str:
    """Return the short label used in result paths."""
    if dataset.startswith("compact_benchmark_"):
        return dataset.replace("compact_benchmark_", "compact_", 1)
    if dataset.startswith("lite_benchmark_"):
        return dataset.replace("lite_benchmark_", "lite_extended_", 1)
    if dataset.startswith("benchmark_"):
        return dataset.replace("benchmark_", "extended_", 1)
    return dataset


def sample_scope_name(
    replicates_per_cell: int,
    *,
    comparison_run_scope: str | None = None,
    k_values: Sequence[int] | None = None,
    m_values: Sequence[int] | None = None,
) -> str:
    """Return a stable folder suffix for a balanced compact replicate sample."""
    parts = []
    if comparison_run_scope:
        parts.append(str(comparison_run_scope))
    if k_values is not None:
        parts.append("k" + "-".join(str(int(value)) for value in k_values))
    if m_values is not None:
        parts.append("m" + "-".join(str(int(value)) for value in m_values))
    parts.append(f"replicate_{replicates_per_cell}")
    return "_".join(parts)
