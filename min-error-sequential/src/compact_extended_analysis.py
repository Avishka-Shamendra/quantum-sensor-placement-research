"""Create one figure comparing compact and extended benchmark results."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from plot_style import (
    METHOD_COLORS,
    METHOD_LINESTYLES,
    METHOD_MARKERS,
    configure_plot_style,
)


HERE = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = HERE / "results"
SENSOR_BUDGET_LABEL = (
    r"Budget ratio $\rho=N/N_{\min}^{opt}$"
)
ALGORITHM_STYLES = {
    algorithm: (
        METHOD_COLORS[algorithm],
        METHOD_LINESTYLES[algorithm],
        METHOD_MARKERS[algorithm],
    )
    for algorithm in ("Optimal", "Greedy", "Heuristic", "Uniform", "Random")
}


@dataclass(frozen=True)
class _Panel:
    """Describe one benchmark panel and its saved result columns."""

    title: str
    result_file: Path
    error_columns: tuple[tuple[str, str], ...]


def make_compact_extended_plot(
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    run_scope: str = "full",
) -> tuple[plt.Figure, Path]:
    """Plot compact and extended overall mean-error curves together."""
    if run_scope not in {"test", "full"}:
        raise ValueError("run_scope must be 'test' or 'full'")

    configure_plot_style()
    results_root = Path(results_root)
    panels = _result_panels(results_root, run_scope)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.2, 9.2),
        sharey=True,
        constrained_layout=True,
    )

    for axis, panel in zip(axes.flat, panels):
        budget_fractions, error_curves = _load_overall_curves(
            panel.result_file,
            panel.error_columns,
        )
        for algorithm, errors in error_curves.items():
            color, line_style, marker = ALGORITHM_STYLES[algorithm]
            axis.plot(
                budget_fractions,
                errors,
                color=color,
                linestyle=line_style,
                marker=marker,
                label=algorithm,
            )
        axis.set_title(panel.title)
        axis.grid(alpha=0.25)

    figure.supxlabel(SENSOR_BUDGET_LABEL)
    figure.supylabel("Mean surrogate final error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(labels),
    )

    output_directory = (
        results_root
        / "compact_extended"
        / f"summary_compact_extended_{run_scope}"
        / "figures"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_directory / "overall_mean_error_curves.png", dpi=300)
    figure.savefig(output_directory / "overall_mean_error_curves.pdf")
    return figure, output_directory


def _result_panels(results_root: Path, run_scope: str) -> tuple[_Panel, ...]:
    """Return the four panels in compact-then-extended order."""
    compact_columns = (
        ("Optimal", "optimal_error"),
        ("Greedy", "greedy_error"),
        ("Heuristic", "tree_aware_error"),
        ("Uniform", "uniform_error"),
        ("Random", "random_error"),
    )
    extended_columns = (
        ("Greedy", "greedy_error"),
        ("Heuristic", "heuristic_error"),
        ("Uniform", "uniform_error"),
        ("Random", "random_error"),
    )
    compact_root = results_root / "cmp_compact"
    extended_root = results_root / "cmp_extended"
    return (
        _Panel(
            "1D – Compact",
            _summary_file(compact_root, "compact_benchmark_1d", run_scope),
            compact_columns,
        ),
        _Panel(
            "2D – Compact",
            _summary_file(compact_root, "compact_benchmark_2d", run_scope),
            compact_columns,
        ),
        _Panel(
            "1D – Extended",
            _summary_file(extended_root, "benchmark_1d", run_scope),
            extended_columns,
        ),
        _Panel(
            "2D – Extended",
            _summary_file(extended_root, "benchmark_2d", run_scope),
            extended_columns,
        ),
    )


def _summary_file(root: Path, dataset: str, run_scope: str) -> Path:
    """Build the path to one saved compact CSV result table."""
    return root / f"{_result_dataset_name(dataset)}_{run_scope}" / "tables" / "summary.csv"


def _result_dataset_name(dataset: str) -> str:
    """Return the short label used in result paths."""
    if dataset.startswith("compact_benchmark_"):
        return dataset.replace("compact_benchmark_", "compact_", 1)
    if dataset.startswith("lite_benchmark_"):
        return dataset.replace("lite_benchmark_", "lite_extended_", 1)
    if dataset.startswith("benchmark_"):
        return dataset.replace("benchmark_", "extended_", 1)
    return dataset


def _load_overall_curves(
    result_file: Path,
    error_columns: tuple[tuple[str, str], ...],
) -> tuple[list[float], dict[str, list[float]]]:
    """Average normalized budget and error values by budget level."""
    if not result_file.is_file():
        raise FileNotFoundError(
            f"Saved results were not found: {result_file}. "
            "Run the matching comparison notebook first."
        )

    counts: defaultdict[int, int] = defaultdict(int)
    budget_sums: defaultdict[int, float] = defaultdict(float)
    error_sums = {
        algorithm: defaultdict(float) for algorithm, _ in error_columns
    }

    with result_file.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {
            "budget_level",
            "budget_fraction_of_N_min",
            *(column for _, column in error_columns),
        }
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"{result_file} is missing columns: {missing_text}. "
                "Rerun the matching comparison notebook."
            )

        for record in reader:
            level = int(record["budget_level"])
            counts[level] += 1
            budget_sums[level] += float(record["budget_fraction_of_N_min"])
            for algorithm, column in error_columns:
                error_sums[algorithm][level] += float(record[column])

    if not counts:
        raise ValueError(f"Saved results contain no rows: {result_file}")

    levels = sorted(counts)
    budget_fractions = [budget_sums[level] / counts[level] for level in levels]
    curves = {
        algorithm: [
            error_sums[algorithm][level] / counts[level] for level in levels
        ]
        for algorithm, _ in error_columns
    }
    return budget_fractions, curves
