"""Statistical summaries and figures for both benchmarks."""

from __future__ import annotations

import csv
import json
import math
import random
import zlib
from run_config import validate_run_mode
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median


FIELD_FAMILY_COLORS = {
    "gaussian": "#0072B2",
    "sinusoidal": "#009E73",
    "quadratic": "#E69F00",
}


# Calculate a percentile using linear interpolation between sample values.
def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sample")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


# Estimate a mean confidence interval while preserving the chosen strata.
def _stratified_bootstrap_mean_interval(
    rows: list[dict[str, object]],
    *,
    value_key: str,
    stratum_keys: tuple[str, ...],
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    strata: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in stratum_keys)
        strata[key].append(float(row[value_key]))
    rng = random.Random(seed)
    estimates: list[float] = []

    # Resample within each stratum so the factorial design stays balanced.
    for _ in range(iterations):
        sample: list[float] = []
        for values in strata.values():
            sample.extend(rng.choices(values, k=len(values)))
        estimates.append(fmean(sample))
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


# Calculate a 95% Wilson interval for a success proportion.
def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return centre - half_width, centre + half_width


def case_metrics(record: dict[str, object]) -> dict[str, object]:
    """Flatten one solver result and derive the comparison measures."""
    optimal = record["optimal"]
    greedy = record["greedy"]
    comparison = record["comparison"]
    assert isinstance(optimal, dict)
    assert isinstance(greedy, dict)
    assert isinstance(comparison, dict)

    ratio = float(comparison["greedy_to_optimal_ratio"])
    theorem_factor = float(comparison["theorem_factor"])
    optimal_seconds = float(optimal["mean_seconds"])
    greedy_seconds = float(greedy["mean_seconds"])
    return {
        "dataset": record.get("dataset"),
        "spatial_dimension": int(record["spatial_dimension"]),
        "case": record["case"],
        "field_model": record["field_model"],
        "replicate": record.get("replicate"),
        "generation_seed": record.get("generation_seed"),
        "phase_scale_c": record.get("phase_scale_c"),
        "K": int(record["K"]),
        "M": int(record["M"]),
        "N_optimal": optimal["minimum_sensors"],
        "N_greedy": greedy["minimum_sensors"],
        "greedy_ratio": ratio,
        "relative_excess_percent": 100 * (ratio - 1),
        "theorem_factor": theorem_factor,
        "theorem_excess_percent": 100 * (theorem_factor - 1),
        "theorem_excess_fraction_used": (
            (ratio - 1) / (theorem_factor - 1) if theorem_factor > 1 else 0.0
        ),
        "exact_match": int(optimal["minimum_sensors"] == greedy["minimum_sensors"]),
        "allocation_match": int(optimal["allocation"] == greedy["allocation"]),
        "optimal_allocation": json.dumps(optimal["allocation"]),
        "greedy_allocation": json.dumps(greedy["allocation"]),
        "optimal_seconds": optimal_seconds,
        "greedy_seconds": greedy_seconds,
        "runtime_speedup": optimal_seconds / greedy_seconds,
    }


def allocation_match_table(rows):
    """Pool all families and M values; distinguish vector and total agreement."""
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["spatial_dimension"]), int(row["K"]))].append(row)
    table = []
    for (dimension, K), group in sorted(groups.items()):
        allocations = sum(int(row["allocation_match"]) for row in group)
        budgets = sum(int(row["exact_match"]) for row in group)
        if any(int(row["allocation_match"]) > int(row["exact_match"]) for row in group):
            raise ValueError("Identical allocations must have identical total budgets")
        table.append({
            "spatial_dimension": dimension,
            "dimension": "One-dimensional" if dimension == 1 else "Two-dimensional",
            "K": K, "case_count": len(group),
            "allocation_matches": allocations, "budget_matches": budgets,
            "allocation_match_percent": 100 * allocations / len(group),
            "budget_match_percent": 100 * budgets / len(group),
        })
    return table


def plot_allocation_match(table):
    """Consolidated Greedy/DP agreement, one panel per spatial dimension."""
    import matplotlib.pyplot as plt
    _configure_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), sharey=True)
    for axis, dimension, title in zip(axes, (1, 2),
            ("One-dimensional dataset", "Two-dimensional dataset")):
        selected = sorted([row for row in table if int(row["spatial_dimension"]) == dimension],
                          key=lambda row: int(row["K"]))
        ks = [int(row["K"]) for row in selected]
        for key, label, color, marker, linestyle in (
            ("allocation_match_percent", "Same allocation", "#0072B2", "o", "-"),
            ("budget_match_percent", "Same total budget", "#D55E00", "s", "--"),
        ):
            axis.plot(ks, [float(row[key]) for row in selected], label=label,
                      color=color, marker=marker, linestyle=linestyle, linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel(r"Number of candidate fields, $K$")
        axis.set_xticks(ks)
        axis.set_ylim(0, 105)
        axis.set_yticks(range(0, 101, 20))
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=10)
    axes[0].set_ylabel("Greedy–DP match rate (%)")
    figure.tight_layout()
    return figure


def summarize_group(
    rows: list[dict[str, object]],
    *,
    group_type: str,
    group_value: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Summarize one pre-specified group with uncertainty intervals."""
    ratios = [float(row["greedy_ratio"]) for row in rows]
    excesses = [float(row["relative_excess_percent"]) for row in rows]
    bound_fractions = [float(row["theorem_excess_fraction_used"]) for row in rows]
    exact_count = sum(int(row["exact_match"]) for row in rows)
    mean_low, mean_high = _stratified_bootstrap_mean_interval(
        rows,
        value_key="greedy_ratio",
        stratum_keys=("spatial_dimension", "field_model", "K", "M"),
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    exact_low, exact_high = _wilson_interval(exact_count, len(rows))
    return {
        "group_type": group_type,
        "group_value": group_value,
        "case_count": len(rows),
        "exact_match_rate": exact_count / len(rows),
        "exact_match_ci95_low": exact_low,
        "exact_match_ci95_high": exact_high,
        "mean_greedy_ratio": fmean(ratios),
        "mean_greedy_ratio_ci95_low": mean_low,
        "mean_greedy_ratio_ci95_high": mean_high,
        "median_greedy_ratio": median(ratios),
        "p95_greedy_ratio": _percentile(ratios, 0.95),
        "max_greedy_ratio": max(ratios),
        "mean_relative_excess_percent": fmean(excesses),
        "p95_relative_excess_percent": _percentile(excesses, 0.95),
        "max_relative_excess_percent": max(excesses),
        "mean_theorem_excess_fraction_used": fmean(bound_fractions),
        "p95_theorem_excess_fraction_used": _percentile(bound_fractions, 0.95),
        "max_theorem_excess_fraction_used": max(bound_fractions),
        "median_optimal_seconds": median(float(row["optimal_seconds"]) for row in rows),
        "median_greedy_seconds": median(float(row["greedy_seconds"]) for row in rows),
        "median_runtime_speedup": median(float(row["runtime_speedup"]) for row in rows),
    }


# Write a table only when it contains at least one row.
def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# Split result rows according to one field.
def _group_rows(
    rows: list[dict[str, object]], key: str
) -> dict[object, list[dict[str, object]]]:
    groups: dict[object, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return dict(groups)


# Save one figure in vector and high-resolution raster formats.
def _save_figure(figure: object, output_directory: Path, stem: str) -> None:
    figure.savefig(output_directory / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(output_directory / f"{stem}.png", dpi=300, bbox_inches="tight")


def _clear_figure_files(output_directory: Path) -> None:
    """Remove previously generated figures from the dedicated output folder."""
    for pattern in ("*.pdf", "*.png"):
        for path in output_directory.glob(pattern):
            path.unlink()


# Apply a consistent figure style before figures are created.
def _configure_plot_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 12,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    })


# Calculate mean greedy sensor excess for every K and M cell.
def _cell_mean_excess(rows: list[dict[str, object]]) -> dict[tuple[int, int], float]:
    cells: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        cells[(int(row["K"]), int(row["M"]))].append(
            float(row["relative_excess_percent"])
        )
    return {cell: fmean(values) for cell, values in cells.items()}


def heatmap_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return the values plotted in the K-by-M quality heatmaps."""
    table: list[dict[str, object]] = []
    for dimension, dimension_rows in sorted(_group_rows(rows, "spatial_dimension").items()):
        cell_means = _cell_mean_excess(dimension_rows)
        for K, M in sorted(cell_means):
            table.append(
                {
                    "dimension": f"{dimension}D",
                    "K": K,
                    "M": M,
                    "mean_additional_sensors_percent": cell_means[(K, M)],
                }
            )
    return table


def family_excess_table(
    rows: list[dict[str, object]],
    *,
    bootstrap_iterations: int = 2_000,
) -> list[dict[str, object]]:
    """Return the values plotted in the field-family excess bar chart."""
    table: list[dict[str, object]] = []
    for dimension, dimension_rows in sorted(_group_rows(rows, "spatial_dimension").items()):
        panel_title = (
            "(a) One-dimensional data set"
            if int(dimension) == 1
            else "(b) Two-dimensional data set"
        )
        by_model = _group_rows(dimension_rows, "field_model")
        for model in ("gaussian", "sinusoidal", "quadratic"):
            by_k = _group_rows(by_model[model], "K")
            for K, group in sorted(by_k.items()):
                mean_excess = fmean(
                    float(row["relative_excess_percent"]) for row in group
                )
                seed = 20261300 + zlib.crc32(
                    f"family-bars:{model}:{K}:{panel_title}".encode()
                )
                low, high = _stratified_bootstrap_mean_interval(
                    group,
                    value_key="relative_excess_percent",
                    stratum_keys=("M",),
                    iterations=bootstrap_iterations,
                    seed=seed,
                )
                table.append(
                    {
                        "dimension": f"{dimension}D",
                        "field_family": model,
                        "K": int(K),
                        "mean_additional_sensors_percent": mean_excess,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return table


def guarantee_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return the values plotted in the approximation-guarantee figure."""
    table: list[dict[str, object]] = []
    for dimension, dimension_rows in sorted(_group_rows(rows, "spatial_dimension").items()):
        by_model = _group_rows(dimension_rows, "field_model")
        for model in ("gaussian", "sinusoidal", "quadratic"):
            by_k = _group_rows(by_model[model], "K")
            for K, group in sorted(by_k.items()):
                table.append(
                    {
                        "dimension": f"{dimension}D",
                        "field_family": model,
                        "K": int(K),
                        "maximum_worst_case_allowance_used_percent": max(
                            100 * float(row["theorem_excess_fraction_used"])
                            for row in group
                        ),
                    }
                )
    return table


def _draw_quality_heatmap(
    axis: object,
    rows: list[dict[str, object]],
    color_limit: float,
    panel_title: str,
) -> object:
    """Draw mean greedy sensor excess over the complete K by M design."""
    k_values = sorted({int(row["K"]) for row in rows})
    m_values = sorted({int(row["M"]) for row in rows})
    cell_means = _cell_mean_excess(rows)
    matrix = [
        [cell_means[(K, M)] for M in m_values]
        for K in k_values
    ]

    image = axis.imshow(
        matrix,
        origin="lower",
        aspect="equal",
        cmap="Blues",
        vmin=0,
        vmax=color_limit,
    )
    axis.set_xticks(range(len(m_values)), m_values)
    axis.set_yticks(range(len(k_values)), k_values)
    axis.set_xlabel(r"Number of candidate sensing locations, $M$")
    axis.set_ylabel(r"Number of candidate fields, $K$")
    axis.set_title(panel_title, loc="center", pad=8)
    for row_index, K in enumerate(k_values):
        for column_index, M in enumerate(m_values):
            value = cell_means[(K, M)]
            axis.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=12,
                color="white" if value > 0.55 * color_limit else "#202020",
            )
    return image


def _draw_family_bars(
    axis: object,
    rows: list[dict[str, object]],
    panel_title: str,
    bootstrap_iterations: int,
) -> tuple[list[object], list[str], float]:
    """Draw mean greedy excess by K and field family with 95% intervals."""
    models = ("gaussian", "sinusoidal", "quadratic")
    labels = {
        "gaussian": "Gaussian",
        "sinusoidal": "Sinusoidal",
        "quadratic": "Quadratic",
    }
    by_model = _group_rows(rows, "field_model")
    k_values = sorted({int(row["K"]) for row in rows})
    base_positions = list(range(len(k_values)))
    bar_width = 0.24
    handles: list[object] = []
    maximum_upper = 0.0

    for model_index, model in enumerate(models):
        by_k = _group_rows(by_model[model], "K")
        means: list[float] = []
        lower_errors: list[float] = []
        upper_errors: list[float] = []
        for K in k_values:
            group = by_k[K]
            mean_excess = fmean(
                float(row["relative_excess_percent"]) for row in group
            )
            seed = 20261300 + zlib.crc32(
                f"family-bars:{model}:{K}:{panel_title}".encode()
            )
            low, high = _stratified_bootstrap_mean_interval(
                group,
                value_key="relative_excess_percent",
                stratum_keys=("M",),
                iterations=bootstrap_iterations,
                seed=seed,
            )
            means.append(mean_excess)
            lower_errors.append(max(0.0, mean_excess - low))
            upper_errors.append(max(0.0, high - mean_excess))
            maximum_upper = max(maximum_upper, high)

        positions = [
            position + (model_index - 1) * bar_width
            for position in base_positions
        ]
        bars = axis.bar(
            positions,
            means,
            width=bar_width,
            color=FIELD_FAMILY_COLORS[model],
            edgecolor="none",
            linewidth=0.0,
            label=labels[model],
            yerr=[lower_errors, upper_errors],
            capsize=2.2,
            error_kw={"ecolor": "#333333", "elinewidth": 0.8},
            alpha=0.88,
        )
        handles.append(bars[0])

    axis.set_xticks(base_positions, [str(K) for K in k_values])
    axis.set_xlabel(r"Number of candidate fields, $K$")
    axis.set_ylabel("Mean additional sensors used by Greedy (%)")
    axis.set_title(panel_title, loc="center", pad=8)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.25)
    return handles, [labels[model] for model in models], maximum_upper


def _draw_guarantee_comparison(
    axis: object,
    rows: list[dict[str, object]],
    panel_title: str,
) -> tuple[list[object], list[str]]:
    """Draw family-specific use of the theorem's worst-case excess allowance."""
    models = ("gaussian", "sinusoidal", "quadratic")
    labels = {
        "gaussian": "Gaussian: observed maximum",
        "sinusoidal": "Sinusoidal: observed maximum",
        "quadratic": "Quadratic: observed maximum",
    }
    markers = {
        "gaussian": "o",
        "sinusoidal": "s",
        "quadratic": "^",
    }
    by_model = _group_rows(rows, "field_model")
    k_values = sorted({int(row["K"]) for row in rows})

    for model in models:
        by_k = _group_rows(by_model[model], "K")
        maximum_allowance_used = [
            max(
                100 * float(row["theorem_excess_fraction_used"])
                for row in by_k[K]
            )
            for K in k_values
        ]
        axis.scatter(
            k_values,
            maximum_allowance_used,
            marker=markers[model],
            color=FIELD_FAMILY_COLORS[model],
            s=42,
            label=labels[model],
            zorder=3,
        )
    axis.axhline(
        100.0,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="Worst-case guarantee limit (100%)",
    )
    axis.set_xticks(k_values)
    axis.set_ylim(0, 105)
    axis.set_xlabel(r"Number of candidate fields, $K$")
    axis.set_ylabel("Worst-case sensor-excess allowance used (%)")
    axis.set_title(panel_title, loc="center", pad=8)
    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    return handles, labels


def _heatmap_figure(
    by_dimension: dict[object, list[dict[str, object]]],
    output_directory: Path,
    color_limit: float,
):
    """Create side-by-side quality heatmaps for both data sets."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    images = []
    for axis, dimension, title in (
        (axes[0], 1, "(a) One-dimensional data set"),
        (axes[1], 2, "(b) Two-dimensional data set"),
    ):
        images.append(
            _draw_quality_heatmap(
                axis,
                by_dimension[dimension],
                color_limit,
                title,
            )
        )
    colorbar = figure.colorbar(images[0], ax=axes, pad=0.02, fraction=0.035)
    colorbar.set_label("Mean additional sensors used by Greedy (%)")
    _save_figure(figure, output_directory, "benchmark_quality_heatmaps")
    return figure


def _family_figure(
    by_dimension: dict[object, list[dict[str, object]]],
    output_directory: Path,
    bootstrap_iterations: int,
):
    """Create grouped family-by-K comparisons for both dimensions."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    handles: list[object] = []
    labels: list[str] = []
    maximum_upper = 0.0
    for axis, dimension, title in (
        (axes[0], 1, "(a) One-dimensional data set"),
        (axes[1], 2, "(b) Two-dimensional data set"),
    ):
        handles, labels, panel_upper = _draw_family_bars(
            axis,
            by_dimension[dimension],
            title,
            bootstrap_iterations,
        )
        maximum_upper = max(maximum_upper, panel_upper)
    excess_limit = max(10.0, 5 * (math.ceil(maximum_upper / 5) + 1))
    axes[0].set_ylim(0, excess_limit)
    axes[1].set_ylabel("")
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    _save_figure(figure, output_directory, "benchmark_family_excess")
    return figure


def _guarantee_figure(
    by_dimension: dict[object, list[dict[str, object]]],
    output_directory: Path,
):
    """Create side-by-side empirical and theoretical guarantee comparisons."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    handles: list[object] = []
    labels: list[str] = []
    for axis, dimension, title in (
        (axes[0], 1, "(a) One-dimensional data set"),
        (axes[1], 2, "(b) Two-dimensional data set"),
    ):
        handles, labels = _draw_guarantee_comparison(
            axis,
            by_dimension[dimension],
            title,
        )
    axes[1].set_ylabel("")
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    _save_figure(figure, output_directory, "benchmark_guarantee_comparison")
    return figure


# Measure how summary statistics change as more replicates are included.
def _convergence_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for dimension, dimension_rows in sorted(_group_rows(rows, "spatial_dimension").items()):
        maximum_replicate = max(int(row["replicate"]) for row in dimension_rows)
        checkpoints = sorted(
            {5, 10, 15, 20, 25, maximum_replicate}
            & set(range(1, maximum_replicate + 1))
        )
        for checkpoint in checkpoints:
            sample = [
                row for row in dimension_rows if int(row["replicate"]) <= checkpoint
            ]
            ratios = [float(row["greedy_ratio"]) for row in sample]
            results.append({
                "spatial_dimension": dimension,
                "replicates_per_cell": checkpoint,
                "case_count": len(sample),
                "mean_greedy_ratio": fmean(ratios),
                "p95_greedy_ratio": _percentile(ratios, 0.95),
                "exact_match_rate": fmean(float(row["exact_match"]) for row in sample),
            })
    return results


# Build a short findings template from the current numerical summaries.
def _claim_text(
    overall_summary: dict[str, object],
    dimension_summaries: list[dict[str, object]],
) -> str:
    lines = [
        "# Empirical findings template",
        "",
        (
            f"Across the two pre-specified synthetic benchmarks "
            f"(n = {overall_summary['case_count']}), the greedy allocation matched "
            f"the exact optimum in "
            f"{100 * float(overall_summary['exact_match_rate']):.1f}% of cases."
        ),
        (
            f"Its mean greedy-to-optimal sensor ratio was "
            f"{float(overall_summary['mean_greedy_ratio']):.3f} "
            f"(95% stratified-bootstrap CI "
            f"{float(overall_summary['mean_greedy_ratio_ci95_low']):.3f}--"
            f"{float(overall_summary['mean_greedy_ratio_ci95_high']):.3f}); "
            f"the 95th percentile was "
            f"{float(overall_summary['p95_greedy_ratio']):.3f}."
        ),
    ]
    for summary in dimension_summaries:
        lines.extend([
            "",
            (
                f"For the {summary['group_value']} benchmark "
                f"(n = {summary['case_count']}), the mean ratio was "
                f"{float(summary['mean_greedy_ratio']):.3f} and the exact-match "
                f"rate was {100 * float(summary['exact_match_rate']):.1f}%."
            ),
        ])
    lines.extend([
        "",
        "These results describe the stated normalized synthetic benchmark distribution. "
        "They do not strengthen the worst-case guarantee or establish performance for "
        "every physical sensing application.",
        "",
    ])
    return "\n".join(lines)


def analyze_benchmarks(
    records_by_dataset: dict[str, list[dict[str, object]]],
    output_directory: str | Path,
    *,
    bootstrap_iterations: int = 2_000,
    run_mode: str = "full",
) -> dict[str, object]:
    """Write combined statistics and figures for both spatial dimensions."""
    validate_run_mode(run_mode)
    required = {"benchmark_1d", "benchmark_2d"}
    missing = required - set(records_by_dataset)
    if missing:
        raise ValueError("Missing benchmark results: " + ", ".join(sorted(missing)))

    output = Path(output_directory)
    tables_directory = output / "tables"
    figures_directory = output / "figures"
    tables_directory.mkdir(parents=True, exist_ok=True)
    figures_directory.mkdir(parents=True, exist_ok=True)
    _clear_figure_files(figures_directory)
    _configure_plot_style()
    rows = [
        case_metrics(record)
        for records in records_by_dataset.values()
        for record in records
    ]
    _write_csv(tables_directory / "case_metrics.csv", rows)
    match_table = allocation_match_table(rows)
    _write_csv(tables_directory / "allocation_match.csv", match_table)

    summary_rows: list[dict[str, object]] = []

    # Add one reproducible bootstrap summary for a selected group.
    def add_summary(group_type: str, group_value: str, group: list[dict[str, object]]):
        seed = 20261201 + zlib.crc32(f"{group_type}:{group_value}".encode())
        summary_rows.append(summarize_group(
            group,
            group_type=group_type,
            group_value=group_value,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=seed,
        ))

    add_summary("benchmark", "combined", rows)
    for key, label in (
        ("spatial_dimension", "dimension"),
        ("field_model", "field_model"),
        ("K", "K"),
        ("M", "M"),
    ):
        for value, group in sorted(_group_rows(rows, key).items(), key=lambda item: str(item[0])):
            display_value = f"{value}D" if key == "spatial_dimension" else str(value)
            add_summary(label, display_value, group)
    for dimension, dimension_rows in sorted(_group_rows(rows, "spatial_dimension").items()):
        for model, group in sorted(_group_rows(dimension_rows, "field_model").items()):
            add_summary("dimension_x_field_model", f"{dimension}D:{model}", group)
    _write_csv(tables_directory / "group_summary.csv", summary_rows)

    overall_summary = summary_rows[0]
    dimension_summaries = [
        row for row in summary_rows if row["group_type"] == "dimension"
    ]
    convergence = _convergence_rows(rows)
    _write_csv(tables_directory / "sample_convergence.csv", convergence)
    (tables_directory / "summary_findings.md").write_text(
        _claim_text(overall_summary, dimension_summaries), encoding="utf-8"
    )

    by_dimension = _group_rows(rows, "spatial_dimension")
    maximum_cell_mean = max(
        value
        for dimension_rows in by_dimension.values()
        for value in _cell_mean_excess(dimension_rows).values()
    )
    color_limit = max(5.0, 5 * math.ceil(maximum_cell_mean / 5))
    figures: dict[str, object] = {
        "heatmaps": _heatmap_figure(
            by_dimension,
            figures_directory,
            color_limit,
        ),
        "families": _family_figure(
            by_dimension,
            figures_directory,
            bootstrap_iterations,
        ),
        "guarantee": _guarantee_figure(
            by_dimension,
            figures_directory,
        ),
    }
    payload = {
        "run_mode": run_mode,
        "bootstrap_iterations": bootstrap_iterations,
        "overall_summary": overall_summary,
        "dimension_summaries": dimension_summaries,
        "sample_convergence": convergence,
    }
    (tables_directory / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    match_figure = plot_allocation_match(match_table)
    if run_mode == "test":
        match_figure.suptitle(f"Test run: {len(rows):,} cases")
        match_figure.tight_layout(rect=(0, 0, 1, 0.93))
    _save_figure(match_figure, figures_directory, "allocation_match")
    figures["allocation_match"] = match_figure
    return {
        "rows": rows,
        "summaries": summary_rows,
        "figures": figures,
        "allocation_match_table": match_table,
        "output_directory": str(output),
        **payload,
    }
