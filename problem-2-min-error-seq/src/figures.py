"""Mean-error curves and case-level comparison bars with 95% intervals."""

from collections import defaultdict
import csv
import json
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev

import matplotlib.pyplot as plt


STYLES = {
    "Exhaustive": ("#1f77b4", "o", "-"),
    "Greedy": ("#d62728", "s", "--"),
    "Heuristic": ("#2ca02c", "^", "-."),
    "Uniform": ("#9467bd", "D", ":"),
    "Random": ("#8c564b", "x", "--"),
}
TITLES = {1: "(a) One-dimensional dataset", 2: "(b) Two-dimensional dataset"}
FAMILY_ORDER = ("gaussian", "sinusoidal", "quadratic")


def _plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _matched_groups(rows):
    methods = {row["method"] for row in rows}
    groups = defaultdict(dict)
    for row in rows:
        groups[(row["dataset"], row["case"], int(row["N"]))][row["method"]] = row
    return [
        group
        for group in groups.values()
        if set(group) == methods
        and all(row["status"] == "ok" for row in group.values())
    ]


def _matched_rows(rows):
    return [row for group in _matched_groups(rows) for row in group.values()]


def _error(row):
    return float(row.get("actual_error", row.get("error_probability")))


def _allocation(row):
    value = row["allocation"]
    if isinstance(value, str):
        value = json.loads(value)
    return tuple(int(count) for count in value)


def _mean_ci95(values):
    mean = fmean(values)
    half_width = 0.0 if len(values) < 2 else 1.96 * stdev(values) / sqrt(len(values))
    return mean, half_width


def allocation_match_table(rows):
    """Summarize exact allocation-vector agreement on matched completed runs."""
    groups = _matched_groups(rows)
    if not groups:
        return []
    present_methods = {method for group in groups for method in group}
    comparisons = [("Greedy–Heuristic", "Greedy", "Heuristic")]
    if "Exhaustive" in present_methods:
        comparisons.append(("Greedy–Exhaustive", "Greedy", "Exhaustive"))
    table = []
    for dimension in (1, 2):
        selected = [
            group for group in groups
            if int(group["Greedy"]["dimension"]) == dimension
        ]
        K_values = sorted({int(group["Greedy"]["K"]) for group in selected})
        for K in K_values:
            K_groups = [
                group for group in selected if int(group["Greedy"]["K"]) == K
            ]
            for label, first_method, second_method in comparisons:
                values = [
                    float(
                        _allocation(group[first_method])
                        == _allocation(group[second_method])
                    )
                    for group in K_groups
                ]
                mean, half_width = _mean_ci95(values)
                table.append({
                    "dimension": dimension,
                    "K": K,
                    "comparison": label,
                    "matches": int(sum(values)),
                    "runs": len(values),
                    "match_percent": 100.0 * mean,
                    "ci95_low": 100.0 * max(0.0, mean - half_width),
                    "ci95_high": 100.0 * min(1.0, mean + half_width),
                })
    return table


def allocation_match_markdown(rows):
    lines = [
        "| Dimension | K | Comparison | Matches | Runs | Match (%) | 95% CI |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        dimension = "One-dimensional" if int(row["dimension"]) == 1 else "Two-dimensional"
        lines.append(
            f"| {dimension} | {row['K']} | {row['comparison']} | "
            f"{row['matches']} | {row['runs']} | {row['match_percent']:.1f} | "
            f"[{row['ci95_low']:.1f}, {row['ci95_high']:.1f}] |"
        )
    return "\n".join(lines)


def _grouped_bars(axis, categories, values_by_method, *, scale=1.0, bounds=None):
    methods = [method for method in STYLES if method in values_by_method]
    positions = list(range(1, len(categories) + 1))
    width = min(0.16, 0.72 / max(1, len(methods)))
    offsets = [(index - (len(methods) - 1) / 2) * width for index in range(len(methods))]
    maximum = 0.0
    for method, offset in zip(methods, offsets):
        means, lower, upper = [], [], []
        for values in values_by_method[method]:
            mean, half_width = _mean_ci95(values)
            low, high = mean - half_width, mean + half_width
            if bounds is not None:
                low = max(bounds[0], low)
                high = min(bounds[1], high)
            means.append(mean * scale)
            lower.append((mean - low) * scale)
            upper.append((high - mean) * scale)
            maximum = max(maximum, high * scale)
        axis.bar(
            [position + offset for position in positions],
            means,
            width=width * 0.9,
            yerr=[lower, upper],
            capsize=3,
            color=STYLES[method][0],
            alpha=0.8,
            label=method,
            error_kw={"elinewidth": 1, "ecolor": "black"},
        )
    axis.set_xticks(positions, categories)
    axis.grid(axis="y", alpha=0.22)
    return maximum


def plot_mean_error(summary):
    if not summary:
        raise ValueError("No matched completed cases to plot; inspect tables/status.csv")
    _plot_style()
    figure, axes = plt.subplots(
        1, 2, figsize=(10.8, 3.8), sharey=True, layout="constrained"
    )
    for dimension, axis in zip((1, 2), axes):
        for method, (color, marker, line) in STYLES.items():
            selected = sorted(
                [row for row in summary
                 if int(row["dimension"]) == dimension and row["method"] == method],
                key=lambda row: int(row["budget_level"]),
            )
            if selected:
                axis.plot(
                    [float(row["mean_budget_fraction"]) for row in selected],
                    [float(row["mean_error_probability"]) for row in selected],
                    label=method,
                    color=color,
                    marker=marker,
                    linestyle=line,
                )
        axis.set_title(TITLES[dimension])
        axis.set_xlabel(r"Mean budget fraction, $N/N_{\min}^{\mathrm{seq}}$")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Mean error probability")
    axes[0].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="outside upper center", ncol=len(labels), frameon=False
    )
    return figure


def plot_allocation_match(rows):
    groups = _matched_groups(rows)
    if not groups:
        raise ValueError("No matched completed cases available for allocation plot")
    present_methods = {method for group in groups for method in group}
    if "Exhaustive" in present_methods:
        comparisons = (
            ("Greedy–Heuristic", "Greedy", "Heuristic", "#1f77b4", "o", "-"),
            ("Greedy–Exhaustive", "Greedy", "Exhaustive", "#d62728", "s", "--"),
        )
    else:
        comparisons = (
            ("Greedy–Heuristic", "Greedy", "Heuristic", "#1f77b4", "o", "-"),
        )
    _plot_style()
    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.2), sharey=True, layout="constrained"
    )
    for dimension, axis in zip((1, 2), axes):
        selected = [group for group in groups
                    if int(group["Greedy"]["dimension"]) == dimension]
        K_values = sorted({int(group["Greedy"]["K"]) for group in selected})
        for label, first_method, second_method, color, marker, line in comparisons:
            means, lower, upper = [], [], []
            for K in K_values:
                values = [
                    float(
                        _allocation(group[first_method])
                        == _allocation(group[second_method])
                    )
                    for group in selected
                    if int(group[first_method]["K"]) == K
                ]
                mean, half_width = _mean_ci95(values)
                low = max(0.0, mean - half_width)
                high = min(1.0, mean + half_width)
                means.append(100.0 * mean)
                lower.append(100.0 * (mean - low))
                upper.append(100.0 * (high - mean))
            axis.errorbar(
                K_values,
                means,
                yerr=[lower, upper],
                color=color,
                marker=marker,
                linestyle=line,
                linewidth=2,
                capsize=3,
                label=label,
            )
        axis.set_title(TITLES[dimension])
        axis.set_xlabel("Number of fields, K")
        axis.set_ylim(0, 100)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Exact allocation match (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    return figure


def plot_runtime_by_k(rows):
    matched = _matched_rows(rows)
    if not matched:
        raise ValueError("No matched completed cases available for runtime plot")
    _plot_style()
    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.2), sharey=True, layout="constrained"
    )
    for dimension, axis in zip((1, 2), axes):
        selected = [row for row in matched if int(row["dimension"]) == dimension]
        K_values = sorted({int(row["K"]) for row in selected})
        for method, (color, marker, line) in STYLES.items():
            if not any(row["method"] == method for row in selected):
                continue
            means, lower, upper = [], [], []
            for K in K_values:
                values = [
                    float(row["seconds"])
                    for row in selected
                    if row["method"] == method and int(row["K"]) == K
                ]
                mean, half_width = _mean_ci95(values)
                low = max(0.0, mean - half_width)
                means.append(mean)
                lower.append(mean - low)
                upper.append(half_width)
            axis.errorbar(
                K_values,
                means,
                yerr=[lower, upper],
                color=color,
                marker=marker,
                linestyle=line,
                linewidth=1.8,
                capsize=3,
                label=method,
            )
        axis.set_title(TITLES[dimension])
        axis.set_xlabel("Number of fields, K")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Mean runtime (s), 95% CI")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)
    return figure


def plot_error_by_family(rows):
    matched = _matched_rows(rows)
    if not matched:
        raise ValueError("No matched completed cases available for family plot")
    _plot_style()
    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.2), sharey=True, layout="constrained"
    )
    panel_maxima = []
    for dimension, axis in zip((1, 2), axes):
        selected = [row for row in matched if int(row["dimension"]) == dimension]
        present = {row["field_family"] for row in selected}
        families = [family for family in FAMILY_ORDER if family in present]
        values_by_method = {
            method: [
                [_error(row) for row in selected
                 if row["method"] == method and row["field_family"] == family]
                for family in families
            ]
            for method in STYLES
            if any(row["method"] == method for row in selected)
        }
        panel_maxima.append(
            _grouped_bars(axis, families, values_by_method, bounds=(0.0, 1.0))
        )
        axis.set_title(TITLES[dimension])
        axis.set_xlabel("Field family")
    maximum = max(panel_maxima)
    axes[0].set_ylim(0, min(1.0, maximum + 0.06 * max(maximum, 0.1)))
    axes[0].set_ylabel("Mean error probability, 95% CI")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)
    return figure


def save_figures(summary, rows, output):
    directory = Path(output) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    for legacy_name in ("runtime_boxplot", "seconds"):
        for extension in ("pdf", "png"):
            (directory / f"{legacy_name}.{extension}").unlink(missing_ok=True)
    methods = {row["method"] for row in rows}
    figures = {"error_probability": plot_mean_error(summary)}
    if "Exhaustive" not in methods:
        figures["allocation_match"] = plot_allocation_match(rows)
        figures["runtime_by_k"] = plot_runtime_by_k(rows)
    else:
        for obsolete_name in ("allocation_match", "runtime_by_k"):
            for extension in ("pdf", "png"):
                (directory / f"{obsolete_name}.{extension}").unlink(missing_ok=True)
    figures.update({
        "error_by_field_family": plot_error_by_family(rows),
    })
    for name, figure in figures.items():
        for extension in ("pdf", "png"):
            figure.savefig(directory / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    return figures


def _load_csv(path, missing_message):
    if not path.is_file():
        raise FileNotFoundError(f"{missing_message}: {path}")
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_summary(project, study, mode="full"):
    path = Path(project) / "results" / study / mode / "tables/error_summary.csv"
    return _load_csv(path, f"Run {study} in {mode} mode first")


def load_case_rows(project, study, mode="full"):
    path = Path(project) / "results" / study / mode / "tables/case_errors.csv"
    return _load_csv(path, f"Run {study} in {mode} mode first")


def summary_markdown(rows):
    lines = [
        "| Dimension | Budget | Method | Cases | Mean Error Probability | Mean Time (s) | Mean Gap |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        dimension = "One-dimensional" if int(row["dimension"]) == 1 else "Two-dimensional"
        gap = "—" if row["mean_error_gap"] == "" else f"{float(row['mean_error_gap']):.5f}"
        lines.append(
            f"| {dimension} | {row['budget_level']} | {row['method']} | {row['case_count']} | "
            f"{float(row['mean_error_probability']):.5f} | {float(row['mean_seconds']):.4f} | {gap} |"
        )
    return "\n".join(lines)
