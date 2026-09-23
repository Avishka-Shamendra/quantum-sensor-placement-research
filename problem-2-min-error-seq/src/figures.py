"""Mean-error curves and case-level runtime and field-family figures."""

from collections import defaultdict
import csv
from pathlib import Path

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


def _matched_rows(rows):
    methods = {row["method"] for row in rows}
    groups = defaultdict(dict)
    for row in rows:
        groups[(row["dataset"], row["case"], int(row["N"]))][row["method"]] = row
    return [
        row
        for group in groups.values()
        if set(group) == methods
        and all(row["status"] == "ok" for row in group.values())
        for row in group.values()
    ]


def _error(row):
    return float(row.get("actual_error", row.get("error_probability")))


def _grouped_boxplot(axis, rows, categories, category_key, value_key):
    methods = [method for method in STYLES if any(r["method"] == method for r in rows)]
    positions = list(range(1, len(categories) + 1))
    width = min(0.16, 0.72 / max(1, len(methods)))
    offsets = [(index - (len(methods) - 1) / 2) * width for index in range(len(methods))]
    for method, offset in zip(methods, offsets):
        values = [
            [value_key(row) for row in rows
             if row["method"] == method and category_key(row) == category]
            for category in categories
        ]
        boxes = axis.boxplot(
            values,
            positions=[position + offset for position in positions],
            widths=width * 0.9,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            meanprops={"marker": "o", "markerfacecolor": "black",
                       "markeredgecolor": "black", "markersize": 3},
            medianprops={"color": "black"},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(STYLES[method][0])
            box.set_alpha(0.5)
        boxes["boxes"][0].set_label(method)
    axis.set_xticks(positions, categories)
    axis.grid(axis="y", alpha=0.22)


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


def plot_runtime_boxplot(rows):
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
        _grouped_boxplot(
            axis, selected, K_values,
            lambda row: int(row["K"]),
            lambda row: float(row["seconds"]),
        )
        axis.set_title(TITLES[dimension])
        axis.set_xlabel("Number of fields, K")
        axis.set_yscale("log")
    axes[0].set_ylabel("Runtime (s, log scale)")
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
    for dimension, axis in zip((1, 2), axes):
        selected = [row for row in matched if int(row["dimension"]) == dimension]
        present = {row["field_family"] for row in selected}
        families = [family for family in FAMILY_ORDER if family in present]
        _grouped_boxplot(
            axis, selected, families,
            lambda row: row["field_family"],
            _error,
        )
        axis.set_title(TITLES[dimension])
        axis.set_xlabel("Field family")
        axis.set_ylim(bottom=0)
    axes[0].set_ylabel("Error probability")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)
    return figure


def save_figures(summary, rows, output):
    directory = Path(output) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    figures = {
        "error_probability": plot_mean_error(summary),
        "runtime_boxplot": plot_runtime_boxplot(rows),
        "error_by_field_family": plot_error_by_family(rows),
    }
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
