"""Error-probability, runtime, and exact-gap figures from saved summaries."""

from pathlib import Path
import csv
import matplotlib.pyplot as plt


STYLES = {"Exhaustive": ("#1f77b4", "o", "-"), "Greedy": ("#d62728", "s", "--"),
          "Heuristic": ("#2ca02c", "^", "-."), "Uniform": ("#777777", "D", ":"),
          "Random": ("#777777", "x", "--")}
TITLES = {1: "(a) One-dimensional dataset", 2: "(b) Two-dimensional dataset"}


def plot_summary(summary, metric="mean_error_probability"):
    if not summary:
        raise ValueError("No matched completed cases to plot; inspect tables/status.csv")
    labels = {"mean_error_probability": "Mean error probability",
              "mean_seconds": "Mean runtime (s)", "mean_error_gap": "Mean error above exhaustive minimum"}
    if metric not in labels:
        raise ValueError(f"Unknown plot metric: {metric}")
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=True, layout="constrained")
    for dimension, axis in zip((1, 2), axes):
        for method, (color, marker, line) in STYLES.items():
            selected = sorted([r for r in summary if int(r["dimension"]) == dimension
                               and r["method"] == method and r[metric] != ""],
                              key=lambda r: int(r["budget_level"]))
            if not selected:
                continue
            axis.plot([float(r["mean_budget_fraction"]) for r in selected],
                      [float(r[metric]) for r in selected], label=method,
                      color=color, marker=marker, linestyle=line)
        axis.set_title(TITLES[dimension])
        axis.set_xlabel(r"Mean budget fraction, $N/N_{\min}^{\mathrm{seq}}$")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel(labels[metric])
    axes[0].set_ylim(bottom=0)
    return fig


def save_figures(summary, output, *, include_exact=False):
    directory = Path(output) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    metrics = ["mean_error_probability", "mean_seconds"]
    if include_exact:
        metrics.append("mean_error_gap")
    figures = {}
    for metric in metrics:
        fig = plot_summary(summary, metric)
        stem = metric.removeprefix("mean_")
        for extension in ("pdf", "png"):
            fig.savefig(directory / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
        figures[stem] = fig
    return figures


def load_summary(project, study, mode="full"):
    path = Path(project) / "results" / study / mode / "tables/error_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Run {study} in {mode} mode first: {path}")
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def summary_markdown(rows):
    lines = ["| Dimension | Budget | Method | Cases | Mean Error Probability | Mean Time (s) | Mean Gap |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        dimension = "One-dimensional" if int(row["dimension"]) == 1 else "Two-dimensional"
        gap = "—" if row["mean_error_gap"] == "" else f"{float(row['mean_error_gap']):.5f}"
        lines.append(f"| {dimension} | {row['budget_level']} | {row['method']} | {row['case_count']} | "
                     f"{float(row['mean_error_probability']):.5f} | {float(row['mean_seconds']):.4f} | {gap} |")
    return "\n".join(lines)
