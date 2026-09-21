"""Independent sensor-budget analyses from raw benchmark phase CSVs.

No notebook outputs or cached solution-method results are read here.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from time import perf_counter

import numpy as np

from eval import _case_entries, _load_manifest, _validate_solution
from optimal import optimal_perfect_pairwise_sensor_placement
from phase_data import compute_nijm, load_phase_csv
from run_config import output_directory, select_cases, validate_run_mode


DATASETS = ("benchmark_1d", "benchmark_2d")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_sensor_budgets(project_directory, *, include_lower_bound=False,
                            progress_every=500, run_mode="full"):
    """Recompute exact budgets for the selected mode, preserving paired inputs."""
    validate_run_mode(run_mode)
    project_directory = Path(project_directory)
    if include_lower_bound:
        from lower_bound import optimal_global_lower_bound

    # Preflight both datasets before any expensive computation.
    datasets = []
    for dataset in DATASETS:
        directory = project_directory / "inputs" / dataset
        manifest = _load_manifest(directory)
        entries = select_cases(_case_entries(directory, manifest), run_mode)
        datasets.append((dataset, entries))

    print(f"Run mode: {run_mode}; selected {sum(len(entries) for _, entries in datasets):,} cases",
          flush=True)

    rows = []
    start = perf_counter()
    for dataset, entries in datasets:
        for index, (path, metadata) in enumerate(entries, 1):
            K, M, phases = load_phase_csv(path)
            if K != metadata["K"] or M != metadata["M"]:
                raise ValueError(f"Manifest dimensions disagree with {path}")
            thresholds = compute_nijm(K, M, phases)
            begin = perf_counter()
            result = optimal_perfect_pairwise_sensor_placement(K, M, thresholds)
            elapsed = perf_counter() - begin
            _validate_solution("Exact DP", result, K, M, thresholds)
            nmin, allocation, _ = result
            row = {
                "run_mode": run_mode,
                "dataset": dataset,
                "spatial_dimension": int(metadata["spatial_dimension"]),
                "case": path.stem, "field_model": metadata["field_model"],
                "replicate": metadata["replicate"],
                "generation_seed": metadata["generation_seed"],
                "phase_scale_c": metadata["phase_scale_c"], "K": K, "M": M,
                "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "N_min": nmin, "nmin_allocation": json.dumps(allocation),
                "nmin_seconds": elapsed,
            }
            if include_lower_bound:
                begin = perf_counter()
                nlb, lb_allocation, diagnostics = optimal_global_lower_bound(K, M, phases)
                if nlb > nmin:
                    raise RuntimeError(f"N_LB > N_min for {path}")
                row.update({
                    "N_LB": nlb, "lb_allocation": json.dumps(lb_allocation),
                    "lb_seconds": perf_counter() - begin,
                    "sensor_gap": nmin - nlb,
                    "lb_to_nmin_ratio": nlb / nmin,
                    "equal_budgets": int(nlb == nmin), **diagnostics,
                })
            rows.append(row)
            if index == 1 or index % progress_every == 0 or index == len(entries):
                print(f"{dataset}: {index:,}/{len(entries):,} cases; "
                      f"{perf_counter() - start:.1f}s elapsed", flush=True)
    return rows


def summarize_budgets(rows, *, group_keys=("spatial_dimension", "K")):
    """Unweighted case means; full benchmarks have balanced M/family strata."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)
    summary = []
    for keys, group in sorted(groups.items()):
        budgets = [row["N_min"] for row in group]
        row = dict(zip(group_keys, keys))
        row.update({
            "dimension": f"{group[0]['spatial_dimension']}D",
            "case_count": len(group), "mean_N_min_opt": fmean(budgets),
            "median_N_min_opt": float(np.median(budgets)),
            "p95_N_min_opt": float(np.percentile(budgets, 95)),
            "maximum_N_min_opt": max(budgets),
        })
        if "N_LB" in group[0]:
            row.update({
                "mean_N_LB": fmean(r["N_LB"] for r in group),
                "mean_sensor_gap": fmean(r["sensor_gap"] for r in group),
                "mean_lb_to_nmin_ratio": fmean(r["lb_to_nmin_ratio"] for r in group),
                "equal_budget_fraction": fmean(r["equal_budgets"] for r in group),
            })
        summary.append(row)
    return summary


def configure_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12, "mathtext.fontset": "stix", "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    })


def plot_budget_comparison(summary):
    """Plot consolidated budgets for one- and two-dimensional datasets."""
    import matplotlib.pyplot as plt
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), sharey=True)
    titles = {1: "One-dimensional dataset", 2: "Two-dimensional dataset"}
    for axis, dimension in zip(axes, (1, 2)):
        selected = sorted([
            row for row in summary if int(row["spatial_dimension"]) == dimension
        ], key=lambda row: int(row["K"]))
        ks = [int(row["K"]) for row in selected]
        for key, label, color, marker, line in (
            ("mean_N_min_opt", r"$N_{\min}^{\mathrm{seq}}$", "#0072B2", "o", "-"),
            ("mean_N_LB", r"$N_{\mathrm{LB}}^{\mathrm{global}}$", "#D55E00", "s", "--"),
        ):
            axis.plot(ks, [float(row[key]) for row in selected], label=label,
                      color=color, marker=marker, linestyle=line, linewidth=1.8)
        axis.set_title(titles[dimension])
        axis.set_xlabel(r"Number of candidate fields, $K$")
        axis.set_xticks(ks)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=10)
    axes[0].set_ylabel("Mean sensor budget")
    # Include both dimensions before fixing the shared y-axis limits.
    maximum = max(float(row[key]) for row in summary
                  for key in ("mean_N_min_opt", "mean_N_LB"))
    axes[0].set_ylim(0, max(1, maximum * 1.08))
    figure.tight_layout()
    return figure


def save_figure(figure, directory, stem):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")


def run_budget_analysis(project_directory, *, include_lower_bound=False, run_mode="full"):
    """Compute from inputs and save only this analysis's own output tables."""
    name = "lb" if include_lower_bound else "nmin"
    output = output_directory(project_directory, name, run_mode)
    rows = evaluate_sensor_budgets(project_directory, include_lower_bound=include_lower_bound,
                                  run_mode=run_mode,
                                  progress_every=10 if run_mode == "test" else 500)
    summary = summarize_budgets(rows)
    write_csv(output / "tables" / "case_budgets.csv", rows)
    stem = "nlb_vs_nmin_by_dimension_k" if include_lower_bound else "nmin_by_dimension_k"
    write_csv(output / "tables" / f"{stem}.csv", summary)
    result = {"rows": rows, "summary": summary, "output_directory": output,
              "run_mode": run_mode}
    if include_lower_bound:
        cell_summary = summarize_budgets(rows, group_keys=("spatial_dimension", "K", "M"))
        write_csv(output / "tables" / "nlb_vs_nmin_by_k_m.csv", cell_summary)
    return result
