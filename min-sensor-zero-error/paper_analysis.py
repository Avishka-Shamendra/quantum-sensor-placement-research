"""Statistical summaries and publication figures for the paper studies."""

from __future__ import annotations

import csv
import json
import math
import random
import zlib
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median


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


def _stratified_bootstrap_mean_interval(
    rows: list[dict[str, object]],
    *,
    value_key: str,
    stratum_keys: tuple[str, ...],
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap within pre-specified factorial cells, preserving cell weights."""
    strata: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in stratum_keys)
        strata[key].append(float(row[value_key]))
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sample: list[float] = []
        for values in strata.values():
            sample.extend(rng.choices(values, k=len(values)))
        estimates.append(fmean(sample))
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


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
    """Flatten one solver record and derive comparison metrics."""
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
        "case": record["case"],
        "field_model": record["field_model"],
        "replicate": record.get("replicate"),
        "generation_seed": record.get("generation_seed"),
        "phase_scale_c": record.get("phase_scale_c"),
        "K": record["K"],
        "M": record["M"],
        "N_optimal": optimal["minimum_sensors"],
        "N_greedy": greedy["minimum_sensors"],
        "greedy_ratio": ratio,
        "relative_excess_percent": 100 * (ratio - 1),
        "theorem_factor": theorem_factor,
        "theorem_excess_percent": 100 * (theorem_factor - 1),
        "theorem_excess_fraction_used": (ratio - 1) / (theorem_factor - 1),
        "exact_match": int(optimal["minimum_sensors"] == greedy["minimum_sensors"]),
        "optimal_seconds": optimal_seconds,
        "greedy_seconds": greedy_seconds,
        "runtime_speedup": optimal_seconds / greedy_seconds,
    }


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
        stratum_keys=("field_model", "K", "M", "phase_scale_c"),
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
        "median_optimal_seconds": median(
            float(row["optimal_seconds"]) for row in rows
        ),
        "median_greedy_seconds": median(
            float(row["greedy_seconds"]) for row in rows
        ),
        "median_runtime_speedup": median(
            float(row["runtime_speedup"]) for row in rows
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _group_rows(
    rows: list[dict[str, object]], key: str
) -> dict[object, list[dict[str, object]]]:
    groups: dict[object, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return dict(groups)


def _save_figure(figure: object, output_directory: Path, stem: str) -> None:
    figure.savefig(output_directory / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(output_directory / f"{stem}.png", dpi=300, bbox_inches="tight")


def _configure_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    })


def _quality_figure(rows: list[dict[str, object]], output_directory: Path):
    import matplotlib.pyplot as plt

    by_k = _group_rows(rows, "K")
    k_values = sorted(int(value) for value in by_k)
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    excess_samples = [
        [float(row["relative_excess_percent"]) for row in by_k[K]]
        for K in k_values
    ]
    boxplot = axes[0].boxplot(
        excess_samples,
        positions=k_values,
        widths=0.55,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for box in boxplot["boxes"]:
        box.set_facecolor("#4C78A8")
        box.set_alpha(0.75)
    axes[0].set_xlabel("Number of candidate fields, K")
    axes[0].set_ylabel("Additional sensors used by greedy (%)")
    axes[0].set_title("(a) Solution-quality distribution", loc="left")
    axes[0].grid(axis="y", alpha=0.25)

    rates = []
    lower_errors = []
    upper_errors = []
    for K in k_values:
        group = by_k[K]
        successes = sum(int(row["exact_match"]) for row in group)
        rate = successes / len(group)
        low, high = _wilson_interval(successes, len(group))
        rates.append(rate)
        lower_errors.append(rate - low)
        upper_errors.append(high - rate)
    axes[1].errorbar(
        k_values,
        rates,
        yerr=[lower_errors, upper_errors],
        marker="o",
        capsize=3,
        color="#59A14F",
        linewidth=1.8,
    )
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Number of candidate fields, K")
    axes[1].set_ylabel("Fraction matching the exact optimum")
    axes[1].set_title("(b) Exact-optimum recovery", loc="left")
    axes[1].grid(alpha=0.25)

    p95_bound_fractions = [
        100 * _percentile(
            [float(row["theorem_excess_fraction_used"]) for row in by_k[K]],
            0.95,
        )
        for K in k_values
    ]
    maximum_bound_fractions = [
        100 * max(
            float(row["theorem_excess_fraction_used"]) for row in by_k[K]
        )
        for K in k_values
    ]
    axes[2].plot(
        k_values,
        p95_bound_fractions,
        marker="o",
        color="#4C78A8",
        label="Observed 95th percentile",
    )
    axes[2].plot(
        k_values,
        maximum_bound_fractions,
        marker="s",
        color="#E15759",
        label="Observed maximum",
    )
    axes[2].axhline(
        100,
        linestyle="--",
        color="black",
        linewidth=1.2,
        label="Worst-case limit",
    )
    axes[2].set_xlabel("Number of candidate fields, K")
    axes[2].set_ylabel("Fraction of guaranteed excess used (%)")
    axes[2].set_title("(c) Gap to worst-case guarantee", loc="left")
    axes[2].set_ylim(bottom=0)
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

    figure.tight_layout()
    _save_figure(figure, output_directory, "primary_quality")
    return figure


def _ecdf_figure(rows: list[dict[str, object]], output_directory: Path):
    import matplotlib.pyplot as plt

    by_model = _group_rows(rows, "field_model")
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    colors = ["#4C78A8", "#F28E2B", "#59A14F", "#B07AA1"]
    for color, model in zip(colors, sorted(str(value) for value in by_model)):
        values = sorted(float(row["greedy_ratio"]) for row in by_model[model])
        probabilities = [(index + 1) / len(values) for index in range(len(values))]
        axis.step(values, probabilities, where="post", label=model.title(), color=color)
    axis.axvline(1.0, linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("Greedy / exact sensor count")
    axis.set_ylabel("Fraction of cases at or below ratio")
    axis.set_title("Empirical distribution of greedy solution quality", loc="left")
    axis.grid(alpha=0.25)
    axis.legend(title="Field model", frameon=False)
    figure.tight_layout()
    _save_figure(figure, output_directory, "primary_quality_ecdf")
    return figure


def _runtime_figure(rows: list[dict[str, object]], output_directory: Path):
    import matplotlib.pyplot as plt

    by_m = _group_rows(rows, "M")
    m_values = sorted(int(value) for value in by_m)
    figure, axes = plt.subplots(1, len(m_values), figsize=(4.7 * len(m_values), 4.2), sharey=True)
    if len(m_values) == 1:
        axes = [axes]
    for axis, M in zip(axes, m_values):
        by_k = _group_rows(by_m[M], "K")
        k_values = sorted(int(value) for value in by_k)
        for key, label, color in (
            ("optimal_seconds", "Exact dynamic programming", "#4C78A8"),
            ("greedy_seconds", "Greedy approximation", "#F28E2B"),
        ):
            medians = [median(float(row[key]) for row in by_k[K]) for K in k_values]
            lowers = [
                _percentile([float(row[key]) for row in by_k[K]], 0.25)
                for K in k_values
            ]
            uppers = [
                _percentile([float(row[key]) for row in by_k[K]], 0.75)
                for K in k_values
            ]
            axis.plot(k_values, medians, marker="o", label=label, color=color)
            axis.fill_between(k_values, lowers, uppers, color=color, alpha=0.18)
        axis.set_yscale("log")
        axis.set_xlabel("Number of candidate fields, K")
        axis.set_title(f"M = {M}")
        axis.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("Runtime per solve (seconds, log scale)")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    _save_figure(figure, output_directory, "primary_runtime_scaling")
    return figure


def _paired_sensitivity_rows(
    rows: list[dict[str, object]], *, bootstrap_iterations: int
) -> list[dict[str, object]]:
    by_k = _group_rows(rows, "K")
    results: list[dict[str, object]] = []
    for K in sorted(int(value) for value in by_k):
        by_c = _group_rows(by_k[K], "phase_scale_c")
        c_values = sorted(float(value) for value in by_c)
        reference = {
            (
                row["field_model"],
                row["M"],
                row["replicate"],
                row["generation_seed"],
            ): float(row["relative_excess_percent"])
            for row in by_c[1.0]
        }
        for c in c_values:
            difference_rows: list[dict[str, object]] = []
            for row in by_c[c]:
                pair_key = (
                    row["field_model"],
                    row["M"],
                    row["replicate"],
                    row["generation_seed"],
                )
                difference_rows.append({
                    "field_model": row["field_model"],
                    "M": row["M"],
                    "difference": (
                        float(row["relative_excess_percent"])
                        - reference[pair_key]
                    ),
                })
            differences = [float(row["difference"]) for row in difference_rows]
            mean = fmean(differences)
            low, high = _stratified_bootstrap_mean_interval(
                difference_rows,
                value_key="difference",
                stratum_keys=("field_model", "M"),
                iterations=bootstrap_iterations,
                seed=20261301 + K * 101 + int(c * 100),
            )
            results.append({
                "K": K,
                "phase_scale_c": c,
                "paired_case_count": len(differences),
                "mean_change_in_greedy_excess_percentage_points": mean,
                "ci95_low": low,
                "ci95_high": high,
            })
    return results


def _sensitivity_figure(
    paired_rows: list[dict[str, object]], output_directory: Path
):
    import matplotlib.pyplot as plt

    by_k = _group_rows(paired_rows, "K")
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for K in sorted(int(value) for value in by_k):
        values = sorted(by_k[K], key=lambda row: float(row["phase_scale_c"]))
        c_values = [float(row["phase_scale_c"]) for row in values]
        means = [
            float(row["mean_change_in_greedy_excess_percentage_points"])
            for row in values
        ]
        lower_errors = [
            mean - float(row["ci95_low"])
            for mean, row in zip(means, values)
        ]
        upper_errors = [
            float(row["ci95_high"]) - mean
            for mean, row in zip(means, values)
        ]
        axis.errorbar(
            c_values,
            means,
            yerr=[lower_errors, upper_errors],
            marker="o",
            capsize=3,
            label=f"K = {K}",
        )
    axis.axhline(0, color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("Phase-scale constant, c")
    axis.set_ylabel("Change in greedy excess from c = 1 (percentage points)")
    axis.set_title("Paired phase-scale sensitivity (mean and 95% bootstrap CI)", loc="left")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, output_directory, "phase_scale_sensitivity")
    return figure


def _convergence_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    maximum_replicate = max(int(row["replicate"]) for row in rows)
    checkpoints = sorted(
        set([5, 10, 15, 20, 25, maximum_replicate])
        & set(range(1, maximum_replicate + 1))
    )
    results: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        sample = [row for row in rows if int(row["replicate"]) <= checkpoint]
        ratios = [float(row["greedy_ratio"]) for row in sample]
        results.append({
            "replicates_per_cell": checkpoint,
            "case_count": len(sample),
            "mean_greedy_ratio": fmean(ratios),
            "p95_greedy_ratio": _percentile(ratios, 0.95),
            "exact_match_rate": fmean(float(row["exact_match"]) for row in sample),
        })
    return results


def _convergence_figure(
    convergence: list[dict[str, object]], output_directory: Path
):
    import matplotlib.pyplot as plt

    x = [int(row["replicates_per_cell"]) for row in convergence]
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    axes[0].plot(
        x,
        [float(row["mean_greedy_ratio"]) for row in convergence],
        marker="o",
        label="Mean",
    )
    axes[0].plot(
        x,
        [float(row["p95_greedy_ratio"]) for row in convergence],
        marker="s",
        label="95th percentile",
    )
    axes[0].set_xlabel("Independent realizations per factorial cell")
    axes[0].set_ylabel("Greedy / exact sensor count")
    axes[0].set_title("(a) Ratio estimates", loc="left")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        x,
        [100 * float(row["exact_match_rate"]) for row in convergence],
        marker="o",
        color="#59A14F",
    )
    axes[1].set_xlabel("Independent realizations per factorial cell")
    axes[1].set_ylabel("Exact-optimum recovery (%)")
    axes[1].set_title("(b) Exact-match estimate", loc="left")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, output_directory, "primary_sample_convergence")
    return figure


def _claim_text(
    primary_summary: dict[str, object],
    replication_summary: dict[str, object] | None,
    convergence: list[dict[str, object]],
    paired_sensitivity: list[dict[str, object]],
) -> str:
    lines = [
        "# Empirical findings template",
        "",
        (
            f"In the pre-specified primary factorial study (n = "
            f"{primary_summary['case_count']}), the greedy allocation matched the "
            f"exact optimum in {100 * float(primary_summary['exact_match_rate']):.1f}% "
            "of cases."
        ),
        (
            f"Its mean greedy-to-optimal sensor ratio was "
            f"{float(primary_summary['mean_greedy_ratio']):.3f} "
            f"(95% bootstrap CI "
            f"{float(primary_summary['mean_greedy_ratio_ci95_low']):.3f}-"
            f"{float(primary_summary['mean_greedy_ratio_ci95_high']):.3f}); "
            f"the 95th percentile was {float(primary_summary['p95_greedy_ratio']):.3f} "
            f"and the maximum was {float(primary_summary['max_greedy_ratio']):.3f}."
        ),
        (
            f"The largest observed greedy excess used "
            f"{100 * float(primary_summary['max_theorem_excess_fraction_used']):.1f}% "
            "of the excess permitted by the worst-case logarithmic guarantee."
        ),
    ]
    if replication_summary is not None:
        lines.extend([
            "",
            (
                f"The independent-seed replication (n = "
                f"{replication_summary['case_count']}) produced a mean ratio of "
                f"{float(replication_summary['mean_greedy_ratio']):.3f}, an exact-match "
                f"rate of {100 * float(replication_summary['exact_match_rate']):.1f}%, "
                f"and a maximum ratio of "
                f"{float(replication_summary['max_greedy_ratio']):.3f}."
            ),
        ])
    if len(convergence) >= 2:
        previous = convergence[-2]
        final = convergence[-1]
        lines.extend([
            "",
            (
                "The sample-stability diagnostic changed the pooled mean ratio by "
                f"{abs(float(final['mean_greedy_ratio']) - float(previous['mean_greedy_ratio'])):.4f} "
                f"and the exact-match estimate by "
                f"{100 * abs(float(final['exact_match_rate']) - float(previous['exact_match_rate'])):.2f} "
                "percentage points when increasing from "
                f"{previous['replicates_per_cell']} to "
                f"{final['replicates_per_cell']} realizations per cell."
            ),
        ])
    nonreference_sensitivity = [
        row for row in paired_sensitivity if float(row["phase_scale_c"]) != 1.0
    ]
    if nonreference_sensitivity:
        means = [
            float(row["mean_change_in_greedy_excess_percentage_points"])
            for row in nonreference_sensitivity
        ]
        intervals_contain_zero = all(
            float(row["ci95_low"]) <= 0 <= float(row["ci95_high"])
            for row in nonreference_sensitivity
        )
        interval_statement = (
            "every 95% paired-bootstrap interval included zero"
            if intervals_contain_zero
            else "at least one 95% paired-bootstrap interval excluded zero"
        )
        lines.extend([
            "",
            (
                "Across the pre-specified phase-scale sensitivity contrasts, "
                "the mean change in greedy excess ranged from "
                f"{min(means):.2f} to {max(means):.2f} percentage points, and "
                f"{interval_statement}."
            ),
        ])
    lines.extend([
        "",
        "Interpretation: these results show that greedy was empirically much closer "
        "to optimal than its worst-case guarantee on the stated synthetic spatial-field "
        "distribution. They do not strengthen the theorem and should not be presented "
        "as a universal average-case guarantee.",
        "",
    ])
    return "\n".join(lines)


def analyze_paper_studies(
    records_by_dataset: dict[str, list[dict[str, object]]],
    output_directory: str | Path,
    *,
    bootstrap_iterations: int = 2_000,
) -> dict[str, object]:
    """Write case data, statistical tables, claims text, and paper figures."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()
    all_rows = [
        case_metrics(record)
        for records in records_by_dataset.values()
        for record in records
    ]
    _write_csv(output / "case_metrics.csv", all_rows)

    primary = [row for row in all_rows if row["dataset"] == "study-primary"]
    replication = [
        row for row in all_rows if row["dataset"] == "study-replication"
    ]
    sensitivity = [
        row for row in all_rows if row["dataset"] == "study-sensitivity"
    ]
    if not primary:
        raise ValueError("study-primary results are required for paper analysis")

    summary_rows: list[dict[str, object]] = []

    def add_summary(group_type: str, group_value: str, group: list[dict[str, object]]):
        seed = 20261201 + zlib.crc32(f"{group_type}:{group_value}".encode())
        summary_rows.append(summarize_group(
            group,
            group_type=group_type,
            group_value=group_value,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=seed,
        ))

    add_summary("study", "primary", primary)
    if replication:
        add_summary("study", "replication", replication)
    if sensitivity:
        add_summary("study", "sensitivity", sensitivity)
    for key, label in (("field_model", "field_model"), ("K", "K"), ("M", "M")):
        for value, group in sorted(_group_rows(primary, key).items(), key=lambda item: str(item[0])):
            add_summary(label, str(value), group)
    _write_csv(output / "group_summary.csv", summary_rows)

    primary_summary = next(
        row for row in summary_rows
        if row["group_type"] == "study" and row["group_value"] == "primary"
    )
    replication_summary = next(
        (
            row for row in summary_rows
            if row["group_type"] == "study" and row["group_value"] == "replication"
        ),
        None,
    )
    convergence = _convergence_rows(primary)
    _write_csv(output / "sample_convergence.csv", convergence)
    paired_sensitivity = (
        _paired_sensitivity_rows(
            sensitivity,
            bootstrap_iterations=bootstrap_iterations,
        )
        if sensitivity
        else []
    )
    _write_csv(output / "sensitivity_paired.csv", paired_sensitivity)
    claims = _claim_text(
        primary_summary,
        replication_summary,
        convergence,
        paired_sensitivity,
    )
    (output / "paper_findings.md").write_text(claims, encoding="utf-8")

    figures = {
        "quality": _quality_figure(primary, output),
        "ecdf": _ecdf_figure(primary, output),
        "runtime": _runtime_figure(primary, output),
        "convergence": _convergence_figure(convergence, output),
    }
    if sensitivity:
        figures["sensitivity"] = _sensitivity_figure(
            paired_sensitivity, output
        )

    payload = {
        "bootstrap_iterations": bootstrap_iterations,
        "primary_summary": primary_summary,
        "replication_summary": replication_summary,
        "sample_convergence": convergence,
        "paired_sensitivity": paired_sensitivity,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "rows": all_rows,
        "summaries": summary_rows,
        "figures": figures,
        "output_directory": str(output),
        **payload,
    }
