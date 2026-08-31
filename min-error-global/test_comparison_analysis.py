"""Tests for the four-method global comparison helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from comparison_analysis import (  # noqa: E402
    STATE_MODELS,
    diagnostic_gap_summary,
    error_curve_summary,
    make_combined_comparison_plots,
    make_extended_comparison_plots,
    phase_range_summary,
    scalable_error_curve_summary,
)


def _diagnostic_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for dimension in (1, 2):
        for state_model in STATE_MODELS:
            for case, optimal, greedy, near_uniform, random in (
                ("case_1", 0.1, 0.1, 0.12, 0.14),
                ("case_2", 0.2, 0.3, 0.25, 0.4),
            ):
                record: dict[str, object] = {
                    "dataset": f"compact_{dimension}d",
                    "case": case,
                    "field_model": (
                        "gaussian" if case == "case_1" else "quadratic"
                    ),
                    "spatial_dimension": dimension,
                    "sensor_budget": 3,
                    "state_model": state_model,
                    "delta_min": -2.0,
                    "delta_max": 3.0,
                    "delta_abs_min": 0.1,
                    "delta_abs_max": 3.0,
                    "n_delta_min": -6.0,
                    "n_delta_max": 9.0,
                    "n_delta_abs_min": 0.3,
                    "n_delta_abs_max": 9.0,
                }
                errors = {
                    "optimal": optimal,
                    "greedy": greedy,
                    "near_uniform": near_uniform,
                    "random": random,
                }
                for method, error in errors.items():
                    record[f"{method}_pgm_error"] = error
                    record[f"{method}_seconds"] = {
                        "optimal": 0.1,
                        "greedy": 0.01,
                        "near_uniform": 0.001,
                        "random": 0.0005,
                    }[method]
                for method in ("greedy", "near_uniform", "random"):
                    gap = errors[method] - optimal
                    record[f"{method}_pgm_error_gap"] = gap
                    record[f"{method}_relative_pgm_error_gap"] = gap / optimal
                records.append(record)
    return records


class ComparisonAnalysisTests(unittest.TestCase):
    def test_error_summary_contains_all_methods_and_gaps(self) -> None:
        rows = error_curve_summary(_diagnostic_records())
        product_1d = next(
            row
            for row in rows
            if row["spatial_dimension"] == 1
            and row["state_model"] == "product"
        )

        self.assertAlmostEqual(product_1d["optimal_mean_pgm_error"], 0.15)
        self.assertAlmostEqual(product_1d["greedy_mean_pgm_error"], 0.2)
        self.assertAlmostEqual(
            product_1d["near_uniform_mean_pgm_error_gap"], 0.035
        )
        self.assertAlmostEqual(product_1d["random_mean_pgm_error_gap"], 0.12)

    def test_gap_diagnostics_include_percentiles_and_optimal_fraction(self) -> None:
        rows = diagnostic_gap_summary(_diagnostic_records())
        product_greedy = next(
            row
            for row in rows
            if row["spatial_dimension"] == 1
            and row["state_model"] == "product"
            and row["method"] == "Greedy"
        )

        self.assertAlmostEqual(product_greedy["mean_relative_gap"], 0.25)
        self.assertAlmostEqual(product_greedy["p90_relative_gap"], 0.45)
        self.assertAlmostEqual(product_greedy["max_relative_gap"], 0.5)
        self.assertAlmostEqual(product_greedy["fraction_exactly_optimal"], 0.5)

    def test_phase_summary_reports_signed_and_absolute_ranges(self) -> None:
        row = phase_range_summary(_diagnostic_records())

        self.assertEqual(row["cases"], 4)
        self.assertEqual(row["delta_min"], -2.0)
        self.assertEqual(row["delta_max"], 3.0)
        self.assertEqual(row["n_delta_abs_max"], 9.0)

    def test_extended_summary_and_plots_exclude_optimal_reference(self) -> None:
        records = _diagnostic_records()
        rows = scalable_error_curve_summary(records)
        figures = make_extended_comparison_plots(records)

        self.assertIn("greedy_mean_pgm_error", rows[0])
        self.assertNotIn("optimal_mean_pgm_error", rows[0])
        self.assertEqual(
            [len(figure.axes) for _, figure in figures],
            [6, 6, 2],
        )
        labels = figures[0][1].axes[0].get_legend_handles_labels()[1]
        self.assertEqual(labels, ["Greedy", "Near-uniform", "Random"])
        for _, figure in figures:
            plt.close(figure)

    def test_plot_helper_builds_requested_panels_and_scales(self) -> None:
        figures = make_combined_comparison_plots(_diagnostic_records())

        self.assertEqual(
            [len(figure.axes) for _, figure in figures],
            [6, 6, 6, 2, 2, 2],
        )
        error_figure = figures[0][1]
        gap_figure = figures[1][1]
        runtime_figure = figures[5][1]
        self.assertGreater(error_figure.axes[4].get_ylim()[0], 0.0)
        self.assertEqual(gap_figure.axes[4].get_ylim()[0], 0.0)
        self.assertEqual(runtime_figure.axes[0].get_yscale(), "log")
        self.assertEqual(plt.rcParams["font.size"], 12.0)
        self.assertEqual(plt.rcParams["font.serif"][0], "Times New Roman")
        for _, figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
