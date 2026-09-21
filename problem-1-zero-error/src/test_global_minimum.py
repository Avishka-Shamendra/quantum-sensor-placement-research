"""Independent small-instance checks: python problem-1-zero-error/src/test_global_minimum.py."""

import itertools
import json
from math import pi
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

sys.path.insert(0, str(Path(__file__).resolve().parent))

from global_minimum import optimal_global_minimum
from optimal import optimal_perfect_pairwise_sensor_placement
from phase_data import compute_nijm
from sensor_budget_analysis import run_budget_analysis
from run_config import select_cases
import eval as dataset_eval
from benchmark_analysis import allocation_match_table, case_metrics
from field_models import generate_case


def brute_force_minimum(phases, M, upper):
    for total in range(upper + 1):
        for allocation in itertools.product(range(total + 1), repeat=M - 1):
            last = total - sum(allocation)
            if last < 0:
                continue
            candidate = (*allocation, last)
            if all(sum(abs(r) * n for r, n in zip(values, candidate)) >= pi - 1e-10
                   for values in phases.values()):
                return total
    raise AssertionError("No feasible enumerated allocation")


class GlobalMinimumTests(unittest.TestCase):
    def test_nearly_feasible_integer_allocation_rejected(self):
        total, _, diagnostics = optimal_global_minimum(
            2, 1, {(0, 1): [pi * (1 - 5e-7) / 2]})
        self.assertEqual(total, 3)
        self.assertGreaterEqual(diagnostics["global_min_pair_coverage_pi"], 1 - 1e-8)

    def test_quadratic_solver_tolerance_regression(self):
        # Reproduces benchmark_1d/case_19614 without needing generated files.
        rows, _ = generate_case("quadratic", 1, 12, 9, seed=19634319743)
        phases = {}
        for i, j, m, phase in rows:
            phases.setdefault((i, j), [0.0] * 9)[m] = phase
        total, allocation, diagnostics = optimal_global_minimum(12, 9, phases)
        self.assertEqual(total, 53)
        self.assertEqual(sum(allocation), total)
        self.assertGreaterEqual(diagnostics["global_min_pair_coverage_pi"], 1 - 1e-8)

    def test_allocation_match_distinguishes_alternative_optima(self):
        rows = []
        for index, allocation in enumerate(([2, 0], [0, 2], [2, 1])):
            record = {
                "spatial_dimension": 1, "case": str(index), "field_model": "gaussian",
                "K": 3, "M": 2,
                "optimal": {"minimum_sensors": 2, "allocation": [2, 0], "mean_seconds": 1.0},
                "greedy": {"minimum_sensors": sum(allocation), "allocation": allocation, "mean_seconds": 1.0},
                "comparison": {"greedy_to_optimal_ratio": sum(allocation) / 2, "theorem_factor": 2.0},
            }
            rows.append(case_metrics(record))
        self.assertEqual([row["allocation_match"] for row in rows], [1, 0, 0])
        self.assertEqual([row["exact_match"] for row in rows], [1, 1, 0])
        summary = allocation_match_table(rows)[0]
        self.assertEqual(summary["case_count"], 3)
        self.assertEqual(summary["allocation_matches"], 1)
        self.assertEqual(summary["budget_matches"], 2)
        self.assertAlmostEqual(summary["allocation_match_percent"], 100 / 3)

    def test_one_pair_closed_form_and_phase_sign(self):
        for phases, expected in [([0, pi / 3], 3), ([-pi, 0], 1), ([0.4*pi, -0.3*pi], 3)]:
            total, allocation, diagnostics = optimal_global_minimum(2, 2, {(0, 1): phases})
            self.assertEqual(total, expected)
            self.assertEqual(total, sum(allocation))
            self.assertEqual(diagnostics["global_solver_status"], 0)

    def test_one_location_matches_sequential(self):
        phases = {(0, 1): [0.3*pi], (0, 2): [0.5*pi], (1, 2): [0.2*pi]}
        actual, _, _ = optimal_global_minimum(3, 1, phases)
        sequential, _, _ = optimal_perfect_pairwise_sensor_placement(3, 1, compute_nijm(3, 1, phases))
        self.assertEqual(actual, sequential)

    def test_strict_gap_for_realizable_fields(self):
        # Field phases (0, .2*pi, .6*pi) at location 1 and
        # (0, .6*pi, .2*pi) at location 2. Shared contributions need 2+1
        # sensors, whereas covering each pair at one location requires 3+2.
        phases = {(0, 1): [-.2*pi, -.6*pi], (0, 2): [-.6*pi, -.2*pi], (1, 2): [-.4*pi, .4*pi]}
        nglobal, _, _ = optimal_global_minimum(3, 2, phases)
        nmin, _, _ = optimal_perfect_pairwise_sensor_placement(3, 2, compute_nijm(3, 2, phases))
        self.assertEqual((nglobal, nmin), (3, 5))

    def test_random_small_instances_against_enumeration(self):
        rng = random.Random(192)
        for _ in range(30):
            phases = {pair: [rng.uniform(.2, .9)*pi for _ in range(3)]
                      for pair in ((0, 1), (0, 2), (1, 2))}
            nmin, _, _ = optimal_perfect_pairwise_sensor_placement(3, 3, compute_nijm(3, 3, phases))
            actual, _, _ = optimal_global_minimum(3, 3, phases)
            self.assertEqual(actual, brute_force_minimum(phases, 3, nmin))
            self.assertLessEqual(actual, nmin)

    def test_indistinguishable_pair_rejected(self):
        with self.assertRaises(ValueError):
            optimal_global_minimum(2, 2, {(0, 1): [0.0, 0.0]})

    def test_malformed_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            optimal_global_minimum(2, 2, {(0, 1): [pi]})

    def test_nonoptimal_incumbent_rejected(self):
        result = OptimizeResult(status=1, success=False, x=np.array([2.]), message="Time limit")
        with patch("global_minimum.milp", return_value=result), self.assertRaises(RuntimeError):
            optimal_global_minimum(2, 1, {(0, 1): [pi]})

    def test_both_analyses_run_without_prior_outputs(self):
        # Input-only project: no notebook 01 output or results directory exists.
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            for dimension in (1, 2):
                directory = project / "inputs" / f"benchmark_{dimension}d"
                directory.mkdir(parents=True)
                phase_csv = "i,j,m,r_ijm\n" + "".join(f"0,1,{m},{pi/2}\n" for m in range(3))
                (directory / "case_00001.csv").write_text(phase_csv)
                (directory / "case_00002.csv").write_text(phase_csv)
                case = {"file": "case_00001.csv", "K": 2, "M": 3,
                        "spatial_dimension": dimension, "field_model": "gaussian",
                        "replicate": 1, "generation_seed": 1, "phase_scale_c": 1.0}
                second = {**case, "file": "case_00002.csv", "replicate": 2}
                (directory / "manifest.json").write_text(json.dumps({
                    "dataset": f"benchmark_{dimension}d", "cases": [case, second]}))
            with patch("builtins.print"):
                comparison = run_budget_analysis(project, include_global_minimum=True)
                minimum = run_budget_analysis(project)
            self.assertEqual([r["N_min"] for r in minimum["rows"]], [2] * 4)
            self.assertEqual([r["N_min_global"] for r in comparison["rows"]], [2] * 4)
            full_table = project / "results/global/full/tables/global_vs_seq_by_dimension_k.csv"
            original = full_table.read_bytes()
            with patch("builtins.print"):
                quick = run_budget_analysis(project, include_global_minimum=True, run_mode="test")
            self.assertEqual(len(quick["rows"]), 2)
            self.assertEqual(quick["output_directory"], project / "results/global/test")
            self.assertTrue((quick["output_directory"] / "tables/case_budgets.csv").is_file())
            self.assertEqual(full_table.read_bytes(), original)
            with patch.object(dataset_eval, "INPUT_ROOT", project / "inputs"), \
                    patch.object(dataset_eval, "RESULTS_ROOT", project / "results/methods"), \
                    patch("builtins.print"):
                methods, methods_path = dataset_eval.evaluate_dataset("benchmark_1d", run_mode="test")
            self.assertEqual(len(methods), 1)
            self.assertEqual(methods_path, project / "results/methods/test/1d/tables")
            self.assertEqual(methods[0]["case"], quick["rows"][0]["case"])
            self.assertEqual(case_metrics(methods[0])["theorem_excess_fraction_used"], 0.0)

    def test_twelve_fields_at_one_location(self):
        rows, metadata = generate_case("gaussian", 2, 12, 1, seed=20260921)
        phases = {(i, j): [phase] for i, j, m, phase in rows}
        thresholds = compute_nijm(12, 1, phases)
        total, allocation, _ = optimal_perfect_pairwise_sensor_placement(12, 1, thresholds)
        self.assertEqual(metadata["M"], 1)
        self.assertEqual(len(rows), 66)
        self.assertEqual(total, max(values[0] for values in thresholds.values()))
        self.assertEqual(allocation, [total])

    def test_mode_selection_keeps_all_families_and_k(self):
        entries = [(Path(f"{family}_{k}_{m}_{r}.csv"),
                    {"field_model": family, "K": k, "M": m, "replicate": r})
                   for family in ("gaussian", "sinusoidal", "quadratic")
                   for k in range(3, 11) for m in range(3, 11) for r in range(1, 31)]
        self.assertEqual(len(select_cases(entries, "test")), 48)
        self.assertEqual(len(select_cases(entries, "full")), 5760)
        with self.assertRaises(ValueError):
            select_cases(entries, "typo")


if __name__ == "__main__":
    unittest.main()
