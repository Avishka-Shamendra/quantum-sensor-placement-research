"""Small-instance checks for physical evaluation and actual-error optimization."""

import csv
from dataclasses import replace
from itertools import combinations, product
import json
from math import cos, pi
from pathlib import Path
from random import Random
import tempfile
import unittest
from unittest.mock import patch

from actual_error import actual_node_probabilities, actual_tree_error, off_pair_probability
from analysis import PROJECT, run_analysis, select_cases
from exhaustive_search import exhaustive_search
from baselines import random_tree
from greedy import allocation_score, build_greedy_tree, greedy
from heuristic import heuristic
from input_data import load_phase_input, pair_error_probability, write_phase_input
from random_baseline import random_baseline
from strategy import DecisionLeaf, DecisionNode, OutcomeModel, SearchTimeout, budget_allocations
from uniform import uniform


PHASES = {(0, 1): [-0.8, -1.1], (0, 2): [-1.4, -0.3], (1, 2): [-0.6, 0.8]}
METHODS = (exhaustive_search, greedy, heuristic, uniform, random_baseline)


def all_trees(remaining, M):
    """Independent literal enumeration, used only for three fields."""
    if len(remaining) == 1:
        yield DecisionLeaf(remaining[0])
        return
    for i, j in combinations(remaining, 2):
        left = list(all_trees(tuple(f for f in remaining if f != i), M))
        right = list(all_trees(tuple(f for f in remaining if f != j), M))
        for location, ti, tj in product(range(M), left, right):
            yield DecisionNode((i, j), location, 0.0, ti, tj)


class ActualErrorTests(unittest.TestCase):
    def test_binary_measurement_matches_pair_error(self):
        for phase in (0.0, 0.3, -0.8, pi):
            for count in (0, 1, 2, 5, 11):
                phases = {(0, 1): [phase]}
                error = pair_error_probability(phase, count)
                self.assertAlmostEqual(1 - actual_node_probabilities((0, 1), 0, count, 0, phases).keep_i, error)
                self.assertAlmostEqual(actual_node_probabilities((0, 1), 0, count, 1, phases).keep_i, error)

    def test_third_field_uses_physical_outcomes(self):
        probability = actual_node_probabilities((0, 1), 0, 1, 2, PHASES).keep_i
        self.assertGreater(abs(probability - 0.5), 0.05)
        self.assertTrue(0 <= probability <= 1)

    def test_appendix_d_exact_threshold_case(self):
        phases = {
            (0, 1): [pi / 2],
            (0, 2): [0.37],
            (1, 2): [0.37 - pi / 2],
        }
        expected = 0.5 + 0.5 * (
            cos(0.37) ** 2 - cos(0.37 - pi / 2) ** 2
        )
        self.assertAlmostEqual(
            off_pair_probability(0, 1, 2, 0, 2, phases), expected
        )

    def test_exhaustive_matches_literal_enumeration(self):
        for budget in (0, 1, 2):
            expected = min(actual_tree_error(t, 3, PHASES, a)
                           for a in budget_allocations(budget, 2) for t in all_trees((0, 1, 2), 2))
            result = exhaustive_search(3, 2, budget, PHASES)
            self.assertAlmostEqual(result.actual_error, expected)

    def test_all_reported_errors_recompute_and_respect_optimum(self):
        optimum = exhaustive_search(3, 2, 2, PHASES).actual_error
        for method in METHODS:
            result = method(3, 2, 2, PHASES)
            self.assertEqual(sum(result.allocation), 2)
            self.assertAlmostEqual(result.actual_error, actual_tree_error(result.decision_tree, 3, PHASES, result.allocation))
            self.assertGreaterEqual(result.actual_error + 1e-10, optimum)

    def test_zero_budget_and_identical_fields(self):
        for method in METHODS:
            self.assertAlmostEqual(method(3, 2, 0, PHASES).actual_error, 2 / 3)
            zeros = {pair: [0.0, 0.0] for pair in PHASES}
            self.assertAlmostEqual(method(3, 2, 3, zeros).actual_error, 2 / 3)

    def test_inconsistent_field_phases_rejected(self):
        with self.assertRaises(ValueError):
            exhaustive_search(3, 2, 1, {**PHASES, (1, 2): [0.0, 0.0]})

    def test_actual_evaluation_ignores_stored_tree_scores(self):
        result = greedy(3, 2, 1, PHASES)
        tree = replace(result.decision_tree, pair_error=0.5)
        self.assertEqual(actual_tree_error(tree, 3, PHASES, result.allocation), result.actual_error)

    def test_timeout_does_not_return_an_optimum(self):
        with patch.object(OutcomeModel, "check_time", side_effect=SearchTimeout):
            with self.assertRaises(SearchTimeout):
                exhaustive_search(3, 2, 2, PHASES)

    def test_greedy_placement_uses_only_average_pair_scores(self):
        model = OutcomeModel(3, 2, 2, PHASES)
        expected = (0, 0)
        for _ in range(2):
            candidates = []
            for location in range(2):
                trial = tuple(
                    count + int(index == location)
                    for index, count in enumerate(expected)
                )
                candidates.append((allocation_score(model, trial), location, trial))
            _, _, expected = min(candidates)

        with patch("greedy.build_greedy_tree", wraps=build_greedy_tree) as builder:
            result = greedy(3, 2, 2, PHASES)
        self.assertEqual(result.allocation, expected)
        self.assertEqual(builder.call_count, 1)

    def test_greedy_tree_uses_only_local_pair_error(self):
        model = OutcomeModel(3, 2, 2, PHASES)
        allocation = (1, 1)
        with patch.object(
            model,
            "probabilities",
            side_effect=AssertionError("Q must not guide greedy tree construction"),
        ):
            tree = build_greedy_tree(model, allocation)

        def check(node, remaining):
            if isinstance(node, DecisionLeaf):
                self.assertEqual(remaining, (node.declared_field,))
                return
            expected_error, expected_pair, expected_location = min(
                (model.pair_error(pair, location, allocation[location]), pair, location)
                for pair in combinations(remaining, 2)
                for location in range(model.M)
            )
            self.assertEqual((node.pair, node.location), (expected_pair, expected_location))
            self.assertAlmostEqual(node.pair_error, expected_error)
            i, j = node.pair
            check(node.discard_i, tuple(field for field in remaining if field != i))
            check(node.discard_j, tuple(field for field in remaining if field != j))

        check(tree, tuple(range(model.K)))

    def test_random_tree_samples_pair_and_location_without_pair_scoring(self):
        model = OutcomeModel(3, 2, 2, PHASES)
        tree = random_tree(model, (1, 1), Random(1))
        self.assertEqual((tree.pair, tree.location), ((0, 1), 1))
        best = min(
            (model.pair_error(pair, location, 1), pair, location)
            for pair in model.pairs for location in range(model.M)
        )
        self.assertNotEqual((tree.pair, tree.location), (best[1], best[2]))

    def test_independent_run_checkpoint_and_changed_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for dimension in (1, 2):
                directory = root / "inputs" / f"compact_benchmark_{dimension}d"
                directory.mkdir(parents=True)
                phases = {pair: [*values, values[0]] for pair, values in PHASES.items()}
                write_phase_input(directory / "case_00001.csv", [
                    (i, j, m, phase)
                    for (i, j), values in phases.items() for m, phase in enumerate(values)])
                self.assertEqual(load_phase_input(directory / "case_00001.csv")[:2], (3, 3))
                metadata = {"file": "case_00001.csv", "K": 3, "M": 3, "replicate": 1,
                            "sensor_budgets": [1], "minimum_sensors_for_zero_error": 4,
                            "field_model": "gaussian", "spatial_dimension": dimension}
                (directory / "manifest.json").write_text(json.dumps({"cases": [metadata]}))
            with patch("analysis.TEST_CASES_PER_DIMENSION", 1), patch("builtins.print"):
                first = run_analysis(project=root)
            self.assertEqual(len(first["rows"]), 10)
            self.assertEqual({r["status"] for r in first["rows"]}, {"ok"})
            with patch("analysis.TEST_CASES_PER_DIMENSION", 1), patch("analysis.SOLVERS", {name: lambda *a, **k: self.fail("Cache missed")
                       for name in ("Exhaustive", "Greedy", "Heuristic", "Uniform", "Random")}), patch("builtins.print"):
                second = run_analysis(project=root)
            self.assertEqual(first["rows"], second["rows"])
            with patch("analysis.TEST_CASES_PER_DIMENSION", 1), patch("builtins.print"):
                third = run_analysis(project=root, time_limit=31)
            self.assertNotEqual(first["rows"][0]["fingerprint"], third["rows"][0]["fingerprint"])
            with (first["output"] / "tables/error_summary.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 10)
            with (first["output"] / "tables/timeouts.csv").open() as stream:
                timeout_reader = csv.DictReader(stream)
                self.assertIn("time_limit_seconds", timeout_reader.fieldnames)
                self.assertEqual(list(timeout_reader), [])

    def test_default_test_runs_have_180_cases_per_dimension(self):
        for study in ("compact", "extended"):
            entries = select_cases(PROJECT, study, "test", exact_k_max=4)
            counts = {
                dimension: sum(
                    case["spatial_dimension"] == dimension
                    for _, _, case in entries
                )
                for dimension in (1, 2)
            }
            self.assertEqual(counts, {1: 180, 2: 180})


if __name__ == "__main__":
    unittest.main()
