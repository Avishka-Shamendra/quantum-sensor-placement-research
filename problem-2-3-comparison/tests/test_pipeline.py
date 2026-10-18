"""Contract checks using synthetic records and mocked discrimination methods."""
import hashlib
import importlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comparison.config import Config, budget_schedule
from comparison.reporting import family_table, paired_interval
from comparison.runner import run_comparison
from comparison.sources import implementations


class ComparisonTests(unittest.TestCase):
    def test_rounding_aliases_and_cap(self):
        self.assertEqual(budget_schedule(1), (0, []))
        maximum, schedule = budget_schedule(3)
        self.assertEqual(maximum, 2)
        self.assertEqual(schedule, [
            {'N': 1, 'target_fractions': [.3, .4, .5, .6, .7]},
            {'N': 2, 'target_fractions': [.8, .9]},
        ])
        self.assertEqual(budget_schedule(100)[0], 50)
        self.assertEqual(budget_schedule(6)[1][0]['target_fractions'], [.1, .2])
        for bound in range(1, 100):
            maximum, schedule = budget_schedule(bound)
            self.assertEqual(len(schedule), len({r['N'] for r in schedule}))
            self.assertTrue(all(0 < r['N'] <= maximum < bound for r in schedule))

    def test_table_equal_instance_weights(self):
        rows = [dict(instance_id=name, N=n, dimension=1, family='gaussian',
                     sequential_error=value, joint_error=value / 2)
                for name, n, value in [('a', 1, .2), ('a', 2, .4), ('b', 1, .9)]]
        table = family_table(rows)
        self.assertEqual(len(table), 6)
        self.assertAlmostEqual(table[0]['Sequential Mean Error'], .6)
        self.assertAlmostEqual(table[0]['Joint Mean Error'], .3)
        with self.assertRaises(ValueError):
            family_table(rows + [rows[0]])

    def test_bootstrap_preserves_pairs(self):
        rows = [dict(instance_id=str(i), family='gaussian', K=2, M=2,
                     sequential_error=x, joint_error=x + .1)
                for i, x in enumerate((.1, .2, .3, .4))]
        result = paired_interval(rows, 100, np.random.default_rng(12))
        for key in ('mean', 'low', 'high'):
            self.assertAlmostEqual(result[key][1] - result[key][0], .1)
        self.assertIsNone(paired_interval([], 100, np.random.default_rng(12)))
        with self.assertRaises(ValueError):
            paired_interval(rows + [rows[0]], 100, np.random.default_rng(12))

    def test_imports_preserve_existing_namespaces(self):
        names = ('input_data', 'actual_error', 'strategy', 'src')
        before = {n: sys.modules.get(n) for n in names}
        _, seq, joint, _, _ = implementations()
        self.assertEqual(before, {n: sys.modules.get(n) for n in names})
        self.assertEqual(seq['heuristic'].heuristic.__module__, '_comparison_seq_heuristic')
        self.assertEqual(joint.vnd_search.__module__, '_comparison_joint.search')

    def test_resume_after_one_method_fails(self):
        # The real generators, lower-bound solver, and algorithms are never called.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / 'case.csv'
            csv_path.write_text('mock input')
            digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            case = dict(instance_id='a', dimension=1, family='gaussian', K=2, M=2,
                        replicate=1, N_LB_joint=4, N_max=3, generation_seed=10,
                        file='case.csv', csv_sha256=digest, status='eligible',
                        budget_schedule=[dict(N=1, target_fractions=[.3])])
            solution = SimpleNamespace(allocation=(1, 0), decision_tree='tree')
            heuristic = Mock(return_value=solution)
            actual = Mock(return_value=.25)
            vnd = Mock(side_effect=[RuntimeError('interrupted'), solution])
            seq = dict(input_data=SimpleNamespace(load_phase_input=lambda p: (2, 2, 'gaps')),
                       heuristic=SimpleNamespace(heuristic=heuristic),
                       actual_error=SimpleNamespace(actual_tree_error=actual),
                       strategy=SimpleNamespace(tree_to_dict=lambda t: {'tree': t}))
            joint = SimpleNamespace(DICKE_POLICY='test', vnd_search=vnd,
                                    probe_gram=Mock(return_value='gram'), pgm_error=Mock(return_value=.15))
            inputs = SimpleNamespace(load_phase_input=lambda p, c: ('fields', 'phases'))
            manifest = dict(dataset_id='test', design={'source_sha256': {}}, cases=[case])
            with patch('comparison.runner.implementations', return_value=(None, seq, joint, inputs, None)), \
                 patch('comparison.runner.source_hashes', return_value={}):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    run_comparison(root, Config(), root, manifest)
                rows, output = run_comparison(root, Config(), root, manifest)
                self.assertEqual(rows[0]['sequential_error'], .25)
                self.assertEqual(rows[0]['joint_error'], .15)
                run_comparison(root, Config(), root, manifest)
                self.assertEqual(heuristic.call_count, 1)
                self.assertEqual(actual.call_count, 1)
                self.assertEqual(vnd.call_count, 2)
                self.assertTrue((output / 'matched_results.csv').exists())
                csv_path.write_text('modified')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    run_comparison(root, Config(), root, manifest)


if __name__ == '__main__':
    unittest.main()
