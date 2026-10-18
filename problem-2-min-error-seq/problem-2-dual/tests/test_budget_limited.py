"""Cutoff selection and cache reuse without heuristic execution."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import budget_limited as runner


class CutoffTests(unittest.TestCase):
    def test_cutoff_includes_boundary(self):
        entries = [('dataset', Path('case.csv'), dict(file=str(n), spatial_dimension=1,
                   field_model='gaussian', K=2, M=1, minimum_sensors_for_zero_error=n))
                   for n in (4999, 5000, 5001)]
        kept, skipped = runner.partition(entries, 5000)
        self.assertEqual([e[2]['minimum_sensors_for_zero_error'] for e in kept], [4999, 5000])
        self.assertEqual([e['N_min_seq'] for e in skipped], [5001])

    def test_original_saved_signatures_match(self):
        settings = dict(beam_width=3, max_iterations=30, time_limit=None)
        roots = {'vnd_beam': 'results/test', 'greedy': 'results/greedy/test'}
        for method, folder in roots.items():
            signature = runner.cache_signature(method, settings)
            info = runner.dual_search.HERE / folder / signature[:16] / 'run_info.json'
            self.assertTrue(info.exists())
            self.assertEqual(json.loads(info.read_text())['signature'], signature)

    def test_completed_target_reuses_cache_without_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root / 'case.csv'
            csv.write_text('i,j,m,r_ijm\n0,1,0,0.2\n')
            case = dict(file='case.csv', spatial_dimension=1, field_model='gaussian', K=2,
                        M=1, minimum_sensors_for_zero_error=20)
            best = dict(N=10, allocation=[10], decision_tree={})
            cache = dict(perfect=dict(N=20, allocation=[20], decision_tree={}, runtime_seconds=0.),
                         targets={'0.1': dict(low=9, high=10, best=best, trace=[],
                                             search_seconds=0., complete=True)})
            solver = Mock(side_effect=AssertionError('Completed cache must not run a heuristic'))
            modules = {'input_data': SimpleNamespace(load_phase_input=lambda p: (2, 1, {})),
                       'actual_error': SimpleNamespace(actual_tree_error=lambda *args: .05),
                       'heuristic': SimpleNamespace(heuristic=solver)}
            with patch.object(runner.dual_search, 'HERE', root), \
                    patch.object(runner, 'cache_signature', return_value='existing'), \
                    patch.object(runner.dual_search, 'dependencies', return_value=modules), \
                    patch.object(runner.dual_search, 'initial_solution', side_effect=AssertionError):
                path = runner.cache_path('existing', csv, case)
                runner.dual_search.save_json(path, cache)
                rows, failures = runner.run_method('vnd_beam', [('dataset', csv, case)], (.1,),
                    dict(beam_width=3, max_iterations=30, time_limit=None), resume=True, run=True)
                self.assertEqual(rows[0]['N'], 10)
                self.assertEqual(failures, [])
                solver.assert_not_called()


if __name__ == '__main__':
    unittest.main()
