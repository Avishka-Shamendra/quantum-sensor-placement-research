"""Storage regression tests use stub solvers; no research study is executed."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comparison.config import Config
from comparison.runner import cache_coverage, run_comparison
from comparison.dataset import prepare_dataset
from comparison.cache import checkpoint_path


class CacheWorkflowTests(unittest.TestCase):
    def test_switch_joint_only_runs_missing_algorithm(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            data = project / 'input.csv'
            data.write_text('stub input')
            case = dict(instance_id='a', dimension=1, family='quadratic', K=2, M=2,
                        replicate=1, N_LB_joint=4, N_max=3, status='eligible',
                        generation_seed=12, file=data.name,
                        csv_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                        budget_schedule=[dict(N=1, target_fractions=[.3])])
            manifest = dict(dataset_id='test', cases=[case])
            solution = SimpleNamespace(allocation=(1, 0), decision_tree=object())
            sequential_solver = Mock(return_value=solution)
            vnd, greedy = Mock(return_value=solution), Mock(return_value=solution)
            seq = dict(input_data=SimpleNamespace(load_phase_input=lambda _: (2, 2, {})),
                       heuristic=SimpleNamespace(heuristic=sequential_solver),
                       greedy=SimpleNamespace(greedy=Mock(return_value=solution)),
                       actual_error=SimpleNamespace(actual_tree_error=lambda *args: .2),
                       strategy=SimpleNamespace(tree_to_dict=lambda _: {'declared_field': 0}))
            joint = SimpleNamespace(DICKE_POLICY='stub', vnd_search=vnd, greedy_search=greedy,
                                    probe_gram=lambda *a, **k: None, pgm_error=lambda _: .1)
            implementations = (None, seq, joint, SimpleNamespace(load_phase_input=lambda *a, **k: (None, None)), None)
            with patch('comparison.runner.implementations', return_value=implementations):
                old = Config(joint_algorithm='vnd')
                run_comparison(project, old, project, manifest)
                current = Config(joint_algorithm='greedy')
                coverage = cache_coverage(project, current, manifest)
                self.assertEqual(coverage['sequential']['available'], 1)
                self.assertEqual(coverage['joint']['missing'], 1)
                with self.assertRaises(FileNotFoundError):
                    run_comparison(project, current, project, manifest, run=False)
                greedy.assert_not_called()
                rows, out = run_comparison(project, current, project, manifest)
                self.assertEqual(out.name, 'compact')
                self.assertEqual(rows[0]['joint_algorithm'], 'greedy')
                sequential_solver.assert_called_once()
                greedy.assert_called_once()
                run_comparison(project, old, project, manifest, run=False)
                vnd.assert_called_once()
                changed = Config(beam_width=4)
                self.assertEqual(cache_coverage(project, changed, manifest)['sequential']['missing'], 1)
                self.assertEqual(cache_coverage(project, changed, manifest)['joint']['available'], 1)
                data.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    run_comparison(project, current, project, manifest)

    def test_dataset_reuse_survives_algorithm_and_reporting_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            directory = project / 'data/test/legacy'
            directory.mkdir(parents=True)
            (directory/'input.csv').write_text('saved data')
            config = Config(mode='compact', joint_algorithm='greedy', bootstrap_samples=10)
            manifest = dict(dataset_id='legacy', design=dict(mode='test', seed=config.seed,
                K=[2, 3], M=[2, 3], replicates=2, c=1., minimum_best_pair_phase_gap=0.),
                cases=[dict(file='input.csv', csv_sha256=hashlib.sha256(b'saved data').hexdigest())])
            (directory/'manifest.json').write_text(json.dumps(manifest))
            with patch('comparison.dataset.implementations', return_value=(None,)*5):
                self.assertEqual(prepare_dataset(project, config), (directory, manifest))


if __name__ == '__main__':
    unittest.main()
