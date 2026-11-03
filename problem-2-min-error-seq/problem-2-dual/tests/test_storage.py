"""No solver calls: check cache isolation and flat-output method preservation."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import budget_limited as study


class StorageTests(unittest.TestCase):
    def test_other_algorithm_source_does_not_invalidate_vnd(self):
        settings = dict(beam_width=3, max_iterations=30, time_limit=None)
        original = Path.read_bytes
        before = {m: study.cache_signature(m, settings) for m in ('vnd_beam', 'greedy')}
        def modified(path):
            value = original(path)
            return value + b'\n# hypothetical Greedy edit\n' if path.name == 'greedy.py' else value
        with patch.object(Path, 'read_bytes', modified):
            self.assertEqual(study.cache_signature('vnd_beam', settings), before['vnd_beam'])
            self.assertNotEqual(study.cache_signature('greedy', settings), before['greedy'])

    def test_flat_exports_keep_unselected_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = root / 'results/compact/methods/greedy/solutions.json'
            previous.parent.mkdir(parents=True)
            previous.write_text('["previous"]')
            modules = {'analysis': SimpleNamespace(select_cases=lambda *a, **k: [])}
            with patch.object(study.dual_search, 'HERE', root), \
                 patch.object(study.dual_search, 'dependencies', return_value=modules), \
                 patch.object(study, 'cache_signature', return_value='stub'), \
                 patch.object(study, 'run_method', return_value=([{'method': 'vnd_beam'}], [])) as run:
                rows, failures, output = study.run_comparison('compact', methods=('vnd_beam',), run=False)
            self.assertEqual(output, root / 'results/compact')
            self.assertEqual(previous.read_text(), '["previous"]')
            self.assertEqual(run.call_args.kwargs['run'], False)
            self.assertTrue((output/'run_info.json').exists())


if __name__ == '__main__':
    unittest.main()
