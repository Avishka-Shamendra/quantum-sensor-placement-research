"""Verify current checkpoints survive stale-record cleanup without solver calls."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import analysis


class ResultStorageTests(unittest.TestCase):
    def test_resume_removes_old_fingerprint_and_keeps_current_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            path = project/'case.csv'
            path.write_text('input')
            metadata = dict(K=2, M=1, sensor_budgets=[1], spatial_dimension=1, field_model='gaussian')
            settings = dict(study='extended', mode='test', exact_k_max=4, time_limit=None,
                            beam_width=3, max_iterations=30)
            fingerprint = hashlib.sha256(path.read_bytes() + json.dumps(settings, sort_keys=True).encode()
                + b'current' + json.dumps(metadata, sort_keys=True).encode()).hexdigest()
            row = dict(fingerprint=fingerprint, dataset='data', case='case', N=1, method='Greedy',
                       allocation=[1], actual_error=.2, decision_tree={}, dimension=1, status='ok')
            checkpoint = project/'results/extended/test/records.jsonl'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text(json.dumps(dict(row, fingerprint='obsolete'))+'\n'+json.dumps(row)+'\n')
            solver = Mock(side_effect=AssertionError('Solver must not run'))
            with patch.object(analysis, 'select_cases', return_value=[('data', path, metadata)]), \
                 patch.object(analysis, 'source_code_hash', return_value='current'), \
                 patch.object(analysis, 'EXTENDED_METHODS', ('Greedy',)), \
                 patch.object(analysis, 'load_phase_input', return_value=(2, 1, {})), \
                 patch.object(analysis, 'SOLVERS', {'Greedy': solver}), \
                 patch.object(analysis, 'summarize', return_value=[]), \
                 patch.object(analysis, 'write_csv'):
                result = analysis.run_analysis(project=project, study='extended')
            self.assertEqual(result['rows'], [row])
            self.assertEqual([json.loads(s) for s in checkpoint.read_text().splitlines() if s.strip()], [row])
            solver.assert_not_called()


if __name__ == '__main__':
    unittest.main()
