"""Flat-checkpoint identities, without running numerical optimizers."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import evaluation_signature


class FlatCheckpointTests(unittest.TestCase):
    def test_identity_tracks_inputs_and_settings_not_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary)/'a', Path(temporary)/'b'
            for directory in (first, second):
                directory.mkdir()
                for name in ('manifest.json', 'set.csv', 'r.csv'):
                    (directory/name).write_text('same data')
            cfg = {'solver': 'auto', 'sdp_eps': 1e-8}
            versions = {'solver_version': 'test'}
            identity = evaluation_signature(first, cfg, versions)
            self.assertEqual(identity, evaluation_signature(second, cfg, versions))
            self.assertNotEqual(identity, evaluation_signature(first, dict(cfg, sdp_eps=1e-6), versions))
            (second/'set.csv').write_text('changed input')
            self.assertNotEqual(identity, evaluation_signature(second, cfg, versions))


if __name__ == '__main__':
    unittest.main()
