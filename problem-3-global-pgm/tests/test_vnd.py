"""Phase-score VND convergence and independent initialization."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.search import vnd_search,separation_score

class VNDTests(unittest.TestCase):
    def test_initialization_and_no_pgm(self):
        with patch('src.search.pgm_error',side_effect=AssertionError('no PGM during allocation')):
            self.assertEqual(vnd_search(8,3,np.ones((3,3))).allocation,(3,3,2))
            self.assertEqual(vnd_search(0,3,np.ones((3,3))).allocation,(0,0,0))

    def test_converges_in_all_transfer_neighborhoods(self):
        d=np.random.default_rng(17).uniform(0,.7,(6,4))
        result=vnd_search(8,4,d)
        self.assertGreaterEqual(result.phase_score,separation_score((2,2,2,2),d))
        self.assertEqual(sum(result.allocation),8)
        for t in (1,2,3):
            for source in range(4):
                for destination in range(4):
                    n=list(result.allocation)
                    if source==destination or n[source]<t:continue
                    n[source]-=t;n[destination]+=t
                    self.assertLessEqual(separation_score(n,d),result.phase_score+1e-12)

    def test_score_saturation_and_validation(self):
        self.assertAlmostEqual(separation_score((2,1),np.array([[2.,0.],[0.,1.]])),(np.pi+1)/2)
        with self.assertRaises(ValueError):vnd_search(2,2,[[-1,1]])
        with self.assertRaises(TimeoutError):vnd_search(2,2,[[1,1]],time_limit=-1)

if __name__=='__main__':unittest.main()
