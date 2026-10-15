"""Phase-only constructive Greedy selection and deterministic tie rules."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.search import greedy_search,phase_differences

class GreedyTests(unittest.TestCase):
    def test_least_pair_uses_smaller_maximum_then_location_index(self):
        d=np.array([[2.,0.],[0.,1.],[1.,1.]])
        with patch('src.search.pgm_error',side_effect=AssertionError('no PGM during allocation')):
            self.assertEqual(greedy_search(1,2,d).allocation,(0,1))
            self.assertEqual(greedy_search(2,2,d).allocation,(1,1))
        self.assertEqual(greedy_search(1,2,np.ones((3,2))).allocation,(1,0))
        self.assertEqual(greedy_search(0,2,d).allocation,(0,0))

    def test_actual_phase_differences_are_not_wrapped_or_log_overlaps(self):
        np.testing.assert_array_equal(phase_differences([[0,1],[4,-2]],c=2),[[8,6]])

if __name__=='__main__':unittest.main()
