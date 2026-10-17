"""Phase-only input preserves PGM errors, including wrapped and nonunit-c data."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.input_data import load_field_input, load_phase_input
from src import FAMILIES, balanced, pgm_error, probe_gram, state_vector_gram, allocations


class PhaseInputTests(unittest.TestCase):
    def test_gauge_invariance_with_wrapping_and_arbitrary_state(self):
        rng=np.random.default_rng(41)
        fields=rng.normal(size=(4,3))*7
        c=2.3
        rows=[]
        for i in range(4):
            for j in range(i+1,4):
                for m in range(3):
                    rows.append((i,j,m,float(np.angle(np.exp(1j*c*(fields[i,m]-fields[j,m]))))))
        psi=rng.normal(size=8)+1j*rng.normal(size=8)
        psi/=np.linalg.norm(psi)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.csv'
            with path.open('w',newline='') as stream:
                writer=csv.writer(stream); writer.writerow(['i','j','m','r_ijm']); writer.writerows(rows)
            relative=load_field_input(path,c=c)
            np.testing.assert_array_equal(relative[0],0)
            for n in allocations(3,3):
                for family in FAMILIES:
                    before=pgm_error(probe_gram(fields,n,family,c=c))
                    after=pgm_error(probe_gram(relative,n,family,c=c))
                    self.assertAlmostEqual(before,after,places=7)
                self.assertAlmostEqual(pgm_error(state_vector_gram(fields,n,psi,c=c)),pgm_error(state_vector_gram(relative,n,psi,c=c)),places=7)

    def test_signed_phases_preserve_wrapped_pairs_and_include_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.csv'
            path.write_text('i,j,m,r_ijm\n0,1,0,2.5\n0,2,0,-2.5\n1,2,0,'+str(2*np.pi-5)+'\n')
            fields, r_ijm=load_phase_input(path,c=2.)
            np.testing.assert_allclose(r_ijm[:,0],[2.5,-2.5,2*np.pi-5])
            self.assertAlmostEqual(abs(2*(fields[1,0]-fields[2,0])),5.)
            self.assertLess(abs(r_ijm[2,0]),np.pi)

    def test_reject_invalid_phase_tables(self):
        valid=[(0,1,0,0.2),(0,2,0,0.4),(1,2,0,0.2)]
        invalid=[valid[:-1],valid+[valid[0]],[(0,1,0,float('nan'))],[(0,1,0,-np.pi)],[(1,0,0,0.2)],valid[:-1]+[(1,2,0,0.3)]]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.csv'
            for rows in invalid:
                with path.open('w',newline='') as stream:
                    writer=csv.writer(stream); writer.writerow(['i','j','m','r_ijm']); writer.writerows(rows)
                with self.assertRaises(ValueError):
                    load_field_input(path)


if __name__=='__main__':
    unittest.main()
