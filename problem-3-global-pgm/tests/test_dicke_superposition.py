"""Check the intended pure superposition against explicit computational states."""
import sys
from pathlib import Path
from math import comb
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import probe_gram, state_vector_gram, pgm_error, allocations


def superposition(n):
    return np.array([1/np.sqrt((n+1)*comb(n,b.bit_count())) for b in range(2**n)])


class DickeSuperpositionTests(unittest.TestCase):
    def test_explicit_pure_states_all_allocations(self):
        fields=np.random.default_rng(99).normal(size=(4,3))*4
        for N in range(6):
            global_state=superposition(N)
            self.assertAlmostEqual(np.vdot(global_state,global_state),1)
            for n in allocations(N,3):
                local=np.ones(1)
                for count in n:local=np.kron(local,superposition(count))
                for family,state in [('local_dicke',local),('global_dicke',global_state)]:
                    np.testing.assert_allclose(probe_gram(fields,n,family,c=1.7),state_vector_gram(fields,n,state,c=1.7),atol=2e-12)

    def test_sector_weights_and_sensing(self):
        psi=superposition(2)
        np.testing.assert_allclose(psi,[1/np.sqrt(3),1/np.sqrt(6),1/np.sqrt(6),1/np.sqrt(3)])
        for family in ['local_dicke','global_dicke']:
            # n=2, common phase gap 2*pi/3: (1+2*cos(gap))/3=0.
            self.assertAlmostEqual(pgm_error(probe_gram([[0],[2*np.pi/3]],(2,),family)),0)
            self.assertAlmostEqual(pgm_error(probe_gram([[0,0],[0,2*np.pi/3]],(2,0),family)),0.5)
            self.assertAlmostEqual(pgm_error(probe_gram([[0,0],[0,2*np.pi/3]],(0,2),family)),0)
            with self.assertRaises(TypeError):probe_gram([[0],[1]],(2,),family,d=1)

    def test_empty_blocks_single_qubit_and_coincident_locations(self):
        f=np.array([[0.1,0.1],[1.3,1.3],[-0.8,-0.8]])
        for family in ['local_dicke','global_dicke']:
            np.testing.assert_allclose(probe_gram(f,(0,0),family),np.ones((3,3)))
            np.testing.assert_allclose(probe_gram(f,(1,0),family),probe_gram(f,(1,0),'product'))
        np.testing.assert_allclose(probe_gram(f,(2,3),'global_dicke'),probe_gram(f,(5,0),'local_dicke'),atol=1e-12)


if __name__=='__main__':unittest.main()
