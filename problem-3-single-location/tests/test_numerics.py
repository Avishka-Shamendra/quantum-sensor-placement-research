"""Fast algebraic checks; no optimizer or SDP solver is invoked."""
import unittest
import numpy as np
from single_location.states import pgm_error, prescribed_states
from single_location.states import validate_weights


class StateChecks(unittest.TestCase):
    def test_known_ensembles(self):
        self.assertAlmostEqual(pgm_error(np.zeros(3), np.ones(3)/3), 2/3)
        self.assertAlmostEqual(pgm_error(2*np.pi*np.arange(3)/3, np.ones(3)/3), 0)

    def test_independent_pgm_operator(self):
        phi = np.array([.1, .4, 1.3, 2.])
        for p in prescribed_states(6).values():
            validate_weights(p, 6)
            states = np.sqrt(p)[:, None] * np.exp(1j*np.arange(7)[:, None]*phi)
            density = states @ states.conj().T / 4
            vals, vecs = np.linalg.eigh(density)
            inv = np.zeros_like(vals)
            mask = vals > 1e-12
            inv[mask] = 1/np.sqrt(vals[mask])
            inverse_root = (vecs*inv) @ vecs.conj().T
            success = sum(abs(s.conj() @ inverse_root @ s)**2 for s in states.T)/16
            np.testing.assert_allclose(pgm_error(phi, p), 1-success, atol=1e-7, rtol=0)

    def test_invalid_weights(self):
        with self.assertRaises(ValueError):
            validate_weights([-.1, 1.1], 1)


if __name__ == '__main__':
    unittest.main()
