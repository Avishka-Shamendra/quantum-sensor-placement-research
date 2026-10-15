"""Tiny in-memory physics and search checks; no benchmark datasets generated."""
import sys
import unittest
from pathlib import Path
from math import comb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import allocations, balanced, exhaustive_search, greedy_search, pgm_error, fixed_excitation_probe_gram as probe_gram, state_vector_gram


def dicke(n, d):
    return np.array([int(s.bit_count() == d)/np.sqrt(comb(n, d)) for s in range(2**n)])


def ghz(n):
    if n == 0:
        return np.ones(1)
    state = np.zeros(2**n)
    state[0] = state[-1] = 1/np.sqrt(2)
    return state


class GlobalPGMTests(unittest.TestCase):
    def test_pgm_limits_and_invalid_grams(self):
        for K in (1, 2, 5):
            self.assertAlmostEqual(pgm_error(np.ones((K, K))), 1-1/K, places=7)
            self.assertAlmostEqual(pgm_error(np.eye(K)), 0)
        for bad in ([[1, 2], [2, 1]], [[1, 1j], [1j, 1]], [[2]]):
            with self.assertRaises(ValueError):
                pgm_error(bad)

    def test_all_families_against_explicit_states(self):
        f = np.random.default_rng(123).normal(size=(4, 3))*5
        for n in allocations(4, 3):
            states = {'product': np.ones(16)/4, 'global_ghz': ghz(4), 'global_dicke': dicke(4, 2)}
            for family, block in [('local_ghz', ghz), ('local_dicke', lambda count: dicke(count, count//2))]:
                state = np.ones(1)
                for count in n:
                    state = np.kron(state, block(count))
                states[family] = state
            for family, state in states.items():
                with self.subTest(n=n, family=family):
                    gram = probe_gram(f, n, family, c=1.3, d=2, local_d=balanced)
                    np.testing.assert_allclose(gram, state_vector_gram(f, n, state, c=1.3), atol=1e-12)
                    if family == 'local_dicke':
                        self.assertAlmostEqual(pgm_error(gram), 0.75, places=14)

    def test_global_dicke_extremes_vacuum_and_relative_phases(self):
        f = [[0, 0], [0, np.pi]]
        for N in (0, 1, 4):
            for d in range(N+1):
                n = (N//2, N-N//2)
                np.testing.assert_allclose(probe_gram(f, n, 'global_dicke', d=d), state_vector_gram(f, n, dicke(N, d)), atol=1e-12)
        self.assertAlmostEqual(pgm_error(probe_gram(f, (1, 1), 'global_dicke', d=1)), 0)
        for family in ('product', 'local_ghz', 'global_ghz', 'local_dicke', 'global_dicke'):
            self.assertAlmostEqual(pgm_error(probe_gram(f, (0, 0), family, d=0, local_d=balanced)), 0.5)
        with self.assertRaises(ValueError):
            probe_gram(f, (1, 1), 'global_dicke')
        with self.assertRaises(ValueError):
            probe_gram(f, (1, 1), 'local_dicke')

    def test_arbitrary_complex_state_and_search(self):
        rng = np.random.default_rng(2)
        f = rng.normal(size=(3, 3))
        state = rng.normal(size=8)+1j*rng.normal(size=8)
        state /= np.linalg.norm(state)
        gram = lambda n: state_vector_gram(f, n, lambda counts: state[:2**sum(counts)]/np.linalg.norm(state[:2**sum(counts)]))
        exact = exhaustive_search(3, 3, gram)
        from src.search import phase_differences
        greedy = greedy_search(3, 3, phase_differences(f))
        self.assertEqual(exact.evaluations, comb(5, 2))
        self.assertEqual(sum(exact.allocation), 3)
        self.assertLessEqual(exact.pgm_error, pgm_error(gram(greedy.allocation))+1e-12)
        with self.assertRaises(ValueError):
            exhaustive_search(3, 3, gram, max_allocations=2)
        with self.assertRaises(ValueError):
            state_vector_gram(f, (1, 1, 1), state*2)

    def test_binary_analytic_error(self):
        theta = 0.71
        actual = pgm_error(probe_gram([[0], [theta]], (1,), 'product'))
        self.assertAlmostEqual(actual, (1-abs(np.sin(theta/2)))/2)


if __name__ == '__main__':
    unittest.main()
