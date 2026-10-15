"""Dicke-sector states and the original Gram-matrix PGM evaluator."""
from math import comb
import numpy as np

def pgm_error(phi, p):
    phi = np.asarray(phi, dtype=float)
    p = np.asarray(p, dtype=float)
    if np.min(p) < -1e-10 or p.sum() <= 0:
        raise ValueError('Invalid Dicke-weight probabilities')
    p = np.clip(p, 0.0, None)
    p /= p.sum()
    w = np.arange(len(p))
    states = np.sqrt(p)[None, :] * np.exp(1j * phi[:, None] * w[None, :])
    gram = states.conj() @ states.T
    gram = (gram + gram.conj().T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(gram)
    if eigvals.min() < -1e-8:
        raise ValueError(f'Gram matrix is not positive semidefinite: {eigvals.min()}')
    root = (eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))) @ eigvecs.conj().T
    success = np.sum(np.abs(np.diag(root)) ** 2) / len(phi)
    return float(np.clip(1.0 - success.real, 0.0, 1.0))


def prescribed_states(n):
    ghz = np.zeros(n + 1)
    ghz[[0, n]] = 0.5
    product = np.array([comb(n, w) for w in range(n + 1)], dtype=float) / (2.0**n)
    equal = np.full(n + 1, 1.0 / (n + 1))
    return {'ghz': ghz, 'product': product, 'equal': equal}

def validate_weights(p, n):
    p = np.asarray(p, dtype=float)
    if p.shape != (n + 1,) or not np.isfinite(p).all():
        raise ValueError('Invalid state dimension or nonfinite weights')
    if p.min() < 0 or abs(p.sum() - 1) > 1e-8:
        raise ValueError('Dicke-sector probabilities violate the simplex')
    return p / p.sum()

