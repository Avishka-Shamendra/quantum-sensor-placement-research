"""Efficient Gram evaluators for the five fixed probe families."""
from math import comb
import numpy as np
from .pgm import integer, validate

FAMILIES = ('product', 'local_ghz', 'global_ghz', 'local_dicke', 'global_dicke')


def balanced(n, m):
    """Explicit local excitation policy d_m=floor(n_m/2), including vacuum."""
    return n // 2


def fixed_excitation_probe_gram(fields, allocation, family, *, c=1.0, d=None, local_d=None):
    f, n = validate(fields, allocation, c)
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    # Unwrapped r_ijm=c*(f_i-f_j); same sign as Problems 1 and 2.
    r = c * (f[:, None, :] - f[None, :, :])
    K = len(f)
    if family == 'product':
        return np.prod(np.cos(r / 2)**np.array(n), axis=2)
    if family == 'local_ghz':
        return np.prod(np.cos(r * np.array(n) / 2), axis=2)
    if family == 'global_ghz':
        return np.cos(r @ np.array(n) / 2)
    if family == 'local_dicke':
        if not callable(local_d):
            raise ValueError("local_d must explicitly specify d_m via local_d(n_m, m)")
        excitations = [integer(local_d(count, m), 'd_m') for m, count in enumerate(n)]
        if any(dm > count for dm, count in zip(excitations, n)):
            raise ValueError("each d_m must be <= n_m (and zero in empty blocks)")
        return np.exp(0.5j * (r @ (np.array(n) - 2*np.array(excitations))))
    N = sum(n)
    d = integer(d, 'd')
    if d > N:
        raise ValueError("d must be <= N")
    coefficients = _dicke_coefficients(r, n, d)
    return np.exp(0.5j*(r @ np.array(n))) * coefficients[:, :, d]


def _dicke_coefficients(r, n, max_d):
    """Normalized fixed-sector overlaps, for all excitation counts <= max_d."""
    K = r.shape[0]
    coefficients = np.ones((K, K, 1), dtype=complex)
    used = 0
    for m, count in enumerate(n):
        new = np.zeros((K, K, min(max_d, used + count) + 1), dtype=complex)
        phases = [np.exp(-1j*q*r[:, :, m]) for q in range(min(count,max_d)+1)]
        for total in range(new.shape[2]):
            for q in range(max(0, total-used), min(count, total)+1):
                weight = (comb(used, total-q)*comb(count, q)) / comb(used+count, total)
                new[:, :, total] += weight * coefficients[:, :, total-q] * phases[q]
        coefficients = new
        used += count
    return coefficients


DICKE_POLICY = 'equal_amplitude_all_excitation_sectors'


def probe_gram(fields, allocation, family, *, c=1.0):
    """Main five-probe model: Dicke means sum_d |D_n^d>/sqrt(n+1).

    Each sector d has probability 1/(n+1); individual basis strings have
    probability 1/((n+1)*comb(n,d)). This is a coherent pure superposition,
    not a mixture. Diagonal Z encoding makes cross-sector overlap terms zero,
    so averaging sector overlaps evaluates this pure-state Gram exactly.
    Local blocks are tensor products; the global probe has one N-qubit block.
    """
    if family not in ('local_dicke', 'global_dicke'):
        return fixed_excitation_probe_gram(fields, allocation, family, c=c)
    f, n = validate(fields, allocation, c)
    r = c * (f[:, None, :] - f[None, :, :])
    if family == 'local_dicke':
        gram = np.ones((len(f),len(f)))
        for m, count in enumerate(n):
            # Finite sum avoids removable singularities of the sine-ratio form.
            gram *= sum(np.cos((count-2*d)*r[:,:,m]/2) for d in range(count+1))/(count+1)
        return gram
    coefficients = _dicke_coefficients(r, n, sum(n))
    return np.exp(0.5j*(r @ np.array(n))) * coefficients.mean(axis=2)
