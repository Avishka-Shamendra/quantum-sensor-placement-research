"""One parallel encoding and one joint equal-prior pure-state PGM."""
from numbers import Integral
import numpy as np


def integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def validate(fields, allocation, c=1.0):
    raw = np.asarray(fields)
    if np.iscomplexobj(raw):
        raise ValueError("fields must be real")
    f = np.asarray(fields, dtype=float)
    if f.ndim != 2 or min(f.shape) < 1 or not np.isfinite(f).all():
        raise ValueError("fields must be a finite nonempty K by M matrix")
    n = tuple(integer(v, "n_m") for v in allocation)
    if len(n) != f.shape[1]:
        raise ValueError("allocation must have M entries")
    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be positive and finite")
    return f, n


def pgm_error(gram):
    """Exact PGM error; reject invalid Grams, clip only roundoff negatives."""
    g = np.asarray(gram, dtype=complex)
    if g.ndim != 2 or g.shape[0] == 0 or g.shape[0] != g.shape[1]:
        raise ValueError("Gram matrix must be nonempty and square")
    if not np.isfinite(g).all() or not np.allclose(g, g.conj().T, atol=1e-10, rtol=0):
        raise ValueError("Gram matrix must be finite and Hermitian")
    if not np.allclose(np.diag(g), 1, atol=1e-10, rtol=0):
        raise ValueError("Gram diagonal must equal one")
    w, v = np.linalg.eigh((g + g.conj().T) / 2)
    if w.min() < -1e-10 * max(1, np.max(np.abs(w))):
        raise ValueError("Gram matrix is not positive semidefinite")
    # Exactly collinear states have error 1-1/K. Detect their rank-one Gram
    # at machine precision, avoiding sqrt(roundoff) spurious sensing gains.
    reference = g[:, :1] @ g[:1, :]
    if np.max(np.abs(g-reference)) <= 8*np.finfo(float).eps:
        return 1.0 - 1.0/len(g)
    diagonal = np.abs(v)**2 @ np.sqrt(np.maximum(w, 0))
    return float(np.clip(1 - np.mean(diagonal**2), 0, 1))


def state_vector_gram(fields, allocation, state, c=1.0):
    """Any normalized pure state; labeled qubits grouped by increasing m.

    state is a vector of length 2**N or a callback state(tuple(n_m)).
    Computational basis is big-endian, Z|0>=|0>. Costs O(K*2**N).
    """
    f, n = validate(fields, allocation, c)
    psi = np.asarray(state(n) if callable(state) else state, dtype=complex)
    N = sum(n)
    if psi.shape != (2**N,) or not np.isfinite(psi).all():
        raise ValueError("state must be a finite vector of length 2**N")
    if not np.isclose(np.vdot(psi, psi), 1, atol=1e-10, rtol=0):
        raise ValueError("input state must be normalized")
    # Iterating basis strings avoids fixed-width integer overflow in indexing.
    z = np.array([[1 - 2*int(b) for b in format(s, f'0{N}b')]
                  for s in range(2**N)]) if N else np.empty((1, 0))
    phases = np.repeat(c*f, n, axis=1) @ z.T
    outputs = np.exp(-0.5j * phases) * psi
    return outputs.conj() @ outputs.T
