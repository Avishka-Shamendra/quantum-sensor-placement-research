"""Read Problem 2 phase gaps and reconstruct a consistent relative-field gauge."""
import csv
import numpy as np


def load_field_input(path, *, c=1.0):
    """Return relative fields with f_0m=0 and c*f_km=-r_0km.

    Absolute fields cannot be recovered from gaps. A common phase rotation at
    each location is hypothesis-independent and leaves every Gram overlap
    invariant. Individual 2*pi lifts change only output global phases, hence
    preserve PGM error for any fixed pure input state. Validate all remaining
    pairs modulo 2*pi instead of using independently wrapped Gram entries.
    Pass the same c here and to the encoding evaluator; r already includes c.
    """
    if not np.isfinite(c) or c <= 0:
        raise ValueError('c must be positive and finite')
    entries = {}
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['i', 'j', 'm', 'r_ijm']:
            raise ValueError('expected columns i,j,m,r_ijm')
        for row in reader:
            i, j, m = int(row['i']), int(row['j']), int(row['m'])
            value = float(row['r_ijm'])
            if not 0 <= i < j or m < 0 or not np.isfinite(value) or not -np.pi < value <= np.pi:
                raise ValueError('invalid phase-gap entry')
            if (i, j, m) in entries:
                raise ValueError('duplicate phase-gap entry')
            entries[i, j, m] = value
    if not entries:
        raise ValueError('empty phase input')
    K = max(j for i, j, m in entries)+1
    M = max(m for i, j, m in entries)+1
    if len(entries) != K*(K-1)//2*M:
        raise ValueError('every canonical i,j,m entry must appear exactly once')
    phases = np.zeros((K, M))
    for k in range(1, K):
        for m in range(M):
            phases[k, m] = -entries[0, k, m]
    for (i, j, m), value in entries.items():
        residual = np.angle(np.exp(1j*(phases[i,m]-phases[j,m]-value)))
        if abs(residual) > 1e-10:
            raise ValueError('phase gaps are inconsistent across field pairs')
    return phases/c
