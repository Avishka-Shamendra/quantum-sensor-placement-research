"""Allocation searches independent of the supplied pure-state family."""
from dataclasses import dataclass, asdict
from math import comb
from time import perf_counter
import numpy as np
from .pgm import integer, pgm_error


@dataclass(frozen=True)
class SearchResult:
    allocation: tuple[int, ...]
    pgm_error: float
    evaluations: int
    runtime_seconds: float
    status: str

    def to_dict(self):
        return asdict(self)


def allocations(N, M):
    N, M = integer(N, 'N'), integer(M, 'M', 1)
    if M == 1:
        yield (N,)
    else:
        for first in range(N+1):
            for tail in allocations(N-first, M-1):
                yield (first, *tail)


def exhaustive_search(N, M, gram_for_allocation, *, max_allocations=1_000_000):
    """Enumerate all C(N+M-1,M-1) allocations; never return a partial optimum."""
    N, M = integer(N, 'N'), integer(M, 'M', 1)
    count = comb(N+M-1, M-1)
    if max_allocations is not None and count > integer(max_allocations, 'max_allocations', 1):
        raise ValueError(f"{count} allocations exceed max_allocations; increase explicitly")
    start = perf_counter()
    best, error = None, float('inf')
    for n in allocations(N, M):
        value = pgm_error(gram_for_allocation(n))
        if value < error:
            best, error = n, value
    return SearchResult(best, error, count, perf_counter()-start, 'optimal')


ALGORITHM_POLICY = 'csv_rijm_pair_greedy_score_vnd_v3_single_random'


@dataclass(frozen=True)
class AllocationResult:
    allocation: tuple[int, ...]
    phase_score: float
    evaluations: int
    runtime_seconds: float
    status: str

    def to_dict(self):
        return asdict(self)


def _validate_phases(N, M, r_ijm):
    N, M = integer(N, 'N'), integer(M, 'M', 1)
    r_ijm = np.asarray(r_ijm, dtype=float)
    if r_ijm.ndim != 2 or r_ijm.shape[1] != M or not len(r_ijm) or not np.all(np.isfinite(r_ijm)) or np.any(r_ijm <= -np.pi) or np.any(r_ijm > np.pi):
        raise ValueError('r_ijm must be a finite pair by M array in (-pi, pi]')
    return N, M, r_ijm


def separation_score(allocation, r_ijm):
    return float(np.minimum(np.pi, np.abs(r_ijm) @ np.asarray(allocation)).mean())


def greedy_search(N, M, r_ijm, *, time_limit=None):
    """Serve the least-separated pair; ties use smaller pair maximum then pair order.

    Location ties use the smallest index. No Gram matrix or PGM is evaluated.
    """
    N, M, r_ijm = _validate_phases(N, M, r_ijm)
    start = perf_counter()
    n = np.zeros(M, dtype=int)
    accumulated = np.zeros(len(r_ijm))
    maxima = np.abs(r_ijm).max(axis=1)
    for _ in range(N):
        if time_limit is not None and perf_counter()-start > time_limit:
            raise TimeoutError('allocation time limit exceeded')
        pair = min(range(len(r_ijm)), key=lambda i: (accumulated[i], maxima[i], i))
        location = int(np.argmax(np.abs(r_ijm[pair])))
        n[location] += 1
        accumulated += np.abs(r_ijm[:, location])
    return AllocationResult(tuple(map(int,n)), separation_score(n,r_ijm), N,
                            perf_counter()-start, 'greedy_complete')


def vnd_search(N, M, r_ijm, *, tolerance=1e-12, time_limit=None):
    """Best-improvement phase-score descent from deterministic near-uniform counts.

    Scan transfers of 1, 2, 3 sensors, restarting at 1 after improvement.
    Ties within tolerance retain the first source/destination in index order.
    Converge without an iteration cap; no Gram matrix or PGM is evaluated.
    """
    N, M, r_ijm = _validate_phases(N, M, r_ijm)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and nonnegative')
    start = perf_counter()
    n = tuple(N//M + (m < N % M) for m in range(M))
    cache = {}
    def score(candidate):
        if time_limit is not None and perf_counter()-start > time_limit:
            raise TimeoutError('allocation time limit exceeded')
        if candidate not in cache:
            cache[candidate] = separation_score(candidate, r_ijm)
        return cache[candidate]
    value = score(n)
    neighborhood = 0
    while neighborhood < 3:
        size = neighborhood+1
        best, best_value = n, value
        for source in range(M):
            if n[source] < size:
                continue
            for destination in range(M):
                if source == destination:
                    continue
                candidate = list(n)
                candidate[source] -= size
                candidate[destination] += size
                candidate = tuple(candidate)
                candidate_value = score(candidate)
                if candidate_value > best_value + tolerance:
                    best, best_value = candidate, candidate_value
        if best == n:
            neighborhood += 1
        else:
            n, value = best, best_value
            neighborhood = 0
    return AllocationResult(n, value, len(cache), perf_counter()-start, 'local_optimum')
