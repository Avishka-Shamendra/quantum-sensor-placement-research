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


ALGORITHM_POLICY = 'pgm_greedy_csv_rijm_score_vnd_v4_single_random'


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


def greedy_search(N, M, gram_for_allocation, *, time_limit=None):
    """Add each sensor at the location giving the lowest PGM error.

    Exact ties retain the smallest location index.
    """
    N, M = integer(N, 'N'), integer(M, 'M', 1)
    start = perf_counter()
    n = (0,) * M
    evaluations = 0

    def evaluate(candidate):
        nonlocal evaluations
        if time_limit is not None and perf_counter()-start > time_limit:
            raise TimeoutError('allocation time limit exceeded')
        value = pgm_error(gram_for_allocation(candidate))
        evaluations += 1
        return value

    if N == 0:
        error = evaluate(n)
    for _ in range(N):
        best, error = None, float('inf')
        for location in range(M):
            candidate = list(n)
            candidate[location] += 1
            candidate = tuple(candidate)
            value = evaluate(candidate)
            if value < error:
                best, error = candidate, value
        n = best
    return SearchResult(n, error, evaluations, perf_counter()-start, 'greedy_complete')


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
