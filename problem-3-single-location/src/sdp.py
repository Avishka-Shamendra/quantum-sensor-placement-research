"""Validated primal and dual SDPs for minimum-error discrimination."""

import cvxpy as cp
import numpy as np


def hermitian(matrix):
    matrix = np.asarray(matrix, dtype=complex)
    return (matrix + matrix.conj().T) / 2


def psd_residual(matrix):
    return max(0., -float(np.linalg.eigvalsh(hermitian(matrix)).min()))


def solver_list(name):
    installed = set(cp.installed_solvers())
    if name != 'auto':
        if name not in installed:
            raise ValueError(f"CVXPY solver '{name}' is not installed")
        return [name]
    found = [s for s in ('CLARABEL', 'SCS', 'CVXOPT') if s in installed]
    if not found:
        raise RuntimeError('No SDP-capable CVXPY solver is installed')
    return found


def solve(problem, solver, eps):
    options = dict(solver=solver, warm_start=False, verbose=False)
    if solver == 'CLARABEL':
        options.update(tol_gap_abs=eps, tol_gap_rel=eps, tol_feas=eps, max_iter=1000)
    elif solver == 'SCS':
        options.update(eps=eps, max_iters=100_000)
    elif solver == 'CVXOPT':
        options.update(abstol=eps, reltol=eps, feastol=eps)
    value = problem.solve(**options)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or value is None or not np.isfinite(value):
        raise ValueError(f'SDP status: {problem.status}')
    return float(value)


def validate_settings(eps, tolerance):
    if not np.isfinite([eps, tolerance]).all() or not 0 < eps < tolerance < 1:
        raise ValueError('Require 0 < solver accuracy < validation tolerance < 1')


def phases(phi, n):
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 1 or len(phi) < 2 or not np.isfinite(phi).all():
        raise ValueError('Expected at least two finite field phases')
    if type(n) is not int or n < 1:
        raise ValueError('Sensor count must be a positive integer')
    return phi


def recover_povm(matrices, weights=None):
    """Use the support pseudoinverse and complete the unused subspace uniformly."""
    d = len(matrices[0])
    support = np.ones(d, dtype=bool) if weights is None else np.asarray(weights) > 0
    indices = np.flatnonzero(support)
    blocks = []
    for matrix in matrices:
        block = hermitian(matrix)[np.ix_(indices, indices)]
        eigenvalues, vectors = np.linalg.eigh(block)
        blocks.append(hermitian((vectors * np.maximum(eigenvalues, 0)) @ vectors.conj().T))
    # The sum equals diag(p) in exact arithmetic; whitening also removes roundoff.
    eigenvalues, vectors = np.linalg.eigh(hermitian(sum(blocks)))
    if eigenvalues.min() <= np.finfo(float).eps * max(1., eigenvalues.max()):
        raise ValueError('Singular measurement on the retained support')
    inverse = (vectors * (1 / np.sqrt(eigenvalues))) @ vectors.conj().T
    complement = np.diag((~support).astype(float)) / len(matrices)
    povm = []
    for block in blocks:
        matrix = complement.astype(complex).copy()
        matrix[np.ix_(indices, indices)] = hermitian(inverse @ block @ inverse)
        povm.append(matrix)
    return povm


def measurement_success(states, povm):
    return float(np.mean([np.vdot(state, measurement @ state).real
                          for state, measurement in zip(states, povm)]))


def validate_bounds(success, upper, k, tolerance):
    if not np.isfinite([success, upper]).all():
        raise ValueError('Nonfinite discrimination probability')
    if not 1 / k - tolerance <= success <= 1 + tolerance:
        raise ValueError('Success probability outside discrimination bounds')
    if upper < success - tolerance or upper - success > tolerance:
        raise ValueError(f'Primal-dual gap exceeds tolerance: {upper-success:.3g}')


class DiscSDP:
    """Optimal POVM for fixed sector weights, checked against its dual."""

    def __init__(self, k, n, solver='auto', eps=1e-8, tolerance=1e-5):
        validate_settings(eps, tolerance)
        self.k, self.n, self.d = k, n, n + 1
        self.eps, self.tolerance = eps, tolerance
        self.solvers = solver_list(solver)
        self.rho = [cp.Parameter((self.d, self.d), hermitian=True) for _ in range(k)]
        self.y = cp.Variable((self.d, self.d), hermitian=True)
        self.prob = cp.Problem(cp.Minimize(cp.real(cp.trace(self.y))),
                               [self.y - r / k >> 0 for r in self.rho])
        self.measurements = [cp.Variable((self.d, self.d), hermitian=True) for _ in range(k)]
        self.primal = cp.Problem(cp.Maximize(sum(cp.real(cp.trace(r @ e)) / k
            for r, e in zip(self.rho, self.measurements))),
            [e >> 0 for e in self.measurements] + [sum(self.measurements) == np.eye(self.d)])

    def error(self, p, phi):
        from .states import validate_weights
        p = validate_weights(p, self.n)
        phi = phases(phi, self.n)
        if len(phi) != self.k:
            raise ValueError('Incompatible phase vector')
        states = np.sqrt(p)[None, :] * np.exp(1j * phi[:, None] * np.arange(self.d))
        for param, state in zip(self.rho, states):
            param.value = hermitian(np.outer(state, state.conj()))
        failures = []
        for solver in self.solvers:
            try:
                solve(self.primal, solver, self.eps)
                solve(self.prob, solver, self.eps)
                matrices = [hermitian(e.value) for e in self.measurements]
                if not all(np.isfinite(e).all() for e in matrices) or not np.isfinite(self.y.value).all():
                    raise ValueError('Nonfinite SDP matrices')
                primal_residual = max(np.linalg.norm(sum(matrices)-np.eye(self.d), 2),
                                      max(psd_residual(e) for e in matrices))
                dual_residual = max(psd_residual(self.y.value-r.value/self.k) for r in self.rho)
                if max(primal_residual, dual_residual) > self.tolerance:
                    raise ValueError('SDP feasibility residual exceeds tolerance')
                self.povm = recover_povm(matrices)
                povm_residual = max(np.linalg.norm(sum(self.povm)-np.eye(self.d), 2),
                                    max(psd_residual(e) for e in self.povm))
                if povm_residual > self.tolerance:
                    raise ValueError("Recovered POVM is infeasible")
                success = measurement_success(states, self.povm)
                upper = min(1., float(np.trace(self.y.value).real) + self.d * dual_residual)
                validate_bounds(success, upper, self.k, self.tolerance)
                self.diagnostics = dict(primal_residual=float(primal_residual), dual_residual=float(dual_residual),
                    optimality_gap=max(0., upper-success), povm_residual=float(povm_residual),
                    success_upper_bound=upper, sdp_status=self.primal.status,
                    dual_status=self.prob.status, solver=solver)
                return float(np.clip(1-success, 0., 1-1/self.k))
            except (cp.error.SolverError, ValueError, np.linalg.LinAlgError) as error:
                failures.append(f'{solver}: {error}')
        raise RuntimeError('SDP validation failed; ' + ' | '.join(failures))
