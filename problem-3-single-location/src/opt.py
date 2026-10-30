"""Exact joint SDP for Dicke-sector weights and the discrimination POVM."""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .sdp import (hermitian, measurement_success, phases, psd_residual,
                  recover_povm, solve, solver_list, validate_bounds, validate_settings)


@dataclass
class JointResult:
    error: float
    p: np.ndarray
    povm: list
    diagnostics: dict


def opt_state(phi, n, cfg):
    """Maximize sum_k Tr(a_k a_k^H X_k)/K with sum_k X_k = diag(p)."""
    phi = phases(phi, n)
    k, d = len(phi), n + 1
    eps, tolerance = cfg['sdp_eps'], cfg.get('tolerance', 1e-5)
    validate_settings(eps, tolerance)
    a = np.exp(-.5j * phi[:, None] * (n-2*np.arange(d)))
    matrices = [hermitian(np.outer(v, v.conj())) / k for v in a]
    p = cp.Variable(d, nonneg=True)
    x = [cp.Variable((d, d), hermitian=True) for _ in range(k)]
    primal = cp.Problem(cp.Maximize(sum(cp.real(cp.trace(b @ q)) for b, q in zip(matrices, x))),
                        [q >> 0 for q in x] + [sum(x) == cp.diag(p), cp.sum(p) == 1])
    y = cp.Variable((d, d), hermitian=True)
    t = cp.Variable()
    dual = cp.Problem(cp.Minimize(t), [y - b >> 0 for b in matrices] + [cp.real(cp.diag(y)) <= t])
    failures = []
    for solver in solver_list(cfg['solver']):
        try:
            primal_value = solve(primal, solver, eps)
            dual_value = solve(dual, solver, eps)
            weights = np.asarray(p.value, dtype=float)
            blocks = [hermitian(q.value) for q in x]
            if not np.isfinite(weights).all() or not all(np.isfinite(q).all() for q in blocks) or not np.isfinite(y.value).all():
                raise ValueError('Nonfinite joint SDP solution')
            primal_residual = max(abs(weights.sum()-1), max(0., -weights.min()),
                np.linalg.norm(sum(blocks)-np.diag(weights), 2), max(psd_residual(q) for q in blocks))
            dual_residual = max(max(psd_residual(y.value-b) for b in matrices),
                                max(0., np.diag(y.value).real.max()-dual_value))
            if max(primal_residual, dual_residual) > tolerance:
                raise ValueError('Joint SDP feasibility residual exceeds tolerance')
            weights = np.maximum(weights, 0)
            weights[weights < 1e-10] = 0
            if weights.sum() <= 0:
                raise ValueError('Joint SDP returned zero total weight')
            weights /= weights.sum()
            povm = recover_povm(blocks, weights)
            states = np.sqrt(weights)[None, :] * a
            success = measurement_success(states, povm)
            shift = max(psd_residual(y.value-b) for b in matrices)
            upper = min(1., float(np.diag(y.value).real.max()) + shift)
            validate_bounds(success, upper, k, tolerance)
            if abs(success-primal_value) > tolerance:
                raise ValueError('POVM recovery changed the objective beyond tolerance')
            povm_residual = max(np.linalg.norm(sum(povm)-np.eye(d), 2), max(psd_residual(q) for q in povm))
            if povm_residual > tolerance:
                raise ValueError('Recovered POVM is infeasible')
            diagnostics = dict(primal_residual=float(primal_residual), dual_residual=float(dual_residual),
                povm_residual=float(povm_residual), optimality_gap=max(0., upper-success),
                primal_success=primal_value, success_upper_bound=upper,
                sdp_status=primal.status, dual_status=dual.status, solver=solver,
                recovery_difference=abs(success-primal_value))
            return JointResult(float(np.clip(1-success, 0., 1-1/k)), weights, povm, diagnostics)
        except (cp.error.SolverError, ValueError, np.linalg.LinAlgError) as error:
            failures.append(f'{solver}: {error}')
    raise RuntimeError('Joint SDP validation failed; ' + ' | '.join(failures))
