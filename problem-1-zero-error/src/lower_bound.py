"""Integer sensor lower bound from Section 3.2.1, equation (14).

This necessary pairwise condition does not certify a shared global input state.
Use the original phase gaps, not the rounded single-location thresholds.
"""

from __future__ import annotations

from math import pi
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from phase_data import compute_nijm, principal_phase


def optimal_global_lower_bound(K, M, phase_gaps):
    """Solve min sum(n) subject to abs(r) @ n >= pi, integer n >= 0.

    Returns the integer optimum, allocation, and numerical solver diagnostics.
    Requires optimal solver termination with zero requested relative MIP gap;
    verifies integrality, objective/dual agreement and the rounded allocation's
    constraints to 1e-8 in units of pi. HiGHS feasibility tolerances are set to
    1e-9, tighter than this independent check (the default MIP tolerance is 1e-6).
    Raises instead of reporting an incumbent
    as an optimum. As with the phase inputs, the computation is floating point.
    """
    compute_nijm(K, M, phase_gaps)  # Shared shape and distinguishability checks.
    matrix = np.array([
        [abs(principal_phase(r)) / pi for r in phase_gaps[pair]]
        for pair in sorted(phase_gaps)
    ])
    # SciPy forwards these documented HiGHS options, but warns because they are
    # outside SciPy's short option allowlist. Suppress only that forwarding notice.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"Unrecognized options detected:.*passed to HiGHS verbatim",
            category=RuntimeWarning,
        )
        result = milp(
            c=np.ones(M), integrality=np.ones(M), bounds=Bounds(0, np.inf),
            constraints=LinearConstraint(matrix, 1, np.inf),
            options={
                "mip_rel_gap": 0.0,
                "mip_feasibility_tolerance": 1e-9,
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
            },
        )
    if result.status != 0 or not result.success or result.x is None:
        raise RuntimeError(f"Lower-bound ILP was not solved optimally: {result.message}")
    rounded = np.rint(result.x)
    if not np.all(np.isfinite(rounded)) or np.any(rounded < 0) or np.max(np.abs(result.x - rounded)) > 1e-8:
        raise RuntimeError("Lower-bound solver returned a noninteger allocation")
    total = int(rounded.sum())
    coverage = matrix @ rounded
    if np.min(coverage) < 1 - 1e-8:
        raise RuntimeError(
            "Rounded lower-bound allocation violates equation (14): "
            f"minimum normalized coverage={np.min(coverage):.16g}"
        )
    if (abs(result.fun - total) > 1e-6
            or abs(result.mip_dual_bound - total) > 1e-6
            or result.mip_gap > 1e-9):
        raise RuntimeError("Lower-bound optimality certificate did not close")
    return total, [int(n) for n in rounded], {
        "lb_solver_status": int(result.status),
        "lb_mip_gap": float(result.mip_gap),
        "lb_dual_bound": float(result.mip_dual_bound),
        "lb_min_pair_coverage_pi": float(np.min(coverage)),
        "lb_feasibility_tolerance": 1e-9,
    }
