"""Global integer-program minimum using principal phase gaps.

The pairwise constraints alone do not certify a shared global input state."""

from __future__ import annotations

from math import pi
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from phase_data import compute_nijm, principal_phase


def optimal_global_minimum(K, M, phase_gaps):
    """Minimize sum(n) with abs(r) @ n >= pi and nonnegative integer n.

    Return the minimum, allocation, and solver diagnostics. Require zero MIP
    relative gap and check integrality, primal/dual agreement, and coverage.
    Solver tolerances are 1e-9; normalized coverage is checked to 1e-8."""
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
        raise RuntimeError(f"Global-minimum ILP was not solved optimally: {result.message}")
    rounded = np.rint(result.x)
    if not np.all(np.isfinite(rounded)) or np.any(rounded < 0) or np.max(np.abs(result.x - rounded)) > 1e-8:
        raise RuntimeError("Global-minimum solver returned a noninteger allocation")
    total = int(rounded.sum())
    coverage = matrix @ rounded
    if np.min(coverage) < 1 - 1e-8:
        raise RuntimeError(
            "Rounded global-minimum allocation violates a pairwise constraint: "
            f"minimum normalized coverage={np.min(coverage):.16g}"
        )
    if (abs(result.fun - total) > 1e-6
            or abs(result.mip_dual_bound - total) > 1e-6
            or result.mip_gap > 1e-9):
        raise RuntimeError("Global-minimum optimality certificate did not close")
    return total, [int(n) for n in rounded], {
        "global_solver_status": int(result.status),
        "global_mip_gap": float(result.mip_gap),
        "global_dual_bound": float(result.mip_dual_bound),
        "global_min_pair_coverage_pi": float(np.min(coverage)),
        "global_feasibility_tolerance": 1e-9,
    }
