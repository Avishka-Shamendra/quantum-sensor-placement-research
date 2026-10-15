import cvxpy as cp
import numpy as np


class DiscSDP:
    """Dual SDP for equiprobable minimum-error pure-state discrimination."""

    def __init__(self, k, n, solver="auto", eps=1e-7):
        self.k = int(k)
        self.n = int(n)
        self.d = self.n + 1
        self.eps = float(eps)
        self.rho = [cp.Parameter((self.d, self.d), hermitian=True) for _ in range(self.k)]
        self.y = cp.Variable((self.d, self.d), hermitian=True)
        cons = [self.y - r / self.k >> 0 for r in self.rho]
        self.prob = cp.Problem(cp.Minimize(cp.real(cp.trace(self.y))), cons)
        self.solvers = self._solver_list(solver)

    @staticmethod
    def _solver_list(name):
        installed = set(cp.installed_solvers())
        if name != "auto":
            if name not in installed:
                raise ValueError(f"CVXPY solver '{name}' is not installed")
            return [name]
        order = ["CLARABEL", "SCS", "CVXOPT"]
        found = [s for s in order if s in installed]
        if not found:
            raise RuntimeError("No SDP-capable CVXPY solver is installed")
        return found

    def _solve(self):
        errors = []
        for solver in self.solvers:
            try:
                opts = {"solver": solver, "warm_start": True, "verbose": False}
                if solver == "SCS":
                    opts.update({"eps": self.eps, "max_iters": 20000})
                elif solver == "CLARABEL":
                    opts.update(
                        {
                            "tol_gap_abs": self.eps,
                            "tol_gap_rel": self.eps,
                            "tol_feas": self.eps,
                            "max_iter": 1000,
                        }
                    )
                value = self.prob.solve(**opts)
                if self.prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and value is not None:
                    return float(np.real(value))
                errors.append(f"{solver}: {self.prob.status}")
            except Exception as exc:  # try the next installed conic solver
                errors.append(f"{solver}: {exc}")
        raise RuntimeError("SDP failed; " + " | ".join(errors))

    def error(self, p, phi):
        p = np.asarray(p, dtype=float)
        phi = np.asarray(phi, dtype=float)
        if p.shape != (self.d,) or phi.shape != (self.k,):
            raise ValueError("Incompatible probability or phase vector")
        if np.min(p) < -1e-10 or not np.isclose(p.sum(), 1.0, atol=1e-8):
            raise ValueError("p must lie on the probability simplex")
        p = np.clip(p, 0.0, None)
        p /= p.sum()
        w = np.arange(self.d)
        states = np.sqrt(p)[None, :] * np.exp(1j * phi[:, None] * w[None, :])
        for param, state in zip(self.rho, states):
            mat = np.outer(state, state.conj())
            param.value = (mat + mat.conj().T) / 2.0
        success = np.clip(self._solve(), 0.0, 1.0)
        return float(1.0 - success)
