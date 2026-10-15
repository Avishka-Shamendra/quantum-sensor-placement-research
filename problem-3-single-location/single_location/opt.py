import numpy as np
from scipy.optimize import dual_annealing, minimize

from .sdp import DiscSDP


def _starts(d, count, rng):
    points = [np.full(d, 1.0 / d)]
    if count > 1:
        edge = np.zeros(d)
        edge[[0, -1]] = 0.5
        points.append(edge)
    while len(points) < count:
        points.append(rng.dirichlet(np.ones(d)))
    return points


def _objective(disc, phi):
    cache = {}

    def obj(x):
        x = np.asarray(x, dtype=float)
        if np.min(x) < -1e-7 or x.sum() <= 0:
            return 1.0
        x = np.clip(x, 0.0, None)
        x /= x.sum()
        key = tuple(np.round(x, 11))
        if key not in cache:
            cache[key] = disc.error(x, phi)
        return cache[key]

    return obj


def opt_state(phi, n, cfg, rng):
    k = len(phi)
    d = n + 1
    disc = DiscSDP(k, n, cfg["solver"], cfg["sdp_eps"])
    obj = _objective(disc, phi)

    best = None
    for x0 in _starts(d, cfg["starts"], rng):
        res = minimize(
            obj,
            x0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * d,
            constraints={"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
            options={
                "maxiter": cfg["slsqp_iter"],
                "ftol": cfg["ftol"],
                "disp": False,
            },
        )
        x = np.clip(res.x, 0.0, None)
        x /= x.sum()
        err = obj(x)
        item = (err, x, res)
        if best is None or err < best[0]:
            best = item
    return best


def polish_state(phi, n, p0, cfg):
    """Polish one supplied feasible state using the same SLSQP/SDP objective."""
    d = n + 1
    x0 = np.clip(np.asarray(p0, dtype=float), 0.0, None)
    if x0.shape != (d,) or x0.sum() <= 0:
        raise ValueError(f"p0 must be a nonzero vector of length {d}")
    x0 /= x0.sum()
    disc = DiscSDP(len(phi), n, cfg["solver"], cfg["sdp_eps"])
    obj = _objective(disc, phi)
    res = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * d,
        constraints={"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
        options={
            "maxiter": cfg["slsqp_iter"],
            "ftol": cfg["ftol"],
            "disp": False,
        },
    )
    p = np.clip(res.x, 0.0, None)
    p /= p.sum()
    return obj(p), p, res


def ann_state(phi, n, cfg, seed):
    """Heuristic cross-check using bounded logits and dual annealing; not a certificate."""
    disc = DiscSDP(len(phi), n, cfg["solver"], cfg["sdp_eps"])
    bound = float(cfg["log_bound"])

    def unpack(y):
        z = np.r_[np.asarray(y, dtype=float), 0.0]
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()

    def obj(y):
        return disc.error(unpack(y), phi)

    res = dual_annealing(
        obj,
        bounds=[(-bound, bound)] * n,
        maxiter=cfg["ann_iter"],
        seed=int(seed),
        no_local_search=False,
    )
    return float(res.fun), unpack(res.x), res
