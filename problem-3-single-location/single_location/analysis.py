"""Validated, atomic checkpoints for single-location experiments."""
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

from .opt import opt_state
from .sdp import DiscSDP
from .states import pgm_error, prescribed_states, validate_weights

FAMILIES = ('gaussian', 'sinusoidal', 'quadratic')
KS = (3, 4, 5)
NS = (2, 4, 6, 8, 10)
TOL = 1e-5


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)



def validate_measurements(item, phi, n):
    p = validate_weights(item['p'], n)
    a, b = item['optimal_error'], item['pgm_error']
    if not np.isfinite([a, b]).all() or min(a, b) < -TOL or max(a, b) > 1 - 1 / len(phi) + TOL:
        raise ValueError('Discrimination errors violate probability bounds')
    if b < a - TOL or abs(b - pgm_error(phi, p)) > TOL:
        raise ValueError('Inconsistent PGM/optimal-POVM errors')
    if item.get('dual_residual', float('inf')) > TOL:
        raise ValueError('Missing or invalid numerical SDP validation')


def evaluate(p, phi, n, cfg):
    p = validate_weights(p, n)
    start = time.perf_counter()
    disc = DiscSDP(len(phi), n, cfg['solver'], cfg['sdp_eps'])
    error = disc.error(p, phi)
    y = disc.y.value
    residual = max(0., max(-float(np.linalg.eigvalsh(y - r.value / len(phi)).min()) for r in disc.rho))
    item = dict(p=p.tolist(), optimal_error=error, pgm_error=pgm_error(phi, p),
                dual_residual=residual, sdp_status=disc.prob.status,
                solver=disc.prob.solver_stats.solver_name,
                measurement_seconds=time.perf_counter() - start)
    validate_measurements(item, phi, n)
    return item


def run_analysis(root, ks=KS, ns=NS, progress=None):
    """Evaluate cases with per-state checkpoints and optional progress reporting."""
    started = time.perf_counter()
    completed = reused = computed = 0
    total = len(FAMILIES) * len(ks) * 30 * len(ns) * 4

    def report(message):
        if progress is None:
            print(message, flush=True)
        else:
            progress(message)

    report('Checking imported files')
    root = Path(root)
    manifest = json.loads((root / 'imported/manifest.json').read_text())
    for name, record in manifest.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Imported-file checksum mismatch: {name}')
    versions = dict(python=platform.python_version(), numpy=np.__version__,
                    cvxpy=cp.__version__, solvers=cp.installed_solvers())
    code = b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))
    rows = []
    for family in FAMILIES:
        base = root / 'imported' / family
        cfg = json.loads((base / 'cfg.json').read_text())
        data = pd.read_csv(base / 'data/set.csv')
        res = pd.read_csv(base / 'out/res.csv')
        prob = pd.read_csv(base / 'out/prob.csv')
        signature = hashlib.sha256(code + json.dumps(versions, sort_keys=True).encode() +
                                   b''.join((base / f).read_bytes() for f in
                                            ('cfg.json', 'data/set.csv', 'out/res.csv', 'out/prob.csv'))).hexdigest()[:20]
        cache = root / 'results/checkpoints' / family / signature
        atomic_json(cache / 'settings.json', dict(configuration=cfg, environment=versions,
                    ks=list(ks), ns=list(ns), tolerance=TOL,
                    interpretation='SDP-optimized state reused for PGM; no PGM-specific state optimization'))
        for k in ks:
            groups = data[data.k == k].groupby('inst', sort=True)
            if set(groups.groups) != set(range(30)):
                raise ValueError(f'{family}: expected 30 instances for K={k}')
            report(f'{family.capitalize()} | K={k} | 30 instances | N={list(ns)}')
            for inst, group in groups:
                label = f'{family.capitalize()} | K={k} | instance {int(inst) + 1}/30'
                report(f'{label} | {completed}/{total} evaluations complete')
                group = group.sort_values('field')
                phi = group.phi.to_numpy(float)
                if len(phi) != k or group.field.nunique() != k or not np.isfinite(phi).all() or group.x.nunique() != 1:
                    raise ValueError('Invalid single-location dataset')
                for n in ns:
                    states = {'optimized': None, **prescribed_states(n)}
                    for state, p in states.items():
                        if progress is not None:
                            report(f'{label} | N={n} | {state} | checking cache')
                        path = cache / f'k{k}_i{inst}_n{n}_{state}.json'
                        item = None
                        if path.exists():
                            try:
                                item = json.loads(path.read_text())
                                validate_measurements(item, phi, n)
                                if p is not None and not np.allclose(item['p'], p, atol=1e-12, rtol=0):
                                    raise ValueError('Prescribed state changed')
                            except (ValueError, KeyError, TypeError):
                                item = None
                        if item is None:
                            if progress is not None:
                                report(f'{label} | N={n} | {state} | evaluating')
                            meta = dict(source='prescribed', optimization_seconds=0., optimizer_converged=None)
                            if state == 'optimized':
                                # Save weights before measurement validation.
                                weights_path = cache / f'k{k}_i{inst}_n{n}_weights.json'
                                if weights_path.exists():
                                    meta = json.loads(weights_path.read_text())
                                    p = validate_weights(meta.pop('p'), n)
                                else:
                                    r = res[(res.k == k) & (res.inst == inst) & (res.n == n)]
                                    q = prob[(prob.k == k) & (prob.inst == inst) & (prob.n == n)].sort_values('w')
                                    valid = len(r) == 1 and list(q.w) == list(range(n + 1))
                                    if valid:
                                        try:
                                            p = validate_weights(q.p.to_numpy(), n)
                                            valid = np.isfinite(r.iloc[0]['err']) and 0 <= r.iloc[0]['err'] <= 1 - 1/k + TOL
                                        except ValueError:
                                            valid = False
                                    if valid:
                                        rr = r.iloc[0]
                                        meta = dict(source='archive', optimization_seconds=float(rr.sec),
                                                    optimizer_converged=str(rr.ok).lower() == 'true',
                                                    imported_error=float(rr.err))
                                    else:
                                        rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], k, int(inst), n]))
                                        start = time.perf_counter()
                                        report(f'{label} | N={n} | optimizing missing state')
                                        err, p, info = opt_state(phi, n, cfg, rng)
                                        meta = dict(source='computed', optimization_seconds=time.perf_counter()-start,
                                                    optimizer_converged=bool(info.success), imported_error=float(err))
                                    atomic_json(weights_path, dict(meta, p=p.tolist()))
                            item = dict(evaluate(p, phi, n, cfg), **meta)
                            if 'imported_error' in item:
                                item['saved_error_difference'] = item['optimal_error'] - item['imported_error']
                            atomic_json(path, item)
                            computed += 1
                        else:
                            reused += 1
                        completed += 1
                        if progress is not None:
                            elapsed = time.perf_counter() - started
                            report(f'{label} | N={n} | {completed}/{total} complete | '
                                   f'{reused} cached, {computed} evaluated | {elapsed / 60:.1f} min')
                        rows.append(dict(family=family, k=k, inst=int(inst), n=n, state=state,
                                         **{a: b for a, b in item.items() if a != 'p'},
                                         reflection_asymmetry=float(np.max(np.abs(np.array(item['p'])-np.array(item['p'])[::-1])))))
    frame = pd.DataFrame(rows)
    out = root / 'results'
    frame.to_csv(out / 'measurements.csv', index=False)
    report(f'Complete: {completed} evaluations | {reused} cached, {computed} evaluated | '
           f'{(time.perf_counter() - started) / 60:.1f} min')
    return frame
