"""Resumable single-location evaluation on the generated 1D and 2D inputs."""

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
    errors = np.array([item['optimal_error'], item['pgm_error']])
    if not np.isfinite(errors).all() or errors.min() < -TOL or errors.max() > 1-1/len(phi)+TOL:
        raise ValueError('Discrimination errors violate probability bounds')
    if errors[1] < errors[0]-TOL or abs(errors[1]-pgm_error(phi, p)) > TOL:
        raise ValueError('Inconsistent PGM/optimal-POVM errors')
    for key in ('primal_residual', 'dual_residual', 'optimality_gap'):
        value = item.get(key, float('inf'))
        if not np.isfinite(value) or not 0 <= value <= TOL:
            raise ValueError(f'Invalid SDP diagnostic: {key}')
    for key in ('sdp_status', 'dual_status'):
        if item.get(key) not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError('Invalid saved SDP status')


def evaluate(p, phi, n, cfg):
    p = validate_weights(p, n)
    start = time.perf_counter()
    disc = DiscSDP(len(phi), n, cfg['solver'], cfg['sdp_eps'], cfg['tolerance'])
    error = disc.error(p, phi)
    item = dict(p=p.tolist(), optimal_error=error, pgm_error=pgm_error(phi, p),
                measurement_seconds=time.perf_counter()-start, **disc.diagnostics)
    validate_measurements(item, phi, n)
    return item


def load_inputs(root, dimension, family):
    base = Path(root) / 'inputs' / f'{dimension}d' / family
    manifest = json.loads((base / 'manifest.json').read_text())
    if (manifest['dimension'], manifest['family'], manifest['M'], manifest['c']) != (dimension, family, 1, 1.):
        raise ValueError('Unexpected single-location input design')
    if manifest['ks'] != list(KS) or manifest['ns'] != list(NS) or manifest['instances_per_k'] != 50:
        raise ValueError('Expected K=3,4,5, N=2,4,6,8,10 and 50 instances per K')
    for name in ('set.csv', 'r.csv'):
        if hashlib.sha256((base/name).read_bytes()).hexdigest() != manifest['files'][name]['sha256']:
            raise ValueError(f'Input checksum mismatch: {base/name}')
    data = pd.read_csv(base/'set.csv')
    gaps = pd.read_csv(base/'r.csv')
    if len(data) != sum(KS)*50 or data.duplicated(['k', 'inst', 'field']).any():
        raise ValueError('Invalid field-instance coverage')
    expected_pairs = sum(k*(k-1)//2 for k in KS)*50
    if len(gaps) != expected_pairs or gaps.duplicated(['k', 'inst', 'i', 'j', 'm']).any():
        raise ValueError('Invalid phase-pair coverage')
    for k in KS:
        selected = data[data.k == k]
        if set(selected.inst) != set(range(50)):
            raise ValueError('Expected 50 instances per family and K')
        for inst, group in selected.groupby('inst'):
            group = group.sort_values('field')
            phi = group.phi.to_numpy(float)
            if list(group.field) != list(range(k)) or not np.isfinite(phi).all():
                raise ValueError('Invalid field phases')
            for coordinate in ('x',) if dimension == 1 else ('x', 'y'):
                if group[coordinate].nunique() != 1 or not group[coordinate].between(0, 1).all():
                    raise ValueError('Fields must share one valid sensing location')
            recorded = gaps[(gaps.k == k) & (gaps.inst == inst)].sort_values(['i', 'j'])
            pairs = [(i, j) for i in range(k) for j in range(i+1, k)]
            if list(zip(recorded.i, recorded.j)) != pairs or not (recorded.m == 0).all():
                raise ValueError('Missing field-pair phases')
            expected = np.array([(phi[i]-phi[j]+np.pi)%(2*np.pi)-np.pi for i, j in pairs])
            residual = (recorded.r_ijm.to_numpy()-expected+np.pi)%(2*np.pi)-np.pi
            if not np.isfinite(residual).all() or np.max(np.abs(residual)) > 1e-12:
                raise ValueError('Field phases disagree with phase differences')
            if np.min(np.abs(expected)) < manifest['min_phase_gap']-1e-12:
                raise ValueError('Input violates the phase-gap threshold')
    return base, data


def run_analysis(root, ks=KS, ns=NS, progress=None, dimensions=(1, 2), solver='auto', sdp_eps=1e-8):
    """Evaluate all four input states; reuse only matching validated checkpoints."""
    root = Path(root)
    if tuple(ks) != KS or tuple(ns) != NS or tuple(dimensions) != (1, 2):
        raise ValueError('Use the complete 1D/2D design: K=3,4,5 and N=2,4,6,8,10')
    cfg = dict(solver=solver, sdp_eps=sdp_eps, tolerance=TOL, method='joint_sdp_v1')
    inputs = {(d, f): load_inputs(root, d, f) for d in dimensions for f in FAMILIES}
    versions = dict(python=platform.python_version(), numpy=np.__version__, cvxpy=cp.__version__, solvers=cp.installed_solvers())
    code = b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))
    out = root / 'results/exact_sdp'
    started = time.perf_counter()
    total, completed, reused = 2*len(FAMILIES)*len(KS)*50*len(NS)*4, 0, 0
    report = progress or (lambda message: print(message, flush=True))
    rows = []
    for (dimension, family), (base, data) in inputs.items():
        signature = hashlib.sha256(code + json.dumps(dict(cfg=cfg, versions=versions), sort_keys=True).encode() +
            b''.join((base/name).read_bytes() for name in ('manifest.json', 'set.csv', 'r.csv'))).hexdigest()[:20]
        cache = out / 'checkpoints' / f'{dimension}d' / family / signature
        atomic_json(cache/'settings.json', dict(configuration=cfg, environment=versions, dimension=dimension, family=family))
        for (k, inst), group in data.groupby(['k', 'inst'], sort=True):
            phi = group.sort_values('field').phi.to_numpy(float)
            label = f'{dimension}D | {family} | K={k} | instance {int(inst)+1}/50'
            for n in ns:
                case_rows = []
                for state, p in {'optimized': None, **prescribed_states(n)}.items():
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
                        if state == 'optimized':
                            start = time.perf_counter()
                            result = opt_state(phi, n, cfg)
                            item = dict(p=result.p.tolist(), optimal_error=result.error,
                                pgm_error=pgm_error(phi, result.p), source='joint_sdp',
                                optimization_seconds=time.perf_counter()-start, measurement_seconds=0.,
                                optimizer_converged=True, **result.diagnostics)
                        else:
                            item = dict(evaluate(p, phi, n, cfg), source='prescribed',
                                        optimization_seconds=0., optimizer_converged=None)
                        validate_measurements(item, phi, n)
                        atomic_json(path, item)
                    else:
                        reused += 1
                    completed += 1
                    case_rows.append(dict(dimension=dimension, family=family, k=int(k), inst=int(inst), n=n, state=state,
                        checkpoint=str(path.relative_to(root)), **{key: value for key, value in item.items() if key != 'p'},
                        reflection_asymmetry=float(np.max(np.abs(np.array(item['p'])-np.array(item['p'])[::-1])))))
                optimum = case_rows[0]['optimal_error']
                if any(optimum > r['optimal_error']+2*TOL for r in case_rows[1:]):
                    raise ValueError(f'Joint optimum exceeds a prescribed-state optimum: {label}, N={n}')
                rows.extend(case_rows)
                report(f'{label} | N={n} | {completed}/{total} complete | {reused} cached | {(time.perf_counter()-started)/60:.1f} min')
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'measurements.csv', index=False)
    atomic_json(out/'run.json', dict(configuration=cfg, environment=versions, evaluations=len(frame), dimensions=list(dimensions)))
    return frame
