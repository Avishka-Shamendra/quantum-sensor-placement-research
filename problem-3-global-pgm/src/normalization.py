"""Normalize budgets by the joint perfect-discrimination ILP lower bound."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def joint_sensor_lower_bound(path):
    """Equation (19): min sum n_m, sum_m n_m |r_ijm| >= pi for all pairs.

    Necessary, not sufficient, for joint perfect discrimination. It is not a
    state-family-specific zero-error threshold. r already includes phase scale.
    """
    from .input_data import load_field_input
    fields = load_field_input(path)  # Validate completeness and phase consistency.
    K, M = fields.shape
    pairs = [(i,j) for i in range(K) for j in range(i+1,K)]
    with Path(path).open() as stream:
        gaps = {(int(r['i']),int(r['j']),int(r['m'])):float(r['r_ijm']) for r in csv.DictReader(stream)}
    matrix = np.array([[abs(gaps[i,j,m])/np.pi for m in range(M)] for i,j in pairs])
    if np.any(np.max(matrix,axis=1)==0):
        raise ValueError('No finite joint lower bound: a pair has identical phases everywhere')
    for presolve, rescale in ((True, False), (False, False), (True, True), (False, True)):
        scale = matrix.max(axis=1) if rescale else np.ones(len(matrix))
        result = milp(c=np.ones(M),integrality=np.ones(M),bounds=Bounds(0,np.inf),
                      constraints=LinearConstraint(matrix/scale[:,None],1/scale,np.inf),
                      options={'mip_rel_gap':0.0, 'presolve':presolve})
        if not result.success or result.status != 0 or result.x is None:
            continue
        n = np.rint(result.x)
        total = int(n.sum())
        valid = (np.all(n>=0) and np.max(np.abs(n-result.x))<=1e-7 and
                 np.min(matrix@n)>=1-1e-7 and abs(result.fun-total)<=1e-6 and
                 abs(result.mip_dual_bound-total)<=1e-6 and result.mip_gap<=1e-9 and total>=1)
        if valid:
            return total
    raise RuntimeError('Joint lower-bound optimality/feasibility checks failed with and without presolve and row scaling')


def normalize_budget(rows, project, study, output):
    """Enrich saved results without rerunning allocation algorithms.

    Cache bounds by exact input CSV hash and this implementation's hash.
    """
    if study not in ('compact','extended'):
        raise ValueError('unknown study')
    project, output = Path(project), Path(output)
    cache_path = output/'joint_lower_bounds.json'
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    bounds = {}
    for row in rows:
        key = (row['dimension'],row['case'])
        if key in bounds:
            continue
        dataset = f'{"compact_" if study=="compact" else ""}benchmark_{key[0]}d'
        path = project/'inputs'/dataset/key[1]
        fingerprint = hashlib.sha256(path.read_bytes()+source_hash.encode()).hexdigest()
        entry = cached.get(fingerprint)
        if entry is None:
            entry = dict(N_LB_joint=joint_sensor_lower_bound(path),dimension=key[0],case=key[1])
            cached[fingerprint] = entry
        bounds[key] = entry['N_LB_joint']
    output.mkdir(parents=True,exist_ok=True)
    cache_path.write_text(json.dumps(cached,indent=2)+'\n')
    normalized = [dict(row,N_LB_joint=bounds[row['dimension'],row['case']],
                       budget_ratio=row['N']/bounds[row['dimension'],row['case']]) for row in rows]
    if normalized:
        with (output/'normalized_plot_data.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(normalized[0]))
            writer.writeheader();writer.writerows(normalized)
    return normalized
