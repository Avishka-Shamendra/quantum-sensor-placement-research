"""Resume matched evaluations using per-method, per-budget checkpoints."""
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import atomic_json
from .cache import checkpoint_path
from .sources import implementations, source_hashes


def run_comparison(project, config, directory, manifest, *, run=True):
    _, seq, joint, joint_input, _ = implementations()
    provenance = dict(config=config.to_dict(), dataset_id=manifest['dataset_id'],
                      source_sha256=source_hashes(), dicke_policy=joint.DICKE_POLICY)
    output = Path(project) / 'results' / ('compact' if config.mode == 'test' else config.mode)
    output.mkdir(parents=True, exist_ok=True)
    reused = {'sequential': 0, 'joint': 0}
    computed = {'sequential': 0, 'joint': 0}
    rows = []
    eligible = [c for c in manifest['cases'] if c['status'] == 'eligible']
    for index, case in enumerate(eligible, 1):
        path = directory / case['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != case['csv_sha256']:
            raise ValueError(f'Input checksum mismatch: {path}')
        K, M, gaps = seq['input_data'].load_phase_input(path)
        fields, signed_phases = joint_input.load_phase_input(path, c=1.0)
        for budget in case['budget_schedule']:
            N = budget['N']
            row = {key: case[key] for key in ('instance_id', 'dimension', 'family', 'K', 'M',
                                              'replicate', 'N_LB_joint', 'N_max')}
            row.update(N=N, target_fractions=budget['target_fractions'])
            for method in ('sequential', 'joint'):
                algorithm = getattr(config, f'{method}_algorithm')
                cache = checkpoint_path(output, config, method, case, N)
                if cache.exists():
                    result = json.loads(cache.read_text())
                    reused[method] += 1
                else:
                    if not run:
                        raise FileNotFoundError(f'Missing {method}/{algorithm} checkpoint: {cache}; enable RUN_ANALYSIS.')
                    computed[method] += 1
                    start = perf_counter()
                    if method == 'sequential':
                        if algorithm == 'vnd_beam':
                            solution = seq['heuristic'].heuristic(
                                K, M, N, gaps, beam_width=config.beam_width,
                                max_iterations=config.sequential_max_iterations,
                                random_seed=case['generation_seed'])
                        else:
                            solution = seq['greedy'].greedy(K, M, N, gaps)
                        error = seq['actual_error'].actual_tree_error(
                            solution.decision_tree, K, gaps, solution.allocation)
                        result = dict(error=error, allocation=list(solution.allocation),
                                      decision_tree=seq['strategy'].tree_to_dict(solution.decision_tree))
                    else:
                        if algorithm == 'vnd':
                            solution = joint.vnd_search(N, M, signed_phases)
                        else:
                            solution = joint.greedy_search(N, M, lambda allocation:
                                joint.probe_gram(fields, allocation, 'local_dicke', c=1.0))
                        gram = joint.probe_gram(fields, solution.allocation, 'local_dicke', c=1.0)
                        result = dict(error=joint.pgm_error(gram),
                                      allocation=list(solution.allocation))
                    result.update(runtime_seconds=perf_counter() - start, N=N,
                                  input_sha256=case['csv_sha256'], method=method, algorithm=algorithm)
                    _validate(result, case, N, M, method)
                    atomic_json(cache, result)
                _validate(result, case, N, M, method)
                row[f'{method}_algorithm'] = algorithm
                row[f'{method}_runtime_seconds'] = result['runtime_seconds']
                row[f'{method}_error'] = result['error']
                row[f'{method}_allocation'] = result['allocation']
            rows.append(row)
        print(f'Instances evaluated/loaded: {index}/{len(eligible)}', flush=True)
    if not rows:
        raise ValueError('No eligible positive budgets in this dataset.')
    with (output / 'matched_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()}
                         for r in rows)
    provenance.update(reused=reused, computed=computed)
    atomic_json(output / 'provenance.json', provenance)
    print(f'Reused: {reused}; computed: {computed}')
    return rows, output


def _validate(result, case, N, M, method):
    allocation = result['allocation']
    if (result['input_sha256'] != case['csv_sha256'] or result['N'] != N
            or result['method'] != method or len(allocation) != M
            or any(type(n) is not int or n < 0 for n in allocation) or sum(allocation) != N
            or not np.isfinite(result['error']) or not -1e-10 <= result['error'] <= 1 + 1e-10):
        raise ValueError(f'Invalid {method} checkpoint for {case["instance_id"]}, N={N}')


def cache_coverage(project, config, manifest):
    """Count reusable checkpoints without running a solver or evaluator."""
    output = Path(project) / 'results' / ('compact' if config.mode == 'test' else config.mode)
    counts = {}
    for side in ('sequential', 'joint'):
        total = available = 0
        for case in manifest['cases']:
            if case['status'] != 'eligible':
                continue
            for budget in case['budget_schedule']:
                total += 1
                available += checkpoint_path(output, config, side, case, budget['N']).exists()
        counts[side] = dict(algorithm=getattr(config, f'{side}_algorithm'),
                            available=available, missing=total-available, total=total)
    return counts
