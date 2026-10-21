"""Reproducible, resumable notebook comparisons with matched-case summaries."""
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from random import Random
from time import perf_counter
import numpy as np
from .input_data import load_phase_input
from .probes import FAMILIES, DICKE_POLICY, probe_gram
from .search import exhaustive_search, greedy_search, vnd_search, ALGORITHM_POLICY
from .pgm import pgm_error

ROOT = Path(__file__).resolve().parents[1]


METHODS = {'compact': ('exhaustive', 'greedy', 'vnd'),
           'extended': ('greedy', 'vnd')}
SOLVERS = {'exhaustive': exhaustive_search, 'greedy': greedy_search, 'vnd': vnd_search}

TEST_REPLICATES_PER_CELL = {'compact': 3, 'extended': 5}


def select_cases(manifest, study, mode):
    """Select a balanced subset of every family/K/M cell, preserving budgets."""
    if study not in TEST_REPLICATES_PER_CELL or mode not in ('test', 'full'):
        raise ValueError('unknown study or mode')
    eligible = manifest['cases']
    if not eligible:
        raise ValueError('dataset has no cases')
    if mode == 'full':
        return eligible
    quota = manifest.get('design', {}).get('test_replicates_per_cell', TEST_REPLICATES_PER_CELL[study])
    if type(quota) is not int or quota < 1:
        raise ValueError('test replicate count must be positive')
    groups = defaultdict(list)
    for case in eligible:
        groups[case['field_model'], case['K'], case['M']].append(case)
    selected = []
    for cell, cases in sorted(groups.items()):
        if len(cases) < quota:
            raise ValueError(f'not enough cases in cell {cell}')
        selected.extend(sorted(cases, key=lambda c: (c['replicate'], c['file']))[:quota])
    return selected


def run_analysis(study, mode='test', *, run=True, resume=True, max_allocations=20_000,
                 time_limit=15.0, dataset_root=None, include_baselines=False):
    if study not in ('compact','extended') or mode not in ('test','full'):
        raise ValueError('unknown study or mode')
    if max_allocations < 1 or not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('limits must be positive')
    methods = METHODS[study] + (('near_uniform','random') if include_baselines else ())
    settings = dict(study=study, mode=mode, max_allocations=max_allocations,
                    time_limit=time_limit, sampling='problem3_balanced_cells', budget_policy='saved_case_budgets',
                    dicke_policy=DICKE_POLICY, algorithm_policy=ALGORITHM_POLICY, methods=list(methods), vnd_move_sizes=[1, 2, 3], vnd_max_iterations=None, vnd_tolerance=1e-12)
    if dataset_root is not None:
        dataset_root = Path(dataset_root).resolve()
        settings['dataset_root'] = str(dataset_root)
    sources = b''.join(p.read_bytes() for p in sorted((ROOT/'src').glob('*.py')))
    digest = hashlib.sha256(sources+json.dumps(settings,sort_keys=True).encode())
    datasets = []
    for dimension in (1,2):
        name = f'{"compact_" if study == "compact" else ""}benchmark_{dimension}d'
        directory = dataset_root/f'{dimension}d' if dataset_root is not None else ROOT/'inputs'/study/f'{dimension}d'
        if dataset_root is None and not directory.exists():
            directory = ROOT/'inputs'/name
        digest.update((directory/'manifest.json').read_bytes())
        manifest = json.loads((directory/'manifest.json').read_text())
        cases = select_cases(manifest, study, mode)
        for case in cases:
            digest.update((directory/case['file']).read_bytes())
        datasets.append((dimension, directory, cases))
    signature = digest.hexdigest()
    output = ROOT/'results'/study/mode
    path = output/'records.jsonl'
    records = []
    info_path = output/'run_info.json'
    saved_info = json.loads(info_path.read_text()) if info_path.exists() else {}
    matching = saved_info.get('signature') == signature
    if path.exists() and matching and (resume or not run):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not run:
        if not records:
            raise FileNotFoundError('No matching saved results. Set RUN_ANALYSIS=True.')
        return records, output
    # A fresh run replaces this study/mode; only matching runs may resume.
    if output.exists() and (not resume or not matching):
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'run_info.json').write_text(json.dumps(dict(signature=signature, settings=settings),indent=2)+'\n')
    selection = [dict(dimension=dimension, case=case['file'], field_model=case['field_model'],
                      K=case['K'], M=case['M'], sensor_budgets=json.dumps(case['sensor_budgets']))
                 for dimension, directory, cases in datasets for case in cases]
    write_table(output/'selected_cases.csv', selection,
                ['dimension', 'case', 'field_model', 'K', 'M', 'sensor_budgets'])
    def key(r):
        return (r['dimension'],r['case'],r['N'],r['probe'],r['method'])
    done = {key(r) for r in records}
    allocation_path = output/'allocations.jsonl'
    allocation_cache = {}
    if allocation_path.exists():
        for line in allocation_path.read_text().splitlines():
            a = json.loads(line)
            allocation_cache[a['dimension'],a['case'],a['N'],a['method']] = a
    with path.open('a') as stream:
        for dimension, directory, cases in datasets:
            print(f'{study} {mode}: {dimension}D, {len(cases)} cases', flush=True)
            for index, case in enumerate(cases):
                print(f"Snapshot {index+1}/{len(cases)} | {dimension}D | {case['field_model']} | "
                      f"K={case['K']}, M={case['M']}", flush=True)
                fields, r_ijm = load_phase_input(directory/case['file'], c=case['phase_scale_c'])
                if fields.shape != (case['K'], case['M']):
                    raise ValueError('Manifest dimensions disagree with the phase CSV')
                budgets = case['sensor_budgets']
                for N in budgets:
                    for probe in FAMILIES:
                        for method in methods:
                            record = dict(dimension=dimension,case=case['file'],field_model=case['field_model'],
                                          K=case['K'],M=case['M'],N=N,probe=probe,method=method,dicke_policy=DICKE_POLICY,algorithm_policy=ALGORITHM_POLICY,
                                          allocation_count=comb(N+case['M']-1,case['M']-1))
                            if 'budget_schedule' in case:
                                schedule = next(b for b in case['budget_schedule'] if b['N'] == N)
                                record.update(N_LB_joint=case['N_LB_joint'],
                                              budget_ratio=schedule['actual_ratio'], target_fractions=schedule['target_fractions'],
                                              dataset_version=dataset_root.parent.name if dataset_root is not None else 'legacy')
                            if key(record) in done:
                                continue
                            start = perf_counter()
                            def gram(n):
                                if perf_counter()-start > time_limit:
                                    raise TimeoutError('PGM search time limit exceeded')
                                return probe_gram(fields,n,probe,c=case['phase_scale_c'])
                            try:
                                if method == 'exhaustive':
                                    if record['allocation_count'] > max_allocations:
                                        record.update(status='allocation_limit', reason=f'exceeds {max_allocations} allocations')
                                    else:
                                        result = exhaustive_search(N,case['M'],gram,max_allocations=max_allocations)
                                        record.update(result.to_dict())
                                        record['allocation'] = list(result.allocation)
                                    record.update(allocation_runtime_seconds=None, pgm_evaluation_seconds=None,
                                                  search_runtime_seconds=perf_counter()-start)
                                    record['runtime_seconds'] = record['search_runtime_seconds']
                                else:
                                    allocation_key = (dimension,case['file'],N,method)
                                    if allocation_key not in allocation_cache:
                                        allocation_start = perf_counter()
                                        a = dict(dimension=dimension,case=case['file'],N=N,method=method)
                                        try:
                                            if method in ('near_uniform','random'):
                                                M=case['M']
                                                if method=='near_uniform':
                                                    candidates=[tuple(N//M+(m<N%M) for m in range(M))]
                                                else:
                                                    seed=int.from_bytes(hashlib.sha256(json.dumps(allocation_key).encode()).digest()[:16],'big')
                                                    rng = Random(seed)
                                                    allocation = [0] * M
                                                    for _ in range(N):
                                                        allocation[rng.randrange(M)] += 1
                                                    candidates = [tuple(allocation)]
                                                a.update(allocation=list(candidates[0]),allocations=[list(n) for n in candidates],status='baseline_complete',
                                                         runtime_seconds=perf_counter()-allocation_start)
                                            else:
                                                a.update(SOLVERS[method](N,case['M'],r_ijm,time_limit=time_limit).to_dict())
                                            a['allocation']=list(a['allocation'])
                                        except TimeoutError as error:
                                            a.update(status='timeout',reason=str(error),runtime_seconds=perf_counter()-allocation_start)
                                        allocation_cache[allocation_key] = a
                                        with allocation_path.open('a') as allocation_stream:
                                            allocation_stream.write(json.dumps(a)+'\n')
                                    a = allocation_cache[allocation_key]
                                    record.update({k:v for k,v in a.items() if k not in ('runtime_seconds',)})
                                    record['allocation_runtime_seconds'] = a['runtime_seconds']
                                    evaluation_start = perf_counter()
                                    if a['status'] != 'timeout':
                                        record['pgm_error'] = pgm_error(probe_gram(fields,a['allocation'],probe,c=case['phase_scale_c']))
                                        record['pgm_evaluations'] = 1
                                    record['pgm_evaluation_seconds'] = perf_counter()-evaluation_start if a['status'] != 'timeout' else 0.0
                                    record['search_runtime_seconds'] = None
                                    record['runtime_seconds'] = a['runtime_seconds']/len(FAMILIES)+record['pgm_evaluation_seconds']
                            except TimeoutError as error:
                                record.update(status='timeout',reason=str(error),runtime_seconds=perf_counter()-start,
                                              allocation_runtime_seconds=None,pgm_evaluation_seconds=None,
                                              search_runtime_seconds=perf_counter()-start)
                            records.append(record)
                            stream.write(json.dumps(record)+'\n')
                            stream.flush()
                            done.add(key(record))
    print('All snapshots complete.', flush=True)
    return records, output


def write_table(path, rows, columns):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records, output, *, methods=None):
    """Use only groups completed by every requested method; report all statuses."""
    if any(r.get('algorithm_policy') != ALGORITHM_POLICY for r in records):
        raise ValueError('Saved allocation results use an older phase-input policy; recompute with CSV r_ijm.')
    if any(r['probe'] in ('local_dicke', 'global_dicke') and r.get('dicke_policy') != DICKE_POLICY for r in records):
        raise ValueError('Saved Dicke results use an older state definition. Rerun the analysis for the equal-sector superposition.')
    output = Path(output)
    methods = tuple(methods) if methods is not None else tuple(m for m in SOLVERS if any(r['method']==m for r in records))
    if not methods or any(m not in SOLVERS for m in methods):
        raise ValueError('select known methods')
    reference = 'exhaustive' if 'exhaustive' in methods else 'greedy'
    if reference not in methods:
        raise ValueError('comparison requires exhaustive or greedy as reference')
    status = Counter((r['dimension'],r['probe'],r['method'],r['status']) for r in records)
    status_rows = [dict(zip(('dimension','probe','method','status','count'),(*key,count))) for key,count in sorted(status.items())]
    write_table(output/'status.csv',status_rows,['dimension','probe','method','status','count'])
    paired = defaultdict(dict)
    for r in records:
        if r['status'] in ('optimal','local_optimum', 'maximum_iterations','greedy_complete'):
            paired[(r['dimension'],r['case'],r['N'],r['probe'])][r['method']] = r
    matches = []
    for pair in paired.values():
        if not all(method in pair for method in methods):
            continue
        ref = pair[reference]
        for method in methods:
            row = pair[method]
            gap = row['pgm_error']-ref['pgm_error']
            if reference == 'exhaustive' and gap < -1e-7:
                raise AssertionError('A method beat exhaustive search')
            matches.append(dict(dimension=row['dimension'],case=row['case'],field_model=row['field_model'],
                                K=row['K'],M=row['M'],N=row['N'],probe=row['probe'],method=method,dicke_policy=row.get('dicke_policy'),algorithm_policy=row.get('algorithm_policy'),
                                pgm_error=row['pgm_error'],runtime_seconds=row['runtime_seconds'],
                                reference=reference,error_gap_to_reference=gap,
                                allocation_match=tuple(row['allocation'])==tuple(ref['allocation'])))
    columns = ['dimension','case','field_model','K','M','N','probe','method','pgm_error',
               'runtime_seconds','reference','error_gap_to_reference','allocation_match','dicke_policy','algorithm_policy']
    write_table(output/'paired_results.csv',matches,columns)
    groups = defaultdict(list)
    for row in matches:
        groups[row['dimension'],row['probe'],row['method']].append(row)
    summary = []
    for (dimension,probe,method),rows in sorted(groups.items()):
        summary.append(dict(dimension=dimension,probe=probe,method=method,paired_case_budgets=len(rows),
                            reference=reference,
                            **{name:float(np.mean([r[name] for r in rows])) for name in
                               ('pgm_error','runtime_seconds','error_gap_to_reference','allocation_match')}))
    write_table(output/'summary.csv',summary,['dimension','probe','method','paired_case_budgets','reference',
                                            'pgm_error','runtime_seconds','error_gap_to_reference','allocation_match'])
    return matches, summary, status_rows


def plot_comparison(matches, output, *, methods=None):
    """Return algorithm overview figures followed by the five state figures."""
    from .figures import plot_by_algorithm, plot_by_state
    methods = tuple(methods) if methods is not None else tuple(m for m in SOLVERS if any(r['method']==m for r in matches))
    return (plot_by_algorithm(matches, output, methods=methods)
            + plot_by_state(matches, output, methods=methods))
