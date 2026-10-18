"""Bound-limited comparisons using the existing per-instance search caches."""
import hashlib
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import dual_search
import greedy_comparison


def cache_signature(method, settings):
    """Match the original runners' cache keys without changing search settings."""
    filename = 'dual_search.py' if method == 'vnd_beam' else 'greedy_comparison.py'
    paths = sorted((dual_search.PROBLEM2 / 'src').glob('*.py')) + [dual_search.HERE / filename] + [
        dual_search.REPOSITORY / 'problem-1-zero-error/src' / f'{name}.py'
        for name in ('optimal', 'phase_data')]
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    if method == 'greedy':
        digest.update((dual_search.HERE / 'dual_search.py').read_bytes())
    return digest.hexdigest()


def cache_path(signature, path, case):
    digest = hashlib.sha256(signature.encode() + path.read_bytes() +
                            json.dumps(case, sort_keys=True).encode()).hexdigest()
    return dual_search.HERE / 'results/cache' / f'{digest}.json'


def partition(entries, limit):
    kept, skipped = [], []
    for entry in entries:
        _, _, case = entry
        if case['minimum_sensors_for_zero_error'] <= limit:
            kept.append(entry)
        else:
            skipped.append(dict(dimension=case['spatial_dimension'], case=case['file'],
                                field_model=case['field_model'], K=case['K'], M=case['M'],
                                N_min_seq=case['minimum_sensors_for_zero_error'],
                                cutoff=limit, reason='sequential_upper_bound_exceeds_cutoff'))
    return kept, skipped


def run_method(method, entries, targets, settings, *, resume, run):
    modules = dual_search.dependencies()
    if method == 'greedy':
        greedy = importlib.import_module('greedy').greedy
        def candidate(K, M, N, phases, **options):
            return greedy(K, M, N, phases, time_limit=options['time_limit'])
        modules['heuristic'] = SimpleNamespace(heuristic=candidate)
    signature = cache_signature(method, settings)
    vnd_signature = cache_signature('vnd_beam', settings)
    rows, failures = [], []
    for index, (_, path, case) in enumerate(entries, 1):
        print(f'{method} | Snapshot {index}/{len(entries)} | {case["spatial_dimension"]}D | '
              f'{case["field_model"]} | K={case["K"]}, M={case["M"]}', flush=True)
        checkpoint_path = cache_path(signature, path, case)
        cache = json.loads(checkpoint_path.read_text()) if resume and checkpoint_path.exists() else {}
        if not resume and checkpoint_path.exists():
            raise FileExistsError('Existing cache: use RESUME=True')
        if not cache and resume and method == 'greedy':
            vnd_path = cache_path(vnd_signature, path, case)
            if vnd_path.exists():
                original = json.loads(vnd_path.read_text())
                if 'perfect' in original:
                    cache = dict(perfect=original['perfect'], targets={})
        identity = dict(method=method, dimension=case['spatial_dimension'], case=case['file'],
                        field_model=case['field_model'], K=case['K'], M=case['M'],
                        N_min_seq=case['minimum_sensors_for_zero_error'])
        K, M, phases = modules['input_data'].load_phase_input(path)
        if (K, M) != (case['K'], case['M']):
            raise ValueError('Manifest and CSV dimensions disagree')
        def checkpoint():
            dual_search.save_json(checkpoint_path, cache)
        try:
            if 'perfect' not in cache:
                if not run:
                    raise FileNotFoundError('Missing perfect-allocation cache; enable RUN_ANALYSIS')
                cache.update(perfect=dual_search.initial_solution(K, M, phases, identity['N_min_seq'], modules), targets={})
                checkpoint()
            for epsilon in targets:
                state = cache['targets'].setdefault(str(epsilon), {})
                if not run and not state.get('complete'):
                    raise FileNotFoundError('Incomplete target cache; enable RUN_ANALYSIS')
                result = dual_search.budget_search(K, M, phases, epsilon, cache['perfect'],
                                                   modules, settings, state, checkpoint)
                rows.append(dict(identity, **result))
        except (ArithmeticError, ValueError, TimeoutError) as error:
            failures.append(dict(identity, reason=f'{type(error).__name__}: {error}'))
            print(f'Snapshot failed verification: {error}', flush=True)
    return rows, failures


def run_comparison(mode='test', *, max_sensor_bound=5000, targets=dual_search.TARGETS,
                   beam_width=3, max_iterations=30, time_limit=None, resume=True, run=True):
    if max_sensor_bound is None:
        return greedy_comparison.run_comparison(mode, targets=targets, beam_width=beam_width,
                    max_iterations=max_iterations, time_limit=time_limit, resume=resume, run=run)
    if type(max_sensor_bound) is not int or max_sensor_bound < 1:
        raise ValueError('MAX_SENSOR_BOUND must be a positive integer or None')
    if mode not in ('test', 'full'):
        raise ValueError('mode must be test or full')
    targets = tuple(float(e) for e in targets)
    if not targets or len(set(targets)) != len(targets) or any(not math.isfinite(e) or not 0 <= e < 1 for e in targets):
        raise ValueError('Targets must be distinct finite probabilities in [0,1)')
    if type(beam_width) is not int or beam_width < 1 or (max_iterations is not None and (type(max_iterations) is not int or max_iterations < 0)):
        raise ValueError('Invalid heuristic settings')
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError('time_limit must be positive or None')
    settings = dict(beam_width=beam_width, max_iterations=max_iterations, time_limit=time_limit)
    modules = dual_search.dependencies()
    original = modules['analysis'].select_cases(dual_search.PROBLEM2, 'extended', mode, exact_k_max=10)
    entries, skipped = partition(original, max_sensor_bound)
    signatures = {m: cache_signature(m, settings) for m in ('vnd_beam', 'greedy')}
    run_key = hashlib.sha256(json.dumps(dict(signatures=signatures, targets=targets,
                        max_sensor_bound=max_sensor_bound), sort_keys=True).encode()).hexdigest()[:16]
    output = dual_search.HERE / 'results/comparison_limited' / mode / run_key
    info = dict(mode=mode, targets=targets, methods=list(signatures), settings=settings,
                max_sensor_bound=max_sensor_bound, original_instances=len(original),
                included_instances=len(entries), skipped_instances=len(skipped),
                cache_signatures=signatures)
    dual_search.save_json(output / 'skipped.json', skipped)
    dual_search.save_json(output / 'run_info.json', info)
    print(f'Upper-bound cutoff: {max_sensor_bound}; included {len(entries)}/{len(original)}; '
          f'skipped {len(skipped)}. Existing search checkpoints are reused.', flush=True)
    rows, failures = [], []
    for method in signatures:
        method_rows, method_failures = run_method(method, entries, targets, settings, resume=resume, run=run)
        rows.extend(method_rows)
        failures.extend(method_failures)
        dual_search.save_json(output / 'solutions.json', rows)
        dual_search.save_json(output / 'failures.json', failures)
    return rows, failures, output
