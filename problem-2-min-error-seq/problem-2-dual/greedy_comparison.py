"""Independent Greedy budget search and paired comparison; original VND caches retained."""
from pathlib import Path
import hashlib
import importlib
import json
from types import SimpleNamespace
import math
from dual_search import (HERE, PROBLEM2, REPOSITORY, TARGETS, dependencies,
                         save_json, initial_solution, budget_search, run_study)


def run_greedy_study(mode='test', *, targets=TARGETS, beam_width=3, max_iterations=30,
              time_limit=None, resume=True, run=True):
    """Read existing Extended inputs; checkpoint each instance/target bisection."""
    if mode not in ('test', 'full'):
        raise ValueError('mode must be test or full')
    targets = tuple(float(e) for e in targets)
    if not targets or len(set(targets)) != len(targets) or any(not math.isfinite(e) or not 0 <= e < 1 for e in targets):
        raise ValueError('targets must be distinct finite probabilities in [0,1)')
    if type(beam_width) is not int or beam_width < 1 or (max_iterations is not None and (type(max_iterations) is not int or max_iterations < 0)):
        raise ValueError('Invalid heuristic settings')
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError('time_limit must be positive or None')
    modules = dependencies()
    greedy = importlib.import_module('greedy').greedy
    def greedy_candidate(K, M, N, phases, **options):
        return greedy(K, M, N, phases, time_limit=options['time_limit'])
    modules['heuristic'] = SimpleNamespace(heuristic=greedy_candidate)
    entries = modules['analysis'].select_cases(PROBLEM2, 'extended', mode, exact_k_max=10)
    settings = dict(beam_width=beam_width, max_iterations=max_iterations, time_limit=time_limit)
    paths = sorted((PROBLEM2/'src').glob('*.py')) + [Path(__file__)] + [
        REPOSITORY/'problem-1-zero-error/src'/f'{name}.py' for name in ('optimal','phase_data')]
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for path in paths:
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    digest.update((HERE/'dual_search.py').read_bytes())
    signature = digest.hexdigest()
    legacy_digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for path in paths:
        original = HERE/'dual_search.py' if path == Path(__file__) else path
        legacy_digest.update(original.name.encode()); legacy_digest.update(original.read_bytes())
    legacy_signature = legacy_digest.hexdigest()
    output = HERE/'results/greedy'/mode/signature[:16]
    output.mkdir(parents=True, exist_ok=True)
    rows = []; failures = []
    for index, (dataset, path, case) in enumerate(entries, 1):
        print(f'Snapshot {index}/{len(entries)} | {case["spatial_dimension"]}D | {case["field_model"]} | K={case["K"]}, M={case["M"]}', flush=True)
        case_hash = hashlib.sha256(signature.encode()+path.read_bytes()+json.dumps(case,sort_keys=True).encode()).hexdigest()
        cache_path = HERE/'results/cache'/f'{case_hash}.json'
        cache = json.loads(cache_path.read_text()) if resume and cache_path.exists() else {}
        if not resume and cache_path.exists():
            raise FileExistsError('Existing cache: use resume=True or explicitly remove the cache')
        if not cache and resume:
            legacy_hash = hashlib.sha256(legacy_signature.encode()+path.read_bytes()+json.dumps(case,sort_keys=True).encode()).hexdigest()
            legacy_path = HERE/'results/cache'/f'{legacy_hash}.json'
            if legacy_path.exists():
                original_cache = json.loads(legacy_path.read_text())
                if 'perfect' in original_cache:
                    cache = dict(perfect=original_cache['perfect'],targets={})
        identity = dict(method='greedy',dimension=case['spatial_dimension'], case=case['file'], field_model=case['field_model'],
                        K=case['K'], M=case['M'], N_min_seq=case['minimum_sensors_for_zero_error'])
        K, M, phases = modules['input_data'].load_phase_input(path)
        if (K,M) != (case['K'],case['M']):
            raise ValueError('Manifest and CSV dimensions disagree')
        checkpoint = lambda: save_json(cache_path, cache)
        try:
            if 'perfect' not in cache:
                if not run: raise FileNotFoundError('No matching perfect-allocation cache; enable RUN_ANALYSIS')
                cache.update(perfect=initial_solution(K,M,phases,identity['N_min_seq'],modules), targets={})
                checkpoint()
            for epsilon in targets:
                state = cache['targets'].setdefault(str(epsilon), {})
                if not run and not state.get('complete'):
                    raise FileNotFoundError('No complete matching target cache; enable RUN_ANALYSIS')
                result = budget_search(K,M,phases,epsilon,cache['perfect'],modules,settings,state,checkpoint)
                rows.append(dict(identity, **result))
        except (ArithmeticError, ValueError, TimeoutError) as error:
            failures.append(dict(identity, reason=f'{type(error).__name__}: {error}'))
            print(f'Snapshot failed verification: {error}', flush=True)
    save_json(output/'solutions.json', rows)
    save_json(output/'failures.json', failures)
    save_json(output/'run_info.json', dict(mode=mode,targets=targets,settings=settings,signature=signature,
        selected_instances=len(entries),verified_solutions=len(rows),failed_instances=len(failures),
        source_files={str(p.relative_to(REPOSITORY)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    return rows, failures, output


def run_comparison(mode='test', **options):
    vnd_rows, vnd_failures, vnd_output = run_study(mode, **options)
    greedy_rows, greedy_failures, greedy_output = run_greedy_study(mode, **options)
    rows = [dict(r,method='vnd_beam') for r in vnd_rows] + greedy_rows
    failures = [dict(r,method='vnd_beam') for r in vnd_failures] + greedy_failures
    output = HERE/'results/comparison'/mode/greedy_output.name
    save_json(output/'solutions.json',rows)
    save_json(output/'failures.json',failures)
    save_json(output/'run_info.json',dict(mode=mode,targets=list(options.get('targets',TARGETS)),
        methods=['vnd_beam','greedy'],vnd_results=str(vnd_output),greedy_results=str(greedy_output)))
    return rows, failures, output
