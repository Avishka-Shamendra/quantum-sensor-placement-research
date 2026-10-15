"""Minimum verified feasible budgets using unchanged sequential heuristics."""
from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
PROBLEM2 = HERE.parent
REPOSITORY = PROBLEM2.parent
TARGETS = (0.1, 0.2, 0.3, 0.4, 0.5)


def dependencies():
    """Use the original flat modules, rejecting incompatible shared-kernel imports."""
    source = PROBLEM2 / 'src'
    for path in source.glob('*.py'):
        loaded = sys.modules.get(path.stem)
        if loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != path.resolve():
            raise RuntimeError(f'Restart the kernel: {path.stem} is loaded from another project')
    sys.path.insert(0, str(source))
    modules = {name: importlib.import_module(name) for name in
               ('heuristic', 'actual_error', 'strategy', 'input_data', 'analysis')}
    for name in ('optimal', 'phase_data'):
        path = REPOSITORY / 'problem-1-zero-error' / 'src' / f'{name}.py'
        spec = importlib.util.spec_from_file_location(f'dual_problem1_{name}', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def perfect_tree(remaining, locations):
    """Lexicographic elimination tree from Problem 1's covering pair locations."""
    if len(remaining) == 1:
        return {'declared_field': remaining[0]}
    i, j = remaining[:2]
    return dict(pair=[i, j], location=locations[i][j], pair_error=0.0,
                discard_i=perfect_tree(tuple(k for k in remaining if k != i), locations),
                discard_j=perfect_tree(tuple(k for k in remaining if k != j), locations))


def verify(solution, K, M, phases, evaluator, epsilon=None):
    allocation = solution['allocation']
    if len(allocation) != M or any(type(n) is not int or n < 0 for n in allocation):
        raise ValueError('Invalid returned allocation')
    if sum(allocation) != solution['N']:
        raise ValueError('Returned allocation does not match the budget')
    error = evaluator(solution['decision_tree'], K, phases, allocation)
    if not math.isfinite(error) or not -1e-12 <= error <= 1 + 1e-12:
        raise ValueError('Invalid actual decision-tree error')
    if epsilon is not None and error > epsilon:
        raise ValueError('Returned solution does not meet its target error')
    solution['actual_error'] = error
    return solution


def initial_solution(K, M, phases, upper, modules):
    start = perf_counter()
    thresholds = modules['phase_data'].compute_nijm(K, M, phases)
    minimum, allocation, locations = modules['optimal'].optimal_perfect_pairwise_sensor_placement(K, M, thresholds)
    if minimum != upper:
        raise ValueError(f'Saved sequential minimum {upper} disagrees with Problem 1 result {minimum}')
    solution = dict(N=minimum, allocation=list(allocation), decision_tree=perfect_tree(tuple(range(K)), locations))
    verify(solution, K, M, phases, modules['actual_error'].actual_tree_error)
    if abs(solution['actual_error']) > 1e-10:
        raise ValueError('Perfect-discrimination fallback failed full-tree error verification')
    solution['runtime_seconds'] = perf_counter() - start
    return solution


def budget_search(K, M, phases, epsilon, perfect, modules, settings, state, checkpoint):
    """Bisection on observed heuristic feasibility, not a monotonicity certificate."""
    evaluator = modules['actual_error'].actual_tree_error
    if not state:
        zero = dict(N=0, allocation=[0]*M, decision_tree={'declared_field': 0})
        verify(zero, K, M, phases, evaluator)
        if zero['actual_error'] <= epsilon:
            state.update(low=0, high=0, best=zero, trace=[], search_seconds=0., complete=True)
        else:
            best = verify(dict(perfect), K, M, phases, evaluator, epsilon)
            state.update(low=0, high=perfect['N'], best=best, trace=[], search_seconds=0., complete=False)
        checkpoint()
    state.pop('interruption', None)
    while state['high'] - state['low'] > 1:
        N = (state['low'] + state['high']) // 2
        start = perf_counter()
        try:
            result = modules['heuristic'].heuristic(K, M, N, phases,
                beam_width=settings['beam_width'], max_iterations=settings['max_iterations'],
                improvement_threshold=0.0, initialization='near_uniform', random_seed=0,
                include_greedy_start=False, move_sizes=(1, 2, 3), time_limit=settings['time_limit'])
            solution = dict(N=N, allocation=list(result.allocation),
                            decision_tree=modules['strategy'].tree_to_dict(result.decision_tree))
            verify(solution, K, M, phases, evaluator)
            if abs(solution['actual_error'] - result.actual_error) > 1e-10:
                raise ValueError('Independent evaluation disagrees with heuristic result')
        except TimeoutError as error:
            elapsed = perf_counter() - start
            state['search_seconds'] += elapsed
            state['trace'].append(dict(N=N, status='timeout', runtime_seconds=elapsed))
            state['interruption'] = str(error)
            checkpoint()
            break
        elapsed = perf_counter() - start
        state['search_seconds'] += elapsed
        feasible = solution['actual_error'] <= epsilon
        state['trace'].append(dict(N=N, actual_error=solution['actual_error'], feasible=feasible,
                                   status='evaluated', runtime_seconds=elapsed))
        if feasible:
            state['best'] = solution
            state['high'] = N
        else:
            state['low'] = N
        checkpoint()
    state['complete'] = state['high'] - state['low'] <= 1
    verify(state['best'], K, M, phases, evaluator, epsilon)
    checkpoint()
    return dict(state['best'], epsilon=epsilon, verified_feasible=True,
                status='completed' if state['complete'] else 'interrupted_feasible',
                runtime_seconds=perfect['runtime_seconds'] + state['search_seconds'],
                perfect_setup_seconds=perfect['runtime_seconds'], search_seconds=state['search_seconds'],
                budget_evaluations=sum(t['status']=='evaluated' for t in state['trace']),
                low=state['low'], high=state['high'], search_trace=state['trace'])


def run_study(mode='test', *, targets=TARGETS, beam_width=3, max_iterations=30,
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
    entries = modules['analysis'].select_cases(PROBLEM2, 'extended', mode, exact_k_max=10)
    settings = dict(beam_width=beam_width, max_iterations=max_iterations, time_limit=time_limit)
    paths = sorted((PROBLEM2/'src').glob('*.py')) + [Path(__file__)] + [
        REPOSITORY/'problem-1-zero-error/src'/f'{name}.py' for name in ('optimal','phase_data')]
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for path in paths:
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    signature = digest.hexdigest()
    output = HERE/'results'/mode/signature[:16]
    output.mkdir(parents=True, exist_ok=True)
    rows = []; failures = []
    for index, (dataset, path, case) in enumerate(entries, 1):
        print(f'Snapshot {index}/{len(entries)} | {case["spatial_dimension"]}D | {case["field_model"]} | K={case["K"]}, M={case["M"]}', flush=True)
        case_hash = hashlib.sha256(signature.encode()+path.read_bytes()+json.dumps(case,sort_keys=True).encode()).hexdigest()
        cache_path = HERE/'results/cache'/f'{case_hash}.json'
        cache = json.loads(cache_path.read_text()) if resume and cache_path.exists() else {}
        if not resume and cache_path.exists():
            raise FileExistsError('Existing cache: use resume=True or explicitly remove the cache')
        identity = dict(dimension=case['spatial_dimension'], case=case['file'], field_model=case['field_model'],
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
