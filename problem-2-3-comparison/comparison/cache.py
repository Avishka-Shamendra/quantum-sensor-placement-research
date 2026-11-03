"""Per-algorithm cache identities; reporting and other algorithms are independent."""
import ast
import hashlib
from functools import lru_cache
from pathlib import Path

from .config import fingerprint
from .sources import ROOT, SEQUENTIAL, JOINT


@lru_cache(maxsize=None)
def algorithm_sources(side, algorithm):
    if side == 'sequential':
        names = ('input_data', 'actual_error', 'strategy',
                 'heuristic' if algorithm == 'vnd_beam' else 'greedy')
        paths = [SEQUENTIAL / f'{name}.py' for name in names]
    else:
        paths = [JOINT / f'{name}.py' for name in ('input_data', 'probes', 'pgm')]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if side == 'joint':
        names = {'greedy_search', 'SearchResult'} if algorithm == 'greedy' else {
            'vnd_search', '_validate_phases', 'separation_score', 'AllocationResult'}
        nodes = [n for n in ast.parse((JOINT/'search.py').read_text()).body
                 if getattr(n, 'name', None) in names]
        hashes['joint_search_functions'] = fingerprint([ast.dump(n) for n in nodes])
    return hashes


def evaluation_spec(config, side, case):
    algorithm = getattr(config, f'{side}_algorithm')
    spec = dict(schema=1, method=side, algorithm=algorithm,
                sources=algorithm_sources(side, algorithm), input_sha256=case['csv_sha256'])
    if side == 'sequential' and algorithm == 'vnd_beam':
        spec.update(beam_width=config.beam_width, max_iterations=config.sequential_max_iterations,
                    random_seed=case['generation_seed'], move_sizes=[1, 2, 3])
    if side == 'joint':
        spec.update(probe='local_dicke', c=1.0)
        if algorithm == 'vnd':
            spec.update(move_sizes=[1, 2, 3], tolerance=1e-12)
    return spec


def checkpoint_path(output, config, side, case, budget):
    spec = evaluation_spec(config, side, case)
    return (Path(output) / 'cache' / side / spec['algorithm'] / case['instance_id'] /
            f'n{budget}_{fingerprint(spec)[:20]}.json')
