"""Expected sensor-use summary from existing matched evaluations and saved trees."""
import hashlib
import json
from math import ceil, pi
from pathlib import Path

import numpy as np

from .cache import checkpoint_path
from .runner import _validate
from .sources import implementations


def tree_uses(tree, K, phases, allocation, node_probabilities, cache=None, *, include_rounds=False):
    """Return expected total uses and error, averaging uniformly over true fields."""
    cache = {} if cache is None else cache
    pending = [(tree, np.full(K, 1.0 / K))]
    uses = success = rounds = 0.0
    while pending:
        node, reach = pending.pop()
        if 'declared_field' in node:
            success += reach[int(node['declared_field'])]
            continue
        pair, location = tuple(node['pair']), int(node['location'])
        count = int(allocation[location])
        if count <= 0:
            raise ValueError('Saved tree tests an unoccupied location')
        gap = abs(float(phases[tuple(sorted(pair))][location]))
        # Above the perfect-discrimination threshold, Appendix D uses only this subset.
        used = min(count, ceil(pi / gap)) if gap else count
        reached = float(reach.sum())
        uses += reached * used
        rounds += reached
        key = pair, location, count
        if key not in cache:
            probabilities = np.array([node_probabilities(pair, location, count, k, phases).keep_i
                                      for k in range(K)])
            if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
                raise ValueError('Invalid node probabilities')
            cache[key] = probabilities
        keep_i = cache[key]
        # Include all paths, even when the true field has already been eliminated.
        pending.append((node['discard_i'], reach * (1.0 - keep_i)))
        pending.append((node['discard_j'], reach * keep_i))
    result = uses, float(1.0 - success)
    return (*result, rounds) if include_rounds else result


def mean_sensor_uses(rows, output, directory, manifest, config, progress=print):
    """Pool unique instance/budget rows equally; no searches, plots or file writes."""
    if (config.sequential_algorithm, config.joint_algorithm) != ('vnd_beam', 'greedy'):
        raise ValueError('This summary requires Sequential Heuristic and Joint Greedy')
    _, seq, _, _, _ = implementations()
    cases = {case['instance_id']: case for case in manifest['cases']}
    seen = set()
    current = None
    sequential_total = joint_total = 0.0
    totals = {d: dict(count=0, allocated=0.0, rounds=0.0, uses=0.0) for d in (1, 2)}
    for index, row in enumerate(rows, 1):
        key = row['instance_id'], row['N']
        if key in seen:
            raise ValueError(f'Duplicate instance/budget: {key}')
        seen.add(key)
        for side in ('sequential', 'joint'):
            if row[f'{side}_algorithm'] != getattr(config, f'{side}_algorithm'):
                raise ValueError('Row algorithms disagree with the summary configuration')
        case = cases[row['instance_id']]
        if row['dimension'] != case['dimension'] or row['K'] != case['K']:
            raise ValueError('Matched row and input manifest disagree')
        if current != row['instance_id']:
            path = Path(directory) / case['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != case['csv_sha256']:
                raise ValueError(f'Input checksum mismatch: {path}')
            K, M, phases = seq['input_data'].load_phase_input(path)
            if (K, M) != (case['K'], case['M']):
                raise ValueError('Input dimensions disagree with the manifest')
            probabilities = {}
            current = row['instance_id']
        saved = json.loads(checkpoint_path(output, config, 'sequential', case, row['N']).read_text())
        _validate(saved, case, row['N'], M, 'sequential')
        if saved['allocation'] != row['sequential_allocation'] or abs(saved['error'] - row['sequential_error']) > 1e-10:
            raise ValueError('Saved tree and matched result disagree')
        uses, error, rounds = tree_uses(saved['decision_tree'], K, phases, saved['allocation'],
                               seq['actual_error'].actual_node_probabilities, probabilities, include_rounds=True)
        if abs(error - row['sequential_error']) > 1e-9:
            raise ValueError('Node reach probabilities do not reproduce the saved error')
        if abs(rounds - (K - 1)) > 1e-9:
            raise ValueError('Saved sequential tree does not have K-1 sensing rounds')
        if sum(row['joint_allocation']) != row['N']:
            raise ValueError('Joint allocation does not match N')
        sequential_total += uses
        joint_total += row['N']
        group = totals[row['dimension']]
        group['count'] += 1
        group['allocated'] += row['N']
        group['rounds'] += rounds
        group['uses'] += uses
        if progress and index % 5000 == 0:
            progress(f'Saved sensor-use summary: {index:,}/{len(rows):,} evaluations')
    if not seen:
        raise ValueError('No matched evaluations to summarize')
    table = []
    for dimension, group in totals.items():
        if not group['count']:
            raise ValueError(f'No saved {dimension}D evaluations')
        n = group['count']
        for method in ('Sequential (Heuristic)', 'Joint (Greedy)'):
            sequential = method.startswith('Sequential')
            table.append({'Dataset': f'{dimension}D', 'Method': method,
                          'Mean allocated sensors (N)': group['allocated'] / n,
                          'Mean sensing rounds': group['rounds'] / n if sequential else 1,
                          'Mean expected total sensor uses': group['uses'] / n if sequential else group['allocated'] / n})
    return dict(table=table, evaluations_by_dimension={d: g['count'] for d, g in totals.items()},
                evaluations=len(seen), sequential_mean_expected_uses=sequential_total / len(seen),
                joint_mean_N=joint_total / len(seen))
