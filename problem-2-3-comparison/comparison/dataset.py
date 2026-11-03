"""Common random inputs, physical validity checks, and lower-bound budgets."""
import hashlib
import json
from itertools import product
from pathlib import Path

from .config import FAMILIES, atomic_json, budget_schedule, fingerprint
from .sources import implementations, source_hashes


def prepare_dataset(project, config):
    """Generate/load independent draws; cache each instance before proceeding."""
    fields, seq, _, joint_input, normalization = implementations()
    hashes = source_hashes()
    data_mode = 'test' if config.mode == 'compact' else config.mode
    design = dict(schema=1, mode=data_mode, seed=config.seed, K=config.sizes,
                  M=config.sizes, replicates=config.replicates, c=1.0,
                  minimum_best_pair_phase_gap=0.0, source_sha256=hashes)
    # Search/report changes must not regenerate common random inputs.
    required = ('mode', 'seed', 'K', 'M', 'replicates', 'c', 'minimum_best_pair_phase_gap')
    for saved in sorted((Path(project) / 'data' / data_mode).glob('*/manifest.json')):
        manifest = json.loads(saved.read_text())
        if all(json.dumps(manifest['design'][key]) == json.dumps(design[key]) for key in required):
            for case in manifest['cases']:
                if hashlib.sha256((saved.parent / case['file']).read_bytes()).hexdigest() != case['csv_sha256']:
                    raise ValueError(f'Input checksum mismatch: {case["file"]}')
            return saved.parent, manifest
    # Only data-generating dependencies belong in a new dataset identity.
    design['source_sha256'] = {k: v for k, v in hashes.items() if
        k.endswith(('field_models.py', 'input_data.py', 'normalization.py', 'optimal.py', 'phase_data.py'))}
    dataset_id = fingerprint(design)
    directory = Path(project) / 'data' / data_mode / dataset_id[:16]
    directory.mkdir(parents=True, exist_ok=True)
    cases = []
    total = 6 * len(config.sizes)**2 * config.replicates
    for dim, family, K, M, rep in product((1, 2), FAMILIES, config.sizes,
                                         config.sizes, range(1, config.replicates + 1)):
        name = f'{dim}d_{family}_k{K}_m{M}_{rep:02d}'
        csv_path, meta_path = directory / f'{name}.csv', directory / f'{name}.json'
        if meta_path.exists():
            case = json.loads(meta_path.read_text())
            if hashlib.sha256(csv_path.read_bytes()).hexdigest() != case['csv_sha256']:
                raise ValueError(f'Input checksum mismatch: {csv_path}')
        else:
            seed = int(fingerprint([config.seed, dim, family, K, M, rep])[:32], 16)
            rows, metadata = fields.generate_case(
                family, dim, K, M, seed=seed, c=1.0, max_attempts=1,
                min_best_pair_phase_gap=0.0,
            )
            seq['input_data'].write_phase_input(csv_path, rows)
            seq['input_data'].load_phase_input(csv_path)
            joint_input.load_phase_input(csv_path)
            minimum_gap = min(fields.best_pair_phase_gaps(rows).values())
            case = dict(metadata, instance_id=name, dimension=dim, family=family,
                        replicate=rep, file=csv_path.name,
                        csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                        observed_min_best_pair_phase_gap=minimum_gap)
            if minimum_gap == 0:
                case.update(status='excluded', exclusion_reason='indistinguishable_pair',
                            N_LB_joint=None, N_max=None, budget_schedule=[])
            else:
                bound = normalization.joint_sensor_lower_bound(csv_path)
                maximum, schedule = budget_schedule(bound)
                case.update(N_LB_joint=bound, N_max=maximum, budget_schedule=schedule,
                            status='eligible' if schedule else 'excluded',
                            exclusion_reason=None if schedule else 'no_positive_budget')
            atomic_json(meta_path, case)
        cases.append(case)
        if len(cases) % 50 == 0 or len(cases) == total:
            print(f'Inputs ready: {len(cases)}/{total}', flush=True)
    manifest = dict(dataset_id=dataset_id, design=design, cases=cases,
                    models={f: dict(formula=fields.FIELD_FORMULAS[f],
                                    parameter_ranges=fields.FIELD_PARAMETER_RANGES[f])
                            for f in FAMILIES})
    atomic_json(directory / 'manifest.json', manifest)
    return directory, manifest
