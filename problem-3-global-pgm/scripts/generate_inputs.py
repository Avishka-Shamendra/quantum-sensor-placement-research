"""Generate phase-gap CSVs using Problem 1's unchanged field ranges."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
from hashlib import sha256

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('problem1_fields', ROOT.parent / 'problem-1-zero-error/src/field_models.py')
fields = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fields)


def generate(config, output):
    required = ('dimension', 'models', 'k_values', 'm_values', 'replicates', 'phase_scale_c', 'sensor_budgets', 'seed')
    if any(key not in config for key in required):
        raise ValueError(f'Config must contain {required}')
    if config['dimension'] not in (1, 2):
        raise ValueError('dimension must be 1 or 2')
    for key, minimum in [('k_values', 2), ('m_values', 1), ('sensor_budgets', 0)]:
        if not config[key] or any(type(v) is not int or v < minimum for v in config[key]):
            raise ValueError(f'Invalid {key}')
    if type(config['replicates']) is not int or config['replicates'] < 1:
        raise ValueError('replicates must be positive')
    if not config['models'] or any(model not in fields.FIELD_MODELS for model in config['models']):
        raise ValueError('unknown field model')
    if type(config['seed']) is not int or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    for key in ('models', 'k_values', 'm_values', 'sensor_budgets'):
        if len(set(config[key])) != len(config[key]):
            raise ValueError(f'duplicate values in {key}')
    test_reps = config.get('test_replicates_per_cell', 1)
    if type(test_reps) is not int or not 1 <= test_reps <= config['replicates']:
        raise ValueError('test_replicates_per_cell must lie between 1 and replicates')
    # Validate c before creating any files.
    fields.phase_gap_rows([[0.0], [0.0]], config['phase_scale_c'])
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Refusing to replace nonempty dataset: {output}')
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for model in config['models']:
        for K in config['k_values']:
            for M in config['m_values']:
                for replicate in range(config['replicates']):
                    # Stable per-cell streams, independent of traversal order and
                    # other studies; no reuse of Problem 2's seed schedule.
                    identity = json.dumps(['problem3-v2', config['seed'], config.get('study', 'custom'),
                                           config['dimension'], model, K, M, replicate], separators=(',', ':'))
                    seed = int.from_bytes(sha256(identity.encode()).digest()[:16], 'big')
                    rows, metadata = fields.generate_case(
                        model, config['dimension'], K, M, seed, c=config['phase_scale_c'],
                        min_best_pair_phase_gap=fields.DEFAULT_MIN_BEST_PAIR_PHASE_GAP)
                    filename = f'case_{len(cases)+1:05d}.csv'
                    with (output / filename).open('w', newline='') as stream:
                        writer = csv.writer(stream)
                        writer.writerow(('i', 'j', 'm', 'r_ijm'))
                        writer.writerows((i, j, m, format(r, '.17g')) for i, j, m, r in rows)
                    cases.append(dict(metadata, file=filename, replicate=replicate+1,
                                      sensor_budgets=config['sensor_budgets']))
    manifest = dict(schema_version=2, design=dict(config, minimum_phase_gap=fields.DEFAULT_MIN_BEST_PAIR_PHASE_GAP),
                    seed_scheme='sha256/problem3-v2',
                    sampling='random field draws conditioned on max_m |r_ijm| >= 0.01 for every pair',
                    input_format={'columns': ['i', 'j', 'm', 'r_ijm'], 'indexing': 'zero-based'},
                    models={model: dict(formula=fields.FIELD_FORMULAS[model][config['dimension']],
                                        parameter_ranges=fields.FIELD_PARAMETER_RANGES[model][config['dimension']]) for model in config['models']},
                    cases=cases)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(generate(json.loads(args.config.read_text()), args.output))


if __name__ == '__main__':
    main()
