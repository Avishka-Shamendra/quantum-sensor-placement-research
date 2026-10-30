"""Generate single-location fields from the shared Table 1 models."""

import csv
import hashlib
import importlib.util
import json
from itertools import combinations
from math import isfinite, pi
from pathlib import Path
from random import Random

PROJECT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT.parent / 'problem-1-zero-error/src/field_models.py'
FAMILIES = ('gaussian', 'sinusoidal', 'quadratic')
KS = (3, 4, 5)
NS = (2, 4, 6, 8, 10)
INSTANCES = 50
DEFAULT_SEED = 20261030


def field_models():
    spec = importlib.util.spec_from_file_location('_single_location_fields', MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed(seed, dimension, family, k, inst):
    label = f'{seed}:{dimension}:{family}:{k}:{inst}'
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:16], 'big')


def _parameters(family, dimension, parameters):
    centre = parameters.get('centre')
    values = {}
    if family in ('gaussian', 'sinusoidal'):
        values['amp'] = parameters['amplitude']
    if centre is not None:
        if dimension == 1:
            values['mu'] = centre
        else:
            values.update(mu1=centre[0], mu2=centre[1])
    if family == 'gaussian':
        values['width'] = parameters['width']
    elif family == 'sinusoidal':
        values.update(q=parameters['frequency'], gamma=parameters['phase_offset'])
        if dimension == 2:
            values['theta'] = parameters['orientation']
    else:
        values.update(b=parameters['background'], a=parameters['curvature'])
    return values


def draw_instance(models, family, dimension, k, seed, min_phase_gap, max_draws):
    """Retain the first draw satisfying the requested phase-gap threshold."""
    rng = Random(seed)
    for attempt in range(1, max_draws + 1):
        location = models.sensing_locations(dimension, 1, rng)[0]
        fields, parameters = models.FIELD_MODELS[family](k, [location], rng)
        if family == 'sinusoidal' and dimension == 2:
            if any(p['orientation'] >= pi for p in parameters):
                continue
        phi = [values[0] for values in fields]
        gaps = [(i, j, models.principal_phase(phi[i] - phi[j]))
                for i, j in combinations(range(k), 2)]
        if not all(isfinite(value) for value in phi):
            raise ValueError('Nonfinite field value')
        if any(abs(gap) < min_phase_gap for _, _, gap in gaps):
            continue
        return location, parameters, phi, gaps, attempt
    raise RuntimeError(f'No valid {dimension}D {family} instance for K={k}, seed={seed}')


def _write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_datasets(dimensions=(1, 2), *, root=PROJECT, seed=DEFAULT_SEED,
                      min_phase_gap=0.01, max_draws=100_000, overwrite=False):
    """Write field values, signed phase differences, and generation metadata."""
    dimensions = tuple(dimensions)
    if not dimensions or len(set(dimensions)) != len(dimensions) or any(d not in (1, 2) for d in dimensions):
        raise ValueError('Select dimension 1, 2, or both')
    if not isfinite(min_phase_gap) or not 0 <= min_phase_gap <= pi:
        raise ValueError('min_phase_gap must lie in [0, pi]')
    if type(max_draws) is not int or max_draws < 1 or type(seed) is not int:
        raise ValueError('Use an integer seed and a positive integer max_draws')
    models = field_models()
    targets = [(d, f, Path(root) / 'inputs' / f'{d}d' / f)
               for d in dimensions for f in FAMILIES]
    for _, _, directory in targets:
        if not overwrite and any((directory / name).exists() for name in ('set.csv', 'r.csv', 'manifest.json')):
            raise FileExistsError(f'Dataset already exists: {directory}; use --overwrite to replace it')
    outputs = []
    for dimension, family, directory in targets:
        rows, phase_rows, instances = [], [], []
        for k in KS:
            for inst in range(INSTANCES):
                case_seed = _seed(seed, dimension, family, k, inst)
                location, parameters, phi, gaps, attempt = draw_instance(
                    models, family, dimension, k, case_seed, min_phase_gap, max_draws)
                for field, (parameters_i, value) in enumerate(zip(parameters, phi)):
                    row = dict(k=k, inst=inst, field=field)
                    row.update(_parameters(family, dimension, parameters_i))
                    row['x'] = location[0]
                    if dimension == 2:
                        row['y'] = location[1]
                    row.update(phi=value, draws=attempt)
                    rows.append(row)
                phase_rows.extend(dict(k=k, inst=inst, i=i, j=j, m=0, r_ijm=gap)
                                  for i, j, gap in gaps)
                instances.append(dict(k=k, inst=inst, seed=case_seed, draws=attempt,
                                      location=list(location), minimum_phase_gap=min(abs(g) for _, _, g in gaps)))
        directory.mkdir(parents=True, exist_ok=True)
        # Write the manifest last so it describes both completed CSV files.
        for name, data in (('set.csv', rows), ('r.csv', phase_rows)):
            temporary = directory / (name + '.tmp')
            _write_csv(temporary, data)
            temporary.replace(directory / name)
        manifest = dict(schema_version=1, dimension=dimension, family=family, M=1,
            ks=list(KS), ns=list(NS), instances_per_k=INSTANCES, instance_count=len(instances),
            instance_budget_count=len(instances)*len(NS), c=1.0, seed=seed,
            min_phase_gap=min_phase_gap, max_draws=max_draws,
            validity_rule='Every pair satisfies abs(r_ij0) >= min_phase_gap',
            formula=models.FIELD_FORMULAS[family][dimension],
            parameter_ranges=models.FIELD_PARAMETER_RANGES[family][dimension],
            location_distribution=f'Uniform on [0,1]^{dimension}',
            phase_convention='phi_i = c*f_i(x); r_ij0 = Arg(exp(i*(phi_i-phi_j))) in (-pi,pi]',
            indexing='inst and field are zero-based within each K; m=0',
            seed_derivation='First 128 bits of SHA256(seed:dimension:family:k:inst)',
            shared_instances_across_sensor_budgets=True,
            source_sha256={str(MODEL_PATH.relative_to(PROJECT.parent)): hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                           'src/dataset.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            files={name: dict(rows=len(data), sha256=hashlib.sha256((directory/name).read_bytes()).hexdigest())
                   for name, data in (('set.csv', rows), ('r.csv', phase_rows))},
            instances=instances)
        if dimension == 2 and family == 'sinusoidal':
            manifest['orientation_interval'] = '[0, pi)'
        temporary = directory / 'manifest.json.tmp'
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        temporary.replace(directory / 'manifest.json')
        print(f'{dimension}D {family}: {len(instances)} instances saved in {directory}', flush=True)
        outputs.append(directory)
    return outputs
