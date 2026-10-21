"""Two publication figures and a runtime table from saved Problem 3 runs."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from publication import SUCCESS, estimate
from src.probes import FAMILIES, DICKE_POLICY
from src.search import ALGORITHM_POLICY

# Problem 2's Exhaustive, Greedy, Heuristic, Uniform, and Random palette.
METHOD_STYLES = {
    'exhaustive': ('Exhaustive', '#1f77b4', 'o', '-'),
    'greedy': ('Greedy', '#d62728', 's', '--'),
    'vnd': ('VND', '#2ca02c', '^', '-.'),
    'near_uniform': ('Near-Uniform', '#9467bd', 'D', ':'),
    'random': ('Random', '#8c564b', 'x', '--'),
}
STATE_STYLES = dict(zip(FAMILIES, [
    ('Product', '#1f77b4', 'o', '-'), ('Local GHZ', '#d62728', 's', '--'),
    ('Global GHZ', '#2ca02c', '^', '-.'), ('Local Dicke', '#9467bd', 'D', ':'),
    ('Global Dicke', '#8c564b', 'x', '--'),
]))
STYLE = {
    'font.family': 'Times New Roman', 'font.size': 12, 'axes.labelsize': 12,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
}
STUDIES = ('compact', 'extended')
ALLOCATION_METHODS = ('greedy', 'vnd', 'near_uniform', 'random')


def load_saved_runs(root):
    """Require complete current-policy coverage; never start or resume a solver."""
    root = Path(root)
    datasets, provenance = {}, {}
    for study in STUDIES:
        run = root / 'results' / study / 'full'
        info = json.loads((run / 'run_info.json').read_text())
        settings = info['settings']
        expected = dict(algorithm_policy=ALGORITHM_POLICY, dicke_policy=DICKE_POLICY,
                        study=study, mode='full')
        mismatches = [f"{key}: saved={settings.get(key)!r}, expected={value!r}"
                      for key, value in expected.items() if settings.get(key) != value]
        if mismatches:
            raise ValueError(f"Incompatible saved run: {run}; " + '; '.join(mismatches))
        required_methods = (*ALLOCATION_METHODS, 'exhaustive') if study == 'compact' else ALLOCATION_METHODS
        expected, manifests = {}, {}
        for dimension in (1, 2):
            path = root / 'inputs' / study / f'{dimension}d' / 'manifest.json'
            manifests[str(dimension)] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = json.loads(path.read_text())
            for case in manifest['cases']:
                for budget in case['budget_schedule']:
                    expected[dimension, case['file'], budget['N']] = (case, budget)
        records_path = run / 'records.jsonl'
        records = {}
        with records_path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record['method'] in required_methods:
                    key = (record['dimension'], record['case'], record['N'], record['probe'], record['method'])
                    records[key] = record
        selected = []
        for key, (case, budget) in expected.items():
            for probe in FAMILIES:
                for method in required_methods:
                    record = records.get((*key, probe, method))
                    if record is None or record['status'] not in SUCCESS:
                        raise ValueError(f'Missing completed record: {study}, {key}, {probe}, {method}')
                    if (record.get('algorithm_policy') != ALGORITHM_POLICY
                            or record.get('dicke_policy') != DICKE_POLICY
                            or record['K'] != case['K'] or record['M'] != case['M']
                            or record['field_model'] != case['field_model']
                            or record['N_LB_joint'] != case['N_LB_joint']
                            or record['target_fractions'] != budget['target_fractions']
                            or not np.isfinite(record['pgm_error'])):
                        raise ValueError(f'Record/manifest mismatch: {study}, {key}, {probe}, {method}')
                    selected.append(record)
        datasets[study] = selected
        provenance[study] = dict(results_directory=str(run.relative_to(root)), settings=settings,
                                 records_sha256=hashlib.sha256(records_path.read_bytes()).hexdigest(),
                                 manifest_sha256=manifests, case_budgets=len(expected),
                                 completed_records=len(selected))
    return datasets, provenance


def _panels(ylabels, xlabel=None):
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), sharey='row')
    figure.subplots_adjust(left=.12, right=.99, bottom=.14, top=.94, hspace=.38, wspace=.12)
    titles = (
        '(a) One-dimensional compact dataset',
        '(b) Two-dimensional compact dataset',
        '(c) One-dimensional extended dataset',
        '(d) Two-dimensional extended dataset',
    )
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title, fontsize=12)
        axis.grid(axis='y', alpha=.2, zorder=0)
        if xlabel:
            axis.set_xlabel(xlabel)
    for axis, label in zip(axes[:, 0], ylabels):
        axis.set_ylabel(label)
    return figure, axes


def _line(axis, x, stats, style):
    label, color, marker, line = style
    axis.plot(x, [v['mean'] for v in stats], label=label, color=color,
              marker=marker, linestyle=line, markersize=4, lw=1.2)


def _save(figure, styles, output, name, data, *, bars=False):
    handles = ([Patch(label=s[0], facecolor=s[1], edgecolor='none', alpha=.85)
                for s in styles] if bars else
               [Line2D([], [], label=s[0], color=s[1], marker=s[2], linestyle=s[3])
                for s in styles])
    figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .025),
                  ncol=len(handles), frameon=False)
    figure.savefig(output / f'{name}.pdf')
    figure.savefig(output / f'{name}.png', dpi=600)
    with (output / f'{name}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def create_outputs(datasets, output):
    """Export two four-panel mean-error figures and a Compact runtime table."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figures = []
    def stats(rows, value):
        clusters = defaultdict(list)
        for record in rows:
            clusters[record['dimension'], record['case']].append(record[value])
        if not clusters:
            raise ValueError('No saved observations for this summary.')
        return dict(mean=float(np.mean([np.mean(v) for v in clusters.values()])),
                    instances=len(clusters))

    with plt.rc_context(STYLE):
        figure, axes = _panels(['Mean PGM error', 'Mean PGM error'],
                              r'Mean normalized budget, $N/N_{\mathrm{LB}}^{\mathrm{joint}}$')
        data = []
        for row, study in enumerate(STUDIES):
            reference = 'exhaustive' if study == 'compact' else 'vnd'
            for column, dimension in enumerate((1, 2)):
                for probe in FAMILIES:
                    selected = [r for r in datasets[study] if r['dimension'] == dimension
                                and r['probe'] == probe and r['method'] == reference]
                    points, x = [], []
                    for fraction in (.25, .5, .75, 1.):
                        group = [r for r in selected if fraction in r['target_fractions']]
                        value = stats(group, 'pgm_error')
                        ratio = float(np.mean([r['budget_ratio'] for r in group]))
                        points.append(value)
                        x.append(ratio)
                        data.append(dict(study=study, dimension=dimension, probe=probe,
                                         method=reference, target_fraction=fraction,
                                         mean_budget_ratio=ratio, **value))
                    _line(axes[row, column], x, points, STATE_STYLES[probe])
                axes[row, column].set_ylim(0, 1)
        _save(figure, STATE_STYLES.values(), output, 'figure_1_input_states', data)
        figures.append(figure)

        figure, axes = _panels(['Mean error gap from Exhaustive', 'Mean error difference from VND'])
        data = []
        for row, study in enumerate(STUDIES):
            reference = 'exhaustive' if study == 'compact' else 'vnd'
            pairs = defaultdict(dict)
            for r in datasets[study]:
                pairs[r['dimension'], r['case'], r['N'], r['probe']][r['method']] = r
            bar_methods = tuple(m for m in ALLOCATION_METHODS
                                if study == 'compact' or m != 'vnd')
            gaps = []
            for group in pairs.values():
                for method in bar_methods:
                    r = group[method]
                    gap = r['pgm_error'] - group[reference]['pgm_error']
                    if study == 'compact' and gap < -1e-7:
                        raise ValueError('Saved result beats Exhaustive beyond numerical tolerance.')
                    gaps.append(dict(r, gap=gap))
            for column, dimension in enumerate((1, 2)):
                axis = axes[row, column]
                for index, method in enumerate(bar_methods):
                    points = [estimate([r for r in gaps if r['dimension'] == dimension
                                     and r['method'] == method and r['probe'] == probe], 'gap')
                              for probe in FAMILIES]
                    x = np.arange(len(FAMILIES)) + (index - (len(bar_methods) - 1) / 2) * .19
                    axis.bar(x, [v['mean'] for v in points], width=.19,
                             color=METHOD_STYLES[method][1], alpha=.85, zorder=2)
                    low = np.array([v['low'] for v in points])
                    high = np.array([v['high'] for v in points])
                    axis.vlines(x, low, high, color='black', lw=.65, zorder=3)
                    axis.hlines(low, x-.035, x+.035, color='black', lw=.65, zorder=3)
                    axis.hlines(high, x-.035, x+.035, color='black', lw=.65, zorder=3)
                    data.extend(dict(study=study, dimension=dimension, probe=probe,
                                     method=method, reference=reference, **value)
                                for probe, value in zip(FAMILIES, points))
                axis.axhline(0, color='.4', lw=.7)
                axis.set_xticks(np.arange(5), ['Product', 'Local\nGHZ', 'Global\nGHZ',
                                             'Local\nDicke', 'Global\nDicke'])
        _save(figure, [METHOD_STYLES[m] for m in ALLOCATION_METHODS], output,
              'figure_2_sensor_allocation', data, bars=True)
        figures.append(figure)

    table = []
    for dimension in (1, 2):
        for method in ('exhaustive', 'greedy', 'vnd'):
            selected = []
            for record in datasets['compact']:
                if record['dimension'] != dimension or record['method'] != method:
                    continue
                seconds = (record['search_runtime_seconds'] if method == 'exhaustive'
                           else record['allocation_runtime_seconds'] + record['pgm_evaluation_seconds'])
                if not np.isfinite(seconds) or seconds <= 0:
                    raise ValueError('Missing or nonpositive saved runtime.')
                selected.append(dict(record, standalone_ms=1000 * seconds))
            value = stats(selected, 'standalone_ms')
            table.append(dict(dimension=f'{dimension}D', method=METHOD_STYLES[method][0],
                              mean_ms=value['mean'], instances=value['instances']))
    with (output / 'table_1_runtime.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = ['| Dimension | Method | Mean runtime (ms) |',
             '|---|---|---:|']
    for row in table:
        lines.append(f"| {row['dimension']} | {row['method']} | {row['mean_ms']:.3f} |")
    markdown = '\n'.join(lines)
    (output / 'table_1_runtime.md').write_text(markdown + '\n')
    return figures, markdown
