"""The two-panel error figure and matched field-family summary."""
from collections import defaultdict
import csv

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .config import FAMILIES, FRACTIONS, atomic_json

METHODS = ('sequential', 'joint')
STYLES = {'sequential': ('Sequential', '#2ca02c', '^', '-.'),
          'joint': ('Joint', '#9467bd', 'D', ':')}
STYLE = {
    'font.family': 'Times New Roman', 'font.size': 12, 'axes.titlesize': 12,
    'axes.labelsize': 12, 'legend.fontsize': 12, 'xtick.labelsize': 12,
    'ytick.labelsize': 12, 'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
}


def paired_interval(rows, samples, rng):
    """Pointwise stratified instance bootstrap, sharing draws across methods."""
    if not rows:
        return None
    strata = defaultdict(list)
    seen = set()
    for row in rows:
        if row['instance_id'] in seen:
            raise ValueError('An instance must occur once per target fraction.')
        seen.add(row['instance_id'])
        strata[row['family'], row['K'], row['M']].append(
            [row[f'{method}_error'] for method in METHODS])
    mean = np.mean([v for values in strata.values() for v in values], axis=0)
    bootstrap = np.zeros((samples, 2))
    for key in sorted(strata):
        values = np.asarray(strata[key])
        indices = rng.integers(len(values), size=(samples, len(values)))
        bootstrap += values[indices].sum(axis=1)
    bootstrap /= len(rows)
    low, high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return dict(n=len(rows), singleton_strata=sum(len(v) == 1 for v in strata.values()),
                mean=mean.tolist(), low=low.tolist(), high=high.tolist())


def family_table(rows):
    """Give each instance equal weight after averaging its unique budgets."""
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row['instance_id'], row['N'])
        if key in seen:
            raise ValueError('Duplicate instance/budget in matched results.')
        seen.add(key)
        grouped[row['dimension'], row['family'], row['instance_id']].append(
            [row[f'{m}_error'] for m in METHODS])
    table = []
    for dimension in (1, 2):
        for family in FAMILIES:
            values = [np.mean(v, axis=0) for (d, f, _), v in grouped.items()
                      if d == dimension and f == family]
            means = np.mean(values, axis=0) if values else (None, None)
            table.append({'Dimension': f'{dimension}D', 'Field Family': family.title(),
                          'Sequential Mean Error': None if means[0] is None else float(means[0]),
                          'Joint Mean Error': None if means[1] is None else float(means[1])})
    return table


def make_report(rows, output, config):
    """Export only Figure 1 and Table 1, plus machine-readable plot estimates."""
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.bootstrap_seed)
    estimates = []
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True)
        figure.subplots_adjust(left=.10, right=.99, bottom=.27, top=.90, wspace=.12)
        for dimension, axis in zip((1, 2), axes):
            points = []
            for fraction in FRACTIONS:
                selected = [r for r in rows if r['dimension'] == dimension
                            and fraction in r['target_fractions']]
                interval = paired_interval(selected, config.bootstrap_samples, rng)
                estimates.append(dict(dimension=dimension, fraction=fraction, estimate=interval))
                points.append(interval)
            for method_index, method in enumerate(METHODS):
                label, color, marker, line = STYLES[method]
                def series(key):
                    return [p[key][method_index] if p else np.nan for p in points]
                axis.plot(FRACTIONS, series('mean'), label=label, color=color,
                          marker=marker, linestyle=line, markersize=3.5, lw=1.1)
                axis.fill_between(FRACTIONS, series('low'), series('high'),
                                  color=color, alpha=.13, lw=0)
            axis.set_title(f'({"a" if dimension == 1 else "b"}) {dimension}D')
            axis.set_xlabel('Target sensor-budget fraction')
            axis.set_xticks(FRACTIONS)
            axis.set_ylim(-.02, 1.02)
            axis.grid(axis='y', alpha=.2, zorder=0)
        axes[0].set_ylabel('Mean discrimination error')
        handles = [Line2D([], [], label=s[0], color=s[1], marker=s[2], linestyle=s[3])
                   for s in STYLES.values()]
        figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .01),
                      ncol=2, frameon=False)
        figure.savefig(output / 'figure_1_sequential_vs_joint_error.pdf')
        figure.savefig(output / 'figure_1_sequential_vs_joint_error.png', dpi=600)
    atomic_json(output / 'figure_1_estimates.json', estimates)
    table = family_table(rows)
    with (output / 'table_1_field_family_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    headings = list(table[0])
    lines = ['| ' + ' | '.join(headings) + ' |', '|---|---|---:|---:|']
    for row in table:
        values = [f'{v:.6f}' if isinstance(v, float) else ('—' if v is None else v)
                  for v in row.values()]
        lines.append('| ' + ' | '.join(values) + ' |')
    markdown = '\n'.join(lines)
    (output / 'table_1_field_family_comparison.md').write_text(markdown + '\n')
    return figure, markdown
