"""State and algorithm comparisons in Problem 2's plotting style."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .probes import FAMILIES, DICKE_POLICY
from .search import ALGORITHM_POLICY

METHOD_STYLES = {
    'exhaustive': ('Exhaustive', '#1f77b4', 'o', '-'),
    'greedy': ('Greedy', '#d62728', 's', '--'),
    'vnd': ('VND', '#2ca02c', '^', '-.'),
}
STATE_STYLES = {
    'product': ('Product', '#1f77b4', 'o', '-'),
    'local_ghz': ('Local GHZ', '#d62728', 's', '--'),
    'global_ghz': ('Global GHZ', '#2ca02c', '^', '-.'),
    'local_dicke': ('Local Dicke', '#9467bd', 'D', ':'),
    'global_dicke': ('Global Dicke', '#8c564b', 'x', '--'),
}
TITLES = {1: '(a) One-dimensional dataset', 2: '(b) Two-dimensional dataset'}


def _figure(title):
    plt.rcParams.update({'font.family':'serif', 'font.size':11,
                         'mathtext.fontset':'stix', 'pdf.fonttype':42,
                         'axes.spines.top':False, 'axes.spines.right':False})
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True)
    for dimension, axis in zip((1, 2), axes):
        axis.set_title(TITLES[dimension])
        axis.set_xlabel(r'Mean sensor-budget ratio, $N/N_{\mathrm{LB}}^{\mathrm{joint}}$')
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel('Mean PGM error probability')
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.75, wspace=0.08)
    figure.suptitle(title, y=0.99)
    return figure, axes


def _curve(axis, rows, style):
    label, color, marker, line = style
    budgets = sorted({r['N'] for r in rows})
    means = [np.mean([r['pgm_error'] for r in rows if r['N']==n]) for n in budgets]
    ratios = [np.mean([r['budget_ratio'] for r in rows if r['N']==n]) for n in budgets]
    axis.plot(ratios, means, label=label, color=color, marker=marker, linestyle=line)


def _save(figure, axes, output, filename, legend_columns):
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center',
                  ncol=legend_columns, frameon=False, bbox_to_anchor=(0.5, 0.92))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output/f'{filename}.png', dpi=200, bbox_inches='tight')
    figure.savefig(output/f'{filename}.pdf', bbox_inches='tight')


def _matched(rows, key, values):
    """Require every plotted series on each case/budget to avoid unequal samples."""
    groups = defaultdict(dict)
    for row in rows:
        if row.get('algorithm_policy') != ALGORITHM_POLICY:
            raise ValueError('Old algorithm results; rerun analysis for incremental Greedy and independent VND.')
        if row['probe'] in ('local_dicke', 'global_dicke') and row.get('dicke_policy') != DICKE_POLICY:
            raise ValueError('Old Dicke results detected; rerun analysis with the equal-sector superposition.')
        if row[key] in values:
            groups[row['dimension'], row['case'], row['N']][row[key]] = row
    return [group[value] for group in groups.values() if all(v in group for v in values)
            for value in values]


def plot_by_algorithm(matches, output, *, methods):
    """One image per algorithm: five state curves, with 1D and 2D panels."""
    figures = []
    for method in methods:
        rows = _matched([r for r in matches if r['method']==method], 'probe', FAMILIES)
        figure, axes = _figure(f'{METHOD_STYLES[method][0]}: comparison of input states')
        for dimension, axis in zip((1, 2), axes):
            selected = [r for r in rows if r['dimension']==dimension]
            for probe in FAMILIES:
                _curve(axis, [r for r in selected if r['probe']==probe], STATE_STYLES[probe])
            if not selected:
                axis.text(0.5,0.5,'No groups completed for all five states',ha='center',transform=axis.transAxes)
        _save(figure, axes, output, f'states_by_algorithm_{method}', 5)
        figures.append(figure)
    return figures


def plot_by_state(matches, output, *, methods):
    """Five images, each comparing study algorithms for one input state."""
    figures = []
    for probe in FAMILIES:
        rows = _matched([r for r in matches if r['probe']==probe], 'method', methods)
        figure, axes = _figure(f'{STATE_STYLES[probe][0]}: comparison of algorithms')
        for dimension, axis in zip((1, 2), axes):
            selected = [r for r in rows if r['dimension']==dimension]
            for method in methods:
                _curve(axis, [r for r in selected if r['method']==method], METHOD_STYLES[method])
            if not selected:
                axis.text(0.5,0.5,'No groups completed by all methods',ha='center',transform=axis.transAxes)
        _save(figure, axes, output, f'algorithms_by_state_{probe}', len(methods))
        figures.append(figure)
    return figures
