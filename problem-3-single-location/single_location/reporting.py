"""Instance-level summaries and the two publication figures."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

from .analysis import FAMILIES

STATE_STYLES = {
    'optimized': ('Optimized Dicke sectors', '#1f77b4', 'o', '-'),
    'product': ('Product', '#2ca02c', '^', '-.'),
    'ghz': ('GHZ', '#9467bd', 'D', ':'),
    'equal': ('Equal Dicke sectors', '#8c564b', 'v', '--'),
}


def summarize(frame, keys, value):
    """Two-sided 95% Student-t intervals across independent instances."""
    s = frame.groupby(keys)[value].agg(['count', 'mean', 'std']).reset_index()
    s['ci95'] = t.ppf(.975, s['count'] - 1) * s['std'] / np.sqrt(s['count'])
    s['lower95'] = s['mean'] - s['ci95']
    s['upper95'] = s['mean'] + s['ci95']
    return s.drop(columns='std')


def strategy_rows(frame):
    rows = []
    for r in frame.to_dict('records'):
        rows.append(dict(family=r['family'], k=r['k'], n=r['n'], inst=r['inst'],
                         strategy=r['state'] + '_pgm', error=r['pgm_error']))
        if r['state'] == 'optimized':
            rows.append(dict(family=r['family'], k=r['k'], n=r['n'], inst=r['inst'],
                             strategy='optimized_povm', error=r['optimal_error']))
    return pd.DataFrame(rows)


def make_results(frame, root):
    out = root / 'results'
    out.mkdir(parents=True, exist_ok=True)
    errors = summarize(strategy_rows(frame), ['family', 'k', 'n', 'strategy'], 'error')
    paired = frame.assign(degradation=frame.pgm_error - frame.optimal_error)
    degradation = summarize(paired, ['family', 'k', 'n', 'state'], 'degradation')
    runtime = frame.groupby(['family', 'k', 'state']).agg(
        cases=('inst', 'size'), mean_measurement_seconds=('measurement_seconds', 'mean'),
        mean_optimization_seconds=('optimization_seconds', 'mean'))
    validation = frame.groupby(['family', 'k']).agg(
        evaluated_states=('state', 'size'), max_dual_residual=('dual_residual', 'max'),
        minimum_pgm_gap=('pgm_error', lambda x: float('nan')))
    validation['minimum_pgm_gap'] = paired.groupby(['family', 'k']).degradation.min()
    optimized = frame[frame.state == 'optimized'].copy()
    optimized['absolute_saved_error_difference'] = optimized.saved_error_difference.abs()
    audit = optimized.groupby(['family', 'k']).agg(
        cases=('inst', 'size'), converged=('optimizer_converged', 'sum'),
        maximum_saved_error_difference=('absolute_saved_error_difference', 'max'),
        maximum_reflection_asymmetry=('reflection_asymmetry', 'max'))
    for name, table in [('error_summary', errors), ('measurement_degradation', degradation),
                        ('runtime_summary', runtime), ('validation_summary', validation), ('optimization_audit', audit)]:
        table.to_csv(out / (name + '.csv'), index=name not in ('error_summary', 'measurement_degradation'))
    figures = make_figures(frame, root)
    return dict(errors=errors, degradation=degradation, runtime=runtime, validation=validation,
                optimization_audit=audit, figures=figures)


def make_figures(frame, root):
    """Export the two K=4 figures from saved per-state measurement errors."""
    out = root / 'results'
    out.mkdir(parents=True, exist_ok=True)
    selected = frame[frame.k == 4]
    expected = pd.MultiIndex.from_product(
        [FAMILIES, [2, 4, 6, 8, 10], STATE_STYLES, range(30)],
        names=['family', 'n', 'state', 'inst'])
    actual = pd.MultiIndex.from_frame(selected[['family', 'n', 'state', 'inst']])
    if actual.has_duplicates or len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError('Figures require 30 saved instances per K=4 family, budget and state')
    if not np.isfinite(selected[['optimal_error', 'pgm_error']].to_numpy()).all():
        raise ValueError('Saved measurement errors must be finite')
    optimal = summarize(selected, ['family', 'k', 'n', 'state'], 'optimal_error')
    paired = selected.assign(degradation=selected.pgm_error - selected.optimal_error)
    degradation = summarize(paired, ['family', 'k', 'n', 'state'], 'degradation')
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                         'font.size': 12, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    figures = []
    for number in (1, 2):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True)
        for ax, family in zip(axes, FAMILIES):
            if number == 1:
                data = optimal[(optimal.family == family) & (optimal.k == 4)]
            else:
                data = degradation[(degradation.family == family) & (degradation.k == 4)]

            for key, (label, color, marker, line) in STATE_STYLES.items():
                g = data[data.state == key].sort_values('n')
                ax.plot(g.n, g['mean'], label=label, color=color, marker=marker, linestyle=line, ms=4, lw=1.4)
            ax.set_title(family.capitalize())
            ax.set_xlabel(r'Sensor count, $N$')
            ax.set_xticks([2, 4, 6, 8, 10])
            ax.grid(alpha=.2)
            if number == 2:
                ax.axhline(0, color='black', lw=.6)
        axes[0].set_ylabel('Mean discrimination error' if number == 1 else 'Mean excess error from PGM')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .87))
        for extension in ('pdf', 'png'):
            fig.savefig(out / f'figure_{number}.{extension}', dpi=600, bbox_inches='tight')
        figures.append(fig)
    return figures
