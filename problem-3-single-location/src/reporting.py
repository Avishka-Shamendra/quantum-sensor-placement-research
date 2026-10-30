"""Saved-result summaries and publication figures."""
import json
from hashlib import sha256

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

from .states import validate_weights

FAMILIES = ('gaussian', 'sinusoidal', 'quadratic')
KS = (3, 4, 5)
NS = (2, 4, 6, 8, 10)
STATE_STYLES = {
    'optimized': ('Optimized Dicke sectors', '#1f77b4', 'o', '-'),
    'product': ('Product', '#2ca02c', '^', '-.'),
    'ghz': ('GHZ', '#9467bd', 'D', ':'),
    'equal': ('Equal Dicke sectors', '#8c564b', 'v', '--'),
}
BUDGET_STYLES = dict(zip(NS, [('#0072B2', 'o'), ('#D55E00', 's'),
                             ('#009E73', '^'), ('#7B2CBF', 'D'), ('#C58A00', 'v')]))
SYMMETRY_TOLERANCE = 1e-5


def summarize(frame, keys, value):
    """Two-sided 95% Student-t intervals across independent instances."""
    s = frame.groupby(keys)[value].agg(['count', 'mean', 'std']).reset_index()
    s['ci95'] = t.ppf(.975, s['count'] - 1) * s['std'] / np.sqrt(s['count'])
    s['lower95'] = s['mean'] - s['ci95']
    s['upper95'] = s['mean'] + s['ci95']
    return s.drop(columns='std')


def validate_measurement_table(frame):
    expected = pd.MultiIndex.from_product(
        [range(1, 3), FAMILIES, KS, NS, STATE_STYLES, range(50)],
        names=['dimension', 'family', 'k', 'n', 'state', 'inst'])
    actual = pd.MultiIndex.from_frame(frame[list(expected.names)])
    if actual.has_duplicates or len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError('Expected 50 instances per dimension, family, K, budget and state')
    if not np.isfinite(frame[['optimal_error', 'pgm_error']].to_numpy()).all():
        raise ValueError('Saved measurement errors must be finite')


def load_saved_weights(frame, root):
    """Reuse consolidated weights only when they match the measurement records."""
    optimized = frame[frame.state == 'optimized']
    columns = ['dimension', 'family', 'k', 'n', 'inst', 'checkpoint',
               'optimal_error', 'pgm_error', 'reflection_asymmetry']
    signature = sha256(optimized[columns].sort_values(columns[:5]).to_csv(index=False).encode()).hexdigest()
    out = root / 'results/exact_sdp'
    csv_path, metadata_path = out / 'saved_weights.csv', out / 'saved_weights_meta.json'
    if csv_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            if (metadata['measurements_sha256'] == signature and
                    metadata['weights_sha256'] == sha256(csv_path.read_bytes()).hexdigest()):
                return pd.read_csv(csv_path)
        except (ValueError, KeyError):
            pass
    print(f"Loading {len(optimized)} saved weight checkpoints; subsequent reports use the consolidated CSV.", flush=True)
    rows = []
    checkpoint_root = (root / 'results/exact_sdp/checkpoints').resolve()
    for r in optimized.itertuples():
        path = (root / r.checkpoint).resolve()
        if not path.is_relative_to(checkpoint_root):
            raise ValueError('Invalid checkpoint path')
        item = json.loads(path.read_text())
        if (item['source'] != 'joint_sdp' or
                abs(item['optimal_error']-r.optimal_error) > 1e-12 or
                abs(item['pgm_error']-r.pgm_error) > 1e-12):
            raise ValueError('Saved weights disagree with the measurement table')
        p = validate_weights(item['p'], r.n)
        if abs(float(np.max(np.abs(p-p[::-1])))-r.reflection_asymmetry) > 1e-10:
            raise ValueError('Saved weights disagree with the symmetry statistic')
        for w, value in enumerate(p):
            rows.append(dict(dimension=r.dimension, family=r.family, k=r.k, m=1, n=r.n,
                             inst=r.inst, w=w, p=value, reflected_p=p[r.n-w]))
    weights = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix('.tmp')
    weights.to_csv(temporary, index=False)
    temporary.replace(csv_path)
    metadata = dict(measurements_sha256=signature, weights_sha256=sha256(csv_path.read_bytes()).hexdigest())
    temporary = metadata_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(metadata) + '\n')
    temporary.replace(metadata_path)
    return weights


def make_results(frame, root):
    validate_measurement_table(frame)
    frame = frame.assign(m=1)
    out = root / 'results/exact_sdp'
    out.mkdir(parents=True, exist_ok=True)
    keys = ['dimension', 'family', 'k', 'm', 'n', 'state']
    errors = summarize(frame, keys, 'optimal_error')
    paired = frame.assign(degradation=frame.pgm_error - frame.optimal_error)
    degradation = summarize(paired, keys, 'degradation')
    runtime = frame.groupby(keys).agg(
        cases=('inst', 'size'), mean_measurement_seconds=('measurement_seconds', 'mean'),
        mean_optimization_seconds=('optimization_seconds', 'mean')).reset_index()
    validation = paired.groupby(keys).agg(
        cases=('inst', 'size'), max_dual_residual=('dual_residual', 'max'),
        minimum_pgm_gap=('degradation', 'min')).reset_index()
    optimized = frame[frame.state == 'optimized'].copy()
    audit = optimized.groupby(['dimension', 'family', 'k', 'm', 'n']).agg(
        cases=('inst', 'size'), converged=('optimizer_converged', 'sum'),
        maximum_optimality_gap=('optimality_gap', 'max'),
        maximum_primal_residual=('primal_residual', 'max')).reset_index()
    weights = load_saved_weights(frame, root)
    weights['absolute_pair_difference'] = abs(weights.p - weights.reflected_p)
    cases = weights.groupby(['dimension', 'family', 'k', 'm', 'n', 'inst']).absolute_pair_difference.max().rename('asymmetry').reset_index()
    group = ['dimension', 'family', 'k', 'm', 'n']
    symmetry = summarize(cases, group, 'asymmetry')
    details = cases.assign(within_tolerance=cases.asymmetry <= SYMMETRY_TOLERANCE).groupby(group).agg(
        maximum_asymmetry=('asymmetry', 'max'), fraction_within_tolerance=('within_tolerance', 'mean')).reset_index()
    symmetry = symmetry.merge(details, on=group)
    symmetry['tolerance'] = SYMMETRY_TOLERANCE
    weight_summary = summarize(weights, group + ['w'], 'p')
    overall_errors = summarize_across_k(frame, 'optimal_error')
    overall_degradation = summarize_across_k(paired, 'degradation')
    tables = dict(overall_errors=overall_errors, overall_degradation=overall_degradation, errors=errors, degradation=degradation, symmetry=symmetry,
                  weight_summary=weight_summary, runtime=runtime, validation=validation,
                  optimization_audit=audit)
    filenames = dict(overall_errors='figure_1_summary', overall_degradation='figure_2_summary', errors='error_summary', degradation='measurement_degradation',
                     symmetry='symmetry_summary', weight_summary='weight_summary',
                     runtime='runtime_summary', validation='validation_summary', optimization_audit='optimization_audit')
    for name, table in tables.items():
        table.to_csv(out / (filenames[name] + '.csv'), index=False)
    cases.to_csv(out / 'symmetry_by_instance.csv', index=False)
    tables['figures'] = make_figures(frame, root, weights=weights)
    return tables


def summarize_across_k(frame, value):
    """Equal-weight dimension/K means with stratified Welch intervals."""
    keys = ['family', 'm', 'n', 'state']
    strata = frame.assign(m=1).groupby(keys + ['dimension', 'k'])[value].agg(['mean', 'var', 'count']).reset_index()
    rows = []
    for key, group in strata.groupby(keys):
        components = group['var'].to_numpy() / group['count'].to_numpy() / len(group)**2
        variance = components.sum()
        denominator = np.sum(components**2 / (group['count'].to_numpy() - 1))
        df = variance**2 / denominator if denominator > 0 else np.inf
        half = float(t.ppf(.975, df) * np.sqrt(variance))
        mean = float(group['mean'].mean())
        rows.append(dict(zip(keys, key), count=int(group['count'].sum()), mean=mean,
                         ci95=half, lower95=mean-half, upper95=mean+half))
    return pd.DataFrame(rows)


def configure_style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
                         'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 12,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
                         'figure.titlesize': 12, 'mathtext.fontset': 'stix',
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})


def export(fig, out, number):
    for extension in ('pdf', 'png'):
        fig.savefig(out / f'figure_{number}.{extension}', dpi=600, bbox_inches='tight')


def make_figures(frame, root, weights=None):
    """Plot all saved K and N values at M=1 without evaluating objectives."""
    validate_measurement_table(frame)
    if weights is None:
        weights = load_saved_weights(frame, root)
    out = root / 'results/exact_sdp'
    out.mkdir(parents=True, exist_ok=True)
    configure_style()
    optimal = summarize_across_k(frame, 'optimal_error')
    paired = frame.assign(degradation=frame.pgm_error - frame.optimal_error)
    degradation = summarize_across_k(paired, 'degradation')
    figures = []
    for number, summary in ((1, optimal), (2, degradation)):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True, sharey=True)
        for ax, family in zip(axes, FAMILIES):
            data = summary[summary.family == family]
            for index, (key, (label, color, marker, line)) in enumerate(STATE_STYLES.items()):
                g = data[data.state == key].sort_values('n')
                if number == 2:
                    width = 0.38
                    offset = (index - (len(STATE_STYLES)-1)/2) * width
                    ax.bar(g.n + offset, g['mean'], width=width, color=color, label=label, zorder=3)
                else:
                    ax.plot(g.n, g['mean'], label=label, color=color, marker=marker, linestyle=line, ms=4, lw=1.4)
            ax.set_title(family.capitalize())
            ax.set_xticks(NS)
            ax.set_xlabel(r'Sensor count, $N$')
            ax.grid(axis='y' if number == 2 else 'both', alpha=.2)
            if number == 2:
                ax.axhline(0, color='black', lw=.6)
        axes[0].set_ylabel('Mean discrimination error' if number == 1 else 'Mean excess error from PGM')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .90))
        export(fig, out, number)
        figures.append(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.3), sharex=True, sharey=True)
    limit = max(.5, float(weights.p.max()) * 1.06)
    for ax, family in zip(axes, FAMILIES):
        selected = weights[weights.family == family]
        for n, (color, marker) in BUDGET_STYLES.items():
            g = selected[(selected.n == n) & (2*selected.w < selected.n)]
            ax.scatter(g.p, g.reflected_p, label=f'$N={n}$', color=color,
                       marker=marker, s=26, alpha=.95, edgecolors='white', linewidths=.25,
                       zorder=3, rasterized=True)
        ax.plot([0, limit], [0, limit], color='black', linestyle='--', lw=.9, zorder=2, label=r'$p_w=p_{N-w}$')
        ax.set_title(family.capitalize())
        ax.set_xlim(-.015, limit)
        ax.set_ylim(-.015, limit)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.2)
        ax.set_xlabel(r'$p_w$')
    axes[0].set_ylabel(r'$p_{N-w}$')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False, markerscale=1.5)
    fig.tight_layout(rect=(0, 0, 1, .91))
    export(fig, out, 3)
    figures.append(fig)
    return figures
