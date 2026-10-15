"""Paired target-error summaries and publication exports for the dual study."""
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def interval(rows, seed, resamples):
    """Field-family-stratified percentile bootstrap of whole field instances."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r['field_model'], r['case'])):
        groups[row['field_model']].append(row['N'])
    draws = np.zeros(resamples)
    for family in sorted(groups):
        values = np.asarray(groups[family], dtype=float)
        draws += values[rng.integers(len(values), size=(resamples, len(values)))].sum(axis=1)
    draws /= len(rows)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def report(rows, failures, output, targets, *, resamples=2000, seed=20261015):
    """Require completed, verified solutions at every target for paired curves."""
    import matplotlib.pyplot as plt
    if type(resamples) is not int or resamples < 2:
        raise ValueError('At least two bootstrap resamples are required')
    targets = tuple(sorted(targets))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = [dict(r, method=r.get('method','vnd_beam')) for r in rows]
    methods = [m for m in ('vnd_beam','greedy') if any(r['method']==m for r in rows)]
    groups = defaultdict(dict)
    for row in rows:
        if not row['verified_feasible'] or row['actual_error'] > row['epsilon']:
            raise ValueError('Unverified or infeasible solution in reported results')
        key = row['dimension'], row['case']
        if (row['epsilon'],row['method']) in groups[key]:
            raise ValueError('Duplicate instance/target result')
        groups[key][row['epsilon'],row['method']] = row
    eligible = {key for key, group in groups.items()
                if all((e,m) in group and group[e,m]['status'] == 'completed' for e in targets for m in methods)}
    paired = [r for r in rows if (r['dimension'],r['case']) in eligible and r['epsilon'] in targets]
    summary = []; coverage = []
    for dim in (1,2):
        attempted = {key for key in groups if key[0] == dim}
        attempted.update((f['dimension'],f['case']) for f in failures if f['dimension'] == dim)
        coverage.append(dict(dimension=f'{dim}D',selected_instances=len(attempted),
            complete_paired_instances=sum(key[0] == dim for key in eligible),
            failed_instances=len({f['case'] for f in failures if f['dimension'] == dim}),
            interrupted_targets=sum(r['dimension'] == dim and r['status'] != 'completed' for r in rows)))
        for method in methods:
            for epsilon in targets:
                selected = [r for r in paired if r['dimension'] == dim and r['epsilon'] == epsilon and r['method']==method]
                if not selected: continue
                low, high = interval(selected,seed,resamples)
                q25, median, q75 = np.percentile([r['N'] for r in selected], [25,50,75])
                summary.append(dict(dimension=f'{dim}D',method=method,epsilon=epsilon,instances=len(selected),
                    median_required_sensors=float(median),q25_required_sensors=float(q25),q75_required_sensors=float(q75),
                    mean_required_sensors=float(np.mean([r['N'] for r in selected])),ci_low=low,ci_high=high,
                    mean_runtime_seconds=float(np.mean([r['runtime_seconds'] for r in selected])),
                    mean_perfect_setup_seconds=float(np.mean([r['perfect_setup_seconds'] for r in selected])),
                    mean_budget_search_seconds=float(np.mean([r['search_seconds'] for r in selected])),
                    mean_achieved_error=float(np.mean([r['actual_error'] for r in selected])),
                    max_achieved_error=max(r['actual_error'] for r in selected),all_feasible=True))
    write_csv(output/'summary.csv',summary); write_csv(output/'coverage.csv',coverage)
    write_csv(output/'verified_solutions.csv', [{k:r[k] for k in
        ('dimension','case','method','field_model','K','M','epsilon','N_min_seq','N','actual_error','runtime_seconds',
         'perfect_setup_seconds','search_seconds','budget_evaluations','status','verified_feasible')} for r in rows])
    style = {'font.family':'Times New Roman','font.size':12,'axes.titlesize':12,
             'axes.labelsize':12,'legend.fontsize':12,'xtick.labelsize':12,'ytick.labelsize':12,
             'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1,2,figsize=(10.8,4.6),sharey=False)
        fig.subplots_adjust(left=.1,right=.98,bottom=.2,top=.88,wspace=.25)
        for dim, ax in zip((1,2),axes):
            selected = [r for r in summary if r['dimension'] == f'{dim}D']
            ax.set_title(f'({"a" if dim==1 else "b"}) {dim}D')
            ax.set_xlabel(r'Target error, $\epsilon$'); ax.set_xticks(targets)
            ax.grid(axis='y',alpha=.2)
            if selected:
                styles = {'vnd_beam':('VND + Beam Search','#2ca02c','o','-'),
                          'greedy':('Greedy','#d62728','s','--')}
                for method in methods:
                    series = [r for r in selected if r['method']==method]
                    if not series: continue
                    label,color,marker,linestyle=styles[method]
                    x=[r['epsilon'] for r in series];y=[r['median_required_sensors'] for r in series]
                    ax.plot(x,y,color=color,marker=marker,linestyle=linestyle,label=label)
                    ax.fill_between(x,[r['q25_required_sensors'] for r in series],[r['q75_required_sensors'] for r in series],color=color,alpha=.15)
                ax.legend(frameon=False,title='Median; band: 25th–75th percentile',title_fontsize=10)
            else:
                ax.text(.5,.5,'No complete verified comparisons',ha='center',transform=ax.transAxes)
        for ax in axes:
            ax.set_ylim(bottom=0)
        axes[0].set_ylabel('Median verified feasible sensor count')
        fig.savefig(output/'required_sensors_vs_target.pdf')
        fig.savefig(output/'required_sensors_vs_target.png',dpi=600)
    return fig, summary, coverage
