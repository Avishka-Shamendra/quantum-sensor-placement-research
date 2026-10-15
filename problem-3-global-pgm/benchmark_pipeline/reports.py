"""Benchmark error and runtime reports for versioned manifests."""
from collections import defaultdict, Counter
from pathlib import Path
import csv,json
import numpy as np
from src.analysis import summarize, METHODS
from src.probes import FAMILIES
from src.figures import METHOD_STYLES, STATE_STYLES


def save(path,rows):
    if rows:
        with Path(path).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def build_report(records,study,output):
    """Summarize the benchmark with matched algorithm comparisons."""
    folder=Path(output)/'benchmark'
    folder.mkdir(parents=True,exist_ok=True)
    selected=records
    matched,summary,status=summarize(selected,folder,methods=METHODS[study])
    lookup={(r['dimension'],r['case'],r['N'],r['probe'],r['method']):r for r in selected}
    for r in matched:
        original=lookup[r['dimension'],r['case'],r['N'],r['probe'],r['method']]
        r.update({k:original[k] for k in ['N_LB_joint','budget_ratio','target_fractions']})
    save(folder/'normalized_paired_results.csv',matched)
    # Pair Greedy and VND directly even if compact Exhaustive timed out.
    pairs=defaultdict(dict)
    for r in selected:
        if r['status'] in ('optimal','greedy_complete','local_optimum', 'maximum_iterations'):
            pairs[r['dimension'],r['case'],r['N'],r['probe']][r['method']]=r
    comparisons=[]
    for key,pair in pairs.items():
        if not all(m in pair for m in ['greedy','vnd']):continue
        g,v=pair['greedy'],pair['vnd'];difference=g['pgm_error']-v['pgm_error']
        comparisons.append(dict(dimension=g['dimension'],case=g['case'],N=g['N'],K=g['K'],M=g['M'],probe=g['probe'],
            field_model=g['field_model'],budget_ratio=g['budget_ratio'],
            target_fractions=g['target_fractions'],greedy_minus_vnd=difference,
            greedy_exhaustive_gap=g['pgm_error']-pair['exhaustive']['pgm_error'] if 'exhaustive' in pair else None,
            vnd_beats=int(difference>1e-7),vnd_ties=int(abs(difference)<=1e-7),vnd_loses=int(difference < -1e-7)))
    save(folder/'algorithm_differences.csv',comparisons)
    grouped=defaultdict(list)
    for r in comparisons:
        for fraction in r['target_fractions']:
            grouped[r['dimension'],r['probe'],r['K'],r['M'],r['field_model'],fraction].append(r)
    frequencies=[]
    for key,values in sorted(grouped.items()):
        frequencies.append(dict(zip(['dimension','probe','K','M','field_family','target_fraction'],key),
            count=len(values),**{metric+'_percent':100*float(np.mean([r[metric] for r in values])) for metric in ['vnd_beats','vnd_ties','vnd_loses']}))
    save(folder/'vnd_frequencies.csv',frequencies)
    # Budgets counted once per selected field instance, not once per method/probe.
    unique={(r['dimension'],r['case'],r['N']):r for r in selected}
    counts=Counter((r['dimension'],r['N']) for r in unique.values())
    save(folder/'actual_budgets.csv',[dict(dimension=d,N=n,count=c) for (d,n),c in sorted(counts.items())])
    return dict(rows=matched,summary=summary,status=status,comparisons=comparisons,output=folder)


def points(rows,value):
    groups=defaultdict(list)
    for r in rows:
        if r.get(value) is not None:
            for fraction in r['target_fractions']:groups[fraction].append(r)
    return [(np.mean([r['budget_ratio'] for r in groups[f]]),np.mean([r[value] for r in groups[f]])) for f in sorted(groups)]


def figures(report,study):
    import matplotlib.pyplot as plt
    output=report['output'];result=[]
    def panel(title,ylabel):
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True,layout='constrained')
        for dim,ax in zip([1,2],axes):
            ax.set_title(f'{dim}D');ax.set_xlabel(r'Mean actual budget ratio, $N/N_{\mathrm{LB}}^{\mathrm{joint}}$');ax.grid(alpha=.2)
        axes[0].set_ylabel(ylabel);fig.suptitle(title)
        return fig,axes
    def finish(fig,axes,name):
        h,l=axes[0].get_legend_handles_labels()
        if not h:h,l=axes[1].get_legend_handles_labels()
        fig.legend(h,l,loc='outside lower center',ncol=5)
        fig.savefig(output/f'{name}.png',dpi=160);fig.savefig(output/f'{name}.pdf');result.append(fig)
    # Preserve both kinds of mean-error comparison, with fractions as the paired groups.
    for method in METHODS[study]:
        fig,axes=panel(f'{METHOD_STYLES[method][0]}: input states','Mean PGM error')
        for dim,ax in zip([1,2],axes):
            for probe in FAMILIES:
                chosen=[r for r in report['rows'] if r['dimension']==dim and r['method']==method and r['probe']==probe]
                xy=points(chosen,'pgm_error');label,color,marker,line=STATE_STYLES[probe]
                if xy:ax.plot(*zip(*xy),label=label,color=color,marker=marker,linestyle=line)
            ax.set_ylim(0,1)
        finish(fig,axes,f'states_by_algorithm_{method}')
    metrics=[('greedy_minus_vnd','Greedy − VND PGM error')]
    for metric,label in metrics:
        fig,axes=panel(label,label)
        for dim,ax in zip([1,2],axes):
            for probe in FAMILIES:
                xy=points([r for r in report['comparisons'] if r['dimension']==dim and r['probe']==probe],metric)
                title,color,marker,line=STATE_STYLES[probe]
                if xy:
                    x,y=zip(*xy);y=np.array(y)*(100 if metric.startswith('vnd_') else 1)
                    ax.plot(x,y,label=title,color=color,marker=marker,linestyle=line)
            if metric.startswith('vnd_'):ax.set_ylim(0,100)
            else:ax.axhline(0,color='black',lw=.5)
        finish(fig,axes,metric)
    return result


def field_family_errors(report,study):
    """One row per dimension/family/state, with matched method means."""
    groups=defaultdict(list)
    for row in report['rows']:
        groups[row['dimension'],row['field_model'],row['probe']].append(row)
    table=[]
    for (dimension,family,probe),rows in sorted(groups.items()):
        entry=dict(Dimension=f'{dimension}D',Field=family,State=STATE_STYLES[probe][0],
                   Case_budgets=len({(r['case'],r['N']) for r in rows}))
        for method in METHODS[study]:
            values=[r['pgm_error'] for r in rows if r['method']==method]
            entry[METHOD_STYLES[method][0]]=float(np.mean(values)) if values else None
        table.append(entry)
    save(report['output']/'field_family_errors.csv',table)
    return table
