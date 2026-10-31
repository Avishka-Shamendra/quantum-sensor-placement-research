"""Publication figures and paired summaries for the fixed benchmark."""
from collections import defaultdict, Counter
from pathlib import Path
import csv
import json
import numpy as np
from src.probes import FAMILIES
from src.search import ALGORITHM_POLICY
from src.figures import STATE_STYLES

METHODS = ('greedy', 'vnd')
STYLES = {'greedy': ('Greedy', '#d62728', 's', '--'),
          'vnd': ('VND (State Independent)', '#2ca02c', '^', '-.'),
          'near_uniform': ('Near-Uniform', '#9467bd', 'D', ':'),
          'random': ('Random', '#7f7f7f', 'x', ':')}
LABELS = {**{k:v[0] for k,v in STYLES.items()}, 'exhaustive':'Exhaustive'}
SUCCESS = {'optimal', 'greedy_complete', 'local_optimum', 'maximum_iterations', 'baseline_complete'}
TOLERANCE = 1e-7


def save_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def estimate(rows,value,seed=20261014,resamples=2000):
    """Equal-instance mean; stratified field-instance bootstrap percentile CI."""
    clusters=defaultdict(list);strata={}
    for r in rows:
        key=r['dimension'],r['case']
        clusters[key].append(r[value])
        strata[key]=(r['K'],r['M'],r['field_model'])
    if not clusters:return dict(mean=np.nan,low=np.nan,high=np.nan,instances=0)
    groups=defaultdict(list)
    for key,values in clusters.items():groups[strata[key]].append(float(np.mean(values)))
    rng=np.random.default_rng(seed);samples=np.zeros(resamples)
    for _,values in sorted(groups.items()):
        a=np.array(values)
        samples+=a[rng.integers(len(a),size=(resamples,len(a)))].sum(axis=1)
    samples/=len(clusters)
    low,high=np.percentile(samples,[2.5,97.5])
    return dict(mean=float(np.mean([np.mean(v) for v in clusters.values()])),low=float(low),high=float(high),instances=len(clusters))


def complete_states(rows,methods):
    groups=defaultdict(dict)
    for r in rows:
        if r['method'] in methods and r['status'] in SUCCESS:
            groups[r['dimension'],r['case'],r['N']][r['probe'],r['method']]=r
    return [group[p,m] for group in groups.values() if all((p,m) in group for p in FAMILIES for m in methods)
            for p in FAMILIES for m in methods]


def prepare(records,study,output,seed=20261014,resamples=2000):
    if any(r.get("algorithm_policy") != ALGORITHM_POLICY for r in records):
        raise ValueError("Saved results do not match the allocation policy; rerun the analysis before reporting.")
    methods=METHODS+tuple(m for m in ('near_uniform','random') if any(r['method']==m for r in records))
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    state_rows=complete_states(records,('vnd',))
    gap_rows=complete_states(records,(*methods,'exhaustive') if study=='compact' else methods)
    pairs=defaultdict(dict)
    for r in gap_rows:pairs[r['dimension'],r['case'],r['N'],r['probe']][r['method']]=r
    gaps=[]
    for group in pairs.values():
        reference=group['exhaustive']['pgm_error'] if study=='compact' else group['greedy']['pgm_error']
        for method in methods:
            r=group[method];gap=r['pgm_error']-reference
            if study=='compact' and gap < -TOLERANCE:raise AssertionError('A compared method beat the exhaustive reference beyond tolerance')
            gaps.append(dict(r,gap=gap,reference_error=reference))
    # Summaries aggregate all instances, with optional field-family grouping.
    quality=[];runtime=[];outcomes=[]
    for dim in [1,2]:
        for family in [None]+sorted({r['field_model'] for r in records}):
            chosen=[r for r in gaps if r['dimension']==dim and (family is None or r['field_model']==family)]
            for probe in FAMILIES:
                for method in methods:
                    group=[r for r in chosen if r['probe']==probe and r['method']==method]
                    if not group:continue
                    stats=estimate(group,'gap',seed,resamples)
                    error=estimate(group,'pgm_error',seed,resamples)['mean']
                    match=estimate([dict(r,match=float(abs(r['gap'])<=TOLERANCE)) for r in group],'match',seed,resamples)['mean']
                    quality.append(dict(dimension=f'{dim}D',family=family or 'Overall',state=STATE_STYLES[probe][0],method=LABELS[method],
                        instances=stats['instances'],mean_error=error,mean_gap=stats['mean'],ci_low=stats['low'],ci_high=stats['high'],
                        **({'optimality_match_percent':100*match} if study=='compact' else {})))
            for method in (('exhaustive',)+methods if study=='compact' else methods):
                attempted=[r for r in records if r['dimension']==dim and (family is None or r['field_model']==family) and r['method']==method]
                completed=[r for r in attempted if r['status'] in SUCCESS]
                if not attempted:continue
                times=[r['runtime_seconds'] for r in completed]
                allocation_runs={(r['case'],r['N'],r['probe'] if method=='greedy' else None):r['allocation_runtime_seconds'] for r in completed
                                 if r.get('allocation_runtime_seconds') is not None}
                allocation_times=list(allocation_runs.values())
                pgm_times=[r['pgm_evaluation_seconds'] for r in completed if r.get('pgm_evaluation_seconds') is not None]
                search_times=[r['search_runtime_seconds'] for r in completed if r.get('search_runtime_seconds') is not None]
                runtime.append(dict(dimension=f'{dim}D',family=family or 'Overall',method=LABELS[method],completed=len(completed),attempted=len(attempted),
                    allocation_runs=len(allocation_times),
                    mean_allocation_seconds=float(np.mean(allocation_times)) if allocation_times else None,
                    mean_final_pgm_seconds=float(np.mean(pgm_times)) if pgm_times else None,
                    mean_exhaustive_search_seconds=float(np.mean(search_times)) if search_times else None,
                    mean_seconds=float(np.mean(times)) if times else None,median_seconds=float(np.median(times)) if times else None,
                    total_observed_seconds=sum(r['runtime_seconds'] for r in attempted)))
            winpairs=defaultdict(dict)
            for r in records:
                if r['dimension']==dim and (family is None or r['field_model']==family) and r['method'] in ('greedy','vnd') and r['status'] in SUCCESS:
                    winpairs[r['case'],r['N'],r['probe']][r['method']]=r
            by_probe=defaultdict(list)
            for pair in winpairs.values():
                if len(pair)!=2:continue
                g,v=pair['greedy'],pair['vnd'];d=g['pgm_error']-v['pgm_error']
                by_probe[g['probe']].append(dict(g,win=float(d>TOLERANCE),tie=float(abs(d)<=TOLERANCE),loss=float(d < -TOLERANCE)))
            for probe,group in by_probe.items():
                outcomes.append(dict(dimension=f'{dim}D',family=family or 'Overall',state=STATE_STYLES[probe][0],case_budgets=len(group),
                    **{k+'_percent':100*estimate(group,k,seed,resamples)['mean'] for k in ['win','tie','loss']}))
    coverage=[]
    for dim in [1,2]:
        requested={(r['case'],r['N']) for r in records if r['dimension']==dim}
        coverage.append(dict(dimension=f'{dim}D',requested_case_budgets=len(requested),
            state_figure_case_budgets=len({(r['case'],r['N']) for r in state_rows if r['dimension']==dim}),
            gap_figure_case_budgets=len({(r['case'],r['N']) for r in gaps if r['dimension']==dim})))
    status=Counter((r['dimension'],r['method'],r['status']) for r in records)
    for name,rows in [('quality_summary',quality),('runtime_summary',runtime),('vnd_outcomes',outcomes),('coverage',coverage),
                      ('status',[dict(dimension=d,method=m,status=s,count=n) for (d,m,s),n in sorted(status.items())])]:save_csv(output/f'{name}.csv',rows)
    (output/'report_settings.json').write_text(json.dumps(dict(seed=seed,resamples=resamples,confidence=.95,
        bootstrap='field-instance clusters stratified by K, M, and field family',
        weighting='equal field-instance weight; average unique budgets within instance for bars',
        input_state_allocation='VND',
        reference='Exhaustive' if study=='compact' else 'Greedy per instance/budget/state; negative gaps indicate lower error than Greedy'),indent=2))
    return dict(methods=methods,state_rows=state_rows,gaps=gaps,quality=quality,runtime=runtime,outcomes=outcomes,coverage=coverage,
                output=output,seed=seed,resamples=resamples)


def curve(rows,value,report):
    result=[]
    for fraction in [.25,.5,.75,1.]:
        selected=[r for r in rows if fraction in r['target_fractions']]
        if selected:
            stats=estimate(selected,value,report['seed'],report['resamples'])
            result.append(dict(target_fraction=fraction,x=estimate(selected,'budget_ratio',report['seed'],report['resamples'])['mean'],**stats))
    return result


def figures(report,study,zoom=True):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    methods=report['methods']
    output=report['output'];result=[];data=[]
    style={'font.family':'Times New Roman','font.size':12,'axes.titlesize':12,'axes.labelsize':12,'legend.fontsize':12,
           'xtick.labelsize':12,'ytick.labelsize':12,'mathtext.fontset':'custom',
           'mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold',
           'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}
    def panel(ylabel,curves=True):
        fig,axes=plt.subplots(1,2,figsize=(10.8,4.6),sharey=True)
        fig.subplots_adjust(left=.10,right=.99,bottom=.27,top=.90,wspace=.12)
        for dim,ax in zip([1,2],axes):
            ax.set_title(f'({"a" if dim==1 else "b"}) {dim}D');ax.grid(axis='y',alpha=.2,zorder=0)
            if curves:ax.set_xlabel(r'Mean normalized budget, $N/N_{\mathrm{LB}}^{\mathrm{joint}}$')
        axes[0].set_ylabel(ylabel)
        return fig,axes
    def finish(fig,handles,labels,name):
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=len(labels),frameon=False)
        fig.savefig(output/f'{name}.pdf')
        fig.savefig(output/f'{name}.png',dpi=600);result.append(fig)
    def line(ax,points,style):
        if not points:return
        label,color,marker,ls=style;x=[p['x'] for p in points]
        ax.plot(x,[p['mean'] for p in points],color=color,marker=marker,linestyle=ls,label=label,markersize=3.5,lw=1.1)
        ax.fill_between(x,[p['low'] for p in points],[p['high'] for p in points],color=color,alpha=.13,lw=0)
    with plt.rc_context(style):
        fig,axes=panel('Mean PGM error')
        number=1 if study=='compact' else 3
        for dim,ax in zip([1,2],axes):
            for probe in FAMILIES:
                pts=curve([r for r in report['state_rows'] if r['dimension']==dim and r['probe']==probe],'pgm_error',report)
                line(ax,pts,STATE_STYLES[probe]);data.extend(dict(figure=number,dimension=dim,series=probe,**p) for p in pts)
            ax.set_ylim(0,1)
            if not any(r['dimension']==dim for r in report['state_rows']):ax.text(.5,.5,'No complete comparisons',ha='center',transform=ax.transAxes)
        handles=[Line2D([],[],color=s[1],marker=s[2],ls=s[3],label=s[0]) for s in STATE_STYLES.values()]
        finish(fig,handles,[s[0] for s in STATE_STYLES.values()],f'figure_{number}_input_states')
        bar_methods=tuple(m for m in methods if study=='compact' or m!='greedy')
        detail_methods=tuple(m for m in bar_methods if m in ('greedy','vnd'))
        number=2 if study=='compact' else 4
        reference='Exhaustive' if study=='compact' else 'Greedy'
        fig,axes=panel(f'Mean error difference from {reference}',False)
        for dim,ax in zip([1,2],axes):
            positions=np.arange(5);width=.8/len(bar_methods);bars={}
            for j,method in enumerate(bar_methods):
                estimates=[estimate([r for r in report['gaps'] if r['dimension']==dim and r['probe']==p and r['method']==method],'gap',report['seed'],report['resamples']) for p in FAMILIES]
                x=positions+(j-(len(bar_methods)-1)/2)*width;y=np.array([v['mean'] for v in estimates]);low=np.array([v['low'] for v in estimates]);high=np.array([v['high'] for v in estimates])
                ax.bar(x,y,width,color=STYLES[method][1],label=LABELS[method],alpha=.85,zorder=2)
                ax.vlines(x,low,high,color='black',lw=.65,zorder=3);ax.hlines(low,x-.035,x+.035,color='black',lw=.65);ax.hlines(high,x-.035,x+.035,color='black',lw=.65)
                bars[method]=(x,y,low,high)
                data.extend(dict(figure=number,dimension=dim,series=method,state=p,**v) for p,v in zip(FAMILIES,estimates))
            ax.set_xticks(positions,['Product','Local\nGHZ','Global\nGHZ','Local\nDicke','Global\nDicke']);ax.axhline(0,color='.4',lw=.6)
            finite=[v for method in bar_methods for values in bars[method][2:] for v in values if np.isfinite(v)]
            heuristic=[v for method in detail_methods for values in bars[method][2:] for v in values if np.isfinite(v)]
            scale=max(map(abs,finite),default=0);detail=max(map(abs,heuristic),default=0)
            if zoom and 0<detail<.35*scale:
                ax.set_ylim(bottom=min(0,min(finite))-.05*scale,top=max(0,max(finite))+.8*scale)
                inset=ax.inset_axes([.13,.65,.82,.27]);inset.set_title('Greedy / VND detail' if study=='compact' else 'VND detail',fontsize=12)
                for method in detail_methods:
                    x,y,lo,hi=bars[method];inset.bar(x,y,width,color=STYLES[method][1]);inset.vlines(x,lo,hi,color='black',lw=.5)
                inset.set_ylim(min(0,min(heuristic))-.1*detail,max(0,max(heuristic))+.1*detail)
                inset.axhline(0,color='.4',lw=.5)
                inset.set_xticks(positions,['P','LG','GG','LD','GD']);inset.tick_params(labelsize=12);inset.grid(axis='y',alpha=.15)
        finish(fig,[plt.Rectangle((0,0),1,1,color=STYLES[m][1]) for m in bar_methods],[LABELS[m] for m in bar_methods],'figure_2_optimality_gaps' if study=='compact' else 'figure_4_performance_gaps')
    # Figure tables have different columns for curves and grouped bars.
    for number in ([1,2] if study=='compact' else [3,4]):save_csv(output/f'figure_{number}_data.csv',[r for r in data if r['figure']==number])
    return result
