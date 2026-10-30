"""Reproducible ordinary-random benchmark generation with physical validity checks."""
from pathlib import Path
import sys, csv, json, hashlib, argparse, tempfile, shutil
from functools import lru_cache
from random import Random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import ceil
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from generate_inputs import fields
from src.probes import DICKE_POLICY
from src.normalization import joint_sensor_lower_bound
FRACTIONS=(.25,.5,.75,1.)
MINIMUM_PHASE_GAP=fields.DEFAULT_MIN_BEST_PAIR_PHASE_GAP


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts,separators=(',',':')).encode()).digest()[:16],'big')


def budgets(bound,cap):
    maximum=min(bound,cap);entries={}
    for fraction in FRACTIONS:
        n=ceil(fraction*maximum)
        entries.setdefault(n,dict(N=n,target_fractions=[],actual_ratio=n/bound))['target_fractions'].append(fraction)
    return maximum,list(entries.values())


def draw(study,dimension,model,K,M,seed,attempt):
    generation_seed=stable_seed('problem3-ordinary-v5',seed,study,dimension,model,K,M,attempt)
    rng=Random(generation_seed)
    locations=fields.sensing_locations(dimension,M,rng)
    values,parameters=fields.FIELD_MODELS[model](K,locations,rng)
    rows=fields.phase_gap_rows(values,1.)
    gaps=np.array([r[3] for r in rows]).reshape(-1,M)
    return rows,dict(K=K,M=M,field_model=model,spatial_dimension=dimension,phase_scale_c=1.,
        generation_seed=generation_seed,generation_attempt=attempt,sensor_locations=locations,
        field_values=values,field_parameters=parameters,
        min_best_pair_phase_gap=float(np.min(np.max(abs(gaps),axis=1))))


def write_input(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['i','j','m','r_ijm'])
        writer.writerows((i,j,m,format(r,'.17g')) for i,j,m,r in rows)


def cell(task):
    study,dimension,model,K,M,seed,quota,limit,output,*source=task
    if study=='extended' and source:
        return copy_cell(task)
    directory=Path(output)/study/f'{dimension}d';directory.mkdir(parents=True,exist_ok=True)
    cellname=f'{model}_k{K}_m{M}'
    accepted=[];attempts=[];cap=16 if study=='compact' else 50
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'candidate.csv'
        for attempt in range(1,limit+1):
            if len(accepted)>=quota:break
            rows,meta=draw(study,dimension,model,K,M,seed,attempt)
            audit=dict(attempt=attempt,generation_seed=meta['generation_seed'])
            if meta['min_best_pair_phase_gap']<MINIMUM_PHASE_GAP:
                audit['reason']='pair_phase_gap_below_threshold';attempts.append(audit);continue
            write_input(path,rows)
            bound=joint_sensor_lower_bound(path)
            maximum,entries=budgets(bound,cap)
            replicate=len(accepted)+1;filename=f'{cellname}_{replicate:02d}.csv'
            write_input(directory/filename,rows)
            meta.update(file=filename,instance_id=f'{study}_{dimension}d_{cellname}_{replicate:02d}',
                dataset=study,replicate=replicate,N_LB_joint=bound,N_max=maximum,
                sensor_budgets=[e['N'] for e in entries],budget_schedule=entries)
            accepted.append(meta);audit.update(reason='accepted',file=filename)
            attempts.append(audit)
    result=dict(study=study,dimension=dimension,field_family=model,K=K,M=M,quota=quota,
        candidates_attempted=len(attempts),validity_rejections=sum(a['reason']!='accepted' for a in attempts),
        accepted=len(accepted),shortfall=quota-len(accepted),attempts=attempts,cases=accepted)
    return result


@lru_cache(maxsize=2)
def source_manifest(directory):
    return json.loads((Path(directory)/'manifest.json').read_text())


def source_values(case):
    rng=Random(case['generation_seed']+case.get('generation_attempt',1)-1)
    locations=fields.sensing_locations(case['spatial_dimension'],case['M'],rng)
    values,parameters=fields.FIELD_MODELS[case['field_model']](case['K'],locations,rng)
    expected=case['sensor_locations']
    if case['spatial_dimension']==1:expected=[(v,) for v in expected]
    np.testing.assert_array_equal(locations,expected)
    if parameters!=case['field_parameters']:raise ValueError('Source parameters do not match their recorded seed')
    return values


def copy_cell(task):
    study,dimension,model,K,M,seed,quota,limit,output,source_root=task
    source=Path(source_root)/f'benchmark_{dimension}d'
    directory=Path(output)/study/f'{dimension}d';directory.mkdir(parents=True,exist_ok=True)
    selected=[c for c in source_manifest(str(source))['cases'] if (c['field_model'],c['K'],c['M'])==(model,K,M)]
    selected=sorted(selected,key=lambda c:c['replicate'])[:quota]
    if len(selected)!=quota:raise ValueError('Problem 2 source has insufficient instances')
    accepted=[];attempts=[]
    for original in selected:
        source_path=source/original['file'];values=source_values(original)
        with source_path.open() as stream:
            actual=np.array([[float(r[k]) for k in ('i','j','m','r_ijm')] for r in csv.DictReader(stream)])
        np.testing.assert_array_equal(actual,np.asarray(fields.phase_gap_rows(values,original['phase_scale_c'])))
        minimum_gap=float(np.min(np.max(abs(actual[:,3].reshape(-1,M)),axis=1)))
        if not np.isfinite(minimum_gap) or minimum_gap<MINIMUM_PHASE_GAP:
            raise ValueError(f'{source_path}: every pair must have max_m |r_ijm| >= {MINIMUM_PHASE_GAP}; regenerate the Problem 2 extended datasets')
        bound=joint_sensor_lower_bound(source_path);maximum,entries=budgets(bound,50)
        shutil.copyfile(source_path,directory/original['file'])
        meta={k:v for k,v in original.items() if k not in ('minimum_sensors_for_zero_error','sensor_budgets','min_best_pair_phase_gap','maximum_best_pair_sensor_threshold')}
        meta.update(instance_id=f'extended_{dimension}d_{original["file"]}',dataset=study,field_values=values,
            min_best_pair_phase_gap=minimum_gap,
            source_dataset=source.name,source_file=original['file'],source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            N_LB_joint=bound,N_max=maximum,sensor_budgets=[e['N'] for e in entries],budget_schedule=entries)
        accepted.append(meta);attempts.append(dict(attempt=original['replicate'],generation_seed=original['generation_seed'],reason='accepted',file=original['file']))
    result=dict(study=study,dimension=dimension,field_family=model,K=K,M=M,quota=quota,candidates_attempted=len(attempts),
        validity_rejections=0,accepted=len(accepted),shortfall=0,attempts=attempts,cases=accepted)
    return result


def finalize(output,settings,results):
    report={}
    for study in settings.get('studies', ['compact','extended']):
        allcases=[];audits=[]
        for dim in [1,2]:
            directory=output/study/f'{dim}d';directory.mkdir(parents=True,exist_ok=True)
            cells=sorted((r for r in results if r['study']==study and r['dimension']==dim),
                         key=lambda r:(r['field_family'],r['K'],r['M']))
            cases=[c for a in cells for c in a['cases']]
            manifest=dict(schema_version=4,dataset_version=output.name,
                design=dict(study=study,dimension=dim,phase_scale_c=1.,test_replicates_per_cell=1,**settings),
                dicke_policy=DICKE_POLICY,sampling='Problem 2 identical CSV instances' if study=='extended' else 'independent random draws conditioned on max_m |r_ijm| >= 0.01 for every pair',
                models={m:dict(formula=fields.FIELD_FORMULAS[m][dim],parameter_ranges=fields.FIELD_PARAMETER_RANGES[m][dim]) for m in fields.FIELD_MODELS},
                input_format=dict(columns=['i','j','m','r_ijm'],indexing='zero-based',
                    heuristic_phase_input='absolute values of signed CSV r_ijm'),cases=cases)
            (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            allcases.extend(cases);audits.extend(cells)
        report[study]=dict(instances=len(allcases),cells_completed=len(audits),
            validity_rejections=sum(a['validity_rejections'] for a in audits),
            N_max_distribution=dict(sorted(Counter(c['N_max'] for c in allcases).items())),
            shortfalls=[{k:a[k] for k in ['dimension','field_family','K','M','shortfall']} for a in audits if a['shortfall']])
    (output/'acceptance_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261014);parser.add_argument('--max-attempts',type=int,default=1000)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--problem2-inputs',type=Path,default=ROOT.parent/'problem-2-min-error-seq/inputs')
    parser.add_argument('--studies',nargs='+',choices=['compact','extended'],default=['compact','extended'])
    parser.add_argument('--compact-replicates',type=int,default=50)
    parser.add_argument('--overwrite',action='store_true',help='Replace selected studies after successful generation')
    args=parser.parse_args();output=args.output.resolve()
    if args.compact_replicates < 1:parser.error('--compact-replicates must be positive')
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error('Output directory must be empty, or use --overwrite')
    output.parent.mkdir(parents=True,exist_ok=True)
    if args.overwrite:
        with tempfile.TemporaryDirectory(prefix='.problem3-inputs-',dir=output.parent) as temporary:
            staging=Path(temporary)
            generate_datasets(args,staging)
            install_datasets(staging,output,args.studies)
    else:
        generate_datasets(args,output)


def install_datasets(staging,output,studies):
    """Replace selected studies with completed datasets."""
    output.mkdir(parents=True,exist_ok=True)
    for study in dict.fromkeys(studies):
        target=output/study
        previous=staging/f'{study}-previous'
        if target.exists():target.rename(previous)
        try:
            (staging/study).rename(target)
        except OSError:
            if previous.exists():previous.rename(target)
            raise
    for name in ('generation_config.json','acceptance_report.json'):
        shutil.copyfile(staging/name,output/name)


def generate_datasets(args,output):
    output.mkdir(parents=True,exist_ok=True)
    settings=dict(studies=args.studies,compact_replicates=args.compact_replicates,seed=args.seed,max_attempts=args.max_attempts,dry_run=args.dry_run,problem2_inputs=str(args.problem2_inputs.resolve()),minimum_phase_gap=MINIMUM_PHASE_GAP)
    protected=[ROOT/'src'/name for name in ['probes.py','pgm.py']]+[ROOT.parent/'problem-1-zero-error/src/field_models.py',ROOT/'src/normalization.py',Path(__file__)]
    provenance=dict(settings=settings,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected})
    source_hashes={}
    for dim in ((1,2) if 'extended' in args.studies else ()):
        directory=args.problem2_inputs/f'benchmark_{dim}d'
        digest=hashlib.sha256((directory/'manifest.json').read_bytes())
        for c in source_manifest(str(directory))['cases']:
            digest.update(c['file'].encode());digest.update((directory/c['file']).read_bytes())
        source_hashes[directory.name]=digest.hexdigest()
    provenance['problem2_source_sha256']=source_hashes
    path=output/'generation_config.json'
    path.write_text(json.dumps(provenance,indent=2))
    tasks=[]
    for study in settings.get('studies', ['compact','extended']):
        ks=[3] if args.dry_run else list(range(3,6 if study=='compact' else 11))
        ms=[3] if args.dry_run else (list(range(2,11)) if study=='extended' else ks)
        quota=1 if args.dry_run else (args.compact_replicates if study=='compact' else 10)
        for dim in [1,2]:
            for model in fields.FIELD_MODELS:
                for K in ks:
                    for M in ms:
                        task=(study,dim,model,K,M,args.seed,quota,args.max_attempts,str(output))
                        tasks.append(task+(str(args.problem2_inputs.resolve()),) if study=='extended' else task)
    print(f'{len(tasks)} configurations; output={output}',flush=True)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,future in enumerate(as_completed([pool.submit(cell,t) for t in tasks]),1):
            r=future.result();results.append(r);print(f'{i}/{len(tasks)} {r["study"]} {r["dimension"]}D {r["field_family"]} K{r["K"]} M{r["M"]}: {r["accepted"]}, shortfall={r["shortfall"]}',flush=True)
    report=finalize(output,settings,results)
    for study,r in report.items():print(study,'instances',r['instances'],'unfilled configurations',len(r['shortfalls']),flush=True)
    if any(r['shortfalls'] for r in report.values()):
        raise RuntimeError('Incomplete datasets; increase --max-attempts and regenerate in an empty directory')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in provenance['source_sha256'].items())
    for study in args.studies:
        for name in ('generation_config.json','acceptance_report.json'):
            shutil.copyfile(output/name,output/study/name)

if __name__=='__main__':main()
