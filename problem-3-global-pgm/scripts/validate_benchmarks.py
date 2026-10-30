"""Validate ordinary-random datasets without executing analysis notebooks."""
from pathlib import Path
from collections import Counter
import argparse,csv,json,hashlib
import numpy as np
from generate_benchmarks import draw,budgets,source_values,fields,MINIMUM_PHASE_GAP
from src.input_data import load_field_input
from src.normalization import joint_sensor_lower_bound
from src.analysis import select_cases


def validate(root,recheck_bounds=True):
    root=Path(root);summary={};shortfalls=[]
    config=root/'generation_config.json'
    if not config.exists() and (root/'1d').is_dir():
        config=root.parent/'generation_config.json'
    settings=json.loads(config.read_text())['settings']
    matched='problem2_inputs' in settings
    threshold=max(settings.get('minimum_phase_gap',MINIMUM_PHASE_GAP),MINIMUM_PHASE_GAP)
    studies=[root.name] if (root/'1d').is_dir() else settings.get('studies', ('compact','extended'))
    for study in studies:
        count=0
        quota=1 if settings['dry_run'] else (settings.get('compact_replicates',20) if study=='compact' else 10)
        for dim in (1,2):
            directory=(root if (root/'1d').is_dir() else root/study)/f'{dim}d';manifest=json.loads((directory/'manifest.json').read_text())
            assert manifest['schema_version']==4
            cases=manifest['cases'];groups=Counter()
            assert {c['file'] for c in cases}=={p.name for p in directory.glob('*.csv')}
            assert len(cases)==len({c['instance_id'] for c in cases})
            ks=[3] if settings['dry_run'] else list(range(3,6 if study=='compact' else 11))
            ms=[3] if settings['dry_run'] else (list(range(2 if matched else 3,11)) if study=='extended' else list(range(3,6)))
            expected={(model,k,m) for model in fields.FIELD_MODELS for k in ks for m in ms}
            for c in cases:
                path=directory/c['file'];f=load_field_input(path)
                if matched and study=='extended':
                    source=Path(settings['problem2_inputs'])/c['source_dataset']/c['source_file']
                    assert path.read_bytes()==source.read_bytes()
                    assert hashlib.sha256(path.read_bytes()).hexdigest()==c['source_sha256']
                    values=source_values(c)
                    np.testing.assert_array_equal(values,c['field_values'])
                    rows=fields.phase_gap_rows(values,c['phase_scale_c'])
                else:
                    rows,meta=draw(study,dim,c['field_model'],c['K'],c['M'],settings['seed'],c['generation_attempt'])
                    for key,value in meta.items():assert json.loads(json.dumps(value))==c[key]
                with path.open() as stream:
                    reader=csv.DictReader(stream);assert reader.fieldnames==['i','j','m','r_ijm']
                    actual=np.array([[float(row[k]) for k in reader.fieldnames] for row in reader])
                np.testing.assert_array_equal(actual,np.asarray(rows))
                raw=np.asarray(c['field_values']);np.testing.assert_allclose(f,raw-raw[0],atol=1e-12)
                minimum_gap=float(np.min(np.max(abs(actual[:,3].reshape(-1,c['M'])),axis=1)))
                assert minimum_gap>=threshold, f'{path}: pair phase gap below {threshold}'
                assert minimum_gap==c['min_best_pair_phase_gap']
                if recheck_bounds:assert joint_sensor_lower_bound(path)==c['N_LB_joint']
                maximum,schedule=budgets(c['N_LB_joint'],16 if study=='compact' else 50)
                assert (c['N_max'],c['budget_schedule'])==(maximum,schedule)
                assert c['sensor_budgets']==[s['N'] for s in schedule]
                assert 'sampling_type' not in c and 'difficulty_metrics' not in c
                groups[c['field_model'],c['K'],c['M']]+=1;count+=1
            assert set(groups)<=expected
            assert all(v<=quota for v in groups.values())
            for model,k,m in sorted(expected):
                missing=quota-groups[model,k,m]
                if missing:
                    shortfalls.append(dict(study=study,dimension=dim,field_family=model,K=k,M=m,shortfall=missing))
            if matched and study=='extended' and not settings['dry_run']:
                original=json.loads((Path(settings['problem2_inputs'])/f'benchmark_{dim}d'/'manifest.json').read_text())
                assert {c['file'] for c in cases}=={c['file'] for c in original['cases']}
                assert {c['M'] for c in cases}==set(range(2,11))
            assert len(select_cases(manifest,study,'test'))==len(groups)
        summary[study]=count
    result=dict(validated_instances=sum(summary.values()),study_counts=summary,shortfalls=shortfalls,
                rechecked_lower_bounds=recheck_bounds)
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset_root',type=Path);args=parser.parse_args()
    print(json.dumps(validate(args.dataset_root),indent=2))
