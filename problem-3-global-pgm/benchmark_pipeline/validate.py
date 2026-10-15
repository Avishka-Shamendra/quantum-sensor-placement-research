"""Validate ordinary-random datasets without executing analysis notebooks."""
from pathlib import Path
from collections import Counter
import argparse,csv,json,hashlib
import numpy as np
from generate import draw,budgets,source_values,fields
from src.input_data import load_field_input
from src.normalization import joint_sensor_lower_bound
from src.analysis import select_cases


def validate(root,recheck_bounds=True):
    root=Path(root);summary={};shortfalls=[]
    settings=json.loads((root/'generation_config.json').read_text())['settings']
    matched='problem2_inputs' in settings
    threshold=settings.get('minimum_phase_gap',.01)
    studies=[root.name] if (root/'1d').is_dir() else settings.get('studies', ('compact','extended'))
    for study in studies:
        count=0
        quota=1 if settings['dry_run'] else (settings.get('compact_replicates',20) if study=='compact' else 10)
        for dim in (1,2):
            directory=(root if (root/'1d').is_dir() else root/study)/f'{dim}d';manifest=json.loads((directory/'manifest.json').read_text())
            assert manifest['schema_version']==4
            cases=manifest['cases'];groups=Counter()
            audits=[json.loads(p.read_text()) for p in sorted(directory.glob('audit_*.json'))]
            assert len(audits)==(3 if settings['dry_run'] else (27 if study=='compact' else (216 if matched else 192)))
            assert {c['file'] for c in cases}=={p.name for p in directory.glob('*.csv')}
            assert len(cases)==len({c['instance_id'] for c in cases})
            assert len(cases)==sum(a['accepted'] for a in audits)
            for audit in audits:
                assert audit['accepted']+audit['shortfall']==quota
                assert audit['candidates_attempted']==len(audit['attempts'])
                assert audit['accepted']+audit['validity_rejections']==audit['candidates_attempted']
                assert all(a['reason'] in ('accepted','phase_separation','indistinguishable_pair') for a in audit['attempts'])
                if audit['shortfall']:shortfalls.append(audit['shortfall'])
                for a in audit['attempts']:
                    if matched and study=='extended':continue
                    _,meta=draw(study,dim,audit['field_family'],audit['K'],audit['M'],settings['seed'],a['attempt'])
                    assert (meta['min_best_pair_phase_gap']>0 if matched else meta['min_best_pair_phase_gap']>=threshold)==(a['reason']=='accepted')
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
                assert c['min_best_pair_phase_gap']>0 if matched else c['min_best_pair_phase_gap']>=threshold
                if recheck_bounds:assert joint_sensor_lower_bound(path)==c['N_LB_joint']
                maximum,schedule=budgets(c['N_LB_joint'],16 if study=='compact' else 50)
                assert (c['N_max'],c['budget_schedule'])==(maximum,schedule)
                assert c['sensor_budgets']==[s['N'] for s in schedule]
                assert 'sampling_type' not in c and 'difficulty_metrics' not in c
                groups[c['field_model'],c['K'],c['M']]+=1;count+=1
            assert all(v<=quota for v in groups.values())
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
