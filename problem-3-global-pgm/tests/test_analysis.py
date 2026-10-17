"""Check sampling coverage and exclusion of incomplete algorithm pairs."""
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.search import ALGORITHM_POLICY
from src.analysis import ROOT, select_cases, summarize, run_analysis


class AnalysisTests(unittest.TestCase):
    def test_selection(self):
        for study, expected, cells, reps in [('compact',27,27,1),('extended',216,216,1)]:
            for dim in (1,2):
                name=f'{"compact_" if study=="compact" else ""}benchmark_{dim}d'
                manifest=json.loads((ROOT/'inputs'/study/f'{dim}d'/'manifest.json').read_text())
                selected=select_cases(manifest,study,'test')
                self.assertEqual(len(selected),expected)
                self.assertEqual(len({c['file'] for c in selected}),expected)
                self.assertEqual(len(manifest['cases']),cells*(50 if study=='compact' else 10))
                self.assertEqual(select_cases(manifest,study,'full'),manifest['cases'])
                self.assertEqual(select_cases(manifest,study,'test'),selected)
                counts=Counter((c['field_model'],c['K'],c['M']) for c in selected)
                self.assertEqual(len(counts),cells)
                self.assertEqual(set(counts.values()),{reps})
                self.assertTrue(all(c['sensor_budgets']==[b['N'] for b in c['budget_schedule']] for c in selected))
                self.assertEqual(set(Counter(c['field_model'] for c in selected).values()),{expected//3})

    def test_allocation_ignores_manifest_fields_and_uses_signed_csv(self):
        import numpy as np
        from src.search import greedy_search, vnd_search
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory);(project/'src').mkdir()
            dataset=project/'dataset'
            phase=2*np.pi-6
            case=dict(file='case.csv',replicate=1,field_model='gaussian',K=2,M=2,
                      phase_scale_c=2.,sensor_budgets=[1],field_values=[[0,0],[3,.2]])
            for dim in (1,2):
                folder=dataset/f'{dim}d';folder.mkdir(parents=True)
                (folder/'manifest.json').write_text(json.dumps(dict(design=dict(test_replicates_per_cell=1),cases=[case])))
                (folder/'case.csv').write_text(f'i,j,m,r_ijm\n0,1,0,{phase}\n0,1,1,-0.4\n')
            def checked(solver):
                def call(N,M,r_ijm,**kwargs):
                    np.testing.assert_allclose(r_ijm,[[phase,-.4]],rtol=0,atol=1e-15)
                    return solver(N,M,r_ijm,**kwargs)
                return call
            with patch('src.analysis.ROOT',project), patch.dict('src.analysis.SOLVERS',
                    {'greedy':checked(greedy_search),'vnd':checked(vnd_search)}):
                records,_=run_analysis('extended',dataset_root=dataset)
            self.assertTrue(all(r['allocation']==[0,1] for r in records))

    def test_runner_preserves_saved_budgets_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory)
            (project/'src').mkdir()
            case=dict(file='case.csv',field_model='gaussian',K=2,M=1,
                      phase_scale_c=1.0,sensor_budgets=[3,7])
            for dim in (1,2):
                dataset=project/'inputs'/f'compact_benchmark_{dim}d'
                dataset.mkdir(parents=True)
                (dataset/'manifest.json').write_text(json.dumps({'cases':[case]}))
                (dataset/'case.csv').write_text('i,j,m,r_ijm\n0,1,0,0.2\n')
            # Exercise the test-mode runner with a tiny selected fixture; selection
            # coverage on the real four datasets is checked independently above.
            with patch('src.analysis.ROOT',project), patch('src.analysis.select_cases',return_value=[case]):
                records,output=run_analysis('compact','test')
                self.assertEqual(len(records),60)
                self.assertEqual({r['N'] for r in records},{3,7})
                self.assertEqual({r['status'] for r in records},{'optimal','local_optimum','greedy_complete'})
                self.assertTrue((output/'selected_cases.csv').exists())
                with patch('src.analysis.probe_gram',side_effect=AssertionError('should reuse results')):
                    resumed,_=run_analysis('compact','test')
                    self.assertEqual(records,resumed)

    def test_state_independent_allocations_and_partial_resume(self):
        from src.search import greedy_search, vnd_search
        from src.probes import probe_gram
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory);(project/'src').mkdir()
            case=dict(file='case.csv',replicate=1,field_model='gaussian',K=2,M=2,
                      phase_scale_c=1.,sensor_budgets=[3],field_values=[[0,0],[.2,.5]])
            dataset=project/'dataset'
            for dim in (1,2):
                d=dataset/f'{dim}d';d.mkdir(parents=True)
                (d/'manifest.json').write_text(json.dumps(dict(design=dict(test_replicates_per_cell=1),cases=[case])))
                (d/'case.csv').write_text('i,j,m,r_ijm\n0,1,0,-0.2\n0,1,1,-0.5\n')
            from unittest.mock import Mock
            g=Mock(wraps=greedy_search);v=Mock(wraps=vnd_search)
            with patch('src.analysis.ROOT',project),patch.dict('src.analysis.SOLVERS',greedy=g,vnd=v),patch('src.analysis.probe_gram',wraps=probe_gram) as pgm:
                records,output=run_analysis('extended',dataset_root=dataset)
                self.assertEqual((g.call_count,v.call_count,pgm.call_count),(2,2,20))
                for dim in (1,2):
                    for method in ('greedy','vnd'):
                        rows=[r for r in records if r['dimension']==dim and r['method']==method]
                        self.assertEqual(len({tuple(r['allocation']) for r in rows}),1)
                        self.assertTrue(all(r['pgm_evaluations']==1 for r in rows))
                path=output/'records.jsonl'
                path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n')
                resumed,_=run_analysis('extended',dataset_root=dataset)
                self.assertEqual(len(resumed),20)
                self.assertEqual((g.call_count,v.call_count,pgm.call_count),(2,2,21))
                baseline,_=run_analysis('extended',dataset_root=dataset,include_baselines=True)
                self.assertEqual(len(baseline),40)
                self.assertTrue(all(r['pgm_evaluations']==20 for r in baseline if r['method']=='random'))

    def test_extended_switch_and_matched_three_methods(self):
        base=dict(algorithm_policy=ALGORITHM_POLICY,dimension=1,case='a',field_model='gaussian',K=3,M=2,N=2,probe='product',runtime_seconds=0.1,allocation=[1,1])
        rows=[dict(base,method='exhaustive',status='optimal',pgm_error=0.1),
              dict(base,method='greedy',status='local_optimum',pgm_error=0.3),
              dict(base,method='vnd',status='local_optimum',pgm_error=0.2)]
        with tempfile.TemporaryDirectory() as output:
            matches,summary,_=summarize(rows,output,methods=('exhaustive','greedy','vnd'))
            self.assertEqual(len(matches),3)
            self.assertTrue(all(r['reference']=='exhaustive' for r in summary))
            timed_out=rows[:2]+[dict(rows[2],status='timeout')]
            self.assertEqual(summarize(timed_out,output,methods=('exhaustive','greedy','vnd'))[0],[])
            matches,summary,_=summarize(rows[1:],output,methods=('greedy','vnd'))
            self.assertEqual(len(matches),2)
            self.assertAlmostEqual(next(r for r in summary if r['method']=='vnd')['error_gap_to_reference'],-0.1)
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory);(project/'src').mkdir()
            case=dict(file='case.csv',field_model='gaussian',K=2,M=1,phase_scale_c=1.0,sensor_budgets=[3])
            for dim in (1,2):
                dataset=project/'inputs'/f'benchmark_{dim}d';dataset.mkdir(parents=True)
                (dataset/'manifest.json').write_text(json.dumps({'cases':[case]}))
                (dataset/'case.csv').write_text('i,j,m,r_ijm\n0,1,0,0.2\n')
            with patch('src.analysis.ROOT',project), patch('src.analysis.select_cases',return_value=[case]), patch('src.analysis.exhaustive_search',side_effect=AssertionError('extended must not run exhaustive')):
                records,_=run_analysis('extended','test')
                self.assertEqual({r['method'] for r in records},{'greedy','vnd'})
                self.assertEqual(len(records),20)

    def test_only_complete_pairs_enter_summary(self):
        base=dict(algorithm_policy=ALGORITHM_POLICY,dimension=1,case='a',field_model='gaussian',K=3,M=2,N=2,probe='product',runtime_seconds=0.1,allocation=[1,1])
        rows=[dict(base,method='exhaustive',status='optimal',pgm_error=0.2),
              dict(base,method='greedy',status='local_optimum',pgm_error=0.3),
              dict(base,case='b',method='exhaustive',status='timeout'),
              dict(base,case='b',method='greedy',status='local_optimum',pgm_error=0.9)]
        with tempfile.TemporaryDirectory() as output:
            pairs, summary, status=summarize(rows,output)
            self.assertEqual(len(pairs),2)
            self.assertAlmostEqual(next(r for r in summary if r['method']=='greedy')['error_gap_to_reference'],0.1)
            self.assertEqual(summary[0]['paired_case_budgets'],1)
            self.assertEqual(sum(r['count'] for r in status),4)
            self.assertTrue((Path(output)/'status.csv').exists())


if __name__=='__main__':
    unittest.main()
