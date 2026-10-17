"""Generation and notebook integration checks without executing notebooks."""
from pathlib import Path
import sys, unittest, tempfile, json
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmark_pipeline')]
from generate import budgets, draw, cell, finalize
from reports import build_report
from src.analysis import select_cases,run_analysis
from src.search import ALGORITHM_POLICY
from src.probes import DICKE_POLICY

class BenchmarkV3Tests(unittest.TestCase):
    def test_budget_ceiling_cap_and_duplicates(self):
        self.assertEqual(budgets(1,16),(1,[dict(N=1,target_fractions=[.25,.5,.75,1.],actual_ratio=1.)]))
        maximum,entries=budgets(7,16)
        self.assertEqual(maximum,7);self.assertEqual([e['N'] for e in entries],[2,4,6,7])
        maximum,entries=budgets(80,50)
        self.assertEqual(maximum,50);self.assertEqual([e['N'] for e in entries],[13,25,38,50])
        self.assertEqual(entries[-1]['actual_ratio'],.625)

    def test_random_generation_reproducible_without_algorithm_calls(self):
        with patch('src.search.greedy_search',side_effect=AssertionError),patch('src.search.vnd_search',side_effect=AssertionError):
            first=draw('compact',1,'gaussian',3,3,77,1)
            self.assertEqual(first,draw('compact',1,'gaussian',3,3,77,1))
            self.assertNotEqual(first,draw('compact',1,'gaussian',3,3,77,2))
        self.assertNotIn('sampling_type',first[1])

    def test_small_nonzero_gaps_are_not_rejected(self):
        rows=[(0,1,0,.001)]
        meta=dict(K=2,M=1,min_best_pair_phase_gap=.001,generation_seed=1)
        with tempfile.TemporaryDirectory() as tmp, patch('generate.draw',return_value=(rows,meta)),patch('generate.joint_sensor_lower_bound',return_value=3142):
            result=cell(('compact',1,'gaussian',2,1,77,1,1,tmp))
            self.assertEqual(result['accepted'],1)
            self.assertEqual(result['validity_rejections'],0)

    def test_generation_has_no_checkpoint_dependency(self):
        rows=[(0,1,0,.2)]
        meta=dict(K=2,M=1,min_best_pair_phase_gap=.2,generation_seed=1)
        with tempfile.TemporaryDirectory() as tmp, patch('generate.draw',return_value=(rows,meta.copy())) as drawing, patch('generate.joint_sensor_lower_bound',return_value=16):
            task=('compact',1,'gaussian',2,1,77,1,1,tmp)
            first=cell(task)
            second=cell(task)
            self.assertEqual(drawing.call_count,2)
            self.assertEqual(first,second)
            self.assertFalse(list(Path(tmp).rglob('audit_*.json')))
            finalize(Path(tmp),dict(studies=['compact'],dry_run=True),[first])
            saved=json.loads((Path(tmp)/'compact/1d/manifest.json').read_text())
            self.assertEqual(saved['cases'],first['cases'])

    def test_extended_copies_problem2_bytes_and_uses_joint_budgets(self):
        source=ROOT.parent/'problem-2-min-error-seq/inputs'
        with tempfile.TemporaryDirectory() as tmp:
            result=cell(('extended',1,'gaussian',3,2,77,1,1,tmp,str(source)))
            case=result['cases'][0]
            self.assertEqual((Path(tmp)/'extended/1d'/case['file']).read_bytes(),
                             (source/'benchmark_1d'/case['source_file']).read_bytes())
            maximum,schedule=budgets(case['N_LB_joint'],50)
            self.assertEqual(case['budget_schedule'],schedule)
            self.assertEqual(case['N_max'],maximum)

    def test_tiny_phase_bound_uses_equivalent_row_scaling(self):
        from src.normalization import joint_sensor_lower_bound
        import math
        phase=math.pi/2_000_000_000
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tiny.csv'
            path.write_text(f'i,j,m,r_ijm\n0,1,0,{phase:.17g}\n')
            self.assertEqual(joint_sensor_lower_bound(path),2_000_000_000)

    def test_versioned_runner_wiring_with_mocked_solver(self):
        # Exercise loading/metadata using mocks, not a benchmark analysis.
        from src.search import AllocationResult
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp);(project/'src').mkdir();dataset=project/'version'/'extended'
            for dim in [1,2]:
                d=dataset/f'{dim}d';d.mkdir(parents=True)
                case=dict(file='x.csv',field_model='gaussian',K=2,M=1,phase_scale_c=1.,sensor_budgets=[2],replicate=1,
                          N_LB_joint=4,budget_schedule=[dict(N=2,target_fractions=[.5],actual_ratio=.5)])
                (d/'manifest.json').write_text(json.dumps(dict(design=dict(test_replicates_per_cell=1),cases=[case])))
                (d/'x.csv').write_text('i,j,m,r_ijm\n0,1,0,0.9\n')
            fake=lambda *args, **kwargs:AllocationResult((2,),.3,1,0.01,'greedy_complete')
            with patch('src.analysis.ROOT',project),patch.dict('src.analysis.SOLVERS',{'greedy':fake,'vnd':fake}),patch('src.analysis.exhaustive_search',side_effect=AssertionError):
                records,output=run_analysis('extended',dataset_root=dataset)
            self.assertEqual(len(records),20)
            self.assertTrue(all(r['budget_ratio']==.5 for r in records))

    def test_pair_reports_keep_vnd_comparison_when_exact_incomplete(self):
        base=dict(dimension=1,case='a',field_model='gaussian',K=3,M=3,N=4,probe='product',runtime_seconds=.1,
                  allocation=[2,1,1],budget_ratio=.5,N_LB_joint=8,target_fractions=[.5],
                  algorithm_policy=ALGORITHM_POLICY,dicke_policy=DICKE_POLICY)
        rows=[dict(base,method='exhaustive',status='timeout'),dict(base,method='greedy',status='greedy_complete',pgm_error=.4),
              dict(base,method='vnd',status='local_optimum',pgm_error=.3)]
        with tempfile.TemporaryDirectory() as tmp:
            report=build_report(rows,'compact',tmp)
            self.assertEqual(report['rows'],[])
            comparison=report['comparisons'][0]
            self.assertEqual(comparison['vnd_beats'],1)
            self.assertIsNone(comparison['greedy_exhaustive_gap'])

if __name__=='__main__':unittest.main()
