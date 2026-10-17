"""Publication estimators, pairing, references, and baseline reproducibility."""
from pathlib import Path
import sys,tempfile,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmark_pipeline')]
from publication import estimate,prepare,complete_states
from src.probes import FAMILIES
from src.search import ALGORITHM_POLICY


def fixture():
    rows=[]
    for dim in [1,2]:
        for sample in ['a','b']:
            for case_index in [0,1]:
                for N,fraction in [(2,.25),(4,.5),(6,.75),(8,1.)]:
                    for pi,probe in enumerate(FAMILIES):
                        for method,gap in [('exhaustive',0),('greedy',.003),('vnd',.001),('near_uniform',.02),('random',.03)]:
                            rows.append(dict(algorithm_policy=ALGORITHM_POLICY,dimension=dim,case=f'{sample}{case_index}',N=N,K=3,M=3,probe=probe,field_model='gaussian',
                                method=method,pgm_error=.4-.03*N+.01*pi+.01*case_index+gap*(1+.1*case_index),
                                status='optimal' if method=='exhaustive' else 'greedy_complete',budget_ratio=N/8,
                                target_fractions=[fraction],runtime_seconds=.02))
    return rows


class PublicationTests(unittest.TestCase):
    def test_rejects_previous_allocation_policy(self):
        rows=fixture()
        rows[0]['algorithm_policy']='phase_only_pair_greedy_score_vnd_v1'
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            prepare(rows,'compact',tmp)

    def test_confidence_clusters_repeated_budgets(self):
        base=dict(dimension=1,K=3,M=3,field_model='gaussian')
        rows=[dict(base,case='a',value=0.)]*5+[dict(base,case='b',value=1.)]
        result=estimate(rows,'value',resamples=1000)
        self.assertEqual(result['mean'],.5);self.assertEqual(result['instances'],2)
        self.assertEqual(result,estimate(rows,'value',resamples=1000))
        self.assertLessEqual(result['low'],.5);self.assertGreaterEqual(result['high'],.5)

    def test_compact_uses_exhaustive_and_error_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=prepare(fixture(),'compact',tmp,resamples=50)
            self.assertTrue(all(r['method']=='exhaustive' for r in report['state_rows']))
            self.assertAlmostEqual(next(r['gap'] for r in report['gaps'] if r['method']=='greedy'),.003)
            self.assertTrue(all(r['optimality_match_percent']==0 for r in report['quality']))

    def test_extended_vnd_reference_allows_negative_greedy_gap(self):
        rows=fixture()
        for r in rows:
            if r['method']=='greedy':r['pgm_error']=.01
        with tempfile.TemporaryDirectory() as tmp:
            report=prepare(rows,'extended',tmp,resamples=50)
            self.assertTrue(all(r['method']=='vnd' for r in report['state_rows']))
            self.assertTrue(all(r['reference_error']>.01 for r in report['gaps']))
            self.assertTrue(all(r['gap']<0 for r in report['gaps'] if r['method']=='greedy'))
            self.assertTrue(all(r['gap']==0 for r in report['gaps'] if r['method']=='vnd'))
            self.assertEqual(len(report['methods']),4)

    def test_missing_state_excluded_from_complete_comparisons(self):
        rows=fixture();rows[0]['status']='timeout'
        exact=complete_states(rows,('exhaustive',))
        self.assertFalse(any(r['dimension']==1 and r['case']=='a0' and r['N']==2 for r in exact))

if __name__=='__main__':unittest.main()
