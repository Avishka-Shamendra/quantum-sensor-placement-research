"""Joint lower bound and per-case budget-ratio checks."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.normalization import joint_sensor_lower_bound, normalize_budget


class NormalizationTests(unittest.TestCase):
    def write(self,path,phases):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['i','j','m','r_ijm'])
            for i in range(len(phases)):
                for j in range(i+1,len(phases)):
                    for m in range(len(phases[0])):
                        value=float(np.angle(np.exp(1j*(phases[i][m]-phases[j][m]))))
                        if value<=-np.pi:value=np.pi
                        writer.writerow([i,j,m,value])

    def test_paper_example_and_two_location_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'case.csv'
            self.write(path,[[0],[np.pi/2],[np.pi]])
            self.assertEqual(joint_sensor_lower_bound(path),2)
            self.write(path,[[0,0],[0.8,1.1]])
            self.assertEqual(joint_sensor_lower_bound(path),3)
            self.write(path,[[0],[0]])
            with self.assertRaises(ValueError):joint_sensor_lower_bound(path)

    def test_ratio_and_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as folder:
            project=Path(folder)
            path=project/'inputs/compact_benchmark_1d/case.csv'
            self.write(path,[[0],[np.pi/2]])
            rows=[dict(dimension=1,case='case.csv',N=3)]
            output=project/'out'
            first=normalize_budget(rows,project,'compact',output)
            self.assertEqual(first[0]['budget_ratio'],1.5)
            with patch('src.normalization.joint_sensor_lower_bound',side_effect=AssertionError('should use cache')):
                self.assertEqual(normalize_budget(rows,project,'compact',output),first)
            self.write(path,[[0],[np.pi/4]])
            second=normalize_budget(rows,project,'compact',output)
            self.assertEqual(second[0]['N_LB_joint'],4)
            self.assertEqual(second[0]['budget_ratio'],0.75)
            self.assertNotIn('budget_ratio',rows[0])


if __name__=='__main__':unittest.main()
