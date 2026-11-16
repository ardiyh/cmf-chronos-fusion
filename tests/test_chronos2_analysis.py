import csv
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from experiments import run_chronos2_controls as c
from test_chronos2_controls import QuantileEndpoint


class Chronos2AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.chronos2_control_analysis'),
                             'Chronos-2 audit/analysis is missing')
        self.a=importlib.import_module('scripts.chronos2_control_analysis')
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.df=pd.DataFrame({'cov':np.arange(1500,dtype=float)*2,'y':np.arange(1500,dtype=float)})
        self.case=dict(dataset='toy',target='y',features=['cov','y'],seasonality=24,
            n_rows=1500,data_sha256=c.frame_hash(self.df,['cov','y']),origins={'24':[1024,1048],'48':[1024,1072]})
        self.run=self.root/'toy'
        c.run_grid(df=self.df,case=self.case,pipeline=QuantileEndpoint(),output_dir=self.run,
            provenance={},device='cpu')

    def rows(self):
        with (self.run/'losses.csv').open() as f:return list(csv.DictReader(f))

    def write_rows(self,rows):
        with (self.run/'losses.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=c.FIELDS);w.writeheader();w.writerows(rows)

    def test_full_audit_recomputes_metrics_and_checks_source_context(self):
        r=self.a.audit_run(self.run,case=self.case,df=self.df)
        self.assertEqual(r['records'],16)
        self.assertEqual(r['replay_pairs'],8)
        self.assertTrue(r['complete'])

    def test_changed_metric_or_seed_is_rejected(self):
        original=self.rows()
        for field,value in [('MASE','99'),('sampler_seed','1')]:
            rows=[dict(r) for r in original];rows[0][field]=value;self.write_rows(rows)
            with self.assertRaises(ValueError):self.a.audit_run(self.run,case=self.case)

    def test_partial_is_rejected_for_inference(self):
        m=json.loads((self.run/'manifest.json').read_text());m['status']='running'
        (self.run/'manifest.json').write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'complete'):self.a.audit_run(self.run)
        self.assertFalse(self.a.audit_run(self.run,require_complete=False)['complete'])

    def test_orphan_or_missing_npz_is_rejected(self):
        p=self.run/self.rows()[0]['prediction_file'];p.rename(p.with_name('orphan.npz'))
        with self.assertRaisesRegex(ValueError,'NPZ'):self.a.audit_run(self.run)

    def test_unexpected_case_grid_is_rejected(self):
        wrong=self.case|{'origins':{'24':[1024],'48':[1024]}}
        with self.assertRaises(ValueError):self.a.audit_run(self.run,case=wrong)

    def test_held_out_source_context_is_checked(self):
        changed=self.df.copy();changed.loc[1023,'cov']=-100
        with self.assertRaises(ValueError):self.a.audit_run(self.run,df=changed)

    def test_family_has_sixteen_holm_tests_and_fixed_direction(self):
        cases=[]
        for d in c.DATASETS:
            for h in [24,48]:
                origins=list(range(1024,1504,24))
                cases.append(dict(dataset=d,horizon=h,origins=origins,
                    target_only={o:2+i/10 for i,o in enumerate(origins)},
                    past_covariates={o:1+i/20 for i,o in enumerate(origins)}))
        result=self.a.analyze_cases(cases)
        self.assertEqual(len(result),16)
        self.assertTrue(all(r['mean_diff']<0 and r['win_rate']==1 for r in result))
        self.assertTrue(all(0<=r['holm_p']<=1 and 0<=r['holm_block_p']<=1 for r in result))


if __name__=='__main__':unittest.main()
