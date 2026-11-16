"""CPU contract tests; no Chronos-2 weights or CUDA are loaded."""
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch


class QuantileEndpoint:
    def __init__(self):
        self.inputs = []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kwargs):
        self.inputs.append(inputs)
        base = float(inputs[0]['target'][-1])
        q = torch.tensor(base + np.array(quantile_levels), dtype=torch.float32)
        return [q.reshape(1, 1, -1).repeat(1, prediction_length, 1)], [torch.full((1,prediction_length),999.)]


class Chronos2ControlsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.run_chronos2_controls'),
                             'Chronos-2 paired runner is missing')
        self.c = importlib.import_module('experiments.run_chronos2_controls')
        self.df = pd.DataFrame({'cov':[10.,20.,30.,40.,5000.,6000.],
                                'y':[1.,2.,3.,4.,5.,6.]})
        self.pipeline = QuantileEndpoint()
        self.args = dict(df=self.df, origin=4, horizon=2, context_length=4,
            feature_cols=['cov','y'], target_col='y', seasonality=2,
            variant='target_only', pipeline=self.pipeline, seed=2026)

    def evaluate(self, **changes):
        return self.c.evaluate_origin(**(self.args|changes))

    def test_target_only_receives_no_covariates(self):
        self.evaluate()
        self.assertEqual(set(self.pipeline.inputs[0][0]), {'target'})
        np.testing.assert_array_equal(self.pipeline.inputs[0][0]['target'],[1,2,3,4])

    def test_covariates_are_past_only_and_exclude_target(self):
        self.evaluate(variant='past_covariates')
        payload=self.pipeline.inputs[0][0]
        self.assertEqual(set(payload),{'target','past_covariates'})
        self.assertEqual(set(payload['past_covariates']),{'cov'})
        np.testing.assert_array_equal(payload['past_covariates']['cov'],[10,20,30,40])
        self.df.loc[4:,'cov']=-999999
        self.evaluate(variant='past_covariates')
        np.testing.assert_array_equal(self.pipeline.inputs[1][0]['past_covariates']['cov'],[10,20,30,40])

    def test_median_quantile_is_used_instead_of_mean_label(self):
        r=self.evaluate()
        np.testing.assert_array_equal(r['median_pred'],[4.5,4.5])
        self.assertAlmostEqual(r['MASE'],.5)
        self.assertAlmostEqual(r['MAE'],1.)
        self.assertAlmostEqual(r['width_50'],.5,places=5)
        self.assertAlmostEqual(r['width_90'],.9,places=5)

    def test_wql_uses_seven_prespecified_levels_in_target_units(self):
        r=self.evaluate()
        expected=0.
        for q in [.05,.1,.2,.5,.8,.9,.95]:
            pred=r['q_preds'][:,self.c.QUANTILES.index(q)]
            error=r['y_true']-pred
            expected+=2*np.maximum(q*error,(q-1)*error).mean()/7
        self.assertAlmostEqual(r['WQL'],expected)

    def test_seed_is_paired_and_rng_is_restored(self):
        first=self.c.seed_for('x','y',24,1024)
        self.assertEqual(first,self.c.seed_for('x','y',24,1024))
        self.assertNotEqual(first,self.c.seed_for('x','y',48,1024))
        state=torch.random.get_rng_state().clone()
        self.evaluate()
        torch.testing.assert_close(torch.random.get_rng_state(),state)

    def test_invalid_quantiles_fail_without_replacing_forecasts(self):
        class Bad(QuantileEndpoint):
            def predict_quantiles(self,*args,**kwargs):
                return [torch.full((1,2,9),float('nan'))],[torch.zeros((1,2))]
        with self.assertRaisesRegex(ValueError,'finite'):
            self.evaluate(pipeline=Bad())

    def test_npz_is_exclusive_and_contains_actual_inputs(self):
        r=self.evaluate(variant='past_covariates')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'record.npz';self.c.write_prediction(p,r)
            with np.load(p,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['past_covariates'],[[10,20,30,40]])
                self.assertEqual(z['q_preds'].shape,(2,9))
                self.assertEqual(json.loads(z['metadata_json'].item())['variant'],'past_covariates')
            with self.assertRaises(FileExistsError):self.c.write_prediction(p,r)

    def test_grid_replays_each_configuration_and_never_overwrites(self):
        df=pd.DataFrame({'cov':np.arange(1500,dtype=float)*2,'y':np.arange(1500,dtype=float)})
        case=dict(dataset='toy',target='y',features=['cov','y'],seasonality=24,
            n_rows=1500,data_sha256=self.c.frame_hash(df,['cov','y']),
            origins={'24':[1024,1048],'48':[1024]})
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'run'
            m=self.c.run_grid(df=df,case=case,pipeline=QuantileEndpoint(),output_dir=p,
                provenance={'checkpoint_revision':self.c.REVISION},device='cpu')
            self.assertEqual(m['n_records'],12)
            self.assertEqual(m['status'],'complete')
            self.assertEqual(m['max_replay_abs_diff'],0)
            with self.assertRaises(FileExistsError):
                self.c.run_grid(df=df,case=case,pipeline=QuantileEndpoint(),output_dir=p,
                    provenance={},device='cpu')


if __name__=='__main__':
    unittest.main()
