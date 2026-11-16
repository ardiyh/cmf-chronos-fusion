"""CPU contracts for a fresh synchronized latency panel; never load weights."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from experiments import run_chronos2_controls as c2
from experiments import run_peer_review_controls as c
from test_chronos2_controls import QuantileEndpoint


class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now
    def advance(self,n):self.now+=n


class LatencyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.measure_guarded_latency'),
                             'Reviewed latency runner missing')
        self.m=importlib.import_module('experiments.measure_guarded_latency')
        self.df=pd.DataFrame({'cov':np.arange(1800,dtype=float)*2,'y':np.arange(1800,dtype=float)})
        case=dict(target='y',features=['cov','y'],seasonality=24,n_rows=1800,
            data_sha256=c.frame_sha256(self.df,['cov','y']),origins={'24':list(range(1024,1456,24)),
            '48':list(range(1024,1456,48))})
        self.pf={'datasets':{d:copy.deepcopy(case) for d in c2.DATASETS}}
        self.selection={(d,h):'ridge' for d in c2.DATASETS for h in (24,48)}
        self.plan=self.m.make_plan(self.pf,self.selection,include_c2_target=True)

    def test_three_endpoint_plan_is_exactly_864_and_optional_target_adds_288(self):
        q=self.m.make_plan(self.pf,self.selection,include_c2_target=False)
        self.assertEqual(q['expected_calls'],864)
        self.assertEqual(q['warmup_calls'],144)
        self.assertEqual(q['measured_calls'],720)
        p=self.m.make_plan(self.pf,self.selection,include_c2_target=True)
        self.assertEqual(p['expected_calls'],1152)
        self.assertEqual(p['measured_calls'],960)

    def test_origins_are_a_priori_evenly_spaced_and_match_across_endpoints(self):
        case=self.plan['cases'][0]
        source=self.pf['datasets'][case['dataset']]['origins'][str(case['horizon'])]
        self.assertEqual(case['origins'],[source[i] for i in np.linspace(0,len(source)-1,5,dtype=int)])
        tasks=self.m.tasks_for_case(case,'t5')
        measured=[t for t in tasks if t['phase']=='measured']
        for method in ['t5_bare','cmf_selected']:
            self.assertEqual({t['origin'] for t in measured if t['method']==method},set(case['origins']))
            self.assertEqual(sum(t['method']==method for t in measured),15)
        self.assertEqual([t['method'] for t in measured[:4]],
            ['t5_bare','cmf_selected','cmf_selected','t5_bare'])

    def test_validation_selection_and_full_sixteen_cases_are_required(self):
        missing=dict(self.selection);missing.pop(('etth1',24))
        with self.assertRaises(ValueError):self.m.make_plan(self.pf,missing)
        bad=copy.deepcopy(self.pf);bad['datasets'].pop('ecl')
        with self.assertRaises(ValueError):self.m.make_plan(bad,self.selection)

    def test_timer_wait_is_outside_and_synchronization_is_inside_components(self):
        clock=Clock();clock.advance(3.) # scheduling delay, before timed region
        def stage(seconds,value):
            def run(*_):clock.advance(seconds);return value
            return run
        result,t=self.m.timed_call(stage(.002,'input'),stage(.005,'output'),
            stage(.003,'result'),sync=lambda:clock.advance(.001),clock=clock)
        self.assertEqual(result,'result')
        np.testing.assert_allclose([t[k] for k in ['prepare_fit_ms','endpoint_ms','postprocess_ms','total_ms']],
                                   [3,6,4,13],atol=1e-10)
        self.assertAlmostEqual(t['total_ms'],sum(t[k] for k in ['prepare_fit_ms','endpoint_ms','postprocess_ms']))

    def test_synchronization_failure_is_not_ignored(self):
        calls=[]
        def fail():raise RuntimeError('sync failed')
        with self.assertRaisesRegex(RuntimeError,'sync failed'):
            self.m.timed_call(lambda:calls.append(1),lambda x:x,lambda x:x,sync=fail)
        self.assertEqual(calls,[])

    def test_t5_preparation_matches_canonical_past_fit_without_reading_horizon(self):
        for adapter in ['identity','ridge','krr','mlp']:
            with c.isolated_rng(42):
                expected,_,_=c.fit_history(df=self.df,origin=1024,adapter_kind=adapter,
                    context_len=512,pred_len=24,feature_cols=['cov','y'],target_col='y',
                    seasonality=24,device_map='cpu',fit_seed=42)
            case=self.plan['cases'][0]|dict(adapter=adapter,adapter_kwargs=c.legacy.ADAPTER_CONFIGS[adapter].copy())
            actual=self.m.prepare_input(self.df,case,1024,'cmf_selected',device='cpu')
            np.testing.assert_array_equal(actual.numpy(),expected.numpy())
            changed=self.df.copy();changed.loc[1024:,:]=np.nan
            actual2=self.m.prepare_input(changed,case,1024,'cmf_selected',device='cpu')
            np.testing.assert_array_equal(actual.numpy(),actual2.numpy())

    def test_c2_payload_is_past_only_and_target_is_excluded_from_covariates(self):
        case=self.plan['cases'][0]
        bare=self.m.prepare_input(self.df,case,1024,'c2_target_only',device='cpu')[0]
        native=self.m.prepare_input(self.df,case,1024,'c2_native',device='cpu')[0]
        self.assertEqual(set(bare),{'target'})
        self.assertEqual(set(native['past_covariates']),{'cov'})
        np.testing.assert_array_equal(native['past_covariates']['cov'],self.df['cov'].to_numpy()[512:1024].astype(np.float32))

    def test_both_endpoints_compute_all_nine_quantiles_and_use_q05(self):
        h=24;samples=torch.arange(50*h,dtype=torch.float32).reshape(1,50,h)
        r=self.m.finish_output(samples,h,'t5_bare')
        self.assertEqual(r['q_preds'].shape,(h,9))
        np.testing.assert_array_equal(r['median_pred'],np.median(samples.numpy()[0],axis=0))
        output=QuantileEndpoint().predict_quantiles([{'target':np.arange(512)}],h,c2.QUANTILES)
        r=self.m.finish_output(output,h,'c2_native')
        np.testing.assert_array_equal(r['median_pred'],np.full(h,511.5))

    def test_nonfinite_and_wrong_output_shapes_fail_without_zero_substitution(self):
        for output in [torch.zeros((1,49,24)),torch.full((1,50,24),float('nan'))]:
            with self.assertRaises(ValueError):self.m.finish_output(output,24,'t5_bare')

    def test_summary_excludes_warmup_and_requires_complete_balanced_panel(self):
        rows=[]
        for case in self.plan['cases']:
            for family in ['t5','c2']:
                for t in self.m.tasks_for_case(case,family):
                    rows.append(t|dict(dataset=case['dataset'],horizon=case['horizon'],
                        adapter_fit_ms=0.,prepare_fit_ms=1.,endpoint_ms=2.,postprocess_ms=3.,total_ms=6.))
        result=self.m.summarize(self.plan,rows)
        self.assertEqual(len(result),64)
        self.assertTrue(all(r['n_origins']==5 and r['n_measured']==15 and r['n_warmup']==3
                            and r['total_ms_median']==6 and r['total_ms_iqr']==0 for r in result))
        with self.assertRaises(ValueError):self.m.summarize(self.plan,rows[:-1])
        with self.assertRaises(ValueError):self.m.summarize(self.plan,rows+[rows[-1]])

    def test_phase_audit_checks_forecast_hash_and_timer_identity(self):
        self.assertTrue(hasattr(self.m,'audit_phase'),'Phase artefact audit missing')
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'predictions').mkdir()
            rows=[]
            for case in self.plan['cases']:
                for task in self.m.tasks_for_case(case,'t5'):
                    samples=torch.arange(50*case['horizon'],dtype=torch.float32).reshape(1,50,case['horizon'])
                    result=self.m.finish_output(samples,case['horizon'],task['method'])
                    rel='predictions/'+str(len(rows))+'.npz';np.savez_compressed(out/rel,**result)
                    row=task|dict(dataset=case['dataset'],horizon=case['horizon'],
                        sampler_seed=c.sampler_seed_for_origin(dataset=case['dataset'],target=case['target'],horizon=case['horizon'],origin=task['origin'],repetition=task['repeat'],base_seed=2026),
                        adapter_fit_ms=0.,prepare_fit_ms=1.,endpoint_ms=2.,postprocess_ms=3.,total_ms=6.,
                        prediction_file=rel,prediction_sha256=c2.digest(out/rel))
                    rows.append(row)
            (out/'timings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertEqual(self.m.audit_phase(out,self.plan,'t5'),rows)
            p=out/rows[0]['prediction_file'];p.write_bytes(p.read_bytes()+b'corrupt')
            with self.assertRaises(ValueError):self.m.audit_phase(out,self.plan,'t5')

    def test_qualification_gate_checks_duration_cap_flags_and_rpm(self):
        self.assertTrue(hasattr(self.m,'assess_hardware'),'Qualification gate missing')
        g=dict(status='complete',returncode=0,required_power_limit_W=160,
               temperature_guard_C=85,fan_rpm_logged=True)
        m=dict(status='qualification_window_complete',stage='qualification',family='c2',
               calls=20,elapsed_prediction_window_s=90.1)
        rows=[dict(temperature=60.,power_limit=160.,power_draw=80.,fan0_rpm=1000.,fan1_rpm=1000.,
                   hw_thermal='Not Active',sw_thermal='Not Active',power_brake='Not Active')]
        self.m.assess_hardware(m,g,rows,self.plan)
        for changed in [dict(elapsed_prediction_window_s=89),dict(calls=11)]:
            with self.assertRaises(ValueError):self.m.assess_hardware(m|changed,g,rows,self.plan)
        for changed in [dict(status='guard_stopped'),dict(temperature_guard_C=95),dict(required_power_limit_W=220)]:
            with self.assertRaises(ValueError):self.m.assess_hardware(m,g|changed,rows,self.plan)
        for changed in [dict(sw_thermal='Active'),dict(temperature=79),dict(temperature=70,fan0_rpm=0)]:
            with self.assertRaises(ValueError):self.m.assess_hardware(m,g,[rows[0]|changed],self.plan)

    def test_gpu_runner_requires_review_receipt_for_the_exact_plan(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'review.json';plan=Path(td)/'plan.json';plan.write_text(json.dumps(self.plan))
            p.write_text(json.dumps(dict(approved=True,reviewer='reviewer',plan_sha256='wrong')))
            with self.assertRaises(ValueError):self.m.require_review(plan,p)
            p.write_text(json.dumps(dict(approved=False,reviewer='reviewer',plan_sha256=c2.digest(plan))))
            with self.assertRaises(ValueError):self.m.require_review(plan,p)
            p.write_text(json.dumps(dict(approved=True,reviewer='',plan_sha256=c2.digest(plan))))
            with self.assertRaises(ValueError):self.m.require_review(plan,p)
            p.write_text(json.dumps(dict(approved=True,reviewer='reviewer',plan_sha256=c2.digest(plan))))
            self.m.require_review(plan,p)


if __name__=='__main__':unittest.main()
