"""Prospective C8 latency panel; plan/review first, then an external GPU guard.

Warm-up, scheduling waits, model loading and evidence IO are excluded from
per-call times. Preparation (including fresh adapter fit), endpoint API and
nine-quantile postprocessing are timed separately with CUDA barriers.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess
import importlib.metadata

import numpy as np
import torch

from experiments import run_chronos2_controls as c2
from experiments import run_peer_review_controls as c

T5_MODEL='amazon/chronos-t5-base'
T5_REVISION='ad294eaacead15db499b740ea4122266dd2a81a2'
SOURCES=('experiments/measure_guarded_latency.py','experiments/run_peer_review_controls.py',
         'experiments/run_revision.py','experiments/run_chronos2_controls.py','src/cmf_core.py',
         'scripts/run_a4_paced.py','scripts/run_with_gpu_guard.py')
TIMES=('adapter_fit_ms','prepare_fit_ms','endpoint_ms','postprocess_ms','total_ms')


def make_plan(preflight,selection,*,include_c2_target=True):
    datasets=preflight['datasets']
    if set(datasets)!=set(c2.DATASETS) or set(selection)!={(d,h) for d in c2.DATASETS for h in (24,48)}:
        raise ValueError('All sixteen frozen cases and validation choices are required')
    methods=['t5_bare','cmf_selected','c2_native']
    if include_c2_target:methods.insert(2,'c2_target_only')
    cases=[]
    for d in c2.DATASETS:
        for h in (24,48):
            v=datasets[d];origins=v['origins'][str(h)]
            if len(origins)<5 or len(set(origins))!=len(origins):raise ValueError('At least five distinct origins required')
            adapter=selection[d,h]
            if adapter not in c.LEGACY_KINDS:raise ValueError('Unsupported frozen adapter')
            kwargs=c.legacy.ADAPTER_CONFIGS[adapter].copy()
            if adapter=='mlp':kwargs['mlp_device']='cuda'
            cases.append(dict(dataset=d,horizon=h,target=v['target'],features=v['features'],
                seasonality=v['seasonality'],n_rows=v['n_rows'],data_sha256=v['data_sha256'],
                origins=[origins[i] for i in np.linspace(0,len(origins)-1,5,dtype=int)],
                adapter=adapter,adapter_kwargs=kwargs,methods=methods))
    measured=16*len(methods)*5*3;warmup=16*len(methods)*3
    root=Path(__file__).resolve().parents[1]
    return dict(schema='cmf-c8-latency-v1',review_status='pending',cases=cases,
        methods=methods,context_length=512,num_samples=50,quantiles=c2.QUANTILES,
        n_origins=5,n_repeats=3,n_warmup=3,base_seed=2026,fit_seed=42,
        delay_before_prediction_s=3,required_power_W=160,guard_C=85,
        primary_outputs='median and nine quantiles in target units; real S50 paths for T5',
        expected_calls=measured+warmup,measured_calls=measured,warmup_calls=warmup,
        t5_model=T5_MODEL,t5_revision=T5_REVISION,c2_model=c2.MODEL,c2_revision=c2.REVISION,
        qualification_seconds={'t5':900,'c2':90},
        source_sha256={p:c2.digest(root/p) for p in SOURCES})


def tasks_for_case(case,family):
    methods=[m for m in case['methods'] if (m.startswith('c2_'))==(family=='c2')]
    if family not in ('t5','c2'):raise ValueError('Unknown family')
    tasks=[]
    for rep in range(3):
        order=methods[rep%len(methods):]+methods[:rep%len(methods)]
        for m in order:tasks.append(dict(method=m,phase='warmup',origin=case['origins'][0],repeat=rep))
    for i,origin in enumerate(case['origins']):
        for rep in range(3):
            k=(i+rep)%len(methods);order=methods[k:]+methods[:k]
            for m in order:tasks.append(dict(method=m,phase='measured',origin=origin,repeat=rep))
    return tasks


def timed_call(prepare,infer,postprocess,*,sync,clock=time.perf_counter):
    sync();start=clock()
    payload=prepare();sync();prepared=clock()
    output=infer(payload);sync();predicted=clock()
    result=postprocess(output);sync();finished=clock()
    t=dict(prepare_fit_ms=1000*(prepared-start),endpoint_ms=1000*(predicted-prepared),
        postprocess_ms=1000*(finished-predicted),total_ms=1000*(finished-start))
    if not all(np.isfinite(v) and v>=0 for v in t.values()):raise ValueError('Invalid timing')
    return result,t


def prepare_input(df,case,origin,method,*,device,fit_timing=None,sync=lambda:None,clock=time.perf_counter):
    if origin<512 or origin>len(df):raise ValueError('Incomplete past context')
    history=df.iloc[origin-512:origin]
    target=history[case['target']].to_numpy(dtype=np.float32).copy()
    if not np.isfinite(target).all():raise ValueError('Target context must be finite')
    if fit_timing is not None:fit_timing['adapter_fit_ms']=0.
    if method=='t5_bare':return torch.tensor(target[None,:],dtype=torch.float32)
    if method.startswith('c2_'):
        if method not in ('c2_target_only','c2_native'):raise ValueError('Unknown endpoint')
        payload={'target':target}
        if method=='c2_native':
            names=[x for x in case['features'] if x!=case['target']]
            cov=history[names].to_numpy(dtype=np.float32)
            if not np.isfinite(cov).all():raise ValueError('Covariates must be finite')
            payload['past_covariates']={str(n):cov[:,i].copy() for i,n in enumerate(names)}
        return [payload]
    if method!='cmf_selected':raise ValueError('Unknown endpoint')
    adapter=case['adapter']
    if adapter=='identity':
        # The selected-identity wrapper still has its own RNG/preparation cost.
        with c.isolated_rng(42):return torch.tensor(target[None,:],dtype=torch.float32)
    # Same float32 normalization, fit_adapter and apply_adapter as the frozen
    # _prepare_single_origin. Only past rows enter; no horizon labels are read.
    X=history[case['features']].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():raise ValueError('Features must be finite')
    X=(X-X.mean(axis=0,keepdims=True))/(X.std(axis=0,keepdims=True)+1e-8)
    mu=float(target.mean());sigma=float(target.std()+1e-8);z=(target-mu)/sigma
    kwargs=case['adapter_kwargs'].copy()
    kwargs['target_idx']=case['features'].index(case['target']) if case['target'] in case['features'] else 0
    if adapter=='mlp':kwargs['mlp_device']=device
    with c.isolated_rng(42):
        sync();start=clock()
        fitted=c.legacy.cmf.fit_adapter(X,z,kind=adapter,**kwargs)
        sync();fit_ms=1000*(clock()-start)
        adapted=c.legacy.cmf.apply_adapter(X[None,:,:],fitted)*sigma+mu
    if fit_timing is not None:fit_timing['adapter_fit_ms']=fit_ms
    result=torch.tensor(adapted,dtype=torch.float32)
    if not torch.isfinite(result).all():raise ValueError('Fused context must be finite')
    return result


def finish_output(output,horizon,method):
    if method.startswith('c2_'):
        quantiles,_=output
        if len(quantiles)!=1 or tuple(quantiles[0].shape)!=(1,horizon,9):raise ValueError('Invalid C2 shape')
        q=quantiles[0][0].detach().cpu().numpy().astype(np.float64)
        result={}
    else:
        if tuple(output.shape)!=(1,50,horizon):raise ValueError('Invalid T5 shape')
        samples=output[0].detach().cpu().numpy().astype(np.float64)
        q=np.quantile(samples,c2.QUANTILES,axis=0).T
        result={'samples':samples}
    if not np.isfinite(q).all():raise ValueError('Nonfinite forecast')
    return result|dict(q_preds=q,median_pred=q[:,c2.QUANTILES.index(.5)],quantile_levels=np.array(c2.QUANTILES))


def summarize(plan,rows):
    expected=[]
    for case in plan['cases']:
        for family in ('t5','c2'):
            expected.extend((case['dataset'],case['horizon'],t['method'],t['phase'],t['origin'],t['repeat'])
                            for t in tasks_for_case(case,family))
    def key(r):return (r['dataset'],int(r['horizon']),r['method'],r['phase'],int(r['origin']),int(r['repeat']))
    if Counter(map(key,rows))!=Counter(expected):raise ValueError('Complete balanced panel required')
    report=[]
    for case in plan['cases']:
        for method in case['methods']:
            chosen=[r for r in rows if r['dataset']==case['dataset'] and int(r['horizon'])==case['horizon'] and r['method']==method]
            measured=[r for r in chosen if r['phase']=='measured']
            result=dict(dataset=case['dataset'],horizon=case['horizon'],method=method,
                adapter=case['adapter'] if method=='cmf_selected' else None,
                n_origins=5,n_repeats=3,n_measured=15,n_warmup=3)
            for name in TIMES:
                values=np.array([float(r[name]) for r in measured])
                if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid measured times')
                q1,med,q3=np.quantile(values,[.25,.5,.75])
                result.update({name+'_median':float(med),name+'_q25':float(q1),name+'_q75':float(q3),
                               name+'_iqr':float(q3-q1)})
            report.append(result)
    return report



def audit_phase(run,plan,family,*,stage='full'):
    run=Path(run)
    with (run/'timings.jsonl').open() as f:rows=[json.loads(l) for l in f]
    cases={(v['dataset'],v['horizon']):v for v in plan['cases']}
    paths=set();same_seed={}
    for row in rows:
        case=cases[row['dataset'],int(row['horizon'])]
        if row['method'] not in case['methods'] or row['method'].startswith('c2_')!=(family=='c2'):
            raise ValueError('Wrong endpoint/family')
        seed=c.sampler_seed_for_origin(dataset=case['dataset'],target=case['target'],horizon=case['horizon'],
            origin=row['origin'],repetition=row['repeat'],base_seed=2026)
        if row['sampler_seed']!=seed:raise ValueError('Wrong sampler seed')
        values=[float(row[k]) for k in TIMES]
        if not all(np.isfinite(v) and v>=0 for v in values):raise ValueError('Nonfinite/negative timer')
        if not np.isclose(row['prepare_fit_ms']+row['endpoint_ms']+row['postprocess_ms'],row['total_ms'],rtol=1e-9,atol=1e-6):
            raise ValueError('Timer components do not sum to end-to-end')
        if row['adapter_fit_ms']>row['prepare_fit_ms']+1e-6:raise ValueError('Fit time exceeds preparation')
        name=row['prediction_file']
        if Path(name).parts[0]!='predictions' or len(Path(name).parts)!=2 or name in paths:
            raise ValueError('Invalid/duplicate forecast path')
        paths.add(name)
        if c2.digest(run/name)!=row['prediction_sha256']:raise ValueError('Forecast bytes changed')
        with np.load(run/name,allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files}
        h=case['horizon'];q=arrays['q_preds']
        if q.shape!=(h,9) or arrays['median_pred'].shape!=(h,) or not all(np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Invalid stored forecast')
        if not np.array_equal(arrays['quantile_levels'],c2.QUANTILES) or not np.array_equal(arrays['median_pred'],q[:,4]):
            raise ValueError('Stored quantile/point estimator changed')
        if family=='t5':
            if arrays['samples'].shape!=(50,h) or not np.array_equal(q,np.quantile(arrays['samples'],c2.QUANTILES,axis=0).T):
                raise ValueError('Stored T5 sample quantiles differ')
        replay_key=(row['dataset'],h,row['method'],row['origin'],row['repeat'])
        if replay_key in same_seed and not np.array_equal(same_seed[replay_key],q):
            raise ValueError('Same-seed warmup/repetition differs')
        same_seed[replay_key]=q
    if paths!={str(v.relative_to(run)) for v in (run/'predictions').glob('*')}:
        raise ValueError('Missing/orphan forecast')
    if stage=='full':
        expected=Counter((v['dataset'],v['horizon'],t['method'],t['phase'],t['origin'],t['repeat'])
            for v in plan['cases'] for t in tasks_for_case(v,family))
        actual=Counter((v['dataset'],int(v['horizon']),v['method'],v['phase'],v['origin'],v['repeat']) for v in rows)
        if actual!=expected:raise ValueError('Incomplete full phase')
    else:
        allowed={('opsd-wind',24),('solar-al',48),('ecl',24)} if family=='t5' else {('ecl',24),('ecl',48)}
        if {(v['dataset'],int(v['horizon'])) for v in rows}!=allowed:raise ValueError('Qualification representative cases missing')
        for v in rows:
            if v['phase']!='qualification' or v['origin']+v['horizon']>int(cases[v['dataset'],v['horizon']]['n_rows']*.6):
                raise ValueError('Qualification must use training contexts')
        for key in allowed:
            expected_methods={m for m in cases[key]['methods'] if m.startswith('c2_')==(family=='c2')}
            if {v['method'] for v in rows if (v['dataset'],v['horizon'])==key}!=expected_methods:
                raise ValueError('Qualification endpoint missing')
    return rows


def assess_hardware(manifest,guard,telemetry,plan):
    seconds=plan['qualification_seconds'][manifest['family']]
    minimum=40 if manifest['family']=='t5' else 12
    if not (manifest['status']=='qualification_window_complete' and manifest['stage']=='qualification' and
        manifest['calls']>=minimum and manifest['elapsed_prediction_window_s']>=seconds and
        guard['status']=='complete' and guard['returncode']==0 and guard['required_power_limit_W']==160 and
        guard['temperature_guard_C']==85 and guard['fan_rpm_logged']):
        raise ValueError('Guarded qualification has not passed')
    if not telemetry:raise ValueError('Qualification telemetry missing')
    for row in telemetry:
        vals={k:float(row[k]) for k in ['temperature','power_limit','power_draw','fan0_rpm','fan1_rpm']}
        if not all(np.isfinite(v) for v in vals.values()) or vals['temperature']>78 or abs(vals['power_limit']-160)>.5:
            raise ValueError('Qualification telemetry unsafe')
        if any(row[k]!='Not Active' for k in ['hw_thermal','sw_thermal','power_brake']):
            raise ValueError('Thermal or hardware power brake flag')
        if min(vals['fan0_rpm'],vals['fan1_rpm'])<0 or (vals['temperature']>=70 and min(vals['fan0_rpm'],vals['fan1_rpm'])<=0):
            raise ValueError('Qualification fan RPM unsafe')


def require_review(plan,receipt):
    r=json.loads(Path(receipt).read_text())
    if not (r.get('approved') is True and str(r.get('reviewer') or '').strip()
            and r.get('plan_sha256')==c2.digest(plan)):
        raise ValueError('A named review approving this exact plan is required before GPU launch')


def require_guard(path):
    deadline=time.monotonic()+5
    while True:
        g=json.loads(Path(path).read_text())
        if g.get('status')=='running' and g.get('pid')==os.getpid():break
        if time.monotonic()>=deadline:raise ValueError('Must be the active guarded child')
        time.sleep(.1)
    if not (g['required_power_limit_W']==160 and g['temperature_guard_C']==85 and g['fan_rpm_logged']):
        raise ValueError('Required power/temperature/RPM guard differs')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0]);sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('plan');q.add_argument('--preflight',required=True);q.add_argument('--selection',required=True)
    q.add_argument('--output',required=True)
    r=sub.add_parser('run');r.add_argument('--plan',required=True);r.add_argument('--review-receipt',required=True)
    r.add_argument('--family',choices=['t5','c2'],required=True);r.add_argument('--stage',choices=['full','qualification'],required=True)
    r.add_argument('--guard-status',required=True);r.add_argument('--output-dir',required=True)
    r.add_argument('--qualification-dir')
    a=sub.add_parser('assess');a.add_argument('--plan',required=True);a.add_argument('--run-dir',required=True)
    a.add_argument('--guard-dir',required=True);a.add_argument('--kernel-log',required=True);a.add_argument('--output',required=True)
    s=sub.add_parser('summarize');s.add_argument('--plan',required=True);s.add_argument('--runs',nargs=2,required=True)
    s.add_argument('--output-dir',required=True);s.add_argument('--guard-dirs',nargs=2,required=True)
    args=p.parse_args(argv)
    if args.action=='plan':
        pf=json.loads(Path(args.preflight).read_text())
        with Path(args.selection).open() as f:selection_rows=list(csv.DictReader(f))
        if len(selection_rows)!=16:raise ValueError('Exactly16 validation choices required')
        selection={(r['dataset'],int(r['H'])):r['validation_winner'] for r in selection_rows}
        plan=make_plan(pf,selection,include_c2_target=True)
        root=Path(__file__).resolve().parents[1]
        a4=root/'results/peer_review_controls/20261114_a4_scale_paced/solar-al/manifest.json'
        runtime=root/'results/gpu_diagnostics/20261115_chronos2_control/CHRONOS2_RUNTIME.json'
        plan.update(preflight_sha256=c2.digest(args.preflight),selection_sha256=c2.digest(args.selection),
            expected_packages={'t5':json.loads(a4.read_text())['environment']['packages'],
                               'c2':json.loads(runtime.read_text())['packages']},
            environment_evidence_sha256={'a4':c2.digest(a4),'c2':c2.digest(runtime)})
        with Path(args.output).open('x') as f:json.dump(plan,f,indent=2)
        print(json.dumps({k:plan[k] for k in ['expected_calls','measured_calls','warmup_calls']}));return
    plan=json.loads(Path(args.plan).read_text())
    if args.action=='assess':
        run=Path(args.run_dir);guard=Path(args.guard_dir)
        m=json.loads((run/'manifest.json').read_text());g=json.loads((guard/'status.json').read_text())
        with (guard/'telemetry.csv').open() as f:telemetry=list(csv.DictReader(f))
        if m['plan_sha256']!=c2.digest(args.plan):raise ValueError('Qualification plan differs')
        rows=audit_phase(run,plan,m['family'],stage='qualification')
        if len(rows)!=m['calls']:raise ValueError('Qualification record count differs')
        assess_hardware(m,g,telemetry,plan)
        times=[datetime.strptime(v['timestamp'],'%Y/%m/%d %H:%M:%S.%f') for v in telemetry]
        if len(times)<2 or max((b-a).total_seconds() for a,b in zip(times,times[1:]))>10 or (times[-1]-times[0]).total_seconds()<m['duration_s']-2:
            raise ValueError('Qualification telemetry duration/gaps invalid')
        if 'NVRM: Xid' in Path(args.kernel_log).read_text():raise ValueError('Kernel Xid present')
        report=dict(qualified=True,family=m['family'],plan_sha256=c2.digest(args.plan),
            records=len(rows),guard_dir=str(guard.resolve()),kernel_log=str(Path(args.kernel_log).resolve()),
            source_sha256={str(v.resolve()):c2.digest(v) for v in [run/'manifest.json',run/'timings.jsonl',
                guard/'status.json',guard/'telemetry.csv',Path(args.kernel_log)]})
        with Path(args.output).open('x') as f:json.dump(report,f,indent=2)
        print(json.dumps(report));return
    out=Path(args.output_dir)
    if out.exists():raise FileExistsError(out)
    if args.action=='summarize':
        rows=[]
        for run,guard in zip(args.runs,args.guard_dirs):
            g=json.loads((Path(guard)/'status.json').read_text())
            if not (g['status']=='complete' and g['returncode']==0 and g['required_power_limit_W']==160 and g['temperature_guard_C']==85 and g['fan_rpm_logged']):
                raise ValueError('Both full guards must complete without failure')
            path=Path(run);m=json.loads((path/'manifest.json').read_text())
            if m['status']!='complete' or m['stage']!='full' or m['plan_sha256']!=c2.digest(args.plan):
                raise ValueError('Both complete full phases on the same plan are required')
            rows.extend(audit_phase(path,plan,m['family']))
        report=summarize(plan,rows);out.mkdir(parents=True)
        with (out/'summary.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(report[0]));w.writeheader();w.writerows(report)
        c2.write_json(out/'analysis.json',dict(plan_sha256=c2.digest(args.plan),rows=len(rows),cells=len(report),
            primary='median/IQR active per-call wall time; warmup excluded; no latency significance tests',
            inputs={str(Path(v)/n):c2.digest(Path(v)/n) for v in args.runs for n in ['manifest.json','timings.jsonl']}))
        return
    require_review(args.plan,args.review_receipt);require_guard(args.guard_status)
    if os.environ.get('HF_HUB_OFFLINE')!='1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise ValueError('Offline cache and deterministic CUBLAS workspace are required')
    if plan['methods']!=['t5_bare','cmf_selected','c2_target_only','c2_native']:
        raise ValueError('Reviewed four-endpoint plan required')
    root=Path(__file__).resolve().parents[1]
    for name,expected in plan['source_sha256'].items():
        if c2.digest(root/name)!=expected:raise ValueError('Planned source changed: '+name)
    for key,value in [('context_length',512),('num_samples',50),('delay_before_prediction_s',3),('required_power_W',160),('guard_C',85)]:
        if plan[key]!=value:raise ValueError('Frozen setting changed: '+key)
    for name,version in plan['expected_packages'][args.family].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Reviewed software changed: '+name)
    if args.stage=='full':
        if not args.qualification_dir:raise ValueError('Full phase requires qualification evidence')
        qual=Path(args.qualification_dir);report=json.loads((qual/'assessment.json').read_text())
        if not (report['qualified'] and report['family']==args.family and report['plan_sha256']==c2.digest(args.plan)):
            raise ValueError('Qualification assessment does not match')
        for name,expected in report['source_sha256'].items():
            if c2.digest(name)!=expected:raise ValueError('Qualification evidence changed')
        audit_phase(qual,plan,args.family,stage='qualification')
    from src import cmf_core as cmf
    frames={}
    for case in plan['cases']:
        ds=case['dataset']
        if ds not in frames:
            df,cols,target,seasonality=cmf.load_dataset(ds)
            if c2.frame_hash(df,list(dict.fromkeys([*cols,target])))!=case['data_sha256']:
                raise ValueError('Frozen dataframe changed: '+ds)
            if cols!=case['features'] or str(target)!=case['target'] or seasonality!=case['seasonality']:
                raise ValueError('Frozen data configuration changed: '+ds)
            frames[ds]=df
    out.mkdir(parents=True);(out/'predictions').mkdir()
    state=dict(status='loading',stage=args.stage,family=args.family,created_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=c2.digest(args.plan),review_sha256=c2.digest(args.review_receipt),environment=c.environment_record())
    c2.write_json(out/'manifest.json',state)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    load_start=time.perf_counter()
    if args.family=='t5':
        pipeline,checkpoint=c._load_chronos(T5_MODEL,T5_REVISION,'cuda');pipeline.inner_model.eval()
        if checkpoint['resolved_commit']!=T5_REVISION:raise ValueError('Wrong T5 revision')
    else:
        from chronos import Chronos2Pipeline
        pipeline=Chronos2Pipeline.from_pretrained(c2.MODEL,revision=c2.REVISION,local_files_only=True,
            device_map='cuda',torch_dtype=torch.float32,attn_implementation='eager');pipeline.model.eval()
        checkpoint=dict(model_id=c2.MODEL,resolved_commit=getattr(pipeline.model.config,'_commit_hash',None))
        if checkpoint['resolved_commit']!=c2.REVISION:raise ValueError('Wrong C2 revision')
    model=pipeline.inner_model if args.family=='t5' else pipeline.model
    if model.training or any(v.dtype!=torch.float32 or v.device.type!='cuda' for v in model.parameters()):
        raise ValueError('Model must be CUDA float32 eval')
    if args.family=='c2' and not set(c2.QUANTILES).issubset(pipeline.quantiles):raise ValueError('Quantiles are not native')
    import chronos
    package=Path(chronos.__file__).parent
    from huggingface_hub import snapshot_download
    cache=Path(snapshot_download(repo_id=T5_MODEL if args.family=='t5' else c2.MODEL,
        revision=T5_REVISION if args.family=='t5' else c2.REVISION,local_files_only=True))
    state.update(status='running',checkpoint=checkpoint,model_load_wall_s=time.perf_counter()-load_start,
        load_finished_utc=datetime.now(timezone.utc).isoformat(),
        chronos_source_sha256={str(v.relative_to(package)):c2.digest(v) for v in package.rglob('*.py')},
        model_file_sha256={v.name:c2.digest(v) for v in cache.iterdir() if v.is_file()},
        actual_attention=getattr(model.config,'_attn_implementation',None),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),tf32=False)

    c2.write_json(out/'manifest.json',state)
    from scripts.run_a4_paced import PacedEndpoint,WindowComplete
    class Task:
        def predict(self,task):
            # Host/device contention check stays outside the timed interval.
            pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
            if {int(v.strip()) for v in pids.splitlines() if v.strip()}!={os.getpid()}:
                raise RuntimeError('CUDA process contention; hold benchmark')
            return task()
    calls=0;paced=None
    try:
        with (out/'pacing.jsonl').open('x') as schedule,(out/'timings.jsonl').open('x') as timings:
            duration=plan['qualification_seconds'][args.family] if args.stage=='qualification' else None
            paced=PacedEndpoint(Task(),schedule,duration=duration)
            paced.sleep=lambda seconds:time.sleep(3. if seconds==1. else seconds)
            sync=lambda:torch.cuda.synchronize(0)
            while True:
                cases=plan['cases']
                if args.stage=='qualification':
                    chosen=[('opsd-wind',24),('solar-al',48),('ecl',24)] if args.family=='t5' else [('ecl',24),('ecl',48)]
                    cases=[v for v in cases if (v['dataset'],v['horizon']) in chosen]
                for case in cases:
                    tasks=tasks_for_case(case,args.family)
                    if args.stage=='qualification':
                        tasks=[dict(method=m,phase='qualification',origin=1024,repeat=calls)
                               for m in case['methods'] if m.startswith('c2_')==(args.family=='c2')]
                        if 1024+case['horizon']>int(case['n_rows']*.6):raise ValueError('Qualification is outside training')
                    for task in tasks:
                        method=task['method'];origin=task['origin'];h=case['horizon'];fit={}
                        seed=c.sampler_seed_for_origin(dataset=case['dataset'],target=case['target'],
                            horizon=h,origin=origin,repetition=task['repeat'],base_seed=2026)
                        def infer(payload):
                            with c.isolated_rng(seed),torch.inference_mode():
                                if method.startswith('c2_'):
                                    return pipeline.predict_quantiles(payload,prediction_length=h,
                                        quantile_levels=c2.QUANTILES,**c2.PREDICT_ARGS)
                                return pipeline.predict(context=payload,prediction_length=h,num_samples=50,**c.SAMPLING_ARGS)
                        result,t= paced.predict(task=lambda:timed_call(
                            lambda:prepare_input(frames[case['dataset']],case,origin,method,device='cuda',fit_timing=fit,sync=sync),
                            infer,lambda v:finish_output(v,h,method),sync=sync))
                        row=task|dict(dataset=case['dataset'],horizon=h,sampler_seed=seed,
                            adapter=case['adapter'] if method=='cmf_selected' else None,
                            adapter_fit_device=('cuda' if case['adapter']=='mlp' else 'cpu') if method=='cmf_selected' and case['adapter']!='identity' else 'none',**t,**fit,
                            utc=datetime.now(timezone.utc).isoformat())
                        name=f'{calls:05d}.npz'
                        with (out/'predictions'/name).open('xb') as f:np.savez_compressed(f,**result)
                        row.update(prediction_file='predictions/'+name,prediction_sha256=c2.digest(out/'predictions'/name))
                        timings.write(json.dumps(row)+'\n');timings.flush();calls+=1
                if args.stage=='full':break
        state['status']='complete'
    except WindowComplete:state['status']='qualification_window_complete'
    except BaseException as exc:
        state.update(status='failed',reason=repr(exc));raise
    finally:
        state.update(calls=calls,elapsed_prediction_window_s=(paced.clock()-paced.started) if paced else None,
                     duration_s=plan['qualification_seconds'][args.family] if args.stage=='qualification' else None,
                     finished_utc=datetime.now(timezone.utc).isoformat())
        c2.write_json(out/'manifest.json',state)
    print(json.dumps(state))


if __name__=='__main__':main()
