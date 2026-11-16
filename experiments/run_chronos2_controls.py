"""Chronos-2 target-only/past-covariate controls, with paired quantile replay.

No synthetic sample paths; q=.5 is the point forecast. CLI requires a frozen
data/origin preflight and always creates fresh evidence directories.
"""
import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import inspect

import numpy as np
import torch

from experiments import run_peer_review_controls as retained

REVISION = '29ec3766d36d6f73f0696f85560a422f50e8498c'
MODEL = 'amazon/chronos-2'
QUANTILES = [.05,.1,.2,.25,.5,.75,.8,.9,.95]
WQL_LEVELS = [.05,.1,.2,.5,.8,.9,.95]
VARIANTS = ('target_only','past_covariates')
DATASETS = ('etth1','etth2','ettm1','ettm2','ecl','opsd-wind','sdwpf','solar-al')
PREDICT_ARGS = dict(context_length=512, batch_size=512, cross_learning=False,
                    limit_prediction_length=True)
ARRAYS = ('target_context_raw','target_input','past_covariates','y_true',
          'q_preds','quantile_levels','median_pred')
METRICS = ('MASE','MAE','WQL','target_abs_mean','coverage_50','width_50',
           'coverage_90','width_90')
FIELDS = ('dataset','target','horizon','origin_idx','variant','replay','sampler_seed',
          *METRICS,'replay_max_abs_diff','prediction_file')


def frame_hash(df, columns):
    return retained.frame_sha256(df,columns)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def seed_for(dataset,target,horizon,origin):
    return retained.sampler_seed_for_origin(dataset=dataset,target=target,horizon=horizon,
        origin=origin,repetition=0,base_seed=2026)


def cell_grid(origins):
    """Interleave horizons to make bounded diagnostics representative of both."""
    return [(int(h),int(values[i])) for i in range(max(map(len,origins.values())))
            for h,values in origins.items() if i<len(values)]


def origin_hashes(origins):
    return {str(h):hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()
            for h,values in origins.items()}


def score(quantiles,y,context,seasonality):
    median=quantiles[:,QUANTILES.index(.5)]
    denom=max(float(np.abs(context[seasonality:]-context[:-seasonality]).mean()),1e-8)
    out=dict(MASE=float(np.abs(y-median).mean()/denom),MAE=float(np.abs(y-median).mean()),
             target_abs_mean=float(np.abs(y).mean()))
    loss=0.
    for q in WQL_LEVELS:
        error=y-quantiles[:,QUANTILES.index(q)]
        loss+=2*np.maximum(q*error,(q-1)*error).mean()
    out['WQL']=float(loss/len(WQL_LEVELS))
    for label,lo,hi in [('50',.25,.75),('90',.05,.95)]:
        lower,upper=(quantiles[:,QUANTILES.index(q)] for q in (lo,hi))
        out['coverage_'+label]=float(((lower<=y)&(y<=upper)).mean())
        out['width_'+label]=float((upper-lower).mean())
    return out


def evaluate_origin(*,df,origin,horizon,feature_cols,target_col,seasonality,variant,
                    pipeline,seed,context_length=512):
    if variant not in VARIANTS or not 0<seasonality<context_length:
        raise ValueError('Invalid variant or seasonality')
    if origin<context_length or origin+horizon>len(df):
        raise ValueError('Incomplete context or horizon')
    raw=df[target_col].to_numpy()[origin-context_length:origin].copy()
    target=raw.astype(np.float32)
    names=[c for c in feature_cols if c!=target_col] if variant=='past_covariates' else []
    cov=df[names].to_numpy(dtype=np.float32)[origin-context_length:origin].T.copy()
    truth=df[target_col].to_numpy()[origin:origin+horizon].copy()
    if not all(np.isfinite(a).all() for a in [raw,target,cov,truth]):
        raise ValueError('Context and target must be finite')
    payload={'target':target}
    if names:payload['past_covariates']={str(c):cov[i] for i,c in enumerate(names)}
    with retained.isolated_rng(seed),torch.inference_mode():
        quantiles,_=pipeline.predict_quantiles([payload],prediction_length=horizon,
            quantile_levels=QUANTILES,**(PREDICT_ARGS|dict(context_length=context_length)))
    if len(quantiles)!=1 or tuple(quantiles[0].shape)!=(1,horizon,len(QUANTILES)):
        raise ValueError('Endpoint quantiles must have shape [1,H,9]')
    q=quantiles[0][0].detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(q).all():raise ValueError('Quantiles must be finite')
    return dict(target=str(target_col),origin_idx=int(origin),horizon=horizon,variant=variant,
        sampler_seed=int(seed),context_length=context_length,seasonality=seasonality,
        covariate_names=[str(c) for c in names],point_estimator='quantile_0.5',
        predict_args=PREDICT_ARGS|dict(context_length=context_length),
        quantile_crossings=int(np.sum(np.diff(q,axis=1)<0)),
        target_context_raw=raw,target_input=target,past_covariates=cov,y_true=truth,
        q_preds=q,quantile_levels=np.array(QUANTILES),median_pred=q[:,QUANTILES.index(.5)],
        **score(q,truth,raw,seasonality))


def write_prediction(path,result):
    meta={k:v for k,v in result.items() if k not in ARRAYS}
    with Path(path).open('xb') as f:
        np.savez_compressed(f,**{k:result[k] for k in ARRAYS},metadata_json=json.dumps(meta))


def write_json(path,value):
    temp=Path(path).with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def run_grid(*,df,case,pipeline,output_dir,provenance,device='cuda',
             alternate_replay_seed=False):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    (out/'predictions').mkdir()
    env=retained.environment_record()
    source=Path(__file__).resolve()
    env['source_sha256']['experiments/run_chronos2_controls.py']=digest(source)
    m=dict(schema='cmf-chronos2-controls-v1',status='running',created_utc=datetime.now(timezone.utc).isoformat(),
        **case,context_length=512,variants=list(VARIANTS),replays_per_variant=1,
        alternate_replay_seed=alternate_replay_seed,base_seed=2026,repetitions=1,
        model_id=MODEL,model_revision=REVISION,dtype='float32',attention='eager',
        quantile_levels=QUANTILES,predict_args=PREDICT_ARGS,device=device,
        environment=env,provenance=provenance)
    write_json(out/'manifest.json',m)
    n=0;max_diff=0.
    with (out/'losses.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
        for horizon,origin in cell_grid(case['origins']):
            first={}
            for replay in (0,1):
                for variant in VARIANTS:
                    seed=(seed_for(case['dataset'],case['target'],horizon,origin)+
                          int(replay and alternate_replay_seed))%(2**32)
                    r=evaluate_origin(df=df,origin=origin,horizon=horizon,feature_cols=case['features'],
                        target_col=case['target'],seasonality=case['seasonality'],variant=variant,
                        pipeline=pipeline,seed=seed)
                    diff=None
                    if not replay:first[variant]=r['q_preds']
                    else:
                        diff=float(np.abs(first[variant]-r['q_preds']).max())
                        if diff!=0:raise RuntimeError('Quantile replay is not exactly deterministic')
                        max_diff=max(max_diff,diff)
                    name=f'H{horizon}_{variant}_p{replay}_o{origin}.npz'
                    r.update(dataset=case['dataset'],replay=replay,replay_max_abs_diff=diff)
                    write_prediction(out/'predictions'/name,r)
                    row={k:r.get(k) for k in FIELDS};row['prediction_file']='predictions/'+name
                    writer.writerow(row);f.flush();n+=1
    m.update(status='complete',n_records=n,max_replay_abs_diff=max_diff,
             finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/'manifest.json',m)
    return m


class QuantileCall:
    def __init__(self,pipeline):self.pipeline=pipeline
    def predict(self,**kwargs):return self.pipeline.predict_quantiles(**kwargs)


class PacedQuantiles:
    def __init__(self,pipeline,log,duration=None):
        from scripts.run_a4_paced import PacedEndpoint
        self.paced=PacedEndpoint(QuantileCall(pipeline),log,duration=duration)
    def predict_quantiles(self,inputs,**kwargs):return self.paced.predict(inputs=inputs,**kwargs)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset',required=True,choices=DATASETS)
    p.add_argument('--preflight',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--pacing-dir',required=True)
    p.add_argument('--stage',choices=['full','pilot','qualification'],default='full')
    args=p.parse_args(argv)
    if Path(args.output_dir).exists():p.error('Output directory already exists')
    preflight=json.loads(Path(args.preflight).read_text())
    case=preflight['datasets'][args.dataset].copy()
    from src import cmf_core as cmf
    df,cols,target,seasonality=cmf.load_dataset(args.dataset)
    if frame_hash(df,list(dict.fromkeys([*cols,target])))!=case['data_sha256']:
        raise ValueError('Frozen input dataframe changed')
    if cols!=case['features'] or str(target)!=case['target'] or seasonality!=case['seasonality']:
        raise ValueError('Frozen dataset configuration changed')
    duration={'pilot':90,'qualification':900,'full':None}[args.stage]
    if args.stage!='full':
        # Train contexts only; interleave H24/H48 in the same widest group.
        end=int(len(df)*.6)
        case['origins']={str(h):list(range(1024,end-h+1,h)) for h in [24,48]}
        case['origin_sha256']=origin_hashes(case['origins'])
        case['split']='train_diagnostic'
    out=Path(args.pacing_dir);out.mkdir(parents=True,exist_ok=False)
    state=dict(status='starting',stage=args.stage,duration_s=duration,
        preflight_sha256=digest(args.preflight),created_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/'status.json',state)
    endpoint=None
    from scripts.run_a4_paced import WindowComplete
    try:
        from chronos import Chronos2Pipeline
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        pipeline=Chronos2Pipeline.from_pretrained(MODEL,revision=REVISION,
            local_files_only=True,device_map='cuda',torch_dtype=torch.float32,
            attn_implementation='eager')
        pipeline.model.eval()
        if not set(QUANTILES).issubset(pipeline.quantiles):
            raise ValueError('Required quantiles are not native to the pinned checkpoint')
        if pipeline.model.training or any(x.dtype!=torch.float32 or x.device.type!='cuda'
                                          for x in pipeline.model.parameters()):
            raise ValueError('Model must be eval/float32')
        model_revision=getattr(pipeline.model.config,'_commit_hash',None)
        if model_revision!=REVISION:raise ValueError('Resolved model commit differs')
        api_file=Path(inspect.getsourcefile(type(pipeline).predict_quantiles))
        provenance=dict(checkpoint_revision=REVISION,resolved_revision=model_revision,
            api_file=str(api_file),api_sha256=digest(api_file),
            preflight_sha256=digest(args.preflight),stage=args.stage,
            required_quantiles_native=True,mean_label_not_used=True,
            deterministic_algorithms=True,tf32=False)
        with (out/'predictions.jsonl').open('x') as log:
            endpoint=PacedQuantiles(pipeline,log,duration)
            state['status']='running';write_json(out/'status.json',state)
            run_grid(df=df,case=case,pipeline=endpoint,output_dir=args.output_dir,
                provenance=provenance,alternate_replay_seed=args.stage!='full')
        state['status']='grid_complete'
    except WindowComplete:
        state['status']='window_complete'
    except BaseException as exc:
        state.update(status='failed',reason=repr(exc));raise
    finally:
        state.update(calls=endpoint.paced.calls if endpoint else 0,
            elapsed_prediction_window_s=(endpoint.paced.clock()-endpoint.paced.started) if endpoint else None,
            finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(out/'status.json',state)
    print(json.dumps(state,indent=2))


if __name__=='__main__':main()
