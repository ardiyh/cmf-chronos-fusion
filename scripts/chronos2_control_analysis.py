"""Audit native quantile evidence before the fixed 16-case Chronos-2 family."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from experiments import run_chronos2_controls as c
from scripts.peer_review_block_sensitivity import block_sums,sign_flip_p
from scripts.sinergi_block_robust import _signed_ranks
from scripts.sinergi_stats import holm_adjust,moving_block_bootstrap_ci,wilcoxon_signed_rank_p


def require(value,message):
    if not value:raise ValueError(message)


def close(a,b,message):
    require(np.allclose(a,b,rtol=1e-9,atol=1e-9),message)


def audit_run(run,*,case=None,require_complete=True,df=None):
    run=Path(run);m=json.loads((run/'manifest.json').read_text())
    complete=m['status']=='complete'
    require(not require_complete or complete,'Run must be complete')
    require(m['schema']=='cmf-chronos2-controls-v1','Unexpected manifest schema')
    require(m['model_revision']==c.REVISION and m['model_id']==c.MODEL and m['dtype']=='float32',
            'Checkpoint/dtype differs')
    require(m['quantile_levels']==c.QUANTILES and m['predict_args']==c.PREDICT_ARGS,'Quantile/API grid differs')
    require(m['variants']==list(c.VARIANTS) and m['repetitions']==1,'Variant/repetition grid differs')
    require(m['base_seed']==2026 and m['context_length']==512 and m['attention']=='eager'
            and m['replays_per_variant']==1,'Frozen runtime/grid settings differ')
    if 'origin_sha256' in m:
        require(m['origin_sha256']==c.origin_hashes(m['origins']),'Origin hashes differ')
    require(m['environment']['source_sha256']['experiments/run_chronos2_controls.py']==c.digest(c.__file__),
            'Runner source differs from saved source')
    if case is not None:
        for k,v in case.items():require(m[k]==v,'Frozen case differs: '+k)
        require(not m['alternate_replay_seed'],'Full test uses same-seed replays')
    if df is not None:
        require(c.frame_hash(df,list(dict.fromkeys([*m['features'],m['target']])))==m['data_sha256'],
                'Source dataframe differs')
    with (run/'losses.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    keys=[(h,o,v,p) for h,o in c.cell_grid(m['origins']) for p in (0,1) for v in c.VARIANTS]
    require(len(rows)<=len(keys),'Too many records')
    if complete:require(len(rows)==len(keys)==m['n_records'],'Incomplete record grid')
    paths={str(p.relative_to(run)) for p in (run/'predictions').glob('*')}
    require(paths=={r['prediction_file'] for r in rows} and len(paths)==len(rows),'Missing/duplicate/orphan NPZ')
    first={};pairs=0;crossings=0
    for row,(h,o,v,p) in zip(rows,keys):
        require((int(row['horizon']),int(row['origin_idx']),row['variant'],int(row['replay']))==(h,o,v,p),
                'Unexpected key/prefix order')
        seed=(c.seed_for(m['dataset'],m['target'],h,o)+int(p and m['alternate_replay_seed']))%(2**32)
        require(int(row['sampler_seed'])==seed,'Seed mismatch')
        require(row['dataset']==m['dataset'] and row['target']==m['target'],'Dataset/target mismatch')
        name=f'predictions/H{h}_{v}_p{p}_o{o}.npz'
        require(row['prediction_file']==name,'Unexpected NPZ path')
        with np.load(run/name,allow_pickle=False) as z:
            arrays={k:z[k] for k in c.ARRAYS};meta=json.loads(z['metadata_json'].item())
        names=[x for x in m['features'] if x!=m['target']] if v=='past_covariates' else []
        for k,value in dict(dataset=m['dataset'],target=m['target'],horizon=h,origin_idx=o,
            variant=v,replay=p,sampler_seed=seed,covariate_names=names,
            point_estimator='quantile_0.5',context_length=512,seasonality=m['seasonality']).items():
            require(meta[k]==value,'NPZ metadata mismatch: '+k)
        shapes=dict(target_context_raw=(512,),target_input=(512,),past_covariates=(len(names),512),
                    y_true=(h,),q_preds=(h,9),quantile_levels=(9,),median_pred=(h,))
        for k,values in arrays.items():
            require(values.shape==shapes[k] and np.isfinite(values).all(),'NPZ shape/nonfinite: '+k)
        close(arrays['quantile_levels'],c.QUANTILES,'Quantile levels differ')
        require(np.array_equal(arrays['target_input'],arrays['target_context_raw'].astype(np.float32)),
                'Target input differs from raw context')
        close(arrays['median_pred'],arrays['q_preds'][:,c.QUANTILES.index(.5)],'Point estimator differs')
        count=int(np.sum(np.diff(arrays['q_preds'],axis=1)<0));crossings+=count
        require(meta['quantile_crossings']==count,'Quantile crossing count differs')
        expected=c.score(arrays['q_preds'],arrays['y_true'],arrays['target_context_raw'],m['seasonality'])
        for k,value in expected.items():
            close(float(row[k]),value,'CSV metric differs: '+k)
            close(meta[k],value,'NPZ metric differs: '+k)
        if df is not None:
            require(np.array_equal(arrays['target_context_raw'],df[m['target']].to_numpy()[o-512:o]),'Source context differs')
            require(np.array_equal(arrays['y_true'],df[m['target']].to_numpy()[o:o+h]),'Source horizon differs')
            cov=df[names].to_numpy(dtype=np.float32)[o-512:o].T
            require(np.array_equal(arrays['past_covariates'],cov),'Source past covariates differ')
        key=(h,o,v)
        if not p:first[key]=arrays['q_preds']
        else:
            diff=float(np.abs(first[key]-arrays['q_preds']).max())
            require(diff==0 and float(row['replay_max_abs_diff'])==0 and meta['replay_max_abs_diff']==0,
                    'Quantile replay differs')
            pairs+=1
    return dict(complete=complete,records=len(rows),expected_records=len(keys),replay_pairs=pairs,
        quantile_crossings=crossings,max_replay_abs_diff=0.,artifact_sha256={
        str(p.relative_to(run)):c.digest(p) for p in sorted(run.rglob('*')) if p.is_file()})


def analyze_cases(cases):
    require(len(cases)==16 and {(r['dataset'],r['horizon']) for r in cases}==
            {(d,h) for d in c.DATASETS for h in (24,48)},'The full 16-case family is required')
    rows=[]
    for i,case in enumerate(cases):
        origins=sorted(case['origins']);a=case['past_covariates'];b=case['target_only']
        require(set(origins)==set(a)==set(b),'Paired origins differ')
        delta=np.array([a[o]-b[o] for o in origins])
        require(np.isfinite(delta).all(),'Loss differences must be finite')
        block=min(math.ceil(512/min(y-x for x,y in zip(origins,origins[1:]))),len(origins))
        ranks=_signed_ranks(delta)
        worst=max(sign_flip_p(block_sums(ranks,block,offset),
            rng=np.random.default_rng([16000,i,offset]))[0] for offset in range(block))
        low,high=moving_block_bootstrap_ci(delta,block_length=block,n_resamples=10000,seed=17000+i)
        mean_a=float(np.mean(list(a.values())));mean_b=float(np.mean(list(b.values())))
        k=math.ceil(len(origins)/block)
        rows.append(dict(dataset=case['dataset'],horizon=case['horizon'],metric='MASE',
            contrast='past_covariates - target_only',n=len(origins),mean_covariates=mean_a,
            mean_target_only=mean_b,mean_diff=float(delta.mean()),median_diff=float(np.median(delta)),
            mean_pct=100*(mean_a/mean_b-1) if mean_b else None,win_rate=float(np.mean(delta<0)),
            wilcoxon_p=wilcoxon_signed_rank_p(delta),block_length=block,block_p_worst=worst,
            n_blocks_offset0=k,min_attainable_p_offset0=2/2**k,ci_lower=low,ci_upper=high))
    for row,p,bp in zip(rows,holm_adjust([r['wilcoxon_p'] for r in rows]),
                            holm_adjust([r['block_p_worst'] for r in rows])):
        row.update(holm_p=p,holm_block_p=bp)
    return rows


def analyze_batch(batch,preflight):
    batch=Path(batch);reports={};cases=[];summaries=[]
    # Finish every artifact/grid audit before any family-level inference.
    for d in c.DATASETS:reports[d]=audit_run(batch/d,case=preflight['datasets'][d])
    for d in c.DATASETS:
        m=json.loads((batch/d/'manifest.json').read_text())
        with (batch/d/'losses.csv').open() as f:rows=[r for r in csv.DictReader(f) if r['replay']=='0']
        for h in (24,48):
            case=dict(dataset=d,horizon=h,origins=m['origins'][str(h)])
            for v in c.VARIANTS:
                chosen=[r for r in rows if r['variant']==v and int(r['horizon'])==h]
                case[v]={int(r['origin_idx']):float(r['MASE']) for r in chosen}
                summaries.append(dict(dataset=d,horizon=h,variant=v,n=len(chosen),
                    **{k:float(np.mean([float(r[k]) for r in chosen])) for k in c.METRICS}))
            cases.append(case)
    return dict(audits=reports,comparisons=analyze_cases(cases),summaries=summaries)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--batch-dir',required=True);p.add_argument('--preflight',required=True)
    p.add_argument('--output-dir',required=True)
    args=p.parse_args(argv);out=Path(args.output_dir)
    if out.exists():raise FileExistsError(out)
    preflight=json.loads(Path(args.preflight).read_text())
    result=analyze_batch(args.batch_dir,preflight)
    out.mkdir(parents=True,exist_ok=False)
    for name in ['comparisons','summaries']:
        with (out/(name+'.csv')).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(result[name][0]));w.writeheader();w.writerows(result[name])
    c.write_json(out/'audit.json',result['audits'])
    c.write_json(out/'analysis.json',dict(family_size=16,primary_point='quantile_0.5',
        direction='past_covariates - target_only; negative favors covariates',
        preflight_sha256=c.digest(args.preflight),analysis_sha256=c.digest(__file__),
        inputs={str(Path(d)/f):c.digest(Path(args.batch_dir)/d/f)
                for d in c.DATASETS for f in ['manifest.json','losses.csv']}))
    print('16 paired primary contrasts written; all eight runs audited')


if __name__=='__main__':main()
