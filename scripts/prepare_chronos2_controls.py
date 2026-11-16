"""CPU-only export of the fixed all-origin retained-rule grid and data hashes."""
import argparse
import contextlib
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

from experiments import run_chronos2_controls as c
from src import cmf_core as cmf


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--retained-dir',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    result=dict(schema='cmf-chronos2-preflight-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        context_length=512,max_origins=150,no_future_mask=True,model_revision=c.REVISION,datasets={})
    n=0
    for d in c.DATASETS:
        with contextlib.redirect_stdout(sys.stderr):df,cols,target,s=cmf.load_dataset(d)
        origins={str(h):cmf.rolling_window_origins(len(df),int(len(df)*.8),512,h,
                    stride=h,max_origins=150) for h in (24,48)}
        historical={}
        for h,values in origins.items():
            f=Path(args.retained_dir)/'per_origin'/f'{d}_identity_H{h}_seed42.csv'
            with f.open() as handle:old=[int(r['origin_idx']) for r in csv.DictReader(handle)]
            historical[h]=dict(path=str(f),sha256=c.digest(f),n_origins=len(old),
                               matches_no_mask_grid=old==values)
        result['datasets'][d]=dict(dataset=d,target=str(target),features=list(cols),
            seasonality=s,n_rows=len(df),data_sha256=c.frame_hash(df,list(dict.fromkeys([*cols,target]))),
            origins=origins,origin_sha256=c.origin_hashes(origins),split='test')
        result.setdefault('historical_grid_audit',{})[d]=historical
        n+=sum(map(len,origins.values()))
    result.update(n_origin_horizon_pairs=n,planned_records=n*4,
                  loader_sha256=c.digest(Path(cmf.__file__)))
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(n_origin_horizon_pairs=n,planned_records=n*4,
                         output=str(out),sha256=c.digest(out)),indent=2))


if __name__=='__main__':main()
