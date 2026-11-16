"""Compare rebuilt CMF datasets with the experiment copies, value by value.

For each dataset: column names, bitwise value equality, largest absolute
difference, the timestamp encoding ratio (rebuilt / experiment) and whether
the unchanged CMF loader reproduces the preflight frame hash from the rebuilt
files. The loader reads paths from CMF_DATA_ROOT at import, so each root is
hashed in its own subprocess.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd

H5 = {'ecl': 'ECL/ecl.h5', 'opsd-wind': 'OPSD/opsd_wind.h5', 'sdwpf': 'SDWPF/sdwpf.h5',
      'solar-al': 'SolarAl/solar_al.h5'}
CSV = {d: f'ETT/{d}.csv' for d in ('etth1', 'etth2', 'ettm1', 'ettm2')}


def read(root, dataset):
    if dataset in H5:
        with h5py.File(Path(root) / H5[dataset], 'r') as f:
            g = f['data']
            return ([c.decode() for c in g['axis0'][:]], g['axis1'][:].astype('int64'),
                    g['block0_values'][:])
    df = pd.read_csv(Path(root) / CSV[dataset])
    return list(df.columns[1:]), None, df.iloc[:, 1:].to_numpy(np.float64)


def frame_hashes(cmf_code, root):
    code = ('import contextlib,io,json,sys;sys.path.insert(0,sys.argv[1]);'
            'from src import cmf_core as cmf;from experiments import run_chronos2_controls as c;'
            'out={}\n'
            'for d in sys.argv[2:]:\n'
            ' with contextlib.redirect_stdout(io.StringIO()):df,cols,t,s=cmf.load_dataset(d)\n'
            ' out[d]=c.frame_hash(df,list(dict.fromkeys([*cols,t])))\n'
            'print(json.dumps(out))')
    result = subprocess.run([sys.executable, '-c', code, cmf_code, *H5, *CSV], check=True,
                            capture_output=True, text=True,
                            env=os.environ | {'CMF_DATA_ROOT': str(root)})
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('--cmf-code', '--rebuilt', '--experiment', '--preflight', '--output'):
        p.add_argument(name, required=True)
    args = p.parse_args()
    preflight = json.loads(Path(args.preflight).read_text())['datasets']
    rebuilt = frame_hashes(args.cmf_code, args.rebuilt)
    report = {}
    for d in [*CSV, *H5]:
        (ca, ta, va), (cb, tb, vb) = read(args.rebuilt, d), read(args.experiment, d)
        same_shape = va.shape == vb.shape
        report[d] = dict(
            columns_equal=ca == cb, shape=list(va.shape), values_bitwise_equal=bool(
                same_shape and np.array_equal(va, vb)),
            max_abs_diff=float(np.max(np.abs(va - vb))) if same_shape else None,
            timestamp_ratio=(sorted({float(x) for x in ta[:100] / tb[:100]})
                             if ta is not None else None),
            rebuilt_frame_matches_preflight=rebuilt[d] == preflight[d]['data_sha256'],
            platform=sys.platform, pandas=pd.__version__)
        print(d, report[d])
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
