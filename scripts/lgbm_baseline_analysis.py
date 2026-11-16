"""Audit and analyze the D5 LightGBM baseline: primary and secondary 16-case families."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments import run_chronos2_controls as c
from experiments import run_lgbm_baseline as b
from scripts.chronos2_control_analysis import analyze_cases, require

REPLAY_DATASETS = ('etth1', 'opsd-wind')
KEYS = [f'H{h}_{v}' for h in (24, 48) for v in c.VARIANTS]


def read_losses(path, replay='0'):
    with Path(path).open() as f:
        return [r for r in csv.DictReader(f) if r['replay'] == replay]


def audit_run(run, frozen, protocol_sha256, replay=False):
    m = json.loads((run / 'manifest.json').read_text())
    require(m['schema'] == 'cmf-lgbm-baseline-v1' and m['status'] == 'complete',
            f'{run}: run must be complete')
    require(m['replay'] == replay and m['protocol_sha256'] == protocol_sha256,
            f'{run}: replay flag or protocol differs')
    require(m['data_sha256'] == frozen['data_sha256'] and
            m['origin_sha256'] == frozen['origin_sha256'] and
            m['origin_sha256'] == c.origin_hashes(m['origins']), f'{run}: frame/origins differ')
    require(not m['environment']['git_dirty'], f'{run}: produced from a dirty checkout')
    require(m['losses_sha256'] == c.digest(run / 'losses.csv'), f'{run}: losses hash differs')
    require(m['grid'] == b.GRID and m['base_params'] == b.BASE_PARAMS and
            m['quantile_levels'] == c.QUANTILES, f'{run}: model settings differ')
    rows = read_losses(run / 'losses.csv', str(int(replay)))
    for h in (24, 48):
        for v in c.VARIANTS:
            got = [int(r['origin_idx']) for r in rows if int(r['horizon']) == h and r['variant'] == v]
            require(got == m['origins'][str(h)], f'{run}: H{h} {v} origins differ')
            npz = np.load(run / f'H{h}_{v}.npz')
            require(npz['q_preds'].shape == (len(got), h, len(c.QUANTILES)) and
                    np.isfinite(npz['q_preds']).all(), f'{run}: H{h} {v} predictions invalid')
    return dict(git_head=m['environment']['git_head'], machine=m['environment']['machine'],
                packages=m['environment']['packages'], seconds=m['seconds'],
                chosen={k: m['selection'][k]['chosen'] for k in KEYS})


def replay_identical(primary, replay):
    return all(np.array_equal(np.load(primary / f'{k}.npz')[a], np.load(replay / f'{k}.npz')[a])
               for k in KEYS for a in ('q_preds', 'origins', 'y_true'))


def mase_by_origin(rows, h, variant):
    return {int(r['origin_idx']): float(r['MASE']) for r in rows
            if int(r['horizon']) == h and r['variant'] == variant}


def analyze(primary, replay, c2_source, preflight, protocol_sha256):
    audits, cases, secondary, summaries = {}, [], [], []
    for d in c.DATASETS:
        audits[d] = audit_run(primary / d, preflight['datasets'][d], protocol_sha256)
    for d in REPLAY_DATASETS:
        audit_run(replay / d, preflight['datasets'][d], protocol_sha256, replay=True)
        require(replay_identical(primary / d, replay / d), f'{d}: replay is not bitwise identical')
    for d in c.DATASETS:
        rows = read_losses(primary / d / 'losses.csv')
        c2 = read_losses(c2_source / d / 'losses.csv')
        for h in (24, 48):
            origins = preflight['datasets'][d]['origins'][str(h)]
            lgbm = {v: mase_by_origin(rows, h, v) for v in c.VARIANTS}
            cases.append(dict(dataset=d, horizon=h, origins=origins, **lgbm))
            secondary.append(dict(dataset=d, horizon=h, origins=origins,
                                  past_covariates=lgbm['past_covariates'],
                                  target_only=mase_by_origin(c2, h, 'past_covariates')))
            for v in c.VARIANTS:
                chosen = [r for r in rows if int(r['horizon']) == h and r['variant'] == v]
                summaries.append(dict(dataset=d, horizon=h, variant=v, n=len(chosen),
                    **{k: float(np.mean([float(r[k]) for r in chosen])) for k in c.METRICS},
                    crossing_forecasts=sum(int(r['quantile_crossings']) > 0 for r in chosen),
                    crossing_steps_total=sum(int(r['quantile_crossings']) for r in chosen)))
    second = analyze_cases(secondary)
    for r in second:
        r['contrast'] = 'lgbm_past_covariates - chronos2_past_covariates'
        r['mean_lgbm_past_covariates'] = r.pop('mean_covariates')
        r['mean_chronos2_past_covariates'] = r.pop('mean_target_only')
    return dict(audits=audits, primary=analyze_cases(cases), secondary=second,
                summaries=summaries)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--run-dir', required=True, help='folder with primary/, replay/, protocol')
    p.add_argument('--c2-source', required=True, help='audited Chronos-2 control source folder')
    p.add_argument('--output-dir', required=True)
    args = p.parse_args(argv)
    run, c2_source, out = Path(args.run_dir), Path(args.c2_source), Path(args.output_dir)
    if out.exists():
        raise FileExistsError(out)
    preflight_path = run / 'CHRONOS2_PREFLIGHT.json'
    require(c.digest(preflight_path) == c.digest(c2_source / 'CHRONOS2_PREFLIGHT.json'),
            'Preflight differs from the Chronos-2 control')
    protocol = c.digest(run / 'D5_LGBM_PROTOCOL_20261116.md')
    result = analyze(run / 'primary', run / 'replay', c2_source,
                     json.loads(preflight_path.read_text()), protocol)
    out.mkdir(parents=True)
    for name in ('primary', 'secondary', 'summaries'):
        with (out / f'{name}.csv').open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(result[name][0]))
            w.writeheader()
            w.writerows(result[name])
    c.write_json(out / 'audit.json', result['audits'])
    c.write_json(out / 'analysis.json', dict(
        family_size=16, point='quantile_0.5', protocol_sha256=protocol,
        primary='lgbm past_covariates - lgbm target_only; negative favors covariates',
        secondary='lgbm past_covariates - chronos2 past_covariates; separate Holm16 family',
        replay_bitwise_identical=list(REPLAY_DATASETS), analysis_sha256=c.digest(__file__),
        preflight_sha256=c.digest(preflight_path),
        inputs={f'{d}/{f}': c.digest(run / 'primary' / d / f)
                for d in c.DATASETS for f in ('manifest.json', 'losses.csv')} |
               {f'c2/{d}/losses.csv': c.digest(c2_source / d / 'losses.csv') for d in c.DATASETS}))
    print('D5: 8 runs audited, 2 replays identical, primary and secondary families written')


if __name__ == '__main__':
    main()
