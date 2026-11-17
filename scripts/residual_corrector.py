"""G3 output-side residual corrector for frozen Chronos-T5 identity forecasts (D5).

Fits a ridge correction of the identity sample median on validation forecasts
only, then applies it to the G2 test identity forecasts. The covariate corrector
adds past-covariate features; the target-only corrector isolates their effect.
"""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path

import numpy as np

from experiments import run_chronos2_controls as c
from experiments import run_lgbm_baseline as b
from experiments import run_revision as legacy
from scripts.chronos2_control_analysis import analyze_cases, require
from scripts.plan_g2 import run_dir
from src import cmf_core as cmf

LAMBDAS = (0.1, 1, 10, 100, 1000)
FIT_FRACTION = 0.7


def load_identity(run, horizon):
    """origin -> prediction arrays of identity, repetition 0, not the replay."""
    with (run / 'losses.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['adapter'] == 'identity' and
                r['replay'] == '0' and r['repetition'] == '0' and int(r['horizon']) == horizon]
    out = {}
    for r in rows:
        with np.load(run / r['prediction_file']) as npz:
            out[int(r['origin_idx'])] = {k: npz[k] for k in
                                         ('median_pred', 'q_preds', 'quantile_levels',
                                          'y_true', 'insample')} | dict(MASE=float(r['MASE']))
    return out


def design(values, origins, preds, horizon, s, floor, covariates):
    """Rows (origin, h): features, normalized residual label, sigma and median."""
    X, y, sigma = [], [], []
    for o in origins:
        p = preds[o]
        ctx = values[o - b.CONTEXT:o] if covariates else values[o - b.CONTEXT:o, :1]
        require(np.array_equal(p['insample'], values[o - b.CONTEXT:o, 0]) and
                np.array_equal(p['y_true'], values[o:o + horizon, 0]), f'origin {o}: data differ')
        mu, sd = ctx.mean(axis=0), np.maximum(ctx.std(axis=0), floor[:ctx.shape[1]])
        z, recent = (ctx - mu) / sd, ((ctx - mu) / sd)[-s:]
        common = [z[-b.N_LAGS:, 0][::-1], [recent[:, 0].mean(), recent[:, 0].std()]]
        if covariates:
            common += [z[-1, 1:], recent[:, 1:].mean(axis=0)]
        common = np.concatenate(common)
        level = lambda q: p['q_preds'][:, int(np.flatnonzero(np.isclose(p['quantile_levels'], q))[0])]
        norm = lambda v: (v - mu[0]) / sd[0]
        for h in range(horizon):
            X.append(np.concatenate([[h + 1, norm(p['median_pred'][h]), norm(level(.1)[h]),
                                      norm(level(.9)[h])], common]))
        y.extend((p['y_true'] - p['median_pred']) / sd[0])
        sigma.append(sd[0])
    return np.array(X), np.array(y), np.array(sigma)


def ridge(X, y, lam):
    m, s = X.mean(axis=0), X.std(axis=0)
    s = np.where(s > 0, s, 1.)
    Z = (X - m) / s
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda A: ((A - m) / s) @ w + y.mean()


def corrected(preds, origins, X, sigma, model, horizon):
    shift = (model(X) * np.repeat(sigma, horizon)).reshape(len(origins), horizon)
    return {o: shift[i] for i, o in enumerate(origins)}


def mase(preds, shifts, s):
    return {o: cmf.compute_mase(p['y_true'], p['median_pred'] + shifts.get(o, 0.), p['insample'], s)
            for o, p in preds.items()}


def fit_corrector(values, val, horizon, s, floor, covariates):
    """Lambda from a 70/30 time split of validation origins, then refit on all."""
    origins = sorted(val)
    cut = int(len(origins) * FIT_FRACTION)
    fit_o, hold_o = origins[:cut], origins[cut:]
    Xf, yf, _ = design(values, fit_o, val, horizon, s, floor, covariates)
    Xh, _, sh = design(values, hold_o, val, horizon, s, floor, covariates)
    hold = {o: val[o] for o in hold_o}
    scores = {}
    for lam in sorted(LAMBDAS, reverse=True):          # ties go to the larger lambda
        shifts = corrected(hold, hold_o, Xh, sh, ridge(Xf, yf, lam), horizon)
        scores[lam] = float(np.mean(list(mase(hold, shifts, s).values())))
    chosen = min(scores, key=lambda lam: (scores[lam], -lam))
    Xa, ya, _ = design(values, origins, val, horizon, s, floor, covariates)
    return ridge(Xa, ya, chosen), dict(lambda_scores=scores, chosen_lambda=chosen,
                                       n_fit=len(fit_o), n_holdout=len(hold_o), n_rows=len(ya))


def run_case(dataset, horizon, val_run, test_run, frozen):
    with contextlib.redirect_stdout(io.StringIO()):
        df, cols, target, s = cmf.load_dataset(dataset)
    order = [target] + [col for col in cols if col != target]
    require(c.frame_hash(df, list(dict.fromkeys([*cols, target]))) == frozen['data_sha256'],
            f'{dataset}: frame differs from preflight')
    values = df[order].to_numpy(np.float64)
    floor = b.floors(values, len(values))
    val, test = load_identity(val_run, horizon), load_identity(test_run, horizon)
    require(sorted(test) == frozen['origins'][str(horizon)], f'{dataset} H{horizon}: test grid differs')
    _, val_end, _ = cmf.get_split_indices(len(values))
    require(max(val) + horizon <= val_end, f'{dataset} H{horizon}: validation labels reach test')
    test_o = sorted(test)
    out = dict(identity={o: test[o]['MASE'] for o in test_o}, selection={})
    recomputed = mase(test, {}, s)
    require(all(np.isclose(recomputed[o], out['identity'][o], rtol=1e-9) for o in test_o),
            f'{dataset} H{horizon}: identity MASE does not reproduce')
    levels = test[test_o[0]]['quantile_levels']
    for name, cov in (('covariates', True), ('target_only', False)):
        model, info = fit_corrector(values, val, horizon, s, floor, cov)
        X, _, sig = design(values, test_o, test, horizon, s, floor, cov)
        shifts = corrected(test, test_o, X, sig, model, horizon)
        out[name] = mase(test, shifts, s)
        out['wql_' + name] = float(np.mean([legacy.weighted_quantile_loss_full(
            test[o]['y_true'], test[o]['q_preds'] + shifts[o][:, None], levels) for o in test_o]))
        out['selection'][name] = info
    out['wql_identity'] = float(np.mean([legacy.weighted_quantile_loss_full(
        test[o]['y_true'], test[o]['q_preds'], levels) for o in test_o]))
    out['origins'] = test_o
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--val-dir', required=True, help='G3 identity validation runs, one folder per dataset')
    p.add_argument('--test-dir', required=True, help='G2 test runs, one folder per dataset and horizon (<dataset>_H<h>)')
    p.add_argument('--preflight', required=True)
    p.add_argument('--protocol', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args(argv)
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(out)
    preflight = json.loads(Path(args.preflight).read_text())
    primary, secondary, rows, selection = [], [], [], {}
    for d in c.DATASETS:
        runs = [(run_dir(args.val_dir, d, h), 'val') for h in (24, 48)]
        runs += [(Path(args.test_dir) / f'{d}_H{h}', 'test') for h in (24, 48)]
        for run, split in runs:
            m = json.loads((run / 'manifest.json').read_text())
            require(m['status'] == 'complete' and m['split']['name'] == split and
                    m['data_sha256'] == preflight['datasets'][d]['data_sha256'],
                    f'{run}: incomplete, wrong split or frame')
        for h in (24, 48):
            r = run_case(d, h, run_dir(args.val_dir, d, h), Path(args.test_dir) / f'{d}_H{h}',
                         preflight['datasets'][d])
            primary.append(dict(dataset=d, horizon=h, origins=r['origins'],
                                past_covariates=r['covariates'], target_only=r['identity']))
            secondary.append(dict(dataset=d, horizon=h, origins=r['origins'],
                                  past_covariates=r['covariates'], target_only=r['target_only']))
            selection[f'{d}_H{h}'] = r['selection']
            rows.append(dict(dataset=d, horizon=h, n=len(r['origins']),
                             **{f'MASE_{k}': float(np.mean(list(r[k].values())))
                                for k in ('identity', 'covariates', 'target_only')},
                             **{k: r[k] for k in ('wql_identity', 'wql_covariates', 'wql_target_only')}))
    results = dict(primary=analyze_cases(primary), secondary=analyze_cases(secondary), summaries=rows)
    for row in results['primary']:
        row['contrast'] = 'covariate corrector - identity'
    for row in results['secondary']:
        row['contrast'] = 'covariate corrector - target-only corrector'
    out.mkdir(parents=True)
    for name, data in results.items():
        with (out / f'{name}.csv').open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    c.write_json(out / 'selection.json', selection)
    c.write_json(out / 'analysis.json', dict(protocol_sha256=c.digest(args.protocol),
        preflight_sha256=c.digest(args.preflight), analysis_sha256=c.digest(__file__),
        point='identity sample median plus ridge shift', lambdas=LAMBDAS, fit_fraction=FIT_FRACTION))
    print('G3: 16 cases corrected; primary and secondary families written')


if __name__ == '__main__':
    main()
