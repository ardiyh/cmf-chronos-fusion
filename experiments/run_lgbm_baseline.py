"""LightGBM direct-quantile baseline, target-only vs past-covariates (reviewer D5).

Uses the frozen Chronos-2 control test grid, frames and metric code. Every
feature comes from the 512-step context before an origin; no future covariates
or calendar features. Each variant is tuned on validation origins only, then
refit on train+validation before the test origins are scored. The q0.5 column
is the point forecast; quantiles are scored raw, without sorting.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import csv
import io
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from experiments import run_chronos2_controls as c
from src import cmf_core as cmf

CONTEXT = 512
N_LAGS = 24
N_PHASE = 3
MAX_TRAIN_ORIGINS = 3000
SIGMA_FLOOR = 0.01            # fraction of the training-split std
GRID = [dict(num_leaves=15, min_child_samples=200), dict(num_leaves=15, min_child_samples=20),
        dict(num_leaves=63, min_child_samples=200), dict(num_leaves=63, min_child_samples=20)]
BASE_PARAMS = dict(objective='quantile', learning_rate=0.05, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255,
                   deterministic=True, force_row_wise=True, num_threads=8, seed=2026,
                   verbose=-1)
MAX_ROUNDS = 2000
PATIENCE = 100
FIELDS = ('dataset','target','horizon','origin_idx','variant','replay',*c.METRICS,
          'quantile_crossings')


def phase_indices(origins, h, s):
    """Most recent observations at the same seasonal phase as step origin+h-1."""
    m = math.ceil(h / s)
    return [np.asarray(origins) + h - 1 - s * (m + k) for k in range(N_PHASE)]


def split_origins(n, horizon):
    train_end, val_end, _ = cmf.get_split_indices(n)
    fit = lambda end: cmf.rolling_window_origins(end, 0, CONTEXT, horizon, stride=1,
                                                 max_origins=MAX_TRAIN_ORIGINS)
    return dict(train=fit(train_end), refit=fit(val_end),
                validation=cmf.rolling_window_origins(val_end, train_end, CONTEXT, horizon,
                                                      stride=horizon))


def floors(values, n):
    """Per-column std floor from the training split only."""
    std = values[:cmf.get_split_indices(n)[0]].std(axis=0)
    return np.where(std > 0, SIGMA_FLOOR * std, 1.0)


def features(values, origins, horizon, s, floor):
    """Rows (origin, h) of instance-normalized context features and labels.

    values[:, 0] is the target; any further columns are past covariates. Only
    values[origin-CONTEXT:origin] enter a row's features.
    """
    origins = np.asarray(origins)
    windows = sliding_window_view(values, CONTEXT, axis=0)                 # [N, C, CONTEXT] view
    stats = []
    for chunk in np.array_split(origins, max(1, len(origins) // 128)):    # bounded memory
        ctx = windows[chunk - CONTEXT]
        recent = ctx[:, :, -s:]
        stats.append(np.stack([ctx.mean(axis=2), ctx.std(axis=2), recent.mean(axis=2),
                               recent[:, 0:1].std(axis=2).repeat(values.shape[1], 1)]))
    mu, raw_sd, recent_mean, recent_sd = np.concatenate(stats, axis=1)     # each [n, C]
    sd = np.maximum(raw_sd, floor)
    z = lambda idx: (values[idx] - mu) / sd                                # idx [n] -> [n, C]
    common = [np.stack([z(origins - k)[:, 0] for k in range(1, N_LAGS + 1)], axis=1),
              ((recent_mean[:, 0] - mu[:, 0]) / sd[:, 0])[:, None],
              (recent_sd[:, 0] / sd[:, 0])[:, None]]
    if values.shape[1] > 1:
        common += [z(origins - 1)[:, 1:], (recent_mean[:, 1:] - mu[:, 1:]) / sd[:, 1:]]
    common = np.concatenate(common, axis=1)
    rows, labels = [], []
    for h in range(1, horizon + 1):
        phase = [z(idx) for idx in phase_indices(origins, h, s)]
        block = [np.full((len(origins), 1), h), common] + [p[:, :1] for p in phase]
        if values.shape[1] > 1:
            block.append(phase[0][:, 1:])
        rows.append(np.concatenate(block, axis=1))
        labels.append(z(origins + h - 1)[:, 0])
    X = np.stack(rows, axis=1).reshape(-1, rows[0].shape[1]).astype(np.float32)
    y = np.stack(labels, axis=1).reshape(-1)
    return X, y, mu[:, 0], sd[:, 0]


def pinball(alpha):
    def metric(pred, data):
        e = data.get_label() - pred
        return f'pinball_{alpha}', float(np.maximum(alpha * e, (alpha - 1) * e).mean()), False
    return metric


def fit(lgb, X, y, params, alpha, *, rounds=MAX_ROUNDS, valid=None):
    p = BASE_PARAMS | params | dict(alpha=alpha)
    kw = {}
    if valid is not None:
        kw = dict(valid_sets=[lgb.Dataset(valid[0], valid[1])], feval=pinball(alpha),
                  callbacks=[lgb.early_stopping(PATIENCE, first_metric_only=True, verbose=False)])
        p['metric'] = 'None'
    return lgb.train(p, lgb.Dataset(X, y), num_boost_round=rounds, **kw)


def predict(models, X, mu, sd, horizon):
    """Back-transform per-level predictions to target units: [n, H, 9]."""
    q = np.stack([m.predict(X, num_iteration=m.best_iteration or None) for m in models], axis=1)
    return mu[:, None, None] + sd[:, None, None] * q.reshape(len(mu), horizon, len(models))


def score_all(q, values, origins, horizon, s):
    out = []
    for i, o in enumerate(origins):
        y = values[o:o + horizon, 0]
        out.append(c.score(q[i], y, values[o - CONTEXT:o, 0], s) |
                   dict(quantile_crossings=int(np.sum(np.diff(q[i], axis=1) < 0))))
    return out


def run_variant(lgb, values, test_origins, horizon, s, floor, log=print):
    """Validation-only selection and early stopping, then a train+validation refit."""
    sets = split_origins(len(values), horizon)
    data = {k: features(values, v, horizon, s, floor) for k, v in sets.items()}
    data['test'] = features(values, test_origins, horizon, s, floor)
    Xt, yt = data['train'][:2]
    Xv, yv, muv, sdv = data['validation']
    median = c.QUANTILES.index(.5)
    scores = []
    for params in GRID:
        model = fit(lgb, Xt, yt, params, .5, valid=(Xv, yv))
        qv = predict([model], Xv, muv, sdv, horizon).repeat(len(c.QUANTILES), axis=2)
        mase = float(np.mean([r['MASE'] for r in score_all(qv, values, sets['validation'],
                                                            horizon, s)]))
        scores.append(dict(params=params, best_iteration=model.best_iteration,
                           validation_MASE=mase))
        log(f'  grid {params} iter={model.best_iteration} val MASE={mase:.6f}')
    chosen = min(range(len(GRID)), key=lambda i: (scores[i]['validation_MASE'], i))
    params = GRID[chosen]
    iterations = [scores[chosen]['best_iteration'] if i == median else
                  fit(lgb, Xt, yt, params, a, valid=(Xv, yv)).best_iteration
                  for i, a in enumerate(c.QUANTILES)]
    Xr, yr = data['refit'][:2]
    models = [fit(lgb, Xr, yr, params, a, rounds=max(it, 1))
              for a, it in zip(c.QUANTILES, iterations)]
    Xs, _, mus, sds = data['test']
    q = predict(models, Xs, mus, sds, horizon)
    selection = dict(grid_scores=scores, chosen=params, chosen_index=chosen,
                     iterations_per_quantile=dict(zip(map(str, c.QUANTILES), iterations)),
                     n_rows=dict(train=len(yt), validation=len(yv), refit=len(yr), test=len(Xs)),
                     n_features=int(Xt.shape[1]),
                     origin_ranges={k: [int(min(v)), int(max(v)), len(v)] for k, v in sets.items()})
    return q, selection


def environment():
    import lightgbm, numpy, pandas, scipy
    git = lambda *a: subprocess.run(['git', *a], cwd=Path(__file__).parents[1],
                                    capture_output=True, text=True).stdout.strip()
    return dict(python=sys.version.split()[0], platform=platform.platform(),
                machine=platform.machine(),
                packages=dict(lightgbm=lightgbm.__version__, numpy=numpy.__version__,
                              pandas=pandas.__version__, scipy=scipy.__version__),
                git_head=git('rev-parse', 'HEAD'), git_dirty=bool(git('status', '--porcelain')),
                source_sha256={p: c.digest(Path(__file__).parents[1] / p) for p in
                               ['experiments/run_lgbm_baseline.py', 'src/cmf_core.py',
                                'experiments/run_chronos2_controls.py']})


def run_dataset(dataset, preflight, out, protocol_sha256, replay=False):
    import lightgbm as lgb
    with contextlib.redirect_stdout(io.StringIO()):
        df, cols, target, s = cmf.load_dataset(dataset)
    frozen = preflight['datasets'][dataset]
    columns = list(dict.fromkeys([*cols, target]))
    origins = {h: cmf.rolling_window_origins(len(df), int(len(df) * .8), CONTEXT, h, stride=h,
                                             max_origins=150) for h in (24, 48)}
    if (c.frame_hash(df, columns) != frozen['data_sha256'] or
            c.origin_hashes({str(h): v for h, v in origins.items()}) != frozen['origin_sha256'] or
            s != frozen['seasonality'] or str(target) != frozen['target']):
        raise ValueError(f'{dataset}: frame, origins or case differ from the preflight')
    out.mkdir(parents=True, exist_ok=False)
    order = [target] + [col for col in cols if col != target]
    full = df[order].to_numpy(dtype=np.float64)
    if not np.isfinite(full).all():
        raise ValueError(f'{dataset}: non-finite values')
    manifest = dict(schema='cmf-lgbm-baseline-v1', status='running', dataset=dataset,
                    target=str(target), seasonality=s, n_rows=len(df),
                    data_sha256=frozen['data_sha256'],
                    origins={str(h): v for h, v in origins.items()},
                    origin_sha256=frozen['origin_sha256'], protocol_sha256=protocol_sha256,
                    replay=replay, quantile_levels=c.QUANTILES, grid=GRID,
                    base_params=BASE_PARAMS, max_rounds=MAX_ROUNDS, patience=PATIENCE,
                    context_length=CONTEXT, n_lags=N_LAGS, n_phase=N_PHASE,
                    max_train_origins=MAX_TRAIN_ORIGINS, sigma_floor=SIGMA_FLOOR,
                    environment=environment(), created_utc=datetime.now(timezone.utc).isoformat(),
                    selection={}, seconds={})
    c.write_json(out / 'manifest.json', manifest)
    log = lambda m: print(f'[{datetime.now():%H:%M:%S}] {dataset} {m}', flush=True)
    rows = []
    for horizon in (24, 48):
        for variant in c.VARIANTS:
            values = full if variant == 'past_covariates' else full[:, :1]
            start = time.time()
            log(f'H{horizon} {variant}: {values.shape[1] - 1} covariates')
            q, selection = run_variant(lgb, values, origins[horizon], horizon, s,
                                       floors(values, len(values)), log)
            key = f'H{horizon}_{variant}'
            manifest['selection'][key] = selection
            manifest['seconds'][key] = round(time.time() - start, 1)
            np.savez_compressed(out / f'{key}.npz', q_preds=q, origins=np.array(origins[horizon]),
                                quantile_levels=np.array(c.QUANTILES),
                                y_true=np.stack([full[o:o + horizon, 0] for o in origins[horizon]]))
            for o, metrics in zip(origins[horizon],
                                  score_all(q, full[:, :1], origins[horizon], horizon, s)):
                rows.append(dict(dataset=dataset, target=str(target), horizon=horizon,
                                 origin_idx=o, variant=variant, replay=int(replay), **metrics))
            c.write_json(out / 'manifest.json', manifest)
            log(f'H{horizon} {variant}: done in {manifest["seconds"][key]} s')
    with (out / 'losses.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                    losses_sha256=c.digest(out / 'losses.csv'))
    c.write_json(out / 'manifest.json', manifest)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--preflight', required=True)
    p.add_argument('--protocol', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--datasets', nargs='+', default=list(c.DATASETS))
    p.add_argument('--replay', action='store_true')
    args = p.parse_args(argv)
    preflight = json.loads(Path(args.preflight).read_text())
    protocol = c.digest(args.protocol)
    for d in args.datasets:
        run_dataset(d, preflight, Path(args.output_dir) / d, protocol, args.replay)


if __name__ == '__main__':
    main()
