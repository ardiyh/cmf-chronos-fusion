"""Audit and analyze G2: mechanism controls, CMF-Select, D4 contrast and noise floor."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from experiments import run_chronos2_controls as c
from scripts.chronos2_control_analysis import require
from scripts.peer_review_block_sensitivity import block_sums, sign_flip_p
from scripts.plan_g2 import DATASETS, FAMILIES, same_model
from scripts.sinergi_block_robust import _signed_ranks
from scripts.sinergi_stats import holm_adjust, moving_block_bootstrap_ci, wilcoxon_signed_rank_p

METRICS = ('MASE', 'MASE_mean', 'WQL', 'coverage_50', 'width_50', 'coverage_90', 'width_90')


def paired_family(cases):
    """analyze_cases without the fixed 16-case size: same tests, seeds and order."""
    rows = []
    for i, case in enumerate(cases):
        origins = sorted(case['origins'])
        a, b = case['method'], case['reference']
        require(set(origins) == set(a) == set(b), 'Paired origins differ')
        delta = np.array([a[o] - b[o] for o in origins])
        block = min(math.ceil(512 / min(y - x for x, y in zip(origins, origins[1:]))), len(origins))
        ranks = _signed_ranks(delta)
        worst = max(sign_flip_p(block_sums(ranks, block, offset),
                                rng=np.random.default_rng([16000, i, offset]))[0]
                    for offset in range(block))
        low, high = moving_block_bootstrap_ci(delta, block_length=block, n_resamples=10000,
                                              seed=17000 + i)
        mean_a, mean_b = float(np.mean(list(a.values()))), float(np.mean(list(b.values())))
        rows.append(dict(dataset=case['dataset'], horizon=case['horizon'], n=len(origins),
                         method=case['method_name'], reference=case['reference_name'],
                         mean_method=mean_a, mean_reference=mean_b, mean_diff=float(delta.mean()),
                         median_diff=float(np.median(delta)), mean_pct=100 * (mean_a / mean_b - 1),
                         win_rate=float(np.mean(delta < 0)), wilcoxon_p=wilcoxon_signed_rank_p(delta),
                         block_length=block, block_p_worst=worst,
                         ci_lower=low, ci_upper=high))
    for row, p, bp in zip(rows, holm_adjust([r['wilcoxon_p'] for r in rows]),
                          holm_adjust([r['block_p_worst'] for r in rows])):
        row.update(holm_p=p, holm_block_p=bp, family_size=len(rows))
    return rows


def load(run, preflight, horizons, adapters):
    m = json.loads((run / 'manifest.json').read_text())
    require(m['status'] == 'complete' and m['split']['name'] == 'test' and
            m['data_sha256'] == preflight['data_sha256'], f'{run}: incomplete, split or frame')
    with (run / 'losses.csv').open() as f:
        rows = list(csv.DictReader(f))
    out = {}
    for h in horizons:
        require(m['origins'][str(h)] == preflight['origins'][str(h)], f'{run}: H{h} grid differs')
        for a in adapters:
            chosen = [r for r in rows if int(r['horizon']) == h and r['adapter'] == a and
                      r['replay'] == '0' and r['mode'] == 'target_units']
            require([int(r['origin_idx']) for r in chosen] == m['origins'][str(h)],
                    f'{run}: H{h} {a} incomplete')
            out[h, a] = {int(r['origin_idx']): {k: float(r[k]) for k in METRICS} for r in chosen}
    replay = [float(r['replay_max_abs_diff']) for r in rows if r['replay'] == '1']
    return out, max(replay) if replay else None


def mase(values):
    return {o: v['MASE'] for o, v in values.items()}


def analyze(test_dir, seed_dir, plan, preflight):
    families = {name: [] for name in ['cmf_select', *FAMILIES, 'cmf_minus_ewma', 'noise_floor']}
    summaries, determinism = [], {}
    for d in DATASETS:
        noise, determinism[f'{d}_seed2027'] = load(seed_dir / d, preflight[d], (24, 48), ['identity'])
        for h in (24, 48):
            case = plan[f'{d}_H{h}']
            data, determinism[f'{d}_H{h}'] = load(test_dir / f'{d}_H{h}', preflight[d], (h,),
                                                  case['adapters'])
            ran = lambda spec: next(a for a in case['adapters'] if same_model(spec, a))
            identity = mase(data[h, 'identity'])
            base = dict(dataset=d, horizon=h, origins=preflight[d]['origins'][str(h)],
                        reference=identity, reference_name='identity')
            winner = case['table5_winner']
            cmf = identity if winner == 'identity' else mase(data[h, ran(winner)])
            if winner != 'identity':
                families['cmf_select'].append(base | dict(method=cmf, method_name=ran(winner)))
            for fam, spec in case['family_winners'].items():
                families[fam].append(base | dict(method=mase(data[h, ran(spec)]),
                                                 method_name=ran(spec)))
            ewma = ran(case['family_winners']['ewma'])
            families['cmf_minus_ewma'].append(base | dict(
                method=cmf, method_name=winner, reference=mase(data[h, ewma]), reference_name=ewma))
            families['noise_floor'].append(base | dict(
                method=mase(noise[h, 'identity']), method_name='identity seed 2027',
                reference_name='identity seed 2026'))
            for a in case['adapters']:
                summaries.append(dict(dataset=d, horizon=h, adapter=a,
                                      **{k: float(np.mean([v[k] for v in data[h, a].values()]))
                                         for k in METRICS}))
    rows = [dict(family=name, **r) for name, cases in families.items() for r in paired_family(cases)]
    return rows, summaries, determinism


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('--test-dir', '--seed-dir', '--plan', '--preflight', '--protocol', '--output-dir'):
        p.add_argument(name, required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads(Path(args.plan).read_text())
    preflight = json.loads(Path(args.preflight).read_text())['datasets']
    rows, summaries, determinism = analyze(Path(args.test_dir), Path(args.seed_dir), plan, preflight)
    require(all(v == 0 for v in determinism.values() if v is not None), 'Same-seed replay differs')
    out.mkdir(parents=True)
    for name, data in (('families', rows), ('summaries', summaries)):
        with (out / f'{name}.csv').open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    c.write_json(out / 'analysis.json', dict(
        point='sample median MASE', determinism_max_abs_diff=determinism,
        protocol_sha256=c.digest(args.protocol), plan_sha256=c.digest(args.plan),
        preflight_sha256=c.digest(args.preflight), analysis_sha256=c.digest(__file__)))
    print(f'G2: {len(rows)} family rows written; same-seed replays identical')


if __name__ == '__main__':
    main()
