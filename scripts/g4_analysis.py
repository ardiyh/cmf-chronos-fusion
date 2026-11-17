"""Audit and analyze G4 (D6): all eligible origins, extra targets, day-ahead horizons."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments import run_chronos2_controls as c
from scripts.chronos2_control_analysis import require
from scripts.g2_analysis import METRICS, mase, paired_family
from scripts.plan_g2 import TABLE5_WINNERS, same_model
from src import cmf_core as cmf

TARGETS = {'etth1': ['HULL', 'MUFL', 'MULL', 'LUFL'], 'etth2': ['HUFL', 'HULL', 'MULL', 'LULL'],
           'ettm1': ['HUFL', 'HULL', 'MUFL', 'MULL'], 'ettm2': ['MUFL', 'MULL', 'LUFL', 'LULL'],
           'ecl': ['MT_100', 'MT_185', 'MT_194', 'MT_207'], 'opsd-wind': ['FR', 'NL', 'CH', 'DK'],
           'sdwpf': ['turbine_16', 'turbine_39', 'turbine_43', 'turbine_98'],
           'solar-al': ['solar_2', 'solar_32', 'solar_45', 'solar_127']}
DAY_AHEAD = (('ettm2', 96), ('sdwpf', 144), ('solar-al', 144))
FROM_G2 = ('etth1', 'etth2')                     # all eligible equals the capped grid


def expected_origins(n, h, cap):
    return cmf.rolling_window_origins(n, int(n * .8), 512, h, stride=h, max_origins=cap)


def load(run, frame_sha256, target, horizons, cap, adapters):
    """{(h, adapter): {origin: metrics}} after checking frame, target and origin rule."""
    m = json.loads((run / 'manifest.json').read_text())
    require(m['status'] == 'complete' and m['split']['name'] == 'test' and
            m['data_sha256'] == frame_sha256 and m['target'] == target,
            f'{run}: incomplete, split, frame or target differs')
    with (run / 'losses.csv').open() as f:
        rows = list(csv.DictReader(f))
    out = {}
    for h in horizons:
        origins = expected_origins(m['n_rows'], h, cap)
        require(m['origins'][str(h)] == origins, f'{run}: H{h} origin rule differs')
        for a in adapters:
            chosen = [r for r in rows if int(r['horizon']) == h and r['adapter'] == a and
                      r['replay'] == '0' and r['mode'] == 'target_units']
            require([int(r['origin_idx']) for r in chosen] == origins, f'{run}: H{h} {a} incomplete')
            out[h, a] = {int(r['origin_idx']): {k: float(r[k]) for k in METRICS} for r in chosen}
    replay = [float(r['replay_max_abs_diff']) for r in rows if r['replay'] == '1']
    require(all(v == 0 for v in replay), f'{run}: same-seed replay differs')
    return out


def case(d, h, data, winner, origins, target):
    return dict(dataset=d, horizon=h, origins=origins, method=mase(data[h, winner]),
                method_name=winner, reference=mase(data[h, 'identity']),
                reference_name='identity', target=target)


def analyze(g4_dir, g2_dir, plan, preflight):
    families = {'all_eligible': [], 'extra_targets': [], 'day_ahead': []}
    summaries = []

    def keep(name, d, h, target, data, adapters):
        for a in adapters:
            summaries.append(dict(family=name, dataset=d, target=target, horizon=h, adapter=a,
                                  n=len(data[h, a]), **{k: float(np.mean([v[k] for v in
                                                                          data[h, a].values()]))
                                                        for k in METRICS}))

    for d, frozen in preflight.items():
        primary = frozen['target']
        for h in (24, 48):
            winner = TABLE5_WINNERS[d, h]
            if winner == 'identity':
                continue
            if d in FROM_G2:
                ran = next(a for a in plan[f'{d}_H{h}']['adapters'] if same_model(winner, a))
                data = load(g2_dir / f'{d}_H{h}', frozen['data_sha256'], primary, (h,), None,
                            ['identity', ran])
                winner = ran
            else:
                data = load(g4_dir / 'g4a' / f'{d}_H{h}', frozen['data_sha256'], primary, (h,), None,
                            ['identity', winner])
            families['all_eligible'].append(case(d, h, data, winner, sorted(data[h, 'identity']), primary))
            keep('all_eligible', d, h, primary, data, ['identity', winner])
        for t in TARGETS[d]:
            winner = TABLE5_WINNERS[d, 24]
            data = load(g4_dir / 'g4b' / f'{d}_{t}', frozen['data_sha256'], t, (24,), 150,
                        ['identity', winner])
            families['extra_targets'].append(case(d, 24, data, winner, sorted(data[24, 'identity']), t))
            keep('extra_targets', d, 24, t, data, ['identity', winner])
    for d, h in DAY_AHEAD:
        frozen, winner = preflight[d], TABLE5_WINNERS[d, 48]
        data = load(g4_dir / 'g4c' / f'{d}_H{h}', frozen['data_sha256'], frozen['target'], (h,), 150,
                    ['identity', winner])
        families['day_ahead'].append(case(d, h, data, winner, sorted(data[h, 'identity']),
                                          frozen['target']))
        keep('day_ahead', d, h, frozen['target'], data, ['identity', winner])
    rows = []
    for name, cases in families.items():
        for r, cs in zip(paired_family(cases), cases):
            rows.append(dict(family=name, target=cs['target'], **r))
    return rows, summaries


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('--g4-dir', '--g2-dir', '--plan', '--preflight', '--protocol', '--output-dir'):
        p.add_argument(name, required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads(Path(args.plan).read_text())
    preflight = json.loads(Path(args.preflight).read_text())['datasets']
    rows, summaries = analyze(Path(args.g4_dir), Path(args.g2_dir), plan, preflight)
    out.mkdir(parents=True)
    for name, data in (('families', rows), ('summaries', summaries)):
        with (out / f'{name}.csv').open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    c.write_json(out / 'analysis.json', dict(
        point='sample median MASE', families={k: sum(r['family'] == k for r in rows)
                                              for k in ('all_eligible', 'extra_targets', 'day_ahead')},
        protocol_sha256=c.digest(args.protocol), plan_sha256=c.digest(args.plan),
        preflight_sha256=c.digest(args.preflight), analysis_sha256=c.digest(__file__)))
    print(f'G4: {len(rows)} contrasts in three families; replays identical')


if __name__ == '__main__':
    main()
