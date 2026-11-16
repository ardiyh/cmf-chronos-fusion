"""A4 scale-consistent ablation; fixed 8-dataset, 16-comparison families.

Protocol: CMF-paper/submission-sinergi/revisions/peer-review-r1/A4_SCALE_PROTOCOL.md.
Requires complete paired origin/repetition grids. Never analyzes a partial batch.
MASE is median-point; MASE_mean is a separate sensitivity family. WQL is the
runner's unnormalized doubled pinball loss, in target units.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from itertools import product
from pathlib import Path

import numpy as np

from scripts.peer_review_control_analysis import _compare
from scripts.sinergi_stats import holm_adjust

DATASETS = ('etth1', 'etth2', 'ettm1', 'ettm2', 'ecl', 'opsd-wind', 'sdwpf', 'solar-al')
MODES = ('target_units', 'target_standardized', 'fused_standardized')
METRICS = ('MASE', 'MASE_mean', 'WQL', 'coverage_50', 'coverage_90', 'width_50', 'width_90')


def load_complete(run, dataset):
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError(f'{dataset}: run must be complete')
    if (manifest['dataset'] != dataset or set(manifest['origins']) != {'24'}
            or manifest['split']['name'] != 'test' or manifest['repetitions'] != 3
            or set(manifest['adapters']) != {'identity', 'ridge'}
            or set(manifest['modes']) != set(MODES)):
        raise ValueError(f'{dataset}: manifest does not match the A4 grid')
    origins = manifest['origins']['24']
    records, seeds = {}, {}
    with (run / 'losses.csv').open(newline='') as handle:
        for r in csv.DictReader(handle):
            if r['replay'] != '0':
                continue
            if r['dataset'] != dataset or int(r['horizon']) != 24:
                raise ValueError(f'{dataset}: unexpected dataset/horizon in grid')
            origin, rep = int(r['origin_idx']), int(r['repetition'])
            key = (r['adapter'], r['mode'], origin, rep)
            if key in records:
                raise ValueError(f'{dataset}: duplicate record {key}')
            seed = int(r['sampler_seed'])
            if seeds.setdefault((origin, rep), seed) != seed:
                raise ValueError(f'{dataset}: unpaired sampler seed at {origin}/{rep}')
            values = {m: float(r[m]) for m in METRICS}
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError(f'{dataset}: metrics must be finite')
            records[key] = values
    expected = set(product(('identity', 'ridge'), MODES, origins, range(3)))
    if set(records) != expected:
        raise ValueError(f'{dataset}: incomplete or unexpected origin/repetition grid')
    if len(set(seeds.values())) != len(seeds):
        raise ValueError(f'{dataset}: sampler seed collision')
    return manifest, {
        (adapter, mode): {
            m: {o: float(np.mean([records[adapter, mode, o, rep][m] for rep in range(3)]))
                for o in origins} for m in METRICS}
        for adapter, mode in product(('identity', 'ridge'), MODES)}


def analyze_batch(batch_dir):
    # Validate every dataset before computing any family-level inference.
    runs = {d: load_complete(Path(batch_dir) / d, d) for d in DATASETS}
    summaries, comparisons, descriptive = [], [], []
    for dataset, (manifest, values) in runs.items():
        for (adapter, mode), metrics in values.items():
            summaries.append(dict(dataset=dataset, H=24, adapter=adapter, mode=mode,
                                  n=len(manifest['origins']['24']), repetitions=3,
                                  **{m: float(np.mean(list(v.values()))) for m, v in metrics.items()}))
    for metric in ('MASE', 'MASE_mean'):
        family = []
        for dataset, (manifest, values) in runs.items():
            origins = sorted(manifest['origins']['24'])
            ridge = {mode: values['ridge', mode][metric] for mode in MODES}
            for mode in MODES[1:]:
                row = dict(dataset=dataset, H=24, metric=metric, adapter='ridge',
                           mode_a='target_units', mode_b=mode,
                           **_compare(ridge, 'target_units', mode, origins=origins,
                                      context_length=manifest['context_length'],
                                      seed=7000 + len(comparisons) + len(family)))
                blocks = math.ceil(row['n'] / row['block_length'])
                row.update(n_blocks_offset0=blocks, min_attainable_p_offset0=2 / 2 ** blocks,
                           informative_at_first_holm_threshold=(blocks >= 10))
                family.append(row)

            def describe(label, diffs):
                descriptive.append(dict(dataset=dataset, H=24, metric=metric, contrast=label,
                                        n=len(diffs), mean_diff=float(np.mean(diffs)),
                                        median_diff=float(np.median(diffs)),
                                        win_rate=float(np.mean(np.array(diffs) < 0))))

            for mode in MODES:
                describe(f'ridge - identity / {mode}',
                         [values['ridge', mode][metric][o] - values['identity', mode][metric][o]
                          for o in origins])
            for mode in MODES[1:]:
                identity_diff = [values['identity', 'target_units'][metric][o]
                                 - values['identity', mode][metric][o] for o in origins]
                describe(f'identity / target_units - {mode}', identity_diff)
                describe(f'(ridge transform effect) - (identity transform effect) / {mode}',
                         [ridge['target_units'][o] - ridge[mode][o] - d
                          for o, d in zip(origins, identity_diff)])
        for r, p, block_p in zip(family, holm_adjust([r['wilcoxon_p'] for r in family]),
                                holm_adjust([r['block_p_worst'] for r in family])):
            r.update(holm_p=p, holm_block_p=block_p)
        comparisons.extend(family)
    return dict(summaries=summaries, comparisons=comparisons, descriptive=descriptive)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--batch-dir', required=True)
    parser.add_argument('--output', required=True, help='new analysis directory')
    args = parser.parse_args(argv)
    out, batch = Path(args.output), Path(args.batch_dir)
    if out.exists():
        raise FileExistsError(out)
    result = analyze_batch(batch)
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in result.items():
        with (out / f'{name}.csv').open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    scripts = Path(_compare.__code__.co_filename).parent
    provenance = dict(
        batch_dir=str(batch.resolve()), family_size_per_metric=16,
        point_primary='sample_median', point_sensitivity='sample_mean',
        WQL='unnormalized doubled pinball loss; target units; equation (6)',
        inputs={str(Path(d) / f): digest(batch / d / f)
                for d in DATASETS for f in ('manifest.json', 'losses.csv')},
        sources={f: digest(scripts / f) for f in (
            'peer_review_control_analysis.py',
            'peer_review_block_sensitivity.py', 'sinergi_stats.py', 'sinergi_block_robust.py')})
    provenance['sources']['peer_review_scale_analysis.py'] = digest(Path(__file__))
    protocol = batch / 'A4_SCALE_PROTOCOL.md'
    provenance['protocol_sha256'] = digest(protocol) if protocol.exists() else None
    (out / 'analysis.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Wrote 16 primary and 16 sensitivity comparisons to {out}')


if __name__ == '__main__':
    main()
