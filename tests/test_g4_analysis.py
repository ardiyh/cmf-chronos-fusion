"""G4 loader: origin rule, target check, replay check (synthetic run folders)."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts import g4_analysis as g


def fake_run(root, *, target='OT', n=20000, h=24, cap=None, replay_diff=0.):
    origins = g.expected_origins(n, h, cap)
    root.mkdir(parents=True)
    (root / 'manifest.json').write_text(json.dumps(dict(
        status='complete', split=dict(name='test'), data_sha256='abc', target=target,
        n_rows=n, origins={str(h): origins})))
    fields = ['horizon', 'adapter', 'replay', 'mode', 'origin_idx', 'replay_max_abs_diff', *g.METRICS]
    with (root / 'losses.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for o in origins:
            for a, rep in (('identity', 0), ('krr', 0), ('identity', 1)):
                w.writerow(dict(horizon=h, adapter=a, replay=rep, mode='target_units', origin_idx=o,
                                replay_max_abs_diff=replay_diff if rep else '',
                                **{k: 1.0 for k in g.METRICS}))
    return origins


class G4AnalysisTests(unittest.TestCase):
    def test_capped_and_uncapped_origin_rules(self):
        self.assertEqual(len(g.expected_origins(20000, 24, 150)), 150)
        self.assertGreater(len(g.expected_origins(20000, 24, None)), 150)

    def test_load_checks_target_rule_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            origins = fake_run(run, cap=150)
            data = g.load(run, 'abc', 'OT', (24,), 150, ['identity', 'krr'])
            self.assertEqual(sorted(data[24, 'krr']), origins)
            for kwargs in (dict(frame_sha256='zzz', target='OT', cap=150),
                           dict(frame_sha256='abc', target='HUFL', cap=150),
                           dict(frame_sha256='abc', target='OT', cap=None)):
                with self.assertRaises(ValueError):
                    g.load(run, kwargs['frame_sha256'], kwargs['target'], (24,), kwargs['cap'],
                           ['identity', 'krr'])
            bad = Path(tmp) / 'bad'
            fake_run(bad, cap=150, replay_diff=1e-6)
            with self.assertRaises(ValueError):
                g.load(bad, 'abc', 'OT', (24,), 150, ['identity'])


if __name__ == '__main__':
    unittest.main()
