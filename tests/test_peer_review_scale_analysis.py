import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import run_peer_review_controls as controls
from scripts import peer_review_scale_analysis as analysis
from scripts.sinergi_stats import holm_adjust
from test_peer_review_controls import GRID_LABEL, RandomEndpoint


class ScaleAnalysisTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.batch = Path(tmp.name)
        for dataset in analysis.DATASETS:
            out = self.batch / dataset
            out.mkdir()
            origins = list(range(1024, 1504, 24))
            manifest = dict(status="complete", dataset=dataset, context_length=512,
                            origins={"24": origins}, repetitions=3,
                            adapters=["identity", "ridge"], modes=list(analysis.MODES),
                            split={"name": "test"})
            rows = []
            for i, origin in enumerate(origins):
                for rep in range(3):
                    for adapter in ("identity", "ridge"):
                        for j, mode in enumerate(analysis.MODES):
                            # Different mean- and median-point effects by construction.
                            loss = 2 + i / 100 + rep / 10 + j / 4
                            if adapter == "ridge":
                                loss -= (0.3 + i / 20) if j == 0 else 0.1
                            rows.append(dict(dataset=dataset, horizon=24, origin_idx=origin,
                                             adapter=adapter, mode=mode, repetition=rep,
                                             replay=0, sampler_seed=100 + i * 3 + rep,
                                             MASE=loss, MASE_mean=loss + (j + 1) ** 2,
                                             WQL=7 + j, coverage_50=0.4, coverage_90=0.8,
                                             width_50=3, width_90=9))
                    replay = next(r for r in rows[-6:]
                                  if r['adapter'] == 'identity' and r['mode'] == 'target_units')
                    rows.append(dict(replay, replay=1, MASE=999))
            manifest['n_records'] = len(rows)
            (out / "manifest.json").write_text(json.dumps(manifest))
            self.write_rows(out, rows)

    def write_rows(self, directory, rows):
        with (directory / "losses.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_scale_pairing_repetition_average_and_replay_exclusion(self):
        result = analysis.analyze_batch(self.batch)
        self.assertEqual(len(result['comparisons']), 32)
        self.assertEqual(len(result['summaries']), 48)
        row = result['comparisons'][0]
        expected = [-0.45 - i / 20 for i in range(20)]
        self.assertEqual(row['metric'], 'MASE')
        self.assertEqual(row['mode_b'], 'target_standardized')
        self.assertEqual(row['n'], 20)
        self.assertAlmostEqual(row['mean_diff'], np.mean(expected))
        self.assertEqual(row['win_rate'], 1)
        identity = next(r for r in result['summaries'] if r['dataset'] == 'etth1'
                        and r['adapter'] == 'identity' and r['mode'] == 'target_units')
        self.assertAlmostEqual(identity['MASE'], 2 + 0.095 + 0.1)
        self.assertEqual(identity['WQL'], 7)  # raw doubled pinball, not normalized
        self.assertAlmostEqual(identity['coverage_90'], 0.8)

    def test_holm_family_is_sixteen_per_metric_and_descriptive_is_separate(self):
        result = analysis.analyze_batch(self.batch)
        for metric in ('MASE', 'MASE_mean'):
            rows = [r for r in result['comparisons'] if r['metric'] == metric]
            self.assertEqual(len(rows), 16)
            self.assertEqual([r['holm_p'] for r in rows],
                             holm_adjust([r['wilcoxon_p'] for r in rows]))
            self.assertEqual([r['holm_block_p'] for r in rows],
                             holm_adjust([r['block_p_worst'] for r in rows]))
        self.assertFalse(any('holm_p' in r for r in result['descriptive']))
        first_mean = next(r for r in result['comparisons'] if r['metric'] == 'MASE_mean')
        self.assertAlmostEqual(first_mean['mean_diff'], result['comparisons'][0]['mean_diff'] - 3)

    def test_incomplete_run_and_missing_dataset_are_rejected(self):
        path = self.batch / 'solar-al' / 'manifest.json'
        doc = json.loads(path.read_text())
        doc['status'] = 'running'
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, 'complete'):
            analysis.analyze_batch(self.batch)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            analysis.analyze_batch(self.batch)

    def test_missing_duplicate_nonfinite_and_unpaired_seed_records_are_rejected(self):
        out = self.batch / 'etth1'
        with (out / 'losses.csv').open() as handle:
            original = list(csv.DictReader(handle))
        for mutation, reason in [('missing', 'grid'), ('duplicate', 'duplicate'),
                                 ('seed', 'seed'), ('nan', 'finite')]:
            with self.subTest(mutation=mutation):
                rows = [dict(r) for r in original]
                if mutation == 'missing':
                    rows.pop(1)
                elif mutation == 'duplicate':
                    rows.append(dict(rows[0]))
                elif mutation == 'seed':
                    rows[1]['sampler_seed'] = '0'
                else:
                    rows[1]['MASE'] = 'nan'
                self.write_rows(out, rows)
                with self.assertRaisesRegex(ValueError, reason):
                    analysis.analyze_batch(self.batch)

    def test_cli_records_hashes_and_refuses_existing_output(self):
        output = self.batch / 'analysis'
        analysis.main(['--batch-dir', str(self.batch), '--output', str(output)])
        provenance = json.loads((output / 'analysis.json').read_text())
        self.assertEqual(len(provenance['inputs']), 16)
        self.assertEqual(provenance['family_size_per_metric'], 16)
        self.assertEqual(provenance['point_primary'], 'sample_median')
        self.assertTrue((output / 'comparisons.csv').exists())
        with self.assertRaises(FileExistsError):
            analysis.main(['--batch-dir', str(self.batch), '--output', str(output)])

    def test_accepts_real_runner_output_with_each_scale_mode(self):
        t = np.arange(3000)
        df = pd.DataFrame({'target': 10 + np.sin(t / 10), 'covariate': np.cos(t / 15)})
        # Use actual fitting, serialization and metrics; replace only inference.
        out = self.batch / 'actual'
        controls.run_grid(df=df, dataset='etth1', feature_cols=list(df.columns),
                          target_col='target', seasonality=24, horizons=[24],
                          adapters=['identity', 'ridge'], modes=list(analysis.MODES),
                          repetitions=3, base_seed=2026, fit_seed=42, max_origins=2,
                          output_dir=out, pipeline=RandomEndpoint(), endpoint_label=GRID_LABEL,
                          device_map='cpu', context_len=512, num_samples=5)
        manifest, values = analysis.load_complete(out, 'etth1')
        self.assertEqual(manifest['n_records'], 42)
        self.assertEqual(len(values), 6)
        self.assertEqual(len(values['ridge', 'fused_standardized']['MASE']), 2)


if __name__ == '__main__':
    unittest.main()
