import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments import run_peer_review_controls as controls
from scripts import peer_review_control_analysis as analysis
from scripts.sinergi_stats import holm_adjust, wilcoxon_signed_rank_p
from test_peer_review_controls import GRID_LABEL, RandomEndpoint, grid_frame


def write_run(directory, rows, *, origins, status="complete", context_length=8):
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({
        "status": status, "context_length": context_length,
        "origins": {"2": origins}, "split": {"name": "test"}}))
    with (directory / "losses.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["horizon", "origin_idx", "adapter", "mode",
                                                    "repetition", "replay", "MASE", "MASE_mean"])
        writer.writeheader()
        writer.writerows(rows)


def row(adapter, origin, rep, mase, replay=0):
    return {"horizon": 2, "origin_idx": origin, "adapter": adapter, "mode": "target_units",
            "repetition": rep, "replay": replay, "MASE": mase, "MASE_mean": mase + 1}


class ControlAnalysisTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_repetitions_are_averaged_per_origin_and_replays_ignored(self):
        origins = [10, 12, 14, 16, 18]
        rows = [row("identity", o, r, 1.0 + 0.1 * r) for o in origins for r in (0, 1)]
        rows += [row("krr", o, r, 0.5 + 0.2 * r) for o in origins for r in (0, 1)]
        rows += [row("identity", o, 0, 99.0, replay=1) for o in origins]
        write_run(self.tmp / "run", rows, origins=origins)
        losses = analysis.origin_losses(self.tmp / "run", metric="MASE")
        self.assertEqual(losses[2]["identity"], {o: 1.05 for o in origins})
        self.assertEqual(losses[2]["krr"], {o: 0.6 for o in origins})

    def test_comparison_statistics_and_holm_only_over_the_family(self):
        origins = list(range(10, 50, 2))
        gains = [0.1 * ((i % 7) - 1.5) for i in range(len(origins))]
        rows = [row("identity", o, 0, 2.0) for o in origins]
        rows += [row("krr", o, 0, 2.0 + g) for o, g in zip(origins, gains)]
        rows += [row("krr_targetonly", o, 0, 2.0 - g / 2) for o, g in zip(origins, gains)]
        write_run(self.tmp / "run", rows, origins=origins)
        result = analysis.analyze(self.tmp / "run",
                                  family=[("krr", "identity"), ("krr", "krr_targetonly")],
                                  descriptive=[("krr_targetonly", "identity")])
        first = result[0]
        self.assertEqual((first["a"], first["b"], first["n"]), ("krr", "identity", len(origins)))
        self.assertAlmostEqual(first["mean_diff"], np.mean(gains))
        self.assertAlmostEqual(first["mean_pct"], 100 * np.mean(gains) / 2.0)
        self.assertAlmostEqual(first["win_rate"], np.mean(np.array(gains) < 0))
        # Same float arithmetic as the script: (2 + g) - 2 can break ties that g has.
        self.assertEqual(first["wilcoxon_p"],
                         wilcoxon_signed_rank_p([(2.0 + g) - 2.0 for g in gains]))
        # ceil(context 8 / min spacing 2) = 4
        self.assertEqual(first["block_length"], 4)
        family_p = [r["wilcoxon_p"] for r in result if r["role"] == "family"]
        self.assertEqual([r["holm_p"] for r in result if r["role"] == "family"],
                         holm_adjust(family_p))
        self.assertIsNone(result[2]["holm_p"])
        self.assertEqual(result[2]["role"], "descriptive")

    def test_mismatched_origins_and_incomplete_runs_are_refused(self):
        rows = [row("identity", o, 0, 1.0) for o in (10, 12, 14)] + [row("krr", 10, 0, 0.5)]
        write_run(self.tmp / "gap", rows, origins=[10, 12, 14])
        with self.assertRaisesRegex(ValueError, "origins"):
            analysis.analyze(self.tmp / "gap", family=[("krr", "identity")])
        write_run(self.tmp / "running", rows, origins=[10, 12, 14], status="running")
        with self.assertRaisesRegex(ValueError, "complete"):
            analysis.analyze(self.tmp / "running", family=[("krr", "identity")])

    def test_runs_on_real_runner_output(self):
        out = self.tmp / "grid"
        controls.run_grid(df=grid_frame(), dataset="toy", feature_cols=["covariate", "target"],
                          target_col="target", seasonality=2, horizons=[2],
                          adapters=["identity", "krr", "krr_targetonly"],
                          modes=["target_units"], repetitions=2, base_seed=11, fit_seed=42,
                          max_origins=None, output_dir=out, pipeline=RandomEndpoint(),
                          endpoint_label=GRID_LABEL, context_len=8, num_samples=5)
        result = analysis.analyze(out, family=[("krr", "identity")], metric="MASE_mean")
        self.assertEqual(result[0]["n"], 6)
        self.assertEqual(result[0]["metric"], "MASE_mean")
        self.assertTrue(math.isfinite(result[0]["block_p_worst"]))

    def test_main_refuses_an_existing_output(self):
        (self.tmp / "exists").mkdir()
        with self.assertRaises(FileExistsError):
            analysis.main(["--run-dir", str(self.tmp), "--family", "krr/identity",
                           "--output", str(self.tmp / "exists")])


if __name__ == "__main__":
    unittest.main()
