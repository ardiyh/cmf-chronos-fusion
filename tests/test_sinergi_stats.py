import math
from pathlib import Path
import tempfile
import unittest

from scripts.sinergi_stats import (
    analyze_retained_results,
    exact_sign_test,
    holm_adjust,
    load_paired_differences,
    moving_block_bootstrap_ci,
    top_gain_share,
    wilcoxon_signed_rank_p,
    write_analysis_csv,
)


class SinergiStatsTests(unittest.TestCase):
    def test_holm_adjustment_preserves_original_order(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual([round(value, 6) for value in adjusted], [0.03, 0.06, 0.06])

    def test_sign_test_discards_ties(self):
        result = exact_sign_test([-1.0, -2.0, 0.0, 3.0])
        self.assertEqual(result["wins"], 2)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(result["ties"], 1)
        self.assertAlmostEqual(result["p_two_sided"], 1.0)

    def test_block_bootstrap_constant_series_has_degenerate_interval(self):
        low, high = moving_block_bootstrap_ci(
            [-2.0] * 20, block_length=4, n_resamples=100, seed=42
        )
        self.assertEqual(low, -2.0)
        self.assertEqual(high, -2.0)

    def test_top_gain_share_can_exceed_one_hundred_percent(self):
        differences = [-10.0] + [0.5] * 18 + [0.0]
        result = top_gain_share(differences, fraction=0.05)
        self.assertEqual(result["n_top"], 1)
        self.assertAlmostEqual(result["share"], 10.0)
        self.assertTrue(math.isclose(result["total_gain"], 1.0))

    def test_retained_ettm2_pairing_reproduces_published_wilcoxon(self):
        result_dir = Path(__file__).parents[1] / "results" / "rtx_20260803_test"
        differences = load_paired_differences(result_dir, "ettm2", "mlp", 24)
        self.assertEqual(len(differences), 150)
        self.assertAlmostEqual(sum(differences) / len(differences), -0.0269, places=4)
        self.assertAlmostEqual(wilcoxon_signed_rank_p(differences), 0.810935, places=6)

    def test_full_analysis_uses_manuscript_selection_and_holm_family(self):
        result_dir = Path(__file__).parents[1] / "results" / "rtx_20260803_test"
        rows = analyze_retained_results(result_dir, n_resamples=100)
        self.assertEqual(len(rows), 16)
        keyed = {(row["dataset"], row["H"]): row for row in rows}
        self.assertEqual(keyed[("sdwpf", 48)]["adapter"], "krr")
        self.assertEqual(keyed[("ecl", 48)]["wilcoxon_p"], 1.0)
        self.assertLess(keyed[("solar-al", 24)]["holm_p"], 0.05)
        self.assertLess(keyed[("solar-al", 48)]["holm_p"], 0.05)

    def test_analysis_writer_creates_sixteen_row_csv(self):
        result_dir = Path(__file__).parents[1] / "results" / "rtx_20260803_test"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "stats.csv"
            write_analysis_csv(result_dir, destination, n_resamples=100)
            lines = destination.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 17)
        self.assertIn("holm_p", lines[0])
        self.assertIn("block_ci_lower", lines[0])


if __name__ == "__main__":
    unittest.main()
