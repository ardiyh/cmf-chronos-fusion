import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

from scripts import peer_review_block_sensitivity as sens
from scripts.sinergi_block_robust import _signed_ranks


def brute_force_p(sums):
    observed = abs(sum(sums))
    totals = [abs(sum(s * b for s, b in zip(signs, sums)))
              for signs in itertools.product((-1, 1), repeat=len(sums))]
    return sum(t >= observed - 1e-9 for t in totals) / len(totals)


class BlockSensitivityTests(unittest.TestCase):
    def test_block_sums_partition_with_and_without_offset(self):
        scores = list(range(1, 11))
        np.testing.assert_array_equal(sens.block_sums(scores, 4, 0), [10, 26, 19])
        np.testing.assert_array_equal(sens.block_sums(scores, 4, 2), [3, 18, 34])

    def test_exact_p_matches_brute_force_with_zeros_and_ties(self):
        diffs = [0.3, -0.1, 0.0, 0.3, 0.5, -0.2, 0.1, 0.0, 0.4]
        sums = sens.block_sums(_signed_ranks(diffs), 2, 1)
        p, method = sens.sign_flip_p(sums, rng=np.random.default_rng(0))
        self.assertEqual(method, "exact")
        self.assertAlmostEqual(p, brute_force_p(list(sums)))

    def test_minimum_attainable_p_is_two_over_two_to_the_k(self):
        p, _ = sens.sign_flip_p(np.ones(5), rng=np.random.default_rng(0))
        self.assertAlmostEqual(p, 2 / 32)

    def test_unit_blocks_equal_scipy_exact_wilcoxon(self):
        diffs = [0.9, -0.4, 1.3, 0.2, -0.7, 1.1, 0.6, -0.15, 0.8, 1.6]
        p, method = sens.sign_flip_p(sens.block_sums(_signed_ranks(diffs), 1, 0),
                                     rng=np.random.default_rng(0))
        self.assertEqual(method, "exact")
        self.assertAlmostEqual(p, stats.wilcoxon(diffs, method="exact").pvalue)

    def test_monte_carlo_agrees_with_exact_beyond_the_enumeration_limit(self):
        sums = np.random.default_rng(3).normal(0.4, 1.0, size=17)
        exact, _ = sens.sign_flip_p(sums, rng=np.random.default_rng(0), exact_max_blocks=17)
        mc, method = sens.sign_flip_p(sums, rng=np.random.default_rng(1))
        self.assertEqual(method, "monte_carlo")
        self.assertAlmostEqual(mc, exact, delta=0.01)

    def test_block_lengths_include_ladder_and_overlap_rules(self):
        lengths = sens.block_lengths(n=150, origins=[0, 24, 48, 96, 144])
        # cube root 6, ceil(512/median spacing 36) = 15, ceil(512/min spacing 24) = 22
        self.assertEqual(lengths, {"ladder_1": 1, "ladder_2": 2, "ladder_4": 4,
                                   "ladder_8": 8, "ladder_16": 16, "ladder_32": 32,
                                   "cube_root": 6, "overlap_median": 15,
                                   "overlap_max": 22})

    def test_existing_output_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                sens.main(["--output-dir", tmp])
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
