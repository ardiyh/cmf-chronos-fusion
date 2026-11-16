import unittest

from scripts.sinergi_block_robust import _signed_ranks, block_sign_flip_p
from scripts.sinergi_stats import wilcoxon_signed_rank_p


class BlockRobustTests(unittest.TestCase):
    def test_unit_blocks_reproduce_wilcoxon(self):
        diffs = [((i * 37) % 11 - 6.5) / 10 for i in range(80)]
        p_flip = block_sign_flip_p(_signed_ranks(diffs), block_length=1,
                                   n_resamples=20_000, seed=0)
        self.assertAlmostEqual(p_flip, wilcoxon_signed_rank_p(diffs), delta=0.02)

    def test_signed_ranks_zero_ties_and_signs(self):
        self.assertEqual(_signed_ranks([0.0, -2.0, 1.0, 2.0]), [0.0, -2.5, 1.0, 2.5])

    def test_one_block_cannot_reject(self):
        p = block_sign_flip_p([-1.0] * 30, block_length=30, n_resamples=2_000, seed=0)
        self.assertGreater(p, 0.9)


if __name__ == "__main__":
    unittest.main()
