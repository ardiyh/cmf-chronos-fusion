"""CPU contract tests for the LightGBM D5 baseline: windows, leakage, determinism."""
import unittest

import numpy as np

from experiments import run_lgbm_baseline as b
from src import cmf_core as cmf


def series(n=2400, channels=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = np.sin(2 * np.pi * t / 24)[:, None] + 0.01 * t[:, None] / n
    return base + 0.3 * rng.standard_normal((n, channels))


class LgbmBaselineTests(unittest.TestCase):
    def test_phase_indices_are_same_phase_inside_context(self):
        for s in (24, 96, 144):
            for h in range(1, 49):
                for idx in b.phase_indices([1000], h, s):
                    gap = 1000 + h - 1 - int(idx[0])
                    self.assertEqual(gap % s, 0)
                    self.assertTrue(1000 - b.CONTEXT <= idx[0] < 1000)

    def test_features_use_only_the_context_window(self):
        values, origin, horizon = series(), 1500, 24
        floor = b.floors(values, len(values))
        X, y, mu, sd = b.features(values, [origin], horizon, 24, floor)
        changed = values.copy()
        rng = np.random.default_rng(1)
        changed[origin:] += rng.standard_normal(changed[origin:].shape)
        changed[:origin - b.CONTEXT] += rng.standard_normal(changed[:origin - b.CONTEXT].shape)
        X2, y2, mu2, sd2 = b.features(changed, [origin], horizon, 24, floor)
        np.testing.assert_array_equal(X, X2)
        np.testing.assert_array_equal(mu, mu2)
        np.testing.assert_array_equal(sd, sd2)
        self.assertFalse(np.allclose(y, y2))
        np.testing.assert_allclose(mu[0] + sd[0] * y, values[origin:origin + horizon, 0])

    def test_target_only_ignores_covariates(self):
        values = series()
        X, *_ = b.features(values[:, :1], [1500], 24, 24, b.floors(values[:, :1], len(values)))
        Xc, *_ = b.features(values, [1500], 24, 24, b.floors(values, len(values)))
        self.assertEqual(X.shape[1], 1 + b.N_LAGS + 2 + b.N_PHASE)
        self.assertEqual(Xc.shape[1], X.shape[1] + 3 * (values.shape[1] - 1))

    def test_split_origins_keep_labels_inside_their_split(self):
        n, horizon = 20000, 48
        train_end, val_end, _ = cmf.get_split_indices(n)
        sets = b.split_origins(n, horizon)
        self.assertLessEqual(len(sets['train']), b.MAX_TRAIN_ORIGINS)
        self.assertLessEqual(max(sets['train']) + horizon, train_end)
        self.assertLessEqual(max(sets['refit']) + horizon, val_end)
        self.assertGreaterEqual(min(sets['validation']), train_end)
        self.assertLessEqual(max(sets['validation']) + horizon, val_end)
        self.assertGreaterEqual(min(sets['train']), b.CONTEXT)

    def test_small_run_is_finite_and_deterministic(self):
        import lightgbm as lgb
        rounds, grid = b.MAX_ROUNDS, b.GRID
        b.MAX_ROUNDS, b.GRID = 30, b.GRID[:2]
        try:
            values = series(n=3000)
            test = cmf.rolling_window_origins(3000, 2400, b.CONTEXT, 24, stride=24)
            runs = [b.run_variant(lgb, values, test, 24, 24, b.floors(values, 3000),
                                  log=lambda m: None) for _ in range(2)]
        finally:
            b.MAX_ROUNDS, b.GRID = rounds, grid
        (q, selection), (q2, _) = runs
        self.assertEqual(q.shape, (len(test), 24, 9))
        self.assertTrue(np.isfinite(q).all())
        np.testing.assert_array_equal(q, q2)
        self.assertEqual(len(selection['grid_scores']), 2)


if __name__ == '__main__':
    unittest.main()
