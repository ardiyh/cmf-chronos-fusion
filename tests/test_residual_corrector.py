"""CPU tests of the G3 residual corrector: context-only features, ridge, shifts."""
import unittest

import numpy as np

from scripts import residual_corrector as g
from experiments import run_lgbm_baseline as b


def prediction(values, origin, horizon, bias):
    y = values[origin:origin + horizon, 0]
    median = y - bias
    levels = np.round(np.arange(.1, 1., .1), 1)
    return dict(median_pred=median, q_preds=median[:, None] + (levels - .5)[None],
                quantile_levels=levels, y_true=y.copy(),
                insample=values[origin - b.CONTEXT:origin, 0].copy(), MASE=0.)


class ResidualCorrectorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        t = np.arange(3000)
        self.values = np.column_stack([np.sin(2 * np.pi * t / 24), rng.standard_normal(3000)])
        self.floor = b.floors(self.values, 3000)

    def test_features_ignore_values_after_the_origin(self):
        o, h = 1500, 24
        preds = {o: prediction(self.values, o, h, .3)}
        X, *_ = g.design(self.values, [o], preds, h, 24, self.floor, True)
        changed = self.values.copy()
        changed[o + h:] += 5.
        X2, *_ = g.design(changed, [o], preds, h, 24, self.floor, True)
        np.testing.assert_array_equal(X, X2)
        Xt, *_ = g.design(self.values, [o], preds, h, 24, self.floor, False)
        self.assertEqual(X.shape[1] - Xt.shape[1], 2)

    def test_constant_bias_is_learned_and_removed(self):
        h = 24
        origins = list(range(600, 2400, h))
        preds = {o: prediction(self.values, o, h, .5) for o in origins}
        model, info = g.fit_corrector(self.values, preds, h, 24, self.floor, False)
        X, _, sigma = g.design(self.values, origins, preds, h, 24, self.floor, False)
        shifts = g.corrected(preds, origins, X, sigma, model, h)
        np.testing.assert_allclose(np.concatenate(list(shifts.values())), .5, atol=1e-6)
        self.assertIn(info['chosen_lambda'], g.LAMBDAS)

    def test_ridge_recovers_linear_signal(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((500, 3))
        y = X @ np.array([1., -2., .5]) + 3.
        np.testing.assert_allclose(g.ridge(X, y, 1e-9)(X), y, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
