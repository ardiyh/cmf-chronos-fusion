import unittest

import numpy as np
import pandas as pd

from experiments.run_revision import evaluate_single_origin_cmf, _prepare_single_origin


class ZeroFutureProtocolTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0, 0.0, 0.0]})
        self.common = dict(
            df=self.df,
            origin=4,
            context_len=4,
            pred_len=2,
            feature_cols=["target"],
            target_col="target",
            seasonality=2,
            device_map="cpu",
        )

    def test_default_mask_excludes_zero_future_for_seasonal_naive(self):
        result = evaluate_single_origin_cmf(
            adapter_kind="seasonal_naive", pipeline=None, **self.common
        )
        self.assertIsNone(result)

    def test_sensitivity_switch_includes_zero_future_for_seasonal_naive(self):
        result = evaluate_single_origin_cmf(
            adapter_kind="seasonal_naive", pipeline=None,
            include_zero_future=True, **self.common
        )
        self.assertIsNotNone(result)

    def test_prepare_origin_uses_the_same_switch(self):
        prepare_common = dict(self.common)
        prepare_common.pop("seasonality")
        default = _prepare_single_origin(
            adapter_kind="identity", seed=42, adapter_kwargs={}, **prepare_common
        )
        included = _prepare_single_origin(
            adapter_kind="identity", seed=42, adapter_kwargs={},
            include_zero_future=True, **prepare_common
        )
        self.assertIsNone(default)
        self.assertTrue(np.array_equal(included[1], np.zeros(2)))


if __name__ == "__main__":
    unittest.main()
