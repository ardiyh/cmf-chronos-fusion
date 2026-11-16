"""CPU tests: pacing must change scheduling only, never seeded predictions."""
import importlib
import importlib.util
import io
import json
import unittest

import numpy as np
import pandas as pd
import torch


class FakeClock:
    def __init__(self):
        self.now = 0.

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Endpoint:
    def predict(self, **kwargs):
        return kwargs


class PacingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.run_a4_paced'),
                             'external pacing wrapper is missing')
        self.p = importlib.import_module('scripts.run_a4_paced')
        self.clock = FakeClock()
        self.log = io.StringIO()

    def wrapper(self, temperatures, pipeline=None, duration=None):
        values = iter(temperatures)
        return self.p.PacedEndpoint(pipeline or Endpoint(), self.log,
            duration=duration, probe=lambda: next(values),
            sleeper=self.clock.sleep, clock=self.clock.clock)

    def test_cooling_waits_until_lower_threshold_and_forwards_arguments(self):
        endpoint = self.wrapper([80., 70., 65.])
        self.assertEqual(endpoint.predict(context='unchanged', top_k=50),
                         dict(context='unchanged', top_k=50))
        self.assertEqual(self.clock.now, 5.)
        event = json.loads(self.log.getvalue())
        self.assertEqual(event['temperature_before_C'], 65.)
        self.assertEqual(event['wait_seconds'], 5.)

    def test_window_ends_before_another_prediction(self):
        endpoint = self.wrapper([60.], duration=1.)
        with self.assertRaises(self.p.WindowComplete):
            endpoint.predict()
        self.assertEqual(endpoint.calls, 0)

    def test_failed_temperature_probe_never_reaches_endpoint(self):
        endpoint = self.wrapper([float('nan')])
        with self.assertRaisesRegex(RuntimeError, 'temperature'):
            endpoint.predict()

    def test_cooldown_has_a_deadline(self):
        endpoint = self.wrapper([80.]*100)
        with self.assertRaisesRegex(RuntimeError, 'cooldown'):
            endpoint.predict()
        self.assertEqual(endpoint.calls, 0)

    def test_seeded_control_samples_are_identical_with_pacing(self):
        from experiments import run_peer_review_controls as c

        class RandomEndpoint:
            def predict(self, prediction_length, num_samples, **kwargs):
                return torch.rand((1, num_samples, prediction_length))

        args = dict(df=pd.DataFrame({'y': [10., 12., 14., 16., 13., 13.]}),
            origin=4, context_len=4, pred_len=2, feature_cols=['y'], target_col='y',
            seasonality=2, device_map='cpu', fit_seed=42, sampler_seed=1729,
            num_samples=9, adapter_kind='identity', mode='target_units')
        first = c.evaluate_control_origin(pipeline=RandomEndpoint(), **args)
        paced = c.evaluate_control_origin(pipeline=self.wrapper([60.], RandomEndpoint()), **args)
        np.testing.assert_array_equal(first['samples'], paced['samples'])


if __name__ == '__main__':
    unittest.main()
