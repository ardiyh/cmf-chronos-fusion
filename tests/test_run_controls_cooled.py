"""CPU tests of the cooldown-only endpoint wrapper; no GPU or model is used."""
import io
import unittest

from scripts.run_controls_cooled import CooledEndpoint


class Pipeline:
    def __init__(self):
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        return kwargs


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class CooledEndpointTests(unittest.TestCase):
    def make(self, temps):
        clock, temps = Clock(), iter(temps)
        return CooledEndpoint(Pipeline(), io.StringIO(), probe=lambda: next(temps),
                              sleeper=clock.sleep, clock=clock), clock

    def test_cool_gpu_predicts_without_delay(self):
        endpoint, clock = self.make([60.])
        self.assertEqual(endpoint.predict(x=1), {'x': 1})
        self.assertEqual(clock.now, 0.)

    def test_hot_gpu_waits_until_resume_temperature(self):
        endpoint, clock = self.make([76., 70., 66., 65.])
        endpoint.predict()
        self.assertEqual(clock.now, 6.)
        self.assertEqual(endpoint.pipeline.calls, 1)

    def test_fixed_delay_precedes_each_prediction(self):
        clock, temps = Clock(), iter([60., 60.])
        endpoint = CooledEndpoint(Pipeline(), io.StringIO(), delay=1., probe=lambda: next(temps),
                                  sleeper=clock.sleep, clock=clock)
        endpoint.predict()
        endpoint.predict()
        self.assertEqual(clock.now, 2.)

    def test_unsafe_or_stuck_temperature_stops(self):
        endpoint, _ = self.make([85.])
        with self.assertRaises(RuntimeError):
            endpoint.predict()
        endpoint, _ = self.make([80.] * 200)
        with self.assertRaises(RuntimeError):
            endpoint.predict()
        self.assertEqual(endpoint.pipeline.calls, 0)


if __name__ == '__main__':
    unittest.main()
