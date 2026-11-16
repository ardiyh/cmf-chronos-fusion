import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_with_gpu_guard as guard


def reading(**changes):
    return dict(timestamp='now', temperature=70., power_limit=160., power_draw=158.,
                fan=60., utilization=90., memory_used=3400.,
                hw_thermal='Not Active', sw_thermal='Not Active',
                power_brake='Not Active', **changes)


class GpuGuardTests(unittest.TestCase):
    def test_duration_limit_stops_and_confirms_child_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(guard, 'read_gpu', side_effect=lambda: reading()):
                result = guard.run([sys.executable, '-c', 'import time; time.sleep(120)'],
                    Path(tmp) / 'guard', power_limit=160, max_temperature=85,
                    interval=.01, max_duration=.03)
        self.assertEqual(result['status'], 'time_limit')
        self.assertTrue(result['child_exit_confirmed'])

    def test_fan_rpm_probe_requires_two_finite_values(self):
        self.assertTrue(hasattr(guard, 'read_fan_rpm'), 'RPM telemetry missing')
        with patch.object(guard.subprocess, 'run') as run:
            run.return_value.stdout = '1200\n1300\n'
            self.assertEqual(guard.read_fan_rpm(), dict(fan0_rpm=1200., fan1_rpm=1300.))
            run.return_value.stdout = 'N/A\n1300\n'
            with self.assertRaises(ValueError):
                guard.read_fan_rpm()

    def test_fan_probe_failure_prevents_launch(self):
        self.assertTrue(hasattr(guard, 'read_fan_rpm'), 'RPM telemetry missing')
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(guard, 'read_gpu', side_effect=lambda: reading()), \
                 patch.object(guard, 'read_fan_rpm', side_effect=RuntimeError('RPM failed')):
                result = guard.run([sys.executable, '-c', 'raise AssertionError()'],
                    Path(tmp) / 'guard', power_limit=160, max_temperature=85,
                    interval=.01, fan_rpm=True)
        self.assertEqual(result['status'], 'guard_stopped')
        self.assertNotIn('pid', result)

    def test_power_and_temperature_limits_and_thermal_flags(self):
        self.assertIsNone(guard.stop_reason(reading(), 160, 85))
        for change in ({'temperature': 85.}, {'temperature': float('nan')},
                       {'power_limit': 220.}, {'hw_thermal': 'Active'},
                       {'sw_thermal': 'Active'}, {'power_brake': 'Active'}):
            self.assertIsNotNone(guard.stop_reason(reading() | change, 160, 85))

    def run_guard(self, code, probe):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        out = Path(temp.name) / 'guard'
        with patch.object(guard, 'read_gpu', side_effect=probe):
            result = guard.run([sys.executable, '-c', code], out,
                               power_limit=160, max_temperature=85, interval=0.01)
        self.assertEqual(json.loads((out / 'status.json').read_text())['status'], result['status'])
        return result, out

    def test_completed_child_is_reported_complete(self):
        result, out = self.run_guard('print("finished")', lambda: reading())
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['returncode'], 0)
        self.assertIn('finished', (out / 'command.log').read_text())

    def test_nonzero_exit_is_reported_failed(self):
        result, _ = self.run_guard('raise SystemExit(7)', lambda: reading())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['returncode'], 7)

    def test_hot_initial_gpu_does_not_launch_the_command(self):
        result, _ = self.run_guard('raise AssertionError("should not run")',
                                   lambda: reading() | {'temperature': 86.})
        self.assertEqual(result['status'], 'guard_stopped')
        self.assertNotIn('pid', result)

    def test_lost_gpu_terminates_running_child_and_records_failure(self):
        probes = iter([reading(), subprocess.TimeoutExpired('nvidia-smi', 5)])
        def probe():
            item = next(probes)
            if isinstance(item, Exception):
                raise item
            return item
        result, _ = self.run_guard('import time; time.sleep(120)', probe)
        self.assertEqual(result['status'], 'guard_stopped')
        self.assertIn('TimeoutExpired', result['reason'])
        self.assertIsNotNone(result['returncode'])

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(FileExistsError):
                guard.run([sys.executable, '-c', 'pass'], Path(out),
                          power_limit=160, max_temperature=85, interval=.01)

    def test_sw_thermal_grace_ignores_only_brief_sw_episodes(self):
        self.assertIsNone(guard.stop_reason((reading() | {'sw_thermal': 'Active'}), 160, 85,
                                            ignore_sw_thermal=True))
        for change in ({'hw_thermal': 'Active'}, {'power_brake': 'Active'}, {'temperature': 85.}):
            self.assertIsNotNone(guard.stop_reason(reading() | change, 160, 85,
                                                   ignore_sw_thermal=True))
        self.assertEqual(guard.sw_thermal_window(False, 5., 1.), (None, 0.))
        self.assertEqual(guard.sw_thermal_window(True, 5., None), (5., 0.))
        self.assertEqual(guard.sw_thermal_window(True, 9., 5.), (5., 4.))

    def test_brief_sw_thermal_episode_lets_the_child_finish(self):
        flags = iter(['Active', 'Active'] + ['Not Active'] * 1000)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(guard, 'read_gpu', side_effect=lambda: (reading() | {'sw_thermal': next(flags)})):
                result = guard.run([sys.executable, '-c', 'import time; time.sleep(.3)'],
                    Path(tmp) / 'guard', power_limit=160, max_temperature=85,
                    interval=.01, sw_thermal_grace=60.)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['sw_thermal_samples'], 2)

    def test_persistent_sw_thermal_stops_after_grace(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(guard, 'read_gpu', side_effect=lambda: (reading() | {'sw_thermal': 'Active'})):
                result = guard.run([sys.executable, '-c', 'import time; time.sleep(120)'],
                    Path(tmp) / 'guard', power_limit=160, max_temperature=85,
                    interval=.01, sw_thermal_grace=.1)
        self.assertEqual(result['status'], 'guard_stopped')
        self.assertIn('SW thermal slowdown active', result['reason'])
        self.assertTrue(result['child_exit_confirmed'])


if __name__ == '__main__':
    unittest.main()
