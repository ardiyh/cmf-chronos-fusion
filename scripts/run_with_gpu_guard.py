"""Run one command with logged GPU telemetry and a bounded failure shutdown.

Read-only GPU monitoring: this does not change power limits or fan settings.
Only the new command's process group is stopped when a guard fails.
"""
import argparse
import csv
import json
import math
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ['timestamp', 'temperature', 'power_limit', 'power_draw', 'fan',
          'utilization', 'memory_used', 'hw_thermal', 'sw_thermal', 'power_brake']
QUERY = ('timestamp,temperature.gpu,power.limit,power.draw,fan.speed,utilization.gpu,'
         'memory.used,clocks_throttle_reasons.hw_thermal_slowdown,'
         'clocks_throttle_reasons.sw_thermal_slowdown,'
         'clocks_throttle_reasons.hw_power_brake_slowdown')


def read_gpu():
    result = subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=' + QUERY,
                             '--format=csv,noheader,nounits'], timeout=5,
                            check=True, capture_output=True, text=True)
    row = next(csv.reader(result.stdout.splitlines()))
    if len(row) != len(FIELDS):
        raise ValueError('Unexpected GPU telemetry fields')
    values = {k: v.strip() for k, v in zip(FIELDS, row)}
    for k in FIELDS[1:7]:
        values[k] = float(values[k])
    return values


def read_fan_rpm():
    result = subprocess.run(['nvidia-settings', '-t', '-q',
        '[fan:0]/GPUCurrentFanSpeedRPM', '-q', '[fan:1]/GPUCurrentFanSpeedRPM'],
        env=os.environ | {'DISPLAY': os.environ.get('DISPLAY', ':0')},
        timeout=5, check=True, capture_output=True, text=True)
    values = [float(v) for v in result.stdout.split()]
    if len(values) != 2 or not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError('Both fan RPM readings must be finite and nonnegative')
    return dict(zip(('fan0_rpm', 'fan1_rpm'), values))


def stop_reason(gpu, power_limit, max_temperature, ignore_sw_thermal=False):
    if not all(math.isfinite(gpu[k]) for k in FIELDS[1:7]):
        return 'Nonfinite GPU telemetry'
    if abs(gpu['power_limit'] - power_limit) > .5:
        return 'Power limit differs from the required setting'
    if gpu['temperature'] >= max_temperature:
        return 'Temperature guard reached'
    flags = [k for k in FIELDS[7:] if not (ignore_sw_thermal and k == 'sw_thermal')]
    if any(gpu[k] != 'Not Active' for k in flags):
        return 'Thermal slowdown or hardware power brake reported'
    return None


def sw_thermal_window(active, now, since):
    """(onset, continuous seconds) of the current SW thermal slowdown episode."""
    if not active:
        return None, 0.
    since = now if since is None else since
    return since, now - since


def run(command, out, *, power_limit, max_temperature, interval=2.,
        max_duration=None, fan_rpm=False, sw_thermal_grace=0.):
    """sw_thermal_grace > 0 stops only after SW thermal slowdown has been active
    continuously for that many seconds; every other condition still stops at once."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    state = dict(status='starting', command=command,
                 started_utc=datetime.now(timezone.utc).isoformat(),
                 required_power_limit_W=power_limit, temperature_guard_C=max_temperature)
    started = time.monotonic()
    state.update(max_duration_s=max_duration, fan_rpm_logged=fan_rpm,
                 sw_thermal_grace_s=sw_thermal_grace, sw_thermal_samples=0,
                 longest_sw_thermal_s=0.)
    sw_since = None

    def save():
        temporary = out / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(out / 'status.json')

    save()
    process = None
    with (out / 'command.log').open('x') as log, (out / 'telemetry.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS +
            (['fan0_rpm', 'fan1_rpm'] if fan_rpm else []))
        writer.writeheader()
        try:
            while True:
                if process is not None and process.poll() is not None:
                    state.update(status='complete' if process.returncode == 0 else 'failed',
                                 returncode=process.returncode)
                    break
                measurement = read_gpu()
                if fan_rpm:
                    measurement.update(read_fan_rpm())
                writer.writerow(measurement)
                handle.flush()
                reason = stop_reason(measurement, power_limit, max_temperature,
                                     ignore_sw_thermal=sw_thermal_grace > 0)
                if reason:
                    raise RuntimeError(reason)
                if sw_thermal_grace > 0:
                    active = measurement['sw_thermal'] != 'Not Active'
                    sw_since, episode = sw_thermal_window(active, time.monotonic(), sw_since)
                    state['sw_thermal_samples'] += active
                    state['longest_sw_thermal_s'] = max(state['longest_sw_thermal_s'], episode)
                    if active and episode >= sw_thermal_grace:
                        raise RuntimeError(f'SW thermal slowdown active for {episode:.0f} s')
                if (fan_rpm and measurement['temperature'] >= 70 and
                    min(measurement['fan0_rpm'], measurement['fan1_rpm']) <= 0):
                    raise RuntimeError('A fan reports zero RPM at >=70 C')
                if max_duration is not None and time.monotonic() - started >= max_duration:
                    raise TimeoutError('Configured command duration reached')
                if process is None:
                    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    state.update(status='running', pid=process.pid)
                    save()
                time.sleep(interval)
        except Exception as exc:
            state.update(status='time_limit' if isinstance(exc, TimeoutError)
                         else 'guard_stopped', reason=repr(exc))
            save()  # Report failure even if a broken driver prevents process exit.
            if process is not None:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                state['returncode'] = process.poll()
                state['child_exit_confirmed'] = process.returncode is not None
        finally:
            state['elapsed_s'] = time.monotonic() - started
            state['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--power-limit', type=float, default=160.)
    parser.add_argument('--max-temperature', type=float, default=85.)
    parser.add_argument('--max-duration', type=float, default=None)
    parser.add_argument('--fan-rpm', action='store_true')
    parser.add_argument('--sw-thermal-grace-seconds', type=float, default=0.,
                        help='0 stops on the first SW thermal sample (latency runs)')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a command is required after --')
    result = run(command, args.output_dir, power_limit=args.power_limit,
                 max_temperature=args.max_temperature,
                 max_duration=args.max_duration, fan_rpm=args.fan_rpm,
                 sw_thermal_grace=args.sw_thermal_grace_seconds)
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result['status'] == 'complete' else 1)


if __name__ == '__main__':
    main()
