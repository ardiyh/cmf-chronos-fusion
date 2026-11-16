"""A4 scheduling mitigation around the unchanged frozen inference runner.

Wait one second before each prediction; at >=75 C, wait until <=65 C.
The independent GPU guard must still enforce 160 W, 85 C and thermal flags.
Timing is diagnostic/operational only and cannot be used as latency evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


class WindowComplete(Exception):
    """Bounded diagnostic ended between predictions; the grid remains partial."""


def read_temperature():
    result = subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=temperature.gpu',
        '--format=csv,noheader,nounits'], timeout=5, check=True,
        capture_output=True, text=True)
    return float(result.stdout.strip())


class PacedEndpoint:
    def __init__(self, pipeline, log, *, duration=None, probe=None,
                 sleeper=time.sleep, clock=time.monotonic):
        self.pipeline, self.log = pipeline, log
        self.duration, self.probe = duration, probe or read_temperature
        self.sleep, self.clock = sleeper, clock
        self.started, self.calls = clock(), 0

    def check_window(self):
        if self.duration is not None and self.clock() - self.started >= self.duration:
            raise WindowComplete()

    def temperature(self):
        value = self.probe()
        if not math.isfinite(value) or value >= 85 or value < 0:
            raise RuntimeError('Invalid or unsafe GPU temperature')
        return value

    def predict(self, **kwargs):
        self.check_window()
        before = self.clock()
        self.sleep(1.)
        self.check_window()
        temperature = self.temperature()
        if temperature >= 75:
            cooling_started = self.clock()
            while temperature > 65:
                self.check_window()
                if self.clock() - cooling_started >= 180:
                    raise RuntimeError('GPU cooldown exceeded 180 seconds')
                self.sleep(2.)
                temperature = self.temperature()
        self.check_window()
        waited = self.clock() - before
        result = self.pipeline.predict(**kwargs)
        self.calls += 1
        self.log.write(json.dumps(dict(call=self.calls,
            utc=datetime.now(timezone.utc).isoformat(),
            temperature_before_C=temperature, wait_seconds=waited)) + '\n')
        self.log.flush()
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', required=True,
        choices=['ettm2', 'ecl', 'opsd-wind', 'sdwpf', 'solar-al'])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--pacing-dir', required=True)
    parser.add_argument('--qualification-seconds', type=float, default=None)
    args = parser.parse_args(argv)
    if Path(args.output_dir).exists():
        parser.error('Output directory already exists')
    if args.qualification_seconds is not None and args.qualification_seconds <= 0:
        parser.error('Qualification duration must be positive')
    out = Path(args.pacing_dir)
    out.mkdir(parents=True, exist_ok=False)
    state = dict(status='starting', dataset=args.dataset, output_dir=args.output_dir,
        qualification_seconds=args.qualification_seconds, delay_before_prediction_s=1.,
        pause_temperature_C=75, resume_temperature_C=65, max_cooldown_s=180,
        created_utc=datetime.now(timezone.utc).isoformat(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def save():
        temp = out / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(out / 'status.json')

    save()
    endpoint = None
    with (out / 'predictions.jsonl').open('x') as log:
        try:
            from experiments import run_peer_review_controls as c

            def load(model, revision, device):
                nonlocal endpoint
                pipeline, checkpoint = c._load_chronos(model, revision, device)
                endpoint = PacedEndpoint(pipeline, log, duration=args.qualification_seconds)
                state['status'] = 'running'
                save()
                return endpoint, checkpoint

            c.main(['--dataset', args.dataset, '--horizons', '24', '--adapters',
                'identity', 'ridge', '--modes', 'target_units', 'target_standardized',
                'fused_standardized', '--repetitions', '3', '--max-origins', '150',
                '--base-seed', '2026', '--fit-seed', '42', '--split', 'test',
                '--device', 'cuda', '--revision',
                'ad294eaacead15db499b740ea4122266dd2a81a2',
                '--output-dir', args.output_dir], load_pipeline=load)
            state['status'] = 'grid_complete'
        except WindowComplete:
            state['status'] = 'window_complete'
        except BaseException as exc:
            state.update(status='failed', reason=repr(exc))
            raise
        finally:
            state.update(calls=endpoint.calls if endpoint else 0,
                elapsed_prediction_window_s=endpoint.clock()-endpoint.started if endpoint else None,
                finished_utc=datetime.now(timezone.utc).isoformat())
            save()
    print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    main()
