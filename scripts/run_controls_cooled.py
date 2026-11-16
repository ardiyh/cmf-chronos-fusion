"""Run the frozen controls runner with an optional fixed delay and a cooldown.

--delay-seconds waits before each prediction (default 0; 1 s after the 220 W and
160 W no-delay qualifications stopped, 16 Nov 2026). At >=75 C the wrapper also
waits until <=65 C (at most 180 s), as in the A4 mitigation. The
independent GPU guard must still enforce the power limit, 85 C and thermal flags.
Timing is operational only and cannot be used as latency evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from scripts.run_a4_paced import WindowComplete, read_temperature


class CooledEndpoint:
    def __init__(self, pipeline, log, *, duration=None, delay=0., probe=read_temperature,
                 sleeper=time.sleep, clock=time.monotonic):
        self.pipeline, self.log, self.duration, self.probe = pipeline, log, duration, probe
        self.delay = delay
        self.sleep, self.clock = sleeper, clock
        self.started, self.calls = clock(), 0

    def check_window(self):
        if self.duration is not None and self.clock() - self.started >= self.duration:
            raise WindowComplete()

    def temperature(self):
        value = self.probe()
        if not 0 <= value < 85:
            raise RuntimeError('Invalid or unsafe GPU temperature')
        return value

    def predict(self, **kwargs):
        self.check_window()
        before = self.clock()
        if self.delay:
            self.sleep(self.delay)
            self.check_window()
        temperature = self.temperature()
        if temperature >= 75:
            while temperature > 65:
                if self.clock() - before >= 180:
                    raise RuntimeError('GPU cooldown exceeded 180 seconds')
                self.sleep(2.)
                self.check_window()
                temperature = self.temperature()
        waited = self.clock() - before
        result = self.pipeline.predict(**kwargs)
        self.calls += 1
        self.log.write(json.dumps(dict(call=self.calls, utc=datetime.now(timezone.utc).isoformat(),
                                       temperature_before_C=temperature,
                                       wait_seconds=waited)) + '\n')
        self.log.flush()
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--pacing-dir', required=True)
    parser.add_argument('--qualification-seconds', type=float, default=None)
    parser.add_argument('--delay-seconds', type=float, default=0.,
                        help='fixed wait before each prediction (A4 used 1 s)')
    parser.add_argument('--runner', default='experiments.run_peer_review_controls',
                        choices=['experiments.run_peer_review_controls',
                                 'experiments.run_target_variants'])
    parser.add_argument('runner_args', nargs=argparse.REMAINDER,
                        help='arguments after -- go to run_peer_review_controls')
    args = parser.parse_args(argv)
    runner_args = args.runner_args[1:] if args.runner_args[:1] == ['--'] else args.runner_args
    out = Path(args.pacing_dir)
    out.mkdir(parents=True, exist_ok=False)
    state = dict(status='starting', runner=args.runner, runner_args=runner_args,
                 qualification_seconds=args.qualification_seconds,
                 delay_before_prediction_s=args.delay_seconds,
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
            import importlib
            from experiments import run_peer_review_controls as c
            runner = importlib.import_module(args.runner)

            def load(model, revision, device):
                nonlocal endpoint
                pipeline, checkpoint = c._load_chronos(model, revision, device)
                endpoint = CooledEndpoint(pipeline, log, duration=args.qualification_seconds,
                                          delay=args.delay_seconds)
                state['status'] = 'running'
                save()
                return endpoint, checkpoint

            runner.main(runner_args, load_pipeline=load)
            state['status'] = 'grid_complete'
        except WindowComplete:
            state['status'] = 'window_complete'
        except BaseException as exc:
            state.update(status='failed', reason=repr(exc))
            raise
        finally:
            state.update(calls=endpoint.calls if endpoint else 0,
                         elapsed_prediction_window_s=(endpoint.clock() - endpoint.started
                                                      if endpoint else None),
                         finished_utc=datetime.now(timezone.utc).isoformat())
            save()
    print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    main()
