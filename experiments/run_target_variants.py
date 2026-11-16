"""Run the frozen controls runner on another target channel or past 64 steps (D6).

--target swaps the forecast target to another stored channel; every other
channel, including the original target, becomes a covariate. --rollout lets
Chronos-T5 forecast beyond its 64-step window with the pipeline's own
autoregressive rollout. The controls runner itself is unchanged.
"""
import argparse

from experiments import run_peer_review_controls as r


def main(argv=None, *, load_pipeline=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--target', default=None)
    parser.add_argument('--rollout', action='store_true')
    args, rest = parser.parse_known_args(argv)
    if args.rollout:
        r.SAMPLING_ARGS['limit_prediction_length'] = False

    def load(name):
        df, cols, target, seasonality = r.legacy.cmf.load_dataset(name)
        if args.target is None:
            return df, cols, target, seasonality
        if args.target not in cols or args.target == target:
            raise ValueError(f'{args.target!r} is not another stored channel of {name}')
        return df, cols, args.target, seasonality

    return r.main(rest, load_data=load, load_pipeline=load_pipeline)


if __name__ == '__main__':
    main()
