"""Freeze the G2 test plan from completed G1 validation runs (D3/D4).

Per case: identity, the Table 5 validation winner (retained defaults; none when
it is identity) and each G1 family winner from select_on_validation. A spec
that resolves to a model already in the list is run once.
"""
import argparse
import json
from pathlib import Path

from experiments import run_peer_review_controls as r

TABLE5_WINNERS = {  # submission-sinergi/manuscript/selection_data.json
    ('ecl', 24): 'ridge', ('ecl', 48): 'identity', ('etth1', 24): 'mlp', ('etth1', 48): 'mlp',
    ('etth2', 24): 'krr', ('etth2', 48): 'krr', ('ettm1', 24): 'mlp', ('ettm1', 48): 'identity',
    ('ettm2', 24): 'mlp', ('ettm2', 48): 'krr', ('opsd-wind', 24): 'mlp', ('opsd-wind', 48): 'krr',
    ('sdwpf', 24): 'ridge', ('sdwpf', 48): 'krr', ('solar-al', 24): 'krr', ('solar-al', 48): 'krr'}
LAGS = 'lag_hours=1+3+6+24'
FAMILIES = {
    'ridge_tuned': ['ridge:ridge_lam=0.01', 'ridge:ridge_lam=1', 'ridge:ridge_lam=100'],
    'covonly': ['ridge_covonly:ridge_lam=0.01', 'ridge_covonly:ridge_lam=1',
                'ridge_covonly:ridge_lam=100'],
    'lagged': [f'ridge_lagged:{LAGS},ridge_lam=1', f'ridge_lagged:{LAGS},ridge_lam=100'],
    'ewma': ['ewma:alpha=0.2', 'ewma:alpha=0.5', 'ewma:alpha=0.8'],
    'residual': ['residual:alpha=0.2,ridge_lam=1', 'residual:alpha=0.2,ridge_lam=100',
                 'residual:alpha=0.5,ridge_lam=1', 'residual:alpha=0.5,ridge_lam=100']}
DATASETS = ('etth1', 'etth2', 'ettm1', 'ettm2', 'opsd-wind', 'sdwpf', 'solar-al', 'ecl')


def same_model(a, b):
    return (r.parse_adapter(a)[:2] == r.parse_adapter(b)[:2] and
            r.resolved_hyperparameters(a) == r.resolved_hyperparameters(b))


def run_dir(root, dataset, horizon):
    """One folder per dataset, or per dataset and horizon after an interruption."""
    combined = Path(root) / dataset
    return combined if combined.exists() else Path(root) / f'{dataset}_H{horizon}'


def plan(g1_dir):
    out = {}
    for d in DATASETS:
        for h in (24, 48):
            selection = {f: r.select_on_validation(run_dir(g1_dir, d, h), specs)
                         for f, specs in FAMILIES.items()}
            chosen = {f: s['by_horizon'][str(h)]['selected'] for f, s in selection.items()}
            winner = TABLE5_WINNERS[d, h]
            adapters = ['identity']
            for spec in ([] if winner == 'identity' else [winner]) + list(chosen.values()):
                if not any(same_model(spec, a) for a in adapters):
                    adapters.append(spec)
            out[f'{d}_H{h}'] = dict(dataset=d, horizon=h, table5_winner=winner,
                                    family_winners=chosen, adapters=adapters,
                                    validation_scores={f: s['by_horizon'][str(h)]['scores']
                                                       for f, s in selection.items()})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--g1-dir', required=True)
    p.add_argument('--output-prefix', required=True, help='writes <prefix>.json and <prefix>.txt')
    args = p.parse_args()
    result = plan(args.g1_dir)
    with open(args.output_prefix + '.json', 'x') as f:
        json.dump(result, f, indent=2)
    with open(args.output_prefix + '.txt', 'x') as f:
        for case in result.values():
            f.write(' '.join([case['dataset'], str(case['horizon']), *case['adapters']]) + '\n')
    print(sum(len(c['adapters']) for c in result.values()), 'case-adapters planned')


if __name__ == '__main__':
    main()
