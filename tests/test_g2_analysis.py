"""paired_family must reproduce analyze_cases exactly on a 16-case family."""
import unittest

import numpy as np

from experiments import run_chronos2_controls as c
from scripts.chronos2_control_analysis import analyze_cases
from scripts.g2_analysis import paired_family


class G2AnalysisTests(unittest.TestCase):
    def test_matches_analyze_cases(self):
        rng = np.random.default_rng(0)
        cases, generic = [], []
        for d in c.DATASETS:
            for h in (24, 48):
                origins = list(range(1000, 1000 + 40 * h, h))
                a = {o: float(rng.gamma(2.)) for o in origins}
                b = {o: a[o] + float(rng.normal(.05, .2)) for o in origins}
                cases.append(dict(dataset=d, horizon=h, origins=origins, past_covariates=a, target_only=b))
                generic.append(dict(dataset=d, horizon=h, origins=origins, method=a, reference=b,
                                    method_name='m', reference_name='r'))
        for x, y in zip(analyze_cases(cases), paired_family(generic)):
            for k in ('mean_diff', 'median_diff', 'win_rate', 'wilcoxon_p', 'block_p_worst',
                      'ci_lower', 'ci_upper', 'holm_p', 'holm_block_p'):
                self.assertEqual(x[k], y[k], k)


if __name__ == '__main__':
    unittest.main()
