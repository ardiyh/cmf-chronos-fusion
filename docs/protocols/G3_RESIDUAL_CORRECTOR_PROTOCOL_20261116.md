# G3 protocol: output-side residual corrector (D5)

Frozen 16 November 2026, while G1/G3 validation inference runs and before any
G2 test output exists. CPU only.

## Question

Does a corrector that adjusts frozen Chronos-T5 identity forecasts after
inference, using past covariates, improve test accuracy over the uncorrected
identity forecast? A target-only corrector isolates the covariate contribution.

## Data

- Fitting: G3 identity forecasts on all eligible validation origins (split
  `val`, stride H, 50 samples, seed rule of the runner). No test value is used.
- Evaluation: the G2 identity forecasts (repetition 0, not the replay) on the
  retained capped test grid, 2,188 origin–horizon pairs.
- Each run's frame hash must equal the Chronos-2 preflight hash.

## Corrector

- Per case (dataset, H), rows are (origin, h). Normalize by the context target
  mean μ and std σ (floored at 1% of the training-split std).
- Label: (y − m̂)/σ, where m̂ is the identity sample median at step h.
- Features: h; the identity median, q0.1 and q0.9 at step h, normalized; target
  lags 1–24 and mean/std of the last s steps (s from the loader); for the
  covariate corrector, each covariate's last value and mean of the last s steps,
  normalized by its own context statistics. Only values before the origin.
- Model: ridge regression with intercept, features standardized on the fitting
  rows. λ ∈ {0.1, 1, 10, 100, 1000} chosen by fitting on the first 70% of
  validation origins (time order) and scoring median-point MASE on the last
  30%; ties to larger λ. Refit on all validation origins with the chosen λ.
- Corrected point forecast: m̂ + σ·ŷ. The same shift moves every sample
  quantile, so WQL and coverage are reported descriptively for the shifted
  distribution.

## Inference

Same machinery as the other 16-case families (two-sided signed-rank with
Holm16, worst-offset block sign-flip with block Holm16, moving-block 95% CI of
the mean difference, median difference, win rate), median-point MASE:

- primary: covariate corrector minus identity;
- secondary: covariate corrector minus target-only corrector.

All cases are reported whatever the direction; non-rejection is not equivalence.

## Limits

The corrector learns from validation origins only (72–580 per case), so it is a
small-sample, global linear correction. It is not tuned on test origins, does not
use future covariates and is not a claim about the best possible corrector.
