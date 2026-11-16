# D5 protocol: LightGBM covariate baseline on the frozen grid

Frozen 16 November 2026, before any test origin is scored. Reviewer D5 asks for
a classical covariate model on the same origin grid. This protocol covers the
LightGBM part only. The output-side residual corrector needs Chronos-T5
train/validation forecasts (GPU) and is not part of this run.

## Question and estimand

Within one classical model, does adding the stored past covariates change
point accuracy on the retained test grid? Secondary: how does that classical
covariate model compare with Chronos-2 past-covariates under the same q0.5
point estimator on the same origins?

## Grid, data and information set

- Eight datasets, H ∈ {24, 48}: 16 cases and 2,188 origin–horizon pairs. Test
  origins are those of `CHRONOS2_PREFLIGHT.json` (start int(0.8n), stride H, at
  most 150 evenly spaced, context 512). The runner refuses to start unless each
  frame hash, origin hash, target and seasonality equals the preflight.
- Frames come from the unchanged loader (`src/cmf_core.py`, SHA-256 03c8e54f…)
  on the RTX x86 host with pandas 3.0.5. The ETT CSV parser output is
  platform-dependent (see A6 note below), so the run uses that host.
- At origin t a row may use only observations in [t−512, t). Target-only sees
  the target; past-covariates also sees every stored non-target channel, as
  Chronos-2 past-covariates did. No future covariates, calendar or position
  features.

## Model

- Direct multi-horizon LightGBM 4.6.0: one model per quantile level with the
  lead h as a feature. Levels are the nine Chronos-2 levels 0.05, 0.1, 0.2,
  0.25, 0.5, 0.75, 0.8, 0.9, 0.95. The q0.5 prediction is the point forecast.
- Instance normalization per origin and channel by the context mean and std;
  the std is floored at 1% of that channel's training-split std (1.0 for a
  constant column). Predictions are mapped back to target units.
- Target features: lags 1–24, three most recent same-phase values (seasonality
  s from the loader), mean and std of the last s steps. Each covariate adds its
  last value, mean of the last s steps and most recent same-phase value.
- Fixed: learning rate 0.05, feature/bagging fraction 0.8, bagging every
  iteration, L2 1.0, 255 bins, deterministic row-wise training, 8 threads,
  seed 2026, at most 2,000 rounds, early-stopping patience 100.

## Splits and tuning (validation only)

- Chronological 60/20/20 split of the loader. Training origins: stride 1 with
  complete horizons inside the training split, at most 3,000 evenly spaced.
  Validation origins: stride H inside the validation split, all eligible.
- Grid, tuned separately per case and variant on q0.5: (num_leaves,
  min_child_samples) ∈ {(15,200), (15,20), (63,200), (63,20)}. Early stopping on
  validation pinball loss in normalized units; selection by mean validation
  MASE in target units; ties go to the earlier configuration.
- Each other quantile level uses the chosen configuration with its own early
  stopping on validation. Every level is then refit on train+validation origins
  (labels before the test split) for its validation-selected number of rounds.
  No test value enters fitting, selection or stopping.

## Metrics and inference

- `score()` of the Chronos-2 control: MASE with the 512-step in-context
  seasonal-naive denominator, MAE, raw doubled-pinball WQL over 0.05, 0.1, 0.2,
  0.5, 0.8, 0.9, 0.95, and 50%/90% coverage/width. Quantiles are not sorted or
  repaired; adjacent crossings are counted and reported.
- Primary family (16 contrasts, negative favors covariates): past-covariates
  minus target-only LightGBM MASE, analyzed with `analyze_cases` of
  `scripts/chronos2_control_analysis.py` (two-sided signed-rank with Holm16,
  worst-offset block sign-flip with separate block Holm16, moving-block 95% CI
  of the mean difference, median difference and win rate).
- Secondary family (16 contrasts, separate Holm): LightGBM past-covariates minus
  Chronos-2 past-covariates q0.5 MASE on the same origins, from the audited C2
  losses. Both use q0.5; this is not an architecture or training-data matched
  comparison, since LightGBM learns from the training split and Chronos-2 is
  zero-shot.
- Descriptive only, no new tests: retained Chronos-T5 identity/CMF mean-point
  MASE, with the estimator mismatch labelled.
- All 16 cases are reported whatever the direction. Non-rejection is not
  equivalence.

## Determinism and audit

- Two datasets (ETTh1, OPSD) are rerun into a separate replay directory; q0.5
  and all quantile arrays must be bitwise identical. Replays never enter
  summaries or tests.
- Each manifest records the frame/origin/protocol hashes, git commit, package
  versions, grid scores, chosen configuration, rounds per level, row counts and
  timings. The analysis refuses incomplete or hash-mismatched runs.

## Scope limits

This does not provide an output-side residual corrector, ARIMAX, multi-target,
uncapped or day-ahead evaluation, latency, or a matched median comparison with
Chronos-T5. ARIMAX is not run: with past covariates only it needs future
covariate values or a separate covariate forecaster, which would add a second
modelling choice outside this protocol.

## A6 note found while preparing this run

The loader reads the ETT CSVs with pandas' default float parser, which is not
correctly rounded and differs by platform: the same bytes give different
float64 values on x86/pandas 3.0.5 and arm64/pandas 2.2.2 (at most 7.1e-15,
in 7,557 of 121,940 ETTh1 cells). A correctly rounded parse
(`float_precision='round_trip'`, Python float, pyarrow) agrees across both.
Historical frames are the x86 values; a future dataset release should pin a
correctly rounded parser and record it.
