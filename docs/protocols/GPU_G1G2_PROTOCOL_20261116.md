# G1/G2 protocol: validation-tuned mechanism controls and their test (D3, D4, C7, D7, D8)

Frozen 16 November 2026 before any G1 validation or G2 test inference.
Scope approved by the author: the lean plan in `GPU_LEAN_PLAN_20261116.md`,
pragmatic but covering each reviewer request.

## Common settings

- Runner `experiments/run_peer_review_controls.py` (unchanged), Chronos-T5-base
  revision ad294eaacead15db499b740ea4122266dd2a81a2, float32, context 512,
  50 samples with the runner's sampling arguments, mode `target_units` (the
  retained CMF mode), fit seed 42, base seed 2026. Every method at one
  (dataset, target, horizon, origin, repetition) shares one sampler seed;
  identity is replayed with the same seed to verify determinism.
- Point forecast: sample median (`MASE`). The sample mean (`MASE_mean`), WQL and
  50%/90% coverage/width come from the same 50 samples.
- GPU operation: power limit 220 W (author decision), no fixed delay before
  predictions (author decision), wrapper `scripts/run_controls_cooled.py` waits
  at ≥75 C until ≤65 C. The independent guard `scripts/run_with_gpu_guard.py
  --power-limit 220 --max-temperature 85` stops the process group on any thermal
  or power-brake flag, 85 C, a changed power limit or GPU access errors.
  Datasets run serially; a guard stop halts the sequence with no automatic retry.
  A 900 s qualification window of the first G1 dataset precedes the full run;
  its partial output is not used.

## G1: validation tuning (D3, D4)

- Split `val`, horizons 24 and 48, 50 evenly spaced validation origins per case
  (`--max-origins 50`), one repetition, all eight datasets and targets as in the
  retained benchmark.
- Candidates, in tie-break order within each family:
  - ridge, all channels (D3 tuning): `ridge:ridge_lam=` 0.01, 1, 100
  - covariates-only ridge (D3): `ridge_covonly:ridge_lam=` 0.01, 1, 100
  - lagged covariates (D4): `ridge_lagged:lag_hours=1+3+6+24,ridge_lam=` 1, 100
  - target smoothing without covariates (D4): `ewma:alpha=` 0.2, 0.5, 0.8
  - input residual (D3): `residual:alpha=0.2|0.5,ridge_lam=1|100` in the order
    (0.2,1), (0.2,100), (0.5,1), (0.5,100)
  - plus `identity`.
- Selection per case and family: `select_on_validation` (lowest mean validation
  median-point MASE, ties to the earlier candidate). Test results do not feed
  back into selection.
- G3 input, same phase: identity only, split `val`, all eligible validation
  origins (`--all-eligible`), one repetition. Used later only to fit the
  output-side residual corrector under its own protocol.

## G2: test (D3, D4, C7, D7, D8)

- Test grid: the retained capped grid, identical to the Chronos-2 preflight
  origins (2,188 origin–horizon pairs); `--max-origins 150`, split `test`.
- Methods per case: identity; the CMF adapter that won validation in Table 5
  with its retained default hyperparameters (none extra when the winner is
  identity: ECL H48, ETTm1 H48); and the five G1 family winners. A spec chosen
  twice runs once. One repetition, plus the same-seed identity replay.
- D7 noise floor: a separate identity-only test run with base seed 2027 on the
  same grid. Identity(seed 2027) minus identity(seed 2026) measures seed-to-seed
  sampling noise; the same-seed replay measures determinism.
- Inference, per family of 16 contrasts with the Chronos-2 control machinery
  (two-sided signed-rank with Holm16, worst-offset block sign-flip with block
  Holm16, moving-block 95% CI of the mean difference, median difference, win rate):
  - each method minus identity, median-point MASE: CMF-selected, ridge-tuned,
    covariates-only, lagged, EWMA, residual (six separate families);
  - D4 key contrast: CMF-selected minus best EWMA (separate family);
  - noise floor: identity seed 2027 minus seed 2026 (separate family).
  Mean-point MASE and WQL are reported descriptively for every method (C7/D8).
- All cases are reported whatever the direction; non-rejection is not
  equivalence. Families stay separate from A4, C2 and the historical benchmark.

## Operational amendment, 16 November 2026 10:40 WIB

The first 220 W qualification was stopped by the guard after 56.5 s (SW Thermal
Slowdown; see `g1_220w_qual_stop_20261116/`). No G1 output was produced. The
author chose a 160 W power limit with no fixed delay and the same cooldown; the
guard runs with `--power-limit 160`. Output goes to a new folder
`20261116_g1_160w/`. Scientific design, candidates and inference are unchanged.

## Operational amendment, 16 November 2026 11:35 WIB

The 160 W qualification without a fixed delay was also stopped (148.4 s, SW
Thermal Slowdown; `g1_160w_qual_stop_20261116/`). As pre-authorized by the
author for this case, runs use 160 W with the A4 1 s pre-prediction delay
(`run_controls_cooled.py --delay-seconds 1`, commit 9ca7750) and the same
cooldown and guard, after a new 900 s qualification. Output goes to
`20261116_g1_160w_d1/`. Scientific design, candidates and inference are unchanged.

## Operational amendment, 16 November 2026 11:40 WIB

The 1 s qualification was also stopped (±153 s; `g1_160w_d1_qual_stop_20261116/`).
Author decision: 160 W with a 3 s pre-prediction delay after a new 900 s
qualification, cooldown and guard unchanged. Each dataset launch reads the delay
from `~/cmf_gpu_delay_seconds` on the RTX host, so a shorter delay can be resumed
at a dataset boundary after cooling improves; the pacing status of every run
records the delay used. Delays change timing only, not forecasts. Output goes
to `20261116_g1_160w_d3/` and `20261116_g2_g4_d3/`. Scientific design is unchanged.

## Operational amendment, 16 November 2026 11:55 WIB: tolerant SW-thermal guard

The 3 s qualification was stopped after 45.8 s at core 53 C (SW Thermal
Slowdown at 50 C, 65 W, fan 30%, utilization 0% in the stopping sample). A
read-only 90 s idle probe showed no SW thermal sample, so the flag follows load.
Four stops in a row (`g1_*_qual_stop_20261116/`, this one in
`g1_160w_d3_qual_stop_20261116/`) left no workable strict configuration.

Author decision for the accuracy runs G1–G4: the guard stops when SW Thermal
Slowdown stays active continuously for 60 s (`run_with_gpu_guard.py
--sw-thermal-grace-seconds 60`, commit cabe91e); HW thermal slowdown, HW power
brake, core ≥85 C, a changed power limit and GPU access errors still stop at
once, and every SW sample is logged in the telemetry. The driver's own clock
capping and the card's 95/98 C protections stay in force. Delay returns to 1 s
(from `~/cmf_gpu_delay_seconds`), cooldown unchanged, new 900 s qualification.
Clock capping changes timing only; seeds and forecasts are unaffected. C8
latency keeps the strict guard (grace 0), since capping would bias timing.
Output goes to `20261116_g1_tol/` and `20261116_g2_g4_tol/`.

## Not covered here

Output-side residual corrector (G3, own protocol after G1), D6 (G4, own
protocol), C8 latency (existing reviewed protocol, run alone), ARIMAX (see D5
protocol). Partial or qualification outputs never enter inferential tables.
