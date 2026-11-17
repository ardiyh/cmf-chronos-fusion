# cmf-chronos-fusion

Code, data-preparation scripts and per-origin results for the article
**"Pre-tokenization multivariate fusion for energy forecasting with a frozen
scalar-interface Chronos-T5"** (A. Y. Hilmi, E. Winarko, M. E. Wibowo and
A. Dharmawan, Universitas Gadjah Mada; submitted to *SINERGI*).

Chronos Multivariate Fusion (CMF) fits a context-local adapter that compresses a
multivariate history into the scalar history accepted by a frozen Chronos-T5
endpoint. The repository contains every experiment reported in the article and
its supplements, together with the controls added during peer review.

## Layout

| Path | Contents |
|---|---|
| `src/` | data loader, adapters, metrics (`cmf_core.py`, `paper_metrics.py`) |
| `experiments/` | runners: retained benchmark and selection, peer-review controls (Chronos-T5), Chronos-2 controls, LightGBM baseline, target/horizon variants, guarded latency |
| `scripts/` | statistics (paired tests, Holm, block sign-flip, moving-block intervals), analyses, GPU guard |
| `tests/` | CPU unit tests (no model weights or GPU needed) |
| `data_prep/` | download-and-rebuild scripts for the eight datasets |
| `results/` | per-origin losses, manifests and analysis outputs (CSV/JSON) |
| `docs/protocols/` | experiment protocols frozen before test scoring |

## Environments

All reported runs used one x86-64 Linux host (NVIDIA RTX 3070, CUDA 12.1,
Python 3.12). Three environments were used; exact versions are pinned in
`requirements/`.

| Environment | Used for | Key packages |
|---|---|---|
| `requirements/t5.txt` | Chronos-T5 benchmark and controls | torch 2.4.1, chronos-forecasting 1.5.3, transformers 4.49.0 |
| `requirements/chronos2.txt` | Chronos-2 controls | torch 2.5.1, chronos-forecasting 2.3.1 |
| `requirements/lgbm.txt` | LightGBM baseline and CPU analyses | lightgbm 4.6.0, numpy 2.5.2, pandas 3.0.5, scipy 1.18.1 |

Checkpoints: `amazon/chronos-t5-base` revision
`ad294eaacead15db499b740ea4122266dd2a81a2`, `amazon/chronos-2` revision
`29ec3766d36d6f73f0696f85560a422f50e8498c`.

**Platform note.** pandas' default CSV float parser is not correctly rounded and
its output depends on the platform: the ETT files parse to values up to
7×10⁻¹⁵ apart on x86-64 and arm64. Frame hashes recorded in the results are
those of the x86-64 parse, so hash-exact replay of the ETT frames needs the same
platform. Results on other architectures may differ at that numerical level.

## Data

The raw data are not redistributed. `data_prep/reconstruct.sh` downloads pinned
sources, rebuilds the processed files with the preprocessing scripts used in the
article and compares each rebuilt frame with the recorded frame hashes.

| Dataset | Source | Terms |
|---|---|---|
| ETTh1/h2/m1/m2 | ETDataset, commit 1d16c8f4 (`ETT-small`) | see the ETDataset repository |
| ECL | UCI Machine Learning Repository, dataset 00321 (LD2011_2014) | CC BY 4.0 |
| OPSD-wind | Open Power System Data, time series 2020-10-06 | see OPSD |
| Solar-AL | LSTNet data repository, commit 7f402f18 (`solar_AL.txt.gz`) | see the repository |
| SDWPF | Baidu KDD Cup 2022 (`sdwpf_245days_v1.csv`, registration required) | KDD Cup terms |

```sh
# place sdwpf_245days_v1.csv and the turbine location file in WORK/raw/sdwpf/ first
bash data_prep/reconstruct.sh WORK data_prep . WORK/rebuilt/data \
  results/peer_review_controls/20261115_chronos2_target_covariates_recovery_1135/CHRONOS2_PREFLIGHT.json
export CMF_DATA_ROOT=WORK/rebuilt/data
```

The rebuilt values match the experiment frames exactly (ETT byte-identical
CSVs; ECL, SDWPF and Solar-AL bitwise; OPSD within 5×10⁻¹³ MW). The experiment
copies of ECL, OPSD and SDWPF stored microsecond timestamps that the loader reads
as nanoseconds, so their recorded frame hashes include dates in January 1970;
all evaluations are positional and use no calendar feature. SDWPF has gaps of up
to 38 h, and missing readings are set to zero by its preprocessing.

## Reproducing the experiments

Set `PYTHONPATH=.` and `CMF_DATA_ROOT` as above. Every runner refuses to
overwrite an existing output folder and writes a manifest with hashes of its
inputs, source files and settings.

- **Retained benchmark and validation selection** (Tables 3, 5, 6):
  `experiments/run_revision.py`, `experiments/run_cmf_select.py`,
  `scripts/cmf_select_and_significance.py`, `scripts/sinergi_zero_future_sensitivity.py`.
- **Scale-consistent ablation** (Table 7, Supplement S1):
  `experiments/run_peer_review_controls.py --adapters identity ridge --modes
  target_units target_standardized fused_standardized --horizons 24
  --repetitions 3 --max-origins 150 --split test ...`, then
  `scripts/peer_review_scale_analysis.py`.
- **Chronos-2 target-only vs past-covariates** (Table 4, Supplement S3):
  `scripts/prepare_chronos2_controls.py`, `experiments/run_chronos2_controls.py`,
  `scripts/chronos2_control_analysis.py`.
- **LightGBM covariate baseline** (Supplement S4.1):
  `experiments/run_lgbm_baseline.py --preflight <CHRONOS2_PREFLIGHT.json>
  --protocol docs/protocols/D5_LGBM_PROTOCOL_20261116.md --output-dir RUN/primary`
  (and `--replay --datasets etth1 opsd-wind --output-dir RUN/replay`), then
  `scripts/lgbm_baseline_analysis.py --run-dir RUN --c2-source <Chronos-2 run>`.
- **Validation-tuned mechanism controls, output-side corrector and expanded
  evaluation**: protocols in `docs/protocols/`; runners
  `experiments/run_peer_review_controls.py` and `experiments/run_target_variants.py`,
  test plan `scripts/plan_g2.py`, analyses `scripts/g2_analysis.py`,
  `scripts/residual_corrector.py` and `scripts/g4_analysis.py`. Validation tuning and
  validation identity runs are in `results/peer_review_controls/20261116_g1_tol/`; test runs,
  the frozen test plan and the three analyses are in `results/peer_review_controls/20261116_g2_g4_tol/`.
- **Latency**: `experiments/measure_guarded_latency.py` (synchronized, warmed-up,
  repeated timing under a strict thermal guard). Both attempts reported in the article
  were stopped by that guard before completion, so no latency results are included.

GPU runs were wrapped by `scripts/run_with_gpu_guard.py` (telemetry, power and
thermal guard) and `scripts/run_controls_cooled.py` (fixed delay and cooldown).
These change timing only; forecasts are deterministic given the seeds.

## Results

`results/` holds per-origin losses, manifests and analysis outputs for every
reported experiment. Raw forecast sample arrays are not included because of
their size. Commit identifiers inside manifests refer to the development
repository from which this package was assembled.

## Tests

```sh
PYTHONPATH=. python -m unittest discover -s tests
```

## Citation and license

Please cite the article (see `CITATION.cff`). Code is released under the MIT
License.
