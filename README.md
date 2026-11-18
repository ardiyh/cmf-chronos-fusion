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

`data_prep/retained-inputs.zip` contains the six archived experiment inputs
(ETTh1, ETTh2, ETTm1, ETTm2, ECL and SDWPF). The ETT files are the original
benchmark CSVs. The archive contains only the processed HDF5 files for ECL and
SDWPF. For attribution, preprocessing and checksum verification, see
[Dataset preparation](data_prep/DATASETS.md).

`data_prep/reconstruct.sh` downloads the pinned sources, runs the original
preprocessing scripts and records the comparison with the experiment preflight
in `WORK/comparison.json`. Pass a separate directory of experiment inputs in
place of `-` to also compare values and timestamp encodings.

| Dataset | Source | Terms |
|---|---|---|
| ETTh1/h2/m1/m2 | ETDataset, commit 1d16c8f4 (`ETT-small`) | CC BY-ND 4.0 |
| ECL | UCI Machine Learning Repository, dataset 00321 (LD2011_2014) | CC BY 4.0 |
| OPSD-wind | Open Power System Data, time series 2020-10-06 | see OPSD |
| Solar-AL | LSTNet data repository, commit 7f402f18 (`solar_AL.txt.gz`) | see the repository |
| SDWPF | Baidu KDD Cup 2022 files from the [authors' Figshare deposit](https://doi.org/10.6084/m9.figshare.24798654) | CC BY 4.0 |

```sh
python3.12 -m venv .venv-prep
.venv-prep/bin/python -m pip install -r requirements/data_prep.txt
python3.12 -m venv .venv-lgbm
.venv-lgbm/bin/python -m pip install -r requirements/lgbm.txt

# place the two SDWPF files listed in data_prep/DATASETS.md in WORK/raw/sdwpf/ first
PREP_PYTHON="$PWD/.venv-prep/bin/python" PYTHON="$PWD/.venv-lgbm/bin/python" \
  bash data_prep/reconstruct.sh WORK data_prep . - \
  results/peer_review_controls/20261115_chronos2_target_covariates_recovery_1135/CHRONOS2_PREFLIGHT.json
export CMF_DATA_ROOT=WORK/rebuilt/data
```

On x86-64 Linux, the rebuilt ETT CSVs were byte-identical to the experiment
copies and the values in all eight datasets matched bit for bit. The arm64
OPSD rebuild differed from the x86-64 experiment copy by at most 4.55×10⁻¹³ MW.

The four ETT datasets and Solar-AL reproduced the recorded frame hashes on
x86-64. ECL, OPSD and SDWPF used microsecond timestamps in the experiments. The
loader read them as nanoseconds, so the recorded frame hashes include dates in
January 1970. The rebuilt files encode the intended dates in nanoseconds and
therefore have different frame hashes. Evaluations use observation positions
and omit calendar features. SDWPF has gaps of up to 38 h, and preprocessing sets
missing readings to zero.

## Reproducing the experiments

Set `PYTHONPATH=.` and `CMF_DATA_ROOT` as above. Every runner refuses to
overwrite an existing output folder and writes a manifest with hashes of its
inputs, source files and settings.

For Chronos-2 and LightGBM runs on the rebuilt data, generate a local preflight
and pass it as `--preflight`. The original preflight remains in `results/` with
the published runs. On x86-64 the local preflight reproduced the original
origin grids and dataset settings, with the three frame-hash differences above.

```sh
PYTHONPATH=. .venv-lgbm/bin/python scripts/prepare_chronos2_controls.py \
  --retained-dir results/rtx_20260803_test --output WORK/REBUILT_PREFLIGHT.json
```

- **Retained benchmark and validation selection** (Tables 3, 5, 6):
  `experiments/run_revision.py`, `experiments/run_cmf_select.py`,
  `scripts/cmf_select_and_significance.py`, `scripts/sinergi_zero_future_sensitivity.py`.
  The retained runs are `results/rtx_20260803_val_from_june/` (validation summary),
  `results/rtx_20260803_test/` (test per-origin losses and the selection table) and
  `results/sinergi_gpu_retry_20260815_140w/` (all-origin Solar-AL/SDWPF reruns); pass
  them as `--val_dir`/`--test_dir` instead of the default folder names.
- **Scale-consistent ablation** (Table 7, Supplement S1):
  `experiments/run_peer_review_controls.py --adapters identity ridge --modes
  target_units target_standardized fused_standardized --horizons 24
  --repetitions 3 --max-origins 150 --split test ...`, then
  `scripts/peer_review_scale_analysis.py`.
- **Chronos-2 target-only vs past-covariates** (Table 4, Supplement S3):
  `scripts/prepare_chronos2_controls.py`, `experiments/run_chronos2_controls.py`,
  `scripts/chronos2_control_analysis.py`.
- **LightGBM covariate baseline** (Table 8, Supplement S4.1):
  `experiments/run_lgbm_baseline.py --preflight <CHRONOS2_PREFLIGHT.json>
  --protocol docs/protocols/D5_LGBM_PROTOCOL_20261116.md --output-dir RUN/primary`
  (and `--replay --datasets etth1 opsd-wind --output-dir RUN/replay`), then
  `scripts/lgbm_baseline_analysis.py --run-dir RUN --c2-source <Chronos-2 run>`.
- **Validation-tuned mechanism controls, output-side corrector and expanded
  evaluation** (Table 8, Supplements S4.2, S4.3 and S5): protocols in `docs/protocols/`; runners
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

The tests need no model weights, GPU or data and run on CPU; the
`requirements/lgbm.txt` environment covers the whole suite.

## Citation and license

The versioned archive of v1.0.0 is available on [Zenodo (10.5281/zenodo.24330607)](https://doi.org/10.5281/zenodo.24330607).

Please cite the article (see `CITATION.cff`). Code is released under the MIT
License.
