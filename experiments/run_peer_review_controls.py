"""Sequential Chronos-T5 controls for SINERGI peer review.

Uses the retained context-fitting implementation, but isolates fitting and
sampling RNGs. Global RNG restoration is process-local: do not run this evaluator
in threads. CPU tests inject a small endpoint double; weights are loaded only by
the CLI, when a run is requested:

    python -m experiments.run_peer_review_controls --help
"""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import random
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from experiments import run_revision as legacy


MODES = ("target_units", "target_standardized", "fused_standardized")
LEGACY_KINDS = ("identity", "linear", "ridge", "pcr", "krr", "mlp")
VARIANTS = ("", "covonly", "targetonly", "lagged")
NEW_KINDS = {"ewma": ("alpha",), "residual": ("alpha", "ridge_lam")}  # required parameters
SAMPLING_ARGS = {"temperature": 1.0, "top_k": 50, "top_p": 1.0,
                 "limit_prediction_length": True}
ARRAY_FIELDS = ("samples", "y_true", "insample", "input_history", "mean_pred",
                "median_pred", "q_preds", "quantile_levels")
LOSS_FIELDS = ("dataset", "target", "horizon", "origin_idx", "origin_time", "adapter",
               "mode", "repetition", "replay", "fit_seed", "sampler_seed", "MASE",
               "MASE_mean", "MAE", "WQL", "target_abs_mean", "coverage_50", "width_50",
               "coverage_90", "width_90", "replay_max_abs_diff", "prediction_file")
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("experiments/run_peer_review_controls.py", "experiments/run_revision.py",
           "src/cmf_core.py")


@contextmanager
def isolated_rng(seed):
    """Save/restore Python, NumPy, and Torch CPU/all visible CUDA RNG states."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def sampler_seed_for_origin(*, dataset, target, horizon, origin, repetition, base_seed):
    """Stable v1 pairing key: deliberately excludes adapter, mode and fit seed."""
    key = ["cmf-peer-review-v1", dataset, str(target), int(horizon), int(origin),
           int(repetition), int(base_seed)]
    digest = hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _parse_value(text):
    if "+" in text:
        return tuple(float(v) for v in text.split("+"))
    try:
        return int(text)
    except ValueError:
        return float(text)


def parse_adapter(spec):
    """'krr_covonly:krr_lam=0.01' -> ('krr', 'covonly', {'krr_lam': 0.01}).

    covonly drops the target channel from the adapter input (it stays the fitting
    label); targetonly keeps only the target channel; lagged feeds covariates at
    lag 0 and at each of lag_hours, past values only. ewma smooths the target
    history; residual adds a covariate ridge fit to the target minus its ewma.
    """
    name, _, params = spec.partition(":")
    base, _, variant = name.partition("_")
    try:
        kwargs = {key: _parse_value(value) for key, value in
                  (item.split("=") for item in params.split(",") if item)}
    except ValueError:
        raise ValueError(f"Malformed adapter parameters in {spec!r}") from None
    if "lag_hours" in kwargs:
        lags = kwargs["lag_hours"]
        kwargs["lag_hours"] = tuple(float(h) for h in (lags if isinstance(lags, tuple) else (lags,)))
    if base in NEW_KINDS:
        variants, allowed, required = ("",), set(NEW_KINDS[base]), set(NEW_KINDS[base])
    elif base in LEGACY_KINDS:
        variants = ("",) if base == "identity" else VARIANTS
        allowed = {k for k in legacy.ADAPTER_CONFIGS[base] if not k.endswith("_device")}
        required = {"lag_hours"} if variant == "lagged" else set()
        allowed |= required
    else:
        raise ValueError(f"Unknown adapter {spec!r}")
    if variant not in variants or not set(kwargs) <= allowed or not required <= set(kwargs):
        raise ValueError(f"Invalid variant or parameters in {spec!r}")
    if "alpha" in kwargs and not 0 < kwargs["alpha"] <= 1:
        raise ValueError("alpha must lie in (0, 1]")
    return base, variant, kwargs


def resolved_hyperparameters(spec):
    base, _, kwargs = parse_adapter(spec)
    if base in NEW_KINDS:
        return kwargs
    return {k: v for k, v in (legacy.ADAPTER_CONFIGS[base] | kwargs).items()
            if not k.endswith("_device")}


def _ewma(values, alpha):
    out = np.empty(len(values), dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _standardize(values):
    """Per-channel context z-score, as in the retained implementation."""
    values = values.astype(np.float32)
    return (values - values.mean(axis=0, keepdims=True)) / (values.std(axis=0, keepdims=True) + 1e-8)


def _lagged(df, origin, context_len, covariates, lags):
    start = origin - context_len - lags[-1]
    if start < 0:
        raise ValueError("Lagged covariates need history before the context")
    values = df[covariates].to_numpy(np.float32)[start:origin]
    if not np.isfinite(values).all():
        raise ValueError("Lagged covariates must be finite")
    end = len(values)
    return np.concatenate([values[end - context_len - lag:end - lag] for lag in lags], axis=1)


def fit_history(*, df, origin, adapter_kind, context_len, pred_len, feature_cols, target_col,
                seasonality, device_map, fit_seed, adapter_kwargs=None):
    """(fitted context in target units [1, L], realized horizon, observed context).

    The adapter output z is fitted to the context-standardized target and mapped
    back by mu + sigma * z, as in the retained implementation.
    """
    base, variant, kwargs = parse_adapter(adapter_kind)
    kwargs.update(adapter_kwargs or {})
    covariates = [c for c in feature_cols if c != target_col]
    if base in LEGACY_KINDS and variant != "lagged":
        features = {"": feature_cols, "covonly": covariates, "targetonly": [target_col]}[variant]
        return legacy._prepare_single_origin(
            df, origin, base, context_len, pred_len, features, target_col, device_map,
            fit_seed, legacy.ADAPTER_CONFIGS[base] | kwargs, include_zero_future=True)
    if not covariates and base != "ewma":
        raise ValueError("This control needs at least one covariate")
    insample = df[target_col].to_numpy()[origin - context_len:origin]
    y_true = df[target_col].to_numpy()[origin:origin + pred_len]
    target = insample.astype(np.float32)
    mu, sigma = float(target.mean()), float(target.std() + 1e-8)
    target_z = (target - mu) / sigma
    if base == "ewma":
        z = _ewma(target_z, kwargs["alpha"])
    elif base == "residual":
        smooth = _ewma(target_z, kwargs["alpha"])
        X = _standardize(df[covariates].to_numpy()[origin - context_len:origin])
        fit = legacy.cmf.fit_ridge(X, target_z - smooth, lam=kwargs["ridge_lam"])
        z = smooth + X @ fit["w"] + fit["bias"]
    else:
        steps = [round(h * seasonality / 24) for h in kwargs.pop("lag_hours")]
        if min(steps) < 1:
            raise ValueError("Every lag must be at least one step long")
        X = _standardize(_lagged(df, origin, context_len, covariates, sorted({0, *steps})))
        params = legacy.ADAPTER_CONFIGS[base] | kwargs
        if base == "mlp":
            params["mlp_device"] = device_map
        adapter = legacy.cmf.fit_adapter(X, target_z, kind=base, **params)
        z = legacy.cmf.apply_adapter(X[None], adapter)[0]
    fitted = mu + sigma * np.asarray(z, dtype=np.float64)
    return torch.tensor(fitted[None], dtype=torch.float32), y_true, insample


def evaluate_control_origin(*, df, origin, adapter_kind, context_len, pred_len,
                            feature_cols, target_col, seasonality, device_map,
                            pipeline, fit_seed, sampler_seed, mode="target_units",
                            num_samples=50, adapter_kwargs=None):
    """One origin, batch size one; restore every path before computing metrics.

    target_standardized undoes the target affine map on the *retained fitted
    history*. Thus it tests placement of that affine map (including float32
    roundoff), without changing the fitting objective or fitting implementation.
    """
    if mode not in MODES:
        raise ValueError("Unknown control mode")
    if torch.device(device_map).type not in ("cpu", "cuda"):
        raise ValueError("Controls currently support CPU or CUDA only")
    if context_len <= 1 or origin < context_len:
        raise ValueError("A complete context is required")
    if pred_len < 1 or origin + pred_len > len(df):
        raise ValueError("A complete horizon is required")
    if not 0 < seasonality < context_len or num_samples < 2:
        raise ValueError("Require 0 < seasonality < context length and at least 2 samples")
    columns = list(dict.fromkeys([*feature_cols, target_col]))
    if not np.isfinite(df.iloc[origin-context_len:origin][columns].to_numpy()).all():
        raise ValueError("Context must be finite")
    with isolated_rng(fit_seed):
        fitted, y_true, insample = fit_history(
            df=df, origin=origin, adapter_kind=adapter_kind, context_len=context_len,
            pred_len=pred_len, feature_cols=feature_cols, target_col=target_col,
            seasonality=seasonality, device_map=device_map, fit_seed=fit_seed,
            adapter_kwargs=adapter_kwargs)
    if not np.isfinite(y_true).all() or not torch.isfinite(fitted).all():
        raise ValueError("Target and fitted history must be finite")

    offset, scale = 0.0, 1.0
    if mode == "target_standardized":
        target = insample.astype(np.float32)
        offset, scale = float(target.mean()), float(target.std() + 1e-8)
    elif mode == "fused_standardized":
        offset, scale = float(fitted.mean()), float(fitted.std() + 1e-5)
    input_history = (fitted - offset) / scale

    # Reset only after fitting has completed, using an origin-specific seed
    # shared by every adapter/control. No fitting workers run concurrently.
    with isolated_rng(sampler_seed), torch.inference_mode():
        output = pipeline.predict(context=input_history, prediction_length=pred_len,
                                  num_samples=num_samples, **SAMPLING_ARGS)
    if tuple(output.shape) != (1, num_samples, pred_len):
        raise ValueError("Endpoint must return [1, samples, horizon]")
    samples = output[0].detach().cpu().numpy().astype(np.float64)
    samples = offset + scale * samples
    if not np.isfinite(samples).all():
        raise ValueError("Restored forecast samples must be finite")
    median_pred, mean_pred = np.median(samples, axis=0), samples.mean(axis=0)
    levels = np.array(legacy.QUANTILE_LEVELS)
    quantiles = np.quantile(samples, levels, axis=0).T
    result = {
        "origin_idx": int(origin), "adapter": adapter_kind, "mode": mode,
        "target": str(target_col), "features": list(feature_cols),
        "context_length": context_len, "horizon": pred_len, "seasonality": seasonality,
        "fit_seed": int(fit_seed), "sampler_seed": int(sampler_seed),
        "num_samples": num_samples, "transform_offset": offset, "transform_scale": scale,
        "point_estimator": "sample_median", "sampling_args": SAMPLING_ARGS,
        "samples": samples, "y_true": y_true, "insample": insample,
        "input_history": input_history[0].cpu().numpy(),
        "median_pred": median_pred, "mean_pred": mean_pred,
        "q_preds": quantiles, "quantile_levels": levels,
        "MASE": legacy.cmf.compute_mase(y_true, median_pred, insample, seasonality),
        "MASE_mean": legacy.cmf.compute_mase(y_true, mean_pred, insample, seasonality),
        "MAE": float(np.abs(y_true - median_pred).mean()),
        "WQL": legacy.weighted_quantile_loss_full(y_true, quantiles, levels),
        "target_abs_mean": float(np.abs(y_true).mean()),
    }
    for label, bounds in [("50", (0.25, 0.75)), ("90", (0.05, 0.95))]:
        lower, upper = np.quantile(samples, bounds, axis=0)
        result[f"coverage_{label}"] = float(((lower <= y_true) & (y_true <= upper)).mean())
        result[f"width_{label}"] = float((upper - lower).mean())
    return result


def write_prediction(path, result):
    """Persist target-unit paths, quantiles, context and provenance without pickle."""
    metadata = {k: v for k, v in result.items() if k not in ARRAY_FIELDS}
    arrays = {k: result[k] for k in ARRAY_FIELDS}
    # Exclusive create protects evidence from silent replacement on a rerun.
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays, metadata_json=json.dumps(metadata))


def frame_sha256(df, columns):
    """Content hash of exactly the values, column names and index the grid reads."""
    digest = hashlib.sha256(json.dumps([str(c) for c in columns]).encode())
    digest.update(np.ascontiguousarray(df[columns].to_numpy(dtype=np.float64)).tobytes())
    digest.update(json.dumps([str(i) for i in df.index]).encode())
    return digest.hexdigest()


def environment_record():
    """Software state of *this* run; never a stand-in for historical versions."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                  text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None

    packages = {}
    for name in ("torch", "numpy", "pandas", "scipy", "transformers", "chronos-forecasting"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    head, status, diff = git("rev-parse", "HEAD"), git("status", "--porcelain"), git("diff", "HEAD")
    cuda = torch.cuda.is_available()
    return {
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "torch_cuda": torch.version.cuda, "cuda_available": cuda,
        "cuda_devices": [torch.cuda.get_device_name(i)
                         for i in range(torch.cuda.device_count())] if cuda else [],
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "source_sha256": {p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest()
                          for p in SOURCES},
        "git_head": head.strip() if head else None,
        "git_status_porcelain": status.splitlines() if status is not None else None,
        "git_diff_head_sha256": hashlib.sha256(diff.encode()).hexdigest()
                                if diff is not None else None,
    }


def identity_noise_summary(rows, endpoint_label):
    """Same-seed replay discrepancy and independent-repetition spread, kept apart."""
    identity = [r for r in rows if (r["adapter"], r["mode"]) == ("identity", "target_units")]
    by_horizon = {}
    for h in sorted({r["horizon"] for r in identity}):
        first = {(r["origin_idx"], r["repetition"]): r for r in identity
                 if r["horizon"] == h and r["replay"] == 0}
        replays = [r for r in identity if r["horizon"] == h and r["replay"] == 1]
        case = [float(np.mean([r["MASE"] for (_, rep), r in first.items() if rep == k]))
                for k in sorted({rep for _, rep in first})]
        by_horizon[str(h)] = {
            "same_seed_replay": {
                "n_pairs": len(replays),
                "max_abs_sample_diff": max(r["replay_max_abs_diff"] for r in replays),
                "max_abs_mase_diff": max(
                    abs(r["MASE"] - first[r["origin_idx"], r["repetition"]]["MASE"])
                    for r in replays),
            },
            "independent_repetitions": {
                "n_repetitions": len(case),
                "case_mean_mase": case,
                "std_case_mean_mase": float(np.std(case, ddof=1)) if len(case) > 1 else None,
                "relative_range_pct": float(100 * (max(case) - min(case)) / np.mean(case)),
            },
        }
    return {"endpoint_label": endpoint_label,
            "scope": "identity/target_units (bare endpoint) on this run's grid only",
            "by_horizon": by_horizon}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def run_grid(*, df, dataset, feature_cols, target_col, seasonality, horizons, adapters,
             modes, repetitions, base_seed, fit_seed, max_origins, output_dir, pipeline,
             endpoint_label, device_map="cpu", context_len=512, num_samples=50,
             run_info=None, split="test"):
    """Sequential grid on the test or validation split; returns the completed manifest.

    Every adapter and mode at one (horizon, origin, repetition) shares a sampler
    seed, and the fit seed stays fixed, so repetitions vary only the sampler.
    identity/target_units is replayed with the same seed after the other
    methods. max_origins=None is the full eligible grid; no future mask exists.
    Validation horizons end before the test split, as in the retained runner.
    """
    if "identity" not in adapters or "target_units" not in modes:
        raise ValueError("The grid needs identity/target_units for the noise-floor replay")
    if repetitions < 1 or (max_origins is not None and max_origins < 1):
        raise ValueError("Require at least one repetition and one origin")
    hyperparameters = {a: resolved_hyperparameters(a) for a in adapters}
    n = len(df)
    train_end, val_end, _ = legacy.get_split_indices(n)
    start, end = {"test": (val_end, n), "val": (train_end, val_end)}[split]
    origins, n_eligible, seeds = {}, {}, {}
    for h in horizons:
        n_eligible[h] = len(legacy.rolling_window_origins(end, start, context_len, h, stride=h))
        origins[h] = legacy.rolling_window_origins(end, start, context_len, h, stride=h,
                                                   max_origins=max_origins)
        for origin in origins[h]:
            for rep in range(repetitions):
                seeds[h, origin, rep] = sampler_seed_for_origin(
                    dataset=dataset, target=target_col, horizon=h, origin=origin,
                    repetition=rep, base_seed=base_seed)
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("Sampler seed collision in this grid; choose another base seed")

    columns = list(dict.fromkeys([*feature_cols, target_col]))
    manifest = {
        "schema": "cmf-peer-review-controls-v1", "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint_label": endpoint_label, "device": device_map,
        "dataset": dataset, "target": str(target_col),
        "features": [str(c) for c in feature_cols],
        "n_rows": n, "data_sha256": frame_sha256(df, columns),
        "split": {"name": split, "rule": "chronological 60/20/20 (legacy get_split_indices)",
                  "origin_start": start, "horizon_end": end},
        "grid_kind": "full_eligible" if max_origins is None else "capped_pilot",
        "max_origins": max_origins,
        "origin_rule": ("stride = H, complete horizons inside the split only, zero-future "
                        "origins kept (no future mask); a cap keeps evenly spaced "
                        "indices, as in the retained 150-origin grid"),
        "origins": {str(h): o for h, o in origins.items()},
        "n_eligible": {str(h): k for h, k in n_eligible.items()},
        "adapters": list(adapters), "modes": list(modes), "repetitions": repetitions,
        "base_seed": base_seed, "fit_seed": fit_seed,
        "adapter_hyperparameters": hyperparameters,
        "hyperparameter_selection": ("as written in each adapter spec, legacy defaults "
                                     "otherwise; no validation tuning inside this run"),
        "context_length": context_len, "seasonality": seasonality,
        "num_samples": num_samples, "sampling_args": SAMPLING_ARGS,
        "quantile_levels": legacy.QUANTILE_LEVELS,
        "point_estimator": "sample median; mean-point MASE kept as sensitivity",
        "environment": environment_record(), "run_info": run_info or {},
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "predictions").mkdir()
    _write_json(out / "manifest.json", manifest)
    dated = isinstance(df.index, pd.DatetimeIndex)
    rows = []
    with (out / "losses.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOSS_FIELDS)
        writer.writeheader()

        def record(result, rep, replay, replay_diff=None):
            spec = re.sub(r"[^A-Za-z0-9_.-]+", "-", result["adapter"])
            name = (f"H{result['horizon']}_{spec}_{result['mode']}"
                    f"_r{rep}_p{replay}_o{result['origin_idx']}.npz")
            result.update(dataset=dataset, repetition=rep, replay=replay)
            write_prediction(out / "predictions" / name, result)
            row = {k: result.get(k) for k in LOSS_FIELDS}
            row.update(origin_time=str(df.index[result["origin_idx"]]) if dated else None,
                       replay_max_abs_diff=replay_diff, prediction_file=f"predictions/{name}")
            writer.writerow(row)
            handle.flush()
            rows.append(row)

        for h in horizons:
            for origin in origins[h]:
                for rep in range(repetitions):
                    common = dict(df=df, origin=origin, context_len=context_len, pred_len=h,
                                  feature_cols=feature_cols, target_col=target_col,
                                  seasonality=seasonality, device_map=device_map,
                                  pipeline=pipeline, fit_seed=fit_seed,
                                  sampler_seed=seeds[h, origin, rep], num_samples=num_samples)
                    for adapter in adapters:
                        for mode in modes:
                            result = evaluate_control_origin(adapter_kind=adapter, mode=mode,
                                                             **common)
                            if (adapter, mode) == ("identity", "target_units"):
                                first_samples = result["samples"]
                            record(result, rep, 0)
                    replay = evaluate_control_origin(adapter_kind="identity",
                                                     mode="target_units", **common)
                    record(replay, rep, 1,
                           float(np.abs(replay["samples"] - first_samples).max()))

    _write_json(out / "identity_noise_summary.json",
                identity_noise_summary(rows, endpoint_label))
    manifest.update(status="complete", n_records=len(rows),
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    noise_summary_file="identity_noise_summary.json")
    _write_json(out / "manifest.json", manifest)
    return manifest


def select_on_validation(validation_dir, candidates, mode="target_units"):
    """Per horizon, the candidate with the lowest mean validation MASE.

    The mean runs over every origin and repetition; ties keep candidate order.
    Only completed validation-split runs are accepted.
    """
    directory = Path(validation_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["split"]["name"] != "val" or manifest["status"] != "complete":
        raise ValueError("Selection needs a completed validation-split run")
    with (directory / "losses.csv").open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["replay"] == "0" and r["mode"] == mode]
    by_horizon = {}
    for h in manifest["origins"]:
        values = {c: [float(r["MASE"]) for r in rows if r["horizon"] == h and r["adapter"] == c]
                  for c in candidates}
        if len({len(v) for v in values.values()}) != 1 or not values[candidates[0]]:
            raise ValueError(f"Candidates are not evaluated on one common grid at H={h}")
        scores = {c: float(np.mean(v)) for c, v in values.items()}
        by_horizon[h] = {"selected": min(candidates, key=lambda c: (scores[c], candidates.index(c))),
                         "scores": scores, "n_per_candidate": len(values[candidates[0]])}
    return {"validation_dir": str(directory), "mode": mode, "candidates": list(candidates),
            "by_horizon": by_horizon}


def _load_chronos(model_id, revision, device):
    """Load the checkpoint only when a run is actually requested."""
    from chronos import BaseChronosPipeline
    kwargs = {"revision": revision} if revision else {}
    pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map=device,
                                                   torch_dtype=torch.float32, **kwargs)
    config = getattr(pipeline.inner_model, "config", None)
    return pipeline, {"model_id": model_id, "requested_revision": revision,
                      "resolved_commit": getattr(config, "_commit_hash", None),
                      "dtype": "float32"}


def _adapter_spec(text):
    try:
        parse_adapter(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    return text


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m experiments.run_peer_review_controls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Peer-review controls on the frozen Chronos-T5 endpoint.\n\n"
            "Origins: stride = H, complete horizons inside the split, and\n"
            "no future mask: zero-future origins are always kept.\n"
            "Hyperparameters are those written in each adapter spec, legacy\n"
            "defaults otherwise; a run never tunes them. Tune by running\n"
            "--split val and calling select_on_validation. Choose --max-origins\n"
            "for a capped pilot or --all-eligible for the full eligible grid.\n\n"
            "Adapter spec: KIND[_VARIANT][:key=value,...], lists joined by '+'.\n"
            "  KIND     identity, linear, ridge, pcr, krr, mlp, ewma, residual\n"
            "  VARIANT  covonly, targetonly, lagged (not for identity/ewma/residual)\n"
            "  e.g.     ridge:ridge_lam=0.1  krr_covonly  ewma:alpha=0.3\n"
            "           residual:alpha=0.5,ridge_lam=1  ridge_lagged:lag_hours=1+24"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", required=True)
    parser.add_argument("--adapters", nargs="+", type=_adapter_spec, required=True,
                        help="adapter specs; must include identity")
    parser.add_argument("--modes", nargs="+", choices=MODES, required=True,
                        help="must include target_units")
    parser.add_argument("--repetitions", type=int, required=True,
                        help="independent sampler seeds per origin")
    grid = parser.add_mutually_exclusive_group(required=True)
    grid.add_argument("--max-origins", type=int,
                      help="capped pilot: evenly spaced origins, the retained-grid rule")
    grid.add_argument("--all-eligible", action="store_true",
                      help="full eligible grid without a cap")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--output-dir", required=True,
                        help="must not exist; evidence is never overwritten")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--model", default="amazon/chronos-t5-base")
    parser.add_argument("--revision", default=None,
                        help="Hugging Face revision to pin; the resolved commit is recorded")
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--fit-seed", type=int, default=42)
    return parser


def main(argv=None, *, load_data=None, load_pipeline=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if Path(args.output_dir).exists():
        parser.error(f"--output-dir {args.output_dir} already exists")
    df, feature_cols, target_col, seasonality = (load_data or legacy.cmf.load_dataset)(
        args.dataset)
    pipeline, checkpoint = (load_pipeline or _load_chronos)(args.model, args.revision,
                                                             args.device)
    manifest = run_grid(
        df=df, dataset=args.dataset, feature_cols=feature_cols, target_col=target_col,
        seasonality=seasonality, horizons=args.horizons, adapters=args.adapters,
        modes=args.modes, repetitions=args.repetitions, base_seed=args.base_seed,
        fit_seed=args.fit_seed, max_origins=None if args.all_eligible else args.max_origins,
        output_dir=args.output_dir, pipeline=pipeline,
        endpoint_label=f"{type(pipeline).__name__}: {args.model}",
        device_map=args.device, context_len=args.context_len, num_samples=args.num_samples,
        split=args.split, run_info={"argv": sys.argv[1:] if argv is None else list(argv),
                  "checkpoint": checkpoint,
                  "data_config": legacy.cmf.DATASET_CONFIGS.get(args.dataset)})
    print(f"{manifest['n_records']} records written to {args.output_dir}")
    return manifest


if __name__ == "__main__":
    main()
