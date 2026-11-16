"""
CMF Revision Experiments — Rolling-window evaluation with external baselines
=============================================================================
Addresses IJIES reviewer concerns:
  [C] External baseline comparisons under matched protocol
  [D] Per-dataset, per-horizon disaggregated results with dispersion
  [F] Calibration diagnostics (empirical coverage, horizon-wise breakdown)

Usage:
  # Smoke test (single dataset, single adapter, few origins)
  python cmf_revision_experiments.py --datasets etth2 --adapters ridge identity --horizons 24 --seeds 1 --max_origins 3 --device cuda

  # Full benchmark
  python cmf_revision_experiments.py --datasets etth1 etth2 ettm1 ettm2 --adapters all --horizons 24 48 --seeds 5 --device cuda --output_dir results/revision_v1

Authors: Ardi Yusri Hilmi et al.
Created: 2026-03-03
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Import from existing CMF module
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from src import cmf_core as cmf
from src.paper_metrics import build_paper_summary_table

# Try importing Chronos
try:
    from chronos import BaseChronosPipeline
except ImportError:
    BaseChronosPipeline = None
    print("[WARN] chronos-forecasting not installed.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUANTILE_LEVELS = [0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95]
NUM_CHRONOS_SAMPLES = 100  # S = 100 sample paths; 20 is insufficient for extreme quantiles

CMF_ADAPTERS = ['linear', 'ridge', 'pcr', 'krr', 'mlp', 'feat_attn']
BASELINES = ['identity', 'seasonal_naive']
ALL_METHODS = CMF_ADAPTERS + BASELINES

# Default adapter hyperparameters (fixed-budget protocol, as stated in paper)
# NOTE: feat_attn uses attn_* keys (not fg_*) — these map to fit_cross_variable_attention args
ADAPTER_CONFIGS = {
    'linear':    {},
    'ridge':     {'ridge_lam': 1e-2},
    'pcr':       {'pcr_n_comp': 3, 'pcr_lam': 0.0},
    'krr':       {'krr_lam': 1e-1, 'krr_gamma': None},
    'mlp':       {'mlp_hidden': 32, 'mlp_epochs': 300, 'mlp_lr': 1e-2,
                  'mlp_wd': 1e-4, 'mlp_device': 'cpu'},
    'feat_attn': {'attn_embed_dim': 32, 'attn_num_heads': 1, 'attn_epochs': 300,
                  'attn_lr': 1e-2, 'attn_device': 'cpu'},
    'identity':        {},
    'seasonal_naive':  {},
}


# ===========================================================================
# Data Splitting & Rolling Window  (canonical implementations live in src/cmf_core.py)
# ===========================================================================

from src import cmf_core as _cmf_utils
get_split_indices     = _cmf_utils.get_split_indices
rolling_window_origins = _cmf_utils.rolling_window_origins


# ===========================================================================
# Evaluation Functions
# ===========================================================================

def evaluate_single_origin_cmf(
    df: pd.DataFrame,
    origin: int,
    adapter_kind: str,
    context_len: int,
    pred_len: int,
    feature_cols: list,
    target_col: str,
    seasonality: int,
    device_map: str,
    pipeline: Any = None,
    seed: int = 42,
    adapter_kwargs: dict = None,
    no_scale_restoration: bool = False,
    num_samples: int = NUM_CHRONOS_SAMPLES,
    include_zero_future: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a CMF adapter at a single forecast origin.
    Returns dict with metrics and raw predictions.
    """
    if adapter_kwargs is None:
        adapter_kwargs = {}
    
    # Set seed for reproducibility of neural adapters
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Extract context and future
    ctx_start = origin - context_len
    ctx_end = origin
    fut_end = origin + pred_len
    
    ctx_df = df.iloc[ctx_start:ctx_end].copy()
    y_future = df[target_col].values[ctx_end:fut_end]
    
    if len(y_future) < pred_len:
        return None  # skip incomplete windows

    if not include_zero_future and np.abs(y_future).max() < 1e-6:
        return None
    
    # --- Seasonal Naive ---
    if adapter_kind == 'seasonal_naive':
        history = ctx_df[target_col].values
        if len(history) < seasonality:
            forecast = np.full(pred_len, history[-1])
        else:
            last_season = history[-seasonality:]
            reps = int(np.ceil(pred_len / seasonality))
            forecast = np.tile(last_season, reps)[:pred_len]
        
        # Deterministic: all quantiles = point forecast
        q_preds = np.tile(forecast[:, None], (1, len(QUANTILE_LEVELS)))
        insample = history
        
        mase_val = cmf.compute_mase(y_future, forecast, insample, seasonality)
        wql_val  = weighted_quantile_loss_full(y_future, q_preds, QUANTILE_LEVELS)
        mae_val  = float(np.mean(np.abs(y_future - forecast)))
        rmse_val = float(np.sqrt(np.mean((y_future - forecast) ** 2)))

        return {
            'MASE': mase_val, 'WQL': wql_val,
            'MAE': mae_val, 'RMSE': rmse_val,
            'Time_ms': 0.0,
            'y_true': y_future, 'y_pred': forecast, 'q_preds': q_preds,
            'insample': insample,
        }
    
    # --- Identity (Vanilla Chronos) ---
    if adapter_kind == 'identity':
        context_tensor = torch.tensor(
            ctx_df[target_col].values[None, :], dtype=torch.float32
        )
    else:
        # --- CMF adapters: fit adapter on context window ---
        context_mv = ctx_df[feature_cols].values    # [L, d]
        target_uni = ctx_df[target_col].values      # [L]

        # Determine target column index within feature_cols (needed by CrossVariableAttention)
        target_idx = feature_cols.index(target_col) if target_col in feature_cols else 0

        # Per-feature z-score normalisation within the context window.
        # Ensures all features enter the adapter at comparable scale regardless of dataset units.
        # The output is denormalised back to original target scale before passing to Chronos.
        ctx_means = context_mv.mean(axis=0, keepdims=True)            # [1, d]
        ctx_stds  = context_mv.std(axis=0, keepdims=True) + 1e-8      # [1, d]  (avoid /0)
        context_mv_norm = (context_mv - ctx_means) / ctx_stds         # [L, d]
        target_uni_norm = (target_uni - ctx_means[0, target_idx]) / ctx_stds[0, target_idx]

        # Build kwargs — use correct key names for each adapter
        kw = dict(adapter_kwargs)
        kw['target_idx'] = target_idx
        if adapter_kind == 'mlp':
            kw['mlp_device'] = device_map
        if adapter_kind == 'feat_attn':
            kw['attn_device'] = device_map   # was 'fg_device' — bug fix

        adapter = cmf.fit_adapter(context_mv_norm, target_uni_norm, kind=adapter_kind, **kw)
        context_mv_3d_norm = context_mv_norm[None, :, :].astype(np.float32)  # [1, L, d]
        context_adapted_norm = cmf.apply_adapter(context_mv_3d_norm, adapter)  # [1, L]

        # Denormalise to original target scale before Chronos tokenisation
        t_mean = float(ctx_means[0, target_idx])
        t_std  = float(ctx_stds[0, target_idx])
        context_adapted = context_adapted_norm * t_std + t_mean        # [1, L]
        context_tensor = torch.tensor(context_adapted, dtype=torch.float32)
    
    # --- Chronos inference ---
    if pipeline is None:
        raise ValueError("Pipeline must be provided")
    
    # Ablation C: simulate lack of Tokeniser Compatibility Constraint.
    # Can be triggered via the `no_scale_restoration` parameter or the CLI flag.
    _no_scale = no_scale_restoration or ('--no_scale_restoration' in sys.argv)
    if _no_scale:
        # Standardise to N(0,1) — destroys the macroscopic scale Chronos relies on
        # for its tokenisation bin boundaries, degrading forecast quality.
        _mean = context_tensor.mean(dim=-1, keepdim=True)
        _std  = context_tensor.std(dim=-1, keepdim=True) + 1e-5
        context_tensor = (context_tensor - _mean) / _std
    
    # Sync before timing
    _sync_device(device_map)
    
    t0 = time.perf_counter()
    # Use predict to get samples (for full quantile computation)
    samples = pipeline.predict(
        context=context_tensor,
        prediction_length=pred_len,
        num_samples=num_samples,
    )  # [1, S, H]
    _sync_device(device_map)
    inf_time_s = time.perf_counter() - t0
    
    samples_np = samples[0].detach().cpu().numpy()  # [S, H]
    
    # Compute quantiles from samples
    q_preds = np.quantile(samples_np, QUANTILE_LEVELS, axis=0).T  # [H, Q]
    mean_pred = samples_np.mean(axis=0)  # [H]
    
    insample = ctx_df[target_col].values
    mase_val = cmf.compute_mase(y_future, mean_pred, insample, seasonality)
    wql_val  = weighted_quantile_loss_full(y_future, q_preds, QUANTILE_LEVELS)
    mae_val  = float(np.mean(np.abs(y_future - mean_pred)))
    rmse_val = float(np.sqrt(np.mean((y_future - mean_pred) ** 2)))

    return {
        'MASE': mase_val, 'WQL': wql_val,
        'MAE': mae_val, 'RMSE': rmse_val,
        'Time_ms': inf_time_s * 1000.0,
        'y_true': y_future, 'y_pred': mean_pred, 'q_preds': q_preds,
        'insample': insample,
    }


def _prepare_single_origin(
    df, origin, adapter_kind, context_len, pred_len,
    feature_cols, target_col, device_map, seed, adapter_kwargs,
    include_zero_future=False,
):
    """
    Fit adapter + prepare context tensor for ONE origin.
    Returns (context_tensor [1,L], y_future [H], insample [L]) or None if skipped.
    CPU-bound for deterministic adapters → safe to parallelize with ThreadPoolExecutor.
    GPU-bound for neural adapters → call sequentially.

    Fixes applied:
    - Bug 1: target_idx uses feature_cols membership; for aggregate targets
             (e.g. total_load not in feat_cols) target_idx=0 for attention,
             but t_mean/t_std always from actual target series (not feat_cols[0]).
    - Bug 2: t_mean/t_std taken from ctx_df[target_col] directly — correct for
             both regular and aggregate targets.
    - Bug 3: Skip all-zero y_future windows (solar night / sparse EV) → return None.
    - Bug 4: insample stored in return tuple → no re-fetch needed in run_experiment.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    ctx_start = origin - context_len
    ctx_df    = df.iloc[ctx_start:origin].copy()
    y_future  = df[target_col].values[origin:origin + pred_len]

    # Skip: window too short
    if len(y_future) < pred_len:
        return None

    # Skip: all-zero future (solar night / idle EV station) → trivially zero metrics
    if not include_zero_future and np.abs(y_future).max() < 1e-6:
        return None

    insample = ctx_df[target_col].values  # ground-truth in-sample for MASE denominator

    if adapter_kind == 'identity':
        ctx_tensor = torch.tensor(insample[None, :], dtype=torch.float32)
        return ctx_tensor, y_future, insample

    # ── Feature matrix (only actual sensor columns, not derived aggregates) ──
    context_mv = ctx_df[feature_cols].values.astype(np.float32)

    # ── Per-feature z-score normalisation of feature matrix ──────────────────
    ctx_means = context_mv.mean(axis=0, keepdims=True)      # [1, d]
    ctx_stds  = context_mv.std(axis=0, keepdims=True) + 1e-8
    context_mv_norm = (context_mv - ctx_means) / ctx_stds   # [L, d]

    # ── Target normalisation (always from actual target series) ──────────────
    # BUG FIX: previously used ctx_means[0, target_idx] which is WRONG for
    # aggregate targets (e.g. total_load not in feature_cols → target_idx=0
    # → stats of station-0 ≈ 0, not of total_load).
    target_raw = insample.astype(np.float32)
    t_mean = float(target_raw.mean())
    t_std  = float(target_raw.std() + 1e-8)
    target_uni_norm = (target_raw - t_mean) / t_std

    # ── CrossVariableAttention target index ───────────────────────────────────
    # For regular datasets: index of target column within feature_cols.
    # For aggregate targets (total_load not in feat_cols): fallback to 0.
    # The adapter loss still trains on the correct target_uni_norm above.
    target_idx = feature_cols.index(target_col) if target_col in feature_cols else 0

    kw = dict(adapter_kwargs)
    kw['target_idx'] = target_idx
    if adapter_kind == 'mlp':
        kw['mlp_device'] = device_map
    if adapter_kind == 'feat_attn':
        kw['attn_device'] = device_map

    adapter      = cmf.fit_adapter(context_mv_norm, target_uni_norm, kind=adapter_kind, **kw)
    ctx_3d_norm  = context_mv_norm[None]                          # [1, L, d]
    adapted_norm = cmf.apply_adapter(ctx_3d_norm, adapter)        # [1, L]

    # ── Denormalise to original TARGET scale (correct for all cases) ─────────
    adapted    = adapted_norm * t_std + t_mean                    # [1, L]
    ctx_tensor = torch.tensor(adapted, dtype=torch.float32)

    return ctx_tensor, y_future, insample


def evaluate_batch_origins_cmf(
    df: pd.DataFrame,
    origins: List[int],
    adapter_kind: str,
    context_len: int,
    pred_len: int,
    feature_cols: list,
    target_col: str,
    seasonality: int,
    device_map: str,
    pipeline: Any,
    seed: int = 42,
    adapter_kwargs: dict = None,
    no_scale_restoration: bool = False,
    num_samples: int = NUM_CHRONOS_SAMPLES,
    n_fit_workers: int = 8,
    include_zero_future: bool = False,
) -> List[Optional[Dict]]:
    """
    Evaluate a CMF adapter over a BATCH of origins in one Chronos call.

    Deterministic adapters are fit in parallel (ThreadPoolExecutor).
    Neural adapters are fit sequentially (GPU serialised).
    All adapted contexts are stacked and sent to Chronos as a single batch.

    Returns list of result dicts (same schema as evaluate_single_origin_cmf),
    with None for any skipped (too-short) windows.
    """
    if adapter_kwargs is None:
        adapter_kwargs = {}

    DETERMINISTIC = {'linear', 'ridge', 'pcr', 'krr', 'identity', 'seasonal_naive'}
    is_deterministic = adapter_kind in DETERMINISTIC

    # ── Seasonal Naive: no Chronos needed, handle individually ────────────────
    if adapter_kind == 'seasonal_naive':
        results = []
        for origin in origins:
            results.append(evaluate_single_origin_cmf(
                df=df, origin=origin, adapter_kind=adapter_kind,
                context_len=context_len, pred_len=pred_len,
                feature_cols=feature_cols, target_col=target_col,
                seasonality=seasonality, device_map=device_map,
                pipeline=pipeline, seed=seed, adapter_kwargs=adapter_kwargs,
                no_scale_restoration=no_scale_restoration, num_samples=num_samples,
                include_zero_future=include_zero_future,
            ))
        return results

    # ── Step 1: Fit adapters (parallel for deterministic, sequential for neural) ──
    def _fit_one(origin):
        return _prepare_single_origin(
            df, origin, adapter_kind, context_len, pred_len,
            feature_cols, target_col, device_map, seed, adapter_kwargs,
            include_zero_future,
        )

    prepared = []
    if is_deterministic:
        with ThreadPoolExecutor(max_workers=min(n_fit_workers, len(origins))) as ex:
            futures = {ex.submit(_fit_one, o): i for i, o in enumerate(origins)}
            # Preserve order
            ordered = [None] * len(origins)
            for fut, idx in futures.items():
                ordered[idx] = fut.result()
        prepared = ordered
    else:
        # Neural: GPU serial (avoid concurrent CUDA ops from threads)
        for origin in origins:
            prepared.append(_fit_one(origin))

    # Filter None (incomplete windows) while keeping track of origin index
    valid = [(i, origins[i], p) for i, p in enumerate(prepared) if p is not None]
    if not valid:
        return [None] * len(origins)

    # ── Step 2: Stack valid contexts → single Chronos call ────────────────────
    ctx_tensors = torch.cat([p[0] for _, _, p in valid], dim=0)  # [B_valid, L]

    if no_scale_restoration:
        _m = ctx_tensors.mean(dim=-1, keepdim=True)
        _s = ctx_tensors.std(dim=-1, keepdim=True) + 1e-5
        ctx_tensors = (ctx_tensors - _m) / _s

    _sync_device(device_map)
    t0 = time.perf_counter()
    all_samples = pipeline.predict(
        context=ctx_tensors,
        prediction_length=pred_len,
        num_samples=num_samples,
    )  # [B_valid, S, H]
    _sync_device(device_map)
    inf_time_s = time.perf_counter() - t0
    inf_time_per_origin_ms = inf_time_s * 1000.0 / len(valid)

    all_samples_np = all_samples.detach().cpu().numpy()  # [B_valid, S, H]

    # ── Step 3: Compute metrics for each valid origin ─────────────────────────
    result_map = {}
    for batch_j, (orig_i, origin, prep) in enumerate(valid):
        _, y_future, insample = prep
        samples_np = all_samples_np[batch_j]       # [S, H]
        q_preds    = np.quantile(samples_np, QUANTILE_LEVELS, axis=0).T  # [H, Q]
        mean_pred  = samples_np.mean(axis=0)        # [H]

        mase_val = cmf.compute_mase(y_future, mean_pred, insample, seasonality)
        wql_val  = weighted_quantile_loss_full(y_future, q_preds, QUANTILE_LEVELS)
        mae_val  = float(np.mean(np.abs(y_future - mean_pred)))
        rmse_val = float(np.sqrt(np.mean((y_future - mean_pred) ** 2)))

        result_map[orig_i] = {
            'MASE': mase_val, 'WQL': wql_val,
            'MAE': mae_val, 'RMSE': rmse_val,
            'Time_ms': inf_time_per_origin_ms,
            'y_true': y_future, 'y_pred': mean_pred, 'q_preds': q_preds,
            'insample': insample,   # stored here — no re-fetch needed in run_experiment
        }

    return [result_map.get(i) for i in range(len(origins))]


def _sync_device(device_map: str):
    """Synchronize GPU/MPS for accurate timing."""
    try:
        if device_map == 'cuda':
            torch.cuda.synchronize()
        elif device_map == 'mps':
            torch.mps.synchronize()
    except Exception:
        pass


# ===========================================================================
# Metrics (Extended)
# ===========================================================================

def weighted_quantile_loss_full(y_true: np.ndarray, q_preds: np.ndarray,
                                quantile_levels: list) -> float:
    """
    WQL with arbitrary quantile levels.
    y_true: [H]
    q_preds: [H, Q]  (Q = len(quantile_levels))
    """
    loss = 0.0
    for i, q in enumerate(quantile_levels):
        e = y_true - q_preds[:, i]
        loss += 2.0 * np.mean(np.maximum(q * e, (q - 1.0) * e))
    return float(loss / len(quantile_levels))


def compute_empirical_coverage(y_true_list: List[np.ndarray],
                                q_preds_list: List[np.ndarray],
                                quantile_levels: list) -> Dict[float, float]:
    """
    Compute empirical coverage for each quantile level.
    coverage(tau) = fraction of test points where y <= q_hat(tau)
    Perfect calibration: coverage(tau) == tau
    """
    coverage = {q: 0.0 for q in quantile_levels}
    total = 0
    
    for y_true, q_preds in zip(y_true_list, q_preds_list):
        n = len(y_true)
        total += n
        for i, q in enumerate(quantile_levels):
            coverage[q] += np.sum(y_true <= q_preds[:, i])
    
    for q in quantile_levels:
        coverage[q] /= max(total, 1)
    
    return coverage


def compute_horizon_breakdown(y_true_list: List[np.ndarray],
                               y_pred_list: List[np.ndarray],
                               q_preds_list: List[np.ndarray],
                               insample_list: List[np.ndarray],
                               seasonality: int,
                               quantile_levels: list) -> pd.DataFrame:
    """
    Compute per-horizon MASE and WQL.
    Returns DataFrame with columns [horizon, MASE, WQL].
    """
    if not y_true_list:
        return pd.DataFrame()
    
    H = len(y_true_list[0])
    n_origins = len(y_true_list)
    
    rows = []
    for h in range(H):
        # Collect all values at horizon h
        y_true_h = np.array([yt[h] for yt in y_true_list])
        y_pred_h = np.array([yp[h] for yp in y_pred_list])
        q_preds_h = np.array([qp[h, :] for qp in q_preds_list])  # [N, Q]
        
        # MAE at horizon h (scaled by mean in-sample naive error)
        mae_h = np.mean(np.abs(y_true_h - y_pred_h))
        naive_errors = []
        for ins in insample_list:
            d = np.abs(ins[seasonality:] - ins[:-seasonality]).mean()
            naive_errors.append(max(d, 1e-8))
        scale = np.mean(naive_errors)
        mase_h = mae_h / scale
        
        # WQL at horizon h
        wql_h = 0.0
        for i, q in enumerate(quantile_levels):
            e = y_true_h - q_preds_h[:, i]
            wql_h += 2.0 * np.mean(np.maximum(q * e, (q - 1.0) * e))
        wql_h /= len(quantile_levels)
        
        rows.append({'horizon': h + 1, 'MASE': mase_h, 'WQL': wql_h})
    
    return pd.DataFrame(rows)


# ===========================================================================
# Main Experiment Runner
# ===========================================================================

def run_experiment(
    dataset_name: str,
    adapter_kind: str,
    horizons: List[int],
    seeds: List[int],
    context_len: int,
    device_map: str,
    max_origins: int = None,
    output_dir: str = None,
    feature_cols_override: List[str] = None,
    no_scale_restoration: bool = False,
    chronos_model_id: str = "amazon/chronos-t5-small",
    num_samples: int = NUM_CHRONOS_SAMPLES,
    torch_dtype: torch.dtype = torch.float32,
    batch_size: int = 1,
    n_fit_workers: int = 8,
    split: str = "test",
    adapter_kwargs_override: Dict[str, Any] = None,
    include_zero_future: bool = False,
) -> List[Dict]:
    """
    Run the full experiment for one dataset × one adapter × multiple horizons × seeds.
    Returns list of per-origin result dicts.

    Args:
        feature_cols_override: If set, replaces the dataset's default feature columns
                               (used by Ablation A — covariate cardinality study).
        no_scale_restoration:  If True, standardises the fused context to N(0,1) before
                               Chronos — ablates the tokeniser compatibility constraint.
        split: Which chronological split supplies the forecast origins.
               "test" (default) → origins in [val_end, n) — the paper-facing numbers.
               "val"            → origins in [train_end, val_end) — used by CMF-Select
                                  to rank adapters on held-out data WITHOUT touching the
                                  test split. Context windows may reach back into train
                                  (past data only); no future/test target ever leaks.
    """
    # Load dataset
    df, feature_cols, target_col, seasonality = cmf.load_dataset(dataset_name)
    n = len(df)
    train_end, val_end, test_end = get_split_indices(n)
    if split not in ("test", "val"):
        raise ValueError(f"split must be 'test' or 'val', got {split!r}")

    # Ablation A: override feature columns via programmatic parameter or CLI flag
    if feature_cols_override is not None:
        feature_cols = feature_cols_override
        print(f"  [ABLATION A] Feature cols overridden: {feature_cols}")
    elif '--features' in sys.argv:
        idx = sys.argv.index('--features')
        if idx + 1 < len(sys.argv):
            feat_arg = sys.argv[idx + 1]
            if feat_arg and feat_arg[0] != '-':
                feature_cols = feat_arg.split(',')
                print(f"  [ABLATION A] Feature cols overridden via CLI: {feature_cols}")

    # Origins are drawn from the selected split. For CMF-Select we evaluate on the
    # validation split so the adapter ranking never sees the test split.
    if split == "val":
        origin_n, origin_start = val_end, train_end
    else:
        origin_n, origin_start = n, val_end

    print(f"\n  Dataset: {dataset_name} | N={n} | "
          f"train:[0,{train_end}) val:[{train_end},{val_end}) test:[{val_end},{test_end})")
    print(f"  Origins from split='{split}' → range [{origin_start},{origin_n})")
    print(f"  Features: {feature_cols} | Target: {target_col} | Seasonality: {seasonality}")
    
    # Get adapter config
    adapter_kwargs = ADAPTER_CONFIGS.get(adapter_kind, {}).copy()
    # Hyperparameter sweep hook (e.g. ridge_lam for the regularisation-strength
    # study). Only keys already meaningful to the adapter are honoured.
    if adapter_kwargs_override:
        adapter_kwargs.update(adapter_kwargs_override)
        print(f"  [OVERRIDE] adapter_kwargs → {adapter_kwargs}")

    # Build Chronos pipeline once (reuse across origins)
    pipeline = None
    if adapter_kind not in ['seasonal_naive'] and BaseChronosPipeline is not None:
        print(f"  Loading Chronos pipeline: {chronos_model_id} (device={device_map}, dtype={torch_dtype})")
        pipeline = BaseChronosPipeline.from_pretrained(
            chronos_model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        # Warm-up
        warm_ctx = torch.zeros(1, context_len, dtype=torch.float32)
        _sync_device(device_map)
        _ = pipeline.predict(
            context=warm_ctx,
            prediction_length=24,
            num_samples=2,
        )
        _sync_device(device_map)
        print("  Pipeline loaded and warmed up.")
    
    all_results = []
    
    for H in horizons:
        origins = rolling_window_origins(
            origin_n, origin_start, context_len, H,
            stride=H,  # non-overlapping
            max_origins=max_origins,
        )
        print(f"\n  H={H} | {len(origins)} forecast origins (split={split})")
        
        # Determine seeds (deterministic adapters only need 1 seed)
        is_deterministic = adapter_kind in ['linear', 'ridge', 'pcr', 'krr',
                                             'identity', 'seasonal_naive']
        effective_seeds = [seeds[0]] if is_deterministic else seeds
        
        for seed in effective_seeds:
            y_true_list   = []
            y_pred_list   = []
            q_preds_list  = []
            insample_list = []
            times_ms      = []
            mae_list      = []
            rmse_list     = []
            origin_idx_list = []

            # ── Batched evaluation ──────────────────────────────────────────
            # batch_size=1 falls back to sequential (same as before).
            # Deterministic adapters benefit from parallel CPU fitting.
            # All adapters benefit from batched Chronos inference.
            B = max(1, batch_size)
            oi = 0
            for batch_start in range(0, len(origins), B):
                batch_origins = origins[batch_start:batch_start + B]
                batch_results = evaluate_batch_origins_cmf(
                    df=df, origins=batch_origins,
                    adapter_kind=adapter_kind,
                    context_len=context_len,
                    pred_len=H,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    seasonality=seasonality,
                    device_map=device_map,
                    pipeline=pipeline,
                    seed=seed,
                    adapter_kwargs=adapter_kwargs,
                    no_scale_restoration=no_scale_restoration,
                    num_samples=num_samples,
                    n_fit_workers=n_fit_workers,
                    include_zero_future=include_zero_future,
                )
                for local_i, result in enumerate(batch_results):
                    oi += 1
                    if result is None:
                        continue
                    # BUG FIX: insample now comes from result dict (set in
                    # _prepare_single_origin), not re-fetched via error-prone
                    # origin index arithmetic.
                    y_true_list.append(result['y_true'])
                    y_pred_list.append(result['y_pred'])
                    q_preds_list.append(result['q_preds'])
                    insample_list.append(result['insample'])
                    times_ms.append(result['Time_ms'])
                    mae_list.append(result['MAE'])
                    rmse_list.append(result['RMSE'])
                    origin_idx_list.append(batch_origins[local_i])

                done = min(batch_start + B, len(origins))
                valid_in_batch = [r for r in batch_results if r is not None]
                if valid_in_batch and (done % max(B, 10) == 0 or done == len(origins)):
                    last = valid_in_batch[-1]
                    print(f"    origin {done}/{len(origins)} | "
                          f"MASE={last['MASE']:.4f}  WQL={last['WQL']:.4f}  "
                          f"MAE={last['MAE']:.4f}  RMSE={last['RMSE']:.4f}  "
                          f"Time={last['Time_ms']:.1f}ms")

            if not y_true_list:
                continue

            # Aggregate metrics across origins
            all_mase = [cmf.compute_mase(yt, yp, ins, seasonality)
                        for yt, yp, ins in zip(y_true_list, y_pred_list, insample_list)]
            all_wql  = [weighted_quantile_loss_full(yt, qp, QUANTILE_LEVELS)
                        for yt, qp in zip(y_true_list, q_preds_list)]

            avg_mase = float(np.mean(all_mase))
            avg_wql  = float(np.mean(all_wql))
            avg_mae  = float(np.mean(mae_list))
            avg_rmse = float(np.mean(rmse_list))
            avg_time = float(np.mean(times_ms))
            avg_target_abs_mean = float(np.mean([np.mean(np.abs(yt)) for yt in y_true_list]))

            # Calibration
            coverage = compute_empirical_coverage(
                y_true_list, q_preds_list, QUANTILE_LEVELS
            )

            # Horizon breakdown
            horizon_df = compute_horizon_breakdown(
                y_true_list, y_pred_list, q_preds_list,
                insample_list, seasonality, QUANTILE_LEVELS
            )

            row = {
                'dataset': dataset_name,
                'adapter': adapter_kind,
                'horizon': H,
                'seed': seed,
                'n_origins': len(y_true_list),
                'MASE_mean': avg_mase,
                'MASE_std': float(np.std(all_mase)),
                'WQL_mean': avg_wql,
                'WQL_std': float(np.std(all_wql)),
                'MAE_mean': avg_mae,
                'MAE_std': float(np.std(mae_list)),
                'RMSE_mean': avg_rmse,
                'RMSE_std': float(np.std(rmse_list)),
                'Time_ms_mean': avg_time,
                'target_abs_mean': avg_target_abs_mean,
            }
            
            # Add coverage columns
            for q in QUANTILE_LEVELS:
                row[f'coverage_{q:.2f}'] = coverage[q]
            
            all_results.append(row)
            
            print(f"    Seed={seed} | MASE={avg_mase:.4f}±{np.std(all_mase):.4f} | "
                  f"WQL={avg_wql:.4f}±{np.std(all_wql):.4f} | Time={avg_time:.1f}ms")
            
            # Save horizon breakdown
            if output_dir:
                hb_path = os.path.join(
                    output_dir, 'horizon_breakdown',
                    f'{dataset_name}_{adapter_kind}_H{H}_seed{seed}.csv'
                )
                os.makedirs(os.path.dirname(hb_path), exist_ok=True)
                horizon_df.to_csv(hb_path, index=False)

                # Per-origin MASE/WQL, paired by origin_idx across adapters —
                # required for significance testing (DM / Wilcoxon on matched
                # forecast origins), which aggregated stats cannot support.
                po_path = os.path.join(
                    output_dir, 'per_origin',
                    f'{dataset_name}_{adapter_kind}_H{H}_seed{seed}.csv'
                )
                os.makedirs(os.path.dirname(po_path), exist_ok=True)
                pd.DataFrame({
                    'origin_idx': origin_idx_list,
                    'MASE': all_mase,
                    'WQL': all_wql,
                }).to_csv(po_path, index=False)

    return all_results


# ===========================================================================
# Calibration Plot
# ===========================================================================

def plot_calibration(results_df: pd.DataFrame, output_dir: str,
                     dataset_name: str = None):
    """
    Plot calibration diagram: nominal quantile vs empirical coverage.
    One line per adapter.
    """
    coverage_cols = [c for c in results_df.columns if c.startswith('coverage_')]
    if not coverage_cols:
        return
    
    nominal = [float(c.replace('coverage_', '')) for c in coverage_cols]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    
    # Filter by dataset if specified
    df_plot = results_df
    if dataset_name:
        df_plot = results_df[results_df['dataset'] == dataset_name]
    
    for adapter in df_plot['adapter'].unique():
        adapter_data = df_plot[df_plot['adapter'] == adapter]
        # Average coverage across seeds and horizons
        empirical = [adapter_data[c].mean() for c in coverage_cols]
        ax.plot(nominal, empirical, 'o-', label=adapter, markersize=5)
    
    ax.set_xlabel('Nominal quantile level (τ)', fontsize=12)
    ax.set_ylabel('Empirical coverage', fontsize=12)
    ax.set_title(f'Calibration Diagram{" — " + dataset_name if dataset_name else ""}',
                 fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    suffix = f'_{dataset_name}' if dataset_name else ''
    fig.savefig(os.path.join(output_dir, f'calibration{suffix}.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved calibration plot: calibration{suffix}.png")


# ===========================================================================
# Summary Table Generation
# ===========================================================================

def generate_summary_tables(results_df: pd.DataFrame, output_dir: str):
    """
    Generate per-dataset × per-horizon summary tables with mean ± std across seeds.
    """
    summary_rows = []
    
    for (ds, adapter, H), group in results_df.groupby(
        ['dataset', 'adapter', 'horizon']
    ):
        row = {
            'dataset': ds,
            'adapter': adapter,
            'horizon': H,
            'n_seeds': len(group),
            'MASE': f"{group['MASE_mean'].mean():.4f}±{group['MASE_mean'].std():.4f}" if len(group) > 1 else f"{group['MASE_mean'].mean():.4f}",
            'WQL': f"{group['WQL_mean'].mean():.4f}±{group['WQL_mean'].std():.4f}" if len(group) > 1 else f"{group['WQL_mean'].mean():.4f}",
            'Time_ms': f"{group['Time_ms_mean'].mean():.1f}",
            'MASE_val': group['MASE_mean'].mean(),
            'WQL_val': group['WQL_mean'].mean(),
            'target_abs_mean': group['target_abs_mean'].mean(),
        }
        
        # Add mean coverage
        coverage_cols = [c for c in results_df.columns if c.startswith('coverage_')]
        for c in coverage_cols:
            row[c] = group[c].mean()
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Print per-dataset tables
    for ds in summary_df['dataset'].unique():
        ds_data = summary_df[summary_df['dataset'] == ds]
        print(f"\n{'='*70}")
        print(f"  DATASET: {ds.upper()}")
        print(f"{'='*70}")
        
        for H in sorted(ds_data['horizon'].unique()):
            h_data = ds_data[ds_data['horizon'] == H].copy()
            print(f"\n  Horizon H={H}:")
            print(f"  {'Adapter':<15} {'MASE↓':<20} {'WQL↓':<20} {'Time(ms)↓':<10}")
            print(f"  {'-'*65}")
            
            # Mark best
            best_mase = h_data['MASE_val'].min()
            best_wql = h_data['WQL_val'].min()
            
            for _, r in h_data.iterrows():
                mase_str = r['MASE']
                wql_str = r['WQL']
                if abs(r['MASE_val'] - best_mase) < 1e-8:
                    mase_str += ' *'
                if abs(r['WQL_val'] - best_wql) < 1e-8:
                    wql_str += ' *'
                print(f"  {r['adapter']:<15} {mase_str:<20} {wql_str:<20} {r['Time_ms']:<10}")
    
    # Save — keep target_abs_mean so build_paper_artifacts can compute nWQL
    out_path = os.path.join(output_dir, 'summary_table.csv')
    summary_df.to_csv(out_path, index=False)
    print(f"\n  Summary saved to: {out_path}")
    
    return summary_df


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CMF Revision Experiments — Rolling-window evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--datasets', nargs='+',
                        default=['etth2'],
                        choices=sorted(cmf.DATASET_CONFIGS.keys()),
                        help='Datasets to evaluate (choices come from '
                             'src/cmf_core.py::DATASET_CONFIGS, so the eight '
                             'paper datasets are all accepted)')
    parser.add_argument('--adapters', nargs='+',
                        default=['ridge'],
                        help='Adapters to evaluate. Use "all" for all CMF adapters + baselines, '
                             'or "cmf" for CMF adapters only.')
    parser.add_argument('--horizons', nargs='+', type=int,
                        default=[24],
                        help='Forecast horizons (e.g. 24 48 96 192 336 720). '
                             'Standard ETT benchmarks use 96 192 336 720.')
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of random seeds (for neural adapters)')
    parser.add_argument('--context_len', type=int, default=512,
                        help='Context length L')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device for Chronos inference')
    parser.add_argument('--max_origins', type=int, default=None,
                        help='Max forecast origins per horizon (for quick tests)')
    parser.add_argument('--output_dir', type=str, default='results/revision',
                        help='Output directory for results')
    parser.add_argument('--write_paper_summary', action='store_true',
                        help='Also write summary_table_paper.csv with paper-facing filtering and nWQL')
    parser.add_argument('--features', type=str, default=None,
                        help='Comma-separated lists of features to include (e.g. OT,MUFL)')
    parser.add_argument('--no_scale_restoration', action='store_true',
                        help='Ablation: skip scale restoration before Chronos inference')
    parser.add_argument('--include_zero_future', action='store_true',
                        help='Sensitivity protocol: retain origins whose realized '
                             'forecast horizon is identically zero')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'],
                        help="Origin split: 'test' (paper-facing, default) or 'val' "
                             "(validation-split pass for CMF-Select adapter ranking)")
    # ── Protocol flags ────────────────────────────────────────────────────
    # These were previously hard-coded inside run_experiment(), which meant the
    # CLI silently ran chronos-t5-SMALL at S=100 — neither of which matches the
    # canonical paper protocol (t5-base, S=50). Exposed here so every run is
    # explicit and recorded in run_config.json.
    parser.add_argument('--model', type=str, default='amazon/chronos-t5-base',
                        help='Chronos backbone. Canonical paper runs use chronos-t5-base.')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Chronos sample paths S. Canonical paper protocol: 50.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Forecast origins per batched Chronos call. Raise on '
                             'large-VRAM GPUs (A100: 16); lower on OOM.')
    parser.add_argument('--n_fit_workers', type=int, default=8,
                        help='Threads for CPU-side adapter fitting')
    parser.add_argument('--ridge_lam', type=float, default=None,
                        help='Override ridge L2 strength (regularisation-sensitivity '
                             'study). Default None keeps ADAPTER_CONFIGS value (1e-2).')
    parser.add_argument('--tag', type=str, default=None,
                        help='Free-text label recorded in run_config.json')

    args = parser.parse_args()

    # Adapter hyperparameter overrides from CLI
    adapter_kwargs_override = {}
    if args.ridge_lam is not None:
        adapter_kwargs_override['ridge_lam'] = args.ridge_lam
    
    # Resolve adapter list
    if 'all' in args.adapters:
        adapters = ALL_METHODS
    elif 'cmf' in args.adapters:
        adapters = CMF_ADAPTERS
    else:
        adapters = args.adapters
    
    seeds = list(range(42, 42 + args.seeds))
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("CMF REVISION EXPERIMENTS")
    print("=" * 70)
    print(f"  Datasets:    {args.datasets}")
    print(f"  Adapters:    {adapters}")
    print(f"  Horizons:    {args.horizons}")
    print(f"  Seeds:       {seeds}")
    print(f"  Context:     L={args.context_len}")
    print(f"  Device:      {args.device}")
    print(f"  Max origins: {args.max_origins or 'all'}")
    print(f"  Split:       {args.split}")
    print(f"  Model:       {args.model}")
    print(f"  Samples S:   {args.num_samples}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Output:      {args.output_dir}")
    print("=" * 70)

    # Provenance: without this, a results directory cannot be traced back to the
    # backbone / S / split that produced it — the exact ambiguity that made the
    # t5-small-vs-t5-base question unanswerable for earlier runs.
    try:
        import subprocess
        _commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                 capture_output=True, text=True,
                                 cwd=os.path.dirname(os.path.abspath(__file__))
                                 ).stdout.strip() or 'unknown'
    except Exception:
        _commit = 'unknown'
    run_config = {
        'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'git_commit': _commit,
        'tag': args.tag,
        'model': args.model,
        'num_samples': args.num_samples,
        'context_len': args.context_len,
        'horizons': args.horizons,
        'datasets': args.datasets,
        'adapters': adapters,
        'seeds': seeds,
        'split': args.split,
        'max_origins': args.max_origins,
        'batch_size': args.batch_size,
        'no_scale_restoration': args.no_scale_restoration,
        'include_zero_future': args.include_zero_future,
        'adapter_kwargs_override': adapter_kwargs_override,
        'quantile_levels': QUANTILE_LEVELS,
    }
    with open(os.path.join(args.output_dir, 'run_config.json'), 'w') as _f:
        json.dump(run_config, _f, indent=2)
    print(f"Run config written to: {args.output_dir}/run_config.json")
    
    all_results = []
    
    for dataset_name in args.datasets:
        for adapter_kind in adapters:
            print(f"\n{'─'*70}")
            print(f"  [{dataset_name.upper()}] Adapter: {adapter_kind}")
            print(f"{'─'*70}")
            
            try:
                results = run_experiment(
                    dataset_name=dataset_name,
                    adapter_kind=adapter_kind,
                    horizons=args.horizons,
                    seeds=seeds,
                    context_len=args.context_len,
                    device_map=args.device,
                    max_origins=args.max_origins,
                    output_dir=args.output_dir,
                    split=args.split,
                    chronos_model_id=args.model,
                    num_samples=args.num_samples,
                    batch_size=args.batch_size,
                    n_fit_workers=args.n_fit_workers,
                    no_scale_restoration=args.no_scale_restoration,
                    include_zero_future=args.include_zero_future,
                    adapter_kwargs_override=adapter_kwargs_override,
                )
                all_results.extend(results)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    if not all_results:
        print("\nNo results collected!")
        return
    
    # Save raw results
    results_df = pd.DataFrame(all_results)
    raw_path = os.path.join(args.output_dir, 'raw_results.csv')
    results_df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved to: {raw_path}")
    
    # Save as JSON too for programmatic access
    json_path = os.path.join(args.output_dir, 'raw_results.json')
    results_df.to_json(json_path, orient='records', indent=2)
    
    # Generate summary table
    summary_df = generate_summary_tables(results_df, args.output_dir)
    if args.write_paper_summary:
        paper_summary = build_paper_summary_table(summary_df)
        paper_summary_path = os.path.join(args.output_dir, 'summary_table_paper.csv')
        paper_summary.to_csv(paper_summary_path, index=False)
        print(f"Paper summary saved to: {paper_summary_path}")
    
    # Generate calibration plots per dataset
    for ds in args.datasets:
        plot_calibration(results_df, args.output_dir, dataset_name=ds)
    
    # Also generate combined calibration plot
    if len(args.datasets) > 1:
        plot_calibration(results_df, args.output_dir, dataset_name=None)
    
    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results in: {args.output_dir}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
