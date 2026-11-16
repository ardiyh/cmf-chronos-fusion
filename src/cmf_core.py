"""
Chronos Multivariate Fusion (CMF) — Single-File Benchmark
Authors: Ardi Yusri Hilmi, Edi Winarko, Moh Edi Wibowo, Andi Dharmawan
Last Updated: 2026-02-09

See README.md for details on usage, installation, and methodology.
"""

# ==========================
# IMPORTS
# ==========================
# Standard library
from __future__ import annotations
import argparse
import gzip
import io
import time
import urllib.request
from typing import Any, Dict, Tuple

# Third-party
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Local/External - Chronos
try:
    from chronos import BaseChronosPipeline
except ImportError as e:
    BaseChronosPipeline = None
    print("[WARN] Could not import BaseChronosPipeline. Install: pip install chronos-forecasting")
    print(f"       Error: {e}")

# ==========================
# DATASET CONFIG + EDA HELPER
# ==========================
# Dataset configurations
# Each entry contains: url, format type, and optional metadata
# All data paths point to <repo>/data/ (self-contained repo).
# Resolve relative to this file so it works both on DGX (/workspace/CMF-code)
# and on local machines without hardcoding an absolute path.
import os as _os
_DATA_ROOT = _os.environ.get(
    "CMF_DATA_ROOT",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "data"),
)

DATASET_CONFIGS = {
    # ── ETT (Electricity Transformer Temperature) ──────────────────────────────
    # Hourly (ETTh) and 15-min (ETTm) energy load data from two substations.
    # 7 features: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT (target = OT, index 6)
    # Status: CLEAN — no NaN, no Inf, min≈-31, max≈108, float64
    'etth1': {
        'url': f"{_DATA_ROOT}/ETT/etth1.csv",
        'format': 'csv',
        'feature_cols': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'target_col': 'OT',
        'seasonality': 24,          # hourly → daily seasonality
    },
    'etth2': {
        'url': f"{_DATA_ROOT}/ETT/etth2.csv",
        'format': 'csv',
        'feature_cols': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'target_col': 'OT',
        'seasonality': 24,
    },
    'ettm1': {
        'url': f"{_DATA_ROOT}/ETT/ettm1.csv",
        'format': 'csv',
        'feature_cols': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'target_col': 'OT',
        'seasonality': 96,          # 15-min → daily = 24×4
    },
    'ettm2': {
        'url': f"{_DATA_ROOT}/ETT/ettm2.csv",
        'format': 'csv',
        'feature_cols': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'target_col': 'OT',
        'seasonality': 96,
    },

    # ── Solar-AL ──────────────────────────────────────────────────────────────
    # 10-min solar irradiance from 137 stations in Alabama.
    # Status: CLEAN — no NaN/Inf, all ≥ 0, 55% zeros (night-time), max=88.9
    'solar-al': {
        'h5_path': f"{_DATA_ROOT}/SolarAl/solar_al.h5",
        'adj_path': f"{_DATA_ROOT}/SolarAl/adj_mx_solar.pkl",
        'format': 'h5_stc',
        'target_col': 0,
        'seasonality': 144,         # 10-min → daily = 24×6
    },

    # ── OPSD Wind ─────────────────────────────────────────────────────────────
    # Hourly wind power generation from 10 European regions (Open Power System Data).
    # Status: CLEAN — no NaN/Inf, all ≥ 0, <1% zeros, max=45,085 MW
    'opsd-wind': {
        'h5_path': f"{_DATA_ROOT}/OPSD/opsd_wind.h5",
        'adj_path': f"{_DATA_ROOT}/OPSD/adj_mx_opsd.pkl",
        'format': 'h5_stc',
        'target_col': 0,
        'seasonality': 24,          # hourly → daily
    },

    # ── SDWPF (Baidu KDD Cup 2022 Wind Power) ─────────────────────────────────
    # 10-min active power output from 134 wind turbines in China.
    # Status: CLEAN — no NaN/Inf, all ≥ 0, 30% zeros (curtailment/low-wind), max=1,567 kW
    'sdwpf': {
        'h5_path': f"{_DATA_ROOT}/SDWPF/sdwpf.h5",
        'adj_path': f"{_DATA_ROOT}/SDWPF/adj_mx_sdwpf.pkl",
        'format': 'h5_stc',
        'target_col': 0,
        'seasonality': 144,         # 10-min → daily = 24×6
    },

    # ── ECL (Electricity Consuming Load) ──────────────────────────────────────
    # Hourly electricity consumption from 370 clients in Portugal (UCI ML Repo).
    # Status: CLEAN — no NaN/Inf, all ≥ 0, 8% zeros, max=764,000 kWh (raw, not normalised)
    # Note: high-dimensional (370 features) — feat_gate auto-falls back to linear
    'ecl': {
        'h5_path': f"{_DATA_ROOT}/ECL/ecl.h5",
        'adj_path': f"{_DATA_ROOT}/ECL/adj_mx_ecl.pkl",
        'format': 'h5_stc',
        'target_col': 0,
        'seasonality': 24,          # hourly → daily
    },

    # ── ACN-EV (Adaptive Charging Network — EV Charging) ─────────────────────
    # 5-min EV charging sessions from 52 stations at Caltech campus.
    # Status: CLEAN — no NaN/Inf, all ≥ 0
    # Target: AGGREGATE total campus load (sum 52 stations) = 87% non-zero
    #   (Single-station target CA-149 was 99% zero — meaningless for forecasting)
    'acn-ev': {
        'h5_path': f"{_DATA_ROOT}/ACN-EV/acn-ev.h5",
        'adj_path': f"{_DATA_ROOT}/ACN-EV/adj_mx_acn.pkl",
        'format': 'h5_stc',
        'target_col': 0,            # overridden by aggregate_target
        'aggregate_target': True,   # target = sum(all 52 stations) → 'total_load'
        'seasonality': 288,         # 5-min → daily = 24×12
    },
}

# Default feature columns (ETT format)
DEFAULT_FEATURE_COLS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
DEFAULT_TARGET_COL = 'OT'

# Current dataset configuration (set by load_dataset)
_current_feature_cols = DEFAULT_FEATURE_COLS
_current_target_col = DEFAULT_TARGET_COL


def load_dataset(dataset_name: str) -> Tuple[pd.DataFrame, list, str, int]:
    """
    Load a dataset by name and return (dataframe, feature_cols, target_col, seasonality).
    Handles different formats (CSV, txt.gz, etc.)
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    fmt = config.get('format', 'csv')
    seasonality = config.get('seasonality', 24)
    
    if fmt == 'csv':
        url = config['url']
        df = pd.read_csv(url)
        feat_cols = config['feature_cols']
        target_col = config['target_col']
    
    elif fmt == 'txt_gz':
        # Download and decompress gzipped text file
        print(f"  Downloading {dataset_name} dataset (this may take a moment)...")
        with urllib.request.urlopen(url) as response:
            compressed = response.read()
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as f:
            data = np.loadtxt(f, delimiter=',')
        
        # Create DataFrame with generated column names
        n_features = config.get('n_features', data.shape[1])
        col_names = [f'PV_{i}' for i in range(n_features)]
        df = pd.DataFrame(data, columns=col_names)
        
        # Use first 7 columns as features (similar to ETT) or all if fewer
        n_feat = min(7, n_features)
        feat_cols = col_names[:n_feat]
        target_col = config.get('target_col', col_names[0])
        
        # Ensure target is in feature cols
        if target_col not in feat_cols:
            feat_cols = feat_cols[:-1] + [target_col]
        
        print(f"  Loaded {len(df)} rows, {n_features} columns. Using {n_feat} features, target={target_col}")
    
    elif fmt == 'h5_stc':
        h5_path = config['h5_path']
        with h5py.File(h5_path, 'r') as f:
            data_group = f['data']
            col_names = [c.decode() for c in data_group['axis0'][:]]
            timestamps_ns = data_group['axis1'][:]
            values = data_group['block0_values'][:]

        df = pd.DataFrame(values, columns=col_names)
        df.index = pd.to_datetime(timestamps_ns, unit='ns')

        feat_cols = col_names
        target_col_idx = config.get('target_col', 0)
        target_col = col_names[target_col_idx] if isinstance(target_col_idx, int) else target_col_idx

        # Aggregate target: sum all stations → single meaningful target series
        # Used for sparse datasets like ACN-EV where single-station target is 99% zero
        if config.get('aggregate_target', False):
            agg_col = 'total_load'
            df[agg_col] = df[feat_cols].sum(axis=1)
            target_col = agg_col
            # feat_cols stay as individual stations (used as covariates by adapters)

        seasonality = config.get('seasonality', 24)

        print(f"  Loaded {dataset_name}: {len(df)} timesteps, {len(col_names)} stations")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}, target={target_col}")
    
    else:
        raise ValueError(f"Unknown format: {fmt}")

    # Update module-level globals so cmf_forecast / mvts_forecast_and_eval work correctly
    global _current_feature_cols, _current_target_col
    _current_feature_cols = feat_cols
    _current_target_col = target_col

    return df, feat_cols, target_col, seasonality


def run_etth2_eda(df: pd.DataFrame, show_plots: bool = False):
    """Optional quick EDA; disabled by default to keep runs fast."""
    print("First 5 rows:\n", df.head())
    print("\nSummary stats:\n", df[_current_feature_cols].describe())
    if show_plots:
        plt.figure(figsize=(12, 5))
        for f in _current_feature_cols:
            plt.plot(df[f].values[:500], label=f)
        plt.title("First 500 Hours: ETTh2 Features")
        plt.legend(); plt.tight_layout(); plt.show()


# ==========================
# DATA WINDOWING
# ==========================

def slice_window(df: pd.DataFrame, target_col: str, context_len: int = 336, prediction_length: int = 24,
                 find_nonzero_window: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Returns the context dataframe (context_len rows ending right before future window)
    and the future target values of length prediction_length.
    
    If find_nonzero_window=True, searches backward from the end to find a window
    where the target values have some non-zero entries (useful for solar data).
    """
    if not find_nonzero_window:
        # Original behavior: use the last available window
        end = len(df) - prediction_length
        if end <= context_len:
            raise ValueError("Dataset too short for the requested context/prediction lengths.")
        start = end - context_len
        ctx_df = df.iloc[start:end]
        y_future = df[target_col].values[end:end+prediction_length]
        return ctx_df, y_future
    
    # Search for a window where y_future has substantial non-zero values
    target_vals = df[target_col].values
    best_end = len(df) - prediction_length
    
    # Search backwards from the end to find a good evaluation window
    for end in range(len(df) - prediction_length, context_len + prediction_length, -prediction_length):
        y_future = target_vals[end:end+prediction_length]
        # Check if at least 50% of values are non-zero
        nonzero_ratio = (np.abs(y_future) > 0.01).sum() / len(y_future)
        if nonzero_ratio >= 0.5:
            best_end = end
            break
    
    end = best_end
    if end <= context_len:
        raise ValueError("Dataset too short for the requested context/prediction lengths.")
    
    start = end - context_len
    ctx_df = df.iloc[start:end]
    y_future = target_vals[end:end+prediction_length]
    return ctx_df, y_future


# ==========================
# ADAPTERS (FUSION METHODS)
# ==========================
# 0) Dispatchers

def fit_adapter(context_mv: np.ndarray, target_uni: np.ndarray, kind: str = "linear", **kwargs) -> Dict[str, Any]:
    if kind == "linear":
        return fit_linear(context_mv, target_uni)
    if kind == "ridge":
        return fit_ridge(context_mv, target_uni, lam=kwargs.get("ridge_lam", 1e-2))
    if kind == "pcr":
        return fit_pcr(context_mv, target_uni, n_comp=kwargs.get("pcr_n_comp", 3), lam=kwargs.get("pcr_lam", 0.0))
    if kind == "krr":
        return fit_krr(context_mv, target_uni, lam=kwargs.get("krr_lam", 1e-1), gamma=kwargs.get("krr_gamma", None))
    if kind == "mlp":
        return fit_mlp(context_mv, target_uni, hidden=kwargs.get("mlp_hidden", 32),
                       epochs=kwargs.get("mlp_epochs", 300), lr=kwargs.get("mlp_lr", 1e-2),
                       weight_decay=kwargs.get("mlp_wd", 1e-4), device=kwargs.get("mlp_device", "cpu"))
    if kind == "identity":
        return {"kind": "identity"}
    if kind == "uniform":
        return {"kind": "uniform"}
    # MAIN CONTRIBUTION: True cross-variable attention
    if kind == "feat_attn":
        return fit_cross_variable_attention(context_mv, target_uni,
                                            embed_dim=kwargs.get("attn_embed_dim", 32),
                                            num_heads=kwargs.get("attn_num_heads", 1),
                                            epochs=kwargs.get("attn_epochs", 300),
                                            lr=kwargs.get("attn_lr", 1e-2),
                                            device=kwargs.get("attn_device", "cpu"),
                                            target_idx=kwargs.get("target_idx", 0))
    # ABLATION: Simple gating mechanism (for comparison)
    if kind == "feat_gate":
        n_features = context_mv.shape[1]
        if n_features > 100:
            print(f"  [WARN] feat_gate skipped for high-dim data ({n_features} features). Using linear instead.")
            return fit_linear(context_mv, target_uni)
        return fit_feature_gate(context_mv, target_uni, hidden=kwargs.get("fg_hidden", 32),
                                epochs=kwargs.get("fg_epochs", 50), lr=kwargs.get("fg_lr", 1e-3),
                                device=kwargs.get("fg_device", "cpu"))
    raise ValueError(f"Unknown adapter kind: {kind}")


def apply_adapter(context_mv_3d: np.ndarray, adapter: Dict[str, Any]) -> np.ndarray:
    kind = adapter["kind"]
    if kind in ["linear", "ridge", "pcr"]:
        return apply_linear_like(context_mv_3d, adapter["w"], adapter["bias"])  # [B,T]
    if kind == "krr":
        return apply_krr(context_mv_3d, adapter)  # [B,T]
    if kind == "mlp":
        return apply_mlp(context_mv_3d, adapter)  # [B,T]
    if kind == "identity":
        # Identity adapter: just return the univariate target series as is (shape [B, T])
        # Requires context_mv_3d to actually contain the target column at the correct index
        # We handle this by assuming the target is the last feature if not specified, 
        # but for safety in `prepare_chronos_input` we extract `target_uni` separately.
        # Here we do a dummy return, the actual logic is handled before calling this or 
        # by passing the target directly throughout. 
        # However, to keep API consistent: `prepare_chronos_input` calls this.
        # We'll update `prepare_chronos_input` to handle identity better.
        # For now, if we reach here, we assume X3 IS the target univariate series reshaped? 
        # No, X3 is multivariate. 
        # Let's handle 'identity' in `prepare_chronos_input` to skip this or 
        # pass index of target. 
        # Ideally: Identity means "Don't fuse, just use the target column".
        raise NotImplementedError("Identity adapter should be handled in prepare_chronos_input directly.")
    if kind == "uniform":
        return context_mv_3d.mean(axis=-1)  # [B, T]
    # MAIN CONTRIBUTION: True cross-variable attention
    if kind == "feat_attn":
        return apply_cross_variable_attention(context_mv_3d, adapter)  # [B,T]
    # ABLATION: Simple gating mechanism
    if kind == "feat_gate":
        return apply_feature_gate(context_mv_3d, adapter)  # [B,T]
    raise ValueError(f"Unknown adapter kind: {kind}")


# 1) Linear OLS

def fit_linear(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    w_full, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    w, b = w_full[:-1], w_full[-1]
    return {"kind": "linear", "w": w.astype(np.float32), "bias": float(b)}


def apply_linear_like(X3: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Project multivariate context to a single series via weights w and bias b.
    Works for X3 with shape [B, T, d]. Uses tensordot to avoid batched matmul shape issues.
    """
    # y[b,t] = sum_k X3[b,t,k] * w[k] + b
    y = np.tensordot(X3, w, axes=([-1], [0])) + b  # -> [B, T]
    return y


# 2) Ridge Regression (L2)

def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = 1e-2) -> Dict[str, Any]:
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
    w = np.linalg.solve(A, Xc.T @ y)
    b = float(y.mean() - (mu @ w).item())
    return {"kind": "ridge", "w": w.astype(np.float32), "bias": b}


# 3) PCR (PCA + regression)

def fit_pcr(X: np.ndarray, y: np.ndarray, n_comp: int = 3, lam: float = 0.0) -> Dict[str, Any]:
    Xc = X - X.mean(axis=0, keepdims=True)
    mu_y = y.mean()
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T[:, :n_comp]   # [d, k]
    Z = Xc @ V             # [T, k]
    if lam > 0:
        A = Z.T @ Z + lam * np.eye(n_comp)
        beta_z = np.linalg.solve(A, Z.T @ (y - mu_y))
    else:
        beta_z, _, _, _ = np.linalg.lstsq(Z, y - mu_y, rcond=None)
    w = V @ beta_z
    b = float(mu_y - (X.mean(axis=0) @ w))
    return {"kind": "pcr", "w": w.astype(np.float32), "bias": b}


# 4) Kernel Ridge (RBF)

def _rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float = None) -> Tuple[np.ndarray, float]:
    if gamma is None:
        diffs = A[:, None, :] - A[None, :, :]
        d2 = np.sum(diffs**2, axis=-1)
        med = np.median(d2[d2 > 0])
        gamma = 1.0 / (2 * med + 1e-8)
    dists = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * (A @ B.T)
    return np.exp(-gamma * np.maximum(dists, 0.0)), gamma


def fit_krr(X: np.ndarray, y: np.ndarray, lam: float = 1e-1, gamma: float = None) -> Dict[str, Any]:
    K, gamma = _rbf_kernel(X, X, gamma)
    alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), y)
    return {"kind": "krr", "X_train": X.astype(np.float32), "alpha": alpha.astype(np.float32), "gamma": float(gamma)}


def apply_krr(X3: np.ndarray, adapter: Dict[str, Any]) -> np.ndarray:
    X = adapter["X_train"]
    alpha = adapter["alpha"]
    gamma = adapter["gamma"]
    B, T, d = X3.shape
    y_out = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        K, _ = _rbf_kernel(X3[b], X, gamma)
        y_out[b] = K @ alpha
    return y_out


# 5) Tiny MLP mixer
class MLP1(nn.Module):
    def __init__(self, d: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):  # x:[B,T,d]
        y = self.net(x)
        return y.squeeze(-1)


def fit_mlp(X: np.ndarray, y: np.ndarray, hidden: int = 32, epochs: int = 300, lr: float = 1e-2,
            weight_decay: float = 1e-4, device: str = "cpu") -> Dict[str, Any]:
    model = MLP1(X.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    Xt = torch.tensor(X[None, :, :], dtype=torch.float32, device=device)
    yt = torch.tensor(y[None, :], dtype=torch.float32, device=device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = crit(pred, yt)
        loss.backward()
        opt.step()
    model.eval()
    return {"kind": "mlp", "model": model, "device": device}


def apply_mlp(X3: np.ndarray, adapter: Dict[str, Any]) -> np.ndarray:
    model = adapter["model"]
    device = adapter["device"]
    with torch.no_grad():
        out = model(torch.tensor(X3, dtype=torch.float32, device=device)).cpu().numpy()
    return out


# 6) Cross-Variable Attention (TRUE ATTENTION Implementation)
# This implements scaled dot-product attention across the variable dimension
# as described in the paper (Section 4.3, Eq. 2)
class CrossVariableAttention(nn.Module):
    """
    True cross-variable attention fusion for multivariate time series.
    
    For each timestep τ, we:
    1. Embed each variable into m-dimensional space
    2. Compute Q, K, V for all variables
    3. Apply scaled dot-product attention ACROSS variables
    4. Project target variable's attended representation to scalar
    
    Input:  x [B, T, d] - batch, time, variables
    Output: y [B, T]    - fused target sequence
    """
    def __init__(self, d: int, embed_dim: int = 32, num_heads: int = 1, target_idx: int = 0):
        super().__init__()
        self.d = d
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.target_idx = target_idx  # which column in feature list is the target variable

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert 0 <= target_idx < d, f"target_idx {target_idx} out of range for d={d}"

        # Per-variable embedding: map scalar to embed_dim
        self.var_embed = nn.Linear(1, embed_dim)

        # Q, K, V projections (applied to each variable's embedding)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Output projection: map target's attended embedding back to scalar
        self.W_out = nn.Linear(embed_dim, 1)
        
    def forward(self, x):  # x: [B, T, d]
        B, T, d = x.shape
        
        # Embed each variable independently: [B, T, d, 1] -> [B, T, d, embed_dim]
        x_unsqueezed = x.unsqueeze(-1)  # [B, T, d, 1]
        E = self.var_embed(x_unsqueezed)  # [B, T, d, embed_dim]
        
        # Compute Q, K, V for all variables
        Q = self.W_q(E)  # [B, T, d, embed_dim]
        K = self.W_k(E)  # [B, T, d, embed_dim]
        V = self.W_v(E)  # [B, T, d, embed_dim]
        
        # For multi-head attention: reshape to [B, T, d, num_heads, head_dim]
        Q = Q.view(B, T, d, self.num_heads, self.head_dim).transpose(2, 3)  # [B, T, num_heads, d, head_dim]
        K = K.view(B, T, d, self.num_heads, self.head_dim).transpose(2, 3)  # [B, T, num_heads, d, head_dim]
        V = V.view(B, T, d, self.num_heads, self.head_dim).transpose(2, 3)  # [B, T, num_heads, d, head_dim]
        
        # Scaled dot-product attention ACROSS variables (dim=3)
        # Q: [B, T, num_heads, d, head_dim]
        # K^T: [B, T, num_heads, head_dim, d]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # [B, T, num_heads, d, d]
        attn_weights = torch.softmax(scores, dim=-1)  # [B, T, num_heads, d, d]
        
        # Apply attention to values
        Z = torch.matmul(attn_weights, V)  # [B, T, num_heads, d, head_dim]
        
        # Concatenate heads and return to [B, T, d, embed_dim]
        Z = Z.transpose(2, 3).contiguous().view(B, T, d, self.embed_dim)
        
        # Extract target variable (by target_idx) and project to scalar
        target_attended = Z[:, :, self.target_idx, :]  # [B, T, embed_dim]
        y = self.W_out(target_attended).squeeze(-1)  # [B, T]
        
        return y


# Legacy gating mechanism (kept for ablation)
class FeatureGate(nn.Module):
    """
    ABLATION: Simple gating mechanism (NOT true attention).
    This learns per-timestep feature weights via MLP + softmax.
    Kept for comparison to show attention superiority.
    Note: Works poorly on high-dimensional datasets (>100 features) due to softmax diffusion.
    """
    def __init__(self, d: int, hidden: int = 32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d, hidden),
            nn.Tanh(),
            nn.Linear(hidden, d)
        )

    def forward(self, x):  # x:[B,T,d]
        logits = self.fc(x)          # [B,T,d]
        w = torch.softmax(logits, dim=-1)
        y = (w * x).sum(dim=-1)      # [B,T]
        return y


def fit_feature_gate(X: np.ndarray, y: np.ndarray, hidden: int = 32, epochs: int = 300, lr: float = 1e-3,
                     device: str = "cpu") -> Dict[str, Any]:
    """Fit gating mechanism (ABLATION: simple weighted average)"""
    model = FeatureGate(X.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    Xt = torch.tensor(X[None, :, :], dtype=torch.float32, device=device)
    yt = torch.tensor(y[None, :], dtype=torch.float32, device=device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = crit(pred, yt)
        loss.backward()
        opt.step()
    model.eval()
    return {"kind": "feat_gate", "model": model, "device": device}


def apply_feature_gate(X3: np.ndarray, adapter: Dict[str, Any]) -> np.ndarray:
    model = adapter["model"]
    device = adapter["device"]
    with torch.no_grad():
        out = model(torch.tensor(X3, dtype=torch.float32, device=device)).cpu().numpy()
    return out


def fit_cross_variable_attention(X: np.ndarray, y: np.ndarray, embed_dim: int = 32,
                                  num_heads: int = 1, epochs: int = 300, lr: float = 1e-2,
                                  device: str = "cpu", target_idx: int = 0) -> Dict[str, Any]:
    """Fit TRUE cross-variable attention (MAIN CONTRIBUTION)"""
    model = CrossVariableAttention(X.shape[1], embed_dim, num_heads, target_idx=target_idx).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    Xt = torch.tensor(X[None, :, :], dtype=torch.float32, device=device)
    yt = torch.tensor(y[None, :], dtype=torch.float32, device=device)
    
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    patience = 30
    
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = crit(pred, yt)
        loss.backward()
        opt.step()
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    model.eval()
    return {"kind": "feat_attn", "model": model, "device": device}


def apply_cross_variable_attention(X3: np.ndarray, adapter: Dict[str, Any]) -> np.ndarray:
    model = adapter["model"]
    device = adapter["device"]
    with torch.no_grad():
        out = model(torch.tensor(X3, dtype=torch.float32, device=device)).cpu().numpy()
    return out


# ==========================
# CHRONOS INPUT + FORECAST
# ==========================

def prepare_chronos_input(df: pd.DataFrame, context_len: int, feature_cols: list, target_col: str,
                          adapter_kind: str = "linear", **adapter_kwargs) -> Tuple[torch.Tensor, Dict[str, Any], pd.DataFrame, np.ndarray]:
    find_nonzero = adapter_kwargs.pop('find_nonzero_window', False)
    ctx_df, y_future = slice_window(df, target_col, context_len=context_len, 
                                    prediction_length=adapter_kwargs.get('prediction_length', 24),
                                    find_nonzero_window=find_nonzero)
    
    # [REVIEW UPDATE] Identity adapter (Vanilla Chronos) - no fusion
    if adapter_kind == "identity":
        target_uni = ctx_df[target_col].values
        context_adapted = target_uni[None, :] # [1, T]
        context_adapted_tensor = torch.tensor(context_adapted, dtype=torch.float32)
        return context_adapted_tensor, {"kind": "identity"}, ctx_df, y_future

    context_mv = ctx_df[feature_cols].values               # [T,d]
    target_uni = ctx_df[target_col].values                 # [T]
    adapter = fit_adapter(context_mv, target_uni, kind=adapter_kind, **adapter_kwargs)
    context_tensor_mv = context_mv[None, :, :].astype(np.float32)  # [1,T,d]
    context_adapted = apply_adapter(context_tensor_mv, adapter)    # [1,T]
    context_adapted_tensor = torch.tensor(context_adapted, dtype=torch.float32)
    return context_adapted_tensor, adapter, ctx_df, y_future


def cmf_forecast(df: pd.DataFrame, context_len: int = 336, prediction_length: int = 24,
                  adapter_kind: str = "linear", device_map: str = "cpu", pipeline: Any = None,
                  **adapter_kwargs):
    if BaseChronosPipeline is None:
        raise ImportError("Chronos pipeline is unavailable. Please install the proper package.")

    # [REVIEW UPDATE] Seasonal Naive Baseline
    if adapter_kind == 'seasonal_naive':
        # Retrieve context and future
        find_nonzero = adapter_kwargs.get('find_nonzero_window', False)
        ctx_df, y_future = slice_window(df, _current_target_col, context_len=context_len, 
                                        prediction_length=prediction_length,
                                        find_nonzero_window=find_nonzero)
        seasonality = adapter_kwargs.get('seasonality', 24)
        
        # Simple seasonal naive: forecast[t] = past[t - seasonality]
        # We repeat the last 'seasonality' steps of the context to fill the prediction horizon
        history = ctx_df[_current_target_col].values
        if len(history) < seasonality:
             # Fallback to naive last value if history is too short
             forecast = np.full(prediction_length, history[-1])
        else:
             # Repeat the last season
             last_season = history[-seasonality:]
             repetitions = int(np.ceil(prediction_length / seasonality))
             forecast = np.tile(last_season, repetitions)[:prediction_length]

        # Construct dummy quantile output (P10=P50=P90=forecast for deterministic baseline)
        quantiles = np.stack([forecast, forecast, forecast], axis=0) # [3, T]
        mean = forecast
        inf_time_s = 0.0 # Negligible
        
        # We still need context_adapted_tensor for metrics/plotting consistency (just use history)
        context_adapted_tensor = torch.tensor(history[None, :], dtype=torch.float32)
        
        return (quantiles, mean, {"kind": "seasonal_naive"}, inf_time_s, None, context_adapted_tensor, ctx_df, y_future)

    # Prepare fused univariate context
    adapter_kwargs = dict(adapter_kwargs)
    adapter_kwargs.setdefault('prediction_length', prediction_length)
    context_adapted_tensor, adapter, ctx_df, y_future = prepare_chronos_input(
        df, context_len, _current_feature_cols, _current_target_col, adapter_kind, **adapter_kwargs
    )

    # Load or reuse Chronos pipeline
    if pipeline is None:
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map=device_map,
            dtype=torch.float32,
        )

    quantile_levels = [0.1, 0.5, 0.9]

    # Synchronize before timing for accurate GPU/MPS timings
    try:
        if device_map == 'cuda':
            torch.cuda.synchronize()
        elif device_map == 'mps':
            torch.mps.synchronize()
    except Exception:
        pass

    t0 = time.perf_counter()
    quantiles, mean = pipeline.predict_quantiles(
        inputs=context_adapted_tensor,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )
    # Synchronize after compute
    try:
        if device_map == 'cuda':
            torch.cuda.synchronize()
        elif device_map == 'mps':
            torch.mps.synchronize()
    except Exception:
        pass

    inf_time_s = time.perf_counter() - t0

    return (quantiles[0].detach().cpu().numpy(),
            mean[0].detach().cpu().numpy(),
            adapter, inf_time_s, pipeline, context_adapted_tensor, ctx_df, y_future)


# ==========================
# METRICS
# ==========================

def weighted_quantile_loss(y_true: np.ndarray, yq_pred: np.ndarray, 
                          quantiles: tuple = (0.1, 0.5, 0.9)) -> float:
    """
    Calculate Weighted Quantile Loss (WQL) for probabilistic forecasts.
    
    WQL measures how well the predicted quantiles match the actual distribution.
    Lower values indicate better probabilistic calibration.
    
    Args:
        y_true: Actual values, shape [T]
        yq_pred: Predicted quantiles, shape [T, n_quantiles]
        quantiles: Quantile levels (default: 10th, 50th, 90th percentiles)
    
    Returns:
        Average quantile loss across all specified quantiles
    """
    loss = 0.0
    for i, q in enumerate(quantiles):
        e = y_true - yq_pred[:, i]
        loss += 2.0 * np.mean(np.maximum(q * e, (q - 1.0) * e))
    return float(loss / len(quantiles))


def compute_mase(y_true: np.ndarray, y_pred: np.ndarray, 
                 y_insample: np.ndarray, seasonality: int = 1) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).
    
    MASE is scale-independent and compares forecast accuracy to a naive 
    seasonal forecast. MASE < 1 means better than naive, > 1 means worse.
    
    Args:
        y_true: Actual future values
        y_pred: Predicted future values  
        y_insample: Historical values used for scaling
        seasonality: Seasonal period (e.g., 24 for hourly data with daily seasonality)
    
    Returns:
        MASE score (lower is better, < 1 is good)
    """
    d = np.abs(y_insample[seasonality:] - y_insample[:-seasonality]).mean()
    d = max(d, 1e-8)  # Avoid division by zero
    return float(np.abs(y_true - y_pred).mean() / d)


def compute_nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalised MAE: MAE / mean(|y_true|).
    Scale-agnostic for cross-dataset comparison.
    Returns np.nan when mean(|y_true|) < 1e-8 (e.g. sparse EV/solar targets).
    """
    denom = float(np.mean(np.abs(y_true)))
    if denom < 1e-8:
        return float('nan')
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def compute_metrics(y_true: np.ndarray, mean: np.ndarray, quantiles: np.ndarray,
                    insample: np.ndarray, seasonality: int = 24) -> Dict[str, float]:
    """
    Compute forecast evaluation metrics.

    Returns:
        dict containing:
        - RMSE: Root Mean Square Error (lower is better)
        - WQL: Weighted Quantile Loss (lower is better)
        - MASE: Mean Absolute Scaled Error (lower is better, < 1 is good)
        - nMAE: Normalised MAE = MAE / mean(|y_true|); nan for zero-dominated targets
    """
    rmse = float(np.sqrt(np.mean((mean - y_true) ** 2)))
    wql = weighted_quantile_loss(y_true, quantiles)
    mase_val = compute_mase(y_true, mean, insample, seasonality=seasonality)
    nmae_val = compute_nmae(y_true, mean)

    return {"RMSE": rmse, "WQL": wql, "MASE": mase_val, "nMAE": nmae_val}


# ==========================
# SHARED EVALUATION UTILITIES
# (imported by run_revision.py and run_baselines.py to avoid duplication)
# ==========================

def get_split_indices(n: int) -> Tuple[int, int, int]:
    """
    Chronological 60/20/20 train/val/test split.
    Returns (train_end, val_end, test_end).
    """
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    return train_end, val_end, n


def rolling_window_origins(n: int, test_start: int, context_len: int,
                            pred_len: int, stride: int = None,
                            max_origins: int = None) -> list:
    """
    Generate forecast origin indices from the test split (non-overlapping by default).

    A forecast origin t means:
      context window : [t - context_len, t)
      prediction window: [t, t + pred_len)
    """
    if stride is None:
        stride = pred_len
    min_origin = max(context_len, test_start)
    max_origin = n - pred_len
    origins = list(range(min_origin, max_origin + 1, stride))
    if max_origins is not None and len(origins) > max_origins:
        indices = np.linspace(0, len(origins) - 1, max_origins, dtype=int)
        origins = [origins[i] for i in indices]
    return origins


# ==========================
# MULTIVARIATE TRANSFORMER BASELINE (MVTS)
# ==========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, L, d]
    def forward(self, x):  # x: [B, T, d]
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)

class MVTransformer(nn.Module):
    def __init__(self, d_in: int, pred_len: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, pred_len)
        )
    def forward(self, x):  # x:[B,T,d_in]
        h = self.in_proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        h_last = h[:, -1, :]
        y = self.head(h_last)  # [B, H]
        return y

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def build_windows(df: pd.DataFrame, feature_cols: list, target_col: str,
                  context_len: int, pred_len: int, start: int = 0, end: int = None):
    if end is None:
        end = len(df) - pred_len
    X_list, Y_list = [], []
    for s in range(start, end - context_len + 1):
        e = s + context_len
        X_list.append(df[feature_cols].values[s:e])
        Y_list.append(df[target_col].values[e:e+pred_len])
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y


def mvts_forecast_and_eval(df: pd.DataFrame, context_len: int, prediction_length: int,
                           device: str = 'cpu', d_model: int = 64, nhead: int = 4,
                           layers: int = 2, dim_ff: int = 128, dropout: float = 0.1,
                           epochs: int = 10, batch_size: int = 64, lr: float = 1e-3,
                           plot: bool = False):
    X, Y = build_windows(df, _current_feature_cols, _current_target_col, context_len, prediction_length)
    X_train, Y_train = X[:-1], Y[:-1]
    X_eval, Y_eval = X[-1:], Y[-1:]

    model = MVTransformer(d_in=len(_current_feature_cols), pred_len=prediction_length,
                          d_model=d_model, nhead=nhead, num_layers=layers,
                          dim_feedforward=dim_ff, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    ds = WindowDataset(X_train, Y_train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()

    model.eval()
    xb = torch.tensor(X_eval, dtype=torch.float32, device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        yhat = model(xb).cpu().numpy()[0]
    inf_ms = (time.perf_counter() - t0) * 1000.0

    actual = Y_eval[0]
    mean = yhat
    q = np.stack([mean, mean, mean], axis=1)

    insample = X_eval[0, :, _current_feature_cols.index(_current_target_col)]
    m = compute_metrics(actual, mean, q, insample, seasonality=24)

    if plot:
        ctx_start = len(df) - prediction_length - context_len
        ctx_df = df.iloc[ctx_start: ctx_start + context_len]
        plot_forecast(ctx_df, q, mean, actual, show_features=True,
                      title="MVTransformer baseline | multi-horizon",
                      metrics=m, inf_time_ms=inf_ms)

    return m, inf_ms, mean, actual

# ==========================
# PLOTTING
# ==========================

def plot_forecast(ctx_df: pd.DataFrame, quantiles: np.ndarray, mean: np.ndarray, y_future: np.ndarray,
                  show_features: bool = True, title: str = "CMF-Chronos Forecast",
                  metrics: Dict[str, float] = None, inf_time_ms: float = None):
    past = ctx_df[_current_target_col].values
    future_idx = np.arange(len(past), len(past) + len(mean))
    plt.figure(figsize=(12, 6))

    # Past target
    plt.plot(range(len(past)), past, label=f"Past Target ({_current_target_col})", linewidth=1.6)

    # Optional: overlay all variables from context
    if show_features:
        for col in _current_feature_cols:
            if col != _current_target_col:
                plt.plot(range(len(past)), ctx_df[col].values, alpha=0.45, linestyle='--', linewidth=0.9, label=col)

    # Forecast interval first (behind), then mean line, then truth dots
    plt.fill_between(future_idx, quantiles[:, 0], quantiles[:, 2], alpha=0.18, label="P10–P90", zorder=1)
    plt.plot(future_idx, mean, label="Forecast (mean)", linewidth=1.8, zorder=3)
    plt.scatter(future_idx, y_future, label="Actual (future)", s=26, zorder=4)

    # Metrics box
    if metrics is not None:
        mt = metrics
        mt_lines = [
            f"MAE = {mt.get('MAE', float('nan')):.3f}",
            f"RMSE = {mt.get('RMSE', float('nan')):.3f}",
            f"MAPE = {mt.get('MAPE', float('nan')):.2f}%",
            f"ACC = {mt.get('ACC', float('nan')):.2f}%",
            f"WQL = {mt.get('WQL', float('nan')):.4f}",
            f"MASE = {mt.get('MASE', float('nan')):.3f}"
        ]
        if inf_time_ms is not None:
            mt_lines.append(f"Time = {inf_time_ms:.1f} ms")
        text = "\n".join(mt_lines)
        ax = plt.gca()
        ax.text(0.02, 0.98, text, transform=ax.transAxes, ha='left', va='top', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="gray", alpha=0.9))

    plt.legend(ncol=3, fontsize=8)
    plt.title(title)
    plt.xlabel("Time (hours)")
    plt.ylabel("Value")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_benchmark_overlay(ctx_df: pd.DataFrame, y_future: np.ndarray, series_list: list,
                           title: str = "Benchmark overlay (all adapters)",
                           show_features: bool = False):
    past = ctx_df[_current_target_col].values
    horizon = len(y_future)
    future_idx = np.arange(len(past), len(past) + horizon)
    plt.figure(figsize=(12, 6))

    # Past target
    plt.plot(range(len(past)), past, label=f"Past Target ({_current_target_col})", linewidth=1.6)

    # Optional context features (off by default to reduce clutter)
    if show_features:
        for col in _current_feature_cols:
            if col != _current_target_col:
                plt.plot(range(len(past)), ctx_df[col].values,
                         alpha=0.25, linestyle='--', linewidth=0.8, label=col)

    # Overlay forecast means from all adapters
    for item in series_list:
        name = item.get("name", "model")
        mean = item["mean"]
        m = item.get("metrics", {})
        lbl = f"{name} (MAE={m.get('MAE', float('nan')):.3f})"
        plt.plot(future_idx, mean, label=lbl, linewidth=1.8)

    # Ground-truth
    plt.scatter(future_idx, y_future, label="Actual (future)", s=26, zorder=5)

    plt.legend(ncol=2, fontsize=9)
    plt.title(title)
    plt.xlabel("Time (hours)")
    plt.ylabel("Value")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================
# BENCHMARK RUNNER
# ==========================

def run_once(df: pd.DataFrame, adapter_kind: str, context_len: int, prediction_length: int,
             device_map: str, plot: bool, **adapter_kwargs) -> Dict[str, Any]:
    # Build and warm-up a pipeline to avoid first-call timing penalty
    if BaseChronosPipeline is None:
        raise ImportError("Chronos pipeline is unavailable. Please install the proper package.")
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=device_map,
        dtype=torch.float32,
    )
    # Warm-up forward on zeros to trigger kernel/init on GPU/MPS/CPU backends
    warm_ctx = torch.zeros(1, context_len, dtype=torch.float32)
    try:
        if device_map == 'cuda':
            torch.cuda.synchronize()
        elif device_map == 'mps':
            torch.mps.synchronize()
    except Exception:
        pass
    _ = pipeline.predict_quantiles(inputs=warm_ctx, prediction_length=prediction_length, quantile_levels=[0.5])
    try:
        if device_map == 'cuda':
            torch.cuda.synchronize()
        elif device_map == 'mps':
            torch.mps.synchronize()
    except Exception:
        pass

    # Now run the actual forecast with the warmed pipeline
    quantiles, mean, adapter, inf_time_s, _, ctx_tensor, ctx_df, y_future = cmf_forecast(
        df, context_len=context_len, prediction_length=prediction_length,
        adapter_kind=adapter_kind, device_map=device_map, pipeline=pipeline, **adapter_kwargs
    )
    insample = ctx_df[_current_target_col].values
    metrics = compute_metrics(y_future, mean, quantiles, insample, seasonality=24)
    inf_time_ms = inf_time_s * 1000.0
    if plot:
        title = f"Adapter={adapter_kind} | {prediction_length}-step forecast"
        plot_forecast(ctx_df, quantiles, mean, y_future, show_features=True, title=title,
                      metrics=metrics, inf_time_ms=inf_time_ms)
    result = {"adapter": adapter_kind, **metrics, "Time_ms": inf_time_ms}
    return result


def run_benchmark(df: pd.DataFrame, context_len: int, prediction_length: int, device_map: str, plot: bool,
                  overlay: bool, configs: Dict[str, Dict[str, Any]], runs: int = 1, find_nonzero_window: bool = False,
                  seasonality: int = 24, verbose: bool = True):
    # Build a single pipeline and warm it up once to avoid first-call timing penalty
    if BaseChronosPipeline is None:
        raise ImportError("Chronos pipeline is unavailable. Please install the proper package.")
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=device_map,
        dtype=torch.float32,
    )
    # Warm-up with a quick forward on a simple zero context
    warm_ctx = torch.zeros(1, context_len, dtype=torch.float32)
    try:
        if device_map == 'cuda':
            torch.cuda.synchronize()
        elif device_map == 'mps':
            torch.mps.synchronize()
    except Exception:
        pass
    _ = pipeline.predict_quantiles(inputs=warm_ctx, prediction_length=prediction_length, quantile_levels=[0.5])
    try:
        if device_map == 'cuda':
            torch.cuda.synchronize()
        elif device_map == 'mps':
            torch.mps.synchronize()
    except Exception:
        pass

    results = []
    overlay_series = []
    ctx_df0, y_future0 = None, None

    for kind, kwargs in configs.items():
        if verbose:
            print(f"\n=== Running adapter: {kind} ===")
        # Add find_nonzero_window to kwargs for solar-type datasets
        kwargs_with_window = dict(kwargs)
        kwargs_with_window['find_nonzero_window'] = find_nonzero_window
        if kind == 'seasonal_naive':
            # Run once is enough for deterministic baseline
            n_runs = 1
        else:
            n_runs = runs

        run_metrics = []
        for r in range(n_runs):
            if verbose and runs > 1:
                print(f"  Run {r+1}/{runs}...")
                
            quantiles, mean, adapter, inf_time_s, _, ctx_tensor, ctx_df, y_future = cmf_forecast(
                df, context_len=context_len, prediction_length=prediction_length,
                adapter_kind=kind, device_map=device_map, pipeline=pipeline, 
                seasonality=seasonality, **kwargs_with_window
            )
            insample = ctx_df[_current_target_col].values
            metrics = compute_metrics(y_future, mean, quantiles, insample, seasonality=seasonality)
            res = {"adapter": kind, **metrics, "Time_ms": inf_time_s * 1000.0}
            run_metrics.append(res)
        
        # Average metrics over runs
        avg_res = run_metrics[0].copy()
        if n_runs > 1:
            for k in ['RMSE', 'WQL', 'MASE', 'Time_ms']:
                vals = [rm[k] for rm in run_metrics]
                avg_res[k] = np.mean(vals)
                avg_res[k + '_std'] = np.std(vals)
        
        if verbose:
            print("Metrics:", avg_res)
        results.append(avg_res)

        # collect for overlay
        if ctx_df0 is None:
            ctx_df0, y_future0 = ctx_df, y_future
        overlay_series.append({"name": kind, "mean": mean, "metrics": metrics})

        # per-adapter plot only if requested and not doing overlay
        if plot and not overlay:
            title = f"Adapter={kind} | {prediction_length}-step forecast"
            plot_forecast(ctx_df, quantiles, mean, y_future, show_features=True, title=title,
                          metrics=metrics, inf_time_ms=inf_time_s*1000.0)

    # Combined overlay plot
    if plot and overlay and overlay_series:
        plot_benchmark_overlay(ctx_df0, y_future0, overlay_series,
                               title=f"Forecast overlay | {prediction_length}-step ahead",
                               show_features=True)

    # print summary
    if results:
        print("\n=== Summary:")
        df_res = pd.DataFrame(results)
        print(df_res.to_string(index=False))
    return results


# ==========================
# MAIN
# ==========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='etth2',
                    choices=['etth1', 'etth2', 'ettm1', 'ettm2', 'solar', 'all'],
                    help='Dataset to use: etth1, etth2, ettm1, ettm2, solar, or "all" to run all datasets')
    ap.add_argument('--url', type=str, default=None, help='CSV URL for dataset (overrides --dataset)')
    ap.add_argument('--context_len', type=int, default=336)
    ap.add_argument('--prediction_length', type=int, default=24)
    ap.add_argument('--device_map', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])
    ap.add_argument('--adapter_kind', type=str, default='linear',
                    choices=['linear', 'ridge', 'pcr', 'krr', 'mlp', 'feat_attn', 'mvts', 'identity', 'seasonal_naive'])
    ap.add_argument('--plot', dest='plot', action='store_true')
    ap.add_argument('--no-plot', dest='plot', action='store_false'); ap.set_defaults(plot=False)
    ap.add_argument('--benchmark', action='store_true', help='Run all adapters with sensible defaults')
    ap.add_argument('--overlay', action='store_true', help='On benchmark: plot a single overlay of all adapters')
    ap.add_argument('--runs', type=int, default=1, help='Number of runs for statistical significance')

    # Ridge
    ap.add_argument('--ridge_lam', type=float, default=1e-2)
    # PCR
    ap.add_argument('--pcr_n_comp', type=int, default=3)
    ap.add_argument('--pcr_lam', type=float, default=0.0)
    # KRR
    ap.add_argument('--krr_lam', type=float, default=1e-1)
    ap.add_argument('--krr_gamma', type=float, default=None)
    # MLP
    ap.add_argument('--mlp_hidden', type=int, default=32)
    ap.add_argument('--mlp_epochs', type=int, default=300)
    ap.add_argument('--mlp_lr', type=float, default=1e-2)
    ap.add_argument('--mlp_wd', type=float, default=1e-4)
    ap.add_argument('--mlp_device', type=str, default='cpu', choices=['cpu','cuda','mps'])
    # Feature Gate
    ap.add_argument('--fg_hidden', type=int, default=32)
    ap.add_argument('--fg_epochs', type=int, default=300)
    ap.add_argument('--fg_lr', type=float, default=1e-2)
    ap.add_argument('--fg_device', type=str, default='cpu', choices=['cpu','cuda','mps'])
    # MVTS baseline args
    ap.add_argument('--mv_d_model', type=int, default=64)
    ap.add_argument('--mv_nhead', type=int, default=4)
    ap.add_argument('--mv_layers', type=int, default=2)
    ap.add_argument('--mv_dim_ff', type=int, default=128)
    ap.add_argument('--mv_dropout', type=float, default=0.1)
    ap.add_argument('--mv_epochs', type=int, default=10)
    ap.add_argument('--mv_batch', type=int, default=64)
    ap.add_argument('--mv_lr', type=float, default=1e-3)
    ap.add_argument('--mv_device', type=str, default='cpu', choices=['cpu','cuda','mps'])

    args = ap.parse_args()

    # Determine which datasets to run
    if args.url is not None:
        # URL overrides dataset selection - use legacy CSV loading
        datasets_to_run = ['custom']
    elif args.dataset == 'all':
        datasets_to_run = list(DATASET_CONFIGS.keys())
    else:
        datasets_to_run = [args.dataset]

    # Adapter configs for benchmark mode
    configs = {
        'linear': {},
        'ridge': {'ridge_lam': args.ridge_lam},
        'pcr':   {'pcr_n_comp': args.pcr_n_comp, 'pcr_lam': args.pcr_lam},
        'krr':   {'krr_lam': args.krr_lam, 'krr_gamma': args.krr_gamma},
        'mlp':   {'mlp_hidden': args.mlp_hidden, 'mlp_epochs': args.mlp_epochs,
                  'mlp_lr': args.mlp_lr, 'mlp_wd': args.mlp_wd, 'mlp_device': args.mlp_device},
        'feat_attn': {'fg_hidden': args.fg_hidden, 'fg_epochs': args.fg_epochs,
                      'fg_lr': args.fg_lr, 'fg_device': args.fg_device},
        # [REVIEW UPDATE] Baselines
        'identity': {},         # Vanilla Chronos
        'seasonal_naive': {},   # Seasonal Naive
    }

    all_results = []

    for dataset_name in datasets_to_run:
        global _current_feature_cols, _current_target_col
        
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*60}")

        # Load data using appropriate loader
        if dataset_name == 'custom':
            df = pd.read_csv(args.url)
            _current_feature_cols = DEFAULT_FEATURE_COLS
            _current_target_col = DEFAULT_TARGET_COL
            seasonality = 24
        else:
            df, _current_feature_cols, _current_target_col, seasonality = load_dataset(dataset_name)
        
        # Optional quick EDA (disabled by default)
        # run_etth2_eda(df, show_plots=False)

        if args.benchmark:
            # Use find_nonzero_window for datasets with many zero values (like solar)
            find_nonzero = dataset_name == 'solar'
            results = run_benchmark(df, args.context_len, args.prediction_length, args.device_map, 
                                   args.plot, args.overlay, configs, runs=args.runs, 
                                   find_nonzero_window=find_nonzero, seasonality=seasonality,
                                   verbose=False)
            for r in results:
                r['dataset'] = dataset_name
            all_results.extend(results)
        else:
            kwargs = {
                'ridge': {'ridge_lam': args.ridge_lam},
                'pcr':   {'pcr_n_comp': args.pcr_n_comp, 'pcr_lam': args.pcr_lam},
                'krr':   {'krr_lam': args.krr_lam, 'krr_gamma': args.krr_gamma},
                'mlp':   {'mlp_hidden': args.mlp_hidden, 'mlp_epochs': args.mlp_epochs,
                          'mlp_lr': args.mlp_lr, 'mlp_wd': args.mlp_wd, 'mlp_device': args.mlp_device},
                'feat_attn': {'fg_hidden': args.fg_hidden, 'fg_epochs': args.fg_epochs,
                              'fg_lr': args.fg_lr, 'fg_device': args.fg_device},
            }.get(args.adapter_kind, {})

            res = run_once(df, args.adapter_kind, args.context_len, args.prediction_length,
                           args.device_map, args.plot, **kwargs)
            res['dataset'] = dataset_name
            all_results.append(res)
            print(f"\nResult ({dataset_name}):")
            for k, v in res.items():
                if isinstance(v, float):
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")

    # Print combined summary for all datasets with best scores highlighted
    if len(datasets_to_run) > 1 and all_results:
        print(f"\n{'='*60}")
        print("COMBINED SUMMARY (ALL DATASETS)")
        print("(* = best score per dataset)")
        print(f"{'='*60}")
        df_all = pd.DataFrame(all_results)
        # Reorder columns to put dataset first
        cols = ['dataset', 'adapter'] + [c for c in df_all.columns if c not in ['dataset', 'adapter']]
        df_all = df_all[cols]
        
        # Format with best scores marked by asterisk
        # Lower is better for all metrics
        lower_better = ['RMSE', 'WQL', 'MASE', 'Time_ms']
        
        # Create formatted output with asterisks for best values
        lines = []
        header = f"{'dataset':>7} {'adapter':>9}"
        for col in cols[2:]:
            header += f" {col:>10}"
        lines.append(header)
        
        for ds in df_all['dataset'].unique():
            ds_data = df_all[df_all['dataset'] == ds]
            # Find best values for this dataset (lower is better for all)
            best_vals = {}
            for col in lower_better:
                if col in ds_data.columns:
                    best_vals[col] = ds_data[col].min()
            
            for _, row in ds_data.iterrows():
                line = f"{row['dataset']:>7} {row['adapter']:>9}"
                for col in cols[2:]:
                    val = row[col]
                    # Check for std dev column
                    std_val = row.get(col + '_std', None)
                    
                    is_best = False
                    if col in best_vals and abs(val - best_vals[col]) < 1e-9:
                        is_best = True
                    
                    if isinstance(val, float):
                        s_val = f"{val:.4f}"
                        if std_val is not None:
                            s_val += f"±{std_val:.3f}"
                        
                        if is_best:
                            line += f" {s_val:>14}*"
                        else:
                            line += f" {s_val:>15}"
                    else:
                        line += f" {val:>10}"
                lines.append(line)
        
        print('\n'.join(lines))


if __name__ == '__main__':
    main()
