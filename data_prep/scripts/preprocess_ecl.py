"""
Preprocess ECL (Electricity Load Diagrams) for ST-Chronos framework.

Input  (data/ECL/LD2011_2014.txt):
  Semicolon-delimited, 15-min intervals, 321 client columns (MT_001…MT_321)
  ~140,256 rows (Jan 2011 – Dec 2014)

Output (data/ECL/):
  ecl.h5           — HDF5 [Time × Clients], resampled to hourly (kWh)
  adj_mx_ecl.pkl   — Correlation-based adjacency (threshold=0.7)
  ecl_clients.csv  — Client list with node index

Design decisions:
  - Resample 15-min → 1-hour by summing (energy, not power → sum is correct)
  - Filter clients with >20% missing → typically leaves ~320 clients
  - Adjacency: Pearson correlation (electricity demand has strong spatial correlation
    among clients in the same grid zone) with threshold=0.7
  - Time window: 2012-01-01 to 2014-12-31 (3 years; skip 2011 warmup period)

Usage:
  python scripts/preprocess_ecl.py
  python scripts/preprocess_ecl.py --start 2012-01-01 --end 2014-12-31 --corr-threshold 0.7
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle


def report_normalization_stats(df, dataset_name, train_ratio=0.7):
    """Report per-node stats and recommend global vs per-node normalization."""
    data = df.values.astype(np.float32)
    n_train = int(len(data) * train_ratio)
    train = data[:n_train]

    node_mean = train.mean(axis=0)
    node_std  = train.std(axis=0)
    cv_mean   = node_mean.std() / (abs(node_mean.mean()) + 1e-8)
    cv_std    = node_std.std()  / (node_std.mean() + 1e-8)

    print("\n" + "-" * 55)
    print(f"  NORMALIZATION STATS: {dataset_name}")
    print(f"  Train timesteps : {n_train:,} / {len(data):,}")
    print(f"  Global mean/std : {train.mean():.4f} / {train.std():.4f}")
    print(f"  Per-node mean   : [{node_mean.min():.4f}, {node_mean.max():.4f}]  CV={cv_mean:.2f}")
    print(f"  Per-node std    : [{node_std.min():.4f},  {node_std.max():.4f}]  CV={cv_std:.2f}")
    if cv_mean > 0.5 or cv_std > 0.5:
        print(f"  ► Recommendation: --normalize --per-node-norm  (heterogeneous nodes)")
    else:
        print(f"  ► Recommendation: --normalize  (nodes are homogeneous)")
    print("-" * 55)


BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data", "ECL")

TXT_FILE  = os.path.join(DATA_DIR, "LD2011_2014.txt")
H5_FILE   = os.path.join(DATA_DIR, "ecl.h5")
ADJ_FILE  = os.path.join(DATA_DIR, "adj_mx_ecl.pkl")
CLI_FILE  = os.path.join(DATA_DIR, "ecl_clients.csv")

DEFAULT_START       = "2012-01-01"
DEFAULT_END         = "2014-12-31 23:00"
DEFAULT_CORR_THR    = 0.7
MAX_MISSING_FRAC    = 0.20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",          default=DEFAULT_START)
    parser.add_argument("--end",            default=DEFAULT_END)
    parser.add_argument("--corr-threshold", type=float, default=DEFAULT_CORR_THR)
    args = parser.parse_args()

    print("=" * 60)
    print("ECL Electricity Load Preprocessing")
    print("=" * 60)

    # ── 1. Load raw text ──────────────────────────────────────────
    print("\n[1/5] Loading raw file...")
    if not os.path.exists(TXT_FILE):
        print(f"  ✗ Not found: {TXT_FILE}")
        print("  Run: python scripts/download_ecl.py")
        sys.exit(1)

    # Large file — read in chunks to avoid OOM
    print("  Reading (this may take 30–60 s for a ~300 MB file)...")
    df_raw = pd.read_csv(
        TXT_FILE,
        sep=";",
        decimal=",",        # European decimal comma
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
        low_memory=True,
    )
    print(f"  Raw shape: {df_raw.shape}")
    print(f"  Columns (first 5): {list(df_raw.columns[:5])}")
    print(f"  Index range: {df_raw.index[0]} → {df_raw.index[-1]}")

    # Keep only MT_* client columns
    client_cols = [c for c in df_raw.columns if c.startswith("MT_")]
    if not client_cols:
        # Try numeric columns fallback
        client_cols = list(df_raw.columns)
    df_raw = df_raw[client_cols].astype(np.float32)
    print(f"  Client columns: {len(client_cols)}")

    # ── 2. Filter time window ─────────────────────────────────────
    print(f"\n[2/5] Filtering time window: {args.start} → {args.end}...")
    df_raw = df_raw.loc[args.start:args.end]
    print(f"  After filter: {df_raw.shape[0]} rows")

    # ── 3. Resample 15-min → hourly (sum kWh) ─────────────────────
    print("\n[3/5] Resampling 15-min → hourly...")
    df_hour = df_raw.resample("1H").sum()
    print(f"  Hourly shape: {df_hour.shape}")

    # Filter clients with too much missing data
    missing_frac = df_hour.isnull().mean()
    good_clients = missing_frac[missing_frac <= MAX_MISSING_FRAC].index.tolist()
    df_hour = df_hour[good_clients].fillna(0.0)
    print(f"  Clients after missing filter: {len(good_clients)} / {len(client_cols)}")

    n_timesteps, n_nodes = df_hour.shape
    print(f"  Final shape: {n_timesteps:,} timesteps × {n_nodes} clients")
    print(f"  Mean hourly consumption: {df_hour.values.mean():.4f} kWh")
    print(f"  Max hourly consumption:  {df_hour.values.max():.4f} kWh")

    # ── 4. Save HDF5 ──────────────────────────────────────────────
    print("\n[4/5] Saving HDF5...")
    df_hour.to_hdf(H5_FILE, key="data", mode="w", complevel=5)
    size_mb = os.path.getsize(H5_FILE) / (1 << 20)
    print(f"  ✓ Saved: {H5_FILE} ({size_mb:.1f} MB)")

    # Save client list
    cli_df = pd.DataFrame({
        "client_id": good_clients,
        "node_idx":  range(n_nodes),
    })
    cli_df.to_csv(CLI_FILE, index=False)
    print(f"  ✓ Saved: {CLI_FILE}")

    # ── 5. Build correlation adjacency ────────────────────────────
    print(f"\n[5/5] Building adjacency (Pearson corr ≥ {args.corr_threshold})...")

    # Sample up to 5000 rows for correlation (full matrix would be huge)
    sample_size = min(len(df_hour), 5000)
    sample_idx  = np.random.choice(len(df_hour), sample_size, replace=False)
    sample_data = df_hour.values[sample_idx]

    corr = np.corrcoef(sample_data.T)          # [N, N]
    corr = np.nan_to_num(corr, nan=0.0)

    adj_mx = np.where(corr >= args.corr_threshold, corr, 0.0).astype(np.float32)
    np.fill_diagonal(adj_mx, 0)

    n_edges  = (adj_mx > 0).sum()
    max_edges = n_nodes * (n_nodes - 1)
    print(f"  Edges: {n_edges} / {max_edges} ({n_edges/max_edges:.1%} density)")
    print(f"  Avg degree: {n_edges / n_nodes:.1f}")

    sensor_ids       = good_clients
    sensor_id_to_ind = {sid: i for i, sid in enumerate(sensor_ids)}
    with open(ADJ_FILE, "wb") as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f)
    print(f"  ✓ Saved: {ADJ_FILE}")

    report_normalization_stats(df_hour, "ecl")
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Dataset:      ECL (Electricity Load Diagrams)")
    print(f"  Nodes:        {n_nodes} clients")
    print(f"  Timesteps:    {n_timesteps:,}")
    print(f"  Interval:     1 hour")
    print(f"  Time range:   {df_hour.index[0]} → {df_hour.index[-1]}")
    print(f"  Edges:        {n_edges} (corr ≥ {args.corr_threshold})")
    print(f"  Files:")
    print(f"    {H5_FILE}")
    print(f"    {ADJ_FILE}")
    print(f"\n  Next: python scripts/train_teacher.py --dataset ecl --normalize")
    print("=" * 60)


if __name__ == "__main__":
    main()
