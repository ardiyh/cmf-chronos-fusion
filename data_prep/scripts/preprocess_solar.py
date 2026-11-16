"""
Preprocess Solar-AL dataset for ST-Chronos framework.

Converts raw solar_AL.txt into:
  1. solar_al.h5  — HDF5 file with DatetimeIndex (compatible with data_loader.py)
  2. adj_mx_solar.pkl — Correlation-based adjacency matrix

Usage:
  python preprocess_solar.py
"""

import numpy as np
import pandas as pd
import pickle
import os


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


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root: STC-code/
DATA_DIR = os.path.join(BASE_DIR, "data", "SolarAl")

# Raw file (despite .gz extension, it's plain text)
RAW_FILE = os.path.join(DATA_DIR, "solar_AL.txt.gz")
H5_FILE = os.path.join(DATA_DIR, "solar_al.h5")
ADJ_FILE = os.path.join(DATA_DIR, "adj_mx_solar.pkl")

# Parameters
CORR_THRESHOLD = 0.9  # Minimum Pearson correlation to create edge
INTERVAL = "10T"       # 10-minute intervals (standard for Solar-AL)
START_DATE = "2006-01-01"  # Approximate start date


def main():
    print("=" * 60)
    print("Solar-AL Dataset Preprocessing")
    print("=" * 60)

    # =========================================================
    # 1. Load raw data
    # =========================================================
    print("\n[1/4] Loading raw data...")
    # File has .gz extension but is actually plain text — open explicitly to bypass numpy gzip auto-detection
    with open(RAW_FILE, 'r') as f:
        data = np.loadtxt(f, delimiter=",")
    n_timesteps, n_nodes = data.shape
    print(f"  Shape: {n_timesteps} timesteps × {n_nodes} nodes")
    print(f"  Value range: [{data.min():.4f}, {data.max():.4f}]")
    print(f"  Mean: {data.mean():.4f}, Std: {data.std():.4f}")
    print(f"  Zero fraction: {(data == 0).sum() / data.size:.2%}")

    # =========================================================
    # 2. Create HDF5 with DatetimeIndex
    # =========================================================
    print("\n[2/4] Creating HDF5 file...")
    timestamps = pd.date_range(start=START_DATE, periods=n_timesteps, freq=INTERVAL)
    columns = [f"solar_{i}" for i in range(n_nodes)]
    df = pd.DataFrame(data, index=timestamps, columns=columns)

    df.to_hdf(H5_FILE, key="data", mode="w", complevel=5)
    size_mb = os.path.getsize(H5_FILE) / (1024 * 1024)
    print(f"  ✓ Saved: {H5_FILE} ({size_mb:.1f} MB)")
    print(f"  Time range: {timestamps[0]} → {timestamps[-1]}")

    # =========================================================
    # 3. Build correlation-based adjacency matrix
    # =========================================================
    print(f"\n[3/4] Building adjacency matrix (correlation threshold={CORR_THRESHOLD})...")

    # Filter out nighttime zeros for better correlation
    # Only use daytime hours for correlation computation
    daytime = data[data.sum(axis=1) > 0]
    print(f"  Daytime samples: {len(daytime)} / {n_timesteps} ({len(daytime)/n_timesteps:.1%})")

    # Pearson correlation matrix
    corr_matrix = np.corrcoef(daytime.T)  # [N, N]
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    print(f"  Correlation range: [{corr_matrix.min():.4f}, {corr_matrix.max():.4f}]")
    print(f"  Mean correlation: {corr_matrix.mean():.4f}")

    # Threshold to create adjacency
    adj_mx = np.zeros_like(corr_matrix)
    adj_mx[corr_matrix >= CORR_THRESHOLD] = corr_matrix[corr_matrix >= CORR_THRESHOLD]
    np.fill_diagonal(adj_mx, 0)  # No self-loops in adjacency

    n_edges = (adj_mx > 0).sum()
    max_edges = n_nodes * (n_nodes - 1)
    print(f"  Edges: {n_edges} / {max_edges} ({n_edges/max_edges:.1%} density)")
    print(f"  Avg degree: {n_edges / n_nodes:.1f}")

    # =========================================================
    # 4. Save adjacency as pickle (same format as METR-LA)
    # =========================================================
    print("\n[4/4] Saving adjacency matrix...")
    sensor_ids = columns
    sensor_id_to_ind = {sid: i for i, sid in enumerate(sensor_ids)}

    with open(ADJ_FILE, "wb") as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f)
    print(f"  ✓ Saved: {ADJ_FILE}")

    # Summary
    report_normalization_stats(df, "solar-al")
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Dataset:    Solar-AL")
    print(f"  Nodes:      {n_nodes}")
    print(f"  Timesteps:  {n_timesteps}")
    print(f"  Interval:   10 minutes")
    print(f"  Edges:      {n_edges} (threshold={CORR_THRESHOLD})")
    print(f"  Files:      {H5_FILE}")
    print(f"              {ADJ_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
