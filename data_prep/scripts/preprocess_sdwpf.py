"""
Preprocess SDWPF dataset for ST-Chronos framework.

Converts raw SDWPF CSVs into:
  1. sdwpf.h5  — HDF5 file with DatetimeIndex (compatible with data_loader.py)
  2. adj_mx_sdwpf.pkl — Distance-based adjacency matrix from turbine coordinates
  3. turb_locations.csv — Clean turbine location file for visualization

The SDWPF dataset has 134 turbines, 245 days of data at 10-min intervals.
We use the "Patv" (active power) column as the target variable.

Usage:
  python preprocess_sdwpf.py
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


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # STC-code/
DATA_DIR = os.path.join(BASE_DIR, "data", "SDWPF")

RAW_DATA_FILE = os.path.join(DATA_DIR, "sdwpf_245days_v1.csv")
LOCATION_FILE = os.path.join(DATA_DIR, "sdwpf_baidukddcup2022_turb_location.csv")
H5_FILE = os.path.join(DATA_DIR, "sdwpf.h5")
ADJ_FILE = os.path.join(DATA_DIR, "adj_mx_sdwpf.pkl")
CLEAN_LOC_FILE = os.path.join(DATA_DIR, "turb_locations.csv")

# Parameters
DISTANCE_THRESHOLD_PERCENTILE = 20  # Top X% closest pairs become edges
TARGET_COL = "Patv"  # Active Power (target variable)
INTERVAL = "10T"  # 10-minute intervals


def build_distance_adjacency(locations, threshold_percentile=20):
    """
    Build adjacency matrix based on Euclidean distance between turbines.
    
    Uses a distance threshold at the given percentile of all pairwise distances,
    following the approach used for METR-LA (but with Euclidean instead of road distance).
    
    Args:
        locations: DataFrame with 'x' and 'y' columns
        threshold_percentile: percentile of distances to use as threshold
    
    Returns:
        adj_mx: numpy array [N, N] with Gaussian kernel weights
    """
    coords = locations[['x', 'y']].values
    n = len(coords)
    
    # Pairwise Euclidean distances
    dist_mx = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_mx[i, j] = np.sqrt((coords[i, 0] - coords[j, 0])**2 + 
                                     (coords[i, 1] - coords[j, 1])**2)
    
    # Distance threshold (percentile of non-zero distances)
    non_diag = dist_mx[np.triu_indices(n, k=1)]
    threshold = np.percentile(non_diag, threshold_percentile)
    
    # Gaussian kernel: w_ij = exp(-d_ij^2 / sigma^2)
    sigma = np.std(non_diag)
    adj_mx = np.exp(-dist_mx**2 / (sigma**2))
    
    # Apply threshold: zero out distant pairs
    adj_mx[dist_mx > threshold] = 0
    np.fill_diagonal(adj_mx, 0)  # No self-loops
    
    return adj_mx, dist_mx


def main():
    print("=" * 60)
    print("SDWPF Dataset Preprocessing")
    print("=" * 60)
    
    # =========================================================
    # 1. Load and validate raw data
    # =========================================================
    print("\n[1/5] Loading raw data...")
    if not os.path.exists(RAW_DATA_FILE):
        print(f"  ✗ Raw data not found: {RAW_DATA_FILE}")
        print("  Run: python download_sdwpf.py")
        return
    
    df_raw = pd.read_csv(RAW_DATA_FILE)
    print(f"  Raw shape: {df_raw.shape}")
    print(f"  Columns: {list(df_raw.columns)}")
    print(f"  Turbine IDs: {sorted(df_raw['TurbID'].unique())}")
    print(f"  # Turbines: {df_raw['TurbID'].nunique()}")
    
    # =========================================================
    # 2. Pivot to wide format: [Time, Turbines]
    # =========================================================
    print(f"\n[2/5] Pivoting '{TARGET_COL}' to wide format...")
    
    # Parse datetime
    if 'Day' in df_raw.columns and 'Tmstamp' in df_raw.columns:
        # Day column is a day index (can exceed 31), not a calendar day.
        # Build datetime from base date + day offset + time.
        base_date = pd.Timestamp("2020-01-01")
        df_raw['datetime'] = (
            base_date
            + pd.to_timedelta(df_raw['Day'].astype(int) - 1, unit='D')
            + pd.to_timedelta(df_raw['Tmstamp'] + ':00')
        )
    
    # Pivot: rows=timestamps, columns=turbines, values=Patv
    df_pivot = df_raw.pivot_table(
        index='datetime',
        columns='TurbID',
        values=TARGET_COL,
        aggfunc='mean'
    )
    df_pivot = df_pivot.sort_index()
    
    # Rename columns to turbine_0, turbine_1, ...
    turbine_ids = sorted(df_pivot.columns)
    col_map = {tid: f"turbine_{i}" for i, tid in enumerate(turbine_ids)}
    df_pivot = df_pivot.rename(columns=col_map)
    
    n_timesteps, n_nodes = df_pivot.shape
    print(f"  Pivot shape: {n_timesteps} timesteps × {n_nodes} turbines")
    print(f"  Time range: {df_pivot.index[0]} → {df_pivot.index[-1]}")
    print(f"  Value range: [{df_pivot.values.min():.2f}, {df_pivot.values.max():.2f}]")
    
    # Handle missing values and negative power
    df_pivot = df_pivot.fillna(0)
    df_pivot = df_pivot.clip(lower=0)  # Negative power values → 0
    
    zero_frac = (df_pivot.values == 0).sum() / df_pivot.values.size
    print(f"  Zero fraction: {zero_frac:.2%}")
    print(f"  Mean power: {df_pivot.values.mean():.2f}")
    
    # =========================================================
    # 3. Save as HDF5
    # =========================================================
    print("\n[3/5] Saving HDF5...")
    df_pivot.to_hdf(H5_FILE, key="data", mode="w", complevel=5)
    size_mb = os.path.getsize(H5_FILE) / (1024 * 1024)
    print(f"  ✓ Saved: {H5_FILE} ({size_mb:.1f} MB)")
    
    # =========================================================
    # 4. Build distance-based adjacency matrix
    # =========================================================
    print(f"\n[4/5] Building adjacency matrix (distance-based, threshold={DISTANCE_THRESHOLD_PERCENTILE}th percentile)...")
    
    if os.path.exists(LOCATION_FILE):
        loc_df = pd.read_csv(LOCATION_FILE)
        print(f"  Location columns: {list(loc_df.columns)}")
        
        # Standardize column names
        loc_df.columns = [c.strip().lower() for c in loc_df.columns]
        if 'turbid' in loc_df.columns:
            loc_df = loc_df.rename(columns={'turbid': 'TurbID'})
        elif 'turbine_id' in loc_df.columns:
            loc_df = loc_df.rename(columns={'turbine_id': 'TurbID'})
        
        # Sort by turbine ID to match pivot column order
        loc_df = loc_df.sort_values('TurbID').reset_index(drop=True)
        
        # Extract x, y coordinates
        x_col = [c for c in loc_df.columns if 'x' in c.lower()][0]
        y_col = [c for c in loc_df.columns if 'y' in c.lower()][0]
        loc_df = loc_df.rename(columns={x_col: 'x', y_col: 'y'})
        
        print(f"  Turbine locations: {len(loc_df)} turbines")
        print(f"  X range: [{loc_df['x'].min():.2f}, {loc_df['x'].max():.2f}]")
        print(f"  Y range: [{loc_df['y'].min():.2f}, {loc_df['y'].max():.2f}]")
        
        adj_mx, dist_mx = build_distance_adjacency(
            loc_df, threshold_percentile=DISTANCE_THRESHOLD_PERCENTILE
        )
        
        n_edges = (adj_mx > 0).sum()
        max_edges = n_nodes * (n_nodes - 1)
        print(f"  Edges: {n_edges} / {max_edges} ({n_edges/max_edges:.1%} density)")
        print(f"  Avg degree: {n_edges / n_nodes:.1f}")
        
        # Save clean location file
        loc_clean = loc_df[['TurbID', 'x', 'y']].copy()
        loc_clean.to_csv(CLEAN_LOC_FILE, index=False)
        print(f"  ✓ Saved clean locations: {CLEAN_LOC_FILE}")
    else:
        print(f"  ✗ Location file not found: {LOCATION_FILE}")
        print("  Building correlation-based adjacency instead...")
        
        # Fallback: correlation-based (like Solar-AL)
        active_data = df_pivot.values[df_pivot.values.sum(axis=1) > 0]
        corr = np.corrcoef(active_data.T)
        corr = np.nan_to_num(corr, nan=0.0)
        adj_mx = np.zeros_like(corr)
        adj_mx[corr >= 0.9] = corr[corr >= 0.9]
        np.fill_diagonal(adj_mx, 0)
        
        n_edges = (adj_mx > 0).sum()
        max_edges = n_nodes * (n_nodes - 1)
        print(f"  Edges: {n_edges} / {max_edges} ({n_edges/max_edges:.1%} density)")
    
    # =========================================================
    # 5. Save adjacency as pickle
    # =========================================================
    print("\n[5/5] Saving adjacency matrix...")
    sensor_ids = list(df_pivot.columns)
    sensor_id_to_ind = {sid: i for i, sid in enumerate(sensor_ids)}
    
    with open(ADJ_FILE, "wb") as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f)
    print(f"  ✓ Saved: {ADJ_FILE}")
    
    # Summary
    report_normalization_stats(df_pivot, "sdwpf")
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Dataset:    SDWPF (Wind Power)")
    print(f"  Nodes:      {n_nodes}")
    print(f"  Timesteps:  {n_timesteps}")
    print(f"  Interval:   10 minutes")
    print(f"  Edges:      {n_edges}")
    print(f"  Files:      {H5_FILE}")
    print(f"              {ADJ_FILE}")
    if os.path.exists(CLEAN_LOC_FILE):
        print(f"              {CLEAN_LOC_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
