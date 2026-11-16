"""
Preprocess OPSD European Energy Time Series for ST-Chronos framework.

Input  (data/OPSD/opsd_raw.csv):
  Wide-format CSV: index=timestamp, columns=<country>_<variable>
  e.g., DE_wind_generation_actual, FR_solar_generation_actual, ...

Output (data/OPSD/):
  opsd_wind.h5        — [Time × 10 countries] wind generation (MW)
  opsd_solar.h5       — [Time × 10 countries] solar generation (MW)
  adj_mx_opsd.pkl     — geographic distance adjacency (Haversine between centroids)
  opsd_countries.csv  — country list, centroids, node indices

Spatial graph:
  Nodes = 10 European countries: DE FR GB NL BE AT CH DK SE NO
  Adjacency = Gaussian kernel on Haversine distance between country centroids
  (threshold: 20th percentile of all pairwise distances)

Usage:
  python scripts/preprocess_opsd.py
  python scripts/preprocess_opsd.py --variable wind --start 2015-01-01 --end 2019-12-31
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import pickle
from math import radians, sin, cos, sqrt, atan2


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


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "OPSD")

RAW_FILE   = os.path.join(DATA_DIR, "opsd_raw.csv")
WIND_H5    = os.path.join(DATA_DIR, "opsd_wind.h5")
SOLAR_H5   = os.path.join(DATA_DIR, "opsd_solar.h5")
ADJ_FILE   = os.path.join(DATA_DIR, "adj_mx_opsd.pkl")
CTRY_FILE  = os.path.join(DATA_DIR, "opsd_countries.csv")

# 10 European countries with approximate geographic centroids (lat, lon)
COUNTRIES = {
    "DE": (51.17, 10.45),   # Germany
    "FR": (46.23,  2.21),   # France
    "GB": (55.38, -3.44),   # Great Britain
    "NL": (52.13,  5.29),   # Netherlands
    "BE": (50.50,  4.47),   # Belgium
    "AT": (47.52, 14.55),   # Austria
    "CH": (46.82,  8.23),   # Switzerland
    "DK": (56.26,  9.50),   # Denmark
    "SE": (60.13, 18.64),   # Sweden
    "NO": (60.47,  8.47),   # Norway
}

DEFAULT_VARIABLE = "wind"
DEFAULT_START    = "2015-01-01"
DEFAULT_END      = "2019-12-31 23:00"
DIST_THRESHOLD_PERCENTILE = 20

VARIABLE_SUFFIXES = {
    "wind": [
        "wind_generation_actual",
        "wind_onshore_generation_actual",
        "wind_offshore_generation_actual",
    ],
    "solar": [
        "solar_generation_actual",
    ],
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi   = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def build_geographic_adjacency(countries, threshold_pct=20):
    """
    Gaussian kernel adjacency on Haversine distance between country centroids.
    Same approach as SDWPF (distance-based), but with real lat/lon.
    """
    ctry_list = list(countries.keys())
    n = len(ctry_list)
    dist_mx = np.zeros((n, n), dtype=np.float32)

    for i, ci in enumerate(ctry_list):
        for j, cj in enumerate(ctry_list):
            if i != j:
                lat1, lon1 = countries[ci]
                lat2, lon2 = countries[cj]
                dist_mx[i, j] = haversine_km(lat1, lon1, lat2, lon2)

    # Gaussian kernel + threshold
    upper_tri = dist_mx[np.triu_indices(n, k=1)]
    threshold = np.percentile(upper_tri, threshold_pct)
    sigma     = upper_tri.std()
    if sigma == 0:
        sigma = 1.0

    adj_mx = np.exp(-dist_mx**2 / sigma**2)
    adj_mx[dist_mx > threshold] = 0.0
    np.fill_diagonal(adj_mx, 0.0)

    n_edges = (adj_mx > 0).sum()
    print(f"  Distance threshold: {threshold:.1f} km  (σ={sigma:.1f} km)")
    print(f"  Edges: {n_edges} / {n*(n-1)} ({n_edges/(n*(n-1)):.1%} density)")
    return adj_mx, dist_mx


def resolve_country_columns(columns, variable, countries=None):
    """Resolve one canonical source column per country.

    Prefer exact national columns first. If a dataset only exposes regional or
    alias national columns (e.g. GB_UKM_*), fall back to the best matching
    alias instead of dropping the country entirely.
    """
    if countries is None:
        countries = COUNTRIES.keys()

    suffixes = VARIABLE_SUFFIXES[variable]
    columns = list(columns)
    resolved = {}

    alias_token_priority = {
        "GB": ["UKM", "GBN", "NIR"],
    }

    for country in countries:
        chosen = None

        # Prefer exact national columns first.
        for suffix in suffixes:
            exact = f"{country}_{suffix}"
            if exact in columns:
                chosen = exact
                break
        if chosen is not None:
            resolved[country] = chosen
            continue

        candidates = []
        for col in columns:
            if not col.startswith(f"{country}_"):
                continue
            for suffix_rank, suffix in enumerate(suffixes):
                if col.endswith(f"_{suffix}"):
                    alias_rank = 50
                    for rank, token in enumerate(alias_token_priority.get(country, [])):
                        if f"_{token}_" in col:
                            alias_rank = rank
                            break
                    candidates.append((suffix_rank, alias_rank, col.count("_"), col))
                    break

        if candidates:
            chosen = min(candidates)[-1]
            resolved[country] = chosen

    return resolved


def should_read_opsd_column(column_name, countries, variable):
    timestamp_cols = {"", "utc_timestamp", "cet_cest_timestamp", "cet_cet_timestamp"}
    if column_name.lower() in timestamp_cols:
        return True

    suffixes = VARIABLE_SUFFIXES[variable]
    return any(
        column_name.startswith(f"{country}_") and
        any(suffix in column_name for suffix in suffixes)
        for country in countries
    )


def load_and_filter(raw_file, countries, variable, start, end):
    """
    Load large OPSD CSV in chunks, extract only the target country columns.
    Returns a DataFrame [Time × Countries] for the selected variable.
    """
    ctry_list = list(countries.keys())
    suffixes = VARIABLE_SUFFIXES[variable]
    print(f"  Target suffixes: {suffixes}")
    print(f"  Reading CSV in chunks (file may be ~150 MB) ...")

    chunks = []
    for chunk in pd.read_csv(
        raw_file, index_col=0, parse_dates=True,
        usecols=lambda c: should_read_opsd_column(c, ctry_list, variable),
        chunksize=50_000,
        low_memory=True,
    ):
        resolved = resolve_country_columns(chunk.columns, variable, countries=ctry_list)
        if resolved:
            rename = {src: country for country, src in resolved.items()}
            selected = chunk[list(rename.keys())].rename(columns=rename)
            selected = selected.loc[:, ~selected.columns.duplicated()]
            chunks.append(selected)

    if not chunks:
        raise ValueError(
            f"No columns matching OPSD {variable} generation found for countries {ctry_list}.\n"
            f"Available sample: check opsd_raw.csv column names."
        )

    df = pd.concat(chunks)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index = df.index.tz_convert("Europe/Berlin").tz_localize(None)
    df = df.sort_index()

    # Filter time window
    df = df.loc[start:end]

    # Keep only country columns that are present
    present = [c for c in ctry_list if c in df.columns]
    missing = [c for c in ctry_list if c not in df.columns]
    if missing:
        print(f"  ⚠ Countries with no data: {missing} (will be excluded)")
    df = df[present]

    # Fill missing values
    df = df.interpolate(method="linear", limit=6).fillna(0.0)
    df = df.clip(lower=0.0)

    return df, present


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", default=DEFAULT_VARIABLE,
                        choices=["wind", "solar"])
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end",   default=DEFAULT_END)
    parser.add_argument("--dist-threshold-pct", type=int,
                        default=DIST_THRESHOLD_PERCENTILE)
    args = parser.parse_args()

    h5_out = WIND_H5 if args.variable == "wind" else SOLAR_H5
    dataset_key = f"opsd-{args.variable}"

    print("=" * 60)
    print(f"OPSD European {args.variable.title()} Generation Preprocessing")
    print(f"10 countries · hourly · {args.start} → {args.end}")
    print("=" * 60)

    if not os.path.exists(RAW_FILE):
        print(f"  ✗ Raw file not found: {RAW_FILE}")
        print("  Run: python scripts/download_opsd.py")
        sys.exit(1)

    # ── 1. Load and filter ────────────────────────────────────────
    print("\n[1/4] Filtering target columns from raw CSV...")
    df, present_countries = load_and_filter(
        RAW_FILE, COUNTRIES, args.variable, args.start, args.end
    )
    n_timesteps, n_nodes = df.shape
    print(f"  Shape: {n_timesteps:,} timesteps × {n_nodes} countries")
    print(f"  Countries: {present_countries}")
    print(f"  Value range: [{df.values.min():.1f}, {df.values.max():.1f}] MW")
    zero_frac = (df.values == 0).mean()
    print(f"  Zero fraction: {zero_frac:.2%}")

    # ── 2. Save HDF5 ──────────────────────────────────────────────
    print(f"\n[2/4] Saving HDF5...")
    df.to_hdf(h5_out, key="data", mode="w", complevel=5)
    size_mb = os.path.getsize(h5_out) / (1 << 20)
    print(f"  ✓ Saved: {h5_out} ({size_mb:.1f} MB)")

    # Country list
    ctry_df = pd.DataFrame({
        "country":  present_countries,
        "node_idx": range(n_nodes),
        "lat":      [COUNTRIES[c][0] for c in present_countries],
        "lon":      [COUNTRIES[c][1] for c in present_countries],
    })
    ctry_df.to_csv(CTRY_FILE, index=False)
    print(f"  ✓ Saved: {CTRY_FILE}")

    # ── 3. Build adjacency ────────────────────────────────────────
    print(f"\n[3/4] Building geographic adjacency (Haversine distance)...")
    present_dict = {c: COUNTRIES[c] for c in present_countries}
    adj_mx, dist_mx = build_geographic_adjacency(
        present_dict, args.dist_threshold_pct
    )

    # ── 4. Save PKL ───────────────────────────────────────────────
    print(f"\n[4/4] Saving adjacency pickle...")
    sensor_ids       = present_countries
    sensor_id_to_ind = {sid: i for i, sid in enumerate(sensor_ids)}
    with open(ADJ_FILE, "wb") as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f)
    print(f"  ✓ Saved: {ADJ_FILE}")

    duration_years = n_timesteps / (365.25 * 24)
    report_normalization_stats(df, f"opsd-{args.variable}")
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Dataset:    OPSD-{args.variable.upper()} (European Grid)")
    print(f"  Nodes:      {n_nodes} countries")
    print(f"  Timesteps:  {n_timesteps:,}")
    print(f"  Interval:   1 hour")
    print(f"  Duration:   {duration_years:.1f} years")
    print(f"  Variable:   {args.variable} generation actual (MW)")
    print(f"  Adj type:   Haversine distance (geographic)")
    print(f"  Edges:      {(adj_mx > 0).sum()}")
    print(f"  Files:")
    print(f"    {h5_out}")
    print(f"    {ADJ_FILE}")
    print(f"\n  Next: python scripts/train_teacher.py --dataset {dataset_key} --normalize")
    print("=" * 60)


if __name__ == "__main__":
    main()
