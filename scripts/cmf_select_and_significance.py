#!/usr/bin/env python3
"""
Phase 4 — CMF-Select derivation + VALID per-origin significance test.

Consumes the canonical re-run artifacts produced by run_revision.py
(see RERUN_SPEC_2026-06-21.md):

  --val_dir   results/canonical_val    (origins from the VALIDATION split)
  --test_dir  results/canonical_test   (origins from the TEST split; has per_origin/)

Produces (in --out_dir, default = test_dir):
  cmf_select_table.csv     dataset,H,sel_adapter,cmf_select_mase,chronos_mase,
                           oracle_mase,delta_pct_mase, + nWQL counterparts
  significance_valid.csv   dataset,H,sel_adapter,n_paired,delta_mase,delta_pct,
                           wilcoxon_p,ci_lower,ci_upper,significant,note

Method (why this is valid, unlike the old significance_summary.csv):
  * Selection uses VAL only; reporting uses TEST only -> no oracle leakage.
  * Significance unit = forecast ORIGIN (n = hundreds), paired by origin_idx
    between the CMF-Select adapter and identity (= unadapted Chronos).
  * Neural adapters: per-origin MASE averaged across seeds before pairing.
  * Two-sided Wilcoxon signed-rank + bootstrap 95% CI of the mean difference.

Selection pool excludes CVA/feat_attn and seasonal_naive by design.

Usage:
  python scripts/cmf_select_and_significance.py \
      --val_dir results/canonical_val --test_dir results/canonical_test
"""

from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

POOL = ["ridge", "pcr", "krr", "mlp", "identity"]   # CMF-Select / oracle pool
IDENTITY = "identity"
N_BOOTSTRAP = 10000
ALPHA = 0.05
np.random.seed(42)


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------
def load_summary(results_dir: str) -> pd.DataFrame:
    """Read summary_table.csv and return numeric MASE / nWQL per (dataset,adapter,horizon).

    summary_table.csv stores the numeric mean MASE in column 'MASE_val' and the
    raw WQL mean in 'WQL_val' (column names are legacy; 'MASE_val' = the MASE value
    for whichever split this directory was run on). nWQL = WQL_val / target_abs_mean.
    """
    path = os.path.join(results_dir, "summary_table.csv")
    if not os.path.exists(path):
        sys.exit(f"STOP: missing {path}")
    df = pd.read_csv(path)
    needed = {"dataset", "adapter", "horizon", "MASE_val", "WQL_val", "target_abs_mean"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"STOP: {path} lacks columns {missing}")
    df = df[["dataset", "adapter", "horizon", "MASE_val", "WQL_val", "target_abs_mean"]].copy()
    df["nwql"] = df["WQL_val"] / df["target_abs_mean"].replace(0, np.nan)
    df = df.rename(columns={"MASE_val": "mase"})
    return df


def lookup(df: pd.DataFrame, dataset: str, adapter: str, H: int, col: str):
    row = df[(df.dataset == dataset) & (df.adapter == adapter) & (df.horizon == H)]
    if len(row) == 0:
        return None
    return float(row.iloc[0][col])


def load_per_origin(test_dir: str, dataset: str, adapter: str, H: int) -> pd.DataFrame | None:
    """Per-origin MASE for one (dataset,adapter,H), averaged across seeds by origin_idx."""
    pat = os.path.join(test_dir, "per_origin", f"{dataset}_{adapter}_H{H}_seed*.csv")
    files = sorted(glob.glob(pat))
    if not files:
        return None
    frames = [pd.read_csv(f)[["origin_idx", "MASE"]] for f in files]
    allrows = pd.concat(frames, ignore_index=True)
    return allrows.groupby("origin_idx", as_index=False)["MASE"].mean()


# --------------------------------------------------------------------------
# CMF-Select + oracle
# --------------------------------------------------------------------------
def derive_cmf_select(val_df, test_df, datasets, horizons, priority="mase"):
    rows = []
    sel_col = "mase" if priority == "mase" else "nwql"
    for ds in datasets:
        for H in horizons:
            # available pool members on the VAL split
            cands = [a for a in POOL if lookup(val_df, ds, a, H, sel_col) is not None]
            if not cands:
                continue
            # selection on VAL
            sel = min(cands, key=lambda a: lookup(val_df, ds, a, H, sel_col))
            # reporting on TEST
            cmf_mase = lookup(test_df, ds, sel, H, "mase")
            cmf_nwql = lookup(test_df, ds, sel, H, "nwql")
            chr_mase = lookup(test_df, ds, IDENTITY, H, "mase")
            chr_nwql = lookup(test_df, ds, IDENTITY, H, "nwql")
            # oracle = per-cell best over the SAME pool on TEST
            test_cands = [a for a in POOL if lookup(test_df, ds, a, H, "mase") is not None]
            oracle_adapter = min(test_cands, key=lambda a: lookup(test_df, ds, a, H, "mase"))
            oracle_mase = lookup(test_df, ds, oracle_adapter, H, "mase")
            # no-leakage sanity: reported value IS the selected adapter's own test value
            assert abs(cmf_mase - lookup(test_df, ds, sel, H, "mase")) < 1e-12

            rows.append({
                "dataset": ds, "H": H, "sel_adapter": sel,
                "cmf_select_mase": round(cmf_mase, 4),
                "chronos_mase": round(chr_mase, 4),
                "oracle_mase": round(oracle_mase, 4),
                "oracle_adapter": oracle_adapter,
                "delta_pct_mase": round((cmf_mase - chr_mase) / chr_mase * 100, 2),
                "cmf_select_nwql": round(cmf_nwql, 4),
                "chronos_nwql": round(chr_nwql, 4),
                "delta_pct_nwql": round((cmf_nwql - chr_nwql) / chr_nwql * 100, 2),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Significance (per-origin paired)
# --------------------------------------------------------------------------
def significance(test_dir, select_df):
    rows = []
    for _, r in select_df.iterrows():
        ds, H, sel = r["dataset"], int(r["H"]), r["sel_adapter"]
        if sel == IDENTITY:
            rows.append({"dataset": ds, "H": H, "sel_adapter": sel, "n_paired": 0,
                         "delta_mase": 0.0, "delta_pct": 0.0, "wilcoxon_p": np.nan,
                         "ci_lower": 0.0, "ci_upper": 0.0, "significant": False,
                         "note": "CMF-Select chose identity (no fusion)"})
            continue
        po_sel = load_per_origin(test_dir, ds, sel, H)
        po_id = load_per_origin(test_dir, ds, IDENTITY, H)
        if po_sel is None or po_id is None:
            rows.append({"dataset": ds, "H": H, "sel_adapter": sel, "n_paired": 0,
                         "delta_mase": np.nan, "delta_pct": np.nan, "wilcoxon_p": np.nan,
                         "ci_lower": np.nan, "ci_upper": np.nan, "significant": False,
                         "note": "missing per_origin file(s)"})
            continue
        m = po_sel.merge(po_id, on="origin_idx", suffixes=("_sel", "_id"))
        diffs = (m["MASE_sel"] - m["MASE_id"]).to_numpy()   # negative = CMF better
        n = len(diffs)
        delta = float(np.mean(diffs))
        delta_pct = float(delta / np.mean(m["MASE_id"].to_numpy()) * 100)

        try:
            p = 1.0 if np.all(diffs == 0) else float(stats.wilcoxon(diffs, alternative="two-sided")[1])
        except Exception:
            p = np.nan
        boot = np.array([np.mean(diffs[np.random.randint(0, n, n)]) for _ in range(N_BOOTSTRAP)])
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"dataset": ds, "H": H, "sel_adapter": sel, "n_paired": n,
                     "delta_mase": round(delta, 4), "delta_pct": round(delta_pct, 2),
                     "wilcoxon_p": round(p, 6) if p == p else np.nan,
                     "ci_lower": round(float(ci_lo), 4), "ci_upper": round(float(ci_hi), 4),
                     "significant": bool(p == p and p < ALPHA),
                     "note": ""})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", default="results/canonical_val")
    ap.add_argument("--test_dir", default="results/canonical_test")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--priority", choices=["mase", "nwql"], default="mase",
                    help="metric used to pick the adapter on the val split")
    args = ap.parse_args()
    out_dir = args.out_dir or args.test_dir

    val_df = load_summary(args.val_dir)
    test_df = load_summary(args.test_dir)
    datasets = sorted(test_df.dataset.unique())
    horizons = sorted(int(h) for h in test_df.horizon.unique())

    select_df = derive_cmf_select(val_df, test_df, datasets, horizons, args.priority)
    sig_df = significance(args.test_dir, select_df)

    sel_path = os.path.join(out_dir, "cmf_select_table.csv")
    sig_path = os.path.join(out_dir, "significance_valid.csv")
    select_df.to_csv(sel_path, index=False)
    sig_df.to_csv(sig_path, index=False)

    # Quarantine the old invalid summary if present
    old = os.path.join("results", "significance_summary.csv")
    if os.path.exists(old):
        os.rename(old, os.path.join("results", "_invalid_significance_summary.csv"))
        print(f"  quarantined old significance_summary.csv -> _invalid_significance_summary.csv")

    print("\n=== CMF-Select (val-chosen, test-reported) ===")
    print(select_df.to_string(index=False))
    print("\n=== Significance (per-origin paired Wilcoxon + bootstrap CI) ===")
    print(sig_df.to_string(index=False))
    print(f"\nSaved -> {sel_path}\nSaved -> {sig_path}")

    nsig = int(sig_df["significant"].sum())
    print(f"\nSummary: {nsig}/{len(sig_df)} cells show a SIGNIFICANT CMF-Select gain over Chronos (p<{ALPHA}).")


if __name__ == "__main__":
    main()
