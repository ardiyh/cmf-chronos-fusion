"""
CMF-Select — validation-based adapter selection (ICIC-EL R2, reviewer comment 7).

Replaces the oracle "CMF (Best)" column with a deployable rule:
  1. Rank candidate adapters by their VALIDATION-split metric (no test peeking).
  2. Report the selected adapter's TEST-split metric.
  3. Quantify the gap to the per-cell oracle (upper bound).

Inputs
------
--val_summary  : summary_table.csv produced by the `--split val` run
                 (scripts/run_full_benchmark.py ... --split val).
                 Columns used: dataset, adapter, horizon, MASE_val, WQL_val, target_abs_mean.
--test_results : canonical all_results.csv (test split).
                 Columns used: model, dataset, horizon, MASE_mean, nWQL.

NOTE: this script is parameterized and schema-checked, but must be validated end-to-end
once the real validation outputs exist (see CMF_SELECT_EXPERIMENT.md §6).

Usage
-----
python experiments/run_cmf_select.py \
  --val_summary results/cmf_select_val/summary_table.csv \
  --test_results paper_artifacts/paper_canonical_20260417_chronos-t5-base_true_qkv/all_results.csv \
  --candidates ridge pcr krr mlp identity \
  --out results/cmf_select_val/cmf_select_table.csv
"""
from __future__ import annotations
import argparse
import sys

import pandas as pd


def _require_cols(df: pd.DataFrame, cols: list, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] {name} is missing required columns {missing}. "
                 f"Found: {list(df.columns)}")


def load_validation(path: str) -> pd.DataFrame:
    """Return val metrics keyed by (dataset, adapter, horizon): val_MASE, val_nWQL."""
    df = pd.read_csv(path)
    _require_cols(df, ["dataset", "adapter", "horizon", "MASE_val", "WQL_val",
                       "target_abs_mean"], "val_summary")
    out = df[["dataset", "adapter", "horizon"]].copy()
    out["val_MASE"] = df["MASE_val"].astype(float)
    # nWQL = raw WQL normalised by mean absolute target scale
    out["val_nWQL"] = df["WQL_val"].astype(float) / df["target_abs_mean"].astype(float)
    return out


def load_test(path: str) -> pd.DataFrame:
    """Return test metrics keyed by (dataset, adapter, horizon): test_MASE, test_nWQL."""
    df = pd.read_csv(path)
    adapter_col = "model" if "model" in df.columns else "adapter"
    mase_col = "MASE_mean" if "MASE_mean" in df.columns else "MASE_val"
    _require_cols(df, [adapter_col, "dataset", "horizon", mase_col, "nWQL"], "test_results")
    out = df[["dataset", "horizon"]].copy()
    out["adapter"] = df[adapter_col]
    out["test_MASE"] = df[mase_col].astype(float)
    out["test_nWQL"] = df["nWQL"].astype(float)
    # collapse any duplicate (e.g. multi-backbone rows) to the mean
    return out.groupby(["dataset", "horizon", "adapter"], as_index=False).mean(numeric_only=True)


def select(val: pd.DataFrame, test: pd.DataFrame, candidates: list) -> pd.DataFrame:
    val = val[val["adapter"].isin(candidates)]
    test = test[test["adapter"].isin(candidates)]
    rows = []
    for (ds, H), vg in val.groupby(["dataset", "horizon"]):
        tg = test[(test["dataset"] == ds) & (test["horizon"] == H)]
        if tg.empty:
            print(f"[WARN] no test rows for {ds} H={H}; skipping")
            continue
        test_lookup = tg.set_index("adapter")
        oracle_mase = tg["test_MASE"].min()
        oracle_nwql = tg["test_nWQL"].min()
        for objective, valcol, testcol, oracle in [
            ("MASE", "val_MASE", "test_MASE", oracle_mase),
            ("nWQL", "val_nWQL", "test_nWQL", oracle_nwql),
        ]:
            pick = vg.loc[vg[valcol].idxmin(), "adapter"]
            if pick not in test_lookup.index:
                print(f"[WARN] selected '{pick}' has no test row for {ds} H={H}")
                continue
            test_val = float(test_lookup.loc[pick, testcol])
            rows.append({
                "dataset": ds, "horizon": H, "select_by": objective,
                "selected_adapter": pick,
                "test_MASE": float(test_lookup.loc[pick, "test_MASE"]),
                "test_nWQL": float(test_lookup.loc[pick, "test_nWQL"]),
                "oracle": float(oracle),
                "gap_vs_oracle": test_val - float(oracle),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="CMF-Select: validation-based adapter selection")
    ap.add_argument("--val_summary", required=True)
    ap.add_argument("--test_results", required=True)
    ap.add_argument("--candidates", nargs="+",
                    default=["ridge", "pcr", "krr", "mlp", "identity"])
    ap.add_argument("--out", default="cmf_select_table.csv")
    args = ap.parse_args()

    val = load_validation(args.val_summary)
    test = load_test(args.test_results)
    result = select(val, test, args.candidates)
    if result.empty:
        sys.exit("[ERROR] no CMF-Select rows produced — check inputs/candidates.")

    result.to_csv(args.out, index=False)
    print(f"\nCMF-Select table → {args.out}\n")
    with pd.option_context("display.width", 120, "display.max_rows", None):
        print(result.to_string(index=False))

    # Headline: how often CMF-Select matches the oracle, and mean optimality gap.
    for obj in ["MASE", "nWQL"]:
        sub = result[result["select_by"] == obj]
        if sub.empty:
            continue
        hit = (sub["gap_vs_oracle"].abs() < 1e-9).mean() * 100
        print(f"\n[{obj}] CMF-Select matches oracle in {hit:.0f}% of cells | "
              f"mean gap vs oracle = {sub['gap_vs_oracle'].mean():.4f}")


if __name__ == "__main__":
    main()
