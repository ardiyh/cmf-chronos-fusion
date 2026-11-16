from __future__ import annotations

import pandas as pd

ACN_EV_EXCLUDED_HORIZONS = {96, 192}


def add_normalized_wql(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if (out["target_abs_mean"] <= 0).any():
        raise ValueError("target_abs_mean must be > 0")
    out["nWQL_val"] = out["WQL_val"] / out["target_abs_mean"]
    return out


def filter_paper_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~((df["dataset"] == "acn-ev") & (df["horizon"].isin(ACN_EV_EXCLUDED_HORIZONS)))
    return df.loc[mask].copy()


def build_paper_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    filtered = filter_paper_rows(df)
    normalized = filtered.copy()

    if "nWQL_val" not in normalized.columns:
        normalized = add_normalized_wql(normalized)

    columns = [
        "dataset",
        "adapter",
        "horizon",
        "MASE_val",
        "nWQL_val",
        "MAE",
        "RMSE",
        "Time_ms",
    ]
    existing_columns = [column for column in columns if column in normalized.columns]
    return normalized[existing_columns].copy()
