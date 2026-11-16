"""All-origin sensitivity check for the realized-zero exclusion mask.

The manuscript's primary analysis excludes an origin whenever its entire
realized horizon satisfies max_h|y_{t+h}| < 1e-6 (paper1_sinergi.md, section
"Evaluation protocol"). That mask affects only Solar-AL and SDWPF, and the
manuscript flags an all-origin rerun as a pending robustness check. This
module recomputes the same paired diagnostics (scripts/sinergi_stats.py)
against the all-origin RTX retry (results/sinergi_gpu_retry_20260815_140w)
and compares them to the retained masked primary run
(results/rtx_20260803_test), cell by cell.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import median

from scripts.sinergi_stats import (
    exact_sign_test,
    holm_adjust,
    load_paired_differences,
    moving_block_bootstrap_ci,
    top_gain_share,
    wilcoxon_signed_rank_p,
)

PRIMARY_DIR = "results/rtx_20260803_test"

# (dataset, adapter, horizon) -> sensitivity (all-origin) result dir
CELLS = {
    ("solar-al", "krr", 24): "results/sinergi_gpu_retry_20260815_140w/zero_future_solar",
    ("solar-al", "krr", 48): "results/sinergi_gpu_retry_20260815_140w/zero_future_solar",
    ("sdwpf", "ridge", 24): "results/sinergi_gpu_retry_20260815_140w/zero_future_sdwpf_h24",
    ("sdwpf", "krr", 48): "results/sinergi_gpu_retry_20260815_140w/zero_future_sdwpf_h48",
}


def _cell_stats(differences, *, seed, n_resamples=10_000):
    n = len(differences)
    sign = exact_sign_test(differences)
    concentration = top_gain_share(differences, fraction=0.05)
    block_length = math.ceil(n ** (1.0 / 3.0))
    block_low, block_high = moving_block_bootstrap_ci(
        differences, block_length=block_length, n_resamples=n_resamples, seed=seed
    )
    return {
        "n": n,
        "mean_diff": sum(differences) / n,
        "median_diff": median(differences),
        "win_rate": sign["wins"] / n,
        "sign_p": sign["p_two_sided"],
        "wilcoxon_p": wilcoxon_signed_rank_p(differences),
        "block_length": block_length,
        "block_ci_lower": block_low,
        "block_ci_upper": block_high,
        "top_5_share": concentration["share"],
    }


def analyze_sensitivity(*, primary_dir=PRIMARY_DIR, cells=None, n_resamples=10_000):
    """Return one row per (dataset, adapter, H) with masked vs all-origin stats."""
    cells = CELLS if cells is None else cells
    rows = []
    raw_p_masked = []
    raw_p_allorigin = []
    for cell_index, ((dataset, adapter, horizon), sensitivity_dir) in enumerate(cells.items()):
        masked = load_paired_differences(primary_dir, dataset, adapter, horizon)
        allorigin = load_paired_differences(sensitivity_dir, dataset, adapter, horizon)
        masked_stats = _cell_stats(masked, seed=1000 + cell_index, n_resamples=n_resamples)
        allorigin_stats = _cell_stats(allorigin, seed=2000 + cell_index, n_resamples=n_resamples)
        row = {
            "dataset": dataset,
            "adapter": adapter,
            "H": horizon,
            "n_masked": masked_stats["n"],
            "n_allorigin": allorigin_stats["n"],
            "n_excluded": allorigin_stats["n"] - masked_stats["n"],
            "mean_diff_masked": masked_stats["mean_diff"],
            "mean_diff_allorigin": allorigin_stats["mean_diff"],
            "median_diff_masked": masked_stats["median_diff"],
            "median_diff_allorigin": allorigin_stats["median_diff"],
            "win_rate_masked": masked_stats["win_rate"],
            "win_rate_allorigin": allorigin_stats["win_rate"],
            "wilcoxon_p_masked": masked_stats["wilcoxon_p"],
            "wilcoxon_p_allorigin": allorigin_stats["wilcoxon_p"],
            "sign_p_masked": masked_stats["sign_p"],
            "sign_p_allorigin": allorigin_stats["sign_p"],
            "block_ci_masked": f"[{masked_stats['block_ci_lower']:.4f},{masked_stats['block_ci_upper']:.4f}]",
            "block_ci_allorigin": f"[{allorigin_stats['block_ci_lower']:.4f},{allorigin_stats['block_ci_upper']:.4f}]",
            "top_5_share_masked": masked_stats["top_5_share"],
            "top_5_share_allorigin": allorigin_stats["top_5_share"],
            "conclusion_flips": (masked_stats["wilcoxon_p"] < 0.05) != (allorigin_stats["wilcoxon_p"] < 0.05),
        }
        raw_p_masked.append(masked_stats["wilcoxon_p"])
        raw_p_allorigin.append(allorigin_stats["wilcoxon_p"])
        rows.append(row)
    for row, holm_m, holm_a in zip(rows, holm_adjust(raw_p_masked), holm_adjust(raw_p_allorigin)):
        row["holm_p_masked"] = holm_m
        row["holm_p_allorigin"] = holm_a
    return rows


def write_sensitivity_csv(destination, *, primary_dir=PRIMARY_DIR, cells=None, n_resamples=10_000):
    rows = analyze_sensitivity(primary_dir=primary_dir, cells=cells, n_resamples=n_resamples)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare masked-primary vs all-origin zero-future sensitivity for Solar-AL and SDWPF."
    )
    parser.add_argument("--output", default="results/sinergi_zero_future_sensitivity.csv")
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    rows = write_sensitivity_csv(args.output, n_resamples=args.resamples)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
