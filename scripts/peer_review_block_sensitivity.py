"""A5 block-length sensitivity for the retained paired losses (CPU reanalysis).

Protocol, fixed before this ran:
CMF-paper/submission-sinergi/revisions/peer-review-r1/BLOCK_SENSITIVITY.md.
Re-tests the signed-rank and sign statistics with block sign flips over a fixed
ladder of block lengths and every block-boundary offset, and recomputes the
moving-block bootstrap interval at each length. No forecasts are run.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.sinergi_block_robust import (
    ALL_ORIGIN_DIRS, CONTEXT_LENGTH, PRIMARY_DIR, SELECTIONS, _signed_ranks)
from scripts.sinergi_stats import holm_adjust, load_paired_differences, moving_block_bootstrap_ci

LADDER = (1, 2, 4, 8, 16, 32)
EXACT_MAX_BLOCKS = 16
N_RESAMPLES = 100_000
BOOTSTRAP_RESAMPLES = 10_000
ALPHA = 0.05
DEFAULT_OUTPUT = "results/peer_review_block_sensitivity_20261113"


def block_sums(scores, block_length, offset=0):
    """Sums over contiguous blocks; a nonzero offset makes the first block that long."""
    starts = list(range(offset, len(scores), block_length))
    if offset:
        starts.insert(0, 0)
    return np.add.reduceat(np.asarray(scores, dtype=float), starts)


def sign_flip_p(sums, *, rng, n_resamples=N_RESAMPLES, exact_max_blocks=EXACT_MAX_BLOCKS):
    """Two-sided p for |sum| under independent block sign flips, exact or Monte Carlo."""
    k = len(sums)
    observed = abs(sums.sum()) - 1e-9
    if k <= exact_max_blocks:
        signs = 1 - 2 * ((np.arange(2 ** k)[:, None] >> np.arange(k)) & 1)
        return float(np.mean(np.abs(signs @ sums) >= observed)), "exact"
    extreme = 0
    for start in range(0, n_resamples, 10_000):
        signs = rng.choice((-1.0, 1.0), size=(min(10_000, n_resamples - start), k))
        extreme += int(np.sum(np.abs(signs @ sums) >= observed))
    return (extreme + 1) / (n_resamples + 1), "monte_carlo"


def block_lengths(n, origins):
    spacing = [b - a for a, b in zip(origins, origins[1:])]
    lengths = {f"ladder_{b}": b for b in LADDER}
    lengths.update(cube_root=math.ceil(n ** (1.0 / 3.0)),
                   overlap_median=math.ceil(CONTEXT_LENGTH / statistics.median(spacing)),
                   overlap_max=math.ceil(CONTEXT_LENGTH / min(spacing)))
    return lengths


def paired_origins(result_dir, dataset, adapter, horizon):
    """The sorted common origins behind load_paired_differences, in the same order."""
    def origins(method):
        found = set()
        for path in glob.glob(str(Path(result_dir) / "per_origin" /
                                  f"{dataset}_{method}_H{horizon}_seed*.csv")):
            with open(path, newline="", encoding="utf-8") as handle:
                found.update(int(row["origin_idx"]) for row in csv.DictReader(handle))
        return found
    return sorted(origins(adapter) & origins("identity"))


def analyze(*, n_resamples=N_RESAMPLES, bootstrap_resamples=BOOTSTRAP_RESAMPLES):
    by_offset, summary, diagnostics = [], [], []
    for index, ((dataset, horizon), adapter) in enumerate(SELECTIONS.items()):
        case = {"dataset": dataset, "H": horizon, "adapter": adapter}
        if adapter == "identity":
            continue
        result_dir = ALL_ORIGIN_DIRS.get((dataset, adapter, horizon), PRIMARY_DIR)
        diffs = load_paired_differences(result_dir, dataset, adapter, horizon)
        origins = paired_origins(result_dir, dataset, adapter, horizon)
        n = len(diffs)
        assert len(origins) == n
        centred = np.asarray(diffs) - np.mean(diffs)
        spacing = [b - a for a, b in zip(origins, origins[1:])]
        diagnostics.append(case | {
            "n": n, "spacing_min": min(spacing), "spacing_median": statistics.median(spacing),
            "spacing_max": max(spacing), "acf_band": 2 / math.sqrt(n),
            **{f"acf_lag{k}": float(centred[:-k] @ centred[k:] / (centred @ centred))
               for k in (1, 2, 3)}})
        scores = {"wilcoxon": _signed_ranks(diffs), "sign": [(d > 0) - (d < 0) for d in diffs]}
        lengths = block_lengths(n, origins)
        for j, (label, b) in enumerate(lengths.items()):
            k0 = math.ceil(n / b)
            row = case | {"n": n, "length_label": label, "block_length": b, "n_blocks": k0,
                          "min_attainable_p": min(1.0, 2 / 2 ** k0),
                          "informative": 2 / 2 ** k0 <= ALPHA / len(SELECTIONS),
                          "within_overlap_max": b <= lengths["overlap_max"]}
            for stat_id, (stat, values) in enumerate(scores.items()):
                ps = []
                for offset in range(min(b, n)):
                    sums = block_sums(values, b, offset)
                    rng = np.random.default_rng([2026, index, b, offset, stat_id])
                    p, method = sign_flip_p(sums, rng=rng, n_resamples=n_resamples)
                    by_offset.append(case | {"statistic": stat, "length_label": label,
                                             "block_length": b, "offset": offset,
                                             "n_blocks": len(sums), "method": method, "p": p})
                    ps.append(p)
                row |= {f"{stat}_p_offset0": ps[0], f"{stat}_p_min": min(ps),
                        f"{stat}_p_worst": max(ps)}
            row["mbb_ci_lower"], row["mbb_ci_upper"] = moving_block_bootstrap_ci(
                diffs, block_length=min(b, n), n_resamples=bootstrap_resamples,
                seed=4000 + 100 * index + j)
            summary.append(row)

    # Holm over all 16 cases per block-length label; identity cases enter as p = 1.
    for label in {r["length_label"] for r in summary}:
        rows = [r for r in summary if r["length_label"] == label]
        for column, source in (("holm_wilcoxon_offset0", "wilcoxon_p_offset0"),
                               ("holm_wilcoxon_worst", "wilcoxon_p_worst"),
                               ("holm_sign_worst", "sign_p_worst")):
            padding = [1.0] * (len(SELECTIONS) - len(rows))
            for r, adjusted in zip(rows, holm_adjust([r[source] for r in rows] + padding)):
                r[column] = adjusted
    return by_offset, summary, diagnostics


def decision(summary):
    """The pre-registered rule from BLOCK_SENSITIVITY.md, section 1."""
    usable = [r for r in summary if r["informative"] and r["within_overlap_max"]]
    target = [r for r in usable if (r["dataset"], r["H"]) == ("solar-al", 48)]
    others = [r for r in usable if (r["dataset"], r["H"]) != ("solar-al", 48)]
    return {
        "a_solar_al_h48_worst_offset_holm_below_alpha": all(
            r["holm_wilcoxon_worst"] < ALPHA for r in target),
        "b_no_other_case_offset0_holm_below_alpha": not any(
            r["holm_wilcoxon_offset0"] < ALPHA for r in others),
        "usable_lengths_solar_al_h48": sorted(r["block_length"] for r in target),
    }


def _write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    by_offset, summary, diagnostics = analyze()
    _write_csv(out / "by_offset.csv", by_offset)
    _write_csv(out / "summary.csv", summary)
    _write_csv(out / "diagnostics.csv", diagnostics)
    inputs = sorted({p for (d, h), a in SELECTIONS.items() for m in {a, "identity"}
                     for p in glob.glob(str(Path(ALL_ORIGIN_DIRS.get((d, a, h), PRIMARY_DIR))
                                            / "per_origin" / f"{d}_{m}_H{h}_seed*.csv"))})
    verdict = decision(summary)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "CMF-paper/submission-sinergi/revisions/peer-review-r1/BLOCK_SENSITIVITY.md",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs},
        "ladder": LADDER, "exact_max_blocks": EXACT_MAX_BLOCKS,
        "n_resamples": N_RESAMPLES, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "decision": verdict,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
