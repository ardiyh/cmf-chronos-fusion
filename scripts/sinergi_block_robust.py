"""Dependence-robust sensitivity for the paired rank and sign tests.

The signed-rank and sign p-values in sinergi_stats.py treat origins as
independent. Consecutive origins share context (L=512 exceeds the origin
spacing), so this module re-tests the same statistics with a block sign-flip
randomization: the ordered differences are cut into contiguous blocks and each
block's signs are flipped together. Under the null of a distribution symmetric
about zero, this stays valid when dependence is confined within blocks.

Two block lengths are reported per cell: ceil(n^(1/3)), the rule already used
for the bootstrap interval, and ceil(L / median origin spacing), the number of
consecutive origins whose context windows overlap. The moving-block bootstrap
interval is also recomputed at the longer length.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import random
import statistics
from pathlib import Path

from scripts.sinergi_stats import (
    holm_adjust,
    load_paired_differences,
    moving_block_bootstrap_ci,
)
from scripts.sinergi_zero_future_sensitivity import CELLS as ALL_ORIGIN_DIRS

PRIMARY_DIR = "results/rtx_20260803_test"
CONTEXT_LENGTH = 512

SELECTIONS = {
    ("ecl", 24): "ridge", ("ecl", 48): "identity",
    ("etth1", 24): "mlp", ("etth1", 48): "mlp",
    ("etth2", 24): "krr", ("etth2", 48): "krr",
    ("ettm1", 24): "mlp", ("ettm1", 48): "identity",
    ("ettm2", 24): "mlp", ("ettm2", 48): "krr",
    ("opsd-wind", 24): "mlp", ("opsd-wind", 48): "krr",
    ("sdwpf", 24): "ridge", ("sdwpf", 48): "krr",
    ("solar-al", 24): "krr", ("solar-al", 48): "krr",
}


def _signed_ranks(differences):
    """Average ranks of |d| over nonzero differences, signed; zeros discarded."""
    values = [d for d in differences if d != 0.0]
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and abs(values[order[end]]) == abs(values[order[position]]):
            end += 1
        for k in range(position, end):
            ranks[order[k]] = (position + 1 + end) / 2.0
        position = end
    signed = []
    it = iter(zip(values, ranks))
    for d in differences:
        if d == 0.0:
            signed.append(0.0)
        else:
            value, rank = next(it)
            signed.append(math.copysign(rank, value))
    return signed


def block_sign_flip_p(scores, *, block_length, n_resamples=10_000, seed=42):
    """Two-sided p-value for sum(scores) under block-wise sign flips."""
    n = len(scores)
    blocks = [sum(scores[i:i + block_length]) for i in range(0, n, block_length)]
    observed = abs(sum(blocks))
    generator = random.Random(seed)
    extreme = 0
    for _ in range(n_resamples):
        total = sum(b if generator.random() < 0.5 else -b for b in blocks)
        if abs(total) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (n_resamples + 1)


def origin_spacing(result_dir, dataset, adapter, horizon):
    path = sorted(glob.glob(str(Path(result_dir) / "per_origin" /
                                f"{dataset}_{adapter}_H{horizon}_seed*.csv")))[0]
    with open(path, newline="", encoding="utf-8") as handle:
        origins = sorted(int(row["origin_idx"]) for row in csv.DictReader(handle))
    return statistics.median(b - a for a, b in zip(origins, origins[1:]))


def analyze(*, n_resamples=10_000):
    rows = []
    for index, ((dataset, horizon), adapter) in enumerate(SELECTIONS.items()):
        if adapter == "identity":
            rows.append({"dataset": dataset, "H": horizon, "adapter": adapter, "n": 0})
            continue
        result_dir = ALL_ORIGIN_DIRS.get((dataset, adapter, horizon), PRIMARY_DIR)
        diffs = load_paired_differences(result_dir, dataset, adapter, horizon)
        n = len(diffs)
        spacing = origin_spacing(result_dir, dataset, adapter, horizon)
        b_short = math.ceil(n ** (1.0 / 3.0))
        b_overlap = math.ceil(CONTEXT_LENGTH / spacing)
        ranks = _signed_ranks(diffs)
        signs = [(d > 0) - (d < 0) for d in diffs]
        seed = 3000 + 10 * index
        low, high = moving_block_bootstrap_ci(
            diffs, block_length=b_overlap, n_resamples=n_resamples, seed=seed)
        rows.append({
            "dataset": dataset, "H": horizon, "adapter": adapter, "n": n,
            "median_spacing": spacing, "b_short": b_short, "b_overlap": b_overlap,
            "wilcoxon_block_p_short": block_sign_flip_p(
                ranks, block_length=b_short, n_resamples=n_resamples, seed=seed + 1),
            "wilcoxon_block_p_overlap": block_sign_flip_p(
                ranks, block_length=b_overlap, n_resamples=n_resamples, seed=seed + 2),
            "sign_block_p_overlap": block_sign_flip_p(
                signs, block_length=b_overlap, n_resamples=n_resamples, seed=seed + 3),
            "block_ci_overlap_lower": low, "block_ci_overlap_upper": high,
        })
    for key in ("wilcoxon_block_p_short", "wilcoxon_block_p_overlap"):
        adjusted = holm_adjust([r.get(key, 1.0) for r in rows])
        for r, a in zip(rows, adjusted):
            if r["n"]:
                r[key.replace("wilcoxon_", "holm_")] = a
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="results/sinergi_block_robust.csv")
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    rows = analyze(n_resamples=args.resamples)
    fields = list(max(rows, key=len))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
