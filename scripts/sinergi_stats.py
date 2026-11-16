"""Dependence-aware statistical utilities for the SINERGI revision.

The module intentionally uses only the Python standard library so that the
stored per-origin CSVs remain auditable on machines without the training
environment.
"""

from __future__ import annotations

import csv
import glob
import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def write_analysis_csv(result_dir, destination, *, n_resamples=10_000):
    rows = analyze_retained_results(result_dir, n_resamples=n_resamples)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def analyze_retained_results(result_dir, *, n_resamples=10_000):
    """Recompute the manuscript's paired diagnostics from retained losses.

    The selection mapping is the validation-only mapping used by Table 1. In
    particular, SDWPF H=48 uses KRR (selected by validation nWQL), repairing the
    mismatch in the older ``significance_valid.csv`` that tested MLP there.
    Identity selections are included in the 16-hypothesis Holm family as p=1.
    """
    selections = {
        ("ecl", 24): "ridge", ("ecl", 48): "identity",
        ("etth1", 24): "mlp", ("etth1", 48): "mlp",
        ("etth2", 24): "krr", ("etth2", 48): "krr",
        ("ettm1", 24): "mlp", ("ettm1", 48): "identity",
        ("ettm2", 24): "mlp", ("ettm2", 48): "krr",
        ("opsd-wind", 24): "mlp", ("opsd-wind", 48): "krr",
        ("sdwpf", 24): "ridge", ("sdwpf", 48): "krr",
        ("solar-al", 24): "krr", ("solar-al", 48): "krr",
    }
    rows = []
    raw_p_values = []
    for cell_index, ((dataset, horizon), adapter) in enumerate(selections.items()):
        if adapter == "identity":
            row = {
                "dataset": dataset, "H": horizon, "adapter": adapter,
                "n": 0, "mean_diff": 0.0, "median_diff": 0.0,
                "win_rate": 0.0, "sign_p": 1.0, "wilcoxon_p": 1.0,
                "block_length": 0, "block_ci_lower": 0.0,
                "block_ci_upper": 0.0, "top_5_n": 0, "top_5_share": None,
            }
        else:
            differences = load_paired_differences(result_dir, dataset, adapter, horizon)
            n = len(differences)
            sign = exact_sign_test(differences)
            concentration = top_gain_share(differences, fraction=0.05)
            block_length = math.ceil(n ** (1.0 / 3.0))
            block_low, block_high = moving_block_bootstrap_ci(
                differences,
                block_length=block_length,
                n_resamples=n_resamples,
                seed=42 + cell_index,
            )
            row = {
                "dataset": dataset, "H": horizon, "adapter": adapter,
                "n": n, "mean_diff": sum(differences) / n,
                "median_diff": median(differences),
                "win_rate": sign["wins"] / n,
                "sign_p": sign["p_two_sided"],
                "wilcoxon_p": wilcoxon_signed_rank_p(differences),
                "block_length": block_length,
                "block_ci_lower": block_low, "block_ci_upper": block_high,
                "top_5_n": concentration["n_top"],
                "top_5_share": concentration["share"],
            }
        raw_p_values.append(row["wilcoxon_p"])
        rows.append(row)
    for row, adjusted in zip(rows, holm_adjust(raw_p_values)):
        row["holm_p"] = adjusted
    return rows


def load_paired_differences(result_dir, dataset, adapter, horizon):
    """Load seed-averaged selected-minus-identity MASE by common origin."""
    per_origin = Path(result_dir) / "per_origin"

    def load_method(method):
        pattern = per_origin / f"{dataset}_{method}_H{horizon}_seed*.csv"
        paths = sorted(glob.glob(str(pattern)))
        if not paths:
            raise FileNotFoundError(f"no per-origin files match {pattern}")
        values = defaultdict(list)
        for path in paths:
            with open(path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    values[int(row["origin_idx"])].append(float(row["MASE"]))
        return {origin: sum(items) / len(items) for origin, items in values.items()}

    selected = load_method(adapter)
    identity = load_method("identity")
    origins = sorted(set(selected).intersection(identity))
    return [selected[origin] - identity[origin] for origin in origins]


def wilcoxon_signed_rank_p(differences):
    """Two-sided large-sample Wilcoxon p-value matching SciPy's defaults.

    Zero differences use the ``wilcox`` convention (discarded). The retained
    manuscript cells all have more than 50 pairs, for which SciPy uses the
    tie-corrected normal approximation without a continuity correction.
    """
    values = [float(value) for value in differences if float(value) != 0.0]
    n = len(values)
    if n == 0:
        return 1.0
    absolute = [abs(value) for value in values]
    order = sorted(range(n), key=absolute.__getitem__)
    ranks = [0.0] * n
    position = 0
    while position < n:
        end = position + 1
        while end < n and absolute[order[end]] == absolute[order[position]]:
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        for sorted_position in range(position, end):
            ranks[order[sorted_position]] = average_rank
        position = end
    positive_rank_sum = sum(rank for rank, value in zip(ranks, values) if value > 0.0)
    expected = n * (n + 1) / 4.0
    tie_correction = sum(count**3 - count for count in Counter(absolute).values())
    variance = (n * (n + 1) * (2 * n + 1) - tie_correction / 2.0) / 24.0
    if variance <= 0.0:
        return 1.0
    z_score = (positive_rank_sum - expected) / math.sqrt(variance)
    return math.erfc(abs(z_score) / math.sqrt(2.0))


def holm_adjust(p_values):
    """Return Holm family-wise-error adjusted p-values in input order."""
    count = len(p_values)
    ordered = sorted(enumerate(float(value) for value in p_values), key=lambda x: x[1])
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, (original_index, value) in enumerate(ordered):
        running_max = max(running_max, (count - rank) * value)
        adjusted[original_index] = min(1.0, running_max)
    return adjusted


def exact_sign_test(differences):
    """Exact two-sided binomial sign test; negative differences are wins."""
    values = [float(value) for value in differences]
    wins = sum(value < 0.0 for value in values)
    losses = sum(value > 0.0 for value in values)
    ties = len(values) - wins - losses
    n = wins + losses
    if n == 0:
        p_value = 1.0
    else:
        tail = min(wins, losses)
        lower_probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
        p_value = min(1.0, 2.0 * lower_probability)
    return {"wins": wins, "losses": losses, "ties": ties, "p_two_sided": p_value}


def moving_block_bootstrap_ci(
    differences, *, block_length, n_resamples=10_000, seed=42, alpha=0.05
):
    """Circular moving-block percentile CI for the arithmetic mean."""
    values = [float(value) for value in differences]
    n = len(values)
    if n == 0:
        raise ValueError("differences must not be empty")
    if not 1 <= block_length <= n:
        raise ValueError("block_length must be between 1 and len(differences)")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")

    generator = random.Random(seed)
    n_blocks = math.ceil(n / block_length)
    means = []
    for _ in range(n_resamples):
        sample = []
        for _ in range(n_blocks):
            start = generator.randrange(n)
            sample.extend(values[(start + offset) % n] for offset in range(block_length))
        means.append(sum(sample[:n]) / n)
    means.sort()
    return (_percentile(means, 100.0 * alpha / 2.0),
            _percentile(means, 100.0 * (1.0 - alpha / 2.0)))


def top_gain_share(differences, *, fraction=0.05):
    """Share of total arithmetic gain supplied by the best floor(fraction*n) origins.

    Differences use selected-minus-identity orientation, so ``-difference`` is
    the gain from fusion. A share above one is valid when losses outside the top
    group offset some of its gain.
    """
    values = [float(value) for value in differences]
    if not values:
        raise ValueError("differences must not be empty")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    n_top = math.floor(fraction * len(values))
    if n_top == 0:
        raise ValueError("fraction selects zero observations")
    gains = sorted((-value for value in values), reverse=True)
    total_gain = sum(gains)
    if math.isclose(total_gain, 0.0, abs_tol=1e-15):
        share = None
    else:
        share = sum(gains[:n_top]) / total_gain
    return {"n_top": n_top, "total_gain": total_gain, "share": share}


def _percentile(sorted_values, percentile):
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def main():
    parser = argparse.ArgumentParser(
        description="Recompute SINERGI paired statistics from retained per-origin losses."
    )
    parser.add_argument("--result-dir", default="results/rtx_20260803_test")
    parser.add_argument(
        "--output", default="results/sinergi_stats_dependence_aware.csv"
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    rows = write_analysis_csv(
        args.result_dir, args.output, n_resamples=args.resamples
    )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
