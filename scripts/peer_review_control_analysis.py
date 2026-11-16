"""Paired analysis of one peer-review control run (losses.csv + manifest.json).

Origin-level loss is the metric averaged over sampler repetitions; same-seed
replays are excluded. For each comparison a/b, d = loss_a - loss_b on the run's
origins (negative favours a): mean, percent change of the mean, median, win
rate, the manuscript's Wilcoxon signed-rank p, the worst-offset block sign-flip
signed-rank p at b = ceil(L / min spacing), and a moving-block 95% interval for
the mean at the same b. Holm runs over the family comparisons only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.peer_review_block_sensitivity import block_sums, sign_flip_p
from scripts.sinergi_block_robust import _signed_ranks
from scripts.sinergi_stats import holm_adjust, moving_block_bootstrap_ci, wilcoxon_signed_rank_p


def origin_losses(run_dir, *, metric="MASE", mode="target_units"):
    """{horizon: {adapter: {origin: metric averaged over repetitions}}}."""
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with (Path(run_dir) / "losses.csv").open(newline="") as handle:
        for r in csv.DictReader(handle):
            if r["replay"] == "0" and r["mode"] == mode:
                values[int(r["horizon"])][r["adapter"]][int(r["origin_idx"])].append(
                    float(r[metric]))
    return {h: {a: {o: float(np.mean(v)) for o, v in by_origin.items()}
                for a, by_origin in by_adapter.items()}
            for h, by_adapter in values.items()}


def _compare(losses, a, b, *, origins, context_length, seed):
    if not set(losses[a]) == set(losses[b]) == set(origins):
        raise ValueError(f"{a} and {b} do not cover the run's origins")
    order = sorted(origins)
    d = [losses[a][o] - losses[b][o] for o in order]
    block = min(math.ceil(context_length / min(y - x for x, y in zip(order, order[1:]))), len(d))
    ranks = _signed_ranks(d)
    worst = max(sign_flip_p(block_sums(ranks, block, s), rng=np.random.default_rng([seed, s]))[0]
                for s in range(block))
    low, high = moving_block_bootstrap_ci(d, block_length=block, n_resamples=10_000, seed=seed)
    mean_a, mean_b = np.mean([losses[a][o] for o in order]), np.mean([losses[b][o] for o in order])
    return {"n": len(d), "mean_a": float(mean_a), "mean_b": float(mean_b),
            "mean_diff": float(np.mean(d)), "mean_pct": float(100 * (mean_a / mean_b - 1)),
            "median_diff": float(np.median(d)), "win_rate": float(np.mean(np.array(d) < 0)),
            "wilcoxon_p": wilcoxon_signed_rank_p(d), "block_length": block,
            "block_p_worst": worst, "ci_lower": low, "ci_upper": high}


def analyze(run_dir, family, descriptive=(), *, metric="MASE", mode="target_units"):
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("The run is not complete")
    losses = origin_losses(run_dir, metric=metric, mode=mode)
    rows = []
    for role, pairs in (("family", family), ("descriptive", descriptive)):
        for h in sorted(losses):
            for a, b in pairs:
                rows.append({"role": role, "metric": metric, "H": h, "a": a, "b": b,
                             **_compare(losses[h], a, b, origins=manifest["origins"][str(h)],
                                        context_length=manifest["context_length"],
                                        seed=5000 + len(rows))})
    family_rows = [r for r in rows if r["role"] == "family"]
    for r, adjusted in zip(family_rows, holm_adjust([r["wilcoxon_p"] for r in family_rows])):
        r["holm_p"] = adjusted
    for r in rows:
        r.setdefault("holm_p", None)
    return rows


def main(argv=None):
    pair = lambda text: tuple(text.split("/"))  # noqa: E731 - adapter specs may contain ':'
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--family", nargs="+", type=pair, required=True, help="a/b pairs")
    parser.add_argument("--descriptive", nargs="*", type=pair, default=[])
    parser.add_argument("--metrics", nargs="+", default=["MASE", "MASE_mean"])
    parser.add_argument("--output", required=True, help="new directory")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = [r for metric in args.metrics
            for r in analyze(args.run_dir, args.family, args.descriptive, metric=metric)]
    with (out / "comparisons.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    run = Path(args.run_dir)
    (out / "analysis.json").write_text(json.dumps({
        "run_dir": str(run), "argv": argv, "script_sha256": digest(__file__),
        "losses_sha256": digest(run / "losses.csv"),
        "manifest_sha256": digest(run / "manifest.json"),
        "noise_summary": json.loads((run / "identity_noise_summary.json").read_text()),
    }, indent=2) + "\n")
    for r in rows:
        holm = "" if r["holm_p"] is None else f" holm={r['holm_p']:.3g}"
        print(f"{r['metric']:9} H={r['H']} {r['a']} vs {r['b']}: n={r['n']} "
              f"mean%={r['mean_pct']:+.2f} win={r['win_rate']:.3f} p={r['wilcoxon_p']:.3g}"
              f"{holm} block_p={r['block_p_worst']:.3g} CI=[{r['ci_lower']:.4f}, "
              f"{r['ci_upper']:.4f}]")


if __name__ == "__main__":
    main()
