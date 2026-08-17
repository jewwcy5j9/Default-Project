"""P5 (deep-review Q8): candidate-quality spread and selection optimism.

The P4 simulator gives near-zero selection optimism at n=20 because it includes
a perfect candidate (candidate 0 is the raw feature x) and its noisy
random-projection candidates are far worse, so inner selection is trivial. At
n=6 it gives 0.018, still ~7x smaller than the biological fixed-to-selected gap
(~0.12). The real five-candidate panel has a near-tie at the top (fold-local
LLR+position 0.1400 vs ESM-2 PCA 0.1477) with only ~0.007 separation, so the
inner-LOO selector on 4-6 observations cannot reliably rank them.

This script isolates that mechanism: all candidates are noisy copies of x with
a deterministic quality ladder rho_j = rho_max - j*gap_step, and the quality
gap (gap_step) plus sample size n are swept. It reuses the frozen P4 machinery
(CLR-ridge alpha=1, fold-local scaling, nested inner-LOO selection).

Outputs:
  results/p5_candidate_heterogeneity.json
  results/p5_candidate_heterogeneity_report.md
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

import p4_support_resolution_selection as P4

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT_JSON = RESULTS / "p5_candidate_heterogeneity.json"
OUT_MD = RESULTS / "p5_candidate_heterogeneity_report.md"

EPSILON = 0.0
DELTA = 1.2
RESOLUTION = "full_k3"
M = 5
REPEATS = 500
BASE_SEED = 20260814
RHO_MAX = 0.90
N_GRID = [6, 8, 20]
GAP_GRID = [0.15, 0.05, 0.02, 0.01]  # quality gap between consecutive candidates


def make_features(n, epsilon, delta, rng, m, gap_step):
    x = rng.uniform(-1.0, 1.0, size=n)
    x[0] = -0.25
    x[1] = -0.25 + epsilon
    targets = P4.latent_targets(x, delta)
    features = np.empty((n, m), dtype=float)
    for j in range(m):
        rho = RHO_MAX - j * gap_step
        rho = max(0.0, rho)
        noise = rng.normal(size=n)
        noise[1] = noise[0]  # keep the collision pair identical across candidates
        features[:, j] = rho * x + math.sqrt(max(1.0 - rho * rho, 0.0)) * noise
    return features, targets


def run_one(n, gap_step, repeat):
    seed = BASE_SEED + n * 1_000_000 + int(round(gap_step * 1000)) * 1000 + repeat
    rng = np.random.default_rng(seed)
    features, targets = make_features(n, EPSILON, DELTA, rng, M, gap_step)
    latent = P4.to_latent(targets, RESOLUTION)
    target = P4.target_for_resolution(targets, RESOLUTION)
    loo_all = P4.ridge_loo_candidates(features, latent, RESOLUTION)
    naive_errors = P4.candidate_mae(loo_all, target)
    nested_pred, selected, audit = P4.nested_predictions(
        features, targets, RESOLUTION, m_values=[M])
    nested_error = float(np.abs(nested_pred[M] - target).mean())
    naive_error = float(np.min(naive_errors[:M]))
    return {
        "n": n, "gap_step": gap_step, "repeat": repeat,
        "naive_error": naive_error,
        "nested_error": nested_error,
        "selection_optimism": nested_error - naive_error,
        "naive_selected_candidate": int(np.argmin(naive_errors[:M])),
    }


def main():
    records = []
    for n in N_GRID:
        for gap_step in GAP_GRID:
            for repeat in range(REPEATS):
                records.append(run_one(n, gap_step, repeat))
            print(f"[n={n}, gap={gap_step}] done")

    summary = []
    for n in N_GRID:
        for gap_step in GAP_GRID:
            rows = [r for r in records if r["n"] == n and r["gap_step"] == gap_step]
            opt = np.array([r["selection_optimism"] for r in rows])
            summary.append({
                "n": n, "gap_step": gap_step, "repeats": len(rows),
                "selection_optimism_mean": float(opt.mean()),
                "selection_optimism_se": float(opt.std(ddof=1) / np.sqrt(len(opt))),
                "nested_error_mean": float(np.mean([r["nested_error"] for r in rows])),
                "naive_error_mean": float(np.mean([r["naive_error"] for r in rows])),
            })

    payload = {
        "experiment": "P5 candidate-quality spread and selection optimism",
        "source_machinery": "p4_support_resolution_selection (frozen CLR-ridge, fold-local, nested LOO)",
        "settings": {"epsilon": EPSILON, "delta": DELTA, "m": M,
                     "resolution": RESOLUTION, "repeats": REPEATS,
                     "n_grid": N_GRID, "gap_grid": GAP_GRID,
                     "rho_max": RHO_MAX, "base_seed": BASE_SEED},
        "method": ("candidate j has rho_j = rho_max - j*gap_step; all candidates share the "
                   "same collision-pair identity; quality gap controls the near-tie at the top"),
        "summary": summary,
        "records": records,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = ["# P5 candidate-quality spread and selection optimism",
             "",
             "Status: **DESCRIPTIVE AUDIT** (2026-08-13). Frozen P4 machinery; only the "
             "candidate-quality ladder differs.",
             "",
             "| n | gap (rho step) | selection optimism (nested - naive) | nested | naive |",
             "|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['n']} | {row['gap_step']} | {row['selection_optimism_mean']:.5f} "
                     f"± {row['selection_optimism_se']:.5f} | {row['nested_error_mean']:.4f} "
                     f"| {row['naive_error_mean']:.4f} |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
