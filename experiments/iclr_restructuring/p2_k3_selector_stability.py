"""P1 selector-stability audit of the P0-1 nested selection (deep-review Q5).

The nested selector chooses among five representations by inner-LOO MAE on
n_train-1 = 4 (Abl1) or 6 (Src) observations. This script quantifies how
stable that choice is without re-running the MLP: it reuses the per-candidate
inner-LOO fold errors already stored in ``p2_k3_nested_pca_results.json``.

Three statistics, per outer fold and aggregated:
  A. Selection margin: inner_mae(rank-2) - inner_mae(rank-1); a fold is a
     "near-tie" when the margin is below the frozen 0.05 tie threshold.
  B. Selection probability: 10,000 within-fold resamples of the inner-LOO
     fold errors (nonparametric bootstrap); the candidate mean is the
     resampled mean and selection is argmin with the frozen tie-break. This
     measures how often the observed selection would change if the same
     inner-LOO folds were resampled.
  C. Regret relative to a per-fold oracle: outer MAE(selected) minus the
     minimum outer MAE over candidates on that fold, using the fixed_mlp_raw
     per-candidate outer predictions stored in the same artifact (available
     for the primary nested_mlp variant only). This oracle inspects the
     fold's held-out target, so it is not a realizable fixed candidate.

Outputs:
  results/p2_k3_selector_stability.json
  results/p2_k3_selector_stability_report.md
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE = RESULTS / "p2_k3_nested_pca_results.json"
OUT_JSON = RESULTS / "p2_k3_selector_stability.json"
OUT_MD = RESULTS / "p2_k3_selector_stability_report.md"

TIE_DELTA = 0.05
N_BOOT = 10_000
RNG = np.random.default_rng(20260813)
CANDIDATE_ORDER = ["pos", "ext", "llr_pos", "llr_only", "pca20"]
# Frozen model order used by the nested model-select evaluation; only
# relevant for the nested_model_select variant, whose inner_scores carry
# full cand::model keys.
MODEL_ORDER = ["CLR-Ridge", "CLR-GP", "LowRankCDST"]


def short_candidate(key):
    """'pos::LowRankCDST' -> 'pos'."""
    return key.split("::", 1)[0]


def combo_rank(key, full_keys):
    """Tie-break rank of an inner-score key under the frozen order."""
    cand, _, model = key.partition("::")
    cand_rank = CANDIDATE_ORDER.index(cand) if cand in CANDIDATE_ORDER else 999
    if not full_keys:
        return cand_rank
    model_rank = MODEL_ORDER.index(model) if model in MODEL_ORDER else 999
    return cand_rank * 10 + model_rank


def outer_mae(pred, target):
    return float(np.mean(np.abs(np.asarray(pred, dtype=float) - np.asarray(target, dtype=float))))


def analyze_variant(folds, fixed_raw, n_outer, include_regret, rng=None):
    """Return per-fold stats + summary for one variant.

    ``inner_scores`` keys are full ``cand::model`` strings. When the variant
    searched multiple models per candidate (nested_model_select), statistics
    are keyed on the FULL cand::model keys — collapsing to the candidate name
    would silently keep only one model's row per candidate. Single-model
    variants (nested_mlp, all keys '::LowRankCDST') use the short candidate
    name for backward-compatible output. Pass a dedicated ``rng`` so one
    variant's bootstrap consumption cannot perturb another variant's frozen
    statistics.
    """
    if rng is None:
        rng = RNG
    per_fold = []
    n_folds = 0
    for holdout, rec in folds.items():
        inner_scores = rec.get("inner_scores") or {}
        if not inner_scores:
            continue
        n_folds += 1
        models_seen = {k.split("::", 1)[1] for k in inner_scores if "::" in k}
        full_keys = len(models_seen) > 1
        key_of = (lambda k: k) if full_keys else short_candidate
        selected = rec.get("candidate")
        if full_keys:
            model = rec.get("model")
            if model:
                selected = f"{selected}::{model}"
        # --- A: selection margin ---
        means = {key_of(k): float(np.mean(v)) for k, v in inner_scores.items()}
        order = sorted(means, key=lambda c: (means[c], combo_rank(c, full_keys)))
        rank1, rank2 = order[0], order[1]
        margin = means[rank2] - means[rank1]
        near_tie = margin < TIE_DELTA

        # --- B: bootstrap selection probability ---
        # Resample the inner-fold error vector of every candidate with the SAME
        # resampled inner-fold indices, so the comparison is paired across
        # candidates (the honest within-fold design).
        arrays = {key_of(k): np.asarray(v, dtype=float)
                  for k, v in inner_scores.items()}
        n_inner = len(next(iter(arrays.values())))
        counts = {c: 0 for c in arrays}
        for _ in range(N_BOOT):
            idx = rng.integers(0, n_inner, size=n_inner)
            boot_means = {c: float(arr[idx].mean()) for c, arr in arrays.items()}
            boot_order = sorted(boot_means, key=lambda c: (
                boot_means[c], combo_rank(c, full_keys)))
            counts[boot_order[0]] += 1
        boot_prob = {c: counts[c] / N_BOOT for c in counts}
        p_selected_top = boot_prob.get(selected, 0.0)

        row = {
            "holdout": holdout,
            "selected": selected,
            "selected_inner_mae": means[selected],
            "runner_up": rank2,
            "margin_to_runner_up": round(margin, 6),
            "near_tie": near_tie,
            "bootstrap_selection_probability": {c: round(p, 4) for c, p in boot_prob.items()},
            "bootstrap_p_selected_top": round(p_selected_top, 4),
        }

        # --- C: regret relative to oracle (primary nested_mlp only) ---
        if include_regret and fixed_raw:
            cand_outer = {}
            for cand, by_holdout in fixed_raw.items():
                pred = by_holdout.get(holdout, {}).get("pred")
                target = by_holdout.get(holdout, {}).get("target")
                if pred is not None and target is not None:
                    cand_outer[cand] = outer_mae(pred, target)
            if cand_outer:
                oracle = min(cand_outer, key=lambda c: cand_outer[c])
                row["oracle_candidate"] = oracle
                row["selected_outer_mae"] = round(cand_outer.get(selected, float("nan")), 6)
                row["oracle_outer_mae"] = round(cand_outer[oracle], 6)
                row["regret"] = round(cand_outer.get(selected, float("nan")) - cand_outer[oracle], 6)
        per_fold.append(row)

    # --- aggregate summary ---
    margins = [r["margin_to_runner_up"] for r in per_fold]
    regrets = [r["regret"] for r in per_fold if "regret" in r]
    selection_counts = {}
    for r in per_fold:
        selection_counts[r["selected"]] = selection_counts.get(r["selected"], 0) + 1
    summary = {
        "n_folds": n_folds,
        "selection_counts": selection_counts,
        "near_tie_threshold": TIE_DELTA,
        "n_near_ties": sum(1 for r in per_fold if r["near_tie"]),
        "near_tie_fraction": round(sum(1 for r in per_fold if r["near_tie"]) / n_folds, 4),
        "margin_min": round(min(margins), 6),
        "margin_median": round(float(np.median(margins)), 6),
        "margin_max": round(max(margins), 6),
        "n_bootstrap": N_BOOT,
        "mean_bootstrap_p_selected_top": round(float(np.mean([r["bootstrap_p_selected_top"] for r in per_fold])), 4),
        "n_folds_unstable": sum(1 for r in per_fold if r["bootstrap_p_selected_top"] < 0.5),
    }
    if regrets:
        summary["regret_mean"] = round(float(np.mean(regrets)), 6)
        summary["regret_max"] = round(float(np.max(regrets)), 6)
    return {"folds": per_fold, "summary": summary}


def main():
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_sha256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()

    out = {
        "experiment": "P1 selector-stability audit of the P0-1 nested selection",
        "source": "experiments/iclr_restructuring/results/p2_k3_nested_pca_results.json",
        "source_sha256": source_sha256,
        "tie_threshold": TIE_DELTA,
        "n_bootstrap_resamples": N_BOOT,
        "method": {
            "selection_margin": "inner_mae(rank-2) - inner_mae(rank-1) per outer fold; near-tie when < 0.05",
            "bootstrap": ("10,000 within-fold resamples of the stored inner-LOO fold errors "
                          "(paired resample indices across candidates); selection = argmin "
                          "resampled mean with the frozen candidate-order tie-break. "
                          "nested_model_select statistics are keyed on the full "
                          "cand::model inner-score keys (15 combos), not the collapsed "
                          "candidate names, with the frozen candidate-then-model "
                          "tie-break order"),
            "regret": ("outer MAE(selected) - min over candidates of outer MAE on that fold "
                       "(a per-fold held-out-label oracle, not one realizable fixed candidate), "
                       "from the fixed_mlp_raw outer predictions (primary nested_mlp only)"),
            "note": ("conditional, descriptive analysis of the frozen artifact; no MLP retraining"),
        },
        "systems": {},
    }

    for sys_name in ("abl1", "src"):
        sys_block = payload["systems"][sys_name]
        folds = sys_block["folds"]
        fixed_raw = folds.get("fixed_mlp_raw", {})
        n_outer = sys_block["n_mutants"]
        out["systems"][sys_name] = {
            "n_outer_mutants": n_outer,
            "nested_mlp": analyze_variant(folds["nested_mlp"], fixed_raw, n_outer, include_regret=True),
            "nested_model_select": analyze_variant(
                folds["nested_model_select"], None, n_outer, include_regret=False,
                rng=np.random.default_rng(20260813)),
        }

    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    # ---------------- report ----------------
    lines = ["# P1 selector-stability audit of the P0-1 nested selection\n",
             "Status: **DESCRIPTIVE AUDIT** (2026-08-13). No MLP was retrained; all "
             "statistics are computed from the frozen `p2_k3_nested_pca_results.json` "
             "inner-LOO fold errors and fixed outer predictions.\n"]
    for sys_name in ("abl1", "src"):
        for variant in ("nested_mlp", "nested_model_select"):
            b = out["systems"][sys_name][variant]
            s = b["summary"]
            lines.append(f"## {sys_name} / {variant}\n")
            lines.append(f"- folds = {s['n_folds']}; selection counts = {s['selection_counts']}")
            lines.append(f"- near-ties (margin < {TIE_DELTA}): {s['n_near_ties']}/{s['n_folds']} "
                         f"({s['near_tie_fraction']:.2f}); margin range "
                         f"{s['margin_min']:.4f}–{s['margin_max']:.4f} (median {s['margin_median']:.4f})")
            lines.append(f"- bootstrap: mean P(selected is top-1) = "
                         f"{s['mean_bootstrap_p_selected_top']:.3f}; folds with P < 0.5 = "
                         f"{s['n_folds_unstable']}/{s['n_folds']}")
            if "regret_mean" in s:
                lines.append(f"- regret vs oracle: mean {s['regret_mean']:.4f}, "
                             f"max {s['regret_max']:.4f}")
            lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT_JSON), "report": str(OUT_MD)}, indent=2))
    for sys_name in ("abl1", "src"):
        b = out["systems"][sys_name]["nested_mlp"]["summary"]
        print(f"[{sys_name} nested_mlp] near-ties={b['n_near_ties']}/{b['n_folds']} "
              f"meanP(top)={b['mean_bootstrap_p_selected_top']:.3f} "
              f"unstable={b['n_folds_unstable']} "
              f"regret_mean={b.get('regret_mean')}")


if __name__ == "__main__":
    main()
