"""Exact paired mutation-level analysis for the full nested selector.

This audit consumes the frozen outer-fold trace produced by
``p2_k3_nested_pca.py``.  It deliberately compares one mutation-level error
per held-out mutation with the corresponding leave-one-out training-mean
error.  Seeds are averaged inside the fold and are retained only as an
optimization-variability summary.  The script does not refit a model and
does not treat seeds as biological observations.

Outputs
-------
``results/p2_k3_paired_exact.json``
    Machine-readable paired statistics and per-mutation predictions.
``results/p2_k3_per_mutation_errors.csv``
    Reviewer-facing long table with population and shift vectors.
``results/p2_k3_paired_exact_report.md``
    Compact text report with exact sign/permutation results.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE = RESULTS / "p2_k3_nested_pca_results.json"
OUT_JSON = RESULTS / "p2_k3_paired_exact.json"
OUT_CSV = RESULTS / "p2_k3_per_mutation_errors.csv"
OUT_MD = RESULTS / "p2_k3_paired_exact_report.md"

# Frozen wild-type populations from ``k3_data.py``.  They are repeated here
# so this audit remains runnable in the lightweight analysis environment,
# which does not need the training stack imported by that module.
WT = {
    "abl1": np.asarray([0.88, 0.06, 0.06], dtype=float),
    "src": np.asarray([0.72, 0.07, 0.21], dtype=float),
}

# These are the prespecified collision groups used by the support analysis.
# The flag is a diagnostic membership label, not a model-failure label.
CONFLICT_GROUPS = {
    "abl1": {
        "F382_exact": {"F382L", "F382Y", "F382V"},
    },
    "src": {
        "L410A_F405A": {"SrcKD-L410A", "SrcKD-F405A"},
        "L325A_V331A": {"SrcKD-L325A", "SrcKD-V331A"},
    },
}

SITE_GROUPS = {
    "abl1": {
        "M290_L301": {"M290L", "L301I", "M290L_L301I"},
        "F382": {"F382L", "F382Y", "F382V"},
    },
    "src": {
        "N_lobe": {
            "SrcKD-L270F_V332I", "SrcKD-L325A", "SrcKD-A311I",
            "SrcKD-V331A", "SrcKD-V332I",
        },
        "C_lobe": {"SrcKD-V380A", "SrcKD-F405A", "SrcKD-L410A"},
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt_vec(values: list[float]) -> str:
    return "(" + ", ".join(f"{float(v):.4f}" for v in values) + ")"


def exact_sign_test(deltas: np.ndarray) -> dict:
    nonzero = deltas[np.abs(deltas) > 1e-12]
    n = int(nonzero.size)
    if n == 0:
        return {
            "n_nonzero": 0,
            "wins_model": 0,
            "wins_mean": 0,
            "p_two_sided": 1.0,
        }
    wins_model = int(np.sum(nonzero < 0.0))
    wins_mean = int(np.sum(nonzero > 0.0))
    # Two-sided exact binomial sign test; the smaller tail is doubled.
    lower = sum(math.comb(n, k) for k in range(0, wins_model + 1)) / (2 ** n)
    upper = sum(math.comb(n, k) for k in range(wins_model, n + 1)) / (2 ** n)
    p = min(1.0, 2.0 * min(lower, upper))
    return {
        "n_nonzero": n,
        "wins_model": wins_model,
        "wins_mean": wins_mean,
        "p_two_sided": float(p),
    }


def exact_paired_permutation(deltas: np.ndarray) -> dict:
    """Enumerate all sign flips of mutation-level paired differences."""
    nonzero = deltas[np.abs(deltas) > 1e-12]
    n = int(nonzero.size)
    if n == 0:
        return {"n_nonzero": 0, "n_permutations": 1, "p_two_sided": 1.0}
    observed = float(np.mean(nonzero))
    null_means = np.asarray(
        [np.mean(nonzero * np.asarray(signs, dtype=float))
         for signs in itertools.product((-1.0, 1.0), repeat=n)],
        dtype=float,
    )
    p = float(np.mean(np.abs(null_means) >= abs(observed) - 1e-12))
    return {
        "n_nonzero": n,
        "n_permutations": int(2 ** n),
        "observed_mean_delta": observed,
        "p_two_sided": p,
    }


def paired_bootstrap_ci(deltas: np.ndarray, n_boot: int = 10000, seed: int = 20260810) -> list[float]:
    if deltas.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, deltas.size, size=(n_boot, deltas.size))
    means = deltas[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def site_group(system: str, mutant: str) -> str:
    for group, members in SITE_GROUPS[system].items():
        if mutant in members:
            return group
    return "unassigned"


def conflict_group(system: str, mutant: str) -> str | None:
    for group, members in CONFLICT_GROUPS[system].items():
        if mutant in members:
            return group
    return None


def collect_system(source: dict, system: str) -> tuple[dict, list[dict]]:
    system_obj = source["systems"][system]
    # The frozen artifact stores the outer-fold trace as one mapping keyed by
    # held-out mutation (under ``folds.nested_mlp``), rather than as a list.
    folds = system_obj["folds"]["nested_mlp"]
    # Every outer fold contains a held-out target; use those values rather
    # than re-importing data so that the table is traceable to the exact run.
    targets: dict[str, np.ndarray] = {}
    for mutant, detail in folds.items():
        targets[mutant] = np.asarray(detail["target"], dtype=float)
    names = list(targets)
    if not names:
        raise ValueError(f"No nested outer-fold targets found for {system}")

    rows: list[dict] = []
    for outer_fold, mutant in enumerate(names):
        detail = folds[mutant]
        pred = np.asarray(detail["pred"], dtype=float)
        target = np.asarray(detail["target"], dtype=float)
        seed_preds = [np.asarray(v, dtype=float) for v in detail["per_seed_predictions"].values()]
        train_targets = np.asarray([targets[m] for m in names if m != mutant], dtype=float)
        mean_pred = train_targets.mean(axis=0)
        model_error = float(np.mean(np.abs(pred - target)))
        baseline_error = float(np.mean(np.abs(mean_pred - target)))
        seed_errors = [float(np.mean(np.abs(seed_pred - target))) for seed_pred in seed_preds]
        true_shift = target - WT[system]
        pred_shift = pred - WT[system]
        cgroup = conflict_group(system, mutant)
        rows.append({
            "system": system,
            "mutation": mutant,
            "site_group": site_group(system, mutant),
            "conflict": cgroup is not None,
            "conflict_group": cgroup,
            "true_population": target.tolist(),
            "predicted_population": pred.tolist(),
            "training_mean_population": mean_pred.tolist(),
            "true_shift": true_shift.tolist(),
            "predicted_shift": pred_shift.tolist(),
            "mae_nested_mlp": model_error,
            "mae_training_mean": baseline_error,
            "delta_model_minus_mean": model_error - baseline_error,
            "seed_mae": seed_errors,
            "seed_mae_sd": float(np.std(seed_errors, ddof=1)) if len(seed_errors) > 1 else 0.0,
            "selected_candidate": detail["candidate"],
            "selected_model": detail["model"],
            "outer_fold": outer_fold,
            "pca_hash": detail.get("meta", {}).get("pca_hash"),
        })

    model_errors = np.asarray([row["mae_nested_mlp"] for row in rows], dtype=float)
    baseline_errors = np.asarray([row["mae_training_mean"] for row in rows], dtype=float)
    deltas = model_errors - baseline_errors
    paired = {
        "n_mutations": len(rows),
        "model_mae": float(model_errors.mean()),
        "training_mean_mae": float(baseline_errors.mean()),
        "mean_delta_model_minus_mean": float(deltas.mean()),
        "median_delta_model_minus_mean": float(np.median(deltas)),
        "model_better": int(np.sum(deltas < -1e-12)),
        "model_worse": int(np.sum(deltas > 1e-12)),
        "ties": int(np.sum(np.abs(deltas) <= 1e-12)),
        "relative_improvement_percent": float(100.0 * (baseline_errors.mean() - model_errors.mean()) / baseline_errors.mean()),
        "mean_seed_mae_sd": float(np.mean([row["seed_mae_sd"] for row in rows])),
        "exact_sign_test": exact_sign_test(deltas),
        "exact_paired_permutation": exact_paired_permutation(deltas),
        "paired_bootstrap_95ci_mean_delta": paired_bootstrap_ci(deltas),
    }
    # Cross-check against the frozen summary, which is rounded only in the
    # per-mutation display but retains a high-precision aggregate MAE.
    if abs(paired["model_mae"] - float(system_obj["nested_mlp"]["mae"])) > 2e-6:
        raise AssertionError(f"Nested MAE mismatch for {system}: {paired['model_mae']} vs {system_obj['nested_mlp']['mae']}")
    if abs(paired["training_mean_mae"] - float(system_obj["training_mean"]["mae"])) > 2e-6:
        raise AssertionError(f"Training-mean MAE mismatch for {system}: {paired['training_mean_mae']} vs {system_obj['training_mean']['mae']}")
    return paired, rows


def write_csv(rows: list[dict]) -> None:
    fields = [
        "system", "mutation", "site_group", "conflict", "conflict_group",
        "true_population", "predicted_population", "training_mean_population",
        "true_shift", "predicted_shift", "mae_nested_mlp", "mae_training_mean",
        "delta_model_minus_mean", "seed_mae_sd", "selected_candidate",
        "selected_model", "outer_fold", "pca_hash",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("true_population", "predicted_population", "training_mean_population", "true_shift", "predicted_shift"):
                out[key] = fmt_vec(out[key])
            writer.writerow({key: out.get(key) for key in fields})


def write_report(result: dict) -> None:
    lines = [
        "# Full-nested selector: exact paired mutation-level audit",
        "",
        "> Comparison: outer-fold full nested MLP selector vs leave-one-out training mean.",
        "> The unit is the held-out mutation; five neural seeds are averaged within each fold and are not extra observations.",
        "",
    ]
    for system in ("abl1", "src"):
        p = result["systems"][system]["paired"]
        sign = p["exact_sign_test"]
        perm = p["exact_paired_permutation"]
        lines.extend([
            f"## {system.upper()} (n={p['n_mutations']})",
            "",
            f"- MAE: nested MLP {p['model_mae']:.4f} vs training mean {p['training_mean_mae']:.4f}.",
            f"- Mean paired difference (model minus mean): **{p['mean_delta_model_minus_mean']:.4f}**; median {p['median_delta_model_minus_mean']:.4f}.",
            f"- Per-mutation outcomes: model better {p['model_better']}/{p['n_mutations']}, worse {p['model_worse']}/{p['n_mutations']}, ties {p['ties']}.",
            f"- Exact two-sided sign test: p={sign['p_two_sided']:.4f} (nonzero pairs n={sign['n_nonzero']}).",
            f"- Exact paired sign-flip permutation: p={perm['p_two_sided']:.4f} ({perm['n_permutations']} permutations).",
            f"- Mutation bootstrap 95% CI for mean difference: [{p['paired_bootstrap_95ci_mean_delta'][0]:.4f}, {p['paired_bootstrap_95ci_mean_delta'][1]:.4f}].",
            f"- Mean seed-to-seed MAE SD (optimization variability): {p['mean_seed_mae_sd']:.4f}.",
            "",
        ])
    lines.extend([
        "The exact tests are descriptive because each system contributes only six or eight mutations. They do not convert the nested estimate into an independent prospective validation claim.",
        "",
        "Source: `results/p2_k3_nested_pca_results.json`; generated by `experiments/iclr_restructuring/p2_k3_paired_exact.py`.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    systems = {}
    all_rows: list[dict] = []
    for system in ("abl1", "src"):
        paired, rows = collect_system(source, system)
        systems[system] = {
            "paired": paired,
            "per_mutation": rows,
        }
        all_rows.extend(rows)

    result = {
        "schema": "p2_k3_paired_exact_v1",
        "protocol": {
            "unit": "held-out mutation",
            "comparison": "full nested MLP selector vs LOO training mean",
            "seeds": 5,
            "seed_role": "optimization variability only",
            "nested_artifact": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
            "exact_sign_flip": "all 2^n sign flips of nonzero mutation-level deltas",
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260810,
            "conflict_definition": "prespecified support-analysis collision groups; diagnostic membership, not a failure label",
        },
        "source_hashes": {
            "nested_artifact_sha256": sha256_file(SOURCE),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "systems": systems,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(all_rows)
    write_report(result)
    print(json.dumps({
        system: {
            "mae": result["systems"][system]["paired"]["model_mae"],
            "mean": result["systems"][system]["paired"]["training_mean_mae"],
            "delta": result["systems"][system]["paired"]["mean_delta_model_minus_mean"],
            "sign_p": result["systems"][system]["paired"]["exact_sign_test"]["p_two_sided"],
            "perm_p": result["systems"][system]["paired"]["exact_paired_permutation"]["p_two_sided"],
        }
        for system in ("abl1", "src")
    }, indent=2))


if __name__ == "__main__":
    main()
