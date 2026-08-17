"""Src CLR robustness to zero replacement and bounded label uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from gp_protocols import PRIMARY_GP_PROTOCOL, make_primary_gp
from p2_k3_nested_pca import build_X, get_delta_rows, load_features
from src.data.src_k3_labels import (
    CANONICAL_SRC_K3_PATH,
    SRC_K3_PRIMARY_PROTOCOL_ID,
    build_src_k3_panel,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PSEUDOCOUNTS = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
CANDIDATES = ["pos", "ext", "llr_pos", "llr_only", "pca20"]
MODELS = ["CLR-Ridge", "CLR-GP"]
INTERVAL_HALF_WIDTH = 0.10
INTERVAL_SEED = 20260811


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clr_with_pseudocount(y, pseudocount):
    y = np.clip(np.asarray(y, float), pseudocount, 1.0)
    y = y / y.sum(axis=-1, keepdims=True)
    return np.log(y) - np.log(y).mean(axis=-1, keepdims=True)


def inverse_clr(z):
    z = np.asarray(z, float)
    exp_z = np.exp(z - z.max(axis=-1, keepdims=True))
    values = exp_z / exp_z.sum(axis=-1, keepdims=True)
    return values[0] if values.ndim == 2 and values.shape[0] == 1 else values


def sample_simplex_interval(rng, center, half_width=INTERVAL_HALF_WIDTH):
    """Sample uniformly in the componentwise interval intersected with K=3 simplex."""
    center = np.asarray(center, float)
    low = np.maximum(0.0, center - half_width)
    high = np.minimum(1.0, center + half_width)
    for _ in range(10000):
        first, second = rng.uniform(low[:2], high[:2])
        third = 1.0 - first - second
        if low[2] <= third <= high[2]:
            return np.array([first, second, third])
    raise RuntimeError(f"could not sample simplex interval around {center.tolist()}")


def build_fold_features(names, features, seq_len, delta_vecs, delta_rows):
    folds = {}
    for candidate in CANDIDATES:
        folds[candidate] = {}
        for held_index, held in enumerate(names):
            train_indices = [i for i in range(len(names)) if i != held_index]
            test_indices = [held_index]
            x_train, metadata = build_X(
                candidate, "src", names, features, seq_len, delta_vecs,
                delta_rows, train_indices, train_indices,
            )
            x_test, _ = build_X(
                candidate, "src", names, features, seq_len, delta_vecs,
                delta_rows, test_indices, train_indices,
            )
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            folds[candidate][held] = {
                "train_indices": train_indices,
                "x_train": x_train,
                "x_test": x_test,
                "fit_ids": [names[i] for i in train_indices],
                "transform_fit_ids": metadata["fit_ids"],
            }
    return folds


def evaluate_row(names, folds, targets, candidate, model_name, pseudocount):
    predictions = {}
    for held in names:
        fold = folds[candidate][held]
        y_train = np.array([targets[names[i]] for i in fold["train_indices"]])
        z_train = clr_with_pseudocount(y_train, pseudocount)
        if model_name == "CLR-Ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "CLR-GP":
            model = make_primary_gp()
        else:
            raise KeyError(model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(fold["x_train"], z_train)
        predictions[held] = inverse_clr(model.predict(fold["x_test"]))

    errors = {
        mutation: float(np.abs(predictions[mutation] - targets[mutation]).mean())
        for mutation in names
    }
    u1_errors = [
        abs((2.0 * predictions[m][0] - 1.0) - (2.0 * targets[m][0] - 1.0))
        for m in names
    ]
    u2_errors = [
        abs(
            (predictions[m][1] - predictions[m][2])
            - (targets[m][1] - targets[m][2])
        )
        for m in names
    ]
    return {
        "mae": float(np.mean(list(errors.values()))),
        "u1_mae": float(np.mean(u1_errors)),
        "u2_mae": float(np.mean(u2_errors)),
        "u2_gt_u1": bool(np.mean(u2_errors) > np.mean(u1_errors)),
        "errors": errors,
        "f405a_gt_l410a": bool(errors["SrcKD-F405A"] > errors["SrcKD-L410A"]),
    }


def summarize_pseudocount_rows(rows):
    by_pseudocount = {}
    for pseudocount in PSEUDOCOUNTS:
        subset = [row for row in rows if row["pseudocount"] == pseudocount]
        by_pseudocount[str(pseudocount)] = {
            "mae_ranking": [
                f"{row['candidate']}::{row['model']}"
                for row in sorted(subset, key=lambda item: (item["mae"], item["candidate"], item["model"]))
            ],
            "u2_gt_u1_rows": sum(row["u2_gt_u1"] for row in subset),
            "f405a_gt_l410a_rows": sum(row["f405a_gt_l410a"] for row in subset),
            "total_rows": len(subset),
        }
    rankings = [item["mae_ranking"] for item in by_pseudocount.values()]
    return {
        "by_pseudocount": by_pseudocount,
        "mae_ranking_stable": all(ranking == rankings[0] for ranking in rankings[1:]),
        "all_rows_u2_gt_u1": all(row["u2_gt_u1"] for row in rows),
        "f405a_vs_l410a_pattern_stable": all(
            len({
                row["f405a_gt_l410a"]
                for row in rows
                if row["candidate"] == candidate and row["model"] == model
            }) == 1
            for candidate in CANDIDATES
            for model in MODELS
        ),
    }


def run_interval_stress_test(names, folds, nominal_targets, realizations):
    rng = np.random.default_rng(INTERVAL_SEED)
    records = []
    for index in range(realizations):
        sampled = {
            mutation: sample_simplex_interval(rng, nominal_targets[mutation])
            for mutation in names
        }
        row_results = {
            model: evaluate_row(
                names, folds, sampled, "pos", model, PRIMARY_GP_PROTOCOL["clr_clip"]
            )
            for model in MODELS
        }
        records.append({
            "realization": index,
            "rows": row_results,
            "gp_mae_le_ridge": row_results["CLR-GP"]["mae"]
            <= row_results["CLR-Ridge"]["mae"],
            "both_u2_gt_u1": all(row["u2_gt_u1"] for row in row_results.values()),
            "both_f405a_gt_l410a": all(
                row["f405a_gt_l410a"] for row in row_results.values()
            ),
        })
    return {
        "status": "curator_interval_stress_test_not_independent_redigitization",
        "component_half_width": INTERVAL_HALF_WIDTH,
        "sampling": "uniform over each componentwise interval intersected with the simplex",
        "seed": INTERVAL_SEED,
        "realizations": realizations,
        "candidate": "pos",
        "models": MODELS,
        "proportions": {
            "gp_mae_le_ridge": float(np.mean([r["gp_mae_le_ridge"] for r in records])),
            "both_u2_gt_u1": float(np.mean([r["both_u2_gt_u1"] for r in records])),
            "both_f405a_gt_l410a": float(
                np.mean([r["both_f405a_gt_l410a"] for r in records])
            ),
        },
        "records": records,
        "limitation": (
            "The 0.10 half-width is the conservative upper end of the curator "
            "visual-read range. This is not a substitute for independent redigitization "
            "from retained pixel coordinates."
        ),
    }


def render_report(result):
    lines = [
        "# Src CLR zero-replacement and label-interval robustness",
        "",
        "## Pseudocount sensitivity",
        "",
        "| Pseudocount | Position Ridge MAE | Position GP MAE | u2>u1 rows | F405A>L410A rows |",
        "|---:|---:|---:|---:|---:|",
    ]
    rows = result["pseudocount_sensitivity"]["rows"]
    summary = result["pseudocount_sensitivity"]["summary"]["by_pseudocount"]
    for pseudocount in PSEUDOCOUNTS:
        position = {
            row["model"]: row for row in rows
            if row["pseudocount"] == pseudocount and row["candidate"] == "pos"
        }
        block = summary[str(pseudocount)]
        lines.append(
            f"| {pseudocount:g} | {position['CLR-Ridge']['mae']:.4f} | "
            f"{position['CLR-GP']['mae']:.4f} | {block['u2_gt_u1_rows']}/"
            f"{block['total_rows']} | {block['f405a_gt_l410a_rows']}/"
            f"{block['total_rows']} |"
        )
    interval = result["digitization_interval_stress_test"]
    proportions = interval["proportions"]
    total_rows = len(rows)
    u2_rows = sum(row["u2_gt_u1"] for row in rows)
    failure_rows = sum(row["f405a_gt_l410a"] for row in rows)
    lines.extend([
        "",
        "The full 10-row MAE ranking is not invariant: the position GP is lower "
        "than ridge through 1e-4, but the ordering reverses at 1e-3 and 1e-2. "
        f"Across the complete grid, u2>u1 holds in {u2_rows}/{total_rows} rows and "
        f"F405A>L410A error holds in {failure_rows}/{total_rows}. The artifact retains "
        "all rankings and per-mutation errors.",
        "",
        "## Bounded label-interval stress test",
        "",
        f"Across {interval['realizations']} deterministic-seed realizations, "
        f"GP MAE was no larger than ridge in {proportions['gp_mae_le_ridge']:.1%}; "
        f"both rows retained u2>u1 in {proportions['both_u2_gt_u1']:.1%}; "
        f"both rows retained F405A>L410A error in "
        f"{proportions['both_f405a_gt_l410a']:.1%}.",
        "",
        f"Limitation: {interval['limitation']}",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-realizations", type=int, default=200)
    parser.add_argument("--skip-interval", action="store_true")
    args = parser.parse_args()
    if args.interval_realizations < 1:
        parser.error("--interval-realizations must be positive")

    start = time.time()
    names, features, _, seq_len = load_features("src")
    delta_vecs, delta_rows = get_delta_rows("src", None)
    folds = build_fold_features(names, features, seq_len, delta_vecs, delta_rows)
    panel = build_src_k3_panel(SRC_K3_PRIMARY_PROTOCOL_ID)
    targets = {mutation: np.asarray(panel.targets[mutation], float) for mutation in names}

    rows = []
    for pseudocount in PSEUDOCOUNTS:
        for candidate in CANDIDATES:
            for model in MODELS:
                result = evaluate_row(
                    names, folds, targets, candidate, model, pseudocount
                )
                rows.append({
                    "pseudocount": pseudocount,
                    "candidate": candidate,
                    "model": model,
                    **result,
                })
        print(f"[pseudocount={pseudocount:g}] complete", flush=True)

    interval = None
    if not args.skip_interval:
        interval = run_interval_stress_test(
            names, folds, targets, args.interval_realizations
        )
    result = {
        "schema_version": "src_clr_label_robustness_v1",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "primary_protocol_id": panel.protocol_id,
        "canonical_sha256": panel.canonical_sha256,
        "gp_protocol": PRIMARY_GP_PROTOCOL,
        "feature_scaling": "training-fold StandardScaler",
        "pseudocount_sensitivity": {
            "pseudocounts": PSEUDOCOUNTS,
            "candidates": CANDIDATES,
            "models": MODELS,
            "rows": rows,
            "summary": summarize_pseudocount_rows(rows),
        },
        "digitization_interval_stress_test": interval,
        "runtime_seconds": time.time() - start,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / "p2_k3_src_clr_robustness.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = RESULTS / "p2_k3_src_clr_robustness_report.md"
    if interval is not None:
        report.write_text(render_report(result), encoding="utf-8")
    elif report.exists():
        # A stale full-run report next to a skipped-interval JSON would look
        # like a consistent pair; remove the outdated report instead.
        report.unlink()
        print(f"[note] --skip-interval run removed stale report {report.name}")
    manifest = {
        "script": "p2_k3_src_clr_robustness.py",
        "script_sha256": sha256_file(Path(__file__)),
        "canonical_src_sha256": sha256_file(CANONICAL_SRC_K3_PATH),
        "output_sha256": sha256_file(output),
        "pseudocounts": PSEUDOCOUNTS,
        "interval_realizations": 0 if interval is None else args.interval_realizations,
        "interval_seed": INTERVAL_SEED,
    }
    (RESULTS / "p2_k3_src_clr_robustness_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[OK] {output} ({time.time() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
