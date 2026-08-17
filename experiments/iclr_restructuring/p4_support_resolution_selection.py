#!/usr/bin/env python
"""Generic K=3 collision, pooling, and representation-selection experiment.

This is an independent experiment.  It does not import, modify, or reinterpret
the earlier low-rank sample-complexity scripts.  Every factorial cell uses 200
fixed Monte Carlo repeats.  CLR-ridge is evaluated with analytic LOO formulas;
1-NN is the capacity control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
FIGURES = ROOT / "paper" / "figures_v2"

N_GRID = [6, 8, 12, 20, 50]
EPS_GRID = [0.0, 0.05, 0.2, 1.0]
DELTA_GRID = [0.0, 0.6, 1.2]
M_GRID = [1, 5, 20]
RESOLUTIONS = ["full_k3", "pooled_k2"]
REPEATS = 200
BASE_SEED = 20260812
ALPHA = 1.0

FIGURE_SLICE_METADATA = {
    "panel_A": {
        "fixed": {"n": 20, "m": 1, "resolution": "full_k3"},
        "varied": ["epsilon", "delta"],
        "averaged_over": [],
        "monte_carlo_repeats_per_setting": REPEATS,
    },
    "panel_B": {
        "fixed": {"n": 20, "m": 1},
        "varied": ["delta", "resolution", "contrast"],
        "averaged_over": ["epsilon"],
        "monte_carlo_repeats_per_setting": REPEATS,
    },
    "panel_C": {
        "fixed": {},
        "varied": ["m", "resolution"],
        "averaged_over": ["n", "epsilon", "delta"],
        "complete_factorial": True,
        "factorial_dimensions": {
            "n": N_GRID, "epsilon": EPS_GRID, "delta": DELTA_GRID,
            "m": M_GRID, "resolution": RESOLUTIONS,
        },
        "monte_carlo_repeats_per_setting": REPEATS,
    },
}


def factorial_counts():
    settings = (len(N_GRID) * len(EPS_GRID) * len(DELTA_GRID)
                * len(M_GRID) * len(RESOLUTIONS))
    return {"settings": settings, "repeats": REPEATS,
            "records": settings * REPEATS}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    ex = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)


def latent_targets(x, delta):
    """Three-state composition with an independently hidden fine contrast."""
    x = np.asarray(x, dtype=float)
    coarse = 1.25 * x + 0.20 * np.sin(np.pi * x)
    # The fine contrast is otherwise absent: delta is the sole controlled
    # source of state-1/state-2 ambiguity, so pooling and collision effects are
    # not confounded by unrelated nonlinear target structure.
    fine = np.zeros_like(x)
    # Members 0 and 1 have an unobserved opposing state-1/state-2 contrast.
    fine[0] -= delta / 2.0
    fine[1] += delta / 2.0
    logits = np.column_stack([coarse, -0.5 * coarse + fine,
                              -0.5 * coarse - fine])
    return softmax(logits)


def make_dataset(n, epsilon, delta, rng, max_m=20):
    x = rng.uniform(-1.0, 1.0, size=n)
    x[0] = -0.25
    x[1] = -0.25 + epsilon
    targets = latent_targets(x, delta)
    features = np.empty((n, max_m), dtype=float)
    features[:, 0] = x
    for j in range(1, max_m):
        rho = rng.uniform(0.0, 0.75)
        noise = rng.normal(size=n)
        # Identical visible features remain exactly identical at epsilon=0.
        noise[1] = noise[0]
        features[:, j] = rho * x + math.sqrt(1.0 - rho * rho) * noise
    return features, targets


def clr(y):
    y = np.clip(np.asarray(y, dtype=float), 1e-6, 1.0)
    y /= y.sum(axis=-1, keepdims=True)
    return np.log(y) - np.log(y).mean(axis=-1, keepdims=True)


def inv_clr(z):
    return softmax(z)


def to_latent(targets, resolution):
    if resolution == "full_k3":
        return clr(targets)
    pooled = np.column_stack([targets[:, 0], targets[:, 1:].sum(axis=1)])
    return np.log(np.clip(pooled[:, :1], 1e-6, 1.0) /
                  np.clip(pooled[:, 1:], 1e-6, 1.0))


def from_latent(latent, resolution):
    latent = np.asarray(latent, dtype=float)
    if resolution == "full_k3":
        return inv_clr(latent)
    p0 = 1.0 / (1.0 + np.exp(-np.clip(latent[..., 0], -40, 40)))
    return np.stack([p0, 1.0 - p0], axis=-1)


def target_for_resolution(targets, resolution):
    if resolution == "full_k3":
        return targets
    return np.column_stack([targets[:, 0], targets[:, 1:].sum(axis=1)])


def fit_scaler(features, fit_ids):
    z = np.asarray(features, float)
    mean = z.mean(axis=0)
    scale = z.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    audit = {
        "fit_ids": [int(i) for i in fit_ids],
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }
    return mean, scale, audit


def ridge_loo_candidates(features, latent, resolution, alpha=ALPHA,
                         sample_ids=None, return_audit=False):
    """LOO predictions with a separately fitted training-only scaler per fold."""
    z = np.asarray(features, float)
    y = np.asarray(latent, float)
    n, m = z.shape
    ids = np.arange(n, dtype=int) if sample_ids is None else np.asarray(sample_ids)
    train_n = n - 1
    all_rows = np.arange(n, dtype=int)
    train_rows = np.stack([np.delete(all_rows, held) for held in range(n)])
    train_features = z[train_rows]
    train_targets = y[train_rows]
    mean = train_features.mean(axis=1)
    scale = train_features.std(axis=1, ddof=0)
    scale[scale < 1e-12] = 1.0

    standardized = (train_features - mean[:, None, :]) / scale[:, None, :]
    target_mean = train_targets.mean(axis=1)
    centered_targets = train_targets - target_mean[:, None, :]
    standardized_cross = np.einsum(
        "nim,nik->nmk", standardized, centered_targets)
    standardized_square = np.sum(standardized * standardized, axis=1)
    slope = standardized_cross / (standardized_square[:, :, None] + alpha)
    held_standardized = (z - mean) / scale
    held_latent = (target_mean[:, None, :]
                   + held_standardized[:, :, None] * slope)
    pred = from_latent(held_latent, resolution)
    if return_audit:
        audits = []
        for held in range(n):
            audits.append({
                "held_out_id": int(ids[held]),
                "fit_ids": [int(i) for i in np.delete(ids, held)],
                "mean": mean[held].tolist(),
                "scale": scale[held].tolist(),
            })
        return pred, audits
    return pred


def _ridge_fit_predict_scaled(z, y, zt, resolution, alpha):
    n, m = z.shape
    sy = y.sum(axis=0)
    sz = z.sum(axis=0)
    szz = (z * z).sum(axis=0) + alpha
    szy = z.T @ y
    det = n * szz - sz * sz
    b0 = (szz[:, None] * sy[None, :] - sz[:, None] * szy) / det[:, None]
    b1 = (-sz[:, None] * sy[None, :] + n * szy) / det[:, None]
    predicted_latent = b0 + zt[:, None] * b1
    return from_latent(predicted_latent, resolution)


def ridge_fit_predict_candidates(train_features, train_latent, test_features,
                                 resolution, alpha=ALPHA, fit_ids=None,
                                 return_audit=False):
    raw_train = np.asarray(train_features, float)
    raw_test = np.asarray(test_features, float)
    y = np.asarray(train_latent, float)
    if fit_ids is None:
        fit_ids = np.arange(len(raw_train), dtype=int)
    mean, scale, audit = fit_scaler(raw_train, fit_ids)
    z = (raw_train - mean) / scale
    zt = (raw_test - mean) / scale
    pred = _ridge_fit_predict_scaled(z, y, zt, resolution, alpha)
    if return_audit:
        return pred, audit
    return pred


def candidate_mae(pred, target):
    # pred: n x m x K; target: n x K
    return np.abs(pred - target[:, None, :]).mean(axis=(0, 2))


def nested_predictions(features, targets, resolution, m_values=M_GRID,
                       sample_ids=None):
    n, max_m = features.shape
    ids = np.arange(n, dtype=int) if sample_ids is None else np.asarray(sample_ids)
    latent = to_latent(targets, resolution)
    target = target_for_resolution(targets, resolution)
    selected = {m: [] for m in m_values}
    predictions = {m: [] for m in m_values}
    preprocessing_audit = []
    for held in range(n):
        train = np.array([i for i in range(n) if i != held], dtype=int)
        inner_pred, inner_audit = ridge_loo_candidates(
            features[train], latent[train], resolution, sample_ids=ids[train],
            return_audit=True)
        inner_error = candidate_mae(inner_pred, target[train])
        outer_pred, outer_audit = ridge_fit_predict_candidates(
            features[train], latent[train], features[held], resolution,
            fit_ids=ids[train], return_audit=True)
        for m in m_values:
            choice = int(np.argmin(inner_error[:m]))
            selected[m].append(choice)
            predictions[m].append(outer_pred[choice])
        preprocessing_audit.append({
            "held_out_id": int(ids[held]),
            "outer_scaler": outer_audit,
            "inner_scalers": inner_audit,
        })
    return ({m: np.asarray(predictions[m]) for m in m_values}, selected,
            preprocessing_audit)


def preprocessing_is_isolated(audit):
    for outer in audit:
        outer_id = outer["held_out_id"]
        if outer_id in outer["outer_scaler"]["fit_ids"]:
            return False
        for inner in outer["inner_scalers"]:
            if outer_id in inner["fit_ids"] or inner["held_out_id"] in inner["fit_ids"]:
                return False
    return True


def pair_separation_report(features, loo_scaler_audit):
    raw = float(abs(features[1, 0] - features[0, 0]))
    transformed = []
    for held in (0, 1):
        scaler = loo_scaler_audit[held]
        transformed.append(float(raw / scaler["scale"][0]))
    return {
        "raw": raw,
        "outer_fold_transformed": transformed,
        "outer_fold_transformed_mean": float(np.mean(transformed)),
    }


def nn_loo(features, targets):
    z = features[:, 0]
    distance = np.abs(z[:, None] - z[None, :])
    np.fill_diagonal(distance, np.inf)
    neighbor = np.argmin(distance, axis=1)
    return targets[neighbor]


def metrics_for_repeat(n, epsilon, delta, repeat):
    seed = BASE_SEED + n * 10_000_000 + int(round(epsilon * 100)) * 100_000 \
           + int(round(delta * 10)) * 1_000 + repeat
    rng = np.random.default_rng(seed)
    features, targets = make_dataset(n, epsilon, delta, rng)
    records = []
    pair_floor_full = 0.5 * float(np.abs(targets[0] - targets[1]).mean())
    for resolution in RESOLUTIONS:
        target = target_for_resolution(targets, resolution)
        latent = to_latent(targets, resolution)
        loo_all, loo_audit = ridge_loo_candidates(
            features, latent, resolution, return_audit=True)
        naive_errors = candidate_mae(loo_all, target)
        nested_pred, selected, audit = nested_predictions(features, targets, resolution)
        separation = pair_separation_report(features, loo_audit)
        isolation_ok = preprocessing_is_isolated(audit)
        nn_pred = nn_loo(features, target)
        fixed_pred = loo_all[:, 0, :]
        shared_fixed = float(np.abs(fixed_pred[:, 0] - target[:, 0]).mean())
        shared_nn = float(np.abs(nn_pred[:, 0] - target[:, 0]).mean())
        if resolution == "full_k3":
            fine_fixed = float(np.abs((fixed_pred[:, 1] - fixed_pred[:, 2])
                                      - (target[:, 1] - target[:, 2])).mean())
            pair_floor = pair_floor_full
        else:
            fine_fixed = None
            pair_floor = 0.5 * float(np.abs(target[0] - target[1]).mean())
        for m in M_GRID:
            nested_error = float(np.abs(nested_pred[m] - target).mean())
            naive_error = float(np.min(naive_errors[:m]))
            records.append({
                "n": n, "epsilon": epsilon, "raw_epsilon": epsilon,
                "raw_pair_separation": separation["raw"],
                "outer_fold_transformed_pair_separation":
                    separation["outer_fold_transformed"],
                "outer_fold_transformed_pair_separation_mean":
                    separation["outer_fold_transformed_mean"],
                "delta": delta,
                "m": m, "resolution": resolution, "repeat": repeat,
                "seed": seed, "clr_ridge_fixed_mae": float(naive_errors[0]),
                "nn_fixed_mae": float(np.abs(nn_pred - target).mean()),
                "pair_clr_ridge_mae": float(np.abs(fixed_pred[:2] - target[:2]).mean()),
                "pair_nn_mae": float(np.abs(nn_pred[:2] - target[:2]).mean()),
                "equal_prediction_pair_floor": pair_floor,
                "shared_contrast_clr_mae": shared_fixed,
                "shared_contrast_nn_mae": shared_nn,
                "fine_contrast_clr_mae": fine_fixed,
                "naive_selected_mae": naive_error,
                "nested_selected_mae": nested_error,
                "selection_optimism": nested_error - naive_error,
                "nested_selected_candidate_mean": float(np.mean(selected[m])),
                "nested_outer_fold_isolation": isolation_ok,
                "nested_inner_fold_isolation": isolation_ok,
                "simplex_ok": bool(np.all(fixed_pred >= -1e-12)
                                   and np.allclose(fixed_pred.sum(axis=1), 1.0)),
                "exact_collision": bool(abs(features[0, 0] - features[1, 0]) < 1e-12),
            })
    return records


NUMERIC_FIELDS = [
    "raw_pair_separation", "outer_fold_transformed_pair_separation_mean",
    "clr_ridge_fixed_mae", "nn_fixed_mae", "pair_clr_ridge_mae",
    "pair_nn_mae", "equal_prediction_pair_floor", "shared_contrast_clr_mae",
    "shared_contrast_nn_mae", "fine_contrast_clr_mae", "naive_selected_mae",
    "nested_selected_mae", "selection_optimism", "nested_selected_candidate_mean"]


def summarize(records):
    groups = {}
    for row in records:
        key = (row["n"], row["epsilon"], row["delta"], row["m"], row["resolution"])
        groups.setdefault(key, []).append(row)
    summary = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3], k[4])):
        rows = groups[key]
        item = {"n": key[0], "epsilon": key[1], "delta": key[2],
                "m": key[3], "resolution": key[4], "repeats": len(rows)}
        for field in NUMERIC_FIELDS:
            values = np.array([r[field] for r in rows if r[field] is not None], float)
            item[field + "_mean"] = None if len(values) == 0 else float(values.mean())
            item[field + "_se"] = None if len(values) == 0 else float(values.std(ddof=1) / np.sqrt(len(values)))
        item["all_simplex_ok"] = all(r["simplex_ok"] for r in rows)
        item["all_nested_outer_fold_isolation"] = all(
            r["nested_outer_fold_isolation"] for r in rows)
        item["all_nested_inner_fold_isolation"] = all(
            r["nested_inner_fold_isolation"] for r in rows)
        summary.append(item)
    return summary


def aggregate_plot(summary):
    def mean_sem(rows, field):
        vals = np.array([r[field] for r in rows], float)
        return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))

    plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.18))
    colors = {"clr": "#0072B2", "nn": "#D55E00", "pool": "#009E73",
              "fine": "#CC79A7", "floor": "#333333"}
    markers = {0.0: "o", 0.6: "s", 1.2: "^"}

    # A: n=20, m=1, full K3; curves stratified by hidden contrast.
    ax = axes[0]
    for delta in DELTA_GRID:
        rows = [r for r in summary if r["n"] == 20 and r["m"] == 1
                and r["resolution"] == "full_k3" and r["delta"] == delta]
        rows.sort(key=lambda r: r["epsilon"])
        ax.plot([r["epsilon"] for r in rows],
                [r["pair_clr_ridge_mae_mean"] for r in rows],
                marker=markers[delta], lw=1.2, ms=3.2, label=fr"ridge, $\delta={delta:g}$")
    floor_row = [r for r in summary if r["n"] == 20 and r["m"] == 1
                 and r["resolution"] == "full_k3" and r["delta"] == 1.2
                 and r["epsilon"] == 0.0][0]
    ax.scatter([0.0], [floor_row["equal_prediction_pair_floor_mean"]],
               color=colors["floor"], marker="*", s=28, zorder=4,
               label=r"strict floor ($\epsilon=0$)")
    ax.set(xlabel=r"collision separation $\epsilon$", ylabel="collision-pair MAE",
           title="A  Exact collisions impose an error floor")
    ax.legend(frameon=False, fontsize=5.7, ncol=2, handlelength=1.5)

    # B: m=1, n=20, averaged over epsilon.
    ax = axes[1]
    for resolution, field, label, color, marker in [
        ("full_k3", "shared_contrast_clr_mae_mean", "full K=3: shared", colors["clr"], "o"),
        ("pooled_k2", "shared_contrast_clr_mae_mean", "pooled K=2: shared", colors["pool"], "s"),
        ("full_k3", "fine_contrast_clr_mae_mean", "full K=3: fine", colors["fine"], "^")]:
        ys = []
        for delta in DELTA_GRID:
            rows = [r for r in summary if r["n"] == 20 and r["m"] == 1
                    and r["resolution"] == resolution and r["delta"] == delta]
            ys.append(float(np.mean([r[field] for r in rows])))
        ax.plot(DELTA_GRID, ys, color=color, marker=marker, lw=1.3, ms=3.3,
                label=label)
    ax.set(xlabel=r"hidden fine contrast $\delta$", ylabel="contrast MAE",
           title="B  Pooling hides fine-state error")
    ax.legend(frameon=False, fontsize=5.8)

    # C: all n/epsilon/delta, by resolution.
    ax = axes[2]
    for resolution, label, color, marker in [
        ("full_k3", "full K=3", colors["clr"], "o"),
        ("pooled_k2", "pooled K=2", colors["pool"], "s")]:
        ys, ses = [], []
        for m in M_GRID:
            vals = np.array([r["selection_optimism_mean"] for r in summary
                             if r["m"] == m and r["resolution"] == resolution], float)
            ys.append(vals.mean()); ses.append(vals.std(ddof=1) / np.sqrt(len(vals)))
        ax.errorbar(M_GRID, ys, yerr=ses, color=color, marker=marker, lw=1.3,
                    ms=3.3, capsize=2, label=label)
    ax.axhline(0, color="#777777", lw=0.7)
    ax.set(xlabel="representation candidates m", ylabel="nested − naive MAE",
           title="C  Naive selection becomes optimistic", xticks=M_GRID)
    ax.legend(frameon=False, fontsize=6)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", lw=0.45)
    fig.tight_layout(w_pad=1.2)
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / "fig4_synthetic_framework.pdf"
    png = FIGURES / "fig4_synthetic_framework.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def self_test():
    rng = np.random.default_rng(7)
    features, targets = make_dataset(8, 0.0, 1.2, rng)
    assert np.allclose(targets.sum(axis=1), 1.0)
    assert np.all(targets >= 0)
    assert features[0, 0] == features[1, 0]
    pooled = target_for_resolution(targets, "pooled_k2")
    assert np.allclose(pooled[:, 0], targets[:, 0])
    assert np.allclose(pooled[:, 1], targets[:, 1] + targets[:, 2])
    pred, _, audit = nested_predictions(features, targets, "full_k3")
    assert preprocessing_is_isolated(audit)
    assert all(fold["outer_scaler"]["fit_ids"] ==
               [j for j in range(8) if j != fold["held_out_id"]]
               for fold in audit)
    assert np.allclose(pred[20].sum(axis=1), 1.0)
    assert factorial_counts() == {"settings": 360, "repeats": 200,
                                  "records": 72000}
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="development only: one repeat; never overwrites frozen outputs")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("SELF_TEST_OK")
        return
    repeats = 1 if args.quick else REPEATS
    if args.quick:
        raise SystemExit("Quick mode is intentionally non-writing; remove --quick for the frozen run")

    started = time.time()
    records = []
    for n in N_GRID:
        for epsilon in EPS_GRID:
            for delta in DELTA_GRID:
                for repeat in range(repeats):
                    records.extend(metrics_for_repeat(n, epsilon, delta, repeat))
        print(f"completed n={n}: {len(records)} records", flush=True)
    summary = summarize(records)
    counts = factorial_counts()
    expected = counts["records"]
    assert len(records) == expected == 72000
    assert len(summary) == counts["settings"] == 360
    assert all(r["all_simplex_ok"] and r["all_nested_outer_fold_isolation"]
               and r["all_nested_inner_fold_isolation"] for r in summary)
    assert all(r["repeats"] == REPEATS for r in summary)

    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "p4_support_resolution_selection.json"
    csv_path = RESULTS / "p4_support_resolution_selection_by_setting.csv"
    payload = {
        "experiment": "generic collision, pooling, and representation-selection audit",
        "independent_of_old_low_rank_sample_complexity_script": True,
        "seed": BASE_SEED,
        "factorial": {"n": N_GRID, "epsilon": EPS_GRID, "delta": DELTA_GRID,
                       "m": M_GRID, "resolution": RESOLUTIONS, "repeats": REPEATS,
                       "expected_settings": counts["settings"],
                       "expected_records": expected},
        "models": {"primary_simulator": "CLR-Ridge alpha=1",
                    "capacity_control": "1-NN"},
        "preprocessing_protocol": {
            "raw_features_retained": True,
            "outer_scaler_fit_scope": "outer_training_ids_only",
            "inner_scaler_fit_scope": "inner_training_ids_only",
            "scaler_audit_fields": ["fit_ids", "mean", "scale"],
        },
        "definitions": {
            "collision_pair": "samples 0 and 1; exact visible collision at epsilon=0",
            "epsilon": "raw first-candidate feature distance before fold-local scaling",
            "transformed_pair_separation":
                "raw pair distance divided by each outer-training-only scaler scale",
            "pooling_matrix": [[1, 0, 0], [0, 1, 1]],
            "selection_optimism": "fully nested outer-LOO MAE minus naive minimum LOO MAE",
            "preprocessing_audit":
                "nested_predictions returns every outer and inner scaler fit_ids, mean, and scale",
            "all_settings_retained": True},
        "figure_slices": FIGURE_SLICE_METADATA,
        "summary": summary, "records": records,
        "runtime_seconds": time.time() - started,
    }
    json_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n",
                         encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    pdf_path, png_path = aggregate_plot(summary)
    manifest = {
        "script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "script_sha256": sha256_file(Path(__file__)),
        "json_sha256": sha256_file(json_path), "csv_sha256": sha256_file(csv_path),
        "figure_pdf_sha256": sha256_file(pdf_path),
        "figure_png_sha256": sha256_file(png_path),
        "records": len(records), "settings": len(summary), "repeats_per_setting": REPEATS,
        "all_settings_retained": True, "self_test": "passed",
    }
    manifest_path = RESULTS / "p4_support_resolution_selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
