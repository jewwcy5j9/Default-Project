#!/usr/bin/env python
"""Frozen prospective-v3 custodian runner.

The modeler prepares ``public_input.json`` and ``esm2_features.npz`` without
mutant-population labels.  The custodian keeps ``private_targets.json`` private
and runs this file once in an isolated environment.  Every outer fold fits its
own PCA using only residue-delta rows belonging to the outer-training mutants.

This module deliberately contains no system-specific positions or site markers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import Ridge

PKG_ROOT = Path(__file__).resolve().parent
ROOT = PKG_ROOT.parents[1]
# Prefer a package-local src/ (the frozen custodian package vendors
# src/models/low_rank_cdst.py so it executes from a clean extraction);
# fall back to the repository root for in-repo runs.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
from src.models.low_rank_cdst import LowRankCDST  # noqa: E402

PROTOCOL_VERSION = "3.0.0"
DIAGNOSTIC_LAYER = "3.1.0"
CHECKPOINT = "facebook/esm2_t33_650M_UR50D"
K = 3
PCA_MAX_DIM = 20
RANK = 2
HIDDEN = 32
EPOCHS = 800
LR = 5e-3
WEIGHT_DECAY = 1e-4
N_SEEDS = 5
EPS = 1e-8
COLLISION_TOL = 1e-8
TARGET_CONFLICT_L1 = 0.10
DIAG_NULL_ALPHA = 0.05
DIAG_P3_BAND = (0.005, 0.15)

# Frozen scoring definitions (mirror preregistration_v3_1_amendment, sec. 2-3).
DIAGNOSTIC_DEFINITIONS = {
    "layer": DIAGNOSTIC_LAYER,
    "p1": ("collision label-permutation null: not estimable OR p>=0.05; "
           "pairwise Euclidean distance on 1280-d mutation vectors, panel-median "
           "normalized, tolerance 1e-8, conflict L1>=0.10"),
    "p2": ("orthonormal q1=(2,-1,-1)/sqrt(6), q2=(0,1,-1)/sqrt(2): "
           "q2 MAE > q1 MAE AND pooled q1-scale MAE >= full q1 MAE "
           "(q1-scale of pooled = (sqrt(6)/2) * shared active MAE)"),
    "p3": ("selection optimism: nested (per-fold inner-LOO over the four fast "
           "candidates) minus naive (argmin panel-mean fold MAE) lies in [0.005, 0.15]"),
    "p4": ("support-error association: Spearman rho of support distance vs "
           "primary LOO MAE is positive"),
    "scoring": ("each of P1-P4 scores 1/0 at reveal; P1 not estimable -> N/A and "
                "denominator 3; framework_validated iff (den=4 and >=3/4) or "
                "(den=3 and 3/3); independent of the v3.0 primary gate; no "
                "post-hoc reinterpretation"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def norm_simplex(p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError("invalid probability vector")
    return p / s


def clr(y):
    y = np.clip(np.asarray(y, dtype=float), 1e-6, 1.0)
    y = y / y.sum(axis=-1, keepdims=True)
    return np.log(y) - np.log(y).mean(axis=-1, keepdims=True)


def inv_clr(z):
    z = np.atleast_2d(np.asarray(z, dtype=float))
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def ilr(y):
    p = np.clip(np.asarray(y, dtype=float), 1e-6, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    coarse = np.sqrt(2.0 / 3.0) * np.log(
        p[..., 0] / np.sqrt(p[..., 1] * p[..., 2]))
    fine = (1.0 / np.sqrt(2.0)) * np.log(p[..., 1] / p[..., 2])
    return np.stack([coarse, fine], axis=-1)


def mae(p, y) -> float:
    return float(np.abs(np.asarray(p) - np.asarray(y)).mean())


def validate_public(public):
    required = ["protocol_version", "panel_id", "wild_type_sequence",
                "wt_population", "state_definitions", "conditions",
                "mutations"]
    missing = [key for key in required if key not in public]
    if missing:
        raise ValueError(f"public input missing: {missing}")
    if public["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("protocol version mismatch")
    wt = np.asarray(public["wt_population"], dtype=float)
    if wt.shape != (K,) or np.any(wt < 0) or not np.isclose(wt.sum(), 1.0):
        raise ValueError("WT population must be a K=3 simplex vector")
    if len(public["state_definitions"]) != K:
        raise ValueError("exactly three frozen state definitions are required")
    ids = [m["mutation_id"] for m in public["mutations"]]
    if len(ids) != len(set(ids)):
        raise ValueError("mutation IDs must be unique")
    if len(ids) < 5:
        raise ValueError("v3 runner requires at least five evaluable mutants")
    seq = public["wild_type_sequence"]
    if not seq or any(a not in "ACDEFGHIKLMNPQRSTVWY" for a in seq):
        raise ValueError("wild_type_sequence must contain canonical amino acids")
    for mut in public["mutations"]:
        if not mut.get("substitutions"):
            raise ValueError(f"{mut['mutation_id']}: no substitutions")
        for sub in mut["substitutions"]:
            i = int(sub["sequence_index_1based"]) - 1
            if i < 0 or i >= len(seq) or seq[i] != sub["from"]:
                raise ValueError(f"{mut['mutation_id']}: sequence-index mismatch")
    tier = "primary_eligible" if len(ids) >= 8 else "supporting_only"
    return ids, wt, tier


def validate_private(private, ids):
    targets = private.get("mutant_populations", {})
    if set(targets) != set(ids):
        raise ValueError("private target IDs do not exactly match public IDs")
    out = {}
    for mid in ids:
        y = np.asarray(targets[mid], dtype=float)
        if y.shape != (K,) or np.any(y < 0) or not np.isclose(y.sum(), 1.0,
                                                               atol=1e-7):
            raise ValueError(f"{mid}: target is not a K=3 simplex vector")
        out[mid] = y
    return out


def load_features(path: Path, ids):
    z = np.load(path, allow_pickle=False)
    vectors, rows = {}, {}
    for mid in ids:
        vk, rk = f"delta_vector::{mid}", f"delta_rows::{mid}"
        if vk not in z or rk not in z:
            raise ValueError(f"feature package missing {vk} or {rk}")
        vectors[mid] = np.asarray(z[vk], dtype=float)
        rows[mid] = np.asarray(z[rk], dtype=float)
        if vectors[mid].ndim != 1 or rows[mid].ndim != 2:
            raise ValueError(f"{mid}: malformed feature dimensions")
        if rows[mid].shape[1] != vectors[mid].shape[0]:
            raise ValueError(f"{mid}: vector/row embedding dimensions differ")
    dims = {v.shape[0] for v in vectors.values()}
    lengths = {r.shape[0] for r in rows.values()}
    if len(dims) != 1 or len(lengths) != 1:
        raise ValueError("all embeddings must share dimension and sequence length")
    return vectors, rows


def fit_outer_pca(train_ids, vectors, rows):
    d_eff = max(1, min(PCA_MAX_DIM, len(train_ids) - 1))
    train_rows = np.vstack([rows[mid] for mid in train_ids])
    pca = PCA(n_components=d_eff, svd_solver="full")
    pca.fit(train_rows)
    meta = {
        "fit_ids": list(train_ids),
        "n_residue_rows": int(train_rows.shape[0]),
        "embedding_dim": int(train_rows.shape[1]),
        "d_eff": int(d_eff),
        "pca_parameter_sha256": hashlib.sha256(
            np.ascontiguousarray(pca.components_).tobytes()
            + np.ascontiguousarray(pca.mean_).tobytes()
            + np.ascontiguousarray(pca.explained_variance_).tobytes()
        ).hexdigest(),
    }
    X = {mid: pca.transform(vectors[mid][None, :])[0] for mid in vectors}
    return X, meta


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def lowrank_predict(X_train, y_train, x_test, wt, seed, k=K):
    set_seed(seed)
    rank = min(RANK, k - 1)
    model = LowRankCDST(K=k, intervention_dim=X_train.shape[1], rank=rank,
                        hidden_dim=HIDDEN).cpu()
    opt = torch.optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    wtr = torch.tensor(np.tile(wt, (len(X_train), 1)), dtype=torch.float32)
    xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    best_loss, best_state = math.inf, None
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = F.mse_loss(model(wtr, xtr), ytr)
        loss.backward()
        opt.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_state = {name: value.detach().clone()
                          for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(wt[None, :], dtype=torch.float32),
                     torch.tensor(x_test[None, :], dtype=torch.float32))[0]
    return norm_simplex(pred.numpy()), best_loss


def fast_predict(kind, X_train, y_train, x_test, wt):
    if kind == "clr_ridge":
        model = Ridge(alpha=1.0).fit(X_train, clr(y_train))
        return norm_simplex(inv_clr(model.predict(x_test[None, :]))[0])
    if kind == "clr_gp":
        model = GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5),
            normalize_y=True, random_state=0, n_restarts_optimizer=0)
        model.fit(X_train, clr(y_train))
        return norm_simplex(inv_clr(model.predict(x_test[None, :]))[0])
    if kind == "training_mean":
        return norm_simplex(np.mean(y_train, axis=0))
    if kind == "constant_wt":
        return norm_simplex(wt)
    raise KeyError(kind)


def exact_spearman_permutation(x, y, max_n=9):
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = float(spearmanr(x, y).statistic)
    if not np.isfinite(obs):
        return {"rho": None, "p_one_sided_greater": None,
                "method": "undefined_constant_rank"}
    if len(x) > max_n:
        # The preregistered exact endpoint remains explicitly unavailable;
        # a deterministic large-sample sensitivity is reported, never used as
        # a success gate.
        rng = np.random.default_rng(3001)
        n_perm = 100000
        exceed = sum(float(spearmanr(x, rng.permutation(y)).statistic) >= obs
                     for _ in range(n_perm))
        return {"rho": obs,
                "p_one_sided_greater": (exceed + 1) / (n_perm + 1),
                "method": "monte_carlo_sensitivity_n_gt_9",
                "n_permutations": n_perm}
    exceed, total = 0, 0
    for perm in itertools.permutations(y.tolist()):
        stat = float(spearmanr(x, perm).statistic)
        exceed += stat >= obs - 1e-15
        total += 1
    return {"rho": obs, "p_one_sided_greater": exceed / total,
            "method": "exhaustive_exact", "n_permutations": total}


def exact_sign_test(deltas):
    deltas = np.asarray(deltas, dtype=float)
    nonzero = deltas[np.abs(deltas) > 1e-15]
    n, k = len(nonzero), int(np.sum(nonzero > 0))
    if n == 0:
        return {"positive": 0, "nonzero": 0, "p_one_sided": 1.0}
    p = sum(math.comb(n, j) for j in range(k, n + 1)) / (2 ** n)
    return {"positive": k, "nonzero": n, "p_one_sided": p,
            "method": "exact_binomial_sign"}


def collision_diagnostic(ids, vectors, targets, predictions):
    scale = np.median([np.linalg.norm(v) for v in vectors.values()]) or 1.0
    pairs = []
    conflict_members = set()
    collision_edges = []
    for i, a in enumerate(ids):
        for j, b in enumerate(ids[i + 1:], start=i + 1):
            fd = float(np.linalg.norm(vectors[a] - vectors[b]) / scale)
            td = float(np.abs(targets[a] - targets[b]).sum())
            is_collision = fd <= COLLISION_TOL
            is_conflict = is_collision and td >= TARGET_CONFLICT_L1
            if is_collision:
                collision_edges.append((i, j))
                pairs.append({"a": a, "b": b, "feature_distance": fd,
                              "target_l1": td, "conflict": is_conflict})
            if is_conflict:
                conflict_members.update([a, b])
    errors = {mid: mae(predictions[mid], targets[mid]) for mid in ids}
    member = [errors[mid] for mid in ids if mid in conflict_members]
    other = [errors[mid] for mid in ids if mid not in conflict_members]
    enrichment = None if not member or not other else float(np.mean(member)
                                                             - np.mean(other))

    def permuted_enrichment(order):
        assigned = [targets[ids[index]] for index in order]
        members = set()
        for i, j in collision_edges:
            if float(np.abs(assigned[i] - assigned[j]).sum()) >= TARGET_CONFLICT_L1:
                members.update([i, j])
        if not members or len(members) == len(ids):
            return None
        perm_errors = [mae(predictions[ids[i]], assigned[i]) for i in range(len(ids))]
        return float(np.mean([perm_errors[i] for i in members])
                     - np.mean([perm_errors[i] for i in range(len(ids)) if i not in members]))

    null = {"status": "not_estimable_without_both_conflict_and_nonconflict_members"}
    if enrichment is not None:
        if len(ids) <= 9:
            orders = itertools.permutations(range(len(ids)))
            method = "exhaustive_exact_label_permutation"
        else:
            rng = np.random.default_rng(3002)
            orders = (rng.permutation(len(ids)).tolist() for _ in range(100000))
            method = "monte_carlo_sensitivity_n_gt_9"
        values = [value for value in (permuted_enrichment(order) for order in orders)
                  if value is not None]
        exceed = sum(value >= enrichment - 1e-15 for value in values)
        null = {"status": "computed", "method": method,
                "defined_permutations": len(values),
                "p_one_sided_greater": None if not values else exceed / len(values)}
    return {"feature_scale": float(scale), "feature_collision_tolerance": COLLISION_TOL,
            "target_conflict_l1_threshold": TARGET_CONFLICT_L1,
            "collision_pairs": pairs, "conflict_members": sorted(conflict_members),
            "error_enrichment": enrichment,
            "label_permutation_null": null}


def inner_candidate_error(kind, outer_train_ids, vectors, rows, targets, wt):
    errs = []
    for held in outer_train_ids:
        inner_train = [mid for mid in outer_train_ids if mid != held]
        X, _ = fit_outer_pca(inner_train, vectors, rows)
        xtr = np.vstack([X[mid] for mid in inner_train])
        ytr = np.vstack([targets[mid] for mid in inner_train])
        pred = fast_predict(kind, xtr, ytr, X[held], wt)
        errs.append(mae(pred, targets[held]))
    return float(np.mean(errs))


def run(public_path: Path, private_path: Path, feature_path: Path,
        output_dir: Path, dry_run=False):
    started = time.time()
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    ids, wt, panel_tier = validate_public(public)
    targets = validate_private(private, ids)
    vectors, rows = load_features(feature_path, ids)

    torch.set_num_threads(1)
    deterministic_ok = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        deterministic_ok = False

    nested_candidates = ["clr_ridge", "clr_gp", "training_mean", "constant_wt"]

    def run_loo(ids_subset, fold_seed_rows, full_diagnostics=True):
        folds, primary_errors, baseline_errors = [], {}, {}
        support_distances, full_preds, pooled_preds = {}, {}, {}
        for test_index, held in enumerate(ids_subset):
            train_ids = [mid for mid in ids_subset if mid != held]
            X, pca_meta = fit_outer_pca(train_ids, vectors, rows)
            xtr = np.vstack([X[mid] for mid in train_ids])
            ytr = np.vstack([targets[mid] for mid in train_ids])
            per_seed, per_loss = [], []
            for s in range(N_SEEDS):
                seed = s * 100 + test_index
                pred, loss = lowrank_predict(xtr, ytr, X[held], wt, seed)
                per_seed.append(pred)
                per_loss.append(loss)
                fold_seed_rows.append({"mutation_id": held, "seed": seed,
                                       "p0": pred[0], "p1": pred[1], "p2": pred[2]})
            pred = norm_simplex(np.mean(per_seed, axis=0))
            baseline = fast_predict("training_mean", xtr, ytr, X[held], wt)
            err, base_err = mae(pred, targets[held]), mae(baseline, targets[held])
            primary_errors[held], baseline_errors[held] = err, base_err
            support = float(min(np.linalg.norm(X[held] - X[mid]) for mid in train_ids))
            support_distances[held] = support
            full_preds[held], pooled_preds[held] = pred, None
            if not full_diagnostics:
                folds.append({"test_id": held, "outer_train_ids": train_ids,
                              "primary_mae": err, "training_mean_mae": base_err})
                continue

            constant_wt = fast_predict("constant_wt", xtr, ytr, X[held], wt)
            ridge = fast_predict("clr_ridge", xtr, ytr, X[held], wt)
            gp = fast_predict("clr_gp", xtr, ytr, X[held], wt)

            pooled_ytr = np.column_stack([ytr[:, 0], ytr[:, 1:].sum(axis=1)])
            pooled_wt = np.array([wt[0], wt[1:].sum()])
            pooled_seed = [lowrank_predict(xtr, pooled_ytr, X[held], pooled_wt,
                                           s * 100 + test_index, k=2)[0]
                           for s in range(N_SEEDS)]
            pooled = norm_simplex(np.mean(pooled_seed, axis=0))

            inner = {kind: inner_candidate_error(kind, train_ids, vectors, rows,
                                                 targets, wt)
                     for kind in nested_candidates}
            chosen = min(nested_candidates,
                         key=lambda key: (inner[key], nested_candidates.index(key)))
            nested_pred = fast_predict(chosen, xtr, ytr, X[held], wt)
            pooled_preds[held] = pooled
            folds.append({
                "test_id": held, "outer_train_ids": train_ids,
                "pca": pca_meta, "prediction": pred.tolist(),
                "per_seed_predictions": [p.tolist() for p in per_seed],
                "per_seed_best_training_loss": per_loss,
                "target": targets[held].tolist(), "primary_mae": err,
                "training_mean_prediction": baseline.tolist(),
                "training_mean_mae": base_err,
                "constant_wt_prediction": constant_wt.tolist(),
                "constant_wt_mae": mae(constant_wt, targets[held]),
                "clr_ridge_prediction": ridge.tolist(),
                "clr_ridge_mae": mae(ridge, targets[held]),
                "clr_gp_prediction": gp.tolist(), "clr_gp_mae": mae(gp, targets[held]),
                "support_distance": support,
                "pooled_k2_prediction": pooled.tolist(),
                "pooled_k2_target": [float(targets[held][0]),
                                     float(targets[held][1:].sum())],
                "nested_selector": {"inner_mae": inner, "selected": chosen,
                                    "prediction": nested_pred.tolist(),
                                    "mae": mae(nested_pred, targets[held])},
            })
        primary_mean = float(np.mean(list(primary_errors.values())))
        baseline_mean = float(np.mean(list(baseline_errors.values())))
        deltas = {mid: baseline_errors[mid] - primary_errors[mid] for mid in ids_subset}
        leave_one_fold = {mid: float(np.mean([deltas[x] for x in ids_subset if x != mid]))
                          for mid in ids_subset}
        sign = exact_sign_test(list(deltas.values()))
        catastrophic = [mid for mid in ids_subset
                        if primary_errors[mid] > 2.0 * baseline_errors[mid] + 1e-15]
        relative = (baseline_mean - primary_mean) / baseline_mean if baseline_mean else None
        success = {
            "mean_paired_improvement_positive": float(np.mean(list(deltas.values()))) > 0,
            "relative_mae_improvement_at_least_15pct": relative is not None and relative >= 0.15,
            "one_sided_exact_p_at_most_0_05": sign["p_one_sided"] <= 0.05,
            "no_catastrophic_fold": not catastrophic,
        }
        return {"folds": folds, "primary_errors": primary_errors,
                "baseline_errors": baseline_errors,
                "support_distances": support_distances,
                "full_preds": full_preds, "pooled_preds": pooled_preds,
                "deltas": deltas, "leave_one_fold_contribution_mean": leave_one_fold,
                "sign": sign, "catastrophic": catastrophic, "relative": relative,
                "primary_mean": primary_mean, "baseline_mean": baseline_mean,
                "success": success, "n_mutants": len(ids_subset)}

    fold_seed_rows = []
    main = run_loo(ids, fold_seed_rows)
    folds = main["folds"]
    primary_errors = main["primary_errors"]
    baseline_errors = main["baseline_errors"]
    support_distances = main["support_distances"]
    full_preds = main["full_preds"]
    pooled_preds = main["pooled_preds"]
    deltas = main["deltas"]
    drop_one = main["leave_one_fold_contribution_mean"]
    sign = main["sign"]
    catastrophic = main["catastrophic"]
    relative = main["relative"]
    primary_mean = main["primary_mean"]
    baseline_mean = main["baseline_mean"]
    success = dict(main["success"])

    # True delete-and-refit robustness gate (v3.0 preregistration intent):
    # removing each mutation in turn, the complete outer LOO is re-run and its
    # mean paired improvement must remain positive.
    delete_one_refit = {}
    for mid in ids:
        subset = [x for x in ids if x != mid]
        sub = run_loo(subset, [], full_diagnostics=False)
        delete_one_refit[mid] = float(np.mean(list(sub["deltas"].values())))
    success["delete_one_refit_all_positive"] = all(
        value > 0 for value in delete_one_refit.values())
    success["all_required"] = all(success.values())

    ilr_pred = np.vstack([ilr(full_preds[mid]) for mid in ids])
    ilr_target = np.vstack([ilr(targets[mid]) for mid in ids])
    shared_full = [abs(full_preds[mid][0] - targets[mid][0]) for mid in ids]
    shared_pooled = [abs(pooled_preds[mid][0] - targets[mid][0]) for mid in ids]
    support_test = exact_spearman_permutation(
        [support_distances[mid] for mid in ids], [primary_errors[mid] for mid in ids])
    nested_mean = float(np.mean([f["nested_selector"]["mae"] for f in folds]))

    # Naive selection (P3): the single candidate with the lowest panel-mean
    # fold MAE, computed on the identical fold outputs the nested selector used.
    candidate_mean_mae = {kind: float(np.mean([f[f"{kind}_mae"] for f in folds]))
                          for kind in nested_candidates}
    naive_selected = min(candidate_mean_mae,
                         key=lambda k: (candidate_mean_mae[k], nested_candidates.index(k)))
    naive_mean = candidate_mean_mae[naive_selected]
    nested_minus_naive = nested_mean - naive_mean

    # Orthonormal contrasts (P2): q1=(2,-1,-1)/sqrt(6), q2=(0,1,-1)/sqrt(2).
    def orthonormal_q(preds):
        e = np.asarray([preds[mid] - targets[mid] for mid in ids], dtype=float)
        q1 = np.abs((2.0 * e[:, 0] - e[:, 1] - e[:, 2]) / math.sqrt(6.0))
        q2 = np.abs((e[:, 1] - e[:, 2]) / math.sqrt(2.0))
        return float(q1.mean()), float(q2.mean())

    q1_mae, q2_mae = orthonormal_q(full_preds)
    pooled_q1_scale = (math.sqrt(6.0) / 2.0) * float(np.mean(shared_pooled))
    full_q1_mae = q1_mae

    result = {
        "status": "DRY RUN / NOT EVIDENCE" if dry_run else "SEALED CUSTODIAN RUN",
        "protocol_version": PROTOCOL_VERSION, "panel_id": public["panel_id"],
        "panel_tier": panel_tier, "n_mutants": len(ids),
        "main_panel_eligible": panel_tier == "primary_eligible",
        "primary_predictor": {
            "checkpoint": CHECKPOINT, "feature": "residue embedding difference",
            "multi_site_aggregation": "sum of mutant-minus-WT residue deltas at all substituted sites",
            "pca": "outer-training residue rows only; d=min(20,n_train-1)",
            "model": {"class": "LowRankCDST", "K": 3, "rank": RANK,
                      "hidden": HIDDEN, "epochs": EPOCHS, "optimizer": "Adam",
                      "lr": LR, "weight_decay": WEIGHT_DECAY,
                      "seeds": N_SEEDS, "seed_scheme": "s*100+outer_test_index"}},
        "input_hashes": {"public_input": sha256_file(public_path),
                         "private_targets": sha256_file(private_path),
                         "esm2_features": sha256_file(feature_path)},
        "aggregate": {"primary_mae": primary_mean,
                      "training_mean_mae": baseline_mean,
                      "mean_paired_improvement": float(np.mean(list(deltas.values()))),
                      "relative_mae_improvement": relative,
                      "exact_test": sign, "catastrophic_folds": catastrophic,
                      "leave_one_fold_contribution_mean": drop_one,
                      "delete_one_refit_mean_improvement": delete_one_refit,
                      "success_gates": success},
        "diagnostics": {
            "support_error": support_test,
            "collision": collision_diagnostic(ids, vectors, targets, full_preds),
            "contrasts": {"coarse_ilr_mae": float(np.abs(ilr_pred[:, 0] - ilr_target[:, 0]).mean()),
                          "fine_ilr_mae": float(np.abs(ilr_pred[:, 1] - ilr_target[:, 1]).mean()),
                          "orthonormal_q": {"q1_mae": q1_mae, "q2_mae": q2_mae}},
            "pooling": {"full_k3_shared_contrast_mae": float(np.mean(shared_full)),
                        "pooled_k2_shared_contrast_mae": float(np.mean(shared_pooled)),
                        "full_k3_q1_mae": full_q1_mae,
                        "pooled_k2_q1_scale_mae": pooled_q1_scale},
            "selection": {"candidates": nested_candidates,
                          "candidate_mean_fold_mae": {k: float(v) for k, v in
                                                      candidate_mean_mae.items()},
                          "naive_selected": naive_selected,
                          "naive_mae": naive_mean,
                          "nested_mae": nested_mean,
                          "nested_minus_naive": nested_minus_naive},
            "fixed_vs_nested": {"fixed_primary_mae": primary_mean,
                                "fully_nested_selector_mae": nested_mean,
                                "nested_minus_fixed": nested_mean - primary_mean}},
        "single_mutant_only_sensitivity": None,
        "folds": folds,
        "runtime": {"seconds": time.time() - started, "python": sys.version,
                    "platform": platform.platform(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                    "torch": torch.__version__, "device": "cpu",
                    # Actual post-call state, not an assertion: if enabling
                    # deterministic algorithms failed on the custodian's
                    # machine the evidence must say so.
                    "deterministic_algorithms": deterministic_ok and
                    torch.are_deterministic_algorithms_enabled()},
    }
    single_ids = [m["mutation_id"] for m in public["mutations"]
                  if len(m["substitutions"]) == 1]
    if len(single_ids) >= 2:
        result["single_mutant_only_sensitivity"] = {
            "n": len(single_ids),
            "primary_mae": float(np.mean([primary_errors[mid] for mid in single_ids])),
            "training_mean_mae": float(np.mean([baseline_errors[mid] for mid in single_ids])),
            "mean_paired_improvement": float(np.mean([deltas[mid] for mid in single_ids]))}

    # ---- v3.1 diagnostic scoring (frozen definitions; see the amendment) ----
    coll = result["diagnostics"]["collision"]
    null = coll["label_permutation_null"]
    if null["status"] == "not_estimable_without_both_conflict_and_nonconflict_members":
        p1 = {"status": "not_estimable", "hit": None}
    else:
        pval = null["p_one_sided_greater"]
        p1 = {"status": "computed", "p": pval,
              "hit": bool(pval is None or pval >= DIAG_NULL_ALPHA)}
    p2_hit = bool(q2_mae > q1_mae and pooled_q1_scale >= full_q1_mae)
    p3_hit = bool(DIAG_P3_BAND[0] <= nested_minus_naive <= DIAG_P3_BAND[1])
    rho = support_test["rho"]
    p4_hit = bool(rho is not None and rho > 0)
    hits = {"P1": p1["hit"], "P2": p2_hit, "P3": p3_hit, "P4": p4_hit}
    if p1["hit"] is None:
        denominator, passed = 3, int(sum(bool(v) for k, v in hits.items() if k != "P1"))
        framework_validated = passed == 3
    else:
        denominator, passed = 4, int(sum(bool(v) for v in hits.values()))
        framework_validated = passed >= 3
    result["v3_1_diagnostic_scoring"] = {
        "layer": DIAGNOSTIC_LAYER,
        "definitions_sha256": hashlib.sha256(
            json.dumps(DIAGNOSTIC_DEFINITIONS, sort_keys=True).encode("utf-8")).hexdigest(),
        "predictions": {
            "P1": {"description": DIAGNOSTIC_DEFINITIONS["p1"], "value": p1, "hit": p1["hit"]},
            "P2": {"description": DIAGNOSTIC_DEFINITIONS["p2"],
                   "value": {"q1_mae": q1_mae, "q2_mae": q2_mae,
                             "pooled_q1_scale_mae": pooled_q1_scale},
                   "hit": p2_hit},
            "P3": {"description": DIAGNOSTIC_DEFINITIONS["p3"],
                   "value": {"nested_minus_naive": nested_minus_naive,
                             "band": list(DIAG_P3_BAND)},
                   "hit": p3_hit},
            "P4": {"description": DIAGNOSTIC_DEFINITIONS["p4"],
                   "value": {"rho": rho}, "hit": p4_hit}},
        "score": passed, "denominator": denominator,
        "framework_validated": framework_validated,
        "independent_of_primary_gate": True,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "custodian_result_v3.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    with (output_dir / "fold_seed_predictions_v3.csv").open("w", newline="",
                                                              encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fold_seed_rows[0]))
        writer.writeheader(); writer.writerows(fold_seed_rows)
    log = {
        "status": result["status"], "protocol_version": PROTOCOL_VERSION,
        "diagnostic_layer": DIAGNOSTIC_LAYER,
        "diagnostic_definitions_sha256": result["v3_1_diagnostic_scoring"][
            "definitions_sha256"],
        "result_sha256": sha256_file(result_path),
        "fold_csv_sha256": sha256_file(output_dir / "fold_seed_predictions_v3.csv"),
        "result_schema_hash": canonical_hash({"top_level_keys": sorted(result)})}
    (output_dir / "run_log_v3.json").write_text(
        json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(args.public, args.private, args.features, args.output_dir,
                 dry_run=args.dry_run)
    print(json.dumps({"status": result["status"],
                      "primary_mae": result["aggregate"]["primary_mae"],
                      "success": result["aggregate"]["success_gates"]["all_required"],
                      "diagnostic_score": result["v3_1_diagnostic_scoring"]["score"],
                      "framework_validated":
                          result["v3_1_diagnostic_scoring"]["framework_validated"]},
                     indent=2))


if __name__ == "__main__":
    main()
