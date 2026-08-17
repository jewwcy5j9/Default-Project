"""P0-2: Src primary-panel versus L410A-substitution sensitivity.

Runs the full Src analysis battery on the primary panel and a hybrid panel:
  PROBE  : canonical probe-level pops (k3_data.SRC_K3, Fig S5 Met305 probe)
  HYBRID : L410A swapped to its Table S2 global CPMG-fit value [0.96, 0.03, 0.01]
           (non_active 0.04 vs probe 0.27); all other core mutants unchanged
           (among core mutants only L410A has a true Table S2 global-fit value
           in cui2025_src_kinase.json; V332I's file row is Fig-S5-sourced and
           A311I's Table S2 row equals the probe row).

Battery per protocol (plan P0-2 items):
  1. training mean row (K=3)
  2. fixed LOO rows: candidates {pos, ext, llr_pos, llr_only, pca20} x
     models {CLR-Ridge, CLR-GP, LowRankCDST} at K=3
     (MLP raw features = k3_benchmark convention; CLR models with fold-local
     StandardScaler = p2_k3_eval_v2 convention)
  3. pooled K=2 rows: same grid on the [Active, E1+E2] collapse
     (metrics: non_active MAE and u1 contrast MAE)
  4. u1 / u2 contrast errors (u1 = A-(E1+E2), u2 = E1-E2) for every K=3 row
  5. F405A per-state errors for key rows
  6. collision pair direction distances (t6 pairs) + per-mutant failure
     pattern stability between protocols

Primary label protocol = PROBE (provenance: Fig S5 Met305 probe is the only
measurement available for all 8 core mutants on one probe; global-fit values
are not available for the full panel). This is a provenance decision, not an
MAE decision (plan P0-2).

Verification against frozen artifacts:
  - probe fixed rows vs P0-1 measured values (pos 0.3213, ext 0.3045,
    llr_pos 0.3471, llr_only 0.3626, pca20 0.3020813, train mean 0.2911)
  - L410A-substitution rows vs src_k3_l410a_sensitivity.json
    (MLP pos 0.3411436539990367, MLP Extended 0.3711850950533213,
     L410A_global_trainmean 0.31857142857142856)

Outputs:
  results/p2_k3_src_label_sensitivity.json
  results/p2_k3_src_label_sensitivity_manifest.json

Usage:
  python p2_k3_src_label_sensitivity.py
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*optimal value found.*")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn.functional as F

from k3_data import SRC_K3, SRC_K3_WT_POP
from src.models.low_rank_cdst import LowRankCDST
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from p2_k3_nested_pca import (CANDIDATES, NEPOCHS, PCA_D,
                              build_X, clr, compute_embeddings,
                              direction_report, fixed_loo, load_features,
                              sha256_file, training_mean_row, u1_u2_contrast,
                              _device, _norm_simplex)
from gp_protocols import PRIMARY_GP_PROTOCOL, make_primary_gp
from src.data.src_k3_labels import (
    CANONICAL_SRC_K3_PATH,
    SRC_K3_L410A_SUBSTITUTION_PROTOCOL_ID,
    SRC_K3_PRIMARY_PROTOCOL_ID,
    build_src_k3_panel,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
K = 3
EPS = 1e-6

SRC_CORE = {m: SRC_K3[m] for m in SRC_K3 if m != "SrcKD-WT"}
PRIMARY_PANEL = build_src_k3_panel(SRC_K3_PRIMARY_PROTOCOL_ID)
L410A_SUBSTITUTION_PANEL = build_src_k3_panel(
    SRC_K3_L410A_SUBSTITUTION_PROTOCOL_ID)
PANELS = {
    "primary_probe": PRIMARY_PANEL,
    "l410a_global_fit_substitution": L410A_SUBSTITUTION_PANEL,
}
PROTOCOLS = {
    key: {m: np.array(panel.targets[m], float) for m in SRC_CORE}
    for key, panel in PANELS.items()
}

MODELS = ["CLR-Ridge", "CLR-GP", "LowRankCDST"]
FROZEN_PROBE = {"pos": 0.3213, "ext": 0.3045, "llr_pos": 0.3471,
                "llr_only": 0.3626, "pca20": 0.3020813,
                "training_mean": 0.2911}
FROZEN_SUBSTITUTION = {"pos": 0.3411436539990367,
                       "ext": 0.3711850950533213,
                       "training_mean": 0.31857142857142856}
COLLISION_PAIRS = [
    ["SrcKD-L410A", "SrcKD-L325A"], ["SrcKD-L410A", "SrcKD-V331A"],
    ["SrcKD-L410A", "SrcKD-F405A"], ["SrcKD-L270F_V332I", "SrcKD-L325A"],
    ["SrcKD-L270F_V332I", "SrcKD-V331A"], ["SrcKD-L325A", "SrcKD-V331A"],
    ["SrcKD-L325A", "SrcKD-F405A"], ["SrcKD-V331A", "SrcKD-F405A"],
]
KEY_ROWS_F405A = ["pos::LowRankCDST", "ext::LowRankCDST", "llr_pos::LowRankCDST",
                  "pca20::LowRankCDST", "pos::CLR-GP", "pca20::CLR-GP",
                  "llr_only::LowRankCDST"]


# ---------------------------------------------------------------------------
# K=2 pooled battery
# ---------------------------------------------------------------------------

def targets_k2(targets3):
    """Collapse [A, E1, E2] -> [A, 1-A]."""
    return {m: np.array([t[0], 1.0 - t[0]]) for m, t in targets3.items()}


def train_mlp_seed_k2(w_tr, c_tr, y_tr, w_te, c_te, d, seed_base, s, dev):
    torch.manual_seed(s * 100 + seed_base)
    np.random.seed(s * 100 + seed_base)
    model = LowRankCDST(K=2, intervention_dim=d, rank=2, hidden_dim=32).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    w_t = torch.FloatTensor(w_tr).to(dev)
    c_t = torch.FloatTensor(c_tr).to(dev)
    t_t = torch.FloatTensor(y_tr).to(dev)
    best_loss, best_state = float("inf"), None
    for _ in range(NEPOCHS):
        opt.zero_grad()
        loss = F.mse_loss(model(w_t, c_t), t_t)
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = model(torch.FloatTensor(np.atleast_2d(w_te)).to(dev),
                  torch.FloatTensor(np.atleast_2d(c_te)).to(dev)).cpu().numpy()[0]
    return p


def predict_fold_k2(cand, mname, system, names, f, seq_len, delta_vecs,
                    delta_rows, tr_i, te_i, targets3, seed_base,
                    use_scaler=False, n_seeds=5):
    """Fold-local (candidate, model) K=2 pooled prediction."""
    Xtr, meta = build_X(cand, system, names, f, seq_len, delta_vecs,
                        delta_rows, tr_i, tr_i)
    Xte, _ = build_X(cand, system, names, f, seq_len, delta_vecs,
                     delta_rows, te_i, tr_i)
    d = Xtr.shape[1]
    t2 = targets_k2(targets3)
    y_tr = np.array([t2[names[i]] for i in tr_i], float)
    w2 = np.array([SRC_K3_WT_POP[0], 1.0 - SRC_K3_WT_POP[0]], float)
    w_tr = np.tile(w2, (len(tr_i), 1))
    w_te = np.atleast_2d(w2)
    dev = _device()
    if use_scaler:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        meta = dict(meta)
        meta["scaler_mean"] = scaler.mean_.tolist()
        meta["scaler_scale"] = scaler.scale_.tolist()
    if mname == "LowRankCDST":
        preds = [_norm_simplex(train_mlp_seed_k2(w_tr, Xtr, y_tr, w_te, Xte, d,
                                                 seed_base, s, dev))
                 for s in range(n_seeds)]
        return _norm_simplex(np.mean(preds, axis=0)), preds, meta
    z_tr = clr(y_tr)[:, :1] * np.sqrt(2.0)
    if mname == "CLR-Ridge":
        m = Ridge(alpha=1.0)
        m.fit(Xtr, z_tr.ravel())
        z_te = m.predict(Xte).reshape(-1, 1) / np.sqrt(2.0)
    elif mname == "CLR-GP":
        m = make_primary_gp()
        m.fit(Xtr, z_tr.ravel())
        z_te = m.predict(Xte).reshape(-1, 1) / np.sqrt(2.0)
    else:
        raise KeyError(mname)
    p = np.empty((z_te.shape[0], 2))
    p[:, 0] = 1.0 / (1.0 + np.exp(-2.0 * z_te[:, 0]))
    p[:, 1] = 1.0 - p[:, 0]
    return p[0], [p[0].copy() for _ in range(n_seeds)], meta


def fixed_loo_k2(system, names, f, seq_len, delta_vecs, delta_rows, targets3,
                 cand, mname, use_scaler=False, n_seeds=5):
    preds, folds = {}, {}
    for i, held in enumerate(names):
        tr_i = [j for j in range(len(names)) if j != i]
        p, per_seed, meta = predict_fold_k2(
            cand, mname, system, names, f, seq_len, delta_vecs, delta_rows,
            tr_i, [i], targets3, seed_base=i, use_scaler=use_scaler,
            n_seeds=n_seeds)
        preds[held] = p
        folds[held] = {"pred": np.round(p, 9).tolist(),
                       "target": np.round(targets_k2(targets3)[held], 9).tolist(),
                       "meta": meta}
    t2 = targets_k2(targets3)
    na_errs = {m: float(abs(preds[m][1] - t2[m][1])) for m in preds}
    u1_errs = {m: float(abs((2.0 * preds[m][0] - 1.0)
                            - (2.0 * t2[m][0] - 1.0))) for m in preds}
    return {"preds": preds, "non_active_errors": na_errs,
            "non_active_mae": float(np.mean(list(na_errs.values()))),
            "u1_contrast_errors": u1_errs,
            "u1_contrast_mae": float(np.mean(list(u1_errs.values()))),
            "folds": folds}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def per_state_errors(preds, targets3, names):
    return {m: [float(abs(preds[m][k] - targets3[m][k])) for k in range(3)]
            for m in names}


def collision_direction_distances(targets3, pairs, wt):
    d = {m: np.asarray(targets3[m]) - np.asarray(wt) for m in targets3}
    out = {}
    for a, b in pairs:
        out[f"{a}|{b}"] = float(np.abs(d[a] - d[b]).sum())
    return out


def failure_pattern_corr(err_a, err_b, names):
    ea = np.array([err_a[m] for m in names])
    eb = np.array([err_b[m] for m in names])
    return float(np.corrcoef(ea, eb)[0, 1]) if np.std(eb) > 0 else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_battery(system, names, f, seq_len, delta_vecs, delta_rows, targets3,
                args):
    tm = training_mean_row(names, targets3)
    fixed, k2 = {}, {}
    for cand in CANDIDATES:
        for mname in MODELS:
            use_scl = mname != "LowRankCDST"
            key = f"{cand}::{mname}"
            fixed[key] = fixed_loo(system, names, f, seq_len, delta_vecs,
                                   delta_rows, targets3, cand, mname,
                                   use_scaler=use_scl, n_seeds=args.seeds)
            k2[key] = fixed_loo_k2(system, names, f, seq_len, delta_vecs,
                                   delta_rows, targets3, cand, mname,
                                   use_scaler=use_scl, n_seeds=args.seeds)
    return tm, fixed, k2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--version", default="1.0.0")
    args = ap.parse_args()

    t0 = time.time()
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    print(f"P0-2 Src label sensitivity | seeds={args.seeds} | "
          f"device={_device()}", flush=True)

    system = "src"
    names, f, targets3, seq_len = load_features(system)
    vecs, rows = compute_embeddings(system, use_cache=True)
    delta_vecs, delta_rows = vecs, rows

    results = {"experiment": "P0-2 Src L410A global-fit substitution "
                              "(iclr_improvement_plan_v2.md)",
               "version": args.version, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
               "system": system, "n_mutants": len(names),
               "protocols": {
                   key: {"protocol_id": panel.protocol_id,
                         "protocol_kind": panel.protocol_kind,
                         "wt_record_id": panel.wt_record_id,
                         "target_record_ids": panel.target_record_ids,
                         "substitutions": panel.substitutions,
                         "canonical_sha256": panel.canonical_sha256}
                   for key, panel in PANELS.items()},
               "protocol_choice": {
                   "primary_label_choice": "probe (provenance: the only "
                                            "measurement available for all 8 "
                                            "core mutants on one probe; not an "
                                            "MAE choice)"},
               "battery": {"candidates": CANDIDATES, "models": MODELS,
                            "gp_protocol": PRIMARY_GP_PROTOCOL,
                           "scaling": "MLP raw (k3_benchmark convention); "
                                      "CLR-Ridge/CLR-GP fold-local "
                                      "StandardScaler (p2 convention)",
                           "pooled_k2": "collapse [A, E1+E2]; metrics: "
                                        "non_active MAE + u1 contrast MAE",
                           "seeds": args.seeds, "epochs": NEPOCHS},
               "systems": {}}

    for proto, targets in PROTOCOLS.items():
        tm, fixed, k2 = run_battery(system, names, f, seq_len, delta_vecs,
                                    delta_rows, targets, args)
        fixed_k3 = {}
        for key, r in fixed.items():
            fixed_k3[key] = {
                "mae": r["mae"], "direction": r["direction"],
                "u1_u2_contrast": u1_u2_contrast(r["preds"], targets),
                "errors": {m: round(e, 6) for m, e in r["errors"].items()},
                "per_mutant_pred": {m: np.round(r["preds"][m], 6).tolist()
                                    for m in names}}
        blk = {"training_mean": {"mae": tm["mae"],
                                 "errors": {k: round(v, 6)
                                            for k, v in tm["errors"].items()}},
               "fixed_k3": fixed_k3,
               "pooled_k2": {k: {"non_active_mae": v["non_active_mae"],
                                 "u1_contrast_mae": v["u1_contrast_mae"],
                                 "non_active_errors": {
                                     m: round(e, 6) for m, e
                                     in v["non_active_errors"].items()},
                                 "u1_contrast_errors": {
                                     m: round(e, 6) for m, e
                                     in v["u1_contrast_errors"].items()}}
                             for k, v in k2.items()},
               "f405a_per_state": {key: per_state_errors(
                   fixed[key]["preds"], targets, names)["SrcKD-F405A"]
                   for key in KEY_ROWS_F405A},
               "collision_distances": collision_direction_distances(
                   targets, COLLISION_PAIRS, SRC_K3_WT_POP)}
        results["systems"][proto] = blk
        print(f"[{proto}] tm={tm['mae']:.4f} | pos/MLP "
              f"{fixed['pos::LowRankCDST']['mae']:.4f} | ext/MLP "
              f"{fixed['ext::LowRankCDST']['mae']:.4f} | pca20/MLP "
              f"{fixed['pca20::LowRankCDST']['mae']:.4f} | pca20 K2 "
              f"{k2['pca20::LowRankCDST']['non_active_mae']:.4f}", flush=True)

    results["verification"] = {
        "primary_probe_vs_p0_1": {k: {"frozen": v,
                              "measured": results["systems"]["primary_probe"]
                              ["fixed_k3"][f"{k}::LowRankCDST"]["mae"]}
                          for k, v in FROZEN_PROBE.items()
                          if k != "training_mean"},
        "primary_probe_training_mean": {"frozen": FROZEN_PROBE["training_mean"],
                                "measured": results["systems"]["primary_probe"]
                                ["training_mean"]["mae"]},
        "l410a_substitution_vs_sensitivity_file": {k: {"frozen": v,
                                           "measured": results["systems"]["l410a_global_fit_substitution"]
                                           ["fixed_k3"][f"{k}::LowRankCDST"]["mae"]}
                                       for k, v in FROZEN_SUBSTITUTION.items()
                                       if k != "training_mean"},
        "l410a_substitution_training_mean": {"frozen": FROZEN_SUBSTITUTION["training_mean"],
                                 "measured": results["systems"]["l410a_global_fit_substitution"]
                                 ["training_mean"]["mae"]},
        "all_within_tol_1e-3": None}
    diffs = []
    for blk in (results["verification"]["primary_probe_vs_p0_1"],
                results["verification"]["l410a_substitution_vs_sensitivity_file"]):
        for k, d in blk.items():
            diffs.append(abs(d["frozen"] - d["measured"]))
    for blk in (results["verification"]["primary_probe_training_mean"],
                results["verification"]["l410a_substitution_training_mean"]):
        diffs.append(abs(blk["frozen"] - blk["measured"]))
    results["verification"]["all_within_tol_1e-3"] = all(
        d <= 1e-3 for d in diffs)
    results["verification"]["max_abs_diff"] = max(diffs)
    if not results["verification"]["all_within_tol_1e-3"]:
        raise SystemExit(
            "verification failed: max abs diff "
            f"{results['verification']['max_abs_diff']:.6f} exceeds 1e-3 "
            "against the frozen values; refusing to write a passing-looking "
            "artifact")

    out = RESULTS / "p2_k3_src_label_sensitivity.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    manifest = {"script": "p2_k3_src_label_sensitivity.py",
                "version": args.version, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "script_sha256": sha256_file(__file__),
                "params": {"seeds": list(range(args.seeds)), "epochs": NEPOCHS,
                            "pca_dim": PCA_D, "candidates": CANDIDATES,
                            "models": MODELS,
                            "gp_protocol": PRIMARY_GP_PROTOCOL,
                            "label_protocol_ids": [panel.protocol_id
                                                   for panel in PANELS.values()]},
                "inputs": {
                    "src_k3_canonical.csv": sha256_file(CANONICAL_SRC_K3_PATH),
                    "p2_k3_nested_pca.py": sha256_file(HERE / "p2_k3_nested_pca.py"),
                    "k3_data.py": sha256_file(HERE / "k3_data.py"),
                    "esm2_encoding.py": sha256_file(HERE / "esm2_encoding.py"),
                    "src_k3_l410a_sensitivity.json": sha256_file(RESULTS / "src_k3_l410a_sensitivity.json")},
                "outputs": ["p2_k3_src_label_sensitivity.json"]}
    (RESULTS / "p2_k3_src_label_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
