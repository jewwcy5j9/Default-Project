"""P2-K3: nested selector evaluation at K=3 with frozen ESM-2 features.

Rebuild of the invalidated p2_eval.py (see p2_protocol_audit.md). Fixes:
  1. K=2 -> K=3: predicts full 3-state population vectors; primary metric =
     per-state MAE averaged over mutants (matches k3_benchmark/K3 pipeline).
  2. Inner-fold index misalignment: all row indices are immutable mutant
     indices (names.index(...)); asserts combo rows == label rows == held-out.
  3. u2 group labels removed; u2 reported as raw E1-E2 contrast MAE (Src).
  4. Raw LLR enters feature matrix; scaling is fold-local (StandardScaler on
     training rows), no full-system max-normalization leak.
  5. Seeds: seed = s*100 + OUTER holdout immutable index; inner folds use the
     outer index too (single sizing rule, no tr.index offset).
  6. Marker control M1 (pos+llr+site markers, direct/experiment-free) runs the
     SAME nested protocol; "no-marker beats control" gate compares nested MAE.
  7. LSO groups frozen: Abl1 F382_family / 290_301; Src N_lobe / C_lobe.
  8. L410A alt-label sensitivity (global [0.96,0.03,0.01]) re-runs nested on
     Src; verdict-reversal check is a gate.

Candidates (frozen): C1[llr] C2[llr,pos] C3[llr,tok] C4[rand4]
C5[llr,tok,rand2] C6[rand8]; models = CLR-Ridge / CLR-GP / SimpleCDST(K=3) /
LowRankCDST(K=3, rank=2, hidden=32), 5 seeds, best-state, 800 epochs.
Inputs: results/p2_llr_features.json + p2_site_deltas.npz (frozen manifests).
Usage: python p2_k3_eval.py [--systems abl1,src] [--skip-abl1] [--skip-src]
       [--skip-alt] [--seeds 5]
Output: results/p2_k3_results.json + results/p2_k3_report.md
"""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from k3_data import ABL1_K3, SRC_K3, ABL1_K3_WT_POP, SRC_K3_WT_POP
from src.models.low_rank_cdst import LowRankCDST
from encoding_ablation_control import SimpleCDST
from esm2_encoding import ABL1_KD, SRC_FULL
from gp_protocols import PRIMARY_GP_PROTOCOL, make_primary_gp
from src.data.src_k3_labels import (
    SRC_K3_L410A_SUBSTITUTION_PROTOCOL_ID,
    SRC_K3_PRIMARY_PROTOCOL_ID,
    build_src_k3_panel,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
EPS = 1e-6
K = 3
RAND_SEED = 0
RAND_DIMS = [2, 4, 8]
TIE_DELTA = 0.05
NEPOCHS = 800

ABL1_CORE = {m: ABL1_K3[m] for m in ("M290L", "L301I", "M290L_L301I",
                                     "F382L", "F382Y", "F382V")}
SRC_CORE = {m: SRC_K3[m] for m in SRC_K3 if m != "SrcKD-WT"}
SEQ_LEN = {"abl1": len(ABL1_KD), "src": len(SRC_FULL)}
WT_POP = {"abl1": ABL1_K3_WT_POP, "src": SRC_K3_WT_POP}

LSO_GROUPS = {
    "abl1": {"F382_family": ["F382L", "F382Y", "F382V"],
             "290_301": ["M290L", "L301I", "M290L_L301I"]},
    "src": {"N_lobe": ["SrcKD-L270F_V332I", "SrcKD-A311I", "SrcKD-L325A",
                       "SrcKD-V331A", "SrcKD-V332I"],
            "C_lobe": ["SrcKD-V380A", "SrcKD-F405A", "SrcKD-L410A"]},
}
COMBO_DEFS = [("C1_llr1", ["llr"]), ("C2_llr_pos", ["llr", "pos"]),
              ("C3_llr_tok", ["llr", "tok"]), ("C4_rand4", ["rand4"]),
              ("C5_llr_tok_rand2", ["llr", "tok", "rand2"]),
              ("C6_rand8", ["rand8"])]
MODEL_NAMES = ["CLR-Ridge", "CLR-GP", "SimpleCDST", "LowRankCDST"]


def clr(y):
    y = np.clip(np.asarray(y, float), EPS, 1.0)
    y = y / y.sum(axis=-1, keepdims=True)
    return np.log(y) - np.log(y).mean(axis=-1, keepdims=True)


def inv_clr(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    v = e / e.sum(axis=-1, keepdims=True)
    return v[0] if v.ndim == 2 else v


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_features(system):
    """Returns (names, feat dict, targets dict of 3-vectors).
    LLR kept RAW (fold-local scaler handles scaling)."""
    feat = json.loads((RESULTS / "p2_llr_features.json").read_text(encoding="utf-8"))
    llr_all = feat["llr"]
    deltas = np.load(RESULTS / "p2_site_deltas.npz")
    rng = np.random.default_rng(RAND_SEED)
    P = {}
    for d in RAND_DIMS:
        M = rng.normal(0.0, 1.0, (d, deltas["abl1::M290L"].shape[0]))
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        P[d] = M
    core = ABL1_CORE if system == "abl1" else SRC_CORE
    names = list(core.keys())
    seq_len = SEQ_LEN[system]
    f = {m: {} for m in names}
    for m in names:
        f[m]["llr"] = float(llr_all[system][m]["llr"])
        f[m]["pos"] = core[m]["pos"] / seq_len
        f[m]["tok"] = llr_all[system][m]["tok_dir_unit"]
        dvec = deltas[f"{system}::{m}"]
        for d in RAND_DIMS:
            f[m][f"rand{d}"] = P[d] @ dvec
    targets = {m: np.array(core[m]["pop"], float) for m in names}
    for m in names:
        assert targets[m].shape == (K,), system
    return names, f, targets


def combo_matrix(f, names, combo):
    return np.array([np.hstack([np.atleast_1d(f[m][k]) for k in combo])
                     for m in names], float)


def marker_matrix(f, names, system):
    marks = [290, 301, 382] if system == "abl1" else [311, 332, 380]
    core = ABL1_CORE if system == "abl1" else SRC_CORE
    seq_len = SEQ_LEN[system]
    X = []
    for m in names:
        r = [core[m]["pos"] / seq_len, f[m]["llr"]]
        r += [1.0 if core[m]["pos"] == p else 0.0 for p in marks]
        X.append(r)
    return np.array(X, float)


def train_lowrank(w_wt_tr, c_tr, y_tr, w_wt_te, c_te, d, seed_base, n_seeds=5):
    dev = _device()
    preds = []
    for s in range(n_seeds):
        torch.manual_seed(s * 100 + seed_base)
        np.random.seed(s * 100 + seed_base)
        model = LowRankCDST(K=K, intervention_dim=d, rank=2, hidden_dim=32).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
        w_t = torch.FloatTensor(w_wt_tr).to(dev)
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
            p = model(torch.FloatTensor(np.atleast_2d(w_wt_te)).to(dev),
                      torch.FloatTensor(np.atleast_2d(c_te)).to(dev)).cpu().numpy()[0]
        preds.append(p)
    return np.mean(preds, axis=0)


def train_simple(w_wt_tr, c_tr, y_tr, w_wt_te, c_te, d, seed_base, n_seeds=5):
    dev = _device()
    preds = []
    for s in range(n_seeds):
        torch.manual_seed(s * 100 + seed_base)
        np.random.seed(s * 100 + seed_base)
        model = SimpleCDST(K=K, intervention_dim=d).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
        w_t = torch.FloatTensor(w_wt_tr).to(dev)
        c_t = torch.FloatTensor(c_tr).to(dev)
        t_t = torch.FloatTensor(y_tr).to(dev)
        best_loss, best_state = float("inf"), None
        for _ in range(NEPOCHS):
            opt.zero_grad()
            p_log = model(w_t, c_t)
            loss = F.mse_loss(torch.exp(p_log), t_t)
            loss.backward()
            opt.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            p = torch.exp(model(torch.FloatTensor(np.atleast_2d(w_wt_te)).to(dev),
                                torch.FloatTensor(np.atleast_2d(c_te)).to(dev))
                          ).cpu().numpy()[0]
        preds.append(p)
    return np.mean(preds, axis=0)


def predict_k3(cname, mname, X_tr, y_tr, X_te, w_wt_tr, w_wt_te, seed_base,
               n_seeds=5):
    d = X_tr.shape[1]
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_tr)
    Xte_s = scaler.transform(X_te)
    if mname == "CLR-Ridge":
        m = Ridge(alpha=1.0)
        m.fit(Xtr_s, clr(y_tr))
        return inv_clr(m.predict(Xte_s))
    if mname == "CLR-GP":
        m = make_primary_gp()
        m.fit(Xtr_s, clr(y_tr))
        return inv_clr(m.predict(Xte_s))
    if mname == "SimpleCDST":
        return train_simple(w_wt_tr, Xtr_s, y_tr, w_wt_te, Xte_s, d,
                            seed_base, n_seeds)
    return train_lowrank(w_wt_tr, Xtr_s, y_tr, w_wt_te, Xte_s, d,
                         seed_base, n_seeds)


# ---------------------------------------------------------------------------
# Protocol: K=3 metrics, fixed-combo LOO, nested LOO, LSO, alt-label
# ---------------------------------------------------------------------------
def per_state_mae(preds, targets):
    return {m: float(np.abs(preds[m] - targets[m]).mean()) for m in preds}


def mean_mae(errs):
    return float(np.mean(list(errs.values())))


def wt_matrix(p_wt, n):
    return np.tile(np.asarray(p_wt, float), (n, 1))


def run_fixed_loo(system, names, f, targets, p_wt, combo_defs, n_seeds=5):
    """Per-(combo, model) fixed LOO at K=3; all indices immutable."""
    X = {c: combo_matrix(f, names, cmb) for c, cmb in combo_defs}
    tgt = {m: targets[m] for m in names}
    out = {}
    hi = {m: i for i, m in enumerate(names)}
    for cname, cmb in combo_defs:
        for mname in MODEL_NAMES:
            preds = {}
            for m in names:
                i = hi[m]
                tr = [j for j in range(len(names)) if j != i]
                p = predict_k3(cname, mname,
                               X[cname][tr], np.array([targets[names[j]] for j in tr], float),
                               X[cname][[i]],
                               wt_matrix(p_wt, len(tr)),
                               wt_matrix(p_wt, 1),
                               seed_base=i, n_seeds=n_seeds)
                preds[m] = p
                assert preds[m].shape == (K,)
            e = per_state_mae(preds, tgt)
            out[f"{cname}::{mname}"] = {"mae": mean_mae(e),
                                        "errors": {k: round(v, 6) for k, v in e.items()}}
    return out


def run_nested(system, names, f, targets, p_wt, combo_defs, control=False,
               n_seeds=3):
    """Nested LOO: inner LOO over training fold selects (combo, model) by
    inner MAE (ties -> lower combo index, then lower model index); outer fold
    refits the selected pair. All indices immutable; seeds = s*100 + outer
    holdout immutable index."""
    n = len(names)
    tgt = [targets[m] for m in names]

    def X_of(cmb, cname):
        return (marker_matrix(f, names, system) if cname == "M1_marker"
                else combo_matrix(f, names, cmb))

    out = {"folds": {}, "preds": {}, "nested_mae": None, "direction": None,
           "per_fold_selected": {}}
    for i, held in enumerate(names):
        tr = [j for j in range(n) if j != i]
        combos = [("M1_marker", None)] if (control or n == 1) else combo_defs
        inner_scores = {}
        for j in tr:
            tr2 = [k for k in tr if k != j]
            for cix, (cname, cmb) in enumerate(combos):
                Xtr2 = X_of(cmb, cname)[tr2]
                for mix, mname in enumerate(MODEL_NAMES):
                    p = predict_k3(cname, mname,
                                   Xtr2, np.array([tgt[j] for j in tr2], float),
                                   X_of(cmb, cname)[[j]],
                                   wt_matrix(p_wt, len(tr2)), wt_matrix(p_wt, 1),
                                   seed_base=i, n_seeds=n_seeds)
                    inner_scores.setdefault((cix, mix), []).append(
                        float(np.abs(p - tgt[j]).mean()))
        best = min(inner_scores, key=lambda km: (np.mean(inner_scores[km]), km[0], km[1]))
        cix, mix = best
        cname, mname = combos[cix][0], MODEL_NAMES[mix]
        Xtr = X_of(combos[cix][1], cname)[tr]
        p = predict_k3(cname, mname,
                       Xtr, np.array([tgt[j] for j in tr], float),
                       X_of(combos[cix][1], cname)[[i]],
                       wt_matrix(p_wt, len(tr)), wt_matrix(p_wt, 1),
                       seed_base=i, n_seeds=n_seeds)
        out["folds"][held] = {"combo": cname, "model": mname,
                              "pred": np.round(p, 6).tolist(),
                              "target": np.round(tgt[i], 6).tolist(),
                              "inner_mae": float(np.mean(inner_scores[best]))}
        out["preds"][held] = np.round(p, 6).tolist()
        out["per_fold_selected"][held] = f"{cname}/{mname}"
    errs = per_state_mae(out["preds"], {m: targets[m] for m in names})
    out["nested_mae"] = mean_mae(errs)
    out["errors"] = {k: round(v, 6) for k, v in errs.items()}
    out["direction"] = direction_report(out["preds"], targets, p_wt)
    return out


def direction_report(preds, targets, p_wt):
    """Direction agreement on the ACTIVE state (u1=2*p_active-1 per contrast
    definition); mutants whose active change is within TIE_DELTA are excluded.
    The 3-state sum is identically 1, so a full-vector diff is meaningless."""
    k, tot = 0, 0
    pw = float(np.asarray(p_wt, float)[0])
    for m in preds:
        td = float(targets[m][0]) - pw
        if abs(td) < TIE_DELTA:
            continue
        tot += 1
        if (float(np.asarray(preds[m])[0]) - pw) * td > 0:
            k += 1
    return f"{k}/{tot}"


def leave_site_out(system, names, f, targets, p_wt, combo_defs, n_seeds=5):
    """Train on mutants not touching the group's sites (CLR-Ridge fixed, K=3);
    report per-group MAE. Frozen groups per plan."""
    X = {c: combo_matrix(f, names, cmb) for c, cmb in combo_defs}
    groups = LSO_GROUPS[system]
    out = {}
    for cname, cmb in combo_defs:
        for gname, members in groups.items():
            tr = [m for m in names if m not in members]
            te = [m for m in names if m in members]
            if not tr or not te:
                continue
            idx = {m: i for i, m in enumerate(names)}
            preds = {}
            for held in te:
                i = idx[held]
                p = predict_k3(cname, "CLR-Ridge",
                               X[cname][[idx[t] for t in tr]],
                               np.array([targets[t] for t in tr], float),
                               X[cname][[i]],
                               wt_matrix(p_wt, len(tr)), wt_matrix(p_wt, 1),
                               seed_base=i, n_seeds=1)
                preds[held] = p
            e = per_state_mae(preds, {m: targets[m] for m in te})
            out[f"{cname}::{gname}"] = {"mae": mean_mae(e),
                                        "errors": {k: round(v, 6) for k, v in e.items()}}
    return out


def lso_baseline(system, names, targets):
    """Per-group training-mean baseline: predict group members by the mean of
    group-EXTERNAL training targets (same train/test split as leave_site_out)."""
    groups = LSO_GROUPS[system]
    b = {}
    for gname, members in groups.items():
        tr = [m for m in names if m not in members]
        if not tr:
            continue
        mu = np.mean([targets[m] for m in tr], axis=0)
        errs = {m: float(np.abs(targets[m] - mu).mean()) for m in members}
        b[gname] = mean_mae(errs)
    return b


def gates(nested, nested_ctl, lso, lso_base, alt=None, system="abl1"):
    """Flat gate booleans per plan 4.6 (K=3)."""
    g = {}
    floor = 0.2329 if system == "abl1" else 0.2560
    g["nested_lt_floor"] = nested["nested_mae"] < floor
    g["nested_le_floor"] = nested["nested_mae"] <= floor
    g["no_marker_beats_control"] = nested["nested_mae"] <= nested_ctl["nested_mae"]
    gains = []
    for k, v in lso.items():
        gname = k.split("::")[1]
        if gname in lso_base:
            gains.append(v["mae"] - lso_base[gname])
    g["lso_impr_any_group"] = bool(gains) and min(gains) < 0
    g["lso_no_major_worsening"] = bool(gains) and max(gains) < 0.5 * max(lso_base.values())
    if alt is not None:
        g["alt_l410a_le_floor"] = alt["nested_mae"] <= 0.2560
        g["alt_no_verdict_reversal"] = (alt["nested_mae"] <= floor)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="abl1,src")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--skip-alt", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    results = {"seed": RAND_SEED, "device": str(_device()), "K": K,
               "combos": [c[0] for c in COMBO_DEFS], "models": MODEL_NAMES,
               "gp_protocol": PRIMARY_GP_PROTOCOL,
               "src_label_protocols": {
                   "primary": SRC_K3_PRIMARY_PROTOCOL_ID,
                   "alternative": SRC_K3_L410A_SUBSTITUTION_PROTOCOL_ID,
               },
               "systems": {}}
    for system in args.systems.split(","):
        system = system.strip()
        print(f"\n[{system}] loading features...", flush=True)
        names, f, targets = load_features(system)
        p_wt = np.array(WT_POP[system], float)
        print(f"  n={len(names)} p_wt={p_wt}", flush=True)
        fixed = run_fixed_loo(system, names, f, targets, p_wt,
                              COMBO_DEFS, n_seeds=args.seeds)
        nested = run_nested(system, names, f, targets, p_wt, COMBO_DEFS,
                            n_seeds=args.seeds)
        ctl = run_nested(system, names, f, targets, p_wt, COMBO_DEFS,
                         control=True, n_seeds=args.seeds)
        lso = leave_site_out(system, names, f, targets, p_wt, COMBO_DEFS)
        alt = None
        if system == "src" and not args.skip_alt:
            alt_panel = build_src_k3_panel(
                SRC_K3_L410A_SUBSTITUTION_PROTOCOL_ID)
            t_alt = {k: np.array(alt_panel.targets[k], float) for k in targets}
            alt = run_nested(system, names, f, t_alt, p_wt, COMBO_DEFS,
                             n_seeds=args.seeds)
        lb = lso_baseline(system, names, targets)
        g = gates(nested, ctl, lso, lb, alt=alt, system=system)
        results["systems"][system] = {"fixed": fixed, "nested": nested,
                                      "marker_ctl": ctl, "lso": lso,
                                      "alt_l410a": alt, "gates": g}
        print(f"  nested={nested['nested_mae']:.4f} ctl={ctl['nested_mae']:.4f} "
              f"dir={nested['direction']}", flush=True)
        print(f"  gates={json.dumps(g)}", flush=True)
    (RESULTS / "p2_k3_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] p2_k3 done in {time.time() - t0:.0f}s -> "
          f"{RESULTS / 'p2_k3_results.json'}")


if __name__ == "__main__":
    main()
