"""P2: Frozen ESM-2 feature evaluation (nested selector, Stage 2).

Reads p2_gpu_features.py outputs:
  results/p2_llr_features.json   (llr, tok_dir_unit per mutation)
  results/p2_site_deltas.npz     (1280-dim site deltas per mutation)

Candidate combos (predefined, dims in {1,2,4,8}, NO system-specific markers):
  C1 [llr] (1)   C2 [llr,pos] (2)   C3 [llr,tok] (2)
  C4 [rand4] (4) C5 [llr,tok,rand2] (4)   C6 [rand8] (8)
  llr  = masked-marginal LLR, normalized per system by max|llr| (frozen constant)
  pos  = mutation position / sequence length (transferable, not a marker)
  tok  = dot(site_delta, emb[mut]-emb[wt]) / |emb[mut]-emb[wt]| (frozen model direction)
  rand = Gaussian projection of site_delta, seed 0, rows unit-normalized (frozen)

Models: CLR-Ridge, CLR-GP, SimpleCDST (linear logit-shift, 5 seeds), LowRankCDST
(K=2, rank=2, hidden=32, 5 seeds). All scalings strictly fold-local.

Protocol: outer LOO; inner LOO selects (combo, model) by inner MAE (ties -> lower
dim, then lower index). Control row M1 = old 5-dim LLR+pos-markers encoding under
the SAME nested protocol (reference only, NOT a candidate).

Gates (plan 4.6): Abl1 nested < 0.2329; Src <= 0.2560; leave-site-out improvement
in >=1 group without major worsening of the other; no-marker combos beat the
marker control under the identical nested protocol.

Output: results/p2_eval_results.json + console summary.
"""
import sys
import json
import time
import argparse
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

RESULTS = Path(__file__).resolve().parent / "results"
EPS = 1e-6
RAND_SEED = 0          # frozen projection seed (pre-registered before labels)
RAND_DIMS = [2, 4, 8]

ABL1_CORE = {m: ABL1_K3[m] for m in ("M290L", "L301I", "M290L_L301I",
                                     "F382L", "F382Y", "F382V")}
SRC_CORE = {m: SRC_K3[m] for m in SRC_K3 if m != "SrcKD-WT"}
ABL1_SEQ_LEN = len(ABL1_KD)
SRC_SEQ_LEN = len(SRC_FULL)

LSO_GROUPS = {
    "abl1": {"F382_family": ["F382L", "F382Y", "F382V"],
             "290_301": ["M290L", "L301I", "M290L_L301I"]},
}
SRC_U1 = ["SrcKD-L410A", "SrcKD-V332I", "SrcKD-L270F_V332I",
          "SrcKD-L325A", "SrcKD-V380A", "SrcKD-V331A"]   # coarse contrast
SRC_U2 = ["SrcKD-A311I", "SrcKD-F405A"]                  # fine state contrast

COMBO_DEFS = [
    ("C1_llr1", ["llr"]),
    ("C2_llr_pos", ["llr", "pos"]),
    ("C3_llr_tok", ["llr", "tok"]),
    ("C4_rand4", ["rand4"]),
    ("C5_llr_tok_rand2", ["llr", "tok", "rand2"]),
    ("C6_rand8", ["rand8"]),
]
MODEL_NAMES = ["CLR-Ridge", "CLR-GP", "SimpleCDST", "LowRankCDST"]
TIE_DELTA = 0.05


def clr(y):
    y = np.clip(y, EPS, 1.0)
    y = y / y.sum(axis=-1, keepdims=True)
    return np.log(y) - np.log(y).mean(axis=-1, keepdims=True)


def inv_clr(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_lowrank(w_wt_tr, c_tr, t_tr, w_wt_te, c_te, d, n_seeds=5,
                  n_epochs=800, seed_base=0):
    dev = _device()
    preds = []
    for s in range(n_seeds):
        torch.manual_seed(s * 100 + seed_base)
        np.random.seed(s * 100 + seed_base)
        model = LowRankCDST(K=2, intervention_dim=d, rank=2, hidden_dim=32).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
        w_t = torch.FloatTensor(w_wt_tr).to(dev)
        c_t = torch.FloatTensor(c_tr).to(dev)
        t_t = torch.FloatTensor(t_tr).to(dev)
        best_loss, best_state = float("inf"), None
        for _ in range(n_epochs):
            opt.zero_grad()
            pred = model(w_t, c_t)
            loss = F.mse_loss(pred, t_t)
            loss.backward()
            opt.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            p = model(torch.FloatTensor(w_wt_te).to(dev),
                      torch.FloatTensor(c_te).to(dev)).cpu().numpy()[0, 1]
        preds.append(float(p))
    return float(np.mean(preds))


def train_simple(w_wt_tr, c_tr, t_tr, w_wt_te, c_te, d, n_seeds=5,
                 n_epochs=800, seed_base=0):
    dev = _device()
    preds = []
    for s in range(n_seeds):
        torch.manual_seed(s * 100 + seed_base)
        np.random.seed(s * 100 + seed_base)
        model = SimpleCDST(K=2, intervention_dim=d).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
        w_t = torch.FloatTensor(w_wt_tr).to(dev)
        c_t = torch.FloatTensor(c_tr).to(dev)
        t_t = torch.FloatTensor(t_tr).to(dev)
        best_loss, best_state = float("inf"), None
        for _ in range(n_epochs):
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
            p = torch.exp(model(torch.FloatTensor(w_wt_te).to(dev),
                                torch.FloatTensor(c_te).to(dev))).cpu().numpy()[0, 1]
        preds.append(float(p))
    return float(np.mean(preds))


def predict_fit(combo_name, model_name, X_tr, y_tr, X_te, w_wt_tr, w_wt_te,
                p_wt, seed_base):
    """Fold-local: scaler fit on training rows only. Returns predicted p (non-active).
    y_tr: 1-D non-active probabilities; converted to [1-p, p] internally."""
    d = X_tr.shape[1]
    y2 = np.stack([1.0 - np.asarray(y_tr, dtype=float),
                   np.asarray(y_tr, dtype=float)], axis=1)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_tr)
    Xte_s = scaler.transform(X_te)
    if model_name == "CLR-Ridge":
        m = Ridge(alpha=1.0)
        m.fit(Xtr_s, clr(y2))
        z = m.predict(Xte_s)[0]
        p = inv_clr(z[None, :])[0][1]
    elif model_name == "CLR-GP":
        m = make_primary_gp()
        m.fit(Xtr_s, clr(y2))
        z = m.predict(Xte_s)[0]
        p = inv_clr(z[None, :])[0][1]
    elif model_name == "SimpleCDST":
        p = train_simple(w_wt_tr, Xtr_s, y2, w_wt_te, Xte_s, d,
                         seed_base=seed_base)
    else:  # LowRankCDST
        p = train_lowrank(w_wt_tr, Xtr_s, y2, w_wt_te, Xte_s, d,
                          seed_base=seed_base)
    return p


def maes(preds, targets):
    return {m: float(abs(preds[m] - targets[m])) for m in preds}


def direction_report(preds, targets, p_wt):
    k, total = 0, 0
    for m in preds:
        td = targets[m] - p_wt
        if abs(td) < TIE_DELTA:
            continue
        total += 1
        if (preds[m] - p_wt) * td > 0:
            k += 1
    return f"{k}/{total}"


def jsd2(a, b):
    a = np.clip(np.asarray(a, float), EPS, 1.0)
    b = np.clip(np.asarray(b, float), EPS, 1.0)
    m = 0.5 * (a + b)
    return float(0.5 * (a * np.log(a / m) + b * np.log(b / m)).sum())


def load_features():
    feat = json.loads((RESULTS / "p2_llr_features.json").read_text(encoding="utf-8"))
    llr_all = feat["llr"]
    deltas = np.load(RESULTS / "p2_site_deltas.npz")

    rng = np.random.default_rng(RAND_SEED)
    P = {}
    for d in RAND_DIMS:
        M = rng.normal(0.0, 1.0, (d, deltas["abl1::M290L"].shape[0]))
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        P[d] = M

    def build(system, core, seq_len, p_wt):
        names = list(core.keys())
        f = {m: {} for m in names}
        for m in names:
            llr_v = llr_all[system][m]["llr"]
            f[m]["llr"] = llr_v
            f[m]["pos"] = core[m]["pos"] / seq_len
            f[m]["tok"] = llr_all[system][m]["tok_dir_unit"]
            dvec = deltas[f"{system}::{m}"]
            for d in RAND_DIMS:
                f[m][f"rand{d}"] = P[d] @ dvec
        mx = max(abs(f[m]["llr"]) for m in names) or 1.0
        for m in names:
            f[m]["llr"] = f[m]["llr"] / mx
        targets = {m: core[m]["pop"][0] for m in names}      # active fraction
        non_active = {m: 1.0 - core[m]["pop"][0] for m in names}
        return names, f, non_active

    d = {}
    d["abl1"] = build("abl1", ABL1_CORE, ABL1_SEQ_LEN, 1 - ABL1_K3_WT_POP[0])
    d["src"] = build("src", SRC_CORE, SRC_SEQ_LEN, 1 - SRC_K3_WT_POP[0])
    return d


def combo_matrix(feat, names, combo):
    return np.array([np.hstack([np.atleast_1d(feat[m][k]) for k in combo])
                     for m in names], dtype=float)


def marker_matrix(feat, names, core, system):
    seq_len = ABL1_SEQ_LEN if system == "abl1" else SRC_SEQ_LEN
    marks = [290, 301, 382] if system == "abl1" else [311, 332, 380]
    X = []
    for m in names:
        r = [core[m]["pos"] / seq_len, feat[m]["llr"]]
        r += [1.0 if core[m]["pos"] == p else 0.0 for p in marks]
        X.append(r)
    return np.array(X, dtype=float)


def run_nested(system, names, feat, core, p_wt, combo_defs, control=False):
    n = len(names)
    out = {"combo_defs": [c[0] for c in combo_defs],
           "models": MODEL_NAMES, "folds": {}, "nested_mae": None,
           "per_fold_selected": {}}
    all_preds = {m: None for m in names}
    targets = {m: 1 - core[m]["pop"][0] for m in names}
    tgt = np.array([targets[m] for m in names], dtype=float)

    def X_of(cmb):
        return (marker_matrix(feat, names, core, system)
                if control else combo_matrix(feat, names, cmb))

    for i, held in enumerate(names):
        tr = [j for j in range(n) if j != i]
        combos = [("M1_marker", None)] if control else combo_defs
        # inner selection (tie -> lower dim, then lower combo index, then model idx)
        best = None
        inner_scores = {}
        for j, hold2 in enumerate(tr):
            tr2 = [k for k in tr if k != hold2]
            for cname, cmb in combos:
                Xall = X_of(cmb)
                for mname in MODEL_NAMES:
                    p = predict_fit(cname, mname,
                                    Xall[tr2], tgt[tr2],
                                    Xall[[tr.index(hold2)]],
                                    np.tile([1 - p_wt, p_wt], (len(tr2), 1)),
                                    np.array([[1 - p_wt, p_wt]]),
                                    p_wt, seed_base=tr.index(hold2))
                    inner_scores.setdefault((cname, mname), []).append(
                        abs(p - tgt[hold2]))
        for ci, (cname, cmb) in enumerate(combos):
            for mi, mname in enumerate(MODEL_NAMES):
                im = float(np.mean(inner_scores[(cname, mname)]))
                if best is None or (im < best[0] - 1e-12 or
                                    (abs(im - best[0]) < 1e-12 and
                                     (ci, mi) < (best[1], best[2]))):
                    best = (im, ci, mi, mname, cname, cmb)
        cname, mname, cmb = best[4], best[3], best[5]
        Xall = X_of(cmb)
        p = predict_fit(cname, mname,
                        Xall[tr], tgt[tr],
                        Xall[[i]],
                        np.tile([1 - p_wt, p_wt], (len(tr), 1)),
                        np.array([[1 - p_wt, p_wt]]),
                        p_wt, seed_base=i)
        all_preds[held] = p
        out["folds"][held] = {"selected_combo": cname, "selected_model": mname,
                              "prediction": float(p), "target": targets[held],
                              "inner_mae": float(best[0]),
                              "inner_scores": {f"{c}::{m}": round(float(np.mean(v)), 6)
                                               for (c, m), v in inner_scores.items()}}
        out["per_fold_selected"][held] = f"{cname}/{mname}"
        print(f"  [{system}] fold {held}: select {cname}/{mname} "
              f"(inner {best[0]:.4f}) pred={p:.4f} target={targets[held]:.4f}",
              flush=True)
    errs = maes(all_preds, targets)
    out["nested_mae"] = float(np.mean(list(errs.values())))
    out["errors"] = errs
    out["predictions"] = all_preds
    out["targets"] = targets
    out["direction"] = direction_report(all_preds, targets, p_wt)
    return out


def run_fixed_combo(system, names, feat, core, p_wt, combo_defs):
    """Per-combo fixed evaluation (CLR-Ridge + LowRankCDST) - reference table."""
    targets = {m: 1 - core[m]["pop"][0] for m in names}
    tgt = np.array([targets[m] for m in names], dtype=float)
    out = {}
    n = len(names)
    for cname, cmb in combo_defs:
        X_all = combo_matrix(feat, names, cmb)
        for mname in ("CLR-Ridge", "LowRankCDST"):
            preds = {}
            for i, held in enumerate(names):
                tr = [j for j in range(n) if j != i]
                p = predict_fit(cname, mname,
                                X_all[tr], tgt[tr],
                                X_all[[i]],
                                np.tile([1 - p_wt, p_wt], (len(tr), 1)),
                                np.array([[1 - p_wt, p_wt]]), p_wt, seed_base=i)
                preds[held] = p
            errs = maes(preds, targets)
            out[f"{cname}::{mname}"] = {"mae": float(np.mean(list(errs.values()))),
                                        "errors": errs}
    return out


def leave_site_out(system, names, feat, core, p_wt, combo_defs):
    """Train on mutants not touching the group's sites; report group MAE (CLR-Ridge)."""
    targets = {m: 1 - core[m]["pop"][0] for m in names}
    out = {}
    for cname, cmb in combo_defs:
        X_all = combo_matrix(feat, names, cmb)
        for gname, members in (LSO_GROUPS[system].items() if system == "abl1"
                               else {"u1": SRC_U1, "u2": SRC_U2}.items()):
            tr = [m for m in names if m not in members]
            te = [m for m in names if m in members]
            if not tr or not te:
                continue
            preds = {}
            for held in te:
                sc = StandardScaler()
                Xtr = sc.fit_transform(X_all[[names.index(t) for t in tr]])
                m = Ridge(alpha=1.0)
                m.fit(Xtr, clr(np.array([[1 - targets[t], targets[t]] for t in tr])))
                p = inv_clr(m.predict(sc.transform(X_all[[names.index(held)]]))[0][None, :])[0][1]
                preds[held] = p
            errs = maes(preds, targets)
            out[f"{cname}::{gname}"] = {"mae": float(np.mean(list(errs.values()))),
                                        "errors": errs}
    return out


def main():
    parser = argparse.ArgumentParser(description="Deprecated K=2 predecessor of p2_k3_eval_v2.py")
    parser.add_argument("--i-know-this-is-deprecated", action="store_true", default=False,
                        help="acknowledge the invalidated normalization and run anyway")
    args = parser.parse_args()
    if not args.i_know_this_is_deprecated:
        raise SystemExit("DEPRECATED (superseded by p2_k3_eval_v2.py / p2_k3_eval.py): "
                         "this K=2 script uses the invalidated system-wide LLR "
                         "normalization; kept for provenance only. Pass "
                         "--i-know-this-is-deprecated to run anyway.")
    t0 = time.time()
    print("=" * 90)
    print("P2: frozen ESM-2 features - nested selector evaluation")
    print(f"combos: {[c[0] for c in COMBO_DEFS]}")
    print(f"models: {MODEL_NAMES}")
    print("=" * 90)
    data = load_features()
    results = {"seed": RAND_SEED, "device": str(_device()),
               "combos": [c[0] for c in COMBO_DEFS],
               "models": MODEL_NAMES, "gp_protocol": PRIMARY_GP_PROTOCOL,
               "systems": {}}
    for system in ("abl1", "src"):
        names, feat, _ = data[system]
        core = ABL1_CORE if system == "abl1" else SRC_CORE
        p_wt = (1 - ABL1_K3_WT_POP[0]) if system == "abl1" else (1 - SRC_K3_WT_POP[0])
        print(f"\n[{system}] nested LOO (n={len(names)})", flush=True)
        nested = run_nested(system, names, feat, core, p_wt, COMBO_DEFS)
        control = run_nested(system, names, feat, core, p_wt, COMBO_DEFS,
                             control=True)
        fixed = run_fixed_combo(system, names, feat, core, p_wt, COMBO_DEFS)
        lso = leave_site_out(system, names, feat, core, p_wt, COMBO_DEFS)
        results["systems"][system] = {"nested": nested,
                                      "marker_control_nested": control,
                                      "fixed_combo": fixed,
                                      "lso": lso, "p_wt": p_wt}
        print(f"  nested MAE = {nested['nested_mae']:.4f} "
              f"(direction {nested['direction']}; marker control "
              f"{control['nested_mae']:.4f})", flush=True)
        for k, v in fixed.items():
            print(f"    {k}: {v['mae']:.4f}", flush=True)

    abl1_mae = results["systems"]["abl1"]["nested"]["nested_mae"]
    src_mae = results["systems"]["src"]["nested"]["nested_mae"]
    abl1_ctl = results["systems"]["abl1"]["marker_control_nested"]["nested_mae"]
    src_ctl = results["systems"]["src"]["marker_control_nested"]["nested_mae"]
    gates = {
        "abl1_nested_lt_0.2329": abl1_mae < 0.2329,
        "src_nested_le_0.2560": src_mae <= 0.2560,
        "abl1_no_marker_beats_control": abl1_mae <= abl1_ctl,
        "src_no_marker_beats_control": src_mae <= src_ctl,
    }
    results["gates"] = gates
    (RESULTS / "p2_eval_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== gates: {json.dumps(gates, ensure_ascii=False)} ===")
    print(f"[OK] p2 eval done in {time.time() - t0:.0f}s -> "
          f"{RESULTS / 'p2_eval_results.json'}")


if __name__ == "__main__":
    main()
