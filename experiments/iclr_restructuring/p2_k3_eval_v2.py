"""P2-K3 nested evaluator v2 - compliant rebuild (NEXT_PHASE B1/B2/B3).

Supersedes p2_k3_eval.py (kept as numeric diagnostic; results not overwritten).
Fixes per NEXT_PHASE_EXECUTION_PLAN.md B1:
  1. tie-break = MAE -> dimension -> model simplicity -> combo ID
  2. scalers fit on training IDs only; transform params + fit IDs recorded
  3. per-inner-split candidate/model scores saved
  4. complete transform params and fit IDs saved
  5. random projection seed=0, matrix hash saved
  6. torch models save five independent per-seed predictions
  7. deterministic CLR models record n_seeds entries flagged deterministic
  8. selector re-run for main, marker control, and alt-label
  9. all metrics computed inside the evaluator (catastrophic + comparator,
     paired improvement, single-mutant contribution, drop-one, support-in/out,
     per-seed variance, fixed combo table marked exploratory)
  10. ILR epsilon=1e-6 with renormalization
  11. JSD stored as divergence, distance saved separately
  12. raw u1/u2 computed on inverse-transformed K=3 simplex
Gates per ADR-003 (FROZEN): independent booleans + overall_go.

Frozen protocol: K=3, seeds=[0..4] (s*100+outer holdout idx), epochs=800,
rank=2, hidden_dim=32, random_projection_seed=0, TIE_DELTA=0.05.
Candidates (frozen): C1[llr] C2[llr,pos] C3[llr,tok] C4[rand4]
C5[llr,tok,rand2] C6[rand8]; models = CLR-Ridge / CLR-GP / SimpleCDST / LowRankCDST.
Inputs: results/p2_llr_features.json + p2_site_deltas.npz (frozen manifests).
Usage: python p2_k3_eval_v2.py [--systems abl1,src] [--skip-alt] [--seeds 5]
Outputs: p2_k3_nested_results.json / p2_k3_l410a_sensitivity.json /
p2_k3_leave_site_out.json / p2_k3_run_manifest.json
"""
import sys, json, time, argparse, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
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

def _find_root():
    p = HERE
    while p != p.parent:
        if (p / "src" / "models" / "low_rank_cdst.py").exists():
            return p
        p = p.parent
    return HERE

ROOT = _find_root()

RESULTS = HERE / "results"
EPS = 1e-6
K = 3
RAND_SEED = 0
RAND_DIMS = [2, 4, 8]
TIE_DELTA = 0.05
NEPOCHS = 800
FLOORS = {"abl1": 0.2329, "src": 0.2560}

GATE_THRESHOLDS = {
    "abl1_nested_lt_0_2329": 0.2329,
    "src_nested_le_0_2560": 0.2560,
    "no_marker_strictly_beats_marker": None,
    "lso_same_route_pass": 0.05,
    "catastrophic_not_worse_than_control": 0,
    "single_mutant_contribution_le_0_50": 0.50,
    "alt_l410a_le_0_2560": 0.2560,
    "alt_verdict_not_reversed": None,
}

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
MODEL_SIMPLICITY = {m: i for i, m in enumerate(MODEL_NAMES)}
COMBO_DIM = {"C1_llr1": 1, "C2_llr_pos": 2, "C3_llr_tok": 2,
             "C4_rand4": 4, "C5_llr_tok_rand2": 3, "C6_rand8": 8}
MARKER_MARKS = {"abl1": [290, 301, 382], "src": [311, 332, 380]}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path, payload):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                              default=lambda value: (value.tolist()
                                                     if isinstance(value, np.ndarray)
                                                     else value.item()
                                                     if isinstance(value, np.generic)
                                                     else str(value))),
                   encoding="utf-8")
    tmp.replace(path)


def checkpoint_signature(system, args, script_hash, adr_hashes):
    dependencies = {
        "encoding_ablation_control.py": HERE / "encoding_ablation_control.py",
        "gp_protocols.py": HERE / "gp_protocols.py",
        "k3_data.py": HERE / "k3_data.py",
        "src_k3_labels.py": ROOT / "src" / "data" / "src_k3_labels.py",
        "src_k3_canonical.csv": ROOT / "data" / "nmr_populations" / "src_k3_canonical.csv",
    }
    return {
        "schema": "p2_k3_eval_v2_system_checkpoint_v1",
        "system": system,
        "script_sha256": script_hash,
        "version": args.version,
        "parameters": {
            "K": K, "seeds": args.seeds, "epochs": NEPOCHS,
            "rank": 2, "hidden_dim": 32,
            "random_projection_seed": RAND_SEED,
            "tie_delta": TIE_DELTA, "ilr_eps": EPS,
            "skip_alt": bool(args.skip_alt),
            "combos": COMBO_DEFS, "models": MODEL_NAMES,
            "gp_protocol": PRIMARY_GP_PROTOCOL,
        },
        "inputs": {
            "primary": adr_hashes,
            "dependencies": {name: sha256_file(path)
                             for name, path in dependencies.items()},
        },
        "environment": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "device": str(_device()),
        },
    }


def load_system_checkpoint(path, expected_signature):
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ignoring unreadable checkpoint {path.name}: {exc}", flush=True)
        return None
    expected = json.loads(json.dumps(expected_signature))
    if payload.get("signature") != expected:
        print(f"ignoring stale checkpoint {path.name}: signature mismatch", flush=True)
        return None
    if not {"block", "leave_site_out", "alternative"} <= payload.keys():
        print(f"ignoring incomplete checkpoint {path.name}", flush=True)
        return None
    return payload


def load_progress_checkpoint(path, expected_signature):
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ignoring unreadable progress checkpoint {path.name}: {exc}",
              flush=True)
        return None
    expected = json.loads(json.dumps(expected_signature))
    if payload.get("signature") != expected or not isinstance(payload.get("stages"), dict):
        print(f"ignoring stale progress checkpoint {path.name}", flush=True)
        return None
    return payload


def save_progress_stage(progress, path, name, value, complete):
    progress.setdefault("stages", {})[name] = {
        "complete": bool(complete),
        "value": value,
    }
    atomic_write_json(path, progress)


def progress_stage(progress, name):
    stage = progress.get("stages", {}).get(name)
    if not isinstance(stage, dict):
        return False, None
    return bool(stage.get("complete")), stage.get("value")


def clr(y):
    y = np.clip(np.asarray(y, float), EPS, 1.0)
    y = y / y.sum(axis=-1, keepdims=True)
    return np.log(y) - np.log(y).mean(axis=-1, keepdims=True)


def inv_clr(z):
    z = np.asarray(z, float)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    v = e / e.sum(axis=-1, keepdims=True)
    return v[0] if v.ndim == 2 and v.shape[0] == 1 else v


def ilr(y):
    """Symmetric ILR on the K=3 simplex: z1 = sqrt(2/3)*log(p1/sqrt(p2*p3)),
    z2 = (1/sqrt(2))*log(p2/p3). Clip at EPS, renormalize, then transform."""
    p = np.clip(np.asarray(y, float), EPS, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    z1 = np.sqrt(2.0 / 3.0) * np.log(p[..., 0] / np.sqrt(p[..., 1] * p[..., 2]))
    z2 = (1.0 / np.sqrt(2.0)) * np.log(p[..., 1] / p[..., 2])
    return np.stack([z1, z2], axis=-1)


def inv_ilr(z):
    """Inverse of ilr: z (...,2) -> simplex (...,3). Orthonormal basis
    v1=(1,-1/2,-1/2)/sqrt(1.5), v2=(0,1,-1)/sqrt(2); log p = z1 v1 + z2 v2 + c."""
    z1 = z[..., 0] / np.sqrt(1.5)
    z2 = z[..., 1] / np.sqrt(2.0)
    l1 = z1
    l2 = -z1 / 2.0 + z2
    l3 = -z1 / 2.0 - z2
    l = np.stack([l1, l2, l3], axis=-1)
    e = np.exp(l - np.max(l, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def jsd(p, q):
    """Jensen-Shannon divergence (natural log); distance = sqrt(JSD) saved separately."""
    p = np.clip(np.asarray(p, float), EPS, 1.0)
    q = np.clip(np.asarray(q, float), EPS, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * (np.log(a) - np.log(b)), axis=-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def raw_u1_u2(pred, target):
    """raw u1 = 2*p_active - 1, u2 = p_E1 - p_E2 (K=3, Active/E1/E2 order)."""
    u1p, u1t = 2.0 * pred[0] - 1.0, 2.0 * target[0] - 1.0
    u2p, u2t = pred[1] - pred[2], target[1] - target[2]
    return {"u1_pred": float(u1p), "u1_target": float(u1t),
            "u2_pred": float(u2p), "u2_target": float(u2t)}


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_features(system):
    """Returns (names, feat dict, targets dict, rand_hash). LLR kept RAW;
    scaling is fold-local (StandardScaler on training rows)."""
    feat = json.loads((RESULTS / "p2_llr_features.json").read_text(encoding="utf-8"))
    llr_all = feat["llr"]
    deltas = np.load(RESULTS / "p2_site_deltas.npz")
    rng = np.random.default_rng(RAND_SEED)
    P, P_blob = {}, b""
    for d in RAND_DIMS:
        M = rng.normal(0.0, 1.0, (d, deltas["abl1::M290L"].shape[0]))
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        P[d] = M
        P_blob += M.tobytes()
    rand_hash = hashlib.sha256(P_blob).hexdigest()
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
    return names, f, targets, rand_hash


def combo_matrix(f, names, combo):
    return np.array([np.hstack([np.atleast_1d(f[m][k]) for k in combo])
                     for m in names], float)


def marker_matrix(f, names, system):
    marks = MARKER_MARKS[system]
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
    return preds


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
    return preds


def _norm_simplex(p):
    p = np.asarray(p, float)
    return p / p.sum() if p.sum() > 0 else p


def predict_k3(cname, mname, X_tr, y_tr, X_te, w_wt_tr, w_wt_te, seed_base,
               n_seeds=5, fit_ids=None):
    """Fold-local prediction. Returns dict with mean, per_seed, deterministic,
    transform_params (scaler mean/scale + fit_ids)."""
    d = X_tr.shape[1]
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_tr)
    Xte_s = scaler.transform(X_te)
    tparams = {"scaler_mean": scaler.mean_.tolist(),
               "scaler_scale": scaler.scale_.tolist(),
               "fit_ids": list(fit_ids) if fit_ids is not None else None}
    if mname == "CLR-Ridge":
        m = Ridge(alpha=1.0)
        m.fit(Xtr_s, clr(y_tr))
        p = _norm_simplex(inv_clr(m.predict(Xte_s)))
        return {"mean": p, "per_seed": [p.copy() for _ in range(n_seeds)],
                "deterministic": True, "transform_params": tparams}
    if mname == "CLR-GP":
        m = make_primary_gp()
        m.fit(Xtr_s, clr(y_tr))
        p = _norm_simplex(inv_clr(m.predict(Xte_s)))
        return {"mean": p, "per_seed": [p.copy() for _ in range(n_seeds)],
                "deterministic": True, "transform_params": tparams}
    if mname == "SimpleCDST":
        preds = [_norm_simplex(p) for p in
                 train_simple(w_wt_tr, Xtr_s, y_tr, w_wt_te, Xte_s, d,
                              seed_base, n_seeds)]
        return {"mean": _norm_simplex(np.mean(preds, axis=0)), "per_seed": preds,
                "deterministic": False, "transform_params": tparams}
    preds = [_norm_simplex(p) for p in
             train_lowrank(w_wt_tr, Xtr_s, y_tr, w_wt_te, Xte_s, d,
                           seed_base, n_seeds)]
    return {"mean": _norm_simplex(np.mean(preds, axis=0)), "per_seed": preds,
            "deterministic": False, "transform_params": tparams}


# ---------------------------------------------------------------------------
# Protocol: K=3 metrics, fixed-combo LOO (exploratory), nested LOO
# ---------------------------------------------------------------------------
def per_state_mae(preds, targets):
    return {m: float(np.abs(np.asarray(preds[m], float)
                            - np.asarray(targets[m], float)).mean())
            for m in preds}


def mean_mae(errs):
    return float(np.mean(list(errs.values())))


def wt_matrix(p_wt, n):
    return np.tile(np.asarray(p_wt, float), (n, 1))


def run_fixed_loo(system, names, f, targets, p_wt, combo_defs, n_seeds=5,
                  resume=None, checkpoint=None):
    """Per-(combo, model) fixed LOO at K=3; exploratory only (not a gate)."""
    X = {c: combo_matrix(f, names, cmb) for c, cmb in combo_defs}
    out = dict(resume or {})
    hi = {m: i for i, m in enumerate(names)}
    for cname, cmb in combo_defs:
        for mname in MODEL_NAMES:
            key = f"{cname}::{mname}"
            if key in out:
                continue
            preds = {}
            for m in names:
                i = hi[m]
                tr = [j for j in range(len(names)) if j != i]
                res = predict_k3(cname, mname,
                                 X[cname][tr],
                                 np.array([targets[names[j]] for j in tr], float),
                                 X[cname][[i]],
                                 wt_matrix(p_wt, len(tr)),
                                 wt_matrix(p_wt, 1),
                                 seed_base=i, n_seeds=n_seeds,
                                 fit_ids=[names[j] for j in tr])
                preds[m] = res["mean"]
                assert preds[m].shape == (K,)
            e = per_state_mae(preds, targets)
            out[key] = {"mae": mean_mae(e),
                        "errors": {k: round(v, 6) for k, v in e.items()},
                        "exploratory": True}
            if checkpoint is not None:
                checkpoint(out)
    return out


def tie_break_key(scores, key):
    """B1 tie-break: (MAE, dimension, model simplicity, combo ID).
    M1_marker (control) has no combo ID; it is the only candidate there."""
    mae = float(np.mean(scores[key]))
    cname, mname = key.split("::")
    cix = ([c[0] for c in COMBO_DEFS].index(cname)
           if cname in [c[0] for c in COMBO_DEFS] else -1)
    return (mae, COMBO_DIM.get(cname, 0), MODEL_SIMPLICITY[mname], cix)


def select_best(inner_scores):
    """Selection-aware inner selector per B1 (all candidate scores recorded)."""
    if not inner_scores:
        return None
    return min(inner_scores, key=lambda k: tie_break_key(inner_scores, k))


def training_mean_mae(targets, tr_idx, held_target):
    """Training-mean baseline for one held-out target (ADR-003 G4/G6)."""
    mu = np.mean([targets[j] for j in tr_idx], axis=0)
    return float(np.abs(mu - held_target).mean())


def catastrophic_flags(preds, targets, names, tr_of, floor_zero=0.05):
    """ADR-003 G6: per-mutant MAE > 2x same-fold training-mean MAE;
    baseline==0 -> candidate error > 0.05 is catastrophic.
    targets: dict keyed by name; tr_of values are immutable int indices."""
    tgt_list = [targets[m] for m in names]
    flags = {}
    for m in preds:
        e = float(np.abs(np.asarray(preds[m]) - np.asarray(targets[m])).mean())
        b = training_mean_mae(tgt_list, tr_of[m], tgt_list[names.index(m)])
        flags[m] = bool(e > 2.0 * b) if b > 0 else bool(e > floor_zero)
    return flags


def support_in_out(names, tr_of, positions):
    """support-in: all mutation sites of the held-out mutant appear in the
    outer training mutants; support-out: at least one site is missing.
    positions: dict keyed by name; tr_of values are immutable int indices."""
    pos_list = [positions[m] for m in names]
    out = {}
    for m in tr_of:
        held_pos = set(pos_list[names.index(m)])
        tr_pos = set()
        for t in tr_of[m]:
            tr_pos |= set(pos_list[t])
        out[m] = "in" if held_pos <= tr_pos else "out"
    return out


def run_nested(system, names, f, targets, p_wt, combo_defs, control=False,
               n_seeds=5, positions=None, train_universe=None, holdouts=None,
               resume=None, checkpoint=None):
    """Nested LOO (selection-aware): inner LOO over training fold selects
    (combo, model) by inner MAE with tie-break (MAE, dim, simplicity, combo ID);
    outer fold refits the selected pair. Records per-seed predictions,
    transform params, inner scores, fit IDs; all row indices immutable.
    control=True runs the marker-control route (M1_marker only).
    train_universe/holdouts: for LSO, training restricted to group-external
    mutants, predictions only for group members."""
    holdouts = holdouts if holdouts is not None else names
    tu = train_universe if train_universe is not None else names
    n = len(names)
    tgt = [targets[m] for m in names]
    hi = {m: i for i, m in enumerate(names)}
    tu_i = [hi[m] for m in tu]

    def X_of(cname, cmb):
        return (marker_matrix(f, names, system) if cname == "M1_marker"
                else combo_matrix(f, names, cmb))

    out = dict(resume or {})
    out.setdefault("folds", {})
    out.setdefault("preds", {})
    out.setdefault("per_fold_selected", {})
    out.setdefault("inner_scores", {})
    out.setdefault("index_alignment_ok", True)
    tr_of = {}
    for held in holdouts:
        i = hi[held]
        tr = [j for j in tu_i if j != i]
        tr_of[held] = tr
        if held in out["folds"]:
            continue
        combos = [("M1_marker", None)] if (control or n == 1) else combo_defs
        inner_scores = {}
        for j in tr:
            tr2 = [k for k in tr if k != j]
            for cix, (cname, cmb) in enumerate(combos):
                Xtr2 = X_of(cname, cmb)[tr2]
                Xj = X_of(cname, cmb)[[j]]
                for mix, mname in enumerate(MODEL_NAMES):
                    res = predict_k3(cname, mname,
                                     Xtr2, np.array([tgt[j] for j in tr2], float),
                                     Xj,
                                     wt_matrix(p_wt, len(tr2)), wt_matrix(p_wt, 1),
                                     seed_base=i, n_seeds=n_seeds,
                                     fit_ids=[names[k] for k in tr2])
                    inner_scores.setdefault(f"{cname}::{mname}", []).append(
                        float(np.abs(res["mean"] - tgt[j]).mean()))
        best_key = select_best(inner_scores)
        cname, mname = best_key.split("::")
        cix = [c[0] for c in combos].index(cname)
        Xtr = X_of(cname, combos[cix][1])[tr]
        res = predict_k3(cname, mname,
                         Xtr, np.array([tgt[j] for j in tr], float),
                         X_of(cname, combos[cix][1])[[i]],
                         wt_matrix(p_wt, len(tr)), wt_matrix(p_wt, 1),
                         seed_base=i, n_seeds=n_seeds,
                         fit_ids=[names[j] for j in tr])
        p = res["mean"]
        fold = {"holdout": held, "combo": cname, "model": mname,
                "pred": np.round(p, 9).tolist(),
                "target": np.round(tgt[i], 9).tolist(),
                "inner_mae": float(np.mean(inner_scores[best_key])),
                "inner_scores": {k: [round(v, 6) for v in vals]
                                 for k, vals in inner_scores.items()},
                "per_seed_predictions": {
                    f"seed_{s}": np.round(np.asarray(res["per_seed"][s]), 9).tolist()
                    for s in range(len(res["per_seed"]))},
                "deterministic": res["deterministic"],
                "transform_params": res["transform_params"]}
        out["folds"][held] = fold
        out["preds"][held] = np.round(p, 9).tolist()
        out["per_fold_selected"][held] = f"{cname}/{mname}"
        out["inner_scores"][held] = {k: [round(v, 6) for v in vals]
                                      for k, vals in inner_scores.items()}
        assert len(inner_scores[best_key]) == len(tr), (system, held)
        if checkpoint is not None:
            out["tr_of"] = {m: [names[j] for j in tr_of[m]] for m in tr_of}
            checkpoint(out)
    errs = per_state_mae(out["preds"], {m: targets[m] for m in names})
    out["nested_mae"] = mean_mae(errs)
    out["errors"] = {k: round(v, 6) for k, v in errs.items()}
    out["direction"] = direction_report(out["preds"], targets, p_wt)
    out["mae_per_mutant"] = out["errors"]
    out["tr_of"] = {m: [names[j] for j in tr_of[m]] for m in tr_of}
    out["catastrophic"] = catastrophic_flags(out["preds"], targets, names, tr_of)
    out["catastrophic_folds"] = sorted(m for m, c in out["catastrophic"].items() if c)
    if positions is not None:
        out["support"] = support_in_out(names, tr_of, positions)
    return out


def direction_report(preds, targets, p_wt):
    """ADR-002: ACTIVE-state sign agreement (u1 = 2*p_active - 1); mutants with
    |target_active - wt_active| < TIE_DELTA are ties, excluded; k/total."""
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


def u1_u2_contrast(preds, targets):
    """raw u1/u2 + ILR z1/z2 contrast MAEs (schema metrics.u1_u2_contrast_mae)."""
    u1e = [abs(2 * preds[m][0] - 1 - (2 * targets[m][0] - 1)) for m in preds]
    u2e = [abs((preds[m][1] - preds[m][2]) - (targets[m][1] - targets[m][2]))
           for m in preds]
    z1e, z2e = [], []
    for m in preds:
        zp = ilr(np.atleast_2d(preds[m]))
        zt = ilr(np.atleast_2d(targets[m]))
        z1e.append(abs(zp[0, 0] - zt[0, 0]))
        z2e.append(abs(zp[0, 1] - zt[0, 1]))
    return {"u1": float(np.mean(u1e)), "u2": float(np.mean(u2e)),
            "ilr_z1": float(np.mean(z1e)), "ilr_z2": float(np.mean(z2e))}


def jsd_metrics(preds, targets):
    divs = {m: float(jsd(np.atleast_2d(preds[m]), np.atleast_2d(targets[m]))[0])
            for m in preds}
    return {"divergence": divs, "mean_divergence": float(np.mean(list(divs.values()))),
            "distance": {m: float(np.sqrt(v)) for m, v in divs.items()}}


def per_seed_variance(nested):
    """Per-fold per-state std of per-seed predictions (torch models)."""
    out = {}
    for held, fold in nested["folds"].items():
        if fold.get("deterministic"):
            continue
        arr = np.array([fold["per_seed_predictions"][f"seed_{s}"]
                        for s in range(len(fold["per_seed_predictions"]))])
        out[held] = {"per_state_std": [round(float(x), 6) for x in arr.std(axis=0)],
                     "max_std": float(arr.std(axis=0).max())}
    return out


def paired_improvement(cand_err, ctl_err):
    """Per-mutant paired improvement: ctl_err - cand_err (positive = improved)."""
    return {m: float(ctl_err[m] - cand_err[m])
            for m in cand_err if m in ctl_err}


def single_mutant_contribution(paired):
    """ADR-003 G7: max |paired| / sum |paired| <= 0.50."""
    tot = sum(abs(v) for v in paired.values())
    if tot <= 0:
        return 0.0
    return float(max(abs(v) for v in paired.values()) / tot)


def drop_one_paired_improvement(paired):
    """Drop-one: remove one mutant, recompute mean paired improvement (no
    retraining, no selection; from stored outer-fold paired errors)."""
    out = {}
    for m in paired:
        rest = {k: v for k, v in paired.items() if k != m}
        out[m] = float(np.mean(list(rest.values()))) if rest else None
    return out


def selection_aware_lso(system, names, f, targets, p_wt, combo_defs, n_seeds=5,
                         positions=None, resume=None, checkpoint=None):
    """ADR-003 G3/G4/G5: same full selection-aware route re-run on each frozen
    LSO group. Leave-site-out: tr = mutants not in the group (group members
    excluded from ALL training); members predicted via the full inner-selection
    route trained only on tr. Comparator = per-split training-mean MAE on the
    held-out members (ADR-003 G4). No cross-config splicing: one route, two
    groups."""
    groups = LSO_GROUPS[system]
    state = dict(resume or {})
    out = state.setdefault("groups", {})
    partial = state.setdefault("partial", {})
    for gname, members in groups.items():
        if gname in out:
            continue
        tr = [m for m in names if m not in members]
        te = [m for m in names if m in members]
        if not tr or not te:
            continue
        def save_partial(value, group=gname):
            partial[group] = value
            if checkpoint is not None:
                checkpoint(state)

        sel = run_nested(system, names, f, targets, p_wt, combo_defs,
                         control=False, n_seeds=n_seeds, positions=None,
                         train_universe=tr, holdouts=te,
                         resume=partial.get(gname), checkpoint=save_partial)
        mu = np.mean([np.asarray(targets[m], float) for m in tr], axis=0)
        group_errs = {}
        for held in te:
            group_errs[held] = float(np.abs(np.asarray(sel["preds"][held])
                                            - np.asarray(targets[held])).mean())
        comparator = float(np.mean([np.abs(mu - np.asarray(targets[m])).mean()
                                    for m in te]))
        g_mae = mean_mae(group_errs)
        out[gname] = {"group_mae": g_mae, "comparator": comparator,
                      "members": members,
                      "errors": {k: round(v, 6) for k, v in group_errs.items()},
                      "selected": {m: sel["per_fold_selected"][m] for m in te}}
        partial.pop(gname, None)
        if checkpoint is not None:
            checkpoint(state)
    return out


def lso_same_route_gate(lso):
    """ADR-003 G5: >=1 group improves (mae < comparator) and the other's
    relative worsening <= 5%."""
    vals = [(g["group_mae"], g["comparator"]) for g in lso.values()]
    if not vals:
        return False
    gains = [g - c for g, c in vals]
    rel_worsen = [((g - c) / c) if c > 0 else 0.0 for g, c in vals]
    return (min(gains) < 0) and (max(rel_worsen) <= 0.05)


def gates(nested, nested_ctl, lso, alt, system):
    """ADR-003 G8: independent booleans + overall_go (no duplicate fields).
    Returns {key: {"passed": bool, "value": float-or-None}} with thresholds
    from GATE_THRESHOLDS (frozen)."""
    g = {}
    floor = FLOORS[system]
    cand = nested["nested_mae"]
    ctl = nested_ctl["nested_mae"]
    paired = paired_improvement(nested["errors"], nested_ctl["errors"])
    contrib = single_mutant_contribution(paired)
    if system == "abl1":
        g["abl1_nested_lt_0_2329"] = {"passed": bool(cand < floor), "value": cand}
    else:
        g["src_nested_le_0_2560"] = {"passed": bool(cand <= floor), "value": cand}
    g["no_marker_strictly_beats_marker"] = {"passed": bool(cand < ctl), "value": cand}
    g["lso_same_route_pass"] = {"passed": lso_same_route_gate(lso),
                                "value": None}
    g["catastrophic_not_worse_than_control"] = {
        "passed": bool(len(nested["catastrophic_folds"]) <= len(nested_ctl["catastrophic_folds"])),
        "value": len(nested["catastrophic_folds"])}
    g["single_mutant_contribution_le_0_50"] = {"passed": bool(contrib <= 0.50),
                                               "value": contrib}
    if system == "src" and alt is not None:
        alt_mae = alt["nested_mae"]
        g["alt_l410a_le_0_2560"] = {"passed": bool(alt_mae <= 0.2560),
                                    "value": alt_mae}
        main_pass = cand <= floor
        alt_pass = alt_mae <= 0.2560
        g["alt_verdict_not_reversed"] = {"passed": bool(main_pass == alt_pass),
                                         "value": None}
    return g


def build_result_json(system, names, f, targets, p_wt, fixed, nested, ctl, lso,
                      alt, g, rand_hash, script_hash, adr_hashes, seed_count,
                      version, positions=None):
    """Schema-conformant output for the main system block."""
    dev = str(_device())
    metrics = {
        "mae_per_mutant": nested["mae_per_mutant"],
        "mae": nested["nested_mae"],
        "direction": nested["direction"],
        "jsd": jsd_metrics(nested["preds"], targets)["mean_divergence"],
        "u1_u2_contrast_mae": u1_u2_contrast(nested["preds"], targets),
        "leave_site_out": {gname: {"group_mae": v["group_mae"],
                                   "members": v["members"],
                                   "comparator": v["comparator"]}
                           for gname, v in lso.items()},
        "catastrophic_folds": nested["catastrophic_folds"],
        "support_stratified_error": (
            {k: float(np.mean([nested["errors"][m] for m in names
                               if positions and nested["support"][m] == k]))
             for k in ("in", "out")
             if positions and any(nested["support"][m] == k for m in names)}
            or None),
        "paired_per_mutant_improvement": paired_improvement(
            nested["errors"], ctl["errors"])}
    if system == "src" and alt is not None:
        metrics["alternative_label_sensitivity"] = {
            "alt_nested_mae": alt["nested_mae"],
            "alt_direction": alt["direction"]}
    return {
        "metadata": {"script": "p2_k3_eval_v2.py", "version": version,
                     "hash": script_hash[:32], "date": time.strftime("%Y-%m-%d"),
                     "device": dev},
        "data_hashes": {"canonical_data": adr_hashes.get("canonical_data", ""),
                        "feature_cache": adr_hashes.get("feature_cache", ""),
                        "model_checkpoint": adr_hashes.get("model_checkpoint", ""),
                        "frozen_random_projection": rand_hash},
        "label_info": {"system": system,
                       "state_order": ["Active", "E1", "E2"],
                       "wt_population": [float(x) for x in p_wt],
                       "label_source": ("Xie 2020 probe" if system == "abl1"
                                        else "Cui 2025 Fig S5 Met305 probe"),
                       "probe_or_global_fit": "probe"},
        "folds": {"outer": names,
                  "inner": {m: [t for t in tr] for m, tr in
                            {m: nested["tr_of"][m] for m in names}.items()},
                  "seed_scheme": "seed = s*100 + outer_holdout_immutable_index",
                  "seeds": list(range(seed_count))},
        "results": {"per_fold": {m: {"holdout": m,
                                     "selected_combo": nested["folds"][m]["combo"],
                                     "selected_model": nested["folds"][m]["model"],
                                     "per_seed_predictions": nested["folds"][m]["per_seed_predictions"],
                                     "mean_predictions": nested["folds"][m]["pred"],
                                     "targets": nested["folds"][m]["target"],
                                     "mae": nested["errors"][m],
                                     "inner_scores": nested["inner_scores"][m],
                                     "transform_params": nested["folds"][m]["transform_params"],
                                     "catastrophic": nested["catastrophic"][m]}
                                for m in names}},
        "metrics": metrics,
        "hard_gates": {"verdict": "GO" if all(v["passed"] for v in g.values()) else "NO_GO",
                       "gates": {k: {"passed": v["passed"],
                                     "value": v["value"],
                                     "threshold": GATE_THRESHOLDS.get(k)}
                                 for k, v in g.items()}},
        "analyses": {"fixed_loo": fixed,
                     "marker_ctl": {"nested_mae": ctl["nested_mae"],
                                    "errors": ctl["errors"],
                                    "direction": ctl["direction"],
                                    "catastrophic_folds": ctl["catastrophic_folds"]},
                     "paired_improvement": paired_improvement(nested["errors"], ctl["errors"]),
                     "single_mutant_contribution": single_mutant_contribution(
                         paired_improvement(nested["errors"], ctl["errors"])),
                     "drop_one_paired_improvement": drop_one_paired_improvement(
                         paired_improvement(nested["errors"], ctl["errors"])),
                     "per_seed_variance": per_seed_variance(nested),
                     "support": nested.get("support"),
                     "lso": lso,
                     "jsd_divergence_per_mutant": jsd_metrics(nested["preds"], targets)["divergence"],
                     "jsd_distance_per_mutant": jsd_metrics(nested["preds"], targets)["distance"],
                     "raw_u1_u2_per_mutant": {m: raw_u1_u2(nested["preds"][m], targets[m])
                                              for m in names},
                     "ilr_per_mutant": {m: {"z1_pred": float(ilr(np.atleast_2d(nested["preds"][m]))[0, 0]),
                                            "z2_pred": float(ilr(np.atleast_2d(nested["preds"][m]))[0, 1]),
                                            "z1_target": float(ilr(np.atleast_2d(targets[m]))[0, 0]),
                                            "z2_target": float(ilr(np.atleast_2d(targets[m]))[0, 1])}
                                        for m in names}},
    }


def positions_from_features(system):
    feat = json.loads((RESULTS / "p2_llr_features.json").read_text(encoding="utf-8"))
    core = ABL1_CORE if system == "abl1" else SRC_CORE
    pos = {}
    for m in core:
        ps = feat["llr"][system][m].get("positions", [])
        pos[m] = sorted({p.get("nominal_pos") for p in ps} - {None})
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="abl1,src")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--skip-alt", action="store_true")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore compatible per-system checkpoints")
    ap.add_argument("--version", default="2.0.0")
    args = ap.parse_args()

    t0 = time.time()
    script_hash = sha256_file(__file__)
    adr_hashes = {
        "canonical_data": sha256_file(HERE / "results" / "p2_llr_features.json")[:32],
        "feature_cache": sha256_file(HERE / "results" / "p2_site_deltas.npz")[:32],
        "model_checkpoint": sha256_file(ROOT / "src" / "models" / "low_rank_cdst.py")[:32],
        "adr_002": sha256_file(HERE / "results" / "ADR-002-direction-definition.md")[:32],
        "adr_003": sha256_file(HERE / "results" / "ADR-003-p2-gate-operationalization.md")[:32],
    }
    results = {"seed": RAND_SEED, "device": str(_device()), "K": K,
               "combos": [c[0] for c in COMBO_DEFS], "models": MODEL_NAMES,
               "gp_protocol": PRIMARY_GP_PROTOCOL,
               "src_label_protocols": {
                   "primary": SRC_K3_PRIMARY_PROTOCOL_ID,
                   "alternative": SRC_K3_L410A_SUBSTITUTION_PROTOCOL_ID,
               },
               "systems": {}, "adr_hashes": adr_hashes,
               "script_sha256": script_hash}
    lso_all, alt_all = {}, {}
    for system in args.systems.split(","):
        system = system.strip()
        signature = checkpoint_signature(system, args, script_hash, adr_hashes)
        checkpoint_path = RESULTS / f".p2_k3_eval_v2_{system}.checkpoint.json"
        progress_path = RESULTS / f".p2_k3_eval_v2_{system}.progress.json"
        checkpoint = (None if args.no_resume else
                      load_system_checkpoint(checkpoint_path, signature))
        if checkpoint is not None:
            results["systems"][system] = checkpoint["block"]
            lso_all[system] = checkpoint["leave_site_out"]
            if checkpoint["alternative"] is not None:
                alt_all[system] = checkpoint["alternative"]
            print(f"\n[{system}] resumed complete system checkpoint", flush=True)
            continue
        progress = (None if args.no_resume else
                    load_progress_checkpoint(progress_path, signature))
        if progress is None:
            progress = {"signature": signature, "stages": {}}

        def save_stage(name, value, complete):
            save_progress_stage(progress, progress_path, name, value, complete)

        def get_stage(name):
            return progress.get("stages", {}).get(name, {})

        print(f"\n[{system}] loading features...", flush=True)
        names, f, targets, rand_hash = load_features(system)
        positions = positions_from_features(system)
        p_wt = np.array(WT_POP[system], float)
        print(f"  n={len(names)} p_wt={p_wt} rand_hash={rand_hash[:12]}", flush=True)
        fixed_stage = get_stage("fixed_loo")
        if fixed_stage.get("complete"):
            fixed = fixed_stage.get("value", {})
        else:
            fixed = run_fixed_loo(
                system, names, f, targets, p_wt, COMBO_DEFS,
                n_seeds=args.seeds, resume=fixed_stage.get("value"),
                checkpoint=lambda value: save_stage("fixed_loo", value, False))
            save_stage("fixed_loo", fixed, True)
        nested_stage = get_stage("nested")
        nested = nested_stage.get("value") if nested_stage.get("complete") else run_nested(
            system, names, f, targets, p_wt, COMBO_DEFS,
            n_seeds=args.seeds, positions=positions,
            resume=nested_stage.get("value"),
            checkpoint=lambda value: save_stage("nested", value, False))
        save_stage("nested", nested, True)
        ctl_stage = get_stage("marker_control")
        ctl = ctl_stage.get("value") if ctl_stage.get("complete") else run_nested(
            system, names, f, targets, p_wt, COMBO_DEFS,
            control=True, n_seeds=args.seeds, positions=positions,
            resume=ctl_stage.get("value"),
            checkpoint=lambda value: save_stage("marker_control", value, False))
        save_stage("marker_control", ctl, True)
        alt = None
        if system == "src" and not args.skip_alt:
            alt_stage = get_stage("alternative")
            if alt_stage.get("complete"):
                alt = alt_stage.get("value")
            else:
                alt_panel = build_src_k3_panel(
                    SRC_K3_L410A_SUBSTITUTION_PROTOCOL_ID)
                t_alt = {k: np.array(alt_panel.targets[k], float) for k in targets}
                alt = run_nested(
                    system, names, f, t_alt, p_wt, COMBO_DEFS,
                    n_seeds=args.seeds, positions=positions,
                    resume=alt_stage.get("value"),
                    checkpoint=lambda value: save_stage("alternative", value, False))
                save_stage("alternative", alt, True)
            alt_all["src"] = alt
        lso_stage = get_stage("leave_site_out")
        if lso_stage.get("complete"):
            lso_value = lso_stage.get("value", {})
            lso = lso_value.get("groups", lso_value)
        else:
            lso_state = lso_stage.get("value")
            lso = selection_aware_lso(
                system, names, f, targets, p_wt, COMBO_DEFS,
                n_seeds=args.seeds, positions=positions, resume=lso_state,
                checkpoint=lambda value: save_stage("leave_site_out", value, False))
            save_stage("leave_site_out", {"groups": lso, "partial": {}}, True)
        lso_all[system] = lso
        g = gates(nested, ctl, lso, alt, system)
        block = build_result_json(system, names, f, targets, p_wt, fixed,
                                  nested, ctl, lso, alt, g, rand_hash,
                                  script_hash, adr_hashes, args.seeds, args.version,
                                  positions=positions)
        block["analyses"].update({"selection_route": {
            "combo_defs": [c[0] for c in COMBO_DEFS], "models": MODEL_NAMES,
            "tie_break": "mae -> dim -> model_simplicity -> combo_id"}})
        results["systems"][system] = block
        atomic_write_json(checkpoint_path, {
            "signature": signature,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "block": block,
            "leave_site_out": lso,
            "alternative": alt,
        })
        print(f"  wrote complete system checkpoint", flush=True)
        print(f"  nested={nested['nested_mae']:.4f} ctl={ctl['nested_mae']:.4f} "
              f"dir={nested['direction']}", flush=True)
        print(f"  gates={json.dumps(g)}", flush=True)
    # Headline outputs use the same atomic write as the checkpoints so an
    # interrupted final write cannot leave a truncated results file.
    atomic_write_json(RESULTS / "p2_k3_nested_results.json", results)
    atomic_write_json(RESULTS / "p2_k3_l410a_sensitivity.json", alt_all)
    atomic_write_json(RESULTS / "p2_k3_leave_site_out.json", lso_all)
    manifest = {"script": "p2_k3_eval_v2.py", "script_sha256": script_hash,
                "version": args.version,
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": str(_device()),
                 "params": {"K": K, "seeds": list(range(args.seeds)),
                            "epochs": NEPOCHS, "rank": 2, "hidden_dim": 32,
                            "random_projection_seed": RAND_SEED,
                            "tie_delta": TIE_DELTA, "ilr_eps": EPS,
                            "gp_protocol": PRIMARY_GP_PROTOCOL,
                            "src_label_protocols": {
                                "primary": SRC_K3_PRIMARY_PROTOCOL_ID,
                                "alternative": SRC_K3_L410A_SUBSTITUTION_PROTOCOL_ID,
                            }},
                "inputs": {"p2_llr_features.json": adr_hashes["canonical_data"],
                           "p2_site_deltas.npz": adr_hashes["feature_cache"],
                           "low_rank_cdst.py": adr_hashes["model_checkpoint"],
                           "adr_002": adr_hashes["adr_002"],
                           "adr_003": adr_hashes["adr_003"]},
                "outputs": ["p2_k3_nested_results.json",
                            "p2_k3_l410a_sensitivity.json",
                            "p2_k3_leave_site_out.json"]}
    (RESULTS / "p2_k3_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] p2_k3_v2 done in {time.time() - t0:.0f}s -> "
          f"{RESULTS / 'p2_k3_nested_results.json'}")


if __name__ == "__main__":
    main()
