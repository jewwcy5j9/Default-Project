"""P0-1 (iclr_improvement_plan_v2.md): full nested selection including
fold-local ESM-2 PCA as a retrospectively defined candidate at K=3.

Purpose
-------
The paper's strongest positive result (Abl1 fixed ESM-2 PCA MAE 0.1477,
t7_fold_local_esm_pca_v2.json) was obtained from a representation identified
after inspecting candidate encodings; it was never a candidate inside the
nested selector (t1 candidates: pos / LLR / Extended; p2 candidates: six
LLR/token/random combos). This script reruns nested LOO selection with a
retrospectively defined candidate set that includes fold-local ESM-2 PCA.

Candidate set (P0-1, all fold-local):
  pos      : position markers, 4 dims [pos/seq, one-hot marks]  (k3_data encoders;
             matches the frozen fixed MLP rows 0.2757 / 0.3213)
  ext      : extended physicochemical features, 10 dims (matches frozen
             fixed MLP rows 0.3003 / 0.3045)
  llr_pos  : [pos/seq, LLR scaled by max|LLR| over the training fold, one-hot
             marks], 5 dims (t1 pipeline; frozen fixed LLR 0.1629 used a
             global scale and is kept as reference)
  llr_only : LLR only, 1 dim, fold-local scale (ESM-2 control WITHOUT position
             identity, plan P0-1 item 5)
  pca20    : fold-local ESM-2 PCA on per-residue delta rows
             (all_delta_rows of esm2_encoding.compute_delta_encodings),
             d = min(20, n_train - 1), raw scores, no extra scaling
             (exactly the t7 construction; frozen fixed rows 0.1477 / 0.3006)

Models
  primary   : LowRankCDST (MLP) only, raw features (matches every frozen fixed
              row's model family). Features are NOT standardized in primary:
              this matches the k3_benchmark / t7 pipelines that produced the
              published fixed numbers.
  secondary : {CLR-Ridge, CLR-GP, LowRankCDST} with fold-local
              StandardScaler on every candidate (p2_k3_eval_v2 convention);
              selection runs over (candidate, model) pairs in the inner loop.

Selection rule:
  inner LOO over the outer training fold; per-candidate inner MAE =
  mean over inner folds; tie-break (inner MAE, candidate list order,
  [model simplicity]). Effective PCA dimension is recorded per fold but is
  not part of the tie-break.

Strict protocol (P0-1):
  per outer fold: PCA / LLR scale / scaler fit on training mutants only;
  inner LOO within the training fold; the selected (candidate, model) is
  refit on the full training fold and evaluated ONCE on the held-out mutant.

Reported blocks (P0-1 "必须报告的结果"):
  fixed rows per candidate (in-script, same construction as nested),
  training mean row, full nested MAE per variant, oracle fixed best,
  per-fold selection trace, selection frequency, per-mutation errors,
  direction, u1/u2 contrast, catastrophic flags, support in/out.

Frozen reference numbers (for verification only, not recomputed here):
  t7 fold-local PCA K=3: abl1 0.1477492904 / src 0.3005821130
  k3_benchmark MLP fixed: abl1 pos 0.2757 / ext 0.3003; src pos 0.3213 /
  ext 0.3045; k3_llr_proxy LLR 0.1629 (global scale)

Outputs:
  results/p2_k3_nested_pca_results.json  (full trace + aggregate)
  results/p2_k3_nested_pca_trace.csv     (per-fold selection trace)
  results/p2_k3_nested_pca_manifest.json (hashes, env, params)
  results/p2_k3_nested_pca_deltas.npz    (cache of delta vectors/rows;
                                           reused on rerun, hashes checked)

Usage:
  python p2_k3_nested_pca.py [--systems abl1,src] [--seeds 5]
                             [--skip-embeddings] [--keep-embeddings]
"""
import argparse
import hashlib
import json
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import esm2_encoding as E
from k3_data import (ABL1_K3, ABL1_K3_WT_POP, SRC_K3, SRC_K3_WT_POP,
                     enc_abl1_extended, enc_abl1_pos_markers,
                     enc_src_extended, enc_src_pos_markers)
from src.models.low_rank_cdst import LowRankCDST
from gp_protocols import PRIMARY_GP_PROTOCOL, make_primary_gp

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

K = 3
EPS = 1e-6
TIE_DELTA = 0.05
NEPOCHS = 800
PCA_D = 20
RAND_SEED = 0

CANDIDATES = ["pos", "ext", "llr_pos", "llr_only", "pca20"]
MODELS_PRIMARY = ["LowRankCDST"]
MODELS_SECONDARY = ["CLR-Ridge", "CLR-GP", "LowRankCDST"]
MODEL_SIMPLICITY = {m: i for i, m in enumerate(MODELS_SECONDARY)}
MARKER_MARKS = {"abl1": [290, 301, 382], "src": [311, 332, 380]}

ABL1_CORE = {m: ABL1_K3[m] for m in ("M290L", "L301I", "M290L_L301I",
                                     "F382L", "F382Y", "F382V")}
SRC_CORE = {m: SRC_K3[m] for m in SRC_K3 if m != "SrcKD-WT"}
WT_POP = {"abl1": ABL1_K3_WT_POP, "src": SRC_K3_WT_POP}

FROZEN_REFERENCE = {
    "abl1": {"t7_pca20": 0.1477492904, "bench_pos": 0.2757,
             "bench_ext": 0.3003, "llr_proxy": 0.1629,
             "training_mean": 0.2328888889},
    "src": {"t7_pca20": 0.3005821130, "bench_pos": 0.3213,
            "bench_ext": 0.3045, "training_mean": 0.2911},
}


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


def checkpoint_signature(system, args, script_hash, hashes, delta_cache_path):
    data_dir = HERE.parent.parent / "data" / "nmr_populations"
    dependencies = {
        "gp_protocols.py": HERE / "gp_protocols.py",
        "src_k3_labels.py": HERE.parent.parent / "src" / "data" / "src_k3_labels.py",
        "src_k3_canonical.csv": data_dir / "src_k3_canonical.csv",
    }
    return {
        "schema": "p2_k3_nested_pca_system_checkpoint_v1",
        "system": system,
        "script_sha256": script_hash,
        "version": args.version,
        "parameters": {
            "K": K, "seeds": args.seeds, "epochs": NEPOCHS,
            "pca_dim": PCA_D, "candidates": CANDIDATES,
            "models_primary": MODELS_PRIMARY,
            "models_secondary": MODELS_SECONDARY,
            "gp_protocol": PRIMARY_GP_PROTOCOL,
            "skip_embeddings": bool(args.skip_embeddings),
        },
        "inputs": {
            "data": hashes["data"],
            "modules": hashes["modules"],
            "dependencies": {name: sha256_file(path)
                             for name, path in dependencies.items()},
            "esm2_model": hashes["esm2_model"],
            "delta_cache_sha256": (
                sha256_file(delta_cache_path)
                if delta_cache_path is not None and delta_cache_path.exists()
                else None),
        },
        "environment": hashes["env"],
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
    if not {"block", "nested_mlp", "nested_model_select", "names"} <= payload.keys():
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


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clr(y):
    y = np.clip(np.asarray(y, float), EPS, 1.0)
    y = y / y.sum(axis=-1, keepdims=True)
    return np.log(y) - np.log(y).mean(axis=-1, keepdims=True)


def inv_clr(z):
    z = np.asarray(z, float)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    v = e / e.sum(axis=-1, keepdims=True)
    return v[0] if v.ndim == 2 and v.shape[0] == 1 else v


def _norm_simplex(p):
    p = np.asarray(p, float)
    return p / p.sum() if p.sum() > 0 else p


def ilr(y):
    p = np.clip(np.asarray(y, float), EPS, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    z1 = np.sqrt(2.0 / 3.0) * np.log(p[..., 0] / np.sqrt(p[..., 1] * p[..., 2]))
    z2 = (1.0 / np.sqrt(2.0)) * np.log(p[..., 1] / p[..., 2])
    return np.stack([z1, z2], axis=-1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features(system):
    """LLR / positions / targets for the core panel (same sources as p2)."""
    feat = json.loads((RESULTS / "p2_llr_features.json").read_text(encoding="utf-8"))
    core = ABL1_CORE if system == "abl1" else SRC_CORE
    names = list(core.keys())
    seq_len = {"abl1": len(E.ABL1_KD), "src": len(E.SRC_FULL)}[system]
    f = {}
    for m in names:
        f[m] = {"llr": float(feat["llr"][system][m]["llr"])}
    targets = {m: np.array(core[m]["pop"], float) for m in names}
    assert all(t.shape == (K,) for t in targets.values())
    return names, f, targets, seq_len


def positions_from_features(system):
    feat = json.loads((RESULTS / "p2_llr_features.json").read_text(encoding="utf-8"))
    core = ABL1_CORE if system == "abl1" else SRC_CORE
    pos = {}
    for m in core:
        ps = feat["llr"][system][m].get("positions", [])
        pos[m] = sorted({p.get("nominal_pos") for p in ps} - {None})
    return pos


def compute_embeddings(system, use_cache=True):
    """Per-residue delta rows + per-mutant site-summed delta vectors,
    exactly as t7 builds them (esm2_encoding path, local model cache)."""
    cache = RESULTS / f"p2_k3_nested_pca_deltas_{system}.npz"
    cache_meta = RESULTS / f"p2_k3_nested_pca_deltas_{system}_meta.json"
    manifest = json.loads((RESULTS / "p2_manifest.json").read_text(encoding="utf-8"))
    model_sha = manifest["model"]["sha256"]
    mutations = ABL1_CORE if system == "abl1" else SRC_CORE
    if use_cache and cache.exists() and cache_meta.exists():
        meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        if meta.get("model_sha256") == model_sha and meta.get("systems") == system:
            print(f"[{system}] reusing cached delta rows (model sha matches "
                  f"{model_sha[:12]}...)", flush=True)
            d = np.load(cache)
            return ([d[f"delta_vec::{system}::{m}"] for m in mutations.keys()],
                    d[f"delta_rows::{system}"])
    wt_seq = E.ABL1_KD if system == "abl1" else E.SRC_FULL
    model, tokenizer, device = E.load_esm2()
    pos_map = E.verify_positions(mutations, wt_seq, system)
    embeddings, mut_positions = E.compute_all_embeddings(
        model, tokenizer, wt_seq, mutations, pos_map, system, device)
    delta_vectors, all_delta_rows = E.compute_delta_encodings(embeddings,
                                                              mut_positions)
    del model, tokenizer
    if use_cache:
        names = list(mutations.keys())
        np.savez_compressed(
            cache,
            **{f"delta_vec::{system}::{names[i]}": delta_vectors[names[i]]
               for i in range(len(names))},
            **{f"delta_rows::{system}": all_delta_rows})
        (cache_meta).write_text(
            json.dumps({"model_sha256": model_sha, "systems": system,
                        "date": time.strftime("%Y-%m-%d %H:%M:%S")}),
            encoding="utf-8")
    return [delta_vectors[m] for m in mutations.keys()], all_delta_rows


def get_delta_rows(system, all_delta_rows_cache):
    """Return (delta_vectors_list, all_delta_rows) in names order, loading
    the cached npz if a previous run stored it."""
    if all_delta_rows_cache is not None:
        return all_delta_rows_cache
    cache = RESULTS / f"p2_k3_nested_pca_deltas_{system}.npz"
    if cache.exists():
        d = np.load(cache)
        core = ABL1_CORE if system == "abl1" else SRC_CORE
        vecs = [d[f"delta_vec::{system}::{m}"] for m in core.keys()]
        rows = d[f"delta_rows::{system}"]
        return vecs, rows
    raise FileNotFoundError(
        "delta cache missing; run without --skip-embeddings once")


def build_X(cand, system, names, f, seq_len, delta_vecs, delta_rows,
            rows, fit_rows):
    """Features for the listed rows (list of indices into names). For pca20
    the PCA is fit on delta_rows[fit_rows] (fold-local); for llr_* the scale
    is max|LLR| over fit_rows."""
    n = len(rows)
    marks = MARKER_MARKS[system]
    if cand == "pos":
        encs = (enc_abl1_pos_markers if system == "abl1" else enc_src_pos_markers)
        core = ABL1_CORE if system == "abl1" else SRC_CORE
        X = np.array([encs(names[i], core[names[i]]) for i in rows], float)
        return X, {"d": X.shape[1], "fit_ids": [names[i] for i in fit_rows]}
    if cand == "ext":
        encs = (enc_abl1_extended if system == "abl1" else enc_src_extended)
        core = ABL1_CORE if system == "abl1" else SRC_CORE
        X = np.array([encs(names[i], core[names[i]]) for i in rows], float)
        return X, {"d": X.shape[1], "fit_ids": [names[i] for i in fit_rows]}
    if cand in ("llr_only", "llr_pos"):
        scale = max(abs(f[names[i]]["llr"]) for i in fit_rows) or 1.0
        core = ABL1_CORE if system == "abl1" else SRC_CORE
        rows_l = []
        for i in rows:
            m = names[i]
            r = [f[m]["llr"] / scale]
            if cand == "llr_pos":
                r = [core[m]["pos"] / seq_len] + r
                r += [1.0 if core[m]["pos"] == p else 0.0 for p in marks]
            rows_l.append(r)
        X = np.array(rows_l, float)
        return X, {"d": X.shape[1], "llr_scale": float(scale),
                   "fit_ids": [names[i] for i in fit_rows]}
    if cand == "pca20":
        if delta_vecs is None or delta_rows is None:
            raise ValueError("pca20 candidate requires ESM-2 delta rows")
        d_eff = min(PCA_D, len(fit_rows) - 1)
        d_eff = max(d_eff, 1)
        seq_len = delta_rows.shape[1] and (delta_rows.shape[0] // len(delta_vecs))
        blocks = [delta_rows[i * seq_len:(i + 1) * seq_len]
                  for i in fit_rows]
        train_rows = np.vstack(blocks)
        pca = PCA(n_components=d_eff, random_state=0)
        pca.fit(train_rows)
        pca_hash = hashlib.sha256(
            pca.components_.tobytes() + pca.mean_.tobytes()).hexdigest()
        X = np.array([pca.transform(delta_vecs[i].reshape(1, -1))[0]
                      for i in rows], float)
        return X, {"d": X.shape[1], "d_eff": d_eff, "pca_hash": pca_hash,
                   "fit_ids": [names[i] for i in fit_rows]}
    raise KeyError(cand)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def train_mlp_seed(w_tr, c_tr, y_tr, w_te, c_te, d, seed_base, s, dev):
    torch.manual_seed(s * 100 + seed_base)
    np.random.seed(s * 100 + seed_base)
    model = LowRankCDST(K=K, intervention_dim=d, rank=2, hidden_dim=32).to(dev)
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


def predict_fold(cand, mname, system, names, f, seq_len, delta_vecs,
                 delta_rows, tr_i, te_i, targets, seed_base,
                 use_scaler=False, n_seeds=5):
    """Fold-local (candidate, model) prediction. Returns
    (mean_simplex, per_seed, meta)."""
    Xtr, meta = build_X(cand, system, names, f, seq_len, delta_vecs,
                        delta_rows, tr_i, tr_i)
    Xte, _ = build_X(cand, system, names, f, seq_len, delta_vecs,
                     delta_rows, te_i, tr_i)
    d = Xtr.shape[1]
    y_tr = np.array([targets[names[i]] for i in tr_i], float)
    w_tr = np.tile(np.asarray(WT_POP[system], float), (len(tr_i), 1))
    w_te = np.atleast_2d(np.asarray(WT_POP[system], float))
    dev = _device()
    if use_scaler:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        meta = dict(meta)
        meta["scaler_mean"] = scaler.mean_.tolist()
        meta["scaler_scale"] = scaler.scale_.tolist()
    if mname == "LowRankCDST":
        preds = [_norm_simplex(train_mlp_seed(w_tr, Xtr, y_tr, w_te, Xte, d,
                                              seed_base, s, dev))
                 for s in range(n_seeds)]
        return _norm_simplex(np.mean(preds, axis=0)), preds, meta
    if mname == "CLR-Ridge":
        m = Ridge(alpha=1.0)
        m.fit(Xtr, clr(y_tr))
        p = _norm_simplex(inv_clr(m.predict(Xte)))
        return p, [p.copy() for _ in range(n_seeds)], meta
    if mname == "CLR-GP":
        m = make_primary_gp()
        m.fit(Xtr, clr(y_tr))
        p = _norm_simplex(inv_clr(m.predict(Xte)))
        return p, [p.copy() for _ in range(n_seeds)], meta
    raise KeyError(mname)


def direction_report(preds, targets, p_wt):
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


def training_mean_row(names, targets):
    preds, errs = {}, {}
    for i, m in enumerate(names):
        mu = np.mean([targets[names[j]] for j in range(len(names)) if j != i],
                     axis=0)
        preds[m] = mu
        errs[m] = float(np.abs(mu - targets[m]).mean())
    return {"preds": preds, "errors": errs,
            "mae": float(np.mean(list(errs.values())))}


def catastrophic_flags(preds, targets, names, tm_errors, floor_zero=0.05):
    flags = {}
    for m in preds:
        e = float(np.abs(np.asarray(preds[m]) - np.asarray(targets[m])).mean())
        b = tm_errors[m]
        flags[m] = bool(e > 2.0 * b) if b > 0 else bool(e > floor_zero)
    return flags


def support_in_out(names, tr_of, positions):
    pos_list = [positions[m] for m in names]
    out = {}
    for m in tr_of:
        held_pos = set(pos_list[names.index(m)])
        tr_pos = set()
        for t in tr_of[m]:
            tr_pos |= set(pos_list[t])
        out[m] = "in" if held_pos <= tr_pos else "out"
    return out


# ---------------------------------------------------------------------------
# Fixed LOO (in-script references, same construction as nested)
# ---------------------------------------------------------------------------

def fixed_loo(system, names, f, seq_len, delta_vecs, delta_rows, targets,
              cand, mname, use_scaler=False, n_seeds=5, resume=None,
              checkpoint=None):
    state = dict(resume or {})
    preds = dict(state.get("preds", {}))
    folds = dict(state.get("folds", {}))
    for i, held in enumerate(names):
        if held in folds:
            continue
        tr_i = [j for j in range(len(names)) if j != i]
        p, per_seed, meta = predict_fold(
            cand, mname, system, names, f, seq_len, delta_vecs, delta_rows,
            tr_i, [i], targets, seed_base=i, use_scaler=use_scaler,
            n_seeds=n_seeds)
        preds[held] = p
        folds[held] = {"pred": np.round(p, 9).tolist(),
                       "per_seed": [np.round(v, 9).tolist() for v in per_seed],
                       "target": np.round(targets[held], 9).tolist(),
                       "meta": meta}
        if checkpoint is not None:
            checkpoint({"preds": preds, "folds": folds})
    errs = {m: float(np.abs(preds[m] - targets[m]).mean()) for m in preds}
    return {"preds": preds, "errors": errs,
            "mae": float(np.mean(list(errs.values()))),
            "direction": direction_report(preds, targets, WT_POP[system]),
            "folds": folds}


# ---------------------------------------------------------------------------
# Nested LOO with full selection
# ---------------------------------------------------------------------------

def tie_break(inner_scores, cands, models, use_model_sel):
    best = None
    for cand in cands:
        for mname in (models if use_model_sel else MODELS_PRIMARY):
            key = f"{cand}::{mname}"
            mae = float(np.mean(inner_scores[key]))
            if np.isnan(mae):
                continue
            candidate_order = CANDIDATES.index(cand)
            tup = (mae, candidate_order)
            if use_model_sel:
                tup = (mae, candidate_order, MODEL_SIMPLICITY[mname])
            if best is None or tup < best[0]:
                best = (tup, key)
    return best[1]


def run_nested(system, names, f, seq_len, delta_vecs, delta_rows, targets,
               cands, models, use_scaler=False, n_seeds=5, positions=None,
               resume=None, checkpoint=None):
    n = len(names)
    tgt = [targets[m] for m in names]
    msel = use_model_sel = len(models) > 1
    out = dict(resume or {})
    out.setdefault("folds", {})
    out.setdefault("preds", {})
    out.setdefault("per_fold_selected", {})
    out.setdefault("inner_scores", {})
    tr_of = {}
    for i, held in enumerate(names):
        tr_i = [j for j in range(n) if j != i]
        tr_of[held] = tr_i
        if held in out["folds"]:
            continue
        inner_scores = {f"{c}::{m}": [] for c in cands for m in models}
        inner_meta = {f"{c}::{m}": [] for c in cands for m in models}
        for j in tr_i:
            tr2 = [k for k in tr_i if k != j]
            for cand in cands:
                for mname in models:
                    p, _, meta = predict_fold(
                        cand, mname, system, names, f, seq_len, delta_vecs,
                        delta_rows, tr2, [j], targets, seed_base=i,
                        use_scaler=use_scaler, n_seeds=n_seeds)
                    inner_scores[f"{cand}::{mname}"].append(
                        float(np.abs(p - tgt[j]).mean()))
                    inner_meta[f"{cand}::{mname}"].append(
                        {"d_eff": meta.get("d_eff"), "d": meta.get("d"),
                         "pca_hash": meta.get("pca_hash"),
                         "llr_scale": meta.get("llr_scale")})
        best_key = tie_break(inner_scores, cands, models, msel)
        bcand, bmodel = best_key.split("::")
        p, per_seed, meta = predict_fold(
            bcand, bmodel, system, names, f, seq_len, delta_vecs, delta_rows,
            tr_i, [i], targets, seed_base=i, use_scaler=use_scaler,
            n_seeds=n_seeds)
        fold = {"holdout": held, "candidate": bcand, "model": bmodel,
                "inner_mae": float(np.mean(inner_scores[best_key])),
                "inner_scores": {k: [round(v, 6) for v in vals]
                                 for k, vals in inner_scores.items()},
                "inner_meta": inner_meta,
                "pred": np.round(p, 9).tolist(),
                "target": np.round(tgt[i], 9).tolist(),
                "per_seed_predictions": {
                    f"seed_{s}": np.round(np.asarray(per_seed[s]), 9).tolist()
                    for s in range(len(per_seed))},
                "meta": {k: v for k, v in meta.items()
                         if k in ("d", "d_eff", "pca_hash", "llr_scale",
                                  "scaler_mean", "scaler_scale")}}
        out["folds"][held] = fold
        out["preds"][held] = np.round(p, 9).tolist()
        out["per_fold_selected"][held] = f"{bcand}/{bmodel}"
        out["inner_scores"][held] = {k: [round(v, 6) for v in vals]
                                      for k, vals in inner_scores.items()}
        if checkpoint is not None:
            out["tr_of"] = {m: [names[j] for j in tr_of[m]] for m in tr_of}
            checkpoint(out)
    errs = {m: float(np.abs(np.asarray(out["preds"][m])
                            - np.asarray(targets[m])).mean()) for m in names}
    out["nested_mae"] = float(np.mean(list(errs.values())))
    out["errors"] = {k: round(v, 6) for k, v in errs.items()}
    out["direction"] = direction_report(out["preds"], targets, WT_POP[system])
    out["mae_per_mutant"] = out["errors"]
    out["tr_of"] = {m: [names[j] for j in tr_of[m]] for m in tr_of}
    sel_counts = {}
    for m in names:
        c = out["folds"][m]["candidate"]
        sel_counts[c] = sel_counts.get(c, 0) + 1
    out["selection_counts"] = sel_counts
    if positions is not None:
        out["support"] = support_in_out(names, tr_of, positions)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_system(system, args, all_delta_cache, progress=None, checkpoint=None):
    t0 = time.time()
    names, f, targets, seq_len = load_features(system)
    positions = positions_from_features(system)
    delta_vecs, delta_rows = None, None
    if "pca20" in CANDIDATES:
        if all_delta_cache is not None:
            delta_vecs, delta_rows = all_delta_cache
        else:
            vecs, rows = compute_embeddings(system, use_cache=True)
            delta_vecs = list(vecs)
            delta_rows = rows
            if delta_rows is None:
                delta_vecs, delta_rows = get_delta_rows(system, None)
    print(f"[{system}] n={len(names)} p_wt={WT_POP[system]}", flush=True)

    tm = training_mean_row(names, targets)
    progress = progress or {"stages": {}}

    def stage(name):
        return progress.get("stages", {}).get(name, {})

    def save(name, value, complete):
        if checkpoint is not None:
            checkpoint(name, value, complete)

    fixed = {}
    for cand in CANDIDATES:
        name = f"fixed_primary::{cand}"
        saved = stage(name)
        if saved.get("complete"):
            fixed[cand] = saved["value"]
        else:
            fixed[cand] = fixed_loo(
                system, names, f, seq_len, delta_vecs, delta_rows, targets,
                cand, "LowRankCDST", use_scaler=False, n_seeds=args.seeds,
                resume=saved.get("value"),
                checkpoint=lambda value, n=name: save(n, value, False))
            save(name, fixed[cand], True)
        print(f"[{system}] fixed {cand:<8} mae={fixed[cand]['mae']:.4f} "
              f"dir={fixed[cand]['direction']}", flush=True)
    oracle_cand = min(CANDIDATES, key=lambda c: fixed[c]["mae"])

    saved = stage("nested_primary")
    if saved.get("complete"):
        nested_mlp = saved["value"]
    else:
        nested_mlp = run_nested(
            system, names, f, seq_len, delta_vecs, delta_rows, targets,
            CANDIDATES, MODELS_PRIMARY, use_scaler=False,
            n_seeds=args.seeds, positions=positions,
            resume=saved.get("value"),
            checkpoint=lambda value: save("nested_primary", value, False))
        save("nested_primary", nested_mlp, True)
    print(f"[{system}] nested(MLP) mae={nested_mlp['nested_mae']:.4f} "
          f"dir={nested_mlp['direction']} counts={nested_mlp['selection_counts']}",
          flush=True)

    fixed_sec = {}
    for cand in CANDIDATES:
        for mname in MODELS_SECONDARY:
            key = f"{cand}::{mname}"
            name = f"fixed_secondary::{key}"
            saved = stage(name)
            if saved.get("complete"):
                fixed_sec[key] = saved["value"]
            else:
                fixed_sec[key] = fixed_loo(
                    system, names, f, seq_len, delta_vecs, delta_rows, targets,
                    cand, mname, use_scaler=True, n_seeds=args.seeds,
                    resume=saved.get("value"),
                    checkpoint=lambda value, n=name: save(n, value, False))
                save(name, fixed_sec[key], True)
    saved = stage("nested_secondary")
    if saved.get("complete"):
        nested_sec = saved["value"]
    else:
        nested_sec = run_nested(
            system, names, f, seq_len, delta_vecs, delta_rows, targets,
            CANDIDATES, MODELS_SECONDARY, use_scaler=True,
            n_seeds=args.seeds, positions=positions,
            resume=saved.get("value"),
            checkpoint=lambda value: save("nested_secondary", value, False))
        save("nested_secondary", nested_sec, True)
    print(f"[{system}] nested(model-sel) mae={nested_sec['nested_mae']:.4f} "
          f"dir={nested_sec['direction']} counts={nested_sec['selection_counts']}",
          flush=True)

    cat_mlp = catastrophic_flags(nested_mlp["preds"], targets, names,
                                 tm["errors"])
    cat_sec = catastrophic_flags(nested_sec["preds"], targets, names,
                                 tm["errors"])

    verification = {}
    ref = FROZEN_REFERENCE[system]
    cpu_repro = {"abl1": 0.1459441622, "src": 0.3026000000}[system]
    verification["t7_pca20_fixed"] = {
        "frozen_gpu_canonical": ref["t7_pca20"],
        "cpu_reproducible_t7_rerun": cpu_repro,
        "measured": fixed["pca20"]["mae"],
        "abs_diff_vs_gpu_canonical": abs(fixed["pca20"]["mae"] - ref["t7_pca20"]),
        "abs_diff_vs_cpu_repro": abs(fixed["pca20"]["mae"] - cpu_repro),
        "note": "GPU canonical (0.147749) and CPU rerun (0.145944) differ by "
                "0.0018: known env-dependent ESM-2 numerics, recorded in "
                "benchmark_registry.json (8/4 value 0.1459 not reproducible "
                "in GPU venv; R4 GPU rerun = 0.147749, canonical). This run is "
                "CPU; tol check uses the CPU-reproducible value."}
    verification["bench_pos_fixed"] = {"frozen": ref["bench_pos"],
                                       "measured": fixed["pos"]["mae"],
                                       "abs_diff": abs(fixed["pos"]["mae"] - ref["bench_pos"])}
    verification["bench_ext_fixed"] = {"frozen": ref["bench_ext"],
                                       "measured": fixed["ext"]["mae"],
                                       "abs_diff": abs(fixed["ext"]["mae"] - ref["bench_ext"])}
    verification["training_mean"] = {"frozen": ref["training_mean"],
                                     "measured": tm["mae"],
                                     "abs_diff": abs(tm["mae"] - ref["training_mean"])}
    if system == "abl1":
        verification["llr_proxy_fixed"] = {"frozen": ref["llr_proxy"],
                                           "measured": fixed["llr_pos"]["mae"],
                                           "note": "frozen LLR row used a global "
                                                   "scale; this row is fold-local"}
    def _tol_diffs(v):
        # t7 block keys its diffs vs GPU canonical / CPU repro; per its note
        # the tolerance check uses the CPU-reproducible value.
        if "abs_diff" in v:
            return [v["abs_diff"]]
        if "abs_diff_vs_cpu_repro" in v:
            return [v["abs_diff_vs_cpu_repro"]]
        return []

    verification["all_within_tol_1e-3"] = all(
        d <= 1e-3 for v in verification.values() for d in _tol_diffs(v))

    block = {
        "system": system,
        "n_mutants": len(names),
        "protocol": {
            "candidates": CANDIDATES,
            "candidate_dims": {c: fixed[c]["folds"][names[0]]["meta"].get("d")
                               for c in CANDIDATES},
            "models_primary": MODELS_PRIMARY,
            "models_secondary": MODELS_SECONDARY,
            "gp_protocol": PRIMARY_GP_PROTOCOL,
            "pca_dim_rule": f"d = min({PCA_D}, n_train - 1), fit on training "
                            "rows only, raw scores",
            "llr_scale_rule": "max|LLR| over training fold",
            "scaling_primary": "none (raw features; matches frozen fixed rows)",
            "scaling_secondary": "fold-local StandardScaler for all candidates",
            "selection": "inner LOO; tie-break (inner MAE, candidate list "
                          "order, [model simplicity]); effective dimension "
                          "is recorded but not used for tie-breaking",
            "seed_scheme": "seed = s*100 + outer_holdout_index",
            "n_seeds": args.seeds, "epochs": NEPOCHS,
            "outer_eval": "selected (candidate, model) refit on full training "
                          "fold; single evaluation on held-out mutant"},
        "training_mean": {"mae": tm["mae"],
                          "errors": {k: round(v, 6) for k, v in tm["errors"].items()}},
        "fixed_mlp_raw": {c: {"mae": fixed[c]["mae"],
                              "direction": fixed[c]["direction"],
                              "errors": {k: round(v, 6) for k, v in fixed[c]["errors"].items()}}
                          for c in CANDIDATES},
        "oracle_fixed_best": {"candidate": oracle_cand,
                              "mae": fixed[oracle_cand]["mae"]},
        "nested_mlp": {"mae": nested_mlp["nested_mae"],
                       "direction": nested_mlp["direction"],
                       "errors": nested_mlp["mae_per_mutant"],
                       "selection_counts": nested_mlp["selection_counts"],
                       "u1_u2_contrast_mae": u1_u2_contrast(nested_mlp["preds"], targets),
                       "catastrophic_folds": sorted(m for m, c in cat_mlp.items() if c),
                       "support": nested_mlp.get("support")},
        "nested_model_select": {"mae": nested_sec["nested_mae"],
                                "direction": nested_sec["direction"],
                                "errors": nested_sec["mae_per_mutant"],
                                "selection_counts": nested_sec["selection_counts"],
                                "u1_u2_contrast_mae": u1_u2_contrast(nested_sec["preds"], targets),
                                "catastrophic_folds": sorted(m for m, c in cat_sec.items() if c),
                                "support": nested_sec.get("support")},
        "verification": verification,
        "folds": {"nested_mlp": nested_mlp["folds"],
                  "nested_model_select": nested_sec["folds"],
                  "fixed_mlp_raw": {c: fixed[c]["folds"] for c in CANDIDATES}},
        "runtime_seconds": float(time.time() - t0),
    }
    return block, nested_mlp, nested_sec, names


def write_trace_csv(systems_data, path):
    import csv
    rows = []
    for system, names, nested_mlp, nested_sec in systems_data:
        for variant, nested in (("mlp", nested_mlp), ("model_select", nested_sec)):
            for m in names:
                fo = nested["folds"][m]
                rows.append({
                    "system": system, "variant": variant, "holdout": m,
                    "selected_candidate": fo["candidate"],
                    "selected_model": fo["model"],
                    "inner_mae": fo["inner_mae"],
                    "pred_active": fo["pred"][0], "pred_E1": fo["pred"][1],
                    "pred_E2": fo["pred"][2],
                    "target_active": fo["target"][0], "target_E1": fo["target"][1],
                    "target_E2": fo["target"][2],
                    "mae": nested["errors"].get(m),
                    "d_eff": (fo["meta"].get("d_eff")
                              if fo["meta"].get("d_eff") is not None
                              else fo["meta"].get("d")),
                    "pca_hash": fo["meta"].get("pca_hash", ""),
                    "support": nested.get("support", {}).get(m, ""),
                })
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="abl1,src")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--skip-embeddings", action="store_true",
                    help="reuse cached delta rows (requires a prior full run)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore compatible per-system checkpoints")
    ap.add_argument("--version", default="1.0.0")
    args = ap.parse_args()

    t0 = time.time()
    torch.set_num_threads(16)
    dev = str(_device())
    print(f"device: {dev} | systems: {args.systems} | seeds: {args.seeds}",
          flush=True)

    all_delta_cache = None
    if args.skip_embeddings:
        core_sets = {"abl1": ABL1_CORE, "src": SRC_CORE}
        vecs, rows = {}, {}
        for s in args.systems.split(","):
            cache = RESULTS / f"p2_k3_nested_pca_deltas_{s}.npz"
            if not cache.exists():
                raise SystemExit(f"--skip-embeddings requested but delta cache "
                                 f"missing for {s}")
            d = np.load(cache)
            vecs[s] = [d[f"delta_vec::{s}::{m}"] for m in core_sets[s].keys()]
            rows[s] = d[f"delta_rows::{s}"]
        all_delta_cache = {s: (vecs[s], rows[s])
                           for s in args.systems.split(",")}
        print("using cached ESM-2 delta rows", flush=True)

    script_hash = sha256_file(__file__)
    data_dir = HERE.parent.parent / "data" / "nmr_populations"
    hashes = {
        "script": {"p2_k3_nested_pca.py": script_hash},
        "data": {
            "xie2020_abl1_FINAL.json": sha256_file(data_dir / "xie2020_abl1_FINAL.json"),
            "cui2025_src_kinase.json": sha256_file(data_dir / "cui2025_src_kinase.json"),
            "p2_llr_features.json": sha256_file(RESULTS / "p2_llr_features.json"),
            "p2_site_deltas.npz": sha256_file(RESULTS / "p2_site_deltas.npz"),
        },
        "modules": {
            "esm2_encoding.py": sha256_file(HERE / "esm2_encoding.py"),
            "k3_data.py": sha256_file(HERE / "k3_data.py"),
            "low_rank_cdst.py": sha256_file(HERE.parent.parent / "src" / "models" / "low_rank_cdst.py"),
        },
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "numpy": np.__version__, "device": dev},
    }
    p2_manifest = json.loads((RESULTS / "p2_manifest.json").read_text(encoding="utf-8"))
    hashes["esm2_model"] = p2_manifest["model"]

    results = {"experiment": "P0-1 full nested selection with fold-local "
                             "ESM-2 PCA (iclr_improvement_plan_v2.md)",
               "version": args.version, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": dev, "K": K, "seed": RAND_SEED,
                "gp_protocol": PRIMARY_GP_PROTOCOL,
                "script_sha256": script_hash[:32], "hashes": hashes,
               "systems": {}}
    systems_data = []
    for system in args.systems.split(","):
        system = system.strip()
        cache = all_delta_cache.get(system) if all_delta_cache else None
        delta_cache_path = (RESULTS / f"p2_k3_nested_pca_deltas_{system}.npz"
                            if args.skip_embeddings else None)
        signature = checkpoint_signature(system, args, script_hash, hashes,
                                         delta_cache_path)
        checkpoint_path = RESULTS / f".p2_k3_nested_pca_{system}.checkpoint.json"
        progress_path = RESULTS / f".p2_k3_nested_pca_{system}.progress.json"
        checkpoint = (None if args.no_resume else
                      load_system_checkpoint(checkpoint_path, signature))
        if checkpoint is not None:
            block = checkpoint["block"]
            nested_mlp = checkpoint["nested_mlp"]
            nested_sec = checkpoint["nested_model_select"]
            names = checkpoint["names"]
            print(f"[{system}] resumed complete system checkpoint", flush=True)
        else:
            progress = (None if args.no_resume else
                        load_progress_checkpoint(progress_path, signature))
            if progress is None:
                progress = {"signature": signature, "stages": {}}

            def save_stage(name, value, complete):
                save_progress_stage(progress, progress_path, name, value, complete)

            block, nested_mlp, nested_sec, names = run_system(
                system, args, cache, progress=progress, checkpoint=save_stage)
            atomic_write_json(checkpoint_path, {
                "signature": signature,
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "block": block,
                "nested_mlp": nested_mlp,
                "nested_model_select": nested_sec,
                "names": names,
            })
            print(f"[{system}] wrote complete system checkpoint", flush=True)
        results["systems"][system] = block
        systems_data.append((system, names, nested_mlp, nested_sec))

    out_path = RESULTS / "p2_k3_nested_pca_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    write_trace_csv(systems_data, RESULTS / "p2_k3_nested_pca_trace.csv")
    manifest = {"script": "p2_k3_nested_pca.py", "version": args.version,
                "script_sha256": script_hash,
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "params": {"K": K, "seeds": list(range(args.seeds)),
                           "epochs": NEPOCHS, "pca_dim": PCA_D,
                            "candidates": CANDIDATES,
                            "models_primary": MODELS_PRIMARY,
                            "models_secondary": MODELS_SECONDARY,
                            "gp_protocol": PRIMARY_GP_PROTOCOL},
                "inputs": {k: v for k, v in hashes.items()},
                "outputs": ["p2_k3_nested_pca_results.json",
                            "p2_k3_nested_pca_trace.csv",
                            "p2_k3_nested_pca_deltas_{abl1,src}.npz"]}
    (RESULTS / "p2_k3_nested_pca_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] p2_k3_nested_pca done in {time.time() - t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
