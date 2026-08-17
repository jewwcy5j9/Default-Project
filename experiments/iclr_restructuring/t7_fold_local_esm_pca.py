"""T7: Fold-local ESM-2 PCA rerun for pooled K=2 and primary K=3 targets.

The original ESM-2 PCA rows were fit transductively:
PCA was fit on the full-panel delta rows, then held-out mutants were projected.
This script fits PCA only on each outer training fold and evaluates both the
pooled K=2 targets and the primary K=3 population vectors.

Design (NEXT_TIER_EXECUTION_PLAN.md, P2):
- dims frozen before running (primary comparison at d=20, the paper's row);
- min(d, n_train-1) cap when training fold cannot support d;
- rank / condition-number diagnostics per fold;
- transductive numbers kept in the JSON as reference.

R4 (NEXT_PHASE_EXECUTION_PLAN.md): per-seed predictions are saved for
every outer fold; PCA fit IDs and a verifiable components+mean hash are
recorded per fold; script/module/data/env hashes are recorded at the top
level; output defaults to t7_fold_local_esm_pca_v2.json (v1 is kept as
archived artifact).

Output: results/t7_fold_local_esm_pca_v2.json
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

import esm2_encoding as E
from k3_data import ABL1_K3, ABL1_K3_WT_POP, SRC_K3, SRC_K3_WT_POP

OUT = Path(__file__).resolve().parent / "results"
PCA_DIMS = [10, 15, 20]
N_SEEDS = int(sys.argv[sys.argv.index("--n-seeds") + 1]) if "--n-seeds" in sys.argv else 5
OUT_NAME = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv \
    else "t7_fold_local_esm_pca_v2.json"


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None


def fold_train_predict(w_wt_train, c_train, y_train, c_test, d, hold_out,
                       n_seeds=None, n_epochs=800):
    """One fold: train LowRankCDST(K=2) on train, predict test (best-state).

    Returns a list of per-seed scalar predictions (len n_seeds).
    """
    n_seeds = N_SEEDS if n_seeds is None else n_seeds
    preds = []
    for seed in range(n_seeds):
        torch.manual_seed(seed * 100 + hold_out)
        model = E.LowRankCDST(K=2, intervention_dim=d, rank=2, hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
        w_t = torch.FloatTensor(w_wt_train)
        c_t = torch.FloatTensor(c_train)
        y_t = torch.FloatTensor(y_train)
        best_loss = float("inf")
        best_state = None
        for _ in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            loss = F.mse_loss(model(w_t, c_t), y_t)
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            p = model(torch.FloatTensor(w_wt_train[:1]),
                      torch.FloatTensor(np.atleast_2d(c_test))).numpy()[0, 1]
        preds.append(float(p))
    return preds


def fold_train_predict_k3(wt, c_train, y_train, c_test, d, hold_out,
                          n_seeds=None, n_epochs=800):
    """One fold: train LowRankCDST(K=3) on train, predict test.

    Returns a list of per-seed K=3 prediction vectors (len n_seeds).
    """
    n_seeds = N_SEEDS if n_seeds is None else n_seeds
    preds = []
    w_train = np.tile(wt, (len(c_train), 1))
    for seed in range(n_seeds):
        torch.manual_seed(seed * 100 + hold_out)
        model = E.LowRankCDST(K=3, intervention_dim=d, rank=2, hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
        w_t = torch.FloatTensor(w_train)
        c_t = torch.FloatTensor(c_train)
        y_t = torch.FloatTensor(y_train)
        best_loss = float("inf")
        best_state = None
        for _ in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            loss = F.mse_loss(model(w_t, c_t), y_t)
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred = model(torch.FloatTensor(np.atleast_2d(wt)),
                         torch.FloatTensor(np.atleast_2d(c_test))).numpy()[0]
        preds.append(pred)
    return preds


def run_fold_local(system_name, mutations, target_key, wt_non_target, wt_seq,
                   system, k3_data, k3_wt):
    print(f"\n[{system_name}] loading ESM-2 and computing embeddings...")
    model, tokenizer, device = E.load_esm2()
    pos_map = E.verify_positions(mutations, wt_seq, system)
    embeddings, mut_positions = E.compute_all_embeddings(
        model, tokenizer, wt_seq, mutations, pos_map, system, device)
    delta_vectors, all_delta_rows = E.compute_delta_encodings(embeddings, mut_positions)
    del model, tokenizer

    names = list(mutations.keys())
    wt_dist = np.array([1 - wt_non_target, wt_non_target])
    w_wt = np.tile(wt_dist, (len(names), 1))
    w_target = np.array([[1 - mutations[m][target_key], mutations[m][target_key]]
                         for m in names])

    # per-mutant delta matrices: all_delta_rows is (n_mutants*seq_len, 1280),
    # one block of seq_len rows per mutant in `names` order.
    seq_len = embeddings['WT'].shape[0]

    # (dead transductive-reference block removed 2026-08-17: pca_t.fit on the
    #  full panel and ref_reduced were computed but never used downstream)

    results = {}
    k3_result = {"per_mutant_pred": {}, "errors": {}, "folds": {}}
    for d in PCA_DIMS:
        fold_detail = {}
        diag = {}
        for held in names:
            tr = [m for m in names if m != held]
            d_eff = min(d, len(tr) - 1)
            tr_idx = [names.index(m) for m in tr]
            train_rows = np.vstack([all_delta_rows[i * seq_len:(i + 1) * seq_len]
                                    for i in tr_idx])
            pca = PCA(n_components=d_eff, random_state=0)
            pca.fit(train_rows)
            pca_hash = hashlib.sha256(
                pca.components_.tobytes() + pca.mean_.tobytes()).hexdigest()
            c_train = np.array([pca.transform(delta_vectors[m].reshape(1, -1))[0]
                                for m in tr])
            c_test = pca.transform(delta_vectors[held].reshape(1, -1))[0]
            y_train = w_target[tr_idx]
            preds = fold_train_predict(
                w_wt[tr_idx], c_train, y_train, c_test, d_eff,
                names.index(held))
            fold_detail[held] = {
                "per_seed": preds,
                "seed_mean": float(np.mean(preds)),
                "target": float(mutations[held][target_key]),
                "fit_ids": tr,
                "pca_hash": pca_hash,
                "n_seeds": N_SEEDS,
            }
            if d == 20:
                y_train_k3 = np.array([k3_data[m]["pop"] for m in tr])
                pred_k3_list = fold_train_predict_k3(
                    np.array(k3_wt), c_train, y_train_k3, c_test, d_eff,
                    names.index(held))
                k3_result["per_mutant_pred"][held] = (
                    np.mean(pred_k3_list, axis=0)).tolist()
                k3_result["errors"][held] = float(
                    np.abs(np.mean(pred_k3_list, axis=0)
                           - np.array(k3_data[held]["pop"])).mean())
                k3_result["folds"][held] = {
                    "per_seed": [v.tolist() for v in pred_k3_list],
                    "seed_mean": k3_result["per_mutant_pred"][held],
                    "target": [float(x) for x in k3_data[held]["pop"]],
                    "fit_ids": tr,
                    "pca_hash": pca_hash,
                    "n_seeds": N_SEEDS,
                }
            ev = pca.explained_variance_
            diag[held] = {
                "d_effective": d_eff,
                "cumulative_variance": float(np.cumsum(pca.explained_variance_ratio_)[-1]),
                "condition_number": float(np.max(ev) / np.min(ev)) if ev.min() > 0 else None,
            }
        per_mutant = {m: fold_detail[m]["seed_mean"] for m in names}
        errs = {m: abs(per_mutant[m] - mutations[m][target_key]) for m in names}
        dir_ok = dir_tot = 0
        dir_detail = {}
        for m in names:
            d_true = mutations[m][target_key] - wt_non_target
            d_pred = per_mutant[m] - wt_non_target
            if abs(d_true) < 0.05:
                dir_detail[m] = 'TIE'
                continue
            dir_tot += 1
            if np.sign(d_true) == np.sign(d_pred):
                dir_ok += 1
                dir_detail[m] = 'OK'
            else:
                dir_detail[m] = 'WRONG'
        # transductive reference for the same dim (fit on full panel)
        results[f"pca_{d}dim"] = {
            "d": d, "d_effective_per_fold": diag,
            "per_mutant_pred": per_mutant,
            "errors": {m: float(e) for m, e in errs.items()},
            "mae": float(np.mean(list(errs.values()))),
            "median": float(np.median(list(errs.values()))),
            "direction": f"{dir_ok}/{dir_tot}",
            "direction_detail": dir_detail,
            "folds": fold_detail,
        }
        print(f"  {system_name} pca{d}: MAE={results[f'pca_{d}dim']['mae']:.4f} "
              f"dir={dir_ok}/{dir_tot}")
    k3_result["mae"] = float(np.mean(list(k3_result["errors"].values())))
    print(f"  {system_name} K=3 pca20: MAE={k3_result['mae']:.4f}")
    return {"system": system_name, "results": results,
            "k3_pca_20dim": k3_result,
            "n_mutants": len(names)}


def main():
    t0 = time.time()
    print("=" * 90)
    print("T7: fold-local ESM-2 PCA (protocol cleanup, R4 per-seed)")
    print("=" * 90)
    out = {"protocol": "PCA fit per outer LOO fold on training mutants only; "
                       "dims frozen [10,15,20]; primary d=20; "
                       f"seed scheme: seed = s*100 + holdout_index, n_seeds={N_SEEDS}; "
                       "per-fold per-seed predictions, seed means, targets, "
                       "PCA fit IDs and components+mean hashes recorded"}
    ref = json.loads((OUT / "esm2_encoding.json").read_text(encoding="utf-8"))["results"]
    out["transductive_reference"] = {
        sysn: {k: v["mae"] for k, v in ref[sysn]["results"].items()}
        for sysn in ["abl1", "src"]}
    for sys_name, mutations, target_key, wt_non_target, wt_seq, system, k3_data, k3_wt in [
            ("Abl1", E.ABL1_DATA, 'non_ground', E.ABL1_WT_NON_GROUND,
             E.ABL1_KD, 'abl1', ABL1_K3, ABL1_K3_WT_POP),
            ("Src", E.SRC_DATA, 'non_active', E.SRC_WT_NON_ACTIVE,
             E.SRC_FULL, 'src', SRC_K3, SRC_K3_WT_POP)]:
        out[sys_name.lower()] = run_fold_local(
            sys_name, mutations, target_key, wt_non_target, wt_seq, system,
            k3_data, k3_wt)
    out["runtime_seconds"] = float(time.time() - t0)
    out["n_seeds"] = N_SEEDS
    here = Path(__file__).resolve().parent
    data_dir = here.parent.parent / "data" / "nmr_populations"
    out["hashes"] = {
        "script": sha256_file(here / "t7_fold_local_esm_pca.py"),
        "modules": {
            "esm2_encoding.py": sha256_file(here / "esm2_encoding.py"),
            "k3_data.py": sha256_file(here / "k3_data.py"),
        },
        "data_files": {
            "esm2_encoding.json": sha256_file(OUT / "esm2_encoding.json"),
            "xie2020_abl1_FINAL.json": sha256_file(data_dir / "xie2020_abl1_FINAL.json"),
            "cui2025_src_kinase.json": sha256_file(data_dir / "cui2025_src_kinase.json"),
        },
        "env": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "device": ("cuda:" + torch.cuda.get_device_name(0))
                      if torch.cuda.is_available() else "cpu",
        },
    }
    out_path = OUT / OUT_NAME
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
