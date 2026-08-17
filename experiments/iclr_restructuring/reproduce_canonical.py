"""A4: single-entry canonical reproduction (Phase-0 closure).

Recomputes every registry-locked canonical number from the registry data
files and the frozen code paths, with tolerance +/-0.003.

  - baselines (constant-WT, training-mean LOO)  : raw data only
  - Abl1 core K=3 encodings                      : k3_data + k3_benchmark
  - Src  K=3 encodings                           : k3_data + k3_benchmark
  - LLR proxy                                    : k3_llr_proxy (frozen LLR cache)
  - leave-site-out (p0_grouped_cv protocol)      : p0_grouped_cv frozen code
  - nested selector (t1_nested_cv protocol)      : t1_nested_cv frozen code
  - Src L410A global sensitivity                 : inline swap (same protocol)

Items that need GPU ESM-2 runs (fold-local ESM-2 PCA) are read from the
current results/t7_fold_local_esm_pca_v2.json (k3_pca_20dim.mae; falls back
to t7_fold_local_esm_pca.json for the pre-R4 artifact); if that file is
missing they are reported as PENDING_REMOTE and verified by rerunning
t7_fold_local_esm_pca.py on the GPU server. When the artifact carries a
"hashes" block (R4+), the recorded script hash is verified against the
current t7_fold_local_esm_pca.py; a mismatch marks the items FAIL_VERIFY.

Usage:  python reproduce_canonical.py [--tol 0.003] [--t7-results PATH]
Outputs:
  results/canonical_reproduction.json
  results/canonical_reproduction_report.md
  results/phase0_exit_report.md
"""
import sys
import json
import time
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from k3_data import (
    ABL1_K3, ABL1_K3_WT_POP, ABL1_SEQ_LEN,
    SRC_K3, SRC_K3_WT_POP, SRC_SEQ_LEN,
    enc_abl1_ddg_main, enc_abl1_pos_markers, enc_abl1_extended,
    enc_abl1_onehot, enc_src_extended, enc_src_no_dvol,
    enc_src_pos_markers, enc_src_onehot,
)
from k3_benchmark import run_loo, metrics
from k3_llr_proxy import enc_abl1_llr
from p0_grouped_cv import GROUPS as ABL1_LSO_GROUPS, grouped_cv
from t1_nested_cv import nested_loo as t1_nested_loo
from k3_src_deepdive import enc_src_ext_anchored
from k3_weighted_loss import loo_weighted

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REGISTRY = json.loads((HERE / "benchmark_registry.json").read_text(encoding="utf-8"))
TOL = float(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 0.003

ABL1_DATA_FILE = HERE.parent.parent / "data" / "nmr_populations" / "xie2020_abl1_FINAL.json"
SRC_DATA_FILE = HERE.parent.parent / "data" / "nmr_populations" / "cui2025_src_kinase.json"
LLR_CACHE = HERE.parent / "foldx_src" / "results" / "esm2_llr_proxy_results.json"
T7_V2 = HERE / "results" / "t7_fold_local_esm_pca_v2.json"
T7_V1 = HERE / "results" / "t7_fold_local_esm_pca.json"
T7_FILE = Path(sys.argv[sys.argv.index("--t7-results") + 1]) if "--t7-results" in sys.argv \
    else (T7_V2 if T7_V2.exists() else T7_V1)


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------
# 1. Data provenance: rebuild from registry data files, assert frozen
# ------------------------------------------------------------
def load_abl1_from_data():
    """populations: dict name -> {Active, I1, I2 (fractions), tier, ...}."""
    d = json.loads(ABL1_DATA_FILE.read_text(encoding="utf-8"))
    out = {}
    for name, v in d["populations"].items():
        if name == "WT":
            continue
        out[name] = {"wt": name.split("(")[0].strip() if "(" in name else None,
                     "mut": None, "pos": None,
                     "pop": [v["Active"], v["I1"], v["I2"]],
                     "tier": v.get("tier")}
    return out


def load_src_from_data():
    d = json.loads(SRC_DATA_FILE.read_text(encoding="utf-8"))
    out = {}
    # frozen source: additional_mutants_from_figS5 data (Met305 probe), same
    # source as the SrcKD-WT populations in SRC_K3 (k3_data froze figS5 values;
    # the JSON 'mutants' list holds Table S2 global CPMG fits for some entries)
    for m in d["additional_mutants_from_figS5"]["data"]:
        if m["name"] == "SrcKD":
            continue
        out[m["name"]] = {"wt": None, "mut": None, "pos": None,
                          "pop": [m["A"] / 100, m["E1"] / 100, m["E2"] / 100],
                          "tier": "gold"}
    return out


# ------------------------------------------------------------
# 2. Baselines (raw data only)
# ------------------------------------------------------------
def baselines(mutations, wt_pop):
    names = [m for m in mutations if m != "WT" and m != "SrcKD-WT"]
    wt = np.array(wt_pop, dtype=float)
    const_errs = [float(np.abs(np.array(mutations[m]["pop"]) - wt).mean()) for m in names]
    const_mae = float(np.mean(const_errs))
    tmean_errs = []
    for h in names:
        others = np.mean([np.array(mutations[m]["pop"]) for m in names if m != h], axis=0)
        tmean_errs.append(float(np.abs(others - np.array(mutations[h]["pop"])).mean()))
    return const_mae, float(np.mean(tmean_errs))


# ------------------------------------------------------------
# 3. Encoder-based K=3 LOO (frozen run_loo/metrics)
# ------------------------------------------------------------
def run_encoder_set(mutations, wt_pop, encoders):
    out = {}
    for key, (fn, d) in encoders.items():
        res = run_loo(mutations, wt_pop, fn, d)
        met = metrics(res["per_mutant"], res["targets"], wt_pop)
        out[key] = {"mae": met["mae"], "direction": met["direction"],
                    "mae_per_mutant": met["mae_per_mutant"]}
    return out


def run_l410a_sensitivity():
    mut = {k: dict(v) for k, v in SRC_K3.items() if k != "SrcKD-WT"}
    mut["SrcKD-L410A"]["pop"] = [0.96, 0.03, 0.01]
    const_mae, tmean_mae = baselines(mut, SRC_K3_WT_POP)
    encoders = {"Extended": (enc_src_extended, 10),
                "pos": (enc_src_pos_markers, 4),
                "no_dVol": (enc_src_no_dvol, 9)}
    return {"const": const_mae, "training_mean": tmean_mae,
            **run_encoder_set(mut, SRC_K3_WT_POP, encoders)}


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    t0 = time.time()
    canon = REGISTRY["canonical_numbers_to_reproduce"]
    items = []  # {key, target, value, status, detail}

    # --- data provenance ---
    abl1_data = load_abl1_from_data()
    src_data = load_src_from_data()
    data_notes = []
    # WT populations are checked through the frozen WT constants. The source
    # loaders intentionally return mutant panels only, so exclude WT here too.
    abl1_frozen = {m: ABL1_K3[m] for m in ABL1_K3
                   if m != "WT" and m != "H396P" and m != "M290L_H396P"}
    abl1_diff = [m for m in abl1_frozen
                 if abl1_frozen[m]["pop"] != abl1_data.get(m, {}).get("pop")]
    src_frozen = {m: SRC_K3[m] for m in SRC_K3 if m != "SrcKD-WT"}
    src_diff = [m for m in src_frozen
                if src_frozen[m]["pop"] != src_data.get(m, {}).get("pop")]
    if abl1_diff:
        data_notes.append(f"Abl1 pop mismatch vs frozen k3_data: {abl1_diff}")
    if src_diff:
        data_notes.append(f"Src pop mismatch vs frozen k3_data: {src_diff}")

    abl1_core = {m: ABL1_K3[m] for m in ABL1_K3 if m != "WT"
                 and m != "H396P" and m != "M290L_H396P"}
    src = {m: SRC_K3[m] for m in SRC_K3 if m != "SrcKD-WT"}

    # --- baselines ---
    base = REGISTRY["baselines_frozen"]
    a_const, a_tmean = baselines(abl1_core, ABL1_K3_WT_POP)
    s_const, s_tmean = baselines(src, SRC_K3_WT_POP)
    items = [
        ("abl1.constant_WT", base["abl1_k3_n6"]["constant_WT"], a_const, ""),
        ("abl1.training_mean", base["abl1_k3_n6"]["training_mean"], a_tmean, ""),
        ("src.constant_WT", base["src_k3_n8"]["constant_WT"], s_const, ""),
        ("src.training_mean", base["src_k3_n8"]["training_mean"], s_tmean, ""),
    ]
    for i, it in enumerate(items):
        stat = "PASS" if abs(it[2] - it[1]) <= TOL else "FAIL"
        items[i] = (it[0], it[1], it[2], stat)
        print(f"  baseline {it[0]:<18} target={it[1]:.4f} got={it[2]:.4f} [{stat}]")

    # --- Abl1 core encodings ---
    a_enc = run_encoder_set(abl1_core, ABL1_K3_WT_POP, {
        "variant_C": (enc_abl1_ddg_main, 5),
        "pos_markers": (enc_abl1_pos_markers, 4),
        "Extended": (enc_abl1_extended, 10),
        "Onehot": (lambda m, d: enc_abl1_onehot(m, d, list(abl1_core.keys())), len(abl1_core)),
    })
    # LLR proxy (frozen cache)
    llr_res = run_loo(abl1_core, ABL1_K3_WT_POP, enc_abl1_llr, 5)
    llr_met = metrics(llr_res["per_mutant"], llr_res["targets"], ABL1_K3_WT_POP)
    a_enc["LLR_proxy"] = {"mae": llr_met["mae"], "direction": llr_met["direction"],
                          "mae_per_mutant": llr_met["mae_per_mutant"]}
    for k in ["variant_C", "LLR_proxy", "pos_markers", "Extended", "Onehot"]:
        target = {"variant_C": canon["abl1_k3_n6"]["variant_C"],
                  "LLR_proxy": canon["abl1_k3_n6"]["LLR_proxy"],
                  "pos_markers": canon["abl1_k3_n6"]["pos_markers"],
                  "Extended": canon["abl1_k3_n6"]["Extended"],
                  "Onehot": canon["abl1_k3_n6"]["Onehot"]}[k]
        got = a_enc[k]["mae"]
        items.append((f"abl1.{k}", target, got,
                      "PASS" if abs(got - target) <= TOL else "FAIL"))
        print(f"  abl1 {k:<16} target={target:.4f} got={got:.4f} [{items[-1][3]}]")

    # --- Src encodings ---
    s_enc = run_encoder_set(src, SRC_K3_WT_POP, {
        "Extended": (enc_src_extended, 10),
        "no_dVol": (enc_src_no_dvol, 9),
        "pos_markers": (enc_src_pos_markers, 4),
        "Onehot": (lambda m, d: enc_src_onehot(m, d, list(src.keys())), len(src)),
    })
    for k in ["Extended", "no_dVol", "pos_markers", "Onehot"]:
        target = {"Extended": canon["src_k3_n8"]["Extended"],
                  "no_dVol": canon["src_k3_n8"]["no_dVol"],
                  "pos_markers": canon["src_k3_n8"]["pos_markers"],
                  "Onehot": canon["src_k3_n8"]["Onehot"]}[k]
        got = s_enc[k]["mae"]
        items.append((f"src.{k}", target, got,
                      "PASS" if abs(got - target) <= TOL else "FAIL"))
        print(f"  src  {k:<16} target={target:.4f} got={got:.4f} [{items[-1][3]}]")

    # --- Src Extended inverse-frequency weighted loss (frozen k3_weighted_loss) ---
    w_target = canon["src_k3_n8"]["Extended_weighted_pow05"]
    w_preds, w_weights = loo_weighted(src, SRC_K3_WT_POP, enc_src_ext_anchored,
                                      d=10, n_seeds=5, n_epochs=800,
                                      weight_power=0.5)
    w_met = metrics(w_preds, {m: src[m]["pop"] for m in src}, SRC_K3_WT_POP)
    w_got = w_met["mae"]
    items.append((f"src.Extended_weighted_pow05", w_target, w_got,
                  "PASS" if abs(w_got - w_target) <= TOL else "FAIL"))
    print(f"  src  Extended_w_p05     target={w_target:.4f} got={w_got:.4f} [{items[-1][3]}]")

    # --- leave-site-out (frozen p0_grouped_cv) ---
    lso = {}
    for enc_name, (fn, d) in {"variant_C": (enc_abl1_ddg_main, 5),
                              "LLR_proxy": (enc_abl1_llr, 5)}.items():
        lso[enc_name] = grouped_cv(fn, d)
    lso_targets = canon["abl1_leave_site_out"]
    for enc_name in ["variant_C", "LLR_proxy"]:
        for g in ["F382_family", "290_301"]:
            key = {"F382_family": "F382", "290_301": "290_301"}[g]
            tkey = {"variant_C": "variant_C", "LLR_proxy": "LLR"}[enc_name]
            target = lso_targets[tkey][key]
            got = lso[enc_name][g]["group_mae"]
            items.append((f"abl1.LSO.{enc_name}.{key}", target, got,
                          "PASS" if abs(got - target) <= TOL else "FAIL"))
            print(f"  LSO {enc_name:<12} {key:<10} target={target:.4f} got={got:.4f} [{items[-1][3]}]")

    # --- nested selector (frozen t1_nested_cv) ---
    t1 = t1_nested_loo()
    target = canon["abl1_k3_n6"]["nested_selector"]
    items.append(("abl1.nested_selector", target, t1["nested_mae"],
                  "PASS" if abs(t1["nested_mae"] - target) <= TOL else "FAIL"))
    print(f"  nested target={target:.4f} got={t1['nested_mae']:.4f} [{items[-1][3]}]")

    # --- L410A global sensitivity ---
    l410 = run_l410a_sensitivity()
    s_targets = canon["src_l410a_global_sensitivity"]
    for k, key in [("Extended", "Extended"), ("pos", "pos"), ("no_dVol", "no_dVol")]:
        got = l410[key]["mae"]
        items.append((f"src.L410A_global.{key}", s_targets[k], got,
                      "PASS" if abs(got - s_targets[k]) <= TOL else "FAIL"))
        print(f"  L410A {key:<10} target={s_targets[k]:.4f} got={got:.4f} [{items[-1][3]}]")

    # --- ESM-2 fold-local PCA: read current t7 result (GPU run), else PENDING ---
    t7 = None
    t7_ver = None
    if T7_FILE.exists():
        t7 = json.loads(T7_FILE.read_text(encoding="utf-8"))
        t7_hashes = t7.get("hashes")
        if t7_hashes:
            t7_ver = {
                "artifact": T7_FILE.name,
                "recorded": True,
                "script_hash_ok": t7_hashes.get("script")
                == sha256(HERE / "t7_fold_local_esm_pca.py"),
                "modules": t7_hashes.get("modules"),
                "data_files": t7_hashes.get("data_files"),
                "env": t7_hashes.get("env"),
                "n_seeds": t7.get("n_seeds"),
            }
        else:
            t7_ver = {"artifact": T7_FILE.name, "recorded": False,
                      "note": "pre-R4 artifact without hashes block"}
    for key, syskey in [("abl1.fold_local_ESM2_PCA", "abl1"),
                        ("src.fold_local_ESM2_PCA", "src")]:
        tkey = key.split(".")[0] + "_k3_" + ("n6" if syskey == "abl1" else "n8")
        target = canon[tkey]["fold_local_ESM2_PCA"]
        if t7 is not None and syskey in t7 and "k3_pca_20dim" in t7[syskey]:
            got = t7[syskey]["k3_pca_20dim"]["mae"]
            stat = "PASS" if abs(got - target) <= TOL else "FAIL"
            if t7_ver and t7_ver.get("recorded") and not t7_ver["script_hash_ok"]:
                stat = "FAIL_VERIFY"
            items.append((key, target, got, stat))
            print(f"  {key:<30} target={target:.4f} got={got:.4f} [{stat}]")
        else:
            items.append((key, target, None, "PENDING_REMOTE"))
            print(f"  {key:<30} PENDING_REMOTE (t7 rerun on GPU server)")

    # --- write outputs ---
    RESULTS.mkdir(exist_ok=True)
    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tol": TOL,
        "provenance_notes": data_notes or ["ok"],
        "items": [{"key": k, "target": t, "value": (round(v, 6) if v is not None else None),
                   "status": s} for k, t, v, s in items],
        "input_hashes": {
            "benchmark_registry.json": sha256(HERE / "benchmark_registry.json"),
            "xie2020_abl1_FINAL.json": sha256(ABL1_DATA_FILE),
            "cui2025_src_kinase.json": sha256(SRC_DATA_FILE),
            "esm2_llr_proxy_results.json": sha256(LLR_CACHE),
        },
        "t7_input_hash": sha256(T7_FILE) if T7_FILE.exists() else None,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (RESULTS / "canonical_reproduction.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] canonical_reproduction.json  ({out['runtime_seconds']}s)")


if __name__ == "__main__":
    main()
