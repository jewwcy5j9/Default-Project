"""Src K=3 L410A sensitivity: replace probe value with global CPMG fit.

L410A probe (Fig S5 Met305) = [0.73, 0.27, 0.0]; global (Table S2) = [0.96, 0.03, 0.01].
Reruns the full Src K=3 LOO benchmark with the global value and compares
against constant-WT / training-mean baselines.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

import k3_data
from k3_benchmark import run_loo, metrics

OUT = Path(__file__).resolve().parent / "results"

SRC_GLOBAL = {
    "SrcKD-WT": [0.72, 0.07, 0.21],
    "SrcKD-L410A": [0.96, 0.03, 0.01],
    "SrcKD-V332I": [0.48, 0.52, 0.0],
    "SrcKD-L270F_V332I": [0.09, 0.91, 0.0],
    "SrcKD-L325A": [0.0, 1.0, 0.0],
    "SrcKD-A311I": [0.0, 1.0, 0.0],
    "SrcKD-V380A": [0.0, 0.62, 0.38],
    "SrcKD-V331A": [0.0, 0.45, 0.55],
    "SrcKD-F405A": [0.0, 0.16, 0.84],
}

def baselines(mutations, wt_pop):
    names = list(mutations.keys())
    pops = {m: np.asarray(mutations[m]["pop"]) for m in names}
    wt = np.asarray(wt_pop)
    const = [np.abs(wt - pops[m]).mean() for m in names]
    tmean = {}
    for held in names:
        others = [pops[m] for m in names if m != held]
        tmean[held] = np.abs(np.mean(others, axis=0) - pops[held]).mean()
    return float(np.mean(const)), float(np.mean(list(tmean.values())))

mut = {k: dict(v) for k, v in k3_data.SRC_K3.items() if k != "SrcKD-WT"}
# FIXED 2026-08-17: this loop previously assigned SRC_GLOBAL values to ALL
# mutants while the docstring/labels say L410A-only. Verified that SRC_GLOBAL
# equals k3_data.SRC_K3 for every mutant except SrcKD-L410A, so the old
# whole-panel assignment was a no-op for the others; restricting the
# substitution to L410A preserves the meaning of the stored
# results/src_k3_l410a_sensitivity.json exactly (keys L410A_global_*).
mut["SrcKD-L410A"]["pop"] = SRC_GLOBAL["SrcKD-L410A"]
const_mae, tmean_mae = baselines(mut, k3_data.SRC_K3_WT_POP)

results = {"L410A_global_const": const_mae, "L410A_global_trainmean": tmean_mae}

encodings = {
    "Extended_10dim": k3_data.enc_src_extended,
    "pos_markers_4dim": k3_data.enc_src_pos_markers,
    "no_dVol_9dim": k3_data.enc_src_no_dvol,
}
# (leftover debug print of dir(k3_data) removed 2026-08-17)
for name, enc in encodings.items():
    dim = len(enc("SrcKD-L410A", mut["SrcKD-L410A"]))
    preds = run_loo(mut, k3_data.SRC_K3_WT_POP, enc, d=dim, n_seeds=5)
    m = metrics(preds["per_mutant"], {k: v["pop"] for k, v in mut.items()},
                k3_data.SRC_K3_WT_POP)
    results[name] = {"mae": m["mae"], "direction": m["direction"]}
    print(f"{name:<20} MAE {m['mae']:.4f}  dir {m['direction']}")

print(f"baselines: constant-WT {const_mae:.4f} | training-mean {tmean_mae:.4f}")
(OUT / "src_k3_l410a_sensitivity.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print("[OK] src_k3_l410a_sensitivity.json written")
