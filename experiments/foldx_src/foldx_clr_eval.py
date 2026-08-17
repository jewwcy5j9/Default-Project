"""C4.2-4: FoldX features into the frozen K=3 CLR protocol.

Per-mutant feature vector = per-state FoldX ddG (kcal/mol, FoldX-measured
mutation effect on each state's reference structure):
  Abl1: [ddg_active, ddg_i1, ddg_i2]
  Src:  [ddg_active, ddg_e1, ddg_e2]
Optional + frozen raw LLR (C4 step 4).

Protocol reuses p2_k3_eval protocol exactly: fixed LOO at K=3, all indices
immutable, fold-local StandardScaler, CLR-Ridge(alpha=1.0)/CLR-GP(RBF+WhiteK)
/SimpleCDST(K=3)/LowRankCDST(K=3,rank=2,hidden=32), 5 seeds, 800 epochs,
seed = s*100 + holdout index.

Track-2 caveat: for degenerate cells (M290L, M290L_L301I on i2) FoldX
returns a constant 0.0 (no recovery), carrying no information about i2.  For
track 2 we therefore use only [ddg_active, ddg_i1] (2-dim) and report why.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "iclr_restructuring"))

from p2_k3_eval import (MODEL_NAMES, leave_site_out, run_fixed_loo, run_nested)
from k3_data import ABL1_K3, SRC_K3, ABL1_K3_WT_POP, SRC_K3_WT_POP

RESULTS = HERE / "results"
STATES = {"abl1": ["active", "i1", "i2"], "src": ["active", "e1", "e2"]}
ABL1_CORE = [m for m in ABL1_K3
             if m not in ("WT", "H396P", "M290L_H396P")]
# n=6 canonical set (same as p2_k3_eval / constant_baselines); H396P and
# M290L_H396P are silver-tier without experimental ddG / LLR features.
SRC_CORE = [m for m in SRC_K3 if m != "SrcKD-WT"]  # matches p2_k3 SRC_CORE
TARGETS = {"abl1": {m: np.array(ABL1_K3[m]["pop"], float) for m in ABL1_CORE},
           "src": {m: np.array(SRC_K3[m]["pop"], float) for m in SRC_CORE}}
WT_POP = {"abl1": ABL1_K3_WT_POP, "src": SRC_K3_WT_POP}
DEGENERATE_I2 = {"M290L", "M290L_L301I"}


def load_llr():
    p = RESULTS / "esm2_llr_proxy_results.json"
    return json.loads(p.read_text(encoding="utf-8"))


def short_name(m):
    return m.split("-")[-1]


def build_feats(system, ddg_path, llr=None, dims=None):
    cells = {(c["mutation"], c["state"]): c for c in json.loads(
        ddg_path.read_text(encoding="utf-8"))["per_cell"]}
    st = STATES[system]
    names = ABL1_CORE if system == "abl1" else SRC_CORE
    f = {}
    for m in names:
        sn = short_name(m)
        dd = []
        for s in st:
            c = cells.get((sn, s), {})
            if c.get("degenerate"):
                continue
            v = c.get("mean_ddg")
            dd.append(float(v) if v is not None else np.nan)
        if dims is not None:
            dd = dd[:dims]
        f[m] = {"ddg": np.array(dd, float)}
        if llr is not None:
            table = llr[system]["llr"]
            f[m]["llr"] = float(table[m])
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=["abl1", "src"])
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--with-llr", action="store_true",
                    help="C4.4: FoldX + frozen raw LLR")
    ap.add_argument("--out", default=str(RESULTS / "foldx_clr_eval.json"))
    args = ap.parse_args()

    llr = load_llr() if args.with_llr else None
    combo_basic = [("FX_ddg", ["ddg"])]
    combo_with_llr = combo_basic + [("FX_ddg_llr", ["ddg", "llr"])]
    report = {"schema": "foldx-clr-k3",
              "combo_models": MODEL_NAMES,
              "track2_note": "degenerate i2 cells (M290L, M290L_L301I) drop the "
                             "i2 dimension (FoldX returns constant 0.0)",
              "per_system": {}}

    for system in args.systems:
        paths = []
        if system == "abl1":
            paths = [("track1", RESULTS / "foldx_abl1_ddg_track1.json"),
                     ("track2", RESULTS / "foldx_abl1_ddg_track2.json")]
        else:
            paths = [("src", RESULTS / "foldx_src_ddg.json")]
        sys_out = {}
        for label, path in paths:
            f = build_feats(system, path, llr,
                            dims=None if system == "src" or label == "track1" else 2)
            names = ABL1_CORE if system == "abl1" else SRC_CORE
            tgt = TARGETS[system]
            p_wt = WT_POP[system]
            combos = combo_with_llr if llr else combo_basic
            fixed = run_fixed_loo(system, names, f, tgt, p_wt, combos,
                                  n_seeds=args.n_seeds)
            lso = leave_site_out(system, names, f, tgt, p_wt, combos,
                                 n_seeds=args.n_seeds)
            sys_out[label] = {"fixed_loo": fixed, "lso": lso}
        report["per_system"][system] = sys_out

    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[OK] {out_path}")

    for system, sys_map in report["per_system"].items():
        print(f"\n=== {system} ===")
        for label, r in sys_map.items():
            if not r["fixed_loo"]:
                print(f"  {label}: empty")
                continue
            best_key = min(r["fixed_loo"],
                           key=lambda k: r["fixed_loo"][k]["mae"])
            best = r["fixed_loo"][best_key]
            print(f"  {label}: best fixed-LOO {best['mae']:.4f} ({best_key})")
            for k, v in sorted(r["fixed_loo"].items(),
                               key=lambda kv: kv[1]["mae"]):
                print(f"     {k}: {v['mae']:.4f}")


if __name__ == "__main__":
    main()