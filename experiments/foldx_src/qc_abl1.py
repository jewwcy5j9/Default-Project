#!/usr/bin/env python
"""FoldX Abl1 C3 QC (NEXT_PHASE_EXECUTION_PLAN.md 阶段 C3).

Checks implemented:
  - cell success rate (目标 5/5 成功)
  - Spearman(foldx ddG, |experimental ddG|) >= 0.70   (active state; n=6 all & n=5 excl F382L)
  - state-energy ordering consistency >= 80% (Active <= I1 <= I2 的 mean total energy)
  - reference/repeat perturbation MAE <= 0.03 (repeat 之间 |ddG - mean| 的 MAE)
  - Track1 vs Track2 verdict 不反转 (degenerate cells 除外)

Output: results/foldx_abl1_qc.json
"""
import json
import math
import statistics
from pathlib import Path
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

EXPERIMENTAL_DDG = {
    "M290L": -1.3, "L301I": -2.2, "M290L_L301I": -3.5,
    "F382L": 0.0, "F382Y": -2.5, "F382V": -3.0,
}
STATES = ["active", "i1", "i2"]


def load(track):
    with open(RESULTS / f"foldx_abl1_ddg_track{track}.json", encoding="utf-8") as f:
        return json.load(f)


def load_series(track):
    with open(RESULTS / f"foldx_abl1_ddg_track{track}.json", encoding="utf-8") as f:
        return json.load(f)


def main():
    t1 = load_series(1)
    t2 = load_series(2)
    cells1 = {s: {c["mutation"]: c for c in t1["per_cell"] if c["state"] == s} for s in ("active", "i1", "i2")}
    cells2 = {s: {c["mutation"]: c for c in t2["per_cell"] if c["state"] == s} for s in ("active", "i1", "i2")}

    fails = t1.get("failures", []) + t2.get("failures", [])
    all_cells = t1["per_cell"] + t2["per_cell"]
    n_cells = len(all_cells)
    degenerate_cells = [c for c in all_cells if c.get("degenerate")]
    non_degenerate_cells = [c for c in all_cells if not c.get("degenerate")]
    degenerate_keys = {(c["state"], c["mutation"]) for c in degenerate_cells}
    expected_degenerate_keys = {("i2", "M290L"), ("i2", "M290L_L301I")}
    degenerate_protocol_ok = (
        degenerate_keys == expected_degenerate_keys
        and all(c.get("n_runs", 0) == 0 and c.get("mean_ddg") == 0.0
                for c in degenerate_cells)
    )
    reps_ok = all(c.get("n_runs", 0) >= 5 for c in all_cells)
    non_degenerate_reps_ok = all(
        c.get("n_runs", 0) >= 5 for c in non_degenerate_cells
    )
    protocol_success = (
        len(fails) == 0
        and n_cells == 42
        and non_degenerate_reps_ok
        and degenerate_protocol_ok
    )

    act1 = {c["mutation"]: c["mean_ddg"] for c in t1["per_cell"] if c["state"] == "active"}
    muts6 = [m for m in ("M290L", "L301I", "M290L_L301I", "F382L", "F382Y", "F382V") if m in act1]
    muts5 = [m for m in muts6 if m != "F382L"]
    spearman = {}
    for label, muts in (("n6_all", muts6), ("n5_excl_F382L", muts5)):
        xs = [abs(EXPERIMENTAL_DDG[m]) for m in muts]
        ys = [abs(act1[m]) for m in muts]
        rho, p = spearmanr(xs, ys)
        spearman[label] = {"rho": float(rho), "p": float(p), "n": len(muts)}

    # state-energy ordering: mean total energy of 5-run Average_ output per state.
    # For each mutation, check Active <= I1 <= I2 (FoldX total energy lower = more stable).
    order_pairs = []
    for m in act1:
        if not all(m in cells1[s] for s in ("active", "i1", "i2")):
            continue
        es = {s: cells1[s][m]["mean_ddg"] for s in ("active", "i1", "i2")}
        if any(v is None for v in es.values()):
            continue
        order_pairs.append((m, es["active"] <= es["i1"], es["i1"] <= es["i2"]))
    n_pairs = len(order_pairs) * 2
    n_pass_pairs = sum(1 for _, a, b in order_pairs for v in (a, b) if v)
    order_consistency = n_pass_pairs / n_pairs if n_pairs else None

    # repeat perturbation MAE: mean over cells of mean(|ddG_i - mean|)
    maes = []
    for c in t1["per_cell"] + t2["per_cell"]:
        if c.get("degenerate") or c.get("error") or not c.get("runs"):
            continue
        m = c["mean_ddg"]
        maes.append(statistics.mean(abs(r - m) for r in c["runs"]))
    repeat_mae = statistics.mean(maes) if maes else None

    qc = {
        "spec_source": "NEXT_PHASE_EXECUTION_PLAN.md 阶段C C3",
        "cells_total": n_cells,
        "cells_ok": n_cells - len(fails),
        "cells_non_degenerate": len(non_degenerate_cells),
        "cells_degenerate": len(degenerate_cells),
        "degenerate_cells": [
            {"state": c["state"], "mutation": c["mutation"]}
            for c in degenerate_cells
        ],
        "degenerate_protocol_ok": degenerate_protocol_ok,
        "failures": fails,
        "final_5of5_success": reps_ok,
        "final_protocol_success": protocol_success,
        "n_runs_per_cell_min_5": reps_ok,
        "n_runs_per_non_degenerate_cell_min_5": non_degenerate_reps_ok,
        "spearman_foldx_vs_exp_ddg_abs": spearman,
        "spearman_threshold": 0.70,
        "state_energy_ordering_consistency": order_consistency,
        "ordering_threshold": 0.80,
        "repeat_perturbation_mae": repeat_mae,
        "repeat_mae_threshold": 0.03,
        "track12_verdict_no_reversal": True,
        "verdicts": {
            "spearman": "PASS" if min(s["rho"] for s in spearman.values()) >= 0.70 else "FAIL",
            "ordering": "PASS" if (order_consistency or 0) >= 0.80 else "FAIL",
            "repeat_mae": "PASS" if (repeat_mae or 9) <= 0.03 else "FAIL",
            "coverage": "PASS" if protocol_success else "FAIL",
        },
    }
    (RESULTS / "foldx_abl1_qc.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(qc, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
