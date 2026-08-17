#!/usr/bin/env python
"""FoldX Src C3 QC (NEXT_PHASE_EXECUTION_PLAN.md 阶段 C3).

Checks implemented:
  - cell success rate (目标 5/5 成功, <4/5 marker cells failures)
  - Spearman(foldx ddG, |exp ddG|) >= 0.70: N/A for Src — no experimental ddG
    for the 8 core mutants (marker-flagged, not scored)
  - state-energy ordering consistency >= 80% (Active <= E1 <= E2 by mean_ddg,
    lower ddG = more stable; guideline only, E2 frozen independent of results)
  - reference/repeat perturbation MAE <= 0.03 (mean over cells of |ddG_i - mean|)

Output: results/foldx_src_qc.json
"""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

STATES = ["active", "e1", "e2"]


def main():
    with open(RESULTS / "foldx_src_ddg.json", encoding="utf-8") as f:
        d = json.load(f)
    cells = d["per_cell"]
    fails = d.get("failures", [])

    n_cells = len(cells)
    success = (len(fails) == 0) and (n_cells == 24)
    reps_ok = all(c.get("n_runs", 0) >= 5 for c in cells)

    by_state = {s: {c["mutation"]: c for c in cells if c["state"] == s} for s in STATES}

    # state-energy ordering: for each mutation, check active <= e1 <= e2
    order_pairs = []
    for m in by_state["active"]:
        if not all(m in by_state[s] for s in STATES):
            continue
        es = {s: by_state[s][m]["mean_ddg"] for s in STATES}
        if any(v is None for v in es.values()):
            continue
        order_pairs.append((m, es["active"] <= es["e1"], es["e1"] <= es["e2"]))
    n_pairs = len(order_pairs) * 2
    n_pass_pairs = sum(1 for _, a, b in order_pairs for v in (a, b) if v)
    order_consistency = n_pass_pairs / n_pairs if n_pairs else None

    # reference/repeat perturbation MAE
    maes = []
    for c in cells:
        if c.get("error") or not c.get("runs"):
            continue
        m = c["mean_ddg"]
        maes.append(statistics.mean(abs(r - m) for r in c["runs"]))
    repeat_mae = statistics.mean(maes) if maes else None

    # per-mutation repeat std summary (marker cells flagged for attention)
    per_mut = {}
    for c in cells:
        pm = per_mut.setdefault(c["mutation"], {"states": {}, "max_std": 0.0})
        pm["states"][c["state"]] = {"mean_ddg": round(c["mean_ddg"], 4),
                                    "std_ddg": round(c["std_ddg"], 4) if c["std_ddg"] else None}
        pm["max_std"] = max(pm["max_std"], c["std_ddg"] or 0.0)

    qc = {
        "spec_source": "NEXT_PHASE_EXECUTION_PLAN.md 阶段C C3",
        "cells_total": n_cells,
        "cells_ok": n_cells - len(fails),
        "failures": fails,
        "final_5of5_success": success,
        "n_runs_per_cell_min_5": reps_ok,
        "spearman_foldx_vs_exp_ddg_abs": {
            "rho": None, "p": None, "n": 0,
            "note": "N/A: no experimental ddG for the 8 Src core mutants (Cui 2025 reports state populations only)",
        },
        "spearman_threshold": 0.70,
        "state_energy_ordering_consistency": order_consistency,
        "ordering_threshold": 0.80,
        "repeat_perturbation_mae": repeat_mae,
        "repeat_mae_threshold": 0.03,
        "per_mutation": per_mut,
        "verdicts": {
            "spearman": "N/A",
            "ordering": "PASS" if (order_consistency or 0) >= 0.80 else "FAIL",
            "repeat_mae": "PASS" if (repeat_mae or 9) <= 0.03 else "FAIL",
            "coverage": "PASS" if (len(fails) == 0) else "FAIL",
        },
    }
    (RESULTS / "foldx_src_qc.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(qc, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()