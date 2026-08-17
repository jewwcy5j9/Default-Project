"""P4c: Orthonormal contrast rebase for paper reporting (C4).

Derives q1/q2 (orthonormal contrast basis) MAEs from the frozen raw u1/u2
contrast MAEs. The basis is defined in main_v2.tex Appendix A:

    state order (Active, E1, E2)
    q1 = (2, -1, -1)/sqrt(6)   # retained by active/non-active pooling
    q2 = (0,  1, -1)/sqrt(2)   # discarded by pooling

Algebra (residual e = pred - target lies in 1^perp, so e_E1 + e_E2 = -e_A):

    q1 . e = (2 e_A - e_E1 - e_E2)/sqrt(6) = 3 e_A/sqrt(6) = (sqrt(6)/4) * u1_err
    q2 . e = (e_E1 - e_E2)/sqrt(2)          = u2_err/sqrt(2)

with raw u1_err = |2 pA_hat - 1 - (2 pA - 1)| = 2|e_A|, u2_err = |e_E1 - e_E2|.
Hence MAE(q1) = (sqrt(6)/4) * MAE(u1) and MAE(q2) = MAE(u2)/sqrt(2) exactly.

This script does NOT refit anything: it verifies the per-mutation identity on
the released predictions, then derives every paper-reported rebased number.
Inputs are frozen artifacts; outputs are new derived artifacts only.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
CANONICAL = Path(__file__).resolve().parents[2] / "data" / "nmr_populations"

C_Q1 = math.sqrt(6) / 4.0          # q1 error = C_Q1 * u1 error
C_Q2 = 1.0 / math.sqrt(2.0)        # q2 error = C_Q2 * u2 error
THRESHOLD = math.sqrt(3.0) / 2.0   # q2 > q1  <=>  u2 > (sqrt(3)/2) * u1

HYBRID_L410A = [0.96, 0.03, 0.01]  # published global-fit L410A value


def load_targets() -> dict[str, list[float]]:
    rows = {}
    with (CANONICAL / "src_k3_canonical.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["used_in_primary"].strip().lower() == "true":
                rows[row["mutation_id"]] = [
                    float(row["state_A"]), float(row["state_E1"]), float(row["state_E2"])]
    return rows


def verify_row(name: str, preds: dict, targets: dict, errors: list[str]) -> None:
    """Verify |q . e| == C * |u_err| for every mutation in one row.

    The identity is exact for simplex residuals (sum e = 0). Released
    predictions sum to 1 only up to float normalization (~1e-6), so the
    numeric tolerance is 1e-6 absolute.
    """
    for m, p in preds.items():
        if m not in targets:
            errors.append(f"{name}: missing target for {m}")
            continue
        y = targets[m]
        e = [p[k] - y[k] for k in range(3)]
        u1_err = abs(2.0 * p[0] - 1.0 - (2.0 * y[0] - 1.0))
        u2_err = abs((p[1] - p[2]) - (y[1] - y[2]))
        q1_err = abs((2.0 * e[0] - e[1] - e[2]) / math.sqrt(6.0))
        q2_err = abs((e[1] - e[2]) / math.sqrt(2.0))
        if abs(q1_err - C_Q1 * u1_err) > 1e-6:
            errors.append(f"{name}/{m}: q1 identity violated")
        if abs(q2_err - C_Q2 * u2_err) > 1e-6:
            errors.append(f"{name}/{m}: q2 identity violated")


def main() -> int:
    errors: list[str] = []
    out: dict = {
        "schema": "contrast_orthonormal_rebase_v1",
        "date": "2026-08-13",
        "basis": {"q1": "(2,-1,-1)/sqrt(6)", "q2": "(0,1,-1)/sqrt(2)"},
        "constants": {"q1_per_u1": C_Q1, "q2_per_u2": C_Q2,
                      "ordering_threshold_u2_over_u1": THRESHOLD},
        "inputs": [],
        "verification": {"rows_checked": 0, "errors": []},
        "paper_numbers": {},
    }

    targets = load_targets()

    # ---- label sensitivity battery ----
    sens_path = RESULTS / "p2_k3_src_label_sensitivity.json"
    sens = json.loads(sens_path.read_text(encoding="utf-8"))
    out["inputs"].append(sens_path.name)
    batteries = {}
    for protocol in ("primary_probe", "l410a_global_fit_substitution"):
        tgt = dict(targets)
        if protocol == "l410a_global_fit_substitution":
            tgt["SrcKD-L410A"] = list(HYBRID_L410A)
        rows = sens["systems"][protocol]["fixed_k3"]
        row_out = {}
        for key, row in rows.items():
            u1 = row["u1_u2_contrast"]["u1"]
            u2 = row["u1_u2_contrast"]["u2"]
            verify_row(f"{protocol}/{key}", row["per_mutant_pred"], tgt, errors)
            out["verification"]["rows_checked"] += 1
            row_out[key] = {
                "u1_raw": u1, "u2_raw": u2,
                "q1": C_Q1 * u1, "q2": C_Q2 * u2,
                "raw_u2_gt_u1": u2 > u1,
                "rebased_q2_gt_q1": u2 > THRESHOLD * u1,
            }
        batteries[protocol] = row_out
    out["paper_numbers"]["sensitivity_battery"] = batteries

    def count_gt(battery: dict, flag: str) -> tuple[int, int]:
        vals = [v[flag] for v in battery.values()]
        return sum(bool(v) for v in vals), len(vals)

    for protocol in ("primary_probe", "l410a_global_fit_substitution"):
        b = batteries[protocol]
        raw_n, tot = count_gt(b, "raw_u2_gt_u1")
        new_n, _ = count_gt(b, "rebased_q2_gt_q1")
        clr_keys = [k for k in b if "Ridge" in k or "GP" in k]
        clr_new = sum(b[k]["rebased_q2_gt_q1"] for k in clr_keys)
        out["paper_numbers"][f"{protocol}_counts"] = {
            "raw_u2_gt_u1": f"{raw_n}/{tot}",
            "rebased_q2_gt_q1": f"{new_n}/{tot}",
            "clr_rows_rebased_q2_gt_q1": f"{clr_new}/10",
        }

    # ---- pseudocount + stress (CLR robustness artifact) ----
    rob_path = RESULTS / "p2_k3_src_clr_robustness.json"
    rob = json.loads(rob_path.read_text(encoding="utf-8"))
    out["inputs"].append(rob_path.name)
    pc_rows = rob["pseudocount_sensitivity"]["rows"]
    pc_new = [r["u2_mae"] > THRESHOLD * r["u1_mae"] for r in pc_rows]
    out["paper_numbers"]["pseudocount"] = {
        "raw_u2_gt_u1": f"{sum(r['u2_gt_u1'] for r in pc_rows)}/50",
        "rebased_q2_gt_q1": f"{sum(pc_new)}/50",
        "per_pseudocount_rebased": {
            pc: sum(f for f, r in zip(pc_new, pc_rows) if r["pseudocount"] == pc)
            for pc in rob["pseudocount_sensitivity"]["pseudocounts"]},
    }
    stress = rob["digitization_interval_stress_test"]
    new_both = sum(
        all(r["u2_mae"] > THRESHOLD * r["u1_mae"] for r in rec["rows"].values())
        for rec in stress["records"])
    out["paper_numbers"]["stress"] = {
        "raw_both_u2_gt_u1_proportion": stress["proportions"]["both_u2_gt_u1"],
        "rebased_both_q2_gt_q1_proportion": new_both / stress["realizations"],
        "rebased_both_q2_gt_q1_runs": f"{new_both}/{stress['realizations']}",
    }

    # ---- T5 frozen diagnostic (Table 2 rows) ----
    t5_path = RESULTS / "t5_review_responses.json"
    t5 = json.loads(t5_path.read_text(encoding="utf-8"))
    t5b_path = RESULTS / "t5b_review_responses.json"
    t5b = json.loads(t5b_path.read_text(encoding="utf-8"))
    out["inputs"] += [t5_path.name, t5b_path.name]
    gp = t5["contrast"]["clrgp_pos"]["summary"]
    out["paper_numbers"]["table_contrasts"] = {
        "gp_t5": {"q1": C_Q1 * gp["u1_mae"], "q2": C_Q2 * gp["u2_mae"],
                  "q1_r2": gp["u1_r2"], "q2_r2": gp["u2_r2"]},
        "mlp_current": {
            "u1": 0.5206792055330889, "u2": 0.6845696578954912,  # reproducible run
            "q1": C_Q1 * 0.5206792055330889, "q2": C_Q2 * 0.6845696578954912,
            "q1_r2": -1.21, "q2_r2": -0.97},
        "pooled_k2": {"non_active_mae": t5b["k2_clrgp_pos"]["non_active_mae"],
                      "q1_scale": C_Q1 * 2.0 * t5b["k2_clrgp_pos"]["non_active_mae"]},
    }

    # ---- paper row values (4-decimal display) ----
    disp = {}
    for protocol in ("primary_probe", "l410a_global_fit_substitution"):
        for key in ("pos::LowRankCDST", "pca20::LowRankCDST",
                    "pos::CLR-Ridge", "pos::CLR-GP"):
            v = batteries[protocol][key]
            disp[f"{protocol}::{key}"] = {
                "q1_4dp": round(v["q1"], 4), "q2_4dp": round(v["q2"], 4)}
    out["paper_numbers"]["table_src_provenance_rebased_4dp"] = disp

    out["verification"]["errors"] = errors
    out_path = RESULTS / "contrast_orthonormal_rebase_20260813.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", out_path)

    # ---- human-readable summary ----
    lines = [
        "# Orthonormal contrast rebase (C4) — derived paper numbers",
        "",
        f"- constants: q1 error = (sqrt(6)/4) * u1 error = {C_Q1:.6f} * u1;"
        f" q2 error = u2/sqrt(2) = {C_Q2:.6f} * u2",
        f"- ordering: q2 > q1  <=>  u2 > (sqrt(3)/2) u1 = {THRESHOLD:.6f} u1",
        "- verified rows:", str(out["verification"]["rows_checked"]),
    ]
    for protocol in ("primary_probe", "l410a_global_fit_substitution"):
        lines.append(f"- {protocol}: raw {out['paper_numbers'][protocol+'_counts']['raw_u2_gt_u1']},"
                     f" rebased {out['paper_numbers'][protocol+'_counts']['rebased_q2_gt_q1']},"
                     f" CLR rows rebased {out['paper_numbers'][protocol+'_counts']['clr_rows_rebased_q2_gt_q1']}")
    lines.append("- pseudocount: raw "
                 + out["paper_numbers"]["pseudocount"]["raw_u2_gt_u1"]
                 + ", rebased "
                 + out["paper_numbers"]["pseudocount"]["rebased_q2_gt_q1"])
    lines.append("- stress both-orderings: raw "
                 + f"{out['paper_numbers']['stress']['raw_both_u2_gt_u1_proportion']:.3f}"
                 + ", rebased "
                 + f"{out['paper_numbers']['stress']['rebased_both_q2_gt_q1_proportion']:.4f}")
    tc = out["paper_numbers"]["table_contrasts"]
    lines.append(f"- Table 2: GP-T5 q1={tc['gp_t5']['q1']:.4f} q2={tc['gp_t5']['q2']:.4f};"
                 f" MLP-current q1={tc['mlp_current']['q1']:.4f} q2={tc['mlp_current']['q2']:.4f};"
                 f" pooled q1-scale={tc['pooled_k2']['q1_scale']:.4f}")
    for k, v in disp.items():
        lines.append(f"- {k}: q1={v['q1_4dp']} q2={v['q2_4dp']}")
    if errors:
        lines.append("- ERRORS: " + "; ".join(errors))
    (RESULTS / "contrast_orthonormal_rebase_20260813_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
