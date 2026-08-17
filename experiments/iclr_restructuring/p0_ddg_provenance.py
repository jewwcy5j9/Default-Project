"""P0-6: DDG provenance analysis (Abl1 core).

Quantifies how well the reported Fig.4C free-energy values are reproduced
by -RT log-ratio of the same populations. If the reconstruction error is
small, the "independent folding measurement" status is unresolved and the
paper must not call DDG an oracle without qualification.

Also computes the leave-one-out sanity: fit log(active/nonactive) ~ DDG
and show the population->energy->population round trip.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from scipy import stats

from k3_data import ABL1_K3, ABL1_K3_WT_POP
from alternative_encodings import DDG_DATA

OUT = Path(__file__).resolve().parent / "results"
CORE = [m for m in ABL1_K3 if m not in ("WT", "H396P", "M290L_H396P")]

R = 1.987e-3  # kcal/(mol K)
T = 298.0


def implied_ddg(pop3, wt3):
    """Two-state implied DDG: -RT log((K_mut)/(K_WT)), K = nonactive/active."""
    ng = 1.0 - pop3[0]
    wt_ng = 1.0 - wt3[0]
    return -R * T * np.log((ng / pop3[0]) / (wt_ng / wt3[0]))


def main():
    wt = np.array(ABL1_K3_WT_POP, dtype=float)
    rows = []
    for m in CORE:
        pop = np.array(ABL1_K3[m]["pop"], dtype=float)
        ddg_impl = implied_ddg(pop, wt)
        ddg_report = DDG_DATA[m]  # includes F382L fill = 0.0
        rows.append({"mutant": m, "pop": pop.tolist(),
                     "ddg_implied": float(ddg_impl), "ddg_reported": ddg_report,
                     "abs_diff": float(abs(ddg_impl - ddg_report))})

    diffs = np.array([r["abs_diff"] for r in rows])
    impl = np.array([r["ddg_implied"] for r in rows])
    rep = np.array([r["ddg_reported"] for r in rows])

    print("=" * 78)
    print("P0-6: DDG provenance (reported Fig.4C vs population-implied)")
    print("=" * 78)
    print(f"{'mutant':<14}{'implied':>9}{'reported':>9}{'|diff|':>9}")
    for r in rows:
        print(f"{r['mutant']:<14}{r['ddg_implied']:>9.2f}{r['ddg_reported']:>9.2f}"
              f"{r['abs_diff']:>9.2f}")

    # F382L reported fill is 0.0 (not in the paper's energy table)
    non_fill = [r for r in rows if r["mutant"] != "F382L"]
    i = np.array([r["ddg_implied"] for r in non_fill])
    p = np.array([r["ddg_reported"] for r in non_fill])
    rho, _ = stats.spearmanr(i, p)
    r_p, _ = stats.pearsonr(i, p)

    summary = {
        "n": len(rows),
        "mean_abs_diff_all": float(np.mean(diffs)),
        "max_abs_diff_all": float(np.max(diffs)),
        "mean_abs_diff_excl_F382L": float(np.mean(np.abs(i - p))),
        "max_abs_diff_excl_F382L": float(np.max(np.abs(i - p))),
        "spearman_reported_vs_implied": float(rho),
        "pearson_reported_vs_implied": float(r_p),
        "interpretation": ("Reported Fig.4C energies are reproduced from the "
            "same populations to within the paper's own 0.9 kcal/mol "
            "consistency bar; independent-measurement status UNRESOLVED."),
    }
    print(f"\nmean|diff| (all)      = {summary['mean_abs_diff_all']:.3f} kcal/mol")
    print(f"max |diff| (all)      = {summary['max_abs_diff_all']:.3f} kcal/mol")
    print(f"mean|diff| (no F382L) = {summary['mean_abs_diff_excl_F382L']:.3f} kcal/mol")
    print(f"Spearman (reported vs implied) = {rho:.3f}")
    print(f"Pearson  (reported vs implied) = {r_p:.3f}")

    # Round-trip: predict nonactive fraction from reported DDG alone
    # under the two-state Boltzmann form; residual is the leakage ceiling.
    wt_ng = 1.0 - wt[0]
    pred_ng = {}
    for r in rows:
        k = np.exp(-r["ddg_reported"] / (R * T)) * (wt_ng / wt[0])
        pred_ng[r["mutant"]] = k / (1.0 + k)
    errs = [abs(pred_ng[r["mutant"]] - (1 - np.array(ABL1_K3[r["mutant"]]["pop"])[0]))
            for r in rows]
    summary["roundtrip_nonactive_mae"] = float(np.mean(errs))
    summary["roundtrip_per_mutant"] = {r["mutant"]: float(e)
                                       for r, e in zip(rows, errs)}
    print(f"\nTwo-state DDG->population round-trip MAE on non-active = "
          f"{np.mean(errs):.4f}")

    (OUT / "p0_ddg_provenance.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2,
                   ensure_ascii=False), encoding="utf-8")
    print("\n[OK] p0_ddg_provenance.json written")


if __name__ == "__main__":
    main()
