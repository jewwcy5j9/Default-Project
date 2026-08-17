"""Phase-1 zero-shot Boltzmann predictor on FoldX per-state ddG.

p_hat_{m,k} proportional to p_WT,k * exp(-beta * ddG_{m,k}), normalized over K.

Evaluation (frozen benchmark registry):
  - 3-state per-mutant MAE, mean MAE, direction (tie |NMR-WT|<0.05)
  - u1 = 2*p_active - 1, u2 = p_I1 - p_I2 contrast MAE
  - vs baselines: training-mean 0.2329, constant-WT 0.3878,
    reference fixed encodings C 0.0804 / ESM2 0.1459 / LLR 0.1629
  - kill check (plan 3.8): corr(computed ddG vector, population-implied energy)
    < 0.2; and reference-structure / repeat stability

Usage:
  python foldx_boltzmann.py --ddg results/foldx_abl1_ddg.json [--track 1|2] [--self-test]
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

from k3_data import ABL1_K3, ABL1_K3_WT_POP

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BETA = 1.677

THRESHOLDS = {
    "adopt_abl1_mae_lt": 0.1459,        # plan 3.7 (current best experiment-free fixed)
    "training_mean": 0.2329,
    "constant_wt": 0.3878,
    "kill_ddg_corr_lt": 0.2,            # plan 3.8
}

MUTS = ["M290L", "L301I", "M290L_L301I", "F382L", "F382Y", "F382V"]
STATES = ["active", "i1", "i2"]  # ABL1_K3 order [Active, I1, I2]


def boltzmann(ddgs, wt_pop, beta=BETA):
    """ddgs: dict state->ddG. Returns normalized 3-vector."""
    p = np.zeros(3)
    for i, s in enumerate(STATES):
        d = ddgs.get(s)
        if d is None or np.isnan(d):
            return None
        p[i] = wt_pop[i] * np.exp(-beta * d)
    s = p.sum()
    if s <= 0 or not np.isfinite(s):
        return None
    return p / s


def mae_per_mutant(pred, true):
    return float(np.abs(np.asarray(pred) - np.asarray(true)).mean())


def direction_ok(pred, true, wt_pop, tie=0.05):
    p, t = np.asarray(pred), np.asarray(true)
    if abs(t - np.asarray(wt_pop)).sum() < tie:
        return None
    return bool(np.dot(p - np.asarray(wt_pop), t - np.asarray(wt_pop)) > 0)


def u_contrasts(pop):
    p = np.asarray(pop)
    return 2 * p[0] - 1, p[1] - p[2]


def population_implied_energy(true, wt):
    """-RT log(K_mut/K_WT) from populations (Xie 2020 Fig 4C convention)."""
    t, w = np.asarray(true, float), np.asarray(wt, float)
    eps = 1e-6
    return -1.0 / BETA * (np.log(np.maximum(t, eps)) - np.log(np.maximum(w, eps)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddg", default=str(RESULTS / "foldx_abl1_ddg.json"))
    ap.add_argument("--track", type=int, default=1, choices=[1, 2])
    ap.add_argument("--self-test", action="store_true",
                    help="evaluate plumbing on synthetic ddG instead of real file")
    args = ap.parse_args()

    if args.self_test:
        rng = np.random.default_rng(0)
        ddg_cells = []
        for m in MUTS:
            for s in STATES:
                ddg_cells.append({"state": s, "mutation": m,
                                  "mean_ddg": float(rng.normal(0, 0.5)),
                                  "std_ddg": 0.1, "n_runs": 5, "degenerate": False})
        ddg = {"per_cell": ddg_cells, "beta": {"value": BETA}, "track": 1}
        label = "SELF-TEST"
    else:
        ddg = json.loads(Path(args.ddg).read_text(encoding="utf-8"))
        label = f"track{ddg.get('track', args.track)}"

    cells = {(c["mutation"], c["state"]): c for c in ddg["per_cell"]}
    wt = np.array(ABL1_K3_WT_POP, dtype=float)

    rows = []
    for m in MUTS:
        dd = {}
        missing = False
        for s in STATES:
            c = cells.get((m, s))
            v = (c or {}).get("mean_ddg")
            if v is None:
                missing = True
                break
            dd[s] = v
        if missing:
            rows.append({"mutation": m, "error": f"missing ddG cell ({s})"})
            continue
        pred = boltzmann(dd, wt)
        if pred is None:
            rows.append({"mutation": m, "error": "non-normalizable prediction"})
            continue
        true = np.array(ABL1_K3[m]["pop"], dtype=float)
        rows.append({
            "mutation": m,
            "true": true.tolist(),
            "pred": pred.tolist(),
            "mae": mae_per_mutant(pred, true),
            "direction": direction_ok(pred, true, wt),
            "u1_true": u_contrasts(true)[0], "u1_pred": u_contrasts(pred)[0],
            "u2_true": u_contrasts(true)[1], "u2_pred": u_contrasts(pred)[1],
        })

    valid = [r for r in rows if "error" not in r]
    mae = np.mean([r["mae"] for r in valid])
    u1 = np.mean([abs(r["u1_pred"] - r["u1_true"]) for r in valid])
    u2 = np.mean([abs(r["u2_pred"] - r["u2_true"]) for r in valid])
    dir_ok = sum(1 for r in valid if r["direction"] is True)
    dir_tot = sum(1 for r in valid if r["direction"] is not None)

    # kill check: ddG vs population-implied energy (per-mutant vector corr)
    implied = {}
    for m in MUTS:
        t = np.array(ABL1_K3[m]["pop"], dtype=float)
        implied[m] = population_implied_energy(t, wt)
    ddg_vec = np.array([(cells.get((m, "active")) or {}).get("mean_ddg") or 0.0
                        for m in MUTS])
    impl_vec = np.array([implied[m][0] for m in MUTS])  # Active-channel energy
    r_active = float(np.corrcoef(ddg_vec, impl_vec)[0, 1]) if ddg_vec.std() > 0 and impl_vec.std() > 0 else None

    out = {
        "label": label,
        "beta": BETA,
        "n_mutants": len(valid),
        "per_mutant": rows,
        "mae": float(mae),
        "u1_mae": float(u1),
        "u2_mae": float(u2),
        "direction": f"{dir_ok}/{dir_tot}",
        "vs_baselines": {
            "training_mean": THRESHOLDS["training_mean"],
            "constant_wt": THRESHOLDS["constant_wt"],
            "skill_vs_training_mean": 1.0 - mae / THRESHOLDS["training_mean"],
            "skill_vs_constant_wt": 1.0 - mae / THRESHOLDS["constant_wt"],
        },
        "adoption_checks": {
            "mae_lt_0_1459": bool(mae < THRESHOLDS["adopt_abl1_mae_lt"]),
            "ddg_corr_population_implied_active": r_active,
            "kill_if_corr_lt_0_2": bool(r_active is not None and r_active < THRESHOLDS["kill_ddg_corr_lt"]),
        },
    }

    print(f"=== {label}: zero-shot Boltzmann (Abl1 K=3, n={len(valid)}) ===")
    for r in valid:
        print(f"  {r['mutation']:<14} MAE={r['mae']:.4f} dir={r['direction']} "
              f"pred={np.round(r['pred'], 3)}")
    print(f"  MEAN MAE = {mae:.4f}  | u1 = {u1:.3f}  | u2 = {u2:.3f}  | dir {out['direction']}")
    print(f"  skill vs training-mean: {out['vs_baselines']['skill_vs_training_mean']:+.3f} "
          f"(adopt if <0.1459: {out['adoption_checks']['mae_lt_0_1459']})")
    print(f"  ddG-Active corr with population-implied energy: {r_active} "
          f"(kill if < 0.2: {out['adoption_checks']['kill_if_corr_lt_0_2']})")

    out_path = RESULTS / f"foldx_boltzmann_abl1_{label}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
