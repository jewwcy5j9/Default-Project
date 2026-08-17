"""P2b: Convex-cone support check (direction requires non-negative combo of
training shift directions). Complements the span-based support score,
which is degenerate at n-1 >= 2.

For each LOO test point: solve min ||D^T a - d||_1 s.t. a >= 0.
  cone_in  : residual < tol (direction realizable from training directions)
  cone_out : residual >= tol (direction requires negative/out-of-cone mixing)

Output: results/p2b_cone_support.json
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from scipy.optimize import linprog

from k3_data import ABL1_K3, ABL1_K3_WT_POP, SRC_K3, SRC_K3_WT_POP

OUT = Path(__file__).resolve().parent / "results"
ABL1_CORE = {m: ABL1_K3[m] for m in ABL1_K3 if m not in ("WT", "H396P", "M290L_H396P")}
SRC_CORE = {m: SRC_K3[m] for m in SRC_K3 if m != "SrcKD-WT"}


def cone_residual(D, d):
    """min_{a>=0} ||D^T a - d||_1  (LP via epigraph)."""
    n_tr, K = D.shape
    if n_tr == 0:
        return float(np.linalg.norm(d, 1))
    c = np.concatenate([np.zeros(n_tr), np.ones(2 * K)])
    A_ub = np.vstack([
        np.hstack([D.T, -np.eye(K), np.zeros((K, K))]),
        np.hstack([-D.T, np.zeros((K, K)), -np.eye(K)]),
    ])
    b_ub = np.concatenate([d, -d])
    bounds = [(0, None)] * n_tr + [(None, None)] * (2 * K)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return float("nan")
    return float(res.fun)


def cone_support(mutations, wt_pop):
    names = list(mutations.keys())
    wt = np.array(wt_pop, dtype=float)
    out = {}
    for held in names:
        tr = [m for m in names if m != held]
        D = np.array([np.array(mutations[m]["pop"], dtype=float) - wt for m in tr])
        d = np.array(mutations[held]["pop"], dtype=float) - wt
        norm_d = float(np.linalg.norm(d, 1))
        if norm_d < 1e-9:
            out[held] = {"cone_in": True, "residual": 0.0, "shift_l1": norm_d}
            continue
        res = cone_residual(D, d)
        out[held] = {"cone_in": bool(res < 1e-6),
                     "residual": res,
                     "residual_rel": float(res / norm_d),
                     "shift_l1": norm_d}
    return out


def main():
    print("=" * 90)
    print("P2b: convex-cone direction support")
    print("=" * 90)
    results = {"abl1": cone_support(ABL1_CORE, ABL1_K3_WT_POP),
               "src": cone_support(SRC_CORE, SRC_K3_WT_POP)}
    for sys_name, d in results.items():
        print(f"  {sys_name}:")
        for m, v in d.items():
            extra = f"res_rel={v.get('residual_rel', float('nan')):.3f}"
            print(f"      {m:<16} cone_in={v['cone_in']} "
                  f"{extra} shift_l1={v['shift_l1']:.3f}")
    (OUT / "p2b_cone_support.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[OK] p2b_cone_support.json written")


if __name__ == "__main__":
    main()
