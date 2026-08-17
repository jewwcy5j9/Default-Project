"""P2: Theory diagnostics for the resolution-support-noise framework.

1. Design audit: singular spectrum / condition number / zero-variance
   columns of each encoding design matrix X (core sets).
2. Support score: for each LOO test point, the cosine/projection of the
   true shift direction d_i = y_i - y_WT onto span of the training shift
   directions. score = ||P_D d_i|| / ||d_i|| (1 = fully supported,
   0 = orthogonal / unsupported).
3. Risk decomposition (Src): pooled (K=2) Bayes risk vs fine-grained
   (K=3) risk; fine contrast variance within the non-active mass.
4. Label-noise resolution: perturb Src labels by +/-5% and +/-10% and
   measure the fraction of pairwise population comparisons that change
   rank (resolution-limited pairs).
5. leave-one-direction-out table: for each test point, report support
   score, true shift norm, and observed error of the best CLR baseline.

Output: results/p2_theory_diagnostics.json
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from k3_data import ABL1_K3, ABL1_K3_WT_POP, SRC_K3, SRC_K3_WT_POP
from alternative_encodings import DDG_DATA
from k3_llr_proxy import LLR

OUT = Path(__file__).resolve().parent / "results"
ABL1_CORE = {m: ABL1_K3[m] for m in ABL1_K3 if m not in ("WT", "H396P", "M290L_H396P")}
SRC_CORE = {m: SRC_K3[m] for m in SRC_K3 if m != "SrcKD-WT"}


def design_audit(X):
    n, d = X.shape
    sv = np.linalg.svd(X, compute_uv=False)
    tol = max(X.shape) * np.finfo(float).eps * (sv[0] if sv.size else 0)
    num_rank = int((sv > tol).sum())
    sv_effective = sv[sv > tol]
    cond = float(sv_effective[0] / sv_effective[-1]) if len(sv_effective) > 1 else float("inf")
    zero_var = int((np.var(X, axis=0) < 1e-12).sum())
    return {"n": n, "d": d, "singulars": [float(s) for s in sv],
            "numerical_rank": num_rank, "condition_number": cond,
            "zero_variance_cols": zero_var}


def support_score(mutations, wt_pop, names_order=None):
    """Projection ratio of each test shift onto training shift span (LOO)."""
    names = list(mutations.keys())
    wt = np.array(wt_pop, dtype=float)
    out = {}
    for held in names:
        tr = [m for m in names if m != held]
        D_tr = np.array([np.array(mutations[m]["pop"], dtype=float) - wt for m in tr])
        d = np.array(mutations[held]["pop"], dtype=float) - wt
        norm_d = np.linalg.norm(d)
        if norm_d < 1e-9:
            out[held] = {"support": 1.0, "shift_norm": float(norm_d), "n_train_dirs": len(tr)}
            continue
        proj = D_tr.T @ np.linalg.lstsq(D_tr.T, d, rcond=None)[0]
        out[held] = {"support": float(np.linalg.norm(proj) / norm_d),
                     "shift_norm": float(norm_d), "n_train_dirs": len(tr)}
    return out


def noise_resolution(mutations, wt_pop, levels=(0.05, 0.10), n_draws=2000, seed=42):
    """Fraction of pairwise population comparisons robust to label noise."""
    names = [m for m in mutations if m != "SrcKD-WT"] if wt_pop is SRC_K3_WT_POP else \
            [m for m in mutations]
    pops = {m: np.array(mutations[m]["pop"], dtype=float) for m in names}
    rng = np.random.default_rng(seed)
    ref_rank = {m: pops[m] for m in names}
    out = {}
    for lvl in levels:
        flip = 0
        total = 0
        for _ in range(n_draws):
            noisy = {}
            for m in names:
                e = rng.uniform(-lvl, lvl, 3)
                p = np.clip(pops[m] + e, 0, 1)
                noisy[m] = p / p.sum()
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    for k in range(3):
                        total += 1
                        if (ref_rank[a][k] - ref_rank[b][k]) * \
                           (noisy[a][k] - noisy[b][k]) < 0:
                            flip += 1
        out[f"lvl_{int(lvl*100)}"] = {"flipped_pairs_frac": float(flip / total)}
    return out


def risk_decomposition(mutations, wt_pop):
    """Pooled vs fine risk of the constant-WT predictor + fine contrast variance."""
    names = list(mutations.keys())
    pops = np.array([mutations[m]["pop"] for m in names])
    wt = np.array(wt_pop, dtype=float)
    ng = 1.0 - pops[:, 0]
    ng_wt = 1.0 - wt[0]
    pooled_risk = float(np.abs(ng - ng_wt).mean())
    fine_risk = float(np.abs(pops - wt).mean())
    # fine contrast within non-active: relative E1/(E1+E2) variance
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_e1 = np.where((pops[:, 1] + pops[:, 2]) > 1e-9,
                          pops[:, 1] / (pops[:, 1] + pops[:, 2]), np.nan)
    rel_e1_valid = rel_e1[~np.isnan(rel_e1)]
    return {"pooled_constant_risk": pooled_risk,
            "fine_constant_risk": fine_risk,
            "fine_contrast_std_E1_share": float(np.nanstd(rel_e1_valid)),
            "n_nonactive_mass": int((ng > 0.05).sum())}


def main():
    print("=" * 90)
    print("P2: theory diagnostics")
    print("=" * 90)
    results = {}

    # 1. design audit
    encs = {
        "abl1_variantC": (ABL1_CORE, lambda n, d: [
            np.zeros(5).__setitem__(0, d["pos"] / 534) or None][0] or np.array(
                [d["pos"] / 534, DDG_DATA.get(n, 0.0) / 3.5,
                 float(d["pos"] == 290), float(d["pos"] == 301),
                 float(d["pos"] == 382)])),
        "abl1_pos": (ABL1_CORE, lambda n, d: np.array(
            [d["pos"] / 534, float(d["pos"] == 290), float(d["pos"] == 301),
             float(d["pos"] == 382)])),
        "src_pos": (SRC_CORE, lambda n, d: np.array(
            [d["pos"] / 536, float(d["pos"] == 311), float(d["pos"] == 332),
             float(d["pos"] == 380)])),
    }
    results["design_audit"] = {}
    for key, (mut, fn) in encs.items():
        X = np.array([fn(m, mut[m]) for m in mut])
        results["design_audit"][key] = design_audit(X)
        print(f"  {key:<14} {results['design_audit'][key]}")

    # 2. support scores
    results["support"] = {
        "abl1": support_score(ABL1_CORE, ABL1_K3_WT_POP),
        "src": support_score(SRC_CORE, SRC_K3_WT_POP)}
    print("\n  support scores (Abl1):")
    for m, v in results["support"]["abl1"].items():
        print(f"      {m:<14} support={v['support']:.3f} shift_norm={v['shift_norm']:.3f}")
    print("  support scores (Src):")
    for m, v in results["support"]["src"].items():
        print(f"      {m:<14} support={v['support']:.3f} shift_norm={v['shift_norm']:.3f}")

    # 3. risk decomposition
    results["risk_decomposition"] = {
        "abl1": risk_decomposition(ABL1_CORE, ABL1_K3_WT_POP),
        "src": risk_decomposition(SRC_CORE, SRC_K3_WT_POP)}
    print(f"\n  risk decomposition: {results['risk_decomposition']}")

    # 4. label-noise resolution
    results["noise_resolution"] = {"src": noise_resolution(SRC_CORE, SRC_K3_WT_POP)}
    print(f"  noise resolution (Src): {results['noise_resolution']['src']}")

    (OUT / "p2_theory_diagnostics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[OK] p2_theory_diagnostics.json written")


if __name__ == "__main__":
    main()
