"""T6: Collision permutation / null calibration.

Upgrades the feature-collision diagnostic from a case study to a calibrated
analysis (plan: NEXT_TIER_EXECUTION_PLAN.md, P1 CPU stream):

A. Label-permutation null on feature-near pairs (d_feat_l2 < tau, tau=0.25):
   two statistics are enumerated exactly (Abl1 6! = 720; Src 8! = 40320):
   severity = sum of d_dir_l1 over feature-near pairs, and the thresholded
   conflict count = number of pairs with d_dir_l1 > delta (delta=0.6),
   matching the paper's two-threshold conflict definition.
B. Output-separation null on the canonical near-collision pair
   (Src pos, L410A/F405A): shuffle populations, refit CLR-Ridge / kNN(1) /
   CLR-GP on all 8 mutants, recompute separation. MLP excluded from the
   permutation (cost); its observed separation is kept from T5.
C. Matched-pair null: within a feature-distance band around each true
   collision pair, where do true-pair direction conflicts and ridge LOO
   errors rank among label-permuted values?

Output: results/t6_collision_null.json
"""
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from k3_data import ABL1_K3, ABL1_K3_WT_POP, SRC_K3, SRC_K3_WT_POP
from p1_core_baselines import clr, inv_clr
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

OUT = Path(__file__).resolve().parent / "results"
ABL1_CORE = {m: ABL1_K3[m] for m in ABL1_K3 if m not in ("WT", "H396P", "M290L_H396P")}
SRC_CORE = {m: SRC_K3[m] for m in SRC_K3 if m != "SrcKD-WT"}
TAU = 0.25
DELTA = 0.6  # d_shift threshold for a thresholded conflict (unified with the paper)
PAIR = ("SrcKD-L410A", "SrcKD-F405A")
RNG = np.random.default_rng(20260803)


def enc_abl1_pos(name, data):
    return np.array([data["pos"] / 534.0, float(data["pos"] == 290),
                     float(data["pos"] == 301), float(data["pos"] == 382)])


def enc_src_pos(name, data):
    return np.array([data["pos"] / 536.0, float(data["pos"] == 311),
                     float(data["pos"] == 332), float(data["pos"] == 380)])


def make_models():
    scaler = StandardScaler()
    ridge = Ridge(alpha=1.0)
    knn = KNeighborsRegressor(n_neighbors=1)
    gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0) + WhiteKernel(0.01),
                                  alpha=1e-4, normalize_y=True, random_state=0)
    return scaler, ridge, knn, gp


def fit_family(scaler, ridge, knn, gp, Cs, pops, wt):
    zs = clr(pops)
    scaler.fit(Cs)
    ridge.fit(scaler.transform(Cs), zs)
    knn.fit(scaler.transform(Cs), zs)
    gp.fit(scaler.transform(Cs), zs)

    def sep(c1, c2):
        z1 = ridge.predict(scaler.transform(np.atleast_2d(c1)))[0]
        z2 = ridge.predict(scaler.transform(np.atleast_2d(c2)))[0]
        return float(np.abs(inv_clr(z1) - inv_clr(z2)).sum())

    def sep_knn(c1, c2):
        z1 = knn.predict(scaler.transform(np.atleast_2d(c1)))[0]
        z2 = knn.predict(scaler.transform(np.atleast_2d(c2)))[0]
        return float(np.abs(inv_clr(z1) - inv_clr(z2)).sum())

    def sep_gp(c1, c2):
        z1 = gp.predict(scaler.transform(np.atleast_2d(c1)))[0]
        z2 = gp.predict(scaler.transform(np.atleast_2d(c2)))[0]
        return float(np.abs(inv_clr(z1) - inv_clr(z2)).sum())

    return sep, sep_knn, sep_gp


def loo_ridge_mae(names, pops, wt, enc_fn):
    Cs = np.array([enc_fn(m, SRC_CORE[m] if m in SRC_CORE else ABL1_CORE[m]) for m in names])
    errs = {}
    for i, held in enumerate(names):
        tr = [j for j in range(len(names)) if j != i]
        zs = clr(pops[tr])
        scaler = StandardScaler().fit(Cs[tr])
        r = Ridge(alpha=1.0).fit(scaler.transform(Cs[tr]), zs)
        z_pred = r.predict(scaler.transform(Cs[i:i + 1]))[0]
        errs[held] = float(np.abs(inv_clr(z_pred) - pops[i]).mean())
    return errs


# ---------------------------------------------------------------
def dir_conflict_pairs(names, pops, enc_fn, wt, tau=TAU):
    """Collision pairs (d_feat < tau) and their d_dir_l1 conflict."""
    Cs = np.array([enc_fn(m, ABL1_CORE[m] if m in ABL1_CORE else SRC_CORE[m]) for m in names])
    D = np.array([pops[i] - wt for i in range(len(names))])
    pairs = []
    for i, j in itertools.combinations(range(len(names)), 2):
        d_feat = float(np.linalg.norm(Cs[i] - Cs[j]))
        if d_feat < tau:
            d_dir = float(np.abs(D[i] - D[j]).sum())
            pairs.append((names[i], names[j], d_feat, d_dir))
    return pairs


def main():
    t0 = time.time()
    print("=" * 90)
    print("T6: collision permutation / null calibration")
    print("=" * 90)
    results = {"tau": TAU, "n_permutations": {"abl1_dir_exact": 720, "src_dir_exact": 40320,
                                               "ridge_knn": 2000, "gp": 200, "loo": 10000}}

    # ---------------- A. label-permutation null on direction conflict ----
    for sys_name, core, wt, enc_fn in [
            ("abl1", ABL1_CORE, ABL1_K3_WT_POP, enc_abl1_pos),
            ("src", SRC_CORE, SRC_K3_WT_POP, enc_src_pos)]:
        names = list(core.keys())
        pops = np.array([core[m]["pop"] for m in names], dtype=float)
        wt_arr = np.array(wt, dtype=float)

        obs_pairs = dir_conflict_pairs(names, pops, enc_fn, wt_arr)
        obs_stat = float(sum(p[3] for p in obs_pairs))
        obs_count = int(sum(1 for p in obs_pairs if p[3] > DELTA))

        if len(names) <= 8:
            perms = list(itertools.permutations(range(len(names))))
            stat = np.zeros(len(perms), dtype=float)
            cnt = np.zeros(len(perms), dtype=float)
            for k, perm in enumerate(perms):
                pp = dir_conflict_pairs(names, pops[list(perm)], enc_fn, wt_arr)
                stat[k] = sum(p[3] for p in pp)
                cnt[k] = sum(1 for p in pp if p[3] > DELTA)
            n_perm = len(perms)
        else:
            perms = [RNG.permutation(len(names)) for _ in range(20000)]
            stat = np.zeros(len(perms), dtype=float)
            cnt = np.zeros(len(perms), dtype=float)
            for k, perm in enumerate(perms):
                pp = dir_conflict_pairs(names, pops[perm], enc_fn, wt_arr)
                stat[k] = sum(p[3] for p in pp)
                cnt[k] = sum(1 for p in pp if p[3] > DELTA)
            n_perm = len(perms)

        p_ge = float(np.mean(stat >= obs_stat))
        p_cnt_ge = float(np.mean(cnt >= obs_count))
        results[f"dir_null_{sys_name}"] = {
            "n_mutants": len(names), "n_exact": n_perm,
            "observed_collision_pairs": [{"pair": list(p[:2]), "d_feat": p[2],
                                          "d_dir_l1": p[3]} for p in obs_pairs],
            "observed_stat": obs_stat,
            "perm_mean": float(stat.mean()), "perm_std": float(stat.std()),
            "perm_pct": {"5": float(np.percentile(stat, 5)),
                         "50": float(np.percentile(stat, 50)),
                         "95": float(np.percentile(stat, 95))},
            "p_perm_ge_obs": p_ge,
            "percentile_obs": float(np.mean(stat <= obs_stat) * 100),
            "thresholded_conflict_count_observed": obs_count,
            "thresholded_conflict_pairs": [{"pair": list(p[:2]), "d_feat": p[2],
                                            "d_dir_l1": p[3]}
                                           for p in obs_pairs if p[3] > DELTA],
            "thresholded_conflict_count_null": {
                "n_exact": n_perm,
                "perm_mean": float(cnt.mean()),
                "p_perm_ge_obs": p_cnt_ge,
                "percentile_obs": float(np.mean(cnt <= obs_count) * 100),
            },
        }
        print(f"\n[A] {sys_name}: feature-near pairs={len(obs_pairs)} "
              f"severity={obs_stat:.3f} (p={p_ge:.4f}) | "
              f"thresholded conflicts={obs_count} (p={p_cnt_ge:.4f})")

    # ---------------- B. output-separation null (Src pos) ----------------
    names = list(SRC_CORE.keys())
    idx_a, idx_b = names.index(PAIR[0]), names.index(PAIR[1])
    pops = np.array([SRC_CORE[m]["pop"] for m in names], dtype=float)
    wt_arr = np.array(SRC_K3_WT_POP, dtype=float)
    Cs = np.array([enc_src_pos(m, SRC_CORE[m]) for m in names])
    c1, c2 = Cs[idx_a], Cs[idx_b]

    scaler, ridge, knn, gp = make_models()
    sep_f, sep_k, sep_g = fit_family(scaler, ridge, knn, gp, Cs, pops, wt_arr)
    obs = {"CLR-Ridge": sep_f(c1, c2), "kNN(1)": sep_k(c1, c2), "CLR-GP": sep_g(c1, c2)}
    print(f"\n[B] observed separation: {obs}")

    n_perm_rk = 2000
    seps_r = np.zeros(n_perm_rk); seps_k = np.zeros(n_perm_rk)
    for it in range(n_perm_rk):
        perm = RNG.permutation(len(names))
        p_perm = pops[perm]
        s, r, k, g = make_models()
        f, fk, fg = fit_family(s, r, k, g, Cs, p_perm, wt_arr)
        seps_r[it] = f(c1, c2)
        seps_k[it] = fk(c1, c2)
    n_perm_gp = 200
    seps_g = np.zeros(n_perm_gp)
    for it in range(n_perm_gp):
        perm = RNG.permutation(len(names))
        p_perm = pops[perm]
        s, r, k, g = make_models()
        f, fk, fg = fit_family(s, r, k, g, Cs, p_perm, wt_arr)
        seps_g[it] = fg(c1, c2)

    results["sep_null_src_pos"] = {"pair": list(PAIR), "observed": obs}
    for label, arr in [("CLR-Ridge", seps_r), ("kNN(1)", seps_k), ("CLR-GP", seps_g)]:
        o = obs[label]
        results["sep_null_src_pos"][label] = {
            "n_permutations": len(arr),
            "observed": float(o),
            "perm_mean": float(arr.mean()), "perm_std": float(arr.std()),
            "perm_pct": {"5": float(np.percentile(arr, 5)),
                         "50": float(np.percentile(arr, 50)),
                         "95": float(np.percentile(arr, 95))},
            "p_perm_ge_obs": float(np.mean(arr >= o)),
            "p_perm_le_obs": float(np.mean(arr <= o)),
        }
        print(f"    {label}: obs={o:.3f} perm50={np.median(arr):.3f} "
              f"p(ge)={np.mean(arr >= o):.4f} p(le)={np.mean(arr <= o):.4f}")

    # ---------------- C. matched-pair null + ridge LOO error ------------
    # d_feat-matched direction-conflict percentile for true collision pairs
    pairs_all = []
    for i, j in itertools.combinations(range(len(names)), 2):
        pairs_all.append((names[i], names[j], float(np.linalg.norm(Cs[i] - Cs[j])),
                          float(np.abs((pops[i] - wt_arr) - (pops[j] - wt_arr)).sum())))
    matched = {}
    for a, b in [("SrcKD-L410A", "SrcKD-F405A"), ("SrcKD-L325A", "SrcKD-V331A")]:
        d_feat_ab = float(np.linalg.norm(Cs[names.index(a)] - Cs[names.index(b)]))
        d_dir_ab = float(np.abs((pops[names.index(a)] - wt_arr) - (pops[names.index(b)] - wt_arr)).sum())
        band_dirs = [p[3] for p in pairs_all if abs(p[2] - d_feat_ab) < 0.02]
        # exclude the true pair itself from the comparison set
        band_dirs_excl = [v for (n1, n2, df, v) in pairs_all
                          if abs(df - d_feat_ab) < 0.02 and not ((n1, n2) in ((a, b), (b, a)))]
        matched[f"{a}|{b}"] = {
            "d_feat": d_feat_ab, "d_dir_l1": d_dir_ab,
            "band_n": len(band_dirs_excl),
            "band_d_dir_mean": float(np.mean(band_dirs_excl)) if band_dirs_excl else None,
            "band_d_dir_max": float(np.max(band_dirs_excl)) if band_dirs_excl else None,
            "d_dir_percentile_in_band": float(np.mean([v >= d_dir_ab for v in band_dirs_excl]) * 100)
            if band_dirs_excl else None,
        }
        print(f"\n[C] {a}|{b}: d_feat={d_feat_ab:.4f} d_dir={d_dir_ab:.2f} "
              f"band_n={len(band_dirs_excl)} pctile={matched[f'{a}|{b}']['d_dir_percentile_in_band']}")

    # ridge LOO per-mutant error, collision members vs label permutation
    errs_obs = loo_ridge_mae(names, pops, wt_arr, enc_src_pos)
    coll_members = ["SrcKD-L410A", "SrcKD-F405A", "SrcKD-L325A", "SrcKD-V331A"]
    obs_mean_coll = float(np.mean([errs_obs[m] for m in coll_members]))
    obs_mean_non = float(np.mean([errs_obs[m] for m in names if m not in coll_members]))
    n_perm_loo = 10000
    diff_null = np.zeros(n_perm_loo)
    for it in range(n_perm_loo):
        perm = RNG.permutation(len(names))
        p_perm = pops[perm]
        errs_perm = loo_ridge_mae(names, p_perm, wt_arr, enc_src_pos)
        diff_null[it] = float(np.mean([errs_perm[m] for m in coll_members]) -
                              np.mean([errs_perm[m] for m in names if m not in coll_members]))
    results["loo_error_null_src_pos"] = {
        "collision_members": coll_members,
        "observed_mean_mae_collision": obs_mean_coll,
        "observed_mean_mae_non_collision": obs_mean_non,
        "observed_diff": float(obs_mean_coll - obs_mean_non),
        "perm_mean_diff": float(diff_null.mean()),
        "perm_pct": {"5": float(np.percentile(diff_null, 5)),
                     "50": float(np.percentile(diff_null, 50)),
                     "95": float(np.percentile(diff_null, 95))},
        "p_perm_diff_ge_obs": float(np.mean(diff_null >= obs_mean_coll - obs_mean_non)),
        "per_mutant_mae": {m: errs_obs[m] for m in names},
    }
    print(f"\n[C] ridge LOO: collision members MAE={obs_mean_coll:.3f} "
          f"vs non={obs_mean_non:.3f} (diff={obs_mean_coll - obs_mean_non:.3f}, "
          f"perm50={np.median(diff_null):.3f})")

    results["runtime_seconds"] = float(time.time() - t0)
    out_path = OUT / "t6_collision_null.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
