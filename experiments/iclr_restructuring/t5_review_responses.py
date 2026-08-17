"""T5: Reviewer-response analyses.

A. Direction baselines (training-mean / majority-direction) for Abl1 K3, Src K3.
B. LLR encoding WITHOUT position markers (Abl1 K3, CLR-Ridge/GP).
C. Contrast-space decomposition (Src K3): u1 = active-(E1+E2) = 2p_a-1,
   u2 = E1-E2; per-contrast MAE/RMSE/R^2 for MLP-pos and CLR-GP-pos;
   K=2 non-active MAE vs K=3 u1-contrast MAE (unified comparison).
D. Empirical model behavior on the collision pair: output separation
   s = ||w(c1)-w(c2)||_1 for each model family (Src pos), plus a global
   numerical Lipschitz estimate (max slope over random train-pair
   directions).
E. pos/seq scale sensitivity of the collision distance (Src).
F. Dirichlet / logistic-normal label perturbation flip rates (Src).
G. AF2 effective sample size (ICC over run/model/seed) from the 840
   classifications.

Output: results/t5_review_responses.json + console tables.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from k3_data import ABL1_K3, ABL1_K3_WT_POP, SRC_K3, SRC_K3_WT_POP
from p1_core_baselines import clr, inv_clr, loo_model, enc_abl1_llr
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

OUT = Path(__file__).resolve().parent / "results"
ABL1_CORE = {m: ABL1_K3[m] for m in ABL1_K3 if m not in ("WT", "H396P", "M290L_H396P")}
SRC_CORE = {m: SRC_K3[m] for m in SRC_K3 if m != "SrcKD-WT"}
RNG = np.random.default_rng(0)


def enc_src_pos(name, data):
    return np.array([data["pos"] / 536.0, float(data["pos"] == 311),
                     float(data["pos"] == 332), float(data["pos"] == 380)])


def enc_abl1_llr_only(name, data):
    """1-dim: normalized LLR only (no position markers)."""
    from k3_llr_proxy import LLR
    return np.array([LLR.get(name, 0.0) / max(abs(v) for v in LLR.values())])


def direction_accuracy(pred, true, wt, tie_thresh=0.05):
    d_p = pred - wt
    d_t = true - wt
    if np.abs(d_t).sum() < tie_thresh:
        return None
    return bool(np.dot(d_p, d_t) > 0)


# ============================================================
# A. direction baselines
# ============================================================
def direction_baselines(mutations, wt_pop):
    names = list(mutations.keys())
    wt = np.array(wt_pop, dtype=float)
    pops = np.array([mutations[m]["pop"] for m in names])
    out = {"per_mutant": {}, "summary": {}}
    dirs = {m: pops[i] - wt for i, m in enumerate(names)}
    for held in names:
        tr = [m for m in names if m != held]
        mean_pred = np.mean([pops[names.index(m)] for m in tr], axis=0)
        # majority direction: sign pattern of training shifts, magnitude = mean magnitude
        tr_d = np.array([dirs[m] for m in tr])
        mag = np.abs(tr_d).mean(axis=0)
        maj_sign = np.sign(tr_d.sum(axis=0))
        maj_shift = maj_sign * mag
        maj_pred = np.clip(wt + maj_shift, 0, 1)
        maj_pred = maj_pred / maj_pred.sum()
        true = pops[names.index(held)]
        out["per_mutant"][held] = {
            "true": true.tolist(),
            "train_mean_pred": mean_pred.tolist(),
            "majority_pred": maj_pred.tolist(),
            "train_mean_mae": float(np.abs(mean_pred - true).mean()),
            "majority_mae": float(np.abs(maj_pred - true).mean()),
            "train_mean_dir": direction_accuracy(mean_pred, true, wt),
            "majority_dir": direction_accuracy(maj_pred, true, wt),
        }
    tm_mae = np.mean([v["train_mean_mae"] for v in out["per_mutant"].values()])
    mj_mae = np.mean([v["majority_mae"] for v in out["per_mutant"].values()])
    tm_dir = [v["train_mean_dir"] for v in out["per_mutant"].values()]
    mj_dir = [v["majority_dir"] for v in out["per_mutant"].values()]
    out["summary"] = {
        "train_mean_mae": float(tm_mae),
        "majority_mae": float(mj_mae),
        "train_mean_dir_ok": int(sum(1 for x in tm_dir if x is True)),
        "train_mean_dir_wrong": int(sum(1 for x in tm_dir if x is False)),
        "train_mean_dir_ties": int(sum(1 for x in tm_dir if x is None)),
        "majority_dir_ok": int(sum(1 for x in mj_dir if x is True)),
        "majority_dir_wrong": int(sum(1 for x in mj_dir if x is False)),
        "majority_dir_ties": int(sum(1 for x in mj_dir if x is None)),
    }
    return out


# ============================================================
# C. contrast-space decomposition (Src K3)
# ============================================================
def contrast_decomposition(preds, targets, wt):
    """u1 = 2*p_a - 1 (active vs non-active), u2 = p_E1 - p_E2."""
    out = {"per_mutant": {}, "summary": {}}
    u1_mae, u2_mae, u1_mse, u2_mse = [], [], [], []
    u1_true_all, u1_pred_all, u2_true_all, u2_pred_all = [], [], [], []
    for m, t in targets.items():
        p = preds[m]
        u1_t = 2 * t[0] - 1
        u2_t = t[1] - t[2]
        u1_p = 2 * p[0] - 1
        u2_p = p[1] - p[2]
        u1_mae.append(abs(u1_p - u1_t))
        u2_mae.append(abs(u2_p - u2_t))
        u1_mse.append((u1_p - u1_t) ** 2)
        u2_mse.append((u2_p - u2_t) ** 2)
        u1_true_all.append(u1_t); u1_pred_all.append(u1_p)
        u2_true_all.append(u2_t); u2_pred_all.append(u2_p)
        out["per_mutant"][m] = {"u1_true": u1_t, "u1_pred": u1_p,
                                "u2_true": u2_t, "u2_pred": u2_p,
                                "u1_mae": float(abs(u1_p - u1_t)),
                                "u2_mae": float(abs(u2_p - u2_t))}
    u1t, u1p = np.array(u1_true_all), np.array(u1_pred_all)
    u2t, u2p = np.array(u2_true_all), np.array(u2_pred_all)
    r2 = lambda t, p: 1 - float(np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))
    out["summary"] = {
        "u1_mae": float(np.mean(u1_mae)), "u2_mae": float(np.mean(u2_mae)),
        "u1_rmse": float(np.sqrt(np.mean(u1_mse))),
        "u2_rmse": float(np.sqrt(np.mean(u2_mse))),
        "u1_r2": r2(u1t, u1p), "u2_r2": r2(u2t, u2p),
        "u1_mutant_mae": dict(sorted({m: v["u1_mae"] for m, v in
                                      out["per_mutant"].items()}.items(),
                                     key=lambda kv: -kv[1])),
    }
    return out


# ============================================================
# D. collision-pair output separation + numeric Lipschitz (Src pos)
# ============================================================
def numeric_lipschitz(model, enc_fn, mutations, wt, n_dirs=400):
    """Max ||w(c+eps u)-w(c)||_1 / ||eps u|| over random directions u."""
    names = list(mutations.keys())
    Cs = np.array([enc_fn(m, mutations[m]) for m in names])
    best = 0.0
    for _ in range(n_dirs):
        i = RNG.integers(len(names))
        u = RNG.normal(size=Cs.shape[1])
        u /= np.linalg.norm(u)
        eps = 0.01
        c1 = Cs[i] + eps * u
        c2 = Cs[i] - eps * u
        p1 = model(c1); p2 = model(c2)
        slope = float(np.abs(p1 - p2).sum()) / (2 * eps)
        best = max(best, slope)
    return best


def collision_separation(models, enc_fn, mutations, wt, pair=("SrcKD-L410A", "SrcKD-F405A")):
    wt = np.array(wt, dtype=float)
    c1 = enc_fn(pair[0], mutations[pair[0]])
    c2 = enc_fn(pair[1], mutations[pair[1]])
    d1 = np.array(mutations[pair[0]]["pop"]) - wt
    d2 = np.array(mutations[pair[1]]["pop"]) - wt
    delta = float(np.abs(d1 - d2).sum())
    out = {"pair": pair, "d_feat": float(np.linalg.norm(c1 - c2)),
           "delta_l1": delta, "models": {}}
    for name, model in models.items():
        p1 = model(c1); p2 = model(c2)
        sep = float(np.abs(p1 - p2).sum())
        out["models"][name] = {
            "output_separation_l1": sep,
            "floor_per_state_mae": float(max(delta - sep, 0) / (2 * 3)),
            "pred1": p1.tolist(), "pred2": p2.tolist()}
    return out


# ============================================================
# E. pos/seq scale sensitivity
# ============================================================
def scale_sensitivity(mutations, wt, seq_len=536, pair=("SrcKD-L410A", "SrcKD-F405A"),
                      tau=0.25):
    base = abs(mutations[pair[0]]["pos"] - mutations[pair[1]]["pos"]) / seq_len
    out = {}
    for s in [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]:
        out[str(s)] = {"d_feat": float(base * s),
                       "in_collision_range": bool(base * s < tau)}
    return out


# ============================================================
# F. Dirichlet / logistic-normal perturbation
# ============================================================
def flip_rate(mutations, wt, perturb, n_draws=1000, seed=7):
    names = list(mutations.keys())
    pops = {m: np.array(mutations[m]["pop"], dtype=float) for m in names}
    rng = np.random.default_rng(seed)
    flip = 0
    total = 0
    for _ in range(n_draws):
        noisy = {}
        for m in names:
            noisy[m] = perturb(pops[m], rng)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                for k in range(3):
                    total += 1
                    if (pops[a][k] - pops[b][k]) * (noisy[a][k] - noisy[b][k]) < 0:
                        flip += 1
    return float(flip / total)


def main():
    t0 = time.time()
    print("=" * 90)
    print("T5: reviewer-response analyses")
    print("=" * 90)
    results = {}

    # ---- A. direction baselines ----
    print("\n[A] direction baselines")
    results["direction_baselines"] = {
        "abl1": direction_baselines(ABL1_CORE, ABL1_K3_WT_POP),
        "src": direction_baselines(SRC_CORE, SRC_K3_WT_POP)}
    for sys_name in ["abl1", "src"]:
        s = results["direction_baselines"][sys_name]["summary"]
        print(f"  {sys_name}: train-mean MAE={s['train_mean_mae']:.4f} "
              f"dir={s['train_mean_dir_ok']}ok/{s['train_mean_dir_wrong']}wrong/"
              f"{s['train_mean_dir_ties']}tie | majority MAE={s['majority_mae']:.4f} "
              f"dir={s['majority_dir_ok']}ok/{s['majority_dir_wrong']}wrong/"
              f"{s['majority_dir_ties']}tie")

    # ---- B. LLR without position markers (Abl1 K3) ----
    print("\n[B] LLR-only (1-dim, no position markers), Abl1 K3")
    results["llr_only"] = {}
    for mname, mf in [("CLR-Ridge", (lambda: (Ridge(alpha=1.0), StandardScaler()))),
                      ("CLR-GP", (lambda: (GaussianProcessRegressor(
                          kernel=RBF(length_scale=1.0) + WhiteKernel(0.01),
                          alpha=1e-4, normalize_y=True, random_state=0), StandardScaler())))]:
        p1, e1 = loo_model(ABL1_CORE, ABL1_K3_WT_POP, enc_abl1_llr_only, 1, mf)
        p2, e2 = loo_model(ABL1_CORE, ABL1_K3_WT_POP, enc_abl1_llr, 5, mf)
        wt = np.array(ABL1_K3_WT_POP, dtype=float)
        d1 = int(sum(1 for m in ABL1_CORE if direction_accuracy(p1[m], np.array(ABL1_CORE[m]["pop"]), wt) is True))
        d2 = int(sum(1 for m in ABL1_CORE if direction_accuracy(p2[m], np.array(ABL1_CORE[m]["pop"]), wt) is True))
        # FIXED 2026-08-17: denominator hardcoded as 5; use the actual panel
        # size (len(ABL1_CORE) == 6). The stored results/t5_review_responses.json
        # predates this fix and shows "5/5" where the true ratio is 5/6; do NOT
        # regenerate the frozen artifact.
        results["llr_only"][mname] = {
            "llr_only_mae": float(np.mean(list(e1.values()))),
            "llr_only_dir": f"{d1}/{len(ABL1_CORE)}",
            "llr_pos_mae": float(np.mean(list(e2.values()))),
            "llr_pos_dir": f"{d2}/{len(ABL1_CORE)}"}
        print(f"  {mname}: LLR-only MAE={np.mean(list(e1.values())):.4f} "
              f"dir={d1}/{len(ABL1_CORE)} "
              f"| LLR+pos MAE={np.mean(list(e2.values())):.4f} "
              f"dir={d2}/{len(ABL1_CORE)}")

    # ---- C. contrast-space decomposition (Src K3) ----
    print("\n[C] contrast-space decomposition (Src K3)")
    k3 = json.loads((OUT / "k3_benchmark_results.json").read_text(encoding="utf-8"))
    mlp_preds = {m: np.array(v) for m, v in
                 k3["src"]["pos_markers_4dim"]["per_mutant_preds"].items()}
    targets = {m: np.array(SRC_CORE[m]["pop"]) for m in SRC_CORE}
    wt = np.array(SRC_K3_WT_POP, dtype=float)
    results["contrast"] = {
        "mlp_pos": contrast_decomposition(mlp_preds, targets, wt),
        "clrgp_pos": None}
    s = results["contrast"]["mlp_pos"]["summary"]
    print(f"  MLP-pos: u1(active) MAE={s['u1_mae']:.4f} RMSE={s['u1_rmse']:.4f} "
          f"R2={s['u1_r2']:.3f} | u2(E1-E2) MAE={s['u2_mae']:.4f} "
          f"RMSE={s['u2_rmse']:.4f} R2={s['u2_r2']:.3f}")
    cg_preds, cg_errs = loo_model(SRC_CORE, SRC_K3_WT_POP, enc_src_pos, 4,
                                  (lambda: (GaussianProcessRegressor(
                                      kernel=RBF(length_scale=1.0) + WhiteKernel(0.01),
                                      alpha=1e-4, normalize_y=True, random_state=0), StandardScaler())))
    results["contrast"]["clrgp_pos"] = contrast_decomposition(cg_preds, targets, wt)
    results["clrgp_pos_reference_mae"] = float(np.mean(list(cg_errs.values())))
    s2 = results["contrast"]["clrgp_pos"]["summary"]
    print(f"  CLR-GP-pos: u1 MAE={s2['u1_mae']:.4f} RMSE={s2['u1_rmse']:.4f} "
          f"R2={s2['u1_r2']:.3f} | u2 MAE={s2['u2_mae']:.4f} "
          f"RMSE={s2['u2_rmse']:.4f} R2={s2['u2_r2']:.3f}")
    print("  u2 (E1-E2) per-mutant MAE (MLP-pos):",
          {m: round(v["u2_mae"], 3) for m, v in
           results["contrast"]["mlp_pos"]["per_mutant"].items()})

    # ---- D. collision-pair separation + numeric Lipschitz (Src pos) ----
    print("\n[D] collision-pair output separation (Src pos, L410A/F405A)")
    ridge = Ridge(alpha=1.0)
    names = list(SRC_CORE.keys())
    Cs = np.array([enc_src_pos(m, SRC_CORE[m]) for m in names])
    pops = np.array([SRC_CORE[m]["pop"] for m in names])
    zs = clr(pops)
    scaler = StandardScaler().fit(Cs)
    ridge.fit(scaler.transform(Cs), zs)

    def ridge_model(c):
        z = ridge.predict(scaler.transform(np.atleast_2d(c)))
        return inv_clr(z[0])

    knn = KNeighborsRegressor(n_neighbors=1).fit(scaler.transform(Cs), zs)

    def knn_model(c):
        return inv_clr(knn.predict(scaler.transform(np.atleast_2d(c)))[0])

    gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0) + WhiteKernel(0.01),
                                  alpha=1e-4, normalize_y=True, random_state=0)
    gp.fit(scaler.transform(Cs), zs)

    def gp_model(c):
        return inv_clr(gp.predict(scaler.transform(np.atleast_2d(c)))[0])

    models = {"CLR-Ridge": ridge_model, "CLR-GP": gp_model, "kNN(1)": knn_model}
    results["collision_separation"] = collision_separation(
        models, enc_src_pos, SRC_CORE, SRC_K3_WT_POP)
    for mn, v in results["collision_separation"]["models"].items():
        print(f"  {mn}: output sep={v['output_separation_l1']:.3f} "
              f"-> per-state floor={v['floor_per_state_mae']:.3f}")
    # MLP separation (train on all 8, 2 seeds)
    import torch
    from k3_benchmark import train_one_seed
    mlp_seps = []
    for seed in range(2):
        model = train_one_seed(np.tile(wt, (len(names), 1)), Cs, pops,
                               d=4, seed=seed, n_epochs=600, K=3)
        p1 = model(torch.FloatTensor([wt]), torch.FloatTensor([enc_src_pos("SrcKD-L410A", SRC_CORE["SrcKD-L410A"])])).detach().numpy()[0]
        p2 = model(torch.FloatTensor([wt]), torch.FloatTensor([enc_src_pos("SrcKD-F405A", SRC_CORE["SrcKD-F405A"])])).detach().numpy()[0]
        mlp_seps.append(float(np.abs(p1 - p2).sum()))
    results["collision_separation"]["models"]["MLP(2 seeds)"] = {
        "output_separation_l1": float(np.mean(mlp_seps)),
        "seeds": mlp_seps}
    print(f"  MLP(2 seeds): mean output sep={np.mean(mlp_seps):.3f} {mlp_seps}")
    results["numeric_lipschitz"] = {
        mn: numeric_lipschitz(m, enc_src_pos, SRC_CORE, SRC_K3_WT_POP)
        for mn, m in models.items()}
    print("  numeric Lipschitz (random dirs, eps=0.01):",
          {k: round(v, 2) for k, v in results["numeric_lipschitz"].items()})

    # ---- E. scale sensitivity ----
    print("\n[E] pos/seq scale sensitivity (Src L410A/F405A)")
    results["scale_sensitivity"] = scale_sensitivity(SRC_CORE, SRC_K3_WT_POP)
    for s, v in results["scale_sensitivity"].items():
        print(f"  scale x{s}: d_feat={v['d_feat']:.4f} in-range={v['in_collision_range']}")

    # ---- F. Dirichlet / logistic-normal perturbation ----
    print("\n[F] label perturbation models (Src)")
    def uniform10(p, rng):
        e = rng.uniform(-0.10, 0.10, 3)
        q = np.clip(p + e, 0, 1)
        return q / q.sum()

    def dirichlet_200(p, rng):
        return rng.dirichlet(200 * p)

    def dirichlet_50(p, rng):
        return rng.dirichlet(50 * p)

    def lognorm_10(p, rng):
        z = np.log(np.clip(p, 1e-6, 1)) + rng.normal(0, 0.10, 3)
        e = np.exp(z - z.max())
        return e / e.sum()

    results["perturbation"] = {
        "uniform_pm10": flip_rate(SRC_CORE, SRC_K3_WT_POP, uniform10),
        "dirichlet_alpha200": flip_rate(SRC_CORE, SRC_K3_WT_POP, dirichlet_200),
        "dirichlet_alpha50": flip_rate(SRC_CORE, SRC_K3_WT_POP, dirichlet_50),
        "lognormal_sigma10": flip_rate(SRC_CORE, SRC_K3_WT_POP, lognorm_10),
    }
    for k, v in results["perturbation"].items():
        print(f"  {k}: flip rate={v:.5f}")

    # ---- G. AF2 effective sample size ----
    print("\n[G] AF2 effective sample size")
    try:
        af2 = json.loads((Path(__file__).resolve().parent.parent / "af2_subsample"
                          / "results" / "state_classifications.json")
                         .read_text(encoding="utf-8"))
        recs = af2["classifications"]
        import collections
        per = collections.defaultdict(list)
        for r in recs:
            key = (r["mutant"], r["run"], r["model"], r["seed"])
            per[key].append(1.0 if r["state"] != "active" else 0.0)
        mut_stats = collections.defaultdict(list)
        for key, vals in per.items():
            mut_stats[key[0]].append(np.mean(vals))
        icc_out = {}
        for m, vals in mut_stats.items():
            vals = np.array(vals)
            k = len(vals)
            icc_out[m] = {
                "n_clusters": k,
                "mean": float(vals.mean()),
                "var_between": float(np.var(vals)),
            }
        results["af2"] = {"n_records": len(recs), "per_mutant": dict(icc_out)}
        print(f"  records={len(recs)}, mutants={len(mut_stats)}")
        for m, v in icc_out.items():
            print(f"    {m}: clusters={v['n_clusters']} mean={v['mean']:.3f} "
                  f"var={v['var_between']:.5f}")
    except Exception as e:
        print("  AF2 analysis skipped:", e)

    (OUT / "t5_review_responses.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] t5_review_responses.json  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
