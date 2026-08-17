"""T5b: (i) K=2 vs K=3 unified comparison on the shared contrast axis u1
(active vs non-active) with the same CLR-GP model; (ii) AF2 coverage
ICC decomposition across MSA seeds / model seeds / runs."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from k3_data import SRC_K3, SRC_K3_WT_POP
from p1_core_baselines import clr, inv_clr, loo_model
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

OUT = Path(__file__).resolve().parent / "results"
SRC_CORE = {m: SRC_K3[m] for m in SRC_K3 if m != "SrcKD-WT"}


def enc_src_pos(name, data):
    return np.array([data["pos"] / 536.0, float(data["pos"] == 311),
                     float(data["pos"] == 332), float(data["pos"] == 380)])


def gp_fn():
    return (GaussianProcessRegressor(
        kernel=RBF(length_scale=1.0) + WhiteKernel(0.01),
        alpha=1e-4, normalize_y=True, random_state=0), StandardScaler())


def main():
    out = {}

    # ---- (i) K=2 (pooled) CLR-GP pos: non-active MAE -> u1-scale MAE ----
    # inline LOO; pooled targets are 1-dim (non-active fraction), use the
    # logit link (CLR degenerates for K=2): z = logit(p), p = sigmoid(z).
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    names = list(SRC_CORE.keys())
    X = np.array([enc_src_pos(m, SRC_CORE[m]) for m in names])
    y_na = np.array([[1.0 - SRC_CORE[m]["pop"][0]] for m in names])
    errs = {}
    for i, held in enumerate(names):
        tr = [j for j in range(len(names)) if j != i]
        z_tr = np.log(np.clip(y_na[tr], 1e-6, 1.0 - 1e-6) /
                      (1.0 - np.clip(y_na[tr], 1e-6, 1.0 - 1e-6)))
        model, scaler = gp_fn()
        model.fit(scaler.fit_transform(X[tr]), z_tr)
        z = model.predict(scaler.transform(X[i:i + 1]))[0]
        p = sigmoid(float(z))
        errs[held] = float(abs(p - y_na[i][0]))
    na_mae = float(np.mean(list(errs.values())))
    out["k2_clrgp_pos"] = {"non_active_mae": na_mae,
                           "u1_scale_mae": 2.0 * na_mae}
    print(f"K=2 CLR-GP pos: non-active MAE={na_mae:.4f} "
          f"(u1-scale MAE={2*na_mae:.4f})")

    # ---- (ii) AF2 coverage ICC ----
    af2 = json.loads((Path(__file__).resolve().parent.parent / "af2_subsample"
                      / "results" / "state_classifications.json")
                     .read_text(encoding="utf-8"))
    recs = af2["classifications"]
    # non-active indicator per PDB
    rows = []
    for r in recs:
        rows.append((r["mutant"], r["run"], r["model"], r["seed"],
                     1.0 if r["state"] != "active" else 0.0))
    import collections
    by_mut = collections.defaultdict(list)
    for m, run, model, seed, v in rows:
        by_mut[m].append((run, model, seed, v))

    def icc(groups):
        # groups: list of lists of values; one-way random-effects ICC
        grand = np.mean([v for g in groups for v in g])
        k = len(groups)
        n_j = np.array([len(g) for g in groups])
        ss_b = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
        ss_w = sum(sum((v - np.mean(g)) ** 2 for v in g) for g in groups)
        n = int(n_j.sum())
        ms_b = ss_b / (k - 1)
        ms_w = ss_w / (n - k)
        n0 = (n - sum(n_j ** 2) / n) / (k - 1)
        icc_val = (ms_b - ms_w) / (ms_b + (n0 - 1) * ms_w)
        return float(icc_val), float(ms_b), float(ms_w)

    per_mut = {}
    for m, vals in by_mut.items():
        # group by model seed (5 groups x 24), by MSA seed (8 x 15), by run (3 x 40)
        by_model = collections.defaultdict(list)
        by_msa = collections.defaultdict(list)
        by_run = collections.defaultdict(list)
        for run, model, seed, v in vals:
            by_model[model].append(v)
            by_msa[seed].append(v)
            by_run[run].append(v)
        per_mut[m] = {
            "icc_model": icc(list(by_model.values()))[0],
            "icc_msa": icc(list(by_msa.values()))[0],
            "icc_run": icc(list(by_run.values()))[0],
            "coverage_mean": float(np.mean([v for *_, v in vals])),
            "n_pdb": len(vals),
        }
    out["af2_icc"] = per_mut
    for m, v in per_mut.items():
        print(f"{m}: cov={v['coverage_mean']:.3f} "
              f"ICC(model)={v['icc_model']:.2f} ICC(msa)={v['icc_msa']:.2f} "
              f"ICC(run)={v['icc_run']:.2f}")
    # mutant-level: coverage means per mutant
    covs = np.array([v["coverage_mean"] for v in per_mut.values()])
    out["af2_mutant_level"] = {
        "n_mutants": len(covs),
        "mean": float(covs.mean()),
        "std_across_mutants": float(covs.std()),
        "range": [float(covs.min()), float(covs.max())]}
    print(f"mutant-level coverage: mean={covs.mean():.3f} "
          f"std={covs.std():.4f} range=[{covs.min():.3f},{covs.max():.3f}]")

    (OUT / "t5b_review_responses.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[OK] t5b_review_responses.json")


if __name__ == "__main__":
    main()
