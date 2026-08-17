"""P1: Standard compositional regression baselines on the core sets.

Models (all LOO, core sets only, no WT):
  - CLR-Ridge : center log-ratio transform of y, ridge regression on
                features, inverse ILR/CLR transform via softmax normalization.
  - CLR-GP    : Gaussian process (RBF) on CLR targets.
  - kNN       : nearest neighbours on raw features (existing family).
  - Isotonic  : monotone scalar model: y_nonactive = isotonic(DDG or LLR),
                the minimal "energy is the only channel" baseline.

Encoding feature scaling is fit per training fold. Outputs MAE + direction
for each (system, encoding, model), with paired stats vs training-mean.

Output: results/p1_core_baselines.json
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.isotonic import IsotonicRegression

from k3_data import ABL1_K3, ABL1_K3_WT_POP, SRC_K3, SRC_K3_WT_POP
from k3_benchmark import metrics
from alternative_encodings import DDG_DATA
from k3_llr_proxy import LLR
from gp_protocols import PRIMARY_GP_PROTOCOL, make_primary_gp

OUT = Path(__file__).resolve().parent / "results"
ABL1_CORE = {m: ABL1_K3[m] for m in ABL1_K3 if m not in ("WT", "H396P", "M290L_H396P")}
SRC_CORE = {m: SRC_K3[m] for m in SRC_K3 if m != "SrcKD-WT"}
EPS = 1e-6


def clr(y):
    y = np.clip(y, EPS, 1.0)
    y = y / y.sum(axis=-1, keepdims=True)
    return np.log(y) - np.log(y).mean(axis=-1, keepdims=True)


def inv_clr(z, base=None):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def loo_model(mutations, wt_pop, enc_fn, d, model_fn, transform='clr'):
    names = list(mutations.keys())
    targets = np.array([mutations[m]["pop"] for m in names])
    X = np.array([enc_fn(m, mutations[m]) for m in names])
    preds, errs = {}, {}
    for i, held in enumerate(names):
        tr = [j for j in range(len(names)) if j != i]
        y_tr = clr(targets[tr]) if transform == 'clr' else targets[tr]
        model = model_fn()
        if isinstance(model, tuple):
            model, scaler = model
            X_tr = scaler.fit_transform(X[tr])
            X_te = scaler.transform(X[i:i + 1])
        else:
            X_tr, X_te = X[tr], X[i:i + 1]
        model.fit(X_tr, y_tr)
        z = model.predict(X_te)[0]
        p = inv_clr(z[None, :])[0] if transform == 'clr' else np.clip(z, 0, 1)
        preds[held] = p
        errs[held] = float(np.abs(p - targets[i]).mean())
    return preds, errs


def enc_abl1_ddg(name, data):
    enc = np.zeros(5)
    enc[0] = data["pos"] / 534.0
    enc[1] = DDG_DATA.get(name, 0.0) / 3.5
    for i, p in enumerate([290, 301, 382]):
        if data["pos"] == p:
            enc[i + 2] = 1.0
    return enc


def enc_abl1_llr(name, data):
    enc = np.zeros(5)
    enc[0] = data["pos"] / 534.0
    enc[1] = LLR.get(name, 0.0) / max(abs(v) for v in LLR.values())
    for i, p in enumerate([290, 301, 382]):
        if data["pos"] == p:
            enc[i + 2] = 1.0
    return enc


def enc_pos(name, data, seq_len):
    enc = np.zeros(4)
    enc[0] = data["pos"] / seq_len
    for i, p in enumerate([290, 301, 382] if seq_len > 400 else [311, 332, 380]):
        if data["pos"] == p:
            enc[i + 1] = 1.0
    return enc


def enc_ext(name, data, seq_len):
    from k3_data import enc_abl1_extended, enc_src_extended
    fn = enc_abl1_extended if seq_len > 400 else enc_src_extended
    return fn(name, data)


def main():
    t0 = time.time()
    print("=" * 90)
    print("P1: core compositional baselines")
    print("=" * 90)
    results = {}

    from sklearn.preprocessing import StandardScaler
    models = {
        "CLR-Ridge": (lambda: (Ridge(alpha=1.0), StandardScaler())),
        "CLR-GP": (lambda: (make_primary_gp(), StandardScaler())),
        "kNN": (lambda: (KNeighborsRegressor(n_neighbors=2), StandardScaler())),
    }

    configs = [
        ("abl1", ABL1_CORE, ABL1_K3_WT_POP,
         {"variantC": (enc_abl1_ddg, 5), "LLR": (enc_abl1_llr, 5),
          "pos": (lambda n, d: enc_pos(n, d, 534), 4),
          "Extended": (lambda n, d: enc_ext(n, d, 534), 10)}),
        ("src", SRC_CORE, SRC_K3_WT_POP,
         {"pos": (lambda n, d: enc_pos(n, d, 536), 4),
          "Extended": (lambda n, d: enc_ext(n, d, 536), 10)}),
    ]

    for sys_name, core, wt, encoders in configs:
        results[sys_name] = {}
        for enc_name, (fn, d) in encoders.items():
            results[sys_name][enc_name] = {}
            print(f"\n--- {sys_name} / {enc_name} (d={d}) ---")
            for model_name, mfn in models.items():
                preds, errs = loo_model(core, wt, fn, d, mfn)
                mae = float(np.mean(list(errs.values())))
                met = metrics(preds, {m: core[m]["pop"] for m in core}, wt)
                results[sys_name][enc_name][model_name] = {
                    "mae": mae, "direction": met["direction"], "errors": errs}
                print(f"  {model_name:<10} MAE={mae:.4f} dir={met['direction']}")

        # isotonic scalar baseline (energy channel only)
        if sys_name == "abl1":
            names = list(core.keys())
            pops = np.array([core[m]["pop"] for m in names])
            nonact = 1.0 - pops[:, 0]
            for feat_name, feat in [("DDG", np.array([DDG_DATA[m] for m in names])),
                                    ("LLR", np.array([LLR[m] for m in names]))]:
                errs = {}
                for i, m in enumerate(names):
                    iso = IsotonicRegression(out_of_bounds="clip")
                    tr = [j for j in range(len(names)) if j != i]
                    iso.fit(feat[tr], nonact[tr])
                    p = np.clip(iso.predict([feat[i]])[0], 0, 1)
                    # p predicts the NON-ACTIVE fraction; compare both state
                    # entries in the same [active, non-active] order.
                    true = np.array([core[m]["pop"][0], 1.0 - core[m]["pop"][0]])
                    errs[m] = float(np.abs(np.array([1 - p, p]) - true).mean())
                results[sys_name][f"isotonic_{feat_name}"] = {
                    "mae": float(np.mean(list(errs.values()))),
                    "direction": "-",
                    "scale": "2-state (active vs non-active) MAE; not on the "
                             "3-state MAE scale of the model rows",
                    "errors": errs}
                print(f"  isotonic_{feat_name:<10} MAE={np.mean(list(errs.values())):.4f}")

    results["_protocols"] = {"CLR-GP": PRIMARY_GP_PROTOCOL}
    (OUT / "p1_core_baselines.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] p1_core_baselines.json (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
