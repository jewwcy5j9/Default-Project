"""T1: Nested CV with encoding selection (post-selection audit, Abl1 core).

The reported LLR advantage was obtained after scanning encodings on the
same labels. Nested CV removes selection bias: for each external LOO
fold, the encoding is chosen by an internal LOO over the training fold
(candidates: pos markers / LLR / Extended), and the chosen model predicts
the external test point.

Output: results/t1_nested_cv.json
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

from k3_data import ABL1_K3, ABL1_K3_WT_POP, ABL1_SEQ_LEN
from k3_benchmark import train_one_seed
from alternative_encodings import DDG_DATA
from k3_llr_proxy import LLR

OUT = Path(__file__).resolve().parent / "results"
CORE = [m for m in ABL1_K3 if m not in ("WT", "H396P", "M290L_H396P")]


def enc_pos(name, data):
    enc = np.zeros(4)
    enc[0] = data["pos"] / ABL1_SEQ_LEN
    for i, p in enumerate([290, 301, 382]):
        if data["pos"] == p:
            enc[i + 1] = 1.0
    return enc


def make_llr_encoder(train_names):
    scale = max(abs(LLR[m]) for m in train_names)

    def enc_llr(name, data):
        enc = np.zeros(5)
        enc[0] = data["pos"] / ABL1_SEQ_LEN
        enc[1] = LLR.get(name, 0.0) / scale
        for i, p in enumerate([290, 301, 382]):
            if data["pos"] == p:
                enc[i + 2] = 1.0
        return enc

    return enc_llr


def enc_ext(name, data):
    from k3_data import enc_abl1_extended
    return enc_abl1_extended(name, data)


ENCODER_DIMS = {"pos": 4, "LLR": 5, "Extended": 10}


def get_encoder(enc_name, train_names):
    if enc_name == "pos":
        return enc_pos
    if enc_name == "LLR":
        return make_llr_encoder(train_names)
    if enc_name == "Extended":
        return enc_ext
    raise KeyError(enc_name)


def fit_predict(train_names, test_name, enc_fn, d, wt, n_seeds=5, n_epochs=800):
    c_tr = np.array([enc_fn(m, ABL1_K3[m]) for m in train_names])
    w_tr = np.tile(wt, (len(train_names), 1))
    y_tr = np.array([ABL1_K3[m]["pop"] for m in train_names])
    preds = []
    for seed in range(n_seeds):
        model = train_one_seed(w_tr, c_tr, y_tr, d=d,
                               seed=seed * 100 + CORE.index(test_name),
                               n_epochs=n_epochs, K=3)
        with torch.no_grad():
            p = model(torch.FloatTensor([wt]),
                      torch.FloatTensor([enc_fn(test_name, ABL1_K3[test_name])]))
        preds.append(p.numpy()[0])
    return np.mean(preds, axis=0)


def internal_loo_cv(train_names, enc_name, wt, n_seeds=5, n_epochs=800):
    errs = []
    for held in train_names:
        tr = [m for m in train_names if m != held]
        enc_fn = get_encoder(enc_name, tr)
        p = fit_predict(tr, held, enc_fn, ENCODER_DIMS[enc_name], wt,
                        n_seeds=n_seeds, n_epochs=n_epochs)
        errs.append(float(np.abs(p - np.array(ABL1_K3[held]["pop"])).mean()))
    return float(np.mean(errs))


def nested_loo():
    wt = np.array(ABL1_K3_WT_POP, dtype=float)
    out = {"per_mutant": {}, "selection": {}, "chosen": {}}
    for test in CORE:
        train = [m for m in CORE if m != test]
        scores = {}
        for enc_name in ENCODER_DIMS:
            scores[enc_name] = internal_loo_cv(train, enc_name, wt)
        best = min(scores, key=scores.get)
        p = fit_predict(train, test, get_encoder(best, train),
                        ENCODER_DIMS[best], wt)
        out["per_mutant"][test] = float(np.abs(p - np.array(ABL1_K3[test]["pop"])).mean())
        out["selection"][test] = {k: round(v, 4) for k, v in scores.items()}
        out["chosen"][test] = best
        print(f"  {test:<14} chosen={best:<9} score={scores[best]:.4f} "
              f"err={out['per_mutant'][test]:.4f}")
    out["nested_mae"] = float(np.mean(list(out["per_mutant"].values())))
    return out


def main():
    t0 = time.time()
    print("=" * 90)
    print("T1: nested LOO with encoding selection (Abl1 core, K=3)")
    print("=" * 90)
    results = nested_loo()
    results["protocol"] = (
        "Nested LOO; each LLR scale is fit on its corresponding training "
        "fold only; candidates fixed to pos/LLR/Extended"
    )
    results["reference"] = {
        "LLR_fixed_mae": 0.1629, "pos_fixed_mae": 0.2757,
        "Extended_fixed_mae": 0.3003, "training_mean": 0.2329,
    }
    (OUT / "t1_nested_cv.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nnested MAE = {results['nested_mae']:.4f} "
          f"(fixed LLR 0.1629, training-mean 0.2329)  [{time.time()-t0:.0f}s]")
    print("[OK] t1_nested_cv.json written")


if __name__ == "__main__":
    main()
