"""T2: Cone-constrained direction-transfer diagnostic (Abl1).

Descriptive diagnostic tied to Prop. 3 (direction transfer): if all
training shifts lie in one half-space of a fixed contrast u (the six gold
mutants are I2-dominant), any model whose predictions stay in the cone of
the training directions must transfer the training direction to a test
shift with <u, d_t> < 0 (the reserved I1-dominant H396P mutants).

Falsifiable prediction: train on the 6 gold mutants (all I2-direction),
predict the reserved I1-direction mutants H396P and M290L_H396P. If the
tested models behave in a cone-constrained way, the predictions should
collapse toward WT or toward the training's I2 direction --- large error
and wrong direction --- in contrast to the in-support leave-site-out
results (~0.08-0.21). Caveat: the model outputs are not verified to lie
in the training cone and the models are not scalar channels, so the
result is consistent with---but does not prove---cone behavior.

Output: results/t2_i1_falsifiable.json
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
GOLD = [m for m in ABL1_K3 if m not in ("WT", "H396P", "M290L_H396P")]
I1_TEST = ["H396P", "M290L_H396P"]


def enc_ddg(name, data):
    enc = np.zeros(5)
    enc[0] = data["pos"] / ABL1_SEQ_LEN
    enc[1] = DDG_DATA.get(name, 0.0) / 3.5
    for i, p in enumerate([290, 301, 382]):
        if data["pos"] == p:
            enc[i + 2] = 1.0
    return enc


def enc_llr(name, data):
    enc = np.zeros(5)
    enc[0] = data["pos"] / ABL1_SEQ_LEN
    enc[1] = LLR.get(name, 0.0) / max(abs(v) for v in LLR.values())
    for i, p in enumerate([290, 301, 382]):
        if data["pos"] == p:
            enc[i + 2] = 1.0
    return enc


def predict(train_names, test_name, enc_fn, d, wt, n_seeds=5, n_epochs=800):
    c_tr = np.array([enc_fn(m, ABL1_K3[m]) for m in train_names])
    w_tr = np.tile(wt, (len(train_names), 1))
    y_tr = np.array([ABL1_K3[m]["pop"] for m in train_names])
    preds = []
    for seed in range(n_seeds):
        model = train_one_seed(w_tr, c_tr, y_tr, d=d,
                               seed=seed * 100 + GOLD.index(train_names[0]) + seed,
                               n_epochs=n_epochs, K=3)
        with torch.no_grad():
            p = model(torch.FloatTensor([wt]),
                      torch.FloatTensor([enc_fn(test_name, ABL1_K3[test_name])]))
        preds.append(p.numpy()[0])
    return np.mean(preds, axis=0)


def main():
    t0 = time.time()
    print("=" * 90)
    print("T2: scalar-channel dimensionality, I1-direction falsifiable test")
    print("=" * 90)
    wt = np.array(ABL1_K3_WT_POP, dtype=float)
    results = {"gold_train": GOLD, "i1_test": I1_TEST, "per_mutant": {}}

    for enc_name, fn, d in [("variant_C", enc_ddg, 5), ("LLR", enc_llr, 5)]:
        results[enc_name] = {}
        for test in I1_TEST:
            p = predict(GOLD, test, fn, d, wt)
            true = np.array(ABL1_K3[test]["pop"], dtype=float)
            err = float(np.abs(p - true).mean())
            shift_t = true - wt
            shift_p = p - wt
            dir_ok = bool(np.dot(shift_p, shift_t) > 0)
            cw_err = float(np.abs(wt - true).mean())
            results[enc_name][test] = {
                "true": true.tolist(), "pred": p.tolist(),
                "mae": err, "constant_wt_mae": cw_err,
                "direction_ok": dir_ok,
                "shift_true": shift_t.tolist(), "shift_pred": shift_p.tolist()}
            print(f"  [{enc_name}] {test:<16} true={np.round(true,2)} "
                  f"pred={np.round(p,2)} mae={err:.4f} (const-WT {cw_err:.4f}) "
                  f"dir_ok={dir_ok}")

    # in-support reference: leave-F382-out on LLR (from P0-2, support-in)
    results["in_support_reference"] = {"LLR_F382_leave_out_mae": 0.2058,
                                       "variantC_F382_leave_out_mae": 0.2204}
    (OUT / "t2_i1_falsifiable.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] t2_i1_falsifiable.json  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
