"""P0-5: Fold-local preprocessing re-run for the key Abl1 encodings.

variant-C and LLR use a max-|value| normalization constant; the original
runs (and k3_benchmark) normalize with the global max over ALL mutants,
which leaks the held-out example's scale into the training fold.

Here the normalization constant is recomputed per LOO fold from the
training mutants only. We also re-run pos-markers as a no-normalization
control. Output: results/p0_fold_local_results.json
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
from k3_benchmark import train_one_seed, metrics
from alternative_encodings import DDG_DATA
from k3_llr_proxy import LLR

OUT = Path(__file__).resolve().parent / "results"
CORE = [m for m in ABL1_K3 if m not in ("WT", "H396P", "M290L_H396P")]


def enc_ddg_foldlocal(name, data, train_names):
    """[pos/seq, ddg/max_train, pos290, pos301, pos382]."""
    enc = np.zeros(5)
    enc[0] = data["pos"] / ABL1_SEQ_LEN
    norm = max(abs(DDG_DATA[m]) for m in train_names) or 1.0
    enc[1] = DDG_DATA.get(name, 0.0) / norm
    if data["pos"] == 290:
        enc[2] = 1.0
    elif data["pos"] == 301:
        enc[3] = 1.0
    elif data["pos"] == 382:
        enc[4] = 1.0
    return enc


def enc_llr_foldlocal(name, data, train_names):
    """[pos/seq, llr/max_train, pos290, pos301, pos382]."""
    enc = np.zeros(5)
    enc[0] = data["pos"] / ABL1_SEQ_LEN
    norm = max(abs(LLR[m]) for m in train_names) or 1.0
    enc[1] = LLR.get(name, 0.0) / norm
    if data["pos"] == 290:
        enc[2] = 1.0
    elif data["pos"] == 301:
        enc[3] = 1.0
    elif data["pos"] == 382:
        enc[4] = 1.0
    return enc


def enc_pos(name, data):
    enc = np.zeros(4)
    enc[0] = data["pos"] / ABL1_SEQ_LEN
    for i, p in enumerate([290, 301, 382]):
        if data["pos"] == p:
            enc[i + 1] = 1.0
    return enc


def loo_foldlocal(enc_fn, d, n_seeds=5, n_epochs=800):
    wt = np.array(ABL1_K3_WT_POP, dtype=float)
    preds, errs = {}, {}
    for held in CORE:
        train = [m for m in CORE if m != held]
        c_tr = np.array([enc_fn(m, ABL1_K3[m], train) for m in train])
        w_tr = np.tile(wt, (len(train), 1))
        y_tr = np.array([ABL1_K3[m]["pop"] for m in train])
        seed_preds = []
        for seed in range(n_seeds):
            model = train_one_seed(w_tr, c_tr, y_tr, d=d,
                                   seed=seed * 100 + CORE.index(held),
                                   n_epochs=n_epochs, K=3)
            with torch.no_grad():
                p = model(torch.FloatTensor([wt]),
                          torch.FloatTensor([enc_fn(held, ABL1_K3[held], train)]))
            seed_preds.append(p.numpy()[0])
        preds[held] = np.mean(seed_preds, axis=0)
        errs[held] = float(np.abs(preds[held] - np.array(ABL1_K3[held]["pop"])).mean())
    return preds, errs


def main():
    t0 = time.time()
    print("=" * 90)
    print("P0-5: fold-local normalization (Abl1 core, K=3)")
    print("=" * 90)
    results = {}
    targets = {m: ABL1_K3[m]["pop"] for m in CORE}
    for enc_name, fn, d in [
        ("variant_C_foldlocal", enc_ddg_foldlocal, 5),
        ("LLR_foldlocal", enc_llr_foldlocal, 5),
    ]:
        preds, errs = loo_foldlocal(fn, d)
        mae = float(np.mean(list(errs.values())))
        met = metrics(preds, targets, ABL1_K3_WT_POP)
        results[enc_name] = {"mae": mae, "direction": met["direction"],
                             "errors": errs, "preds": {m: v.tolist()
                                                       for m, v in preds.items()}}
        print(f"  {enc_name:<22} MAE={mae:.4f} dir={met['direction']}")
        for m in CORE:
            print(f"      {m:<14} err={errs[m]:.4f}")

    # reference: global-normalized runs (from saved canonical results)
    k3 = json.loads((OUT / "k3_benchmark_results.json").read_text(encoding="utf-8"))
    ref = k3["abl1_core"]["C_ddg_5dim"]["mae"]
    results["reference_global_norm_variantC"] = {"mae": ref}
    print(f"  reference (global-norm variant C) MAE={ref:.4f}")

    (OUT / "p0_fold_local_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] p0_fold_local_results.json (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
