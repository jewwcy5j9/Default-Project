"""P0-2: True grouped (leave-site-out) CV for Abl1 core set.

Fixes the bug in k3_followup.grouped_cv (which trained on every mutant
except the current one, i.e. plain LOO). Here the training set is exactly
the mutants OUTSIDE the held-out site.

Runs variant-C (ddG+pos), LLR proxy (experiment-free) and pos-markers.
Also records which one-hot position columns have zero variance in the
training fold (feature-support diagnostics).
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

from k3_data import (ABL1_K3, ABL1_K3_WT_POP, ABL1_SEQ_LEN,
                     enc_abl1_ddg_main, enc_abl1_pos_markers)
from k3_benchmark import train_one_seed
from k3_llr_proxy import enc_abl1_llr

OUT = Path(__file__).resolve().parent / "results"
ABL1_CORE = {m: ABL1_K3[m] for m in ABL1_K3 if m not in ("WT", "H396P", "M290L_H396P")}
GROUPS = {"F382_family": ["F382L", "F382Y", "F382V"],
          "290_301": ["M290L", "L301I", "M290L_L301I"]}

ENCODERS = {
    "variant_C": (enc_abl1_ddg_main, 5),
    "LLR_proxy": (enc_abl1_llr, 5),
    "pos_markers": (enc_abl1_pos_markers, 4),
}


def grouped_cv(encoder_fn, d, n_seeds=5, n_epochs=800):
    names = list(ABL1_CORE.keys())
    wt = np.array(ABL1_K3_WT_POP, dtype=float)
    out = {}
    for gname, group in GROUPS.items():
        train = [m for m in names if m not in group]
        c_tr = np.array([encoder_fn(m, ABL1_CORE[m]) for m in train])
        zero_var_cols = [int(i) for i in range(c_tr.shape[1])
                         if np.allclose(np.var(c_tr[:, i], axis=0), 0)]
        w_tr = np.tile(wt, (len(train), 1))
        y_tr = np.array([ABL1_CORE[m]["pop"] for m in train])
        g_preds, g_errs = {}, {}
        for m in group:
            seed_preds = []
            for seed in range(n_seeds):
                model = train_one_seed(w_tr, c_tr, y_tr, d=d,
                                       seed=seed * 100 + names.index(m),
                                       n_epochs=n_epochs, K=3)
                with torch.no_grad():
                    p = model(torch.FloatTensor([wt]),
                              torch.FloatTensor([encoder_fn(m, ABL1_CORE[m])]))
                seed_preds.append(p.numpy()[0])
            g_preds[m] = np.mean(seed_preds, axis=0)
            g_errs[m] = float(np.abs(g_preds[m] - np.array(ABL1_CORE[m]["pop"])).mean())
        out[gname] = {"members": group, "n_train": len(train),
                      "zero_var_feature_cols": zero_var_cols,
                      "errors": g_errs,
                      "group_mae": float(np.mean(list(g_errs.values())))}
        print(f"  [{gname}] n_train={len(train)} zero_var_cols={zero_var_cols} "
              f"group_mae={out[gname]['group_mae']:.4f}")
        for m in group:
            print(f"      {m:<14} true={np.round(np.array(ABL1_CORE[m]['pop']),2)} "
                  f"pred={np.round(g_preds[m],2)} err={g_errs[m]:.4f}")
    return out


def main():
    t0 = __import__("time").time()
    print("=" * 90)
    print("P0-2: true leave-site-out CV (Abl1 core, K=3)")
    print("=" * 90)
    results = {}
    for enc_name, (fn, d) in ENCODERS.items():
        print(f"\n--- {enc_name} (d={d}) ---")
        results[enc_name] = grouped_cv(fn, d)
    results["groups"] = GROUPS
    out_json = OUT / "p0_grouped_cv.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n[OK] {out_json} ({(__import__('time').time() - t0):.0f}s)")


if __name__ == "__main__":
    main()
