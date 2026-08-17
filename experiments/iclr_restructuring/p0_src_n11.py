"""P0-4: Src n=11 perturbation LOO, WT excluded from evaluation.

Fixes k3_src_perturbation.py, which ran LOO over SRC_K3_EXT (12 entries
including SrcKD-WT). Here LOO covers only the 11 perturbations; the WT
entry is scored separately as an in-sample reference and excluded from
the MAE/direction aggregates.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from k3_data import (SRC_K3_EXT, SRC_K3_WT_POP,
                     enc_src_extended_type, enc_src_pos_type, enc_src_no_dvol_type)
from k3_benchmark import run_loo, metrics, paired_tests

OUT = Path(__file__).resolve().parent / "results"
WT_NAME = "SrcKD-WT"
PERT_NAMES = [m for m in SRC_K3_EXT if m != WT_NAME]  # 11 perturbations


def main():
    t0 = time.time()
    print("=" * 90)
    print("P0-4: Src n=11 LOO, WT excluded from evaluation")
    print("=" * 90)
    results = {"protocol": "LOO over 11 perturbations (WT scored separately, "
                           "excluded from aggregates)"}

    encoders = {"Extended_type": (enc_src_extended_type, 11),
                "pos_markers_type": (enc_src_pos_type, 5),
                "no_dVol_type": (enc_src_no_dvol_type, 10)}
    eval_set = {m: SRC_K3_EXT[m] for m in PERT_NAMES}
    for enc_name, (fn, d) in encoders.items():
        res = run_loo(eval_set, SRC_K3_WT_POP, fn, d)
        met = metrics(res["per_mutant"], res["targets"], SRC_K3_WT_POP)
        results[enc_name] = {"mae": met["mae"], "direction": met["direction"],
                             "errors": met["mae_per_mutant"],
                             "preds": {m: res["per_mutant"][m].tolist()
                                       for m in res["per_mutant"]}}
        print(f"  {enc_name:<20} MAE={met['mae']:.4f} dir={met['direction']}")
        for m in PERT_NAMES:
            print(f"      {m:<16} true={np.round(np.array(SRC_K3_EXT[m]['pop']),2)} "
                  f"pred={np.round(res['per_mutant'][m],2)} "
                  f"err={met['mae_per_mutant'][m]:.4f}")

    paired = paired_tests(results["Extended_type"]["errors"],
                          results["pos_markers_type"]["errors"])
    results["paired_ext_vs_pos"] = {"wilcoxon_p": paired["wilcoxon_p"],
                                    "mean_diff": paired["mean_diff_a_minus_b"],
                                    "n_pairs": paired["n_pairs"]}
    print(f"  paired Extended vs pos (n=11): Wilcoxon p={paired['wilcoxon_p']:.4f}")

    (OUT / "p0_src_n11_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] p0_src_n11_results.json  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
