"""P0-1: Core-set constant / training-mean baselines (audit-corrected).

Fixes the contamination in constant_baselines.py:
  - Abl1: only the 6 core mutants (WT, H396P, M290L_H396P excluded)
  - Src  : only the 8 core mutants (WT excluded)
  - Pooled K=2 collapse uses the same core sets.

Output: results/constant_baselines_core.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from k3_data import ABL1_K3, ABL1_K3_WT_POP, SRC_K3, SRC_K3_WT_POP

OUT = Path(__file__).resolve().parent / "results"

ABL1_CORE = {m: ABL1_K3[m] for m in ABL1_K3 if m not in ("WT", "H396P", "M290L_H396P")}
SRC_CORE = {m: SRC_K3[m] for m in SRC_K3 if m != "SrcKD-WT"}


def mae(pred, true):
    return float(np.abs(np.asarray(pred) - np.asarray(true)).mean())


def baselines(mutations, wt_pop):
    names = list(mutations.keys())
    pops = {m: np.asarray(mutations[m]["pop"], dtype=float) for m in names}
    wt = np.asarray(wt_pop, dtype=float)

    const = {m: mae(wt, pops[m]) for m in names}
    tmean = {}
    for held in names:
        others = [pops[m] for m in names if m != held]
        pred = np.mean(others, axis=0)
        tmean[held] = mae(pred, pops[held])
    return {
        "n": len(names),
        "mutants": names,
        "constant_WT": {"per_mutant": const,
                        "mean_mae": float(np.mean(list(const.values())))},
        "training_mean_LOO": {"per_mutant": tmean,
                              "mean_mae": float(np.mean(list(tmean.values())))},
    }


def two_state(pop3):
    p = np.asarray(pop3, dtype=float)
    return np.array([p[0], 1.0 - p[0]])


res = {
    "abl1_k3_n6": baselines(ABL1_CORE, ABL1_K3_WT_POP),
    "src_k3_n8": baselines(SRC_CORE, SRC_K3_WT_POP),
    "abl1_pooled_n6": baselines({m: {"pop": two_state(v["pop"])}
                                 for m, v in ABL1_CORE.items()},
                                two_state(ABL1_K3_WT_POP)),
    "src_pooled_n8": baselines({m: {"pop": two_state(v["pop"])}
                                for m, v in SRC_CORE.items()},
                               two_state(SRC_K3_WT_POP)),
}

# skill of reference results against the corrected core baselines
ref = {
    "abl1_k3_n6": {"variant_C": 0.0804, "LLR_proxy": 0.1629,
                   "Extended": 0.3003, "pos": 0.2757, "ESM2": 0.3088,
                   "onehot": 0.2425},
    "src_k3_n8": {"Extended": 0.3045, "pos": 0.3213, "ESM2": 0.3468,
                  "onehot": 0.2606},
    "abl1_pooled_n6": {"Extended": 0.4134, "variant_C": 0.1046},
    "src_pooled_n8": {"Extended": 0.4443, "pos": 0.2508},
}

print("=" * 88)
print(f"{'set':<18}{'metric':<16}{'MAE':>8}   skill vs const")
print("-" * 88)
for set_key, bl in res.items():
    cw = bl["constant_WT"]["mean_mae"]
    tm = bl["training_mean_LOO"]["mean_mae"]
    print(f"{set_key:<18}{'constant-WT':<16}{cw:>8.4f}")
    print(f"{set_key:<18}{'training-mean':<16}{tm:>8.4f}")
    for name, val in ref.get(set_key, {}).items():
        skill_const = 1.0 - val / cw
        skill_tm = 1.0 - val / tm
        print(f"{set_key:<18}{'  ' + name:<16}{val:>8.4f}   "
              f"skill={skill_const:+.3f} / vs-tmean {skill_tm:+.3f}")
    print("-" * 88)

(OUT / "constant_baselines_core.json").write_text(
    json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print("\n[OK] constant_baselines_core.json written")
