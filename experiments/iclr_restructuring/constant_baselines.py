"""Constant / training-mean baselines for all MAE tables (K=3 and pooled K=2).

SUPERSEDED by constant_baselines_core.py (audit-corrected core sets).
This module now delegates: the legacy WT/silver-contaminated counts
(Abl1 including WT/H396P/M290L_H396P; Src including SrcKD-WT) are retired.
Legacy output results/constant_baselines.json (frozen, archive-only)
predates this delegation and uses the retired contaminated sets.

The successor writes results/constant_baselines_core.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def mae(pred, true):
    import numpy as np
    return float(np.abs(np.asarray(pred) - np.asarray(true)).mean())


def main():
    print("[NOTE] constant_baselines.py delegates to constant_baselines_core.py "
          "(WT/silver-contaminated legacy counts retired)")
    # constant_baselines_core.py has no main(): it is a module-level script,
    # so importing it runs its full entry point (mirroring its __main__ form).
    import constant_baselines_core
    return constant_baselines_core  # module executes on import


if __name__ == "__main__":
    main()
