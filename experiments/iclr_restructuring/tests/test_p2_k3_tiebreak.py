"""B4 regression: tie-break order (MAE -> dimension -> model simplicity -> combo ID).

Plan: NEXT_PHASE_EXECUTION_PLAN.md B4. Historical bug: tie-break used
(mean_mae, combo index, model index) which could pick a more complex candidate
with the same MAE. This test pins the frozen order.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent))

import numpy as np
import p2_k3_eval_v2 as ev


def _mk_scores():
    return {
        "C1_llr1::CLR-Ridge": [0.30],
        "C1_llr1::CLR-GP": [0.30],
        "C1_llr1::SimpleCDST": [0.30],
        "C1_llr1::LowRankCDST": [0.30],
        "C2_llr_pos::CLR-Ridge": [0.30],
        "C4_rand4::CLR-Ridge": [0.30],
        "C6_rand8::LowRankCDST": [0.30],
    }


def test_tie_break_same_mae_prefers_lower_dim():
    s = _mk_scores()
    keys = [k for k in s if "::" in k]
    best = min(keys, key=lambda k: ev.tie_break_key(s, k))
    # C1 (dim 1) must win over C2 (dim 2) and C4/C6 at equal MAE
    assert best == "C1_llr1::CLR-Ridge", best


def test_tie_break_same_dim_prefers_simpler_model():
    s = {
        "C1_llr1::CLR-Ridge": [0.30],
        "C1_llr1::CLR-GP": [0.30],
        "C1_llr1::SimpleCDST": [0.30],
        "C1_llr1::LowRankCDST": [0.30],
    }
    best = min(s, key=lambda k: ev.tie_break_key(s, k))
    assert best == "C1_llr1::CLR-Ridge", best


def test_tie_break_mae_dominates():
    s = {
        "C1_llr1::CLR-Ridge": [0.20],
        "C2_llr_pos2d::CLR-GP": [0.19],
        "C6_rand8::LowRankCDST": [0.21],
    }
    best = min(s, key=lambda k: ev.tie_break_key(s, k))
    assert best == "C2_llr_pos2d::CLR-GP", best


def test_tie_break_combo_id_last():
    s = {
        "C1_llr1::CLR-Ridge": [0.30],
        "C1_llr1::CLR-GP": [0.30],
    }
    best = min(s, key=lambda k: ev.tie_break_key(s, k))
    assert best == "C1_llr1::CLR-Ridge", best


def test_select_best_m1_marker_only_candidate():
    s = {"M1_marker::CLR-Ridge": [0.25]}
    assert ev.select_best(s) == "M1_marker::CLR-Ridge"


def test_select_best_empty_returns_none():
    assert ev.select_best({}) is None


def _run_all():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run_all()
