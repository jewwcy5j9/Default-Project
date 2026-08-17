"""B4 regression: metric definitions (MAE, u1/u2, ILR, JSD, direction).

Plan: NEXT_PHASE_EXECUTION_PLAN.md B4. Pins hand-computed values for the
K=3 metrics used as gates: per-state MAE, raw u1/u2 contrast, ILR z1/z2,
JSD divergence, and ACTIVE-state direction agreement (ADR-002).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent))

import numpy as np
import p2_k3_eval_v2 as ev


def test_per_state_mae_hand_computed():
    preds = {"a": np.array([0.80, 0.10, 0.10]), "b": np.array([0.60, 0.20, 0.20])}
    targets = {"a": np.array([0.90, 0.05, 0.05]), "b": np.array([0.50, 0.30, 0.20])}
    e = ev.per_state_mae(preds, targets)
    assert abs(e["a"] - (0.10 + 0.05 + 0.05) / 3) < 1e-12
    assert abs(e["b"] - (0.10 + 0.10 + 0.00) / 3) < 1e-12
    assert abs(ev.mean_mae(e) - ((0.20 / 3 + 0.20 / 3) / 2)) < 1e-12


def test_u1_u2_contrast_hand_computed():
    preds = {"a": np.array([0.75, 0.20, 0.05])}
    targets = {"a": np.array([0.90, 0.07, 0.03])}
    c = ev.u1_u2_contrast(preds, targets)
    assert abs(c["u1"] - abs(2 * 0.75 - 1 - (2 * 0.90 - 1))) < 1e-12
    assert abs(c["u2"] - abs((0.20 - 0.05) - (0.07 - 0.03))) < 1e-12
    # ILR of the two vectors
    zp = ev.ilr(np.atleast_2d(preds["a"]))[0]
    zt = ev.ilr(np.atleast_2d(targets["a"]))[0]
    assert abs(c["ilr_z1"] - abs(zp[0] - zt[0])) < 1e-12
    assert abs(c["ilr_z2"] - abs(zp[1] - zt[1])) < 1e-12


def test_direction_active_sign():
    p_wt = np.array([0.88, 0.06, 0.06])
    preds = {"up": np.array([0.90, 0.05, 0.05]),   # active up, target up
             "down": np.array([0.10, 0.60, 0.30]),  # active down, target down
             "tie": np.array([0.85, 0.10, 0.05])}   # target within TIE_DELTA
    targets = {"up": np.array([0.95, 0.03, 0.02]),
               "down": np.array([0.30, 0.40, 0.30]),
               "tie": np.array([0.86, 0.08, 0.06])}
    d = ev.direction_report(preds, targets, p_wt)
    assert d == "2/2", d  # tie excluded


def test_direction_wrong_sign_fails():
    p_wt = np.array([0.88, 0.06, 0.06])
    preds = {"x": np.array([0.10, 0.80, 0.10])}
    targets = {"x": np.array([0.95, 0.03, 0.02])}
    d = ev.direction_report(preds, targets, p_wt)
    assert d == "0/1", d


def test_direction_tie_delta_excluded():
    p_wt = np.array([0.88, 0.06, 0.06])
    preds = {"x": np.array([0.90, 0.05, 0.05])}
    targets = {"x": np.array([0.90, 0.05, 0.05])}  # |0.90-0.88|=0.02 < 0.05
    d = ev.direction_report(preds, targets, p_wt)
    assert d == "0/0", d


def test_training_mean_mae():
    targets = [np.array([0.90, 0.05, 0.05]), np.array([0.50, 0.30, 0.20]),
               np.array([0.10, 0.60, 0.30])]
    held = np.array([0.70, 0.20, 0.10])
    mu = np.mean(targets, axis=0)
    expected = np.abs(mu - held).mean()
    assert abs(ev.training_mean_mae(targets, [0, 1, 2], held) - expected) < 1e-12


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
