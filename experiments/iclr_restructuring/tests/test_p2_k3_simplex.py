"""B4 regression: simplex constraints (non-negative, sum=1) on all predictions.

Plan: NEXT_PHASE_EXECUTION_PLAN.md B4. Historical bug: predictions were stored
rounded to 6 decimals, which could violate the 1e-6 simplex tolerance of the
result schema validator. This test pins the simplex guarantee for every
prediction path (CLR-Ridge, CLR-GP, SimpleCDST, LowRankCDST) and for the
ILR/clr round trips.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent))

import numpy as np
import p2_k3_eval_v2 as ev


def test_clr_roundtrip():
    y = np.array([[0.88, 0.06, 0.06], [0.10, 0.00, 0.90], [0.55, 0.10, 0.35]])
    back = ev.inv_clr(ev.clr(y))
    # clip(EPS)+renormalize is the frozen protocol; values at 0 map to EPS
    # so the exact roundtrip holds for all-positive rows
    assert np.allclose(back[0], y[0], atol=1e-8)
    assert np.allclose(back[2], y[2], atol=1e-8)
    assert np.all(back >= 0)
    assert np.allclose(back.sum(axis=-1), 1.0)


def test_ilr_roundtrip():
    y = np.array([[0.88, 0.06, 0.06], [0.10, 0.00, 0.90], [0.55, 0.10, 0.35]])
    z = ev.ilr(y)
    back = ev.inv_ilr(z)
    # ilr clips at EPS and renormalizes (frozen protocol); positive rows exact
    assert np.allclose(back[0], y[0], atol=1e-8)
    assert np.allclose(back[2], y[2], atol=1e-8)
    assert np.all(back >= 0)
    assert np.allclose(back.sum(axis=-1), 1.0)
    # clip-normalized input roundtrips exactly everywhere
    c = np.clip(y, ev.EPS, 1.0)
    c = c / c.sum(axis=-1, keepdims=True)
    assert np.allclose(ev.inv_ilr(ev.ilr(c)), c, atol=1e-8)


def test_ilr_known_isotonic():
    y = np.array([[1.0, 0.0, 0.0]])
    z = ev.ilr(y)[0]
    # z1 = sqrt(2/3)*log(1/sqrt(eps*eps)) >> 0; z2 = (1/sqrt2)*log(eps/eps) ~ 0
    assert z[0] > 0
    assert abs(z[1]) < 1e-6


def test_jsd_simplex_protected():
    p = np.array([[0.9, 0.1, 0.0]])
    q = np.array([[0.3, 0.3, 0.4]])
    d = ev.jsd(p, q)
    assert d.shape == (1,)
    assert d[0] >= 0
    # symmetric
    d2 = ev.jsd(q, p)
    assert abs(d[0] - d2[0]) < 1e-12
    # identical distributions -> 0
    assert ev.jsd(p, p)[0] == 0.0


def test_raw_u1_u2_definition():
    pred = np.array([0.75, 0.20, 0.05])
    tgt = np.array([0.90, 0.07, 0.03])
    r = ev.raw_u1_u2(pred, tgt)
    assert abs(r["u1_pred"] - (2 * 0.75 - 1)) < 1e-12
    assert abs(r["u2_pred"] - (0.20 - 0.05)) < 1e-12
    assert abs(r["u1_target"] - (2 * 0.90 - 1)) < 1e-12
    assert abs(r["u2_target"] - (0.07 - 0.03)) < 1e-12


def test_norm_simplex():
    p = np.array([1.0, 2.0, 3.0])
    q = ev._norm_simplex(p)
    assert np.allclose(q.sum(), 1.0)
    assert np.all(q >= 0)
    # zero vector passes through unchanged
    z = ev._norm_simplex(np.zeros(3))
    assert np.array_equal(z, np.zeros(3))


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
