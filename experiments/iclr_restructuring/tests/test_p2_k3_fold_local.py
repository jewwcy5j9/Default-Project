"""B4 regression: fold-local scaling (held-out extreme must not change the
training scaler).

Plan: NEXT_PHASE_EXECUTION_PLAN.md B4. Historical bug: scaler was fit on the
full feature matrix including the held-out row, so an extreme held-out value
leaked into training standardization. The evaluator must fit scalers on
training IDs only and record fit IDs + transform params.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent))

import numpy as np
import p2_k3_eval_v2 as ev


def _synthetic():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(6, 3))
    y = np.abs(rng.normal(size=(6, 3)))
    y = y / y.sum(axis=1, keepdims=True)
    return X, y


def test_scaler_fit_ids_recorded():
    X, y = _synthetic()
    res = ev.predict_k3("C1_llr1", "CLR-Ridge", X[1:], y[1:],
                        X[[0]], ev.wt_matrix(np.array([0.88, 0.06, 0.06]), 5),
                        ev.wt_matrix(np.array([0.88, 0.06, 0.06]), 1),
                        seed_base=0, n_seeds=2, fit_ids=["b", "c", "d", "e", "f"])
    tp = res["transform_params"]
    assert tp["fit_ids"] == ["b", "c", "d", "e", "f"]
    # scaler mean equals mean of the training rows only (rows 1..5), not row 0
    Xtr = X[1:]
    assert np.allclose(tp["scaler_mean"], Xtr.mean(axis=0), atol=1e-12)
    assert np.allclose(tp["scaler_scale"], Xtr.std(axis=0), atol=1e-12)


def test_held_out_extreme_does_not_shift_scaler():
    X, y = _synthetic()
    # held-out row 0 gets an extreme value; training rows are 1..5
    X_ext = X.copy()
    X_ext[0] = 1e6
    res = ev.predict_k3("C1_llr1", "CLR-Ridge", X_ext[1:], y[1:],
                        X_ext[[0]], ev.wt_matrix(np.array([0.88, 0.06, 0.06]), 5),
                        ev.wt_matrix(np.array([0.88, 0.06, 0.06]), 1),
                        seed_base=0, n_seeds=2, fit_ids=["b", "c", "d", "e", "f"])
    tp = res["transform_params"]
    # mean/scale come from training rows 1..5; the extreme row 0 is excluded
    Xtr = X[1:]
    assert np.allclose(tp["scaler_mean"], Xtr.mean(axis=0), atol=1e-6)
    assert np.allclose(tp["scaler_scale"], Xtr.std(axis=0), atol=1e-6)
    # and the prediction is finite (no overflow from the extreme row)
    assert np.all(np.isfinite(res["mean"]))


def test_deterministic_models_report_flag():
    X, y = _synthetic()
    res = ev.predict_k3("C1_llr1", "CLR-Ridge", X[1:], y[1:],
                        X[[0]], ev.wt_matrix(np.array([0.88, 0.06, 0.06]), 5),
                        ev.wt_matrix(np.array([0.88, 0.06, 0.06]), 1),
                        seed_base=0, n_seeds=3, fit_ids=["b", "c", "d", "e", "f"])
    assert res["deterministic"] is True
    assert len(res["per_seed"]) == 3
    for s in res["per_seed"]:
        assert np.allclose(s, res["mean"])


def test_torch_models_per_seed_distinct():
    X, y = _synthetic()
    res = ev.predict_k3("C1_llr1", "SimpleCDST", X[1:], y[1:],
                        X[[0]], ev.wt_matrix(np.array([0.88, 0.06, 0.06]), 5),
                        ev.wt_matrix(np.array([0.88, 0.06, 0.06]), 1),
                        seed_base=0, n_seeds=2, fit_ids=["b", "c", "d", "e", "f"])
    assert res["deterministic"] is False
    assert len(res["per_seed"]) == 2


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
