"""B4 regression: determinism (same seed + same input -> same result).

Plan: NEXT_PHASE_EXECUTION_PLAN.md B4. Torch models must produce identical
per-seed predictions given identical seeds and inputs; CLR models are
deterministic by construction.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent))

import numpy as np
import p2_k3_eval_v2 as ev


def _synthetic():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(6, 3))
    y = np.abs(rng.normal(size=(6, 3)))
    y = y / y.sum(axis=1, keepdims=True)
    return X, y


def test_torch_per_seed_repeatable():
    X, y = _synthetic()
    kw = dict(seed_base=0, n_seeds=2, fit_ids=["a", "b", "c", "d", "e", "f"])
    r1 = ev.predict_k3("C1_llr1", "LowRankCDST", X, y, X[[0]],
                       ev.wt_matrix(np.array([0.88, 0.06, 0.06]), 6),
                       ev.wt_matrix(np.array([0.88, 0.06, 0.06]), 1), **kw)
    r2 = ev.predict_k3("C1_llr1", "LowRankCDST", X, y, X[[0]],
                       ev.wt_matrix(np.array([0.88, 0.06, 0.06]), 6),
                       ev.wt_matrix(np.array([0.88, 0.06, 0.06]), 1), **kw)
    for s in range(2):
        assert np.allclose(r1["per_seed"][s], r2["per_seed"][s], atol=1e-12)
    assert np.allclose(r1["mean"], r2["mean"], atol=1e-12)


def test_torch_seed_change_changes_result(monkeypatch):
    # LowRankCDST has a randomly initialized transition net; different seed
    # bases must (almost surely) give different predictions. SimpleCDST is a
    # zero-init convex model converging to the same solution regardless of
    # seed, so it is not a valid probe of seed sensitivity.
    X, y = _synthetic()
    kw = dict(n_seeds=1, fit_ids=["a", "b", "c", "d", "e", "f"])
    # monkeypatch auto-restores NEPOCHS (the old finally-restore left it at
    # 2 instead of the module default 800, leaking stale config to any test
    # that ran afterwards under -k/reordered selection).
    monkeypatch.setattr(ev, "NEPOCHS", 20)
    r1 = ev.predict_k3("C1_llr1", "LowRankCDST", X, y, X[[0]],
                       ev.wt_matrix(np.array([0.88, 0.06, 0.06]), 6),
                       ev.wt_matrix(np.array([0.88, 0.06, 0.06]), 1),
                       seed_base=0, **kw)
    r2 = ev.predict_k3("C1_llr1", "LowRankCDST", X, y, X[[0]],
                       ev.wt_matrix(np.array([0.88, 0.06, 0.06]), 6),
                       ev.wt_matrix(np.array([0.88, 0.06, 0.06]), 1),
                       seed_base=1, **kw)
    # different seed base should (almost surely) give different predictions
    assert not np.allclose(r1["mean"], r2["mean"], atol=1e-6)


def test_clr_deterministic_flag():
    X, y = _synthetic()
    r = ev.predict_k3("C1_llr1", "CLR-GP", X, y, X[[0]],
                      ev.wt_matrix(np.array([0.88, 0.06, 0.06]), 6),
                      ev.wt_matrix(np.array([0.88, 0.06, 0.06]), 1),
                      seed_base=0, n_seeds=3, fit_ids=["a", "b", "c", "d", "e", "f"])
    assert r["deterministic"] is True
    for s in r["per_seed"]:
        assert np.allclose(s, r["mean"])


def test_run_nested_repeatable():
    rng = np.random.default_rng(5)
    names = [f"m{i}" for i in range(4)]
    f = {m: {"llr": float(rng.normal()), "pos": float(rng.uniform(0, 1)),
             "tok": float(rng.normal())} for m in names}
    for m in names:
        for d in ev.RAND_DIMS:
            f[m][f"rand{d}"] = rng.normal(size=(d,)).tolist()
    targets = {}
    for m in names:
        v = np.abs(rng.normal(size=(3,)))
        targets[m] = v / v.sum()
    p_wt = np.array([0.88, 0.06, 0.06])
    ev.NEPOCHS = 2
    try:
        a = ev.run_nested("test", names, f, targets, p_wt,
                          [("C1_llr1", ["llr"])], n_seeds=1)
        b = ev.run_nested("test", names, f, targets, p_wt,
                          [("C1_llr1", ["llr"])], n_seeds=1)
    finally:
        ev.NEPOCHS = 800
    assert a["nested_mae"] == b["nested_mae"]
    for m in names:
        assert np.allclose(a["folds"][m]["pred"], b["folds"][m]["pred"])
        assert a["folds"][m]["combo"] == b["folds"][m]["combo"]
        assert a["folds"][m]["model"] == b["folds"][m]["model"]


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
