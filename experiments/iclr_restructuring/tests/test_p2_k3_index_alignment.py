"""B4 regression: index alignment (immutable row indices; inner features and
labels must never mismatch).

Plan: NEXT_PHASE_EXECUTION_PLAN.md B4. Historical bug (p2_k3_eval.py): the
inner selector indexed labels with row positions that did not match the
feature rows after filtering, so training labels could be misaligned with
features. The evaluator must use immutable indices throughout and record
fit IDs per fold.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent))

import numpy as np
import p2_k3_eval_v2 as ev


def _synthetic_system(n=5):
    rng = np.random.default_rng(7)
    names = [f"m{i}" for i in range(n)]
    f = {}
    for m in names:
        f[m] = {"llr": float(rng.normal()), "pos": float(rng.uniform(0, 1)),
                "tok": float(rng.normal())}
        for d in ev.RAND_DIMS:
            f[m][f"rand{d}"] = rng.normal(size=(d,)).tolist()
    targets = {}
    for m in names:
        v = np.abs(rng.normal(size=(3,)))
        targets[m] = v / v.sum()
    p_wt = np.array([0.88, 0.06, 0.06])
    return names, f, targets, p_wt


def test_combo_matrix_rows_match_names():
    names, f, targets, p_wt = _synthetic_system(5)
    X = ev.combo_matrix(f, names, ["llr"])
    assert X.shape == (5, 1)
    # row order equals names order
    for i, m in enumerate(names):
        assert abs(X[i, 0] - f[m]["llr"]) < 1e-12


def test_marker_matrix_rows_match_names():
    # marker_matrix reads ABL1_CORE/SRC_CORE by system; use real src names
    names_src = list(ev.SRC_CORE.keys())[:4]
    f_src = {m: {"llr": float(i), "pos": 0.5} for i, m in enumerate(names_src)}
    X = ev.marker_matrix(f_src, names_src, "src")
    assert X.shape == (4, 2 + len(ev.MARKER_MARKS["src"]))


def test_nested_folds_targets_match_holdout():
    names, f, targets, p_wt = _synthetic_system(4)
    combo_defs = [("C1_llr1", ["llr"])]
    ev.NEPOCHS = 2
    try:
        out = ev.run_nested("test", names, f, targets, p_wt, combo_defs,
                            n_seeds=1, positions=None)
    finally:
        ev.NEPOCHS = 800
    assert len(out["folds"]) == 4
    for m in names:
        fold = out["folds"][m]
        assert np.allclose(fold["target"], targets[m], atol=1e-9)
        # per-fold selected pair must be in the candidate list
        assert fold["combo"] in ["C1_llr1", "M1_marker"]
        assert fold["model"] in ev.MODEL_NAMES
    # every fold's inner scores come from the same training fold -> count = n-1
    for m in names:
        for k, v in out["inner_scores"][m].items():
            assert len(v) == 3, (m, k, len(v))


def test_tr_of_immutable_index_count():
    names, f, targets, p_wt = _synthetic_system(4)
    combo_defs = [("C1_llr1", ["llr"])]
    ev.NEPOCHS = 2
    try:
        out = ev.run_nested("test", names, f, targets, p_wt, combo_defs,
                            n_seeds=1, positions=None)
    finally:
        ev.NEPOCHS = 800
    for m in names:
        assert len(out["tr_of"][m]) == 3


def test_fit_ids_recorded_per_fold():
    names, f, targets, p_wt = _synthetic_system(4)
    combo_defs = [("C1_llr1", ["llr"])]
    ev.NEPOCHS = 2
    try:
        out = ev.run_nested("test", names, f, targets, p_wt, combo_defs,
                            n_seeds=1, positions=None)
    finally:
        ev.NEPOCHS = 800
    for m in names:
        tp = out["folds"][m]["transform_params"]
        assert tp["fit_ids"] is not None
        assert set(tp["fit_ids"]) == set(names) - {m}
        assert len(tp["fit_ids"]) == 3
        # Strengthened (2026-08-17): the recorded scaler statistics must be
        # the statistics OF THOSE TRAINING ROWS (mirrors the p4 synthetic
        # fold-local test). A mis-sliced matrix with correct fit_ids would
        # previously pass.
        train_rows = np.array([[f[x]["llr"]] for x in tp["fit_ids"]])
        assert np.allclose(tp["scaler_mean"], train_rows.mean(axis=0))
        assert np.allclose(tp["scaler_scale"], train_rows.std(axis=0))


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
