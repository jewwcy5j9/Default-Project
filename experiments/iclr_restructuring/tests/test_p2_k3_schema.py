"""B4 regression: full output must validate; any missing key field must fail.

Plan: NEXT_PHASE_EXECUTION_PLAN.md B4. Uses the actual evaluator's
build_result_json on a synthetic system, validates the block against
result_schema.json with validate_result_schema.validate, and checks that
removing each required top-level key makes it fail.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent))

import numpy as np
import p2_k3_eval_v2 as ev
from validate_result_schema import load_schema, validate


def _synthetic_system(n=8):
    # use real SRC_CORE mutant names (marker_matrix reads SRC_CORE by name);
    # n=8 covers both LSO groups with non-empty training folds
    rng = np.random.default_rng(13)
    names = list(ev.SRC_CORE.keys())[:n]
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
    p_wt = np.array([0.72, 0.07, 0.21])
    return names, f, targets, p_wt


def _make_block():
    names, f, targets, p_wt = _synthetic_system(8)
    first = names[0]
    combo_defs = [("C1_llr1", ["llr"])]
    positions = {m: [int(round(f[m]["pos"] * 100))] for m in names}
    ev.NEPOCHS = 2
    try:
        fixed = ev.run_fixed_loo("src", names, f, targets, p_wt,
                                 combo_defs, n_seeds=1)
        nested = ev.run_nested("src", names, f, targets, p_wt, combo_defs,
                               n_seeds=1, positions=positions)
        ctl = ev.run_nested("src", names, f, targets, p_wt, combo_defs,
                            control=True, n_seeds=1, positions=positions)
        lso = ev.selection_aware_lso("src", names, f, targets, p_wt,
                                     combo_defs, n_seeds=1)
        g = ev.gates(nested, ctl, lso, None, "src")
    finally:
        ev.NEPOCHS = 800
    return ev.build_result_json("src", names, f, targets, p_wt, fixed,
                                nested, ctl, lso, None, g, "a" * 32,
                                "b" * 32, {"canonical_data": "c" * 32,
                                           "feature_cache": "d" * 32,
                                           "model_checkpoint": "e" * 32},
                                1, "9.9.9-test", positions=positions)


def test_block_validates():
    block = _make_block()
    schema = load_schema()
    errs = validate(block, schema)
    assert not errs, errs


def test_missing_top_level_fails():
    block = _make_block()
    schema = load_schema()
    del block["metrics"]
    errs = validate(block, schema)
    assert errs, "expected failure when metrics is removed"


def test_missing_per_fold_field_fails():
    block = _make_block()
    schema = load_schema()
    key = list(block["results"]["per_fold"])[0]
    del block["results"]["per_fold"][key]["mean_predictions"]
    errs = validate(block, schema)
    assert errs, "expected failure when per_fold mean_predictions removed"


def test_simplex_violation_fails():
    block = _make_block()
    schema = load_schema()
    key = list(block["results"]["per_fold"])[0]
    block["results"]["per_fold"][key]["mean_predictions"] = [0.9, 0.1, 0.1]
    errs = validate(block, schema)
    assert errs, "expected failure when simplex sum != 1"


def test_gate_thresholds_are_numeric():
    block = _make_block()
    for k, v in block["hard_gates"]["gates"].items():
        th = v["threshold"]
        assert th is None or isinstance(th, (int, float)), (k, th)


def test_verdict_enum():
    block = _make_block()
    assert block["hard_gates"]["verdict"] in ("GO", "NO_GO", "PENDING")


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
