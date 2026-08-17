"""B4 regression: alternative label run must re-execute the selector.

Plan: NEXT_PHASE_EXECUTION_PLAN.md B4 + B1 fix 8. Historical bug: the
alt-label run reused the main selector's choice. The evaluator must re-run
the full selection-aware route with alt targets (L410A set to [0.96,0.03,0.01]).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent))

import numpy as np
import p2_k3_eval_v2 as ev


def _synthetic_system(n=4):
    rng = np.random.default_rng(11)
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


def test_alt_target_replaced_for_l410a():
    names, f, targets, p_wt = _synthetic_system(4)
    alt_targets = {k: ([0.96, 0.03, 0.01] if k == "m0" else v)
                   for k, v in targets.items()}
    assert np.allclose(alt_targets["m0"], [0.96, 0.03, 0.01])
    assert np.allclose(alt_targets["m1"], targets["m1"])


def test_alt_route_reruns_selector(monkeypatch):
    # The alt-label run must produce its own selection, not reuse main's.
    # Run the main route and the alt route; both must return per-fold
    # selected pairs and independent inner scores.
    names, f, targets, p_wt = _synthetic_system(4)
    combo_defs = [("C1_llr1", ["llr"]), ("C2_llr_pos2d", ["llr", "pos"])]
    alt_targets = {k: ([0.96, 0.03, 0.01] if k == "m0" else v)
                   for k, v in targets.items()}
    monkeypatch.setattr(ev, "NEPOCHS", 2)
    main_out = ev.run_nested("test", names, f, targets, p_wt, combo_defs,
                             n_seeds=1, positions=None)
    alt_out = ev.run_nested("test", names, f, alt_targets, p_wt,
                            combo_defs, n_seeds=1, positions=None)
    # Both routes ran the selector: per-fold inner scores exist for all folds
    assert len(main_out["inner_scores"]) == 4
    assert len(alt_out["inner_scores"]) == 4
    # The alt route's inner scores were computed on alt targets, so at least
    # one fold differs (targets differ) -> selection may differ
    assert main_out["folds"]["m0"]["target"] != alt_out["folds"]["m0"]["target"]
    # Strengthened (2026-08-17): the historical bug reused the main run's
    # inner scores. Any fold whose inner LOO training set CONTAINS m0 must
    # see different inner scores under the alt target, so a reused-scores
    # regression now fails here instead of passing vacuously.
    differing_folds = [
        fold for fold in names
        if fold != "m0"  # folds other than m0 train on m0's target
        and main_out["inner_scores"][fold] != alt_out["inner_scores"][fold]
    ]
    assert differing_folds, (
        "alt-label run produced identical inner scores to the main run for "
        "every fold that trains on the substituted target — selector reuse "
        "regression")


def test_gates_include_alt_verdict_for_src():
    # gates() for src with alt returns alt_l410a_le_0_2560 and
    # alt_verdict_not_reversed as independent booleans (ADR-003 G8).
    nested = {"nested_mae": 0.20, "errors": {"a": 0.20},
              "catastrophic_folds": []}
    ctl = {"nested_mae": 0.30, "errors": {"a": 0.30},
           "catastrophic_folds": []}
    lso = {}
    alt = {"nested_mae": 0.10, "direction": "1/1"}
    g = ev.gates(nested, ctl, lso, alt, "src")
    assert "alt_l410a_le_0_2560" in g
    assert "alt_verdict_not_reversed" in g
    assert g["alt_l410a_le_0_2560"]["passed"] is True
    assert g["alt_verdict_not_reversed"]["passed"] is True


def test_gates_alt_verdict_reversal_detected():
    nested = {"nested_mae": 0.20, "errors": {"a": 0.20},
              "catastrophic_folds": []}
    ctl = {"nested_mae": 0.30, "errors": {"a": 0.30},
           "catastrophic_folds": []}
    # main passes (0.20 <= 0.2560) but alt fails (0.30 > 0.2560)
    alt = {"nested_mae": 0.30, "direction": "0/1"}
    g = ev.gates(nested, ctl, {}, alt, "src")
    assert g["alt_l410a_le_0_2560"]["passed"] is False
    assert g["alt_verdict_not_reversed"]["passed"] is False


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
