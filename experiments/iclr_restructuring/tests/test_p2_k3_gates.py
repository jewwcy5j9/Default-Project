"""B4 regression: gates per ADR-003 (no cross-config LSO splicing, alt
reversal judged independently, catastrophic comparator = same-fold marker
control, single-mutant contribution <= 50%).

Plan: NEXT_PHASE_EXECUTION_PLAN.md B4 + ADR-003 G1-G8 (FROZEN).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent))

import numpy as np
import p2_k3_eval_v2 as ev


def test_abl1_floor_strict_lt():
    nested = {"nested_mae": 0.2329, "errors": {"a": 0.2329},
              "catastrophic_folds": []}
    ctl = {"nested_mae": 0.30, "errors": {"a": 0.30},
           "catastrophic_folds": []}
    g = ev.gates(nested, ctl, {}, None, "abl1")
    # equal to the floor is NOT a pass for abl1 (strict <)
    assert g["abl1_nested_lt_0_2329"]["passed"] is False
    nested2 = dict(nested, nested_mae=0.20)
    g2 = ev.gates(nested2, ctl, {}, None, "abl1")
    assert g2["abl1_nested_lt_0_2329"]["passed"] is True


def test_src_floor_le():
    nested = {"nested_mae": 0.2560, "errors": {"a": 0.2560},
              "catastrophic_folds": []}
    ctl = {"nested_mae": 0.30, "errors": {"a": 0.30},
           "catastrophic_folds": []}
    g = ev.gates(nested, ctl, {}, None, "src")
    # equal to the floor IS a pass for src (<=)
    assert g["src_nested_le_0_2560"]["passed"] is True


def test_no_marker_strictly_beats_marker():
    nested = {"nested_mae": 0.30, "errors": {"a": 0.30},
              "catastrophic_folds": []}
    ctl = {"nested_mae": 0.30, "errors": {"a": 0.30},
           "catastrophic_folds": []}
    g = ev.gates(nested, ctl, {}, None, "src")
    assert g["no_marker_strictly_beats_marker"]["passed"] is False
    nested2 = dict(nested, nested_mae=0.2999)
    g2 = ev.gates(nested2, ctl, {}, None, "src")
    assert g2["no_marker_strictly_beats_marker"]["passed"] is True


def test_lso_same_route_gate():
    # one group improves, other worsens <= 5% -> pass
    lso = {"g1": {"group_mae": 0.20, "comparator": 0.30},
           "g2": {"group_mae": 0.31, "comparator": 0.30}}
    assert ev.lso_same_route_gate(lso) is True
    # both worsen -> fail
    lso2 = {"g1": {"group_mae": 0.31, "comparator": 0.30},
            "g2": {"group_mae": 0.32, "comparator": 0.30}}
    assert ev.lso_same_route_gate(lso2) is False
    # one improves, other worsens 6% -> fail
    lso3 = {"g1": {"group_mae": 0.20, "comparator": 0.30},
            "g2": {"group_mae": 0.318, "comparator": 0.30}}
    assert ev.lso_same_route_gate(lso3) is False
    # one improves, other equal -> pass
    lso4 = {"g1": {"group_mae": 0.20, "comparator": 0.30},
            "g2": {"group_mae": 0.30, "comparator": 0.30}}
    assert ev.lso_same_route_gate(lso4) is True


def test_lso_comparator_is_training_mean_on_held_members():
    # Verify the comparator equals the mean per-state training-mean error
    # on the held-out members (ADR-003 G4) with a tiny hand case.
    rng = np.random.default_rng(5)
    names = list(ev.SRC_CORE.keys())
    targets = {}
    for m in names:
        v = np.abs(rng.normal(size=(3,)))
        targets[m] = v / v.sum()
    f = {m: {"llr": float(rng.normal()), "pos": 0.5, "tok": 0.0,
             "rand2": [0.0, 0.0], "rand4": [0.0] * 4, "rand8": [0.0] * 8}
         for m in names}
    p_wt = np.array([0.72, 0.07, 0.21])
    ev.NEPOCHS = 2
    try:
        lso = ev.selection_aware_lso("src", names, f, targets, p_wt,
                                     [("C1_llr1", ["llr"])], n_seeds=1)
    finally:
        ev.NEPOCHS = 800
    assert set(lso) == set(ev.LSO_GROUPS["src"]), set(lso)
    for gname, g in lso.items():
        # hand-computed training-mean comparator over the members
        tr_names = [m for m in names if m not in g["members"]]
        mu = np.mean([targets[m] for m in tr_names], axis=0)
        cmp = np.mean([np.abs(mu - targets[m]).mean() for m in g["members"]])
        assert abs(g["comparator"] - cmp) < 1e-9, (gname, g["comparator"], cmp)


def test_single_mutant_contribution_cap():
    paired = {"a": 0.05, "b": 0.05, "c": 0.4}
    assert ev.single_mutant_contribution(paired) > 0.50
    paired2 = {"a": 0.05, "b": 0.05, "c": 0.10}
    assert ev.single_mutant_contribution(paired2) <= 0.50
    assert ev.single_mutant_contribution({}) == 0.0


def test_catastrophic_comparator_uses_same_fold_control():
    # catastrophic: per-mutant error > 2x same-fold training-mean error;
    # baseline 0 -> error > 0.05 (ADR-003 G6)
    targets = {"a": np.array([0.9, 0.05, 0.05]),
               "b": np.array([0.8, 0.1, 0.1]),
               "c": np.array([0.7, 0.2, 0.1])}
    preds = {"a": np.array([0.1, 0.6, 0.3]),  # err 0.533 > 2*0.1 baseline
             "b": np.array([0.8, 0.1, 0.1]),
             "c": np.array([0.7, 0.2, 0.1])}
    tr_of = {"a": [1, 2], "b": [0, 2], "c": [0, 1]}
    flags = ev.catastrophic_flags(preds, targets, ["a", "b", "c"], tr_of)
    assert flags["a"] is True   # far from training mean
    assert flags["b"] is False
    assert flags["c"] is False


def test_catastrophic_zero_baseline_floor():
    targets = {"a": np.array([1.0, 0.0, 0.0]),
               "b": np.array([1.0, 0.0, 0.0]),
               "c": np.array([1.0, 0.0, 0.0])}
    preds = {"a": np.array([0.90, 0.10, 0.00]),  # err 0.0667 > 0.05
             "b": np.array([0.98, 0.01, 0.01]),  # err 0.0133 < 0.05
             "c": np.array([1.0, 0.0, 0.0])}
    tr_of = {"a": [1, 2], "b": [0, 2], "c": [0, 1]}
    flags = ev.catastrophic_flags(preds, targets, ["a", "b", "c"], tr_of,
                                  floor_zero=0.05)
    assert flags["a"] is True
    assert flags["b"] is False
    assert flags["c"] is False


def test_gates_keys_are_independent():
    nested = {"nested_mae": 0.2, "errors": {"a": 0.2},
              "catastrophic_folds": []}
    ctl = {"nested_mae": 0.3, "errors": {"a": 0.3},
           "catastrophic_folds": []}
    g = ev.gates(nested, ctl, {}, None, "src")
    expected = {"src_nested_le_0_2560", "no_marker_strictly_beats_marker",
                "lso_same_route_pass", "catastrophic_not_worse_than_control",
                "single_mutant_contribution_le_0_50"}
    assert set(g) == expected, set(g)


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
