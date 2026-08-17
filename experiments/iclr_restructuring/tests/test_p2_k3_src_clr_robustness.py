import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from p2_k3_src_clr_robustness import (
    PSEUDOCOUNTS,
    clr_with_pseudocount,
    inverse_clr,
    sample_simplex_interval,
    summarize_pseudocount_rows,
)


def test_clr_round_trip_respects_pseudocount():
    values = np.array([[0.0, 1.0, 0.0], [0.72, 0.07, 0.21]])
    recovered = inverse_clr(clr_with_pseudocount(values, 1e-4))
    expected = np.clip(values, 1e-4, 1.0)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(recovered, expected)


def test_interval_samples_stay_in_simplex_and_component_bounds():
    rng = np.random.default_rng(7)
    center = np.array([0.0, 1.0, 0.0])
    samples = np.array([sample_simplex_interval(rng, center, 0.1) for _ in range(100)])
    np.testing.assert_allclose(samples.sum(axis=1), 1.0)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)
    assert np.all(np.abs(samples - center) <= 0.1 + 1e-12)


def test_pseudocount_summary_detects_stable_patterns():
    rows = []
    for pseudocount in PSEUDOCOUNTS:
        for candidate in ("pos", "ext", "llr_pos", "llr_only", "pca20"):
            for model_index, model in enumerate(("CLR-Ridge", "CLR-GP")):
                rows.append({
                    "pseudocount": pseudocount,
                    "candidate": candidate,
                    "model": model,
                    "mae": 0.1 + 0.01 * model_index,
                    "u2_gt_u1": True,
                    "f405a_gt_l410a": True,
                })
    summary = summarize_pseudocount_rows(rows)
    assert summary["mae_ranking_stable"] is True
    assert summary["all_rows_u2_gt_u1"] is True
    assert summary["f405a_vs_l410a_pattern_stable"] is True
