"""Regression tests for the synthetic experiment's fold-local protocol."""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import p4_support_resolution_selection as synthetic


def _dataset(epsilon=0.2):
    return synthetic.make_dataset(
        n=8, epsilon=epsilon, delta=0.6,
        rng=np.random.default_rng(123), max_m=5)


def test_nonzero_epsilon_is_preserved_or_reported():
    epsilon = 0.2
    features, targets = _dataset(epsilon)
    assert features[1, 0] - features[0, 0] == epsilon

    latent = synthetic.to_latent(targets, "full_k3")
    _, scaler_audit = synthetic.ridge_loo_candidates(
        features, latent, "full_k3", return_audit=True)
    report = synthetic.pair_separation_report(features, scaler_audit)
    assert report["raw"] == epsilon
    assert len(report["outer_fold_transformed"]) == 2
    assert np.isclose(
        report["outer_fold_transformed_mean"],
        np.mean(report["outer_fold_transformed"]))


def test_outer_covariate_does_not_change_inner_preprocessing():
    features, targets = _dataset()
    _, selected, audit = synthetic.nested_predictions(
        features, targets, "full_k3", m_values=[1, 5])

    changed = features.copy()
    changed[0] = 1e9
    _, changed_selected, changed_audit = synthetic.nested_predictions(
        changed, targets, "full_k3", m_values=[1, 5])

    assert audit[0] == changed_audit[0]
    assert selected[1][0] == changed_selected[1][0]
    assert selected[5][0] == changed_selected[5][0]


def test_nested_scaler_uses_outer_training_only():
    features, targets = _dataset()
    ids = np.arange(100, 108)
    _, _, audit = synthetic.nested_predictions(
        features, targets, "full_k3", m_values=[5], sample_ids=ids)

    for outer in audit:
        outer_id = outer["held_out_id"]
        outer_positions = [i for i, sample_id in enumerate(ids)
                           if sample_id != outer_id]
        outer_scaler = outer["outer_scaler"]
        assert outer_scaler["fit_ids"] == ids[outer_positions].tolist()
        assert np.allclose(outer_scaler["mean"],
                           features[outer_positions].mean(axis=0))
        assert np.allclose(outer_scaler["scale"],
                           features[outer_positions].std(axis=0))

        for inner in outer["inner_scalers"]:
            inner_positions = [i for i, sample_id in enumerate(ids)
                               if sample_id not in {outer_id, inner["held_out_id"]}]
            assert inner["fit_ids"] == ids[inner_positions].tolist()
            assert np.allclose(inner["mean"],
                               features[inner_positions].mean(axis=0))
            assert np.allclose(inner["scale"],
                               features[inner_positions].std(axis=0))
    assert synthetic.preprocessing_is_isolated(audit)


def test_figure_slice_matches_caption():
    slices = synthetic.FIGURE_SLICE_METADATA
    assert slices["panel_A"]["fixed"]["n"] == 20
    assert slices["panel_A"]["fixed"]["m"] == 1
    assert slices["panel_A"]["averaged_over"] == []
    assert slices["panel_B"]["fixed"] == {"n": 20, "m": 1}
    assert slices["panel_B"]["averaged_over"] == ["epsilon"]
    assert slices["panel_C"]["complete_factorial"] is True
    assert slices["panel_C"]["factorial_dimensions"] == {
        "n": synthetic.N_GRID,
        "epsilon": synthetic.EPS_GRID,
        "delta": synthetic.DELTA_GRID,
        "m": synthetic.M_GRID,
        "resolution": synthetic.RESOLUTIONS,
    }


def test_all_72000_records_retained():
    assert synthetic.factorial_counts() == {
        "settings": 360,
        "repeats": 200,
        "records": 72000,
    }
