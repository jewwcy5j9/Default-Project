from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np
import pytest

V3 = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_custodian_v3", V3 / "run_custodian_v3.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def minimal_public(n=8):
    seq = "ACDEFGHIKLMNPQRSTVWY"
    return {
        "protocol_version": "3.0.0", "panel_id": "TEST", "wild_type_sequence": seq,
        "wt_population": [0.6, 0.2, 0.2], "state_definitions": ["a", "b", "c"],
        "conditions": {"construct": "x", "ligand": "x", "temperature": "x",
                       "buffer": "x", "state_model": "x"},
        "mutations": [{"mutation_id": f"m{i}", "substitutions": [
            {"sequence_index_1based": i + 1, "from": seq[i], "to": "A" if seq[i] != "A" else "V"}]}
            for i in range(n)]}


def test_main_gate_and_simplex():
    ids, wt, tier = runner.validate_public(minimal_public(8))
    assert len(ids) == 8 and tier == "primary_eligible"
    assert np.allclose(wt.sum(), 1.0)
    assert np.allclose(runner.norm_simplex([2, 3, 5]), [0.2, 0.3, 0.5])


def test_v3_1_diagnostic_definitions_frozen():
    d = runner.DIAGNOSTIC_DEFINITIONS
    assert set(d) == {"layer", "p1", "p2", "p3", "p4", "scoring"}
    assert d["layer"] == "3.1.0"
    assert runner.DIAG_NULL_ALPHA == 0.05
    assert runner.DIAG_P3_BAND == (0.005, 0.15)
    frozen = hashlib.sha256(
        json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()
    assert frozen == "4c4d90f8ce5499b71ffdd148198f088bed0e9ad0a44aa8821b7f00e0c5e3fb1e"


def test_supporting_gate():
    assert runner.validate_public(minimal_public(7))[2] == "supporting_only"


def test_pca_is_outer_train_only():
    ids = [f"m{i}" for i in range(8)]
    rng = np.random.default_rng(1)
    vectors = {mid: rng.normal(size=12) for mid in ids}
    rows = {mid: rng.normal(size=(20, 12)) for mid in ids}
    train = ids[:-1]
    X, meta = runner.fit_outer_pca(train, vectors, rows)
    assert meta["fit_ids"] == train
    assert ids[-1] not in meta["fit_ids"]
    assert meta["d_eff"] == min(20, len(train) - 1)
    assert set(X) == set(ids)


def test_exact_sign_and_contrasts():
    test = runner.exact_sign_test([1, 2, 3, 4, 5, 6])
    assert test["p_one_sided"] == 1 / 64
    z = runner.ilr([0.6, 0.3, 0.1])
    assert z.shape == (2,) and np.isfinite(z).all()


def test_collision_label_permutation_endpoint():
    ids = ["a", "b", "c"]
    vectors = {"a": np.array([0.0, 0.0]), "b": np.array([0.0, 0.0]),
               "c": np.array([1.0, 0.0])}
    targets = {"a": np.array([0.8, 0.1, 0.1]),
               "b": np.array([0.1, 0.8, 0.1]),
               "c": np.array([0.1, 0.1, 0.8])}
    predictions = {key: value.copy() for key, value in targets.items()}
    result = runner.collision_diagnostic(ids, vectors, targets, predictions)
    assert result["conflict_members"] == ["a", "b"]
    assert result["label_permutation_null"]["method"] == "exhaustive_exact_label_permutation"


def test_dry_run_artifact_has_no_outer_leakage():
    path = V3 / "dry_run_outputs" / "custodian_result_v3.json"
    if not path.exists():
        # dry_run_outputs/ is repository-only evidence (excluded from the
        # custodian package); skip loudly rather than pass vacuously.
        pytest.skip("dry_run_outputs not present (repository-only evidence)")
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["status"] == "DRY RUN / NOT EVIDENCE"
    assert result["panel_tier"] == "supporting_only"
    for fold in result["folds"]:
        assert fold["test_id"] not in fold["outer_train_ids"]
        assert fold["pca"]["fit_ids"] == fold["outer_train_ids"]
        assert np.isclose(sum(fold["prediction"]), 1.0)
        assert min(fold["prediction"]) >= 0
