import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent))

import p2_k3_eval_v2 as eval_v2
import p2_k3_nested_pca as nested_pca


def test_nested_checkpoint_load_requires_exact_signature(tmp_path: Path):
    path = tmp_path / "nested.json"
    signature = {"schema": "nested", "system": "abl1"}
    payload = {
        "signature": signature,
        "block": {},
        "nested_mlp": {},
        "nested_model_select": {},
        "names": [],
    }
    nested_pca.atomic_write_json(path, payload)

    assert nested_pca.load_system_checkpoint(path, signature) == payload
    assert nested_pca.load_system_checkpoint(path, {**signature, "system": "src"}) is None


def test_eval_checkpoint_rejects_incomplete_payload(tmp_path: Path):
    path = tmp_path / "eval.json"
    signature = {"schema": "eval", "system": "src"}
    path.write_text(json.dumps({"signature": signature, "block": {}}),
                    encoding="utf-8")

    assert eval_v2.load_system_checkpoint(path, signature) is None


def test_eval_signature_changes_with_scientific_parameters(monkeypatch):
    monkeypatch.setattr(eval_v2, "sha256_file", lambda path: str(path))
    base = argparse.Namespace(version="2.0.0", seeds=5, skip_alt=False)
    changed = argparse.Namespace(version="2.0.0", seeds=4, skip_alt=False)
    hashes = {"canonical_data": "a", "feature_cache": "b",
              "model_checkpoint": "c", "adr_002": "d", "adr_003": "e"}

    sig_base = eval_v2.checkpoint_signature("src", base, "script", hashes)
    sig_changed = eval_v2.checkpoint_signature("src", changed, "script", hashes)

    assert sig_base != sig_changed


def test_nested_progress_signature_survives_json_normalization(tmp_path: Path):
    path = tmp_path / "nested-progress.json"
    signature = {"schema": "nested", "candidates": [("pos", [1, 2])]}
    progress = {"signature": signature, "stages": {}}

    nested_pca.atomic_write_json(path, progress)

    loaded = nested_pca.load_progress_checkpoint(path, signature)
    assert loaded is not None
    assert loaded["signature"]["candidates"] == [["pos", [1, 2]]]


def test_nested_fixed_loo_resume_skips_completed_folds(monkeypatch):
    calls = []

    def fake_predict(*args, **kwargs):
        held_index = args[9][0]
        calls.append(held_index)
        pred = np.array([0.5, 0.3, 0.2])
        return pred, [pred], {"d": 1}

    monkeypatch.setattr(nested_pca, "predict_fold", fake_predict)
    names = ["a", "b"]
    targets = {name: np.array([0.5, 0.3, 0.2]) for name in names}
    resume = {
        "preds": {"a": [0.5, 0.3, 0.2]},
        "folds": {"a": {"pred": [0.5, 0.3, 0.2]}},
    }

    result = nested_pca.fixed_loo(
        "abl1", names, {}, 1, None, None, targets, "pos", "LowRankCDST",
        n_seeds=1, resume=resume)

    assert calls == [1]
    assert set(result["folds"]) == {"a", "b"}


def test_eval_fixed_loo_resume_preserves_public_keys(monkeypatch):
    calls = []

    def fake_predict(cname, mname, *args, **kwargs):
        calls.append((cname, mname))
        return {"mean": np.array([0.5, 0.3, 0.2])}

    monkeypatch.setattr(eval_v2, "predict_k3", fake_predict)
    monkeypatch.setattr(eval_v2, "combo_matrix",
                        lambda features, names, combo: np.ones((len(names), 1)))
    monkeypatch.setattr(eval_v2, "MODEL_NAMES", ["m1", "m2"])
    names = ["a", "b"]
    targets = {name: np.array([0.5, 0.3, 0.2]) for name in names}
    resume = {"c1::m1": {"mae": 0.0, "errors": {}, "exploratory": True}}

    result = eval_v2.run_fixed_loo(
        "abl1", names, {}, targets, np.array([0.5, 0.3, 0.2]),
        [("c1", ["x"])], n_seeds=1, resume=resume)

    assert set(result) == {"c1::m1", "c1::m2"}
    assert all(model == "m2" for _, model in calls)
