import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from p2_k3_unified_robustness import build_result, summarize_frozen_errors


def test_summarize_frozen_errors_distinguishes_deletion_from_refit():
    result = summarize_frozen_errors({"single_a": 0.1, "double": 0.4, "single_b": 0.2}, "double")

    assert result["all_mutation_mae"] == pytest.approx(0.7 / 3)
    assert result["leave_one_observation_out"]["minimum_mae"] == pytest.approx(0.15)
    assert result["leave_one_observation_out"]["minimum_when_deleted"] == "double"
    assert result["double_mutant_exclusion"]["mae"] == pytest.approx(0.15)
    assert "no model or selector is refit" in result["double_mutant_exclusion"]["definition"]


def test_build_result_keeps_routes_separate_and_marks_lso_as_retrained(monkeypatch):
    monkeypatch.setattr(
        "p2_k3_unified_robustness.sha256_file", lambda path: f"hash:{path.name}"
    )
    paired = {
        "systems": {
            "abl1": {"per_mutation": [
                {"mutation": "M290L", "mae_nested_mlp": 0.1},
                {"mutation": "M290L_L301I", "mae_nested_mlp": 0.3},
            ]},
            "src": {"per_mutation": [
                {"mutation": "SrcKD-L410A", "mae_nested_mlp": 0.2},
                {"mutation": "SrcKD-L270F_V332I", "mae_nested_mlp": 0.4},
            ]},
        }
    }
    confirmatory = {
        "systems": {
            "abl1": {"metrics": {
                "mae": 0.3,
                "mae_per_mutant": {"M290L": 0.2, "M290L_L301I": 0.4},
                "leave_site_out": {"site": {"group_mae": 0.5, "comparator": 0.3}},
            }},
            "src": {"metrics": {
                "mae": 0.4,
                "mae_per_mutant": {"SrcKD-L410A": 0.3, "SrcKD-L270F_V332I": 0.5},
                "leave_site_out": {"site": {"group_mae": 0.6, "comparator": 0.4}},
            }},
        }
    }

    result = build_result(paired, confirmatory)

    assert result["protocol"]["routes_must_remain_separate"] is True
    assert set(result["routes"]) == {
        "representation_selection_audit", "candidate_model_confirmatory"
    }
    lso = result["routes"]["candidate_model_confirmatory"]["systems"]["abl1"]["leave_site_out_retraining"]
    assert "retrained" in lso["definition"]
    assert lso["groups"]["site"]["group_mae"] == 0.5
