"""Tests for result_schema.json + validate_result_schema.py.

Run: python tests/test_result_schema.py   (no pytest required)
     pytest tests/test_result_schema.py   (also works)
Plan: SOTA_FOLLOWUP_EXECUTION_PLAN.md Workstream A1.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from validate_result_schema import load_schema, validate, validate_file  # noqa: E402

FIX_PASS = HERE / "fixtures" / "confirmatory_fixture.json"
FIX_FAIL = HERE / "fixtures" / "missing_fields_fixture.json"


def test_valid_fixture_passes():
    errors = validate_file(FIX_PASS, verbose=False)
    assert errors == [], f"fixture should be valid, got: {errors}"


def test_missing_fields_fixture_fails():
    errors = validate_file(FIX_FAIL, verbose=False)
    assert errors, "missing-field fixture should be invalid"
    joined = "\n".join(errors)
    assert "metadata.hash" in joined
    assert "data_hashes" in joined
    assert "label_info" in joined
    assert "folds" in joined
    assert "hard_gates" in joined


def test_schema_file_itself_is_valid_json():
    load_schema()  # raises on malformed JSON


def test_p2_system_wrapper_validates():
    result = ROOT / "results" / "p2_k3_nested_results.json"
    errors = validate_file(result, verbose=False)
    assert errors == [], f"P2 wrapper should be valid, got: {errors}"


def test_required_top_level_fields():
    schema = load_schema()
    data = json.loads(FIX_PASS.read_text(encoding="utf-8"))
    for key in ["metadata", "data_hashes", "label_info", "folds", "results", "metrics", "hard_gates"]:
        assert key in data, f"fixture missing {key}"


def test_simplex_predictions_rejected():
    """K=3 predictions must be length-3 arrays; 2-length must be caught."""
    schema = load_schema()
    data = json.loads(FIX_PASS.read_text(encoding="utf-8"))
    fold = data["results"]["per_fold"]["M290L"]
    fold["mean_predictions"] = [0.5, 0.5]  # K=2 collapse
    errors = validate(data, schema)
    assert any("minItems" in e or "maxItems" in e for e in errors), errors


def test_simplex_sum_enforced():
    """Probability vectors must sum to 1 within 1e-6 (custom simplex keyword)."""
    schema = load_schema()
    data = json.loads(FIX_PASS.read_text(encoding="utf-8"))
    fold = data["results"]["per_fold"]["M290L"]
    fold["mean_predictions"] = [0.9, 0.1, 0.05]  # sum = 1.05
    errors = validate(data, schema)
    assert any("simplex" in e for e in errors), errors


def test_wt_population_simplex_enforced():
    schema = load_schema()
    data = json.loads(FIX_PASS.read_text(encoding="utf-8"))
    data["label_info"]["wt_population"] = [0.88, 0.06, 0.06, 0.0]  # wrong length -> length error
    errors = validate(data, schema)
    assert any("maxItems" in e for e in errors), errors
    data2 = json.loads(FIX_PASS.read_text(encoding="utf-8"))
    data2["label_info"]["wt_population"] = [0.9, 0.1, 0.1]  # sum = 1.1
    errors2 = validate(data2, schema)
    assert any("simplex" in e for e in errors2), errors2


def test_min_properties_enforced():
    """Maps that must be populated (per_fold, mae_per_mutant, leave_site_out,
    support_stratified_error) reject empty dicts."""
    schema = load_schema()
    data = json.loads(FIX_PASS.read_text(encoding="utf-8"))
    data["metrics"]["mae_per_mutant"] = {}
    errors = validate(data, schema)
    assert any("minProperties" in e for e in errors), errors
    data2 = json.loads(FIX_PASS.read_text(encoding="utf-8"))
    data2["results"]["per_fold"] = {}
    errors2 = validate(data2, schema)
    assert any("minProperties" in e for e in errors2), errors2


def test_probe_or_global_fit_required():
    schema = load_schema()
    data = json.loads(FIX_PASS.read_text(encoding="utf-8"))
    del data["label_info"]["probe_or_global_fit"]
    errors = validate(data, schema)
    assert any("probe_or_global_fit" in e for e in errors), errors


def test_ilr_required():
    schema = load_schema()
    data = json.loads(FIX_PASS.read_text(encoding="utf-8"))
    del data["metrics"]["u1_u2_contrast_mae"]["ilr_z2"]
    errors = validate(data, schema)
    assert any("ilr_z2" in e for e in errors), errors


def test_direction_format():
    schema = load_schema()
    data = json.loads(FIX_PASS.read_text(encoding="utf-8"))
    data["metrics"]["direction"] = "seven"
    errors = validate(data, schema)
    assert any("direction" in e for e in errors), errors


def test_short_hash_rejected():
    schema = load_schema()
    data = json.loads(FIX_PASS.read_text(encoding="utf-8"))
    data["metadata"]["hash"] = "short"
    errors = validate(data, schema)
    assert any("hash" in e for e in errors), errors


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
