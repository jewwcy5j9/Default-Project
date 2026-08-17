import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import calibrate_assignments as calibration


def test_threshold_curve_uses_strict_existing_assignment_rule():
    records = [{
        "rmsds": {"active": 3.0, "i1": 4.0, "i2": 5.0},
    }]
    curve = calibration.threshold_curve(records)
    at_3 = next(point for point in curve if point["threshold_angstrom"] == 3.0)
    above_3 = next(point for point in curve if point["threshold_angstrom"] == 3.25)
    assert at_3["hard_assignment_counts"]["unclassified"] == 1
    assert above_3["hard_assignment_counts"]["active"] == 1


def test_margin_rule_separates_ambiguous_from_unclassified():
    ambiguous = calibration.assign_with_margin(
        {"active": 2.0, "i1": 2.4, "i2": 5.0}, threshold=3.0
    )
    unclassified = calibration.assign_with_margin(
        {"active": 3.0, "i1": 3.2, "i2": 5.0}, threshold=3.0
    )
    assigned = calibration.assign_with_margin(
        {"active": 2.0, "i1": 2.5, "i2": 5.0}, threshold=3.0
    )
    assert ambiguous["state"] == "ambiguous"
    assert ambiguous["margin"] == pytest.approx(0.4)
    assert unclassified["state"] == "unclassified"
    assert assigned["state"] == "active"


def test_reference_calibration_has_all_pairs_and_regions():
    parser = calibration.PDBParser(QUIET=True)
    references, _ = calibration.load_references(parser)
    region_offsets = calibration.compute_region_offsets(references)
    result = calibration.reference_calibration(references, region_offsets)
    assert set(result) == set(calibration.REGIONS)
    for region in result.values():
        pairs = region["pairwise_distances"]
        assert set(pairs) == {"active__I1", "active__I2", "I1__I2"}
        assert all(value["rmsd_angstrom"] > 0 for value in pairs.values())
        assert all(value["n_ca_atoms"] >= 10 for value in pairs.values())


def test_report_disclaims_population_interpretation():
    result = {
        "ambiguity_rule": {"margin_cutoff_angstrom": 0.5},
        "reference_calibration": {
            "full_protein": {"pairwise_distances": {
                "active__I1": {"rmsd_angstrom": 1.0},
                "active__I2": {"rmsd_angstrom": 2.0},
                "I1__I2": {"rmsd_angstrom": 3.0},
            }},
        },
        "protocols": {},
        "method": {"implementation_path": "classify_states.py",
                   "implementation_sha256": "abc"},
        "references": {},
    }
    report = calibration.render_markdown(result).lower()
    assert "not estimates of thermodynamic populations" in report
    assert "estimated population" not in report
    assert "inferred population" not in report
    assert "true population" not in report
