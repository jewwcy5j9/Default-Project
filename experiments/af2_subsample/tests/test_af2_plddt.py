import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from aggregate_af2_plddt import build_payload, mean_ca_plddt


def _pdb_line(record, atom, bfactor):
    return (
        f"{record:<6}{1:>5} {atom:^4} {'ALA':>3} A{1:>4}    "
        f"{0.0:>8.3f}{0.0:>8.3f}{0.0:>8.3f}{1.0:>6.2f}{bfactor:>6.2f}"
        f"          {'C':>2}\n"
    )


def test_mean_ca_plddt_uses_only_atom_ca(tmp_path):
    path = tmp_path / "one.pdb"
    path.write_text(
        _pdb_line("ATOM", "CA", 80.0)
        + _pdb_line("ATOM", "N", 10.0)
        + _pdb_line("HETATM", "CA", 20.0)
        + _pdb_line("ATOM", "CA", 90.0),
        encoding="utf-8",
    )
    mean, count = mean_ca_plddt(path)
    assert count == 2
    assert mean == 85.0


def test_empty_pdb_is_rejected(tmp_path):
    path = tmp_path / "empty.pdb"
    path.write_text("END\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No CA"):
        mean_ca_plddt(path)


def test_repository_af2_inventory_and_regression_values():
    payload = build_payload()
    original = payload["protocols"]["original"]
    fresh = payload["protocols"]["fresh_msa"]
    assert (original["found_structures"], fresh["found_structures"]) == (840, 480)
    assert (original["n_ca_min"], original["n_ca_max"]) == (263, 263)
    assert (fresh["n_ca_min"], fresh["n_ca_max"]) == (263, 263)
    assert original["mutant_mean_range"] == pytest.approx((80.9587442966, 81.7324987326))
    assert fresh["mutant_mean_range"] == pytest.approx((54.5635621039, 57.3558282636))
