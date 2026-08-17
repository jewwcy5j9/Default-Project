import csv
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.src_k3_labels import (
    CANONICAL_SRC_K3_PATH,
    SRC_K3_CORE_IDS,
    SRC_K3_L410A_SUBSTITUTION_PROTOCOL_ID,
    SRC_K3_PRIMARY_PROTOCOL_ID,
    build_src_k3_panel,
    collapse_src_k3_non_active,
    load_src_k3_records,
    src_k3_sha256,
)


def test_primary_panel_values_and_order():
    panel = build_src_k3_panel(SRC_K3_PRIMARY_PROTOCOL_ID)
    assert panel.wt_population == (0.72, 0.07, 0.21)
    assert tuple(panel.targets) == SRC_K3_CORE_IDS
    assert panel.targets["SrcKD-V332I"] == (0.48, 0.52, 0.0)
    assert panel.targets["SrcKD-F405A"] == (0.0, 0.16, 0.84)
    assert len(panel.canonical_sha256) == 64
    assert panel.canonical_sha256 == src_k3_sha256()


def test_l410a_panel_changes_exactly_one_measurement_record():
    primary = build_src_k3_panel(SRC_K3_PRIMARY_PROTOCOL_ID)
    hybrid = build_src_k3_panel(SRC_K3_L410A_SUBSTITUTION_PROTOCOL_ID)
    assert hybrid.wt_record_id == primary.wt_record_id
    assert hybrid.wt_population == primary.wt_population
    changed = {name for name in primary.targets if primary.targets[name] != hybrid.targets[name]}
    assert changed == {"SrcKD-L410A"}
    assert hybrid.targets["SrcKD-L410A"] == (0.96, 0.03, 0.01)
    assert hybrid.protocol_kind == "hybrid_single_substitution"
    assert len(hybrid.substitutions) == 1


def test_ambiguous_v332i_is_quarantined():
    rows = load_src_k3_records()
    ambiguous = next(row for row in rows if row["record_id"] == "legacy_ambiguous__SrcKD-V332I")
    assert ambiguous["population"] == (0.73, 0.27, 0.0)
    assert ambiguous["status"] == "ambiguous_not_used"
    assert ambiguous["used_in_primary"] is False
    for protocol in (SRC_K3_PRIMARY_PROTOCOL_ID, SRC_K3_L410A_SUBSTITUTION_PROTOCOL_ID):
        assert "legacy_ambiguous__SrcKD-V332I" not in build_src_k3_panel(protocol).target_record_ids.values()


def test_k2_collapse_matches_frozen_values():
    panel = build_src_k3_panel(SRC_K3_PRIMARY_PROTOCOL_ID)
    targets, wt = collapse_src_k3_non_active(panel)
    assert wt == pytest.approx(0.28)
    assert tuple(targets.values()) == pytest.approx((0.27, 0.52, 0.91, 1, 1, 1, 1, 1))


def test_duplicate_record_id_is_rejected(tmp_path):
    rows = list(csv.reader(CANONICAL_SRC_K3_PATH.open(newline="", encoding="utf-8")))
    rows.append(rows[1])
    path = tmp_path / "duplicate.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    with pytest.raises(ValueError, match="Duplicate"):
        load_src_k3_records(path)


def test_generic_global_protocol_is_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        build_src_k3_panel("global")
