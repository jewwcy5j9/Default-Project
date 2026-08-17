"""Validated label authority for the Src three-state benchmark."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


CANONICAL_SRC_K3_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "nmr_populations" / "src_k3_canonical.csv"
)
SRC_K3_STATE_ORDER = ("Active", "E1", "E2")
SRC_K3_CORE_IDS = (
    "SrcKD-L410A",
    "SrcKD-V332I",
    "SrcKD-L270F_V332I",
    "SrcKD-L325A",
    "SrcKD-A311I",
    "SrcKD-V380A",
    "SrcKD-V331A",
    "SrcKD-F405A",
)
SRC_K3_PRIMARY_PROTOCOL_ID = "src_k3_figs5_met305_primary_v1"
SRC_K3_L410A_SUBSTITUTION_PROTOCOL_ID = (
    "src_k3_figs5_met305_with_table_s2_l410a_substitution_v1"
)

_FIELDS = (
    "record_id", "mutation_id", "state_A", "state_E1", "state_E2",
    "measurement_scope", "probe_id", "source_location", "extraction_method",
    "uncertainty_kind", "uncertainty", "used_in_primary", "panel_order",
    "status", "dataset_version", "notes",
)
_SCOPES = {"probe_specific", "global_fit", "ambiguous_legacy"}
_STATUSES = {"accepted", "ambiguous_not_used"}
_EXTRACTION_METHODS = {"visual_bar_read", "direct_table", "legacy_curated_entry"}
_UNCERTAINTY_KINDS = {
    "curator_visual_range_fraction", "reported_sd_fraction", "not_available"
}


@dataclass(frozen=True)
class SrcK3Panel:
    protocol_id: str
    protocol_kind: str
    wt_record_id: str
    wt_population: tuple[float, float, float]
    targets: dict[str, tuple[float, float, float]]
    target_record_ids: dict[str, str]
    substitutions: tuple[dict[str, str], ...]
    canonical_sha256: str


def src_k3_sha256(path: Path | str = CANONICAL_SRC_K3_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_src_k3_records(path: Path | str = CANONICAL_SRC_K3_PATH) -> tuple[dict, ...]:
    """Load and strictly validate measurement records from the canonical CSV."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _FIELDS:
            raise ValueError("Unsupported Src K3 CSV schema")
        raw_rows = list(reader)

    records = []
    seen = set()
    for raw in raw_rows:
        record_id = raw["record_id"]
        if not record_id or record_id in seen:
            raise ValueError(f"Duplicate or empty record_id: {record_id!r}")
        seen.add(record_id)
        values = tuple(Decimal(raw[key]) for key in ("state_A", "state_E1", "state_E2"))
        if any(not value.is_finite() or value < 0 or value > 1 for value in values):
            raise ValueError(f"Invalid population bounds for {record_id}")
        if sum(values) != Decimal("1"):
            raise ValueError(f"Population does not sum exactly to one for {record_id}")
        if raw["measurement_scope"] not in _SCOPES:
            raise ValueError(f"Invalid measurement_scope for {record_id}")
        if raw["status"] not in _STATUSES:
            raise ValueError(f"Invalid status for {record_id}")
        if raw["extraction_method"] not in _EXTRACTION_METHODS:
            raise ValueError(f"Invalid extraction_method for {record_id}")
        if raw["uncertainty_kind"] not in _UNCERTAINTY_KINDS:
            raise ValueError(f"Invalid uncertainty_kind for {record_id}")
        scope = raw["measurement_scope"]
        if (scope == "probe_specific") != (raw["probe_id"] == "Met305"):
            raise ValueError(f"Invalid probe/scope combination for {record_id}")
        primary = raw["used_in_primary"].lower()
        if primary not in {"true", "false"}:
            raise ValueError(f"Invalid used_in_primary for {record_id}")
        panel_order = int(raw["panel_order"]) if raw["panel_order"] else None
        records.append({
            **raw,
            "population": tuple(float(value) for value in values),
            "used_in_primary": primary == "true",
            "panel_order": panel_order,
        })

    primary_rows = sorted(
        (row for row in records if row["used_in_primary"]),
        key=lambda row: -1 if row["panel_order"] is None else row["panel_order"],
    )
    expected_ids = ("SrcKD-WT",) + SRC_K3_CORE_IDS
    if len(primary_rows) != 9 or tuple(row["mutation_id"] for row in primary_rows) != expected_ids:
        raise ValueError("Primary panel must be ordered WT plus the eight frozen core mutations")
    if tuple(row["panel_order"] for row in primary_rows) != tuple(range(9)):
        raise ValueError("Primary panel_order must be exactly 0 through 8")
    if any(row["status"] != "accepted" or row["measurement_scope"] != "probe_specific"
           for row in primary_rows):
        raise ValueError("Every primary record must be an accepted probe-specific measurement")

    by_id = {row["record_id"]: row for row in records}
    for required in (
        "table_s2_global__SrcKD-WT",
        "table_s2_global__SrcKD-L410A",
        "legacy_ambiguous__SrcKD-V332I",
    ):
        if required not in by_id:
            raise ValueError(f"Missing required provenance record: {required}")
    ambiguous = by_id["legacy_ambiguous__SrcKD-V332I"]
    if ambiguous["status"] != "ambiguous_not_used" or ambiguous["used_in_primary"]:
        raise ValueError("The ambiguous V332I record must remain quarantined")
    return tuple(records)


def build_src_k3_panel(
    protocol_id: str,
    path: Path | str = CANONICAL_SRC_K3_PATH,
) -> SrcK3Panel:
    """Construct either the primary panel or the one-row L410A sensitivity panel."""
    records = load_src_k3_records(path)
    primary = sorted(
        (row for row in records if row["used_in_primary"]),
        key=lambda row: row["panel_order"],
    )
    wt = primary[0]
    targets = {row["mutation_id"]: row["population"] for row in primary[1:]}
    record_ids = {row["mutation_id"]: row["record_id"] for row in primary[1:]}
    substitutions: tuple[dict[str, str], ...] = ()
    kind = "primary_probe"
    if protocol_id == SRC_K3_L410A_SUBSTITUTION_PROTOCOL_ID:
        replacement = next(
            row for row in records if row["record_id"] == "table_s2_global__SrcKD-L410A"
        )
        previous = record_ids["SrcKD-L410A"]
        targets["SrcKD-L410A"] = replacement["population"]
        record_ids["SrcKD-L410A"] = replacement["record_id"]
        substitutions = ({
            "mutation_id": "SrcKD-L410A",
            "from_record_id": previous,
            "to_record_id": replacement["record_id"],
        },)
        kind = "hybrid_single_substitution"
    elif protocol_id != SRC_K3_PRIMARY_PROTOCOL_ID:
        raise ValueError(f"Unknown or unsupported Src K3 protocol: {protocol_id}")
    return SrcK3Panel(
        protocol_id=protocol_id,
        protocol_kind=kind,
        wt_record_id=wt["record_id"],
        wt_population=wt["population"],
        targets=targets,
        target_record_ids=record_ids,
        substitutions=substitutions,
        canonical_sha256=src_k3_sha256(path),
    )


def collapse_src_k3_non_active(panel: SrcK3Panel) -> tuple[dict[str, float], float]:
    targets = {mutation: population[1] + population[2]
               for mutation, population in panel.targets.items()}
    return targets, panel.wt_population[1] + panel.wt_population[2]
