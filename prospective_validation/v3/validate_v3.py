#!/usr/bin/env python
"""Validate v2 immutability, v3.0 frozen history, v3.1 package hashes, dry run, and synthetic outputs.

The v3.1 manifest (archive_manifest_v3_1.json) is authoritative for the
working-tree files and the custodian archive. The v3.0 zip is frozen
history: it is verified against its own manifest inside the zip only,
because its superseded working-tree files legitimately evolved for v3.1.
The dry-run and synthetic sections require repository-only artifacts that
are deliberately excluded from the custodian package; their absence is
reported loudly as an error rather than skipped silently.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Package members whose sources live outside prospective_validation/v3;
# keep in sync with V3_1_EXTRA / V3_1_VENDORED in build_v3_package.py.
V3_1_EXTERNAL_SOURCES = {
    "p6_audit_detection_benchmark.json":
        ROOT / "experiments" / "iclr_restructuring" / "results"
        / "p6_audit_detection_benchmark.json",
    "src/models/low_rank_cdst.py":
        ROOT / "src" / "models" / "low_rank_cdst.py",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    errors = []

    # --- v3.1 package (authoritative for the working tree) ---
    manifest = json.loads((HERE / "archive_manifest_v3_1.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        source = V3_1_EXTERNAL_SOURCES.get(name, HERE / name)
        if not source.exists():
            errors.append(f"v3.1 source missing: {name} ({source})")
        elif sha256(source) != expected:
            errors.append(f"v3.1 content hash mismatch: {name}")
    archive = HERE / manifest["archive"]
    if sha256(archive) != manifest["archive_sha256"]:
        errors.append("v3.1 archive hash mismatch")
    else:
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
            expected_names = set(manifest["files"]) | {"package_manifest.json"}
            if names != expected_names:
                errors.append("v3.1 archive entry set mismatch")
            for name, expected in manifest["files"].items():
                if hashlib.sha256(zf.read(name)).hexdigest() != expected:
                    errors.append(f"v3.1 zipped content hash mismatch: {name}")
    for name, expected in manifest["v2_immutable_hashes"].items():
        if sha256(HERE.parent / name) != expected:
            errors.append(f"v2 immutability failure: {name}")

    # --- v3.0 archive (frozen history; zip-internal verification only) ---
    v30 = json.loads((HERE / "archive_manifest_v3.json").read_text(encoding="utf-8"))
    archive30 = HERE / v30["archive"]
    if sha256(archive30) != v30["archive_sha256"]:
        errors.append("v3.0 archive hash mismatch (frozen history)")
    else:
        with zipfile.ZipFile(archive30) as zf:
            names30 = set(zf.namelist())
            if names30 != set(v30["files"]) | {"package_manifest.json"}:
                errors.append("v3.0 archive entry set mismatch")
            for name, expected in v30["files"].items():
                if hashlib.sha256(zf.read(name)).hexdigest() != expected:
                    errors.append(f"v3.0 zipped content hash mismatch: {name}")
    if manifest["v3_0_immutable"]["archive_sha256"] != v30["archive_sha256"]:
        errors.append("v3.1 manifest v3.0-immutability pin disagrees with v3.0 manifest")

    # --- dry run (repository-only evidence) ---
    dry_path = HERE / "dry_run_outputs" / "custodian_result_v3.json"
    if not dry_path.exists():
        errors.append(
            "dry-run outputs not present (repository-only evidence, excluded "
            "from the custodian package; run dry_run_v3.py from the repo tree)")
        dry = None
    else:
        dry = json.loads(dry_path.read_text(encoding="utf-8"))
        if dry["status"] != "DRY RUN / NOT EVIDENCE" or dry["panel_tier"] != "supporting_only":
            errors.append("dry-run evidence/tier label failure")
        for fold in dry["folds"]:
            if fold["test_id"] in fold["outer_train_ids"]:
                errors.append(f"outer leakage: {fold['test_id']}")
            if fold["pca"]["fit_ids"] != fold["outer_train_ids"]:
                errors.append(f"PCA fit-ID mismatch: {fold['test_id']}")
            for pred in fold["per_seed_predictions"] + [fold["prediction"]]:
                if min(pred) < 0 or not np.isclose(sum(pred), 1.0, atol=1e-7):
                    errors.append(f"simplex failure: {fold['test_id']}")

    # --- controlled synthetic suite (repository-only evidence) ---
    synthetic_dir = ROOT / "experiments" / "iclr_restructuring" / "results"
    sm_path = synthetic_dir / "p4_support_resolution_selection_manifest.json"
    if not sm_path.exists():
        errors.append(
            "synthetic manifest not present (repository-only evidence, "
            "excluded from the custodian package)")
        sm = None
    else:
        sm = json.loads(sm_path.read_text(encoding="utf-8"))
        if (sm["records"], sm["settings"], sm["repeats_per_setting"]) != (72000, 360, 200):
            errors.append("synthetic factorial count failure")
        if sha256(synthetic_dir / "p4_support_resolution_selection.json") != sm["json_sha256"]:
            errors.append("synthetic JSON hash mismatch")
        if sha256(synthetic_dir / "p4_support_resolution_selection_by_setting.csv") != sm["csv_sha256"]:
            errors.append("synthetic CSV hash mismatch")

    if errors:
        raise SystemExit("\n".join(errors))
    print(json.dumps({"status": "PASS",
                      "v2_files_unchanged": len(manifest["v2_immutable_hashes"]),
                      "v3_1_files_verified": len(manifest["files"]),
                      "v3_0_zip_frozen": True,
                      "dry_run_folds": len(dry["folds"]) if dry else None,
                      "synthetic_records": sm["records"] if sm else None}, indent=2))


if __name__ == "__main__":
    main()
