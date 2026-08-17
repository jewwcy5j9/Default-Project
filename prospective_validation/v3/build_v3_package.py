#!/usr/bin/env python
"""Build deterministic prospective-v3 custodian and outreach artifacts."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent

V2_FILES = [
    "preregistration.md", "analysis_plan.md", "protocol_lock.json",
    "model_manifest.json", "archive_20260808_adr001.zip"]
PACKAGE_FILES = [
    "README.md", "preregistration_v3.md", "analysis_plan_v3.md",
    "protocol_lock_v3.json", "model_manifest_v3.json",
    "public_input_schema_v3.json", "private_target_schema_v3.json",
    "public_input_template_v3.json", "private_targets_template_v3.json",
    "environment_lock_v3.txt", "run_custodian_v3.py",
    "prepare_esm2_features_v3.py", "dry_run_v3.py",
    "reveal_log_v3.md", "validate_v3.py", "validation_report_v3.md",
    "tests/test_v3_protocol.py"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_zip_member(zf, arcname, data):
    info = zipfile.ZipInfo(arcname, date_time=(2026, 8, 10, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    zf.writestr(info, data)


V3_1_ARCHIVE = HERE / "prospective_v3_1_20260815.zip"
V3_1_FREEZE_DATE = "2026-08-15"
V3_1_FILES = [
    "README.md", "preregistration_v3.md", "analysis_plan_v3.md",
    "protocol_lock_v3.json", "model_manifest_v3.json",
    "public_input_schema_v3.json", "private_target_schema_v3.json",
    "public_input_template_v3.json", "private_targets_template_v3.json",
    "environment_lock_v3.txt", "run_custodian_v3.py",
    "prepare_esm2_features_v3.py", "dry_run_v3.py",
    "reveal_log_v3.md", "validate_v3.py", "validation_report_v3.md",
    "tests/test_v3_protocol.py",
    "preregistration_v3_1_amendment_20260814.md"]
# Extra artifacts stored in the v3.1 package but not in prospective_validation/v3.
V3_1_EXTRA = {
    "p6_audit_detection_benchmark.json":
        HERE.parent.parent / "experiments" / "iclr_restructuring" / "results"
        / "p6_audit_detection_benchmark.json",
}
# Vendored dependency: run_custodian_v3.py imports src.models.low_rank_cdst,
# so the frozen package must carry it to execute from a clean extraction.
V3_1_VENDORED = {
    "src/models/low_rank_cdst.py": HERE.parent.parent / "src" / "models"
    / "low_rank_cdst.py",
}
GENERATED_NOTE = (
    "Generated 2026-08-17 (v3.1 package rebuilt pre-send: vendored "
    "src/models/low_rank_cdst.py for self-contained custodian execution). "
    "Recompute immediately before actual send."
)


def main():
    # The v3.0 archive (prospective_v3_20260810.zip) is frozen history and is
    # never rebuilt; its hash is read from its manifest.
    v30_manifest = json.loads((HERE / "archive_manifest_v3.json").read_text(encoding="utf-8"))

    missing = [name for name in V3_1_FILES if not (HERE / name).exists()]
    if missing:
        raise FileNotFoundError(f"package inputs missing: {missing}")
    for name, path in {**V3_1_EXTRA, **V3_1_VENDORED}.items():
        if not path.exists():
            raise FileNotFoundError(f"extra artifact missing: {path}")

    # Self-containment: every src.* module that run_custodian_v3.py imports
    # must be vendored into the package.
    import ast
    tree = ast.parse((HERE / "run_custodian_v3.py").read_text(encoding="utf-8"))
    packaged = set(V3_1_FILES) | set(V3_1_EXTRA) | set(V3_1_VENDORED)
    uncontained = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
            if f"{'/'.join(node.module.split('.'))}.py" not in packaged:
                uncontained.append(node.module)
    if uncontained:
        raise AssertionError(
            f"custodian package not self-contained; unvendored imports: {uncontained}")

    content_hashes = {name: sha256(HERE / name) for name in V3_1_FILES}
    for name, path in {**V3_1_EXTRA, **V3_1_VENDORED}.items():
        content_hashes[name] = sha256(path)
    internal_manifest = {
        "package": "prospective_v3_1_20260815",
        "protocol_version": "3.1.0", "primary_protocol_version": "3.0.0",
        "freeze_date": V3_1_FREEZE_DATE, "files": content_hashes,
        "exclusions": ["outreach", "dry_run_outputs", "private target data",
                       "model weights", "v2 historical files"],
        "v3_0_immutable": {"archive": v30_manifest["archive"],
                           "archive_sha256": v30_manifest["archive_sha256"]}}
    with zipfile.ZipFile(V3_1_ARCHIVE, "w") as zf:
        for name in sorted(V3_1_FILES):
            write_zip_member(zf, name, (HERE / name).read_bytes())
        for name, path in sorted({**V3_1_EXTRA, **V3_1_VENDORED}.items()):
            write_zip_member(zf, name, path.read_bytes())
        write_zip_member(zf, "package_manifest.json",
                         (json.dumps(internal_manifest, indent=2) + "\n").encode("utf-8"))
    manifest = {
        **internal_manifest,
        "archive": V3_1_ARCHIVE.name, "archive_sha256": sha256(V3_1_ARCHIVE),
        "v2_immutable_hashes": {name: sha256(PARENT / name) for name in V2_FILES}}
    (HERE / "archive_manifest_v3_1.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    outreach = HERE / "outreach"
    for candidate in ["CAND-001", "CAND-002", "CAND-003", "CAND-004"]:
        attachments = [
            f"outreach/{candidate}_eligibility.md",
            f"outreach/{candidate}_protocol_summary.md",
            "preregistration_v3.md", "protocol_lock_v3.json",
            "public_input_schema_v3.json",
            "preregistration_v3_1_amendment_20260814.md",
            V3_1_ARCHIVE.name]
        lines = ["SHA256  FILE"]
        for relative in attachments:
            path = HERE / relative
            lines.append(f"{sha256(path)}  {Path(relative).name}")
        lines.extend(["", GENERATED_NOTE, "Status: NOT SENT."])
        (outreach / f"{candidate}_attachment_hashes.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(V3_1_ARCHIVE), "sha256": sha256(V3_1_ARCHIVE),
                      "files": len(V3_1_FILES) + len(V3_1_EXTRA) + len(V3_1_VENDORED)},
                     indent=2))


if __name__ == "__main__":
    main()
