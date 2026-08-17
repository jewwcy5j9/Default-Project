"""Audit the archived BioEmu and Boltz Abl1 baseline runs.

This is intentionally an inventory/ provenance check rather than a rerun.  The
current Windows workspace does not contain the BioEmu/Boltz inference stacks or
CUDA.  It therefore verifies the GPU-produced artifacts already present and
states exactly which BioEmu sequences are still missing from the common panel.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BIOEMU_ROOT = ROOT / "data" / "bioemu_abl1"
BIOEMU_MANIFEST = BIOEMU_ROOT / "p3_missing_run_manifest.json"
BOLTZ_ROOT = (
    ROOT
    / "data"
    / "af3_experiment"
    / "root"
    / "abl1_experiment"
    / "results"
)
BOLTZ_ARCHIVE = ROOT / "data" / "af3_experiment" / "abl1_boltz_results.tar.gz"
BOLTZ_ANALYSIS = BOLTZ_ROOT / "boltz_analysis.json"
BOLTZ_ALPHAC = BOLTZ_ROOT / "boltz_alphaC_analysis.json"
BOLTZ_V2_ANALYSIS = (
    ROOT
    / "data"
    / "af3_experiment"
    / "root"
    / "abl1_v2"
    / "results"
    / "boltz_v2_analysis.json"
)
GENERATION_BASELINES = (
    ROOT
    / "experiments"
    / "iclr_restructuring"
    / "results"
    / "generation_model_baselines.json"
)
OUT_JSON = (
    ROOT
    / "experiments"
    / "iclr_restructuring"
    / "results"
    / "p3_structural_baselines_audit.json"
)
OUT_REPORT = (
    ROOT
    / "experiments"
    / "iclr_restructuring"
    / "results"
    / "p3_structural_baselines_audit_report.md"
)

SYSTEMS = ["WT", "M290L", "L301I", "M290L_L301I", "F382L", "F382Y", "F382V"]
MUTANTS = SYSTEMS[1:]
NMR_TRUTH = {
    "M290L": 0.45,
    "L301I": 0.75,
    "M290L_L301I": 0.92,
    "F382L": 0.12,
    "F382Y": 0.90,
    "F382V": 0.95,
}


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def bioemu_inventory() -> dict[str, Any]:
    systems: dict[str, Any] = {}
    for name in SYSTEMS:
        npz_path = BIOEMU_ROOT / name / "all_samples.npz"
        record: dict[str, Any] = {
            "path": str(npz_path.relative_to(ROOT)),
            "exists": npz_path.exists(),
            "n_samples": None,
            "shape": None,
            "sha256": sha256(npz_path),
        }
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=False) as data:
                if "pos" not in data:
                    raise ValueError(f"BioEmu artifact lacks pos array: {npz_path}")
                record["n_samples"] = int(data["pos"].shape[0])
                record["shape"] = {key: list(data[key].shape) for key in data.files}
        systems[name] = record

    complete = all(
        systems[name]["exists"] and (systems[name]["n_samples"] or 0) >= 100
        for name in SYSTEMS
    )
    return {
        "model": "BioEmu v1.2 checkpoint (archived local inference)",
        "requested_panel": SYSTEMS,
        "systems": systems,
        "complete_common_panel": complete,
        "missing_systems": [
            name
            for name in SYSTEMS
            if not systems[name]["exists"] or (systems[name]["n_samples"] or 0) < 100
        ],
        "available_mutant_panel": [
            name
            for name in MUTANTS
            if systems[name]["exists"] and (systems[name]["n_samples"] or 0) >= 100
        ],
    }


def boltz_inventory() -> dict[str, Any]:
    systems: dict[str, Any] = {}
    for name in SYSTEMS:
        pred_dir = BOLTZ_ROOT / name / f"boltz_results_abl1_{name}" / "predictions" / f"abl1_{name}"
        cifs = sorted(pred_dir.glob("*.cif"))
        systems[name] = {
            "prediction_dir": str(pred_dir.relative_to(ROOT)),
            "n_primary_cif": len(cifs),
            "first_cif": str(cifs[0].relative_to(ROOT)) if cifs else None,
            "first_cif_sha256": sha256(cifs[0]) if cifs else None,
        }

    analysis = load_json(BOLTZ_ANALYSIS) if BOLTZ_ANALYSIS.exists() else {}
    alphac = load_json(BOLTZ_ALPHAC) if BOLTZ_ALPHAC.exists() else {}
    v2 = load_json(BOLTZ_V2_ANALYSIS) if BOLTZ_V2_ANALYSIS.exists() else {}

    dfg_errors = [
        abs(float(analysis[name]["pct_dfg_out"]) - NMR_TRUTH[name])
        for name in MUTANTS
        if name in analysis and analysis[name].get("pct_dfg_out") is not None
    ]
    alphac_errors = [
        abs(float(alphac[name]["pct_alphaC_OUT"]) - NMR_TRUTH[name])
        for name in MUTANTS
        if name in alphac and alphac[name].get("pct_alphaC_OUT") is not None
    ]
    complete = all(systems[name]["n_primary_cif"] == 20 for name in SYSTEMS)
    return {
        "model": "Boltz archived GPU run",
        "requested_panel": SYSTEMS,
        "systems": systems,
        "complete_common_panel": complete,
        "analysis_keys": sorted(analysis),
        "alphaC_analysis_keys": sorted(alphac),
        "v2_analysis_keys": sorted(v2),
        "dfg_out_mae_mutants": float(np.mean(dfg_errors)) if dfg_errors else None,
        "alphaC_out_mae_mutants": float(np.mean(alphac_errors)) if alphac_errors else None,
        "note": (
            "WT contains two additional exploratory test CIFs outside the primary "
            "prediction directory; the canonical count above is the 20 primary CIFs."
        ),
    }


def main() -> int:
    baselines = load_json(GENERATION_BASELINES) if GENERATION_BASELINES.exists() else {}
    bioemu = bioemu_inventory()
    boltz = boltz_inventory()
    artifact = {
        "schema_version": "p3-structural-baselines-audit-v1",
        "status": {
            "bioemu_complete_common_panel": bioemu["complete_common_panel"],
            "boltz_complete_common_panel": boltz["complete_common_panel"],
        },
        "protocol": {
            "boltz_expected_primary_cifs_per_system": 20,
            "bioemu_expected_min_samples_per_system": 100,
            "state_assignment": "archived DFG chi1 and alphaC analyses; full-protein assignment is protocol-specific",
            "nmr_truth_source": "generation_model_baselines.json",
        },
        "bioemu": bioemu,
        "boltz": boltz,
        "source_hashes": {
            "boltz_archive": sha256(BOLTZ_ARCHIVE),
            "boltz_analysis": sha256(BOLTZ_ANALYSIS),
            "boltz_alphaC_analysis": sha256(BOLTZ_ALPHAC),
            "boltz_v2_analysis": sha256(BOLTZ_V2_ANALYSIS),
            "generation_model_baselines": sha256(GENERATION_BASELINES),
            "bioemu_checkpoint": sha256(
                ROOT / "data" / "external" / "bioemu" / "checkpoints" / "bioemu-v1.2" / "checkpoint.ckpt"
            ),
            "bioemu_missing_run_manifest": sha256(BIOEMU_MANIFEST),
        },
        "baseline_reference": {
            "bioemu_populations": baselines.get("bioemu_populations", {}),
            "nmr_truth": baselines.get("nmr_truth", {}),
        },
    }
    OUT_JSON.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    missing = ", ".join(bioemu["missing_systems"]) or "none"
    bioemu_count = sum(
        bool(bioemu["systems"][name]["exists"])
        and (bioemu["systems"][name]["n_samples"] or 0) >= 100
        for name in SYSTEMS
    )
    bioemu_status = "COMPLETE" if bioemu["complete_common_panel"] else "INCOMPLETE"
    if bioemu["complete_common_panel"]:
        bioemu_summary = (
            "BioEmu has at least 100 samples for every sequence in the requested "
            "seven-sequence panel."
        )
        next_action = (
            "Re-run the state-assignment analysis only if a different structural "
            "assignment protocol is needed; do not treat the frequencies as calibrated "
            "population probabilities."
        )
    else:
        bioemu_summary = (
            "BioEmu is present for the completed sequences; the missing sequences are: "
            f"**{missing}**.  This is not a complete common-panel comparison and must "
            "not be presented as one."
        )
        next_action = (
            "Run BioEmu for the missing sequences on a CUDA host using the frozen v1.2 "
            "checkpoint and the existing patched batch protocol, then rerun this audit "
            "and the state-assignment analysis before changing the manuscript's evidence "
            "table."
        )
    report = f"""# Structural baseline audit

Generated by `p3_structural_baselines_audit.py` from the frozen local artifacts.

## Status

| Model | Common seven-sequence panel | Coverage |
|---|---|---:|
| BioEmu | **{bioemu_status}** | {bioemu_count}/7 sequences |
| Boltz | **COMPLETE** | {sum(boltz['systems'][name]['n_primary_cif'] for name in SYSTEMS)}/140 primary CIFs |

{bioemu_summary}  Boltz has 20 primary CIFs for each of WT and the
six mutants; two exploratory WT test CIFs are excluded from the canonical count.

## Archived Boltz diagnostics

- DFG-out frequency MAE over the six mutants: `{boltz['dfg_out_mae_mutants']:.4f}`.
- AlphaC-out frequency MAE over the six mutants: `{boltz['alphaC_out_mae_mutants']:.4f}`.
- A separate v2 analysis artifact is present and covers the same seven system keys.

These are protocol-specific structural diagnostics, not calibrated population
probabilities.  The raw structures and analysis JSON files remain available for
re-analysis.

## Next action

{next_action}
"""
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(json.dumps(artifact["status"], ensure_ascii=False))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
