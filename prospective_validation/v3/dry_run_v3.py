#!/usr/bin/env python
"""Build an Abl1 fixture from frozen caches and run v3 end to end.

Every generated artifact is labelled ``DRY RUN / NOT EVIDENCE``.  This script
does not modify prospective v2 files or archives.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "experiments" / "iclr_restructuring"
# This is a repository-tree rehearsal tool: it drives the frozen custodian
# entrypoint against the full experiment tree (ESM-2 encoding pipeline and
# K=3 panels). It is shipped in the custodian package for protocol
# completeness but cannot run from a clean package extraction.
if not (EXP / "esm2_encoding.py").exists():
    raise SystemExit(
        "dry_run_v3.py is a repository-tree rehearsal tool: run it from the "
        "full repository checkout, not from the extracted custodian package "
        "(the sealed custodian flow only needs run_custodian_v3.py)."
    )
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT))

import esm2_encoding as esm  # noqa: E402
from k3_data import ABL1_K3, ABL1_K3_WT_POP  # noqa: E402
from run_custodian_v3 import run  # noqa: E402


def main():
    out = HERE / "dry_run_outputs"
    out.mkdir(parents=True, exist_ok=True)
    ids = ["M290L", "L301I", "M290L_L301I", "F382L", "F382Y", "F382V"]
    substitutions = {
        "M290L": [(290, "M", "L")], "L301I": [(301, "L", "I")],
        "M290L_L301I": [(290, "M", "L"), (301, "L", "I")],
        "F382L": [(382, "F", "L")], "F382Y": [(382, "F", "Y")],
        "F382V": [(382, "F", "V")],
    }
    # Abl1 numbering in the source is Abl1a; the cached sequence begins at 229.
    # Resolve the already-audited nominal positions to explicit sequence indices.
    mutations = []
    for mid in ids:
        entries = []
        for nominal, aa_from, aa_to in substitutions[mid]:
            index, *_ = esm.find_position(esm.ABL1_KD, nominal, aa_from, system="abl1")
            entries.append({"reported_position": nominal,
                            "sequence_index_1based": index + 1,
                            "from": aa_from, "to": aa_to})
        mutations.append({"mutation_id": mid, "substitutions": entries})
    public = {
        "protocol_version": "3.0.0", "panel_id": "ABL1_V3_DRY_RUN_NOT_EVIDENCE",
        "system_name": "Abl1 historical development panel",
        "wild_type_sequence": esm.ABL1_KD,
        "wt_population": ABL1_K3_WT_POP,
        "state_definitions": ["Active", "I1", "I2"],
        "conditions": {"construct": "historical development fixture",
                       "ligand": "historical development fixture",
                       "temperature": "25 C", "buffer": "historical development fixture",
                       "state_model": "K=3 historical fixture"},
        "uncertainty_metadata": {"note": "not used by primary predictor"},
        "mutations": mutations,
        "warning": "DRY RUN / NOT EVIDENCE",
    }
    private = {"panel_id": public["panel_id"],
               "mutant_populations": {mid: ABL1_K3[mid]["pop"] for mid in ids},
               "warning": "DRY RUN / NOT EVIDENCE"}
    public_path, private_path = out / "public_input_abl1_dry_run.json", out / "private_targets_abl1_dry_run.json"
    public_path.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    private_path.write_text(json.dumps(private, indent=2) + "\n", encoding="utf-8")

    cache = np.load(EXP / "results" / "p2_k3_nested_pca_deltas_abl1.npz")
    combined = cache["delta_rows::abl1"]
    seq_len = len(esm.ABL1_KD)
    arrays = {}
    for index, mid in enumerate(ids):
        arrays[f"delta_vector::{mid}"] = cache[f"delta_vec::abl1::{mid}"]
        arrays[f"delta_rows::{mid}"] = combined[index * seq_len:(index + 1) * seq_len]
    feature_path = out / "esm2_features_abl1_dry_run.npz"
    np.savez_compressed(feature_path, **arrays)
    result = run(public_path, private_path, feature_path, out, dry_run=True)
    assert result["status"] == "DRY RUN / NOT EVIDENCE"
    assert all(fold["pca"]["fit_ids"] == fold["outer_train_ids"] for fold in result["folds"])
    assert all(abs(sum(fold["prediction"]) - 1.0) < 1e-7 for fold in result["folds"])
    print(json.dumps({"status": result["status"],
                      "primary_mae": result["aggregate"]["primary_mae"],
                      "panel_tier": result["panel_tier"]}, indent=2))


if __name__ == "__main__":
    main()
