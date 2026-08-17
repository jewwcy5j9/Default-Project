"""Audit system-scale-normalized errors for the primary K=3 panels.

The normalization is descriptive and fixed by the prediction target:

    normalized MAE = model MAE / constant-WT MAE

The denominator is the mean absolute mutant shift from the system WT over all
mutation-state coordinates.  It therefore equals the raw MAE of predicting the
WT population for every mutation.  A value of 1.0 means no improvement over
constant WT; lower is better.  This audit does not select a model or alter any
primary result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "iclr_restructuring" / "results"
NESTED_PATH = RESULTS / "p2_k3_nested_pca_results.json"
ABL1_PATH = ROOT / "data" / "nmr_populations" / "xie2020_abl1_FINAL.json"
SRC_PATH = ROOT / "data" / "nmr_populations" / "cui2025_src_kinase.json"
OUT_JSON = RESULTS / "p3_dynamic_range_normalization.json"
OUT_REPORT = RESULTS / "p3_dynamic_range_normalization_report.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def system_targets() -> dict[str, dict[str, Any]]:
    abl1_source = json.loads(ABL1_PATH.read_text(encoding="utf-8"))
    src_source = json.loads(SRC_PATH.read_text(encoding="utf-8"))
    abl1_names = [
        "M290L",
        "L301I",
        "M290L_L301I",
        "F382L",
        "F382Y",
        "F382V",
    ]
    src_names = [
        "SrcKD-L410A",
        "SrcKD-V332I",
        "SrcKD-L270F_V332I",
        "SrcKD-L325A",
        "SrcKD-A311I",
        "SrcKD-V380A",
        "SrcKD-V331A",
        "SrcKD-F405A",
    ]
    abl1_populations = abl1_source["populations"]
    src_rows = {
        row["name"]: row
        for row in src_source["additional_mutants_from_figS5"]["data"]
    }
    return {
        "abl1": {
            "names": abl1_names,
            "targets": np.asarray(
                [
                    [
                        abl1_populations[name]["Active"],
                        abl1_populations[name]["I1"],
                        abl1_populations[name]["I2"],
                    ]
                    for name in abl1_names
                ]
            ),
            "wt": np.asarray(
                [
                    abl1_populations["WT"]["Active"],
                    abl1_populations["WT"]["I1"],
                    abl1_populations["WT"]["I2"],
                ]
            ),
        },
        "src": {
            "names": src_names,
            "targets": np.asarray(
                [
                    [
                        src_rows[name]["A"] / 100.0,
                        src_rows[name]["E1"] / 100.0,
                        src_rows[name]["E2"] / 100.0,
                    ]
                    for name in src_names
                ]
            ),
            "wt": np.asarray(
                [
                    src_rows["SrcKD"]["A"] / 100.0,
                    src_rows["SrcKD"]["E1"] / 100.0,
                    src_rows["SrcKD"]["E2"] / 100.0,
                ]
            ),
        },
    }


def main() -> int:
    nested = json.loads(NESTED_PATH.read_text(encoding="utf-8"))
    artifact: dict[str, Any] = {
        "schema_version": "p3-dynamic-range-normalization-v1",
        "definition": {
            "normalized_mae": "raw_mae / constant_wt_mae",
            "constant_wt_mae": (
                "mean absolute mutant-minus-WT population shift over all "
                "mutation-state coordinates"
            ),
            "interpretation": (
                "1.0 equals constant-WT error; lower is better; descriptive "
                "only and not a model-selection criterion"
            ),
        },
        "source_hashes": {
            "nested_artifact": sha256(NESTED_PATH),
            "abl1_population_source": sha256(ABL1_PATH),
            "src_population_source": sha256(SRC_PATH),
            "script": sha256(Path(__file__)),
        },
        "systems": {},
    }

    for system, source in system_targets().items():
        targets = source["targets"]
        wt = source["wt"]
        constant_wt_mae = float(np.mean(np.abs(targets - wt)))
        state_ranges = np.ptp(targets, axis=0)
        record = nested["systems"][system]
        raw = {
            "constant_wt": constant_wt_mae,
            "loo_training_mean": float(record["training_mean"]["mae"]),
            "fixed_llr_plus_position_fold_local": float(
                record["fixed_mlp_raw"]["llr_pos"]["mae"]
            ),
            "fixed_pca_gpu_canonical": float(
                record["verification"]["t7_pca20_fixed"]["frozen_gpu_canonical"]
            ),
            "full_nested_mlp": float(record["nested_mlp"]["mae"]),
        }
        normalized = {key: value / constant_wt_mae for key, value in raw.items()}
        artifact["systems"][system] = {
            "n_mutations": int(len(source["names"])),
            "mutation_ids": source["names"],
            "state_order": ["Active", "I1", "I2"]
            if system == "abl1"
            else ["Active", "E1", "E2"],
            "state_ranges": [float(value) for value in state_ranges],
            "raw_mae": raw,
            "normalized_mae": normalized,
        }

    # Frozen benchmark checks.  These prevent a target-panel or denominator
    # change from silently altering the manuscript table.
    assert np.isclose(
        artifact["systems"]["abl1"]["raw_mae"]["constant_wt"],
        0.3877777777777778,
    )
    assert np.isclose(
        artifact["systems"]["src"]["raw_mae"]["constant_wt"], 0.46
    )
    assert np.allclose(
        artifact["systems"]["abl1"]["state_ranges"], [0.83, 0.10, 0.89]
    )
    assert np.allclose(
        artifact["systems"]["src"]["state_ranges"], [0.73, 0.84, 0.84]
    )

    OUT_JSON.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    rows = []
    for system in ("abl1", "src"):
        result = artifact["systems"][system]
        raw = result["raw_mae"]
        norm = result["normalized_mae"]
        rows.append(
            "| {system} | {scale:.4f} | {ranges} | {mean:.4f} | {llr:.4f} | "
            "{pca:.4f} | {nested:.4f} |".format(
                system="Abl1" if system == "abl1" else "Src",
                scale=raw["constant_wt"],
                ranges="/".join(f"{value:.2f}" for value in result["state_ranges"]),
                mean=norm["loo_training_mean"],
                llr=norm["fixed_llr_plus_position_fold_local"],
                pca=norm["fixed_pca_gpu_canonical"],
                nested=norm["full_nested_mlp"],
            )
        )
    OUT_REPORT.write_text(
        """# Dynamic-range-normalized K=3 errors

Normalized MAE is raw MAE divided by the constant-WT MAE for the same system.
The denominator is the mean absolute mutant-minus-WT population shift over all
mutation-state coordinates.  Thus 1.0 equals constant-WT prediction and lower
is better.  These values are descriptive and are not used for model selection.

| System | Constant-WT MAE | State ranges | Train mean | LLR+pos fold-local | PCA | Full nested |
|---|---:|---:|---:|---:|---:|---:|
"""
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["systems"], indent=2, ensure_ascii=False))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
