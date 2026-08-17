#!/usr/bin/env python
"""Aggregate original and fresh-MSA AF2 pLDDT directly from PDB files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import fmean


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "af2_plddt_raw_pdb.json"
PROTOCOLS = {
    "original": {
        "root": HERE / "output",
        "mutants": ("WT", "M290L", "L301I", "M290L_L301I", "F382L", "F382Y", "F382V"),
    },
    "fresh_msa": {
        "root": HERE / "output_independent_msa" / "output",
        "mutants": ("WT", "L301I", "M290L_L301I", "F382V"),
    },
}
PDB_PATTERN = re.compile(
    r"(?P<mutant>[^/]+)/run_(?P<run>\d+)/model_(?P<model>\d+)_seed_(?P<seed>\d+)\.pdb"
)


def mean_ca_plddt(path: Path) -> tuple[float, int]:
    values = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    values.append(float(line[60:66]))
                except ValueError as exc:
                    raise ValueError(f"Malformed CA B-factor at {path}:{line_number}") from exc
    if not values:
        raise ValueError(f"No CA pLDDT values found in {path}")
    return fmean(values), len(values)


def scan_protocol(root: Path, mutants: tuple[str, ...]) -> dict:
    rows = []
    seen = set()
    for path in sorted(root.glob("*/run_*/model_*_seed_*.pdb")):
        relative = path.relative_to(root).as_posix()
        match = PDB_PATTERN.fullmatch(relative)
        if match is None:
            raise ValueError(f"Unexpected PDB path: {path}")
        key = (match["mutant"], int(match["run"]), int(match["model"]), int(match["seed"]))
        if key in seen:
            raise ValueError(f"Duplicate AF2 grid member: {key}")
        seen.add(key)
        mean_plddt, n_ca = mean_ca_plddt(path)
        rows.append({
            "path": relative,
            "mutant": key[0],
            "run": key[1],
            "model": key[2],
            "seed": key[3],
            "n_ca": n_ca,
            "mean_plddt": mean_plddt,
        })

    expected = {(mutant, run, model, seed) for mutant in mutants
                for run in range(3) for model in range(1, 6) for seed in range(8)}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"AF2 grid mismatch: missing={missing[:5]}, extra={extra[:5]}")
    per_mutant = {}
    for mutant in mutants:
        values = [row["mean_plddt"] for row in rows if row["mutant"] == mutant]
        per_mutant[mutant] = {"n_structures": len(values), "mean_plddt": fmean(values)}
    structure_means = [row["mean_plddt"] for row in rows]
    mutant_means = [record["mean_plddt"] for record in per_mutant.values()]
    return {
        "root": str(root),
        "expected_structures": len(expected),
        "found_structures": len(rows),
        "parse_errors": 0,
        "n_ca_min": min(row["n_ca"] for row in rows),
        "n_ca_max": max(row["n_ca"] for row in rows),
        "overall_mean_plddt": fmean(structure_means),
        "structure_mean_range": [min(structure_means), max(structure_means)],
        "mutant_mean_range": [min(mutant_means), max(mutant_means)],
        "per_mutant": per_mutant,
        "structures": rows,
    }


def build_payload() -> dict:
    protocols = {
        name: scan_protocol(config["root"], config["mutants"])
        for name, config in PROTOCOLS.items()
    }
    shared = PROTOCOLS["fresh_msa"]["mutants"]
    original_shared = [protocols["original"]["per_mutant"][mutant]["mean_plddt"]
                       for mutant in shared]
    protocols["original"]["shared_four_sequence_mean_plddt"] = fmean(original_shared)
    return {
        "schema_version": "af2_plddt_raw_pdb_v1",
        "metric": {
            "record_filter": "ATOM records whose PDB atom-name field is CA",
            "value_field": "PDB B-factor columns 61-66",
            "per_structure": "arithmetic mean across CA values",
            "per_mutant": "unweighted arithmetic mean across structure means",
            "overall": "unweighted arithmetic mean across structure means",
        },
        "protocols": protocols,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for name, record in payload["protocols"].items():
        low, high = record["mutant_mean_range"]
        print(f"{name}: n={record['found_structures']} mutant mean range={low:.4f}-{high:.4f}")
    print(args.output)


if __name__ == "__main__":
    main()
