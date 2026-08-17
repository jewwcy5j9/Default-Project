#!/usr/bin/env python
"""Calibrate AF2 reference assignment across alignment regions and cutoffs.

This analysis is additive: it recomputes assignment diagnostics from existing
PDBs without changing the preregistered 3.0 A classifications.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

sys.path.insert(0, str(Path(__file__).resolve().parent))

import classify_states as CS
from t9_reclassify_verify import compute_region_offsets, region_rmsd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
DEFAULT_JSON = HERE / "results" / "assignment_calibration.json"
DEFAULT_MARKDOWN = HERE / "results" / "assignment_calibration.md"
THRESHOLDS = tuple(round(2.0 + 0.25 * i, 2) for i in range(13))
AMBIGUOUS_MARGIN = 0.5
STATE_LABELS = {"active": "active", "i1": "I1", "i2": "I2"}
COUNT_LABELS = ("active", "I1", "I2", "ambiguous", "unclassified")
REGIONS = {
    "full_protein": None,
    "n_lobe_act": (235, 400),
    "alphaC_only": (260, 300),
}
PROTOCOLS = {
    "original": {
        "label": "original MSA-subsample ensemble",
        "structure_dir": HERE / "output",
        "generation_manifest": HERE / "output" / "manifest.json",
        "stored_classifications": HERE / "results" / "state_classifications.json",
    },
    "fresh_msa": {
        "label": "independent/fresh-MSA ensemble",
        "structure_dir": HERE / "output_independent_msa" / "output",
        "generation_manifest": None,
        "stored_classifications": (
            HERE / "output_independent_msa" / "results" / "state_classifications.json"
        ),
    },
}


def display_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collection_sha256(paths):
    """Hash sorted relative paths and file hashes into one collection digest."""
    digest = hashlib.sha256()
    for path in sorted((Path(p).resolve() for p in paths), key=lambda p: display_path(p)):
        digest.update(display_path(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def assign_with_margin(rmsds, threshold, ambiguous_margin=AMBIGUOUS_MARGIN):
    """Apply the existing strict cutoff/argmin rule, then flag low margins.

    The margin is second-nearest RMSD minus nearest RMSD. A structure is
    ambiguous only if it passes the assignment cutoff and margin < the stated
    ambiguity threshold. Cutoff failures remain unclassified.
    """
    ranked = sorted(rmsds.items(), key=lambda item: (item[1], item[0]))
    nearest_state, nearest = ranked[0]
    second_state, second = ranked[1]
    margin = second - nearest
    if nearest >= threshold:
        state = "unclassified"
    elif margin < ambiguous_margin:
        state = "ambiguous"
    else:
        state = STATE_LABELS[nearest_state]
    return {
        "state": state,
        "nearest_state": STATE_LABELS[nearest_state],
        "second_nearest_state": STATE_LABELS[second_state],
        "nearest_rmsd": float(nearest),
        "second_nearest_rmsd": float(second),
        "margin": float(margin),
    }


def distribution(values):
    values = np.asarray(values, dtype=float)
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "sorted_values": [float(value) for value in np.sort(values)],
    }


def load_references(parser):
    references = {}
    paths = {
        "active": CS.resolve_path(CS.DEFAULT_REF_ACTIVE),
        "i1": CS.resolve_path(CS.DEFAULT_REF_I1),
        "i2": CS.resolve_path(CS.DEFAULT_REF_I2),
    }
    for state, path in paths.items():
        ordered, by_resseq, n_atoms = CS.load_reference(parser, str(path), state)
        references[state] = (ordered, by_resseq)
        paths[state] = Path(path).resolve()
    return references, paths


def calculate_rmsds(pred_ordered, pred_by, references, region,
                    region_offsets=None):
    """RMSDs against every reference.

    ``region`` bounds are stated in the 2HYY/I1 numbering frame;
    ``region_offsets`` translates them into each target reference's own
    numbering (see compute_region_offsets)."""
    region_offsets = region_offsets or {}
    rmsds = {}
    atom_counts = {}
    for state, (ref_ordered, ref_by) in references.items():
        if region is None:
            rmsd, n_atoms = CS.compute_ca_rmsd(
                pred_ordered, pred_by, ref_ordered, ref_by
            )
        else:
            rmsd, n_atoms = region_rmsd(
                pred_ordered, pred_by, ref_ordered, ref_by, region,
                region_offsets.get(state, 0),
            )
        if rmsd is None:
            raise ValueError(f"No valid RMSD for {state} in region {region}")
        rmsds[state] = float(rmsd)
        atom_counts[state] = int(n_atoms)
    return rmsds, atom_counts


def reference_calibration(references, region_offsets):
    output = {}
    pairs = (("active", "i1"), ("active", "i2"), ("i1", "i2"))
    for region_name, region in REGIONS.items():
        distances = {}
        for state_a, state_b in pairs:
            rmsd, atom_counts = calculate_rmsds(
                references[state_a][0], references[state_a][1],
                {state_b: references[state_b]}, region, region_offsets,
            )
            distances[f"{STATE_LABELS[state_a]}__{STATE_LABELS[state_b]}"] = {
                "rmsd_angstrom": rmsd[state_b],
                "n_ca_atoms": atom_counts[state_b],
            }
        output[region_name] = {
            "residue_range": list(region) if region else None,
            "pairwise_distances": distances,
        }
    return output


def threshold_curve(records):
    curve = []
    for threshold in THRESHOLDS:
        counts = Counter()
        hard_counts = Counter()
        for record in records:
            assignment = assign_with_margin(record["rmsds"], threshold)
            counts[assignment["state"]] += 1
            hard_state = CS.classify_state(
                record["rmsds"]["active"], record["rmsds"]["i1"],
                record["rmsds"]["i2"], threshold,
            )
            hard_counts[hard_state] += 1
        n = len(records)
        curve.append({
            "threshold_angstrom": threshold,
            "hard_assignment_counts": {
                state: int(hard_counts.get(state, 0))
                for state in ("active", "I1", "I2", "unclassified")
            },
            "hard_assignment_fractions": {
                state: float(hard_counts.get(state, 0) / n)
                for state in ("active", "I1", "I2", "unclassified")
            },
            "margin_aware_counts": {
                state: int(counts.get(state, 0)) for state in COUNT_LABELS
            },
            "margin_aware_fractions": {
                state: float(counts.get(state, 0) / n) for state in COUNT_LABELS
            },
        })
    return curve


def compare_frozen(records, stored_path):
    stored = json.loads(Path(stored_path).read_text(encoding="utf-8"))
    key = lambda row: (row["mutant"], row["run"], row["model"], row["seed"])
    expected = {key(row): row["state"] for row in stored["classifications"]}
    mismatches = []
    for record in records:
        rmsds = record["rmsds"]
        observed = CS.classify_state(
            rmsds["active"], rmsds["i1"], rmsds["i2"], CS.RMSD_THRESHOLD
        )
        if expected.get(record["key"]) != observed:
            mismatches.append({
                "key": list(record["key"]),
                "stored": expected.get(record["key"]),
                "recomputed": observed,
            })
    return {
        "stored_records": len(expected),
        "recomputed_records": len(records),
        "matching_records": len(records) - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
    }


def analyze_protocol(parser, protocol, references, region_offsets):
    predictions = CS.find_predictions(str(protocol["structure_dir"]))
    if not predictions:
        raise FileNotFoundError(f"No predictions found in {protocol['structure_dir']}")
    by_region = {name: [] for name in REGIONS}
    source_paths = []
    for pdb_path, mutant, run, model, seed in predictions:
        path = Path(pdb_path).resolve()
        source_paths.append(path)
        structure = parser.get_structure("prediction", str(path))
        pred_ordered, pred_by = CS.get_ca_atoms(structure, chain_id="A", model_id=0)
        if not pred_ordered:
            pred_ordered, pred_by = CS.get_ca_atoms(structure)
        for region_name, region in REGIONS.items():
            rmsds, atom_counts = calculate_rmsds(
                pred_ordered, pred_by, references, region, region_offsets
            )
            by_region[region_name].append({
                "key": (mutant, run, model, seed),
                "rmsds": rmsds,
                "atom_counts": atom_counts,
            })

    region_output = {}
    for region_name, records in by_region.items():
        assignments = [assign_with_margin(record["rmsds"], CS.RMSD_THRESHOLD)
                       for record in records]
        margins = [assignment["margin"] for assignment in assignments]
        nearest_counts = Counter(a["nearest_state"] for a in assignments)
        ambiguous_counts = Counter(a["state"] for a in assignments)
        atom_counts = [count for record in records
                       for count in record["atom_counts"].values()]
        region_output[region_name] = {
            "residue_range": list(REGIONS[region_name]) if REGIONS[region_name] else None,
            "n_structures": len(records),
            "n_ca_atoms_min": min(atom_counts),
            "n_ca_atoms_max": max(atom_counts),
            "threshold_curve": threshold_curve(records),
            "margin_distribution_angstrom": distribution(margins),
            "nearest_reference_counts": dict(nearest_counts),
            "ambiguity_at_frozen_threshold": {
                "threshold_angstrom": CS.RMSD_THRESHOLD,
                "margin_cutoff_angstrom": AMBIGUOUS_MARGIN,
                "counts": {state: int(ambiguous_counts.get(state, 0))
                           for state in COUNT_LABELS},
            },
        }

    full_records = by_region["full_protein"]
    stored_path = Path(protocol["stored_classifications"]).resolve()
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    manifest_path = protocol.get("generation_manifest")
    manifest = None
    if manifest_path is not None:
        manifest_path = Path(manifest_path).resolve()
        if manifest_path.exists():
            manifest = {
                "path": display_path(manifest_path),
                "sha256": sha256_file(manifest_path),
            }
    return {
        "label": protocol["label"],
        "n_structures": len(predictions),
        "source": {
            "structure_directory": display_path(protocol["structure_dir"]),
            "structure_collection_sha256": collection_sha256(source_paths),
            "collection_hash_definition": (
                "SHA-256 over sorted project-relative path, NUL, per-file SHA-256, newline"
            ),
            "stored_classifications_path": display_path(stored_path),
            "stored_classifications_sha256": sha256_file(stored_path),
            "stored_protocol": stored.get("protocol"),
            "stored_rmsd_threshold_angstrom": stored.get("rmsd_threshold"),
            "stored_rmsd_method": stored.get("rmsd_method"),
            "stored_classification_rule": stored.get("classification_rule"),
            "generation_manifest": manifest,
            "generation_manifest_status": (
                "hashed" if manifest else "not available for this protocol"
            ),
        },
        "frozen_3A_full_protein_consistency": compare_frozen(full_records, stored_path),
        "regions": region_output,
    }


def render_markdown(result):
    lines = [
        "# AF2 Assignment Calibration",
        "",
        "> These results are protocol-dependent structure assignments, not estimates of "
        "thermodynamic populations. Structures generated with shared models, seeds, or MSAs "
        "are not treated as independent biological samples.",
        "",
        "The preregistered classifications remain unchanged. This report adds cutoff and "
        "alignment sensitivity diagnostics. `ambiguous` means the nearest reference passes "
        f"the cutoff but the RMSD margin (second-nearest minus nearest) is < "
        f"{result['ambiguity_rule']['margin_cutoff_angstrom']:.2f} Angstrom. Cutoff failures "
        "remain `unclassified`.",
        "",
        "## Reference-to-reference calibration",
        "",
        "| Alignment region | Active-I1 | Active-I2 | I1-I2 |",
        "|---|---:|---:|---:|",
    ]
    for region_name, region in result["reference_calibration"].items():
        pairs = region["pairwise_distances"]
        lines.append(
            f"| {region_name} | {pairs['active__I1']['rmsd_angstrom']:.3f} | "
            f"{pairs['active__I2']['rmsd_angstrom']:.3f} | "
            f"{pairs['I1__I2']['rmsd_angstrom']:.3f} |"
        )

    lines.extend(["", "## Assignment calibration at 3.0 Angstrom", ""])
    for protocol_name, protocol in result["protocols"].items():
        lines.extend([
            f"### {protocol_name}",
            "",
            "| Alignment region | active | I1 | I2 | ambiguous | unclassified | "
            "margin median | margin q05-q95 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for region_name, region in protocol["regions"].items():
            counts = region["ambiguity_at_frozen_threshold"]["counts"]
            margin = region["margin_distribution_angstrom"]
            lines.append(
                f"| {region_name} | {counts['active']} | {counts['I1']} | "
                f"{counts['I2']} | {counts['ambiguous']} | {counts['unclassified']} | "
                f"{margin['median']:.3f} | {margin['q05']:.3f}-{margin['q95']:.3f} |"
            )
        consistency = protocol["frozen_3A_full_protein_consistency"]
        lines.extend([
            "",
            f"Frozen full-protein result check: {consistency['matching_records']}/"
            f"{consistency['recomputed_records']} records match the stored assignments.",
            "",
        ])

    lines.extend([
        "## Threshold curves",
        "",
        "Counts below use the original strict cutoff plus argmin convention. The separate "
        "ambiguity-aware counts are available in the JSON output for every 0.25 Angstrom "
        "increment from 2.0 through 5.0 Angstrom.",
        "",
    ])
    for protocol_name, protocol in result["protocols"].items():
        for region_name, region in protocol["regions"].items():
            lines.extend([
                f"### {protocol_name}: {region_name}",
                "",
                "| Cutoff | active | I1 | I2 | unclassified |",
                "|---:|---:|---:|---:|---:|",
            ])
            for point in region["threshold_curve"]:
                counts = point["hard_assignment_counts"]
                lines.append(
                    f"| {point['threshold_angstrom']:.2f} | {counts['active']} | "
                    f"{counts['I1']} | {counts['I2']} | {counts['unclassified']} |"
                )
            lines.append("")

    lines.extend([
        "## Provenance",
        "",
        f"RMSD implementation: `{result['method']['implementation_path']}` "
        f"(`{result['method']['implementation_sha256']}`).",
        "",
    ])
    for state, ref in result["references"].items():
        lines.append(f"- {state}: `{ref['path']}` (`{ref['sha256']}`)")
    for protocol_name, protocol in result["protocols"].items():
        source = protocol["source"]
        lines.append(
            f"- {protocol_name} structures: `{source['structure_directory']}` "
            f"(collection SHA-256 `{source['structure_collection_sha256']}`)"
        )
    lines.extend([
        "",
        "## Limitations",
        "",
        "The 0.5 Angstrom ambiguity cutoff is an explicit diagnostic convention, not a "
        "validated physical boundary. Reference perturbation and leave-one-reference "
        "stability are not evaluated here. Results depend on the three selected references, "
        "the first model/chain-A parsing convention, residue correspondence, and the stated "
        "Kabsch alignment regions.",
        "",
    ])
    return "\n".join(lines)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    started = time.time()
    pdb_parser = PDBParser(QUIET=True)
    references, reference_paths = load_references(pdb_parser)
    region_offsets = compute_region_offsets(references)
    result = {
        "schema_version": "af2_assignment_calibration_v1",
        "purpose": "assignment sensitivity calibration; does not replace frozen classifications",
        "method": {
            "rmsd": "C-alpha RMSD after Kabsch alignment via Bio.PDB.Superimposer",
            "classification": "argmin among references with RMSD strictly below cutoff",
            "implementation_path": display_path(Path(CS.__file__)),
            "implementation_sha256": sha256_file(CS.__file__),
            "alignment_helper_path": display_path(HERE / "t9_reclassify_verify.py"),
            "alignment_helper_sha256": sha256_file(HERE / "t9_reclassify_verify.py"),
            "region_frame": {
                "frame": "2HYY/I1 numbering (canonical Abl1 residue numbering)",
                "reference_region_offsets": region_offsets,
            },
            "thresholds_angstrom": list(THRESHOLDS),
        },
        "ambiguity_rule": {
            "definition": (
                "ambiguous if nearest RMSD < assignment cutoff and "
                "second-nearest RMSD - nearest RMSD < margin cutoff"
            ),
            "margin_cutoff_angstrom": AMBIGUOUS_MARGIN,
            "cutoff_failure_label": "unclassified",
            "interpretation": "diagnostic assignment uncertainty; not a thermodynamic population",
        },
        "references": {
            STATE_LABELS[state]: {
                "path": display_path(path), "sha256": sha256_file(path)
            }
            for state, path in reference_paths.items()
        },
        "reference_calibration": reference_calibration(references, region_offsets),
        "protocols": {},
    }
    for protocol_name, protocol in PROTOCOLS.items():
        print(f"Analyzing {protocol_name}: {protocol['structure_dir']}")
        result["protocols"][protocol_name] = analyze_protocol(
            pdb_parser, protocol, references, region_offsets
        )
    result["runtime_seconds"] = float(time.time() - started)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(result), encoding="utf-8")
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.markdown_output}")


if __name__ == "__main__":
    run()
