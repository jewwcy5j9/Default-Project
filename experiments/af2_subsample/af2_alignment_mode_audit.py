"""P1 AF2 alignment-mode audit (deep-review Q12).

Records, for every predicted structure in both protocols (840 original and
480 fresh-MSA), whether the full-protein C-alpha alignment used
sequence-offset matching or the first-N C-alpha fallback, plus the residue
offset and matched-residue count for each of the three reference states.

This is additive: it recomputes the full-protein RMSDs and the frozen 3.0 A
assignment to verify consistency (840/840 and 480/480), and it does not change
the preregistered classifications.

Outputs:
  results/af2_alignment_mode_audit.json
  results/af2_alignment_mode_audit.md
"""
import json
import sys
from collections import Counter
from pathlib import Path

from Bio.PDB import PDBParser

sys.path.insert(0, str(Path(__file__).resolve().parent))

import classify_states as CS

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT_JSON = RESULTS / "af2_alignment_mode_audit.json"
OUT_MD = RESULTS / "af2_alignment_mode_audit.md"


def alignment_mode(pred_ordered, pred_by, ref_ordered, ref_by):
    """Return (mode, offset, matched_residues) using the exact matching logic
    of classify_states.compute_ca_rmsd, without computing the RMSD."""
    offset = CS.detect_residue_offset(pred_ordered, ref_ordered)
    if offset is not None:
        pred_mapped = {r + offset: atom for r, atom in pred_by.items()}
        common = sorted(set(pred_mapped) & set(ref_by))
        if len(common) >= max(10, int(0.3 * min(len(pred_ordered), len(ref_ordered)))):
            return "offset", int(offset), len(common)
    n = min(len(pred_ordered), len(ref_ordered))
    return "fallback", None, n


def main():
    parser = PDBParser(QUIET=True)
    references = {}
    for state, rel in (("active", CS.DEFAULT_REF_ACTIVE),
                       ("i1", CS.DEFAULT_REF_I1),
                       ("i2", CS.DEFAULT_REF_I2)):
        ordered, by_resseq, _ = CS.load_reference(parser, str(CS.resolve_path(rel)), state)
        references[state] = (ordered, by_resseq)

    protocols = {
        "original": (HERE / "output", HERE / "results" / "state_classifications.json"),
        "fresh_msa": (
            HERE / "output_independent_msa" / "output",
            HERE / "output_independent_msa" / "results" / "state_classifications.json",
        ),
    }

    out = {"protocols": {}}
    for pname, (structure_dir, stored_path) in protocols.items():
        predictions = CS.find_predictions(str(structure_dir))
        if not predictions:
            raise FileNotFoundError(f"No predictions under {structure_dir}")

        stored = json.loads(Path(stored_path).read_text(encoding="utf-8"))
        expected = {(r["mutant"], r["run"], r["model"], r["seed"]): r["state"]
                    for r in stored["classifications"]}

        mode_counts = Counter()
        per_reference = {s: Counter() for s in ("active", "i1", "i2")}
        records = []
        n_match = 0

        for pdb_path, mutant, run, model, seed in predictions:
            structure = parser.get_structure("prediction", str(pdb_path))
            ordered, by_resseq = CS.get_ca_atoms(structure, chain_id="A", model_id=0)
            if not ordered:
                ordered, by_resseq = CS.get_ca_atoms(structure)

            rmsds = {}
            diags = {}
            for state, (ro, rb) in references.items():
                mode, offset, matched = alignment_mode(ordered, by_resseq, ro, rb)
                rmsd, n_atoms = CS.compute_ca_rmsd(ordered, by_resseq, ro, rb)
                rmsds[state] = float(rmsd)
                diags[state] = {"mode": mode, "offset": offset, "matched_residues": matched}
                mode_counts[mode] += 1
                per_reference[state][mode] += 1

            assigned = CS.classify_state(
                rmsds["active"], rmsds["i1"], rmsds["i2"], CS.RMSD_THRESHOLD
            )
            key = (mutant, run, model, seed)
            if expected.get(key) == assigned:
                n_match += 1
            records.append({
                "mutant": mutant, "run": run, "model": model, "seed": seed,
                "alignment": diags, "assigned_state": assigned,
            })

        out["protocols"][pname] = {
            "n_structures": len(records),
            "alignment_mode_counts": dict(mode_counts),
            "per_reference_mode_counts": {s: dict(per_reference[s]) for s in per_reference},
            "frozen_assignment_matches": n_match,
            "frozen_assignment_total": len(expected),
            "records": records,
        }
        print(f"[{pname}] structures={len(records)} modes={dict(mode_counts)} "
              f"frozen_match={n_match}/{len(expected)}")

    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    lines = ["# AF2 alignment-mode audit (full-protein C-alpha)",
             "",
             "Status: **ADDITIVE AUDIT** (2026-08-13). Recomputes full-protein RMSDs "
             "and the frozen 3.0 A assignment; does not change preregistered classifications.",
             ""]
    for pname, block in out["protocols"].items():
        lines.append(f"## {pname}")
        lines.append(f"- structures: {block['n_structures']}")
        lines.append(f"- alignment-mode counts: {block['alignment_mode_counts']}")
        lines.append(f"- per-reference mode counts: {block['per_reference_mode_counts']}")
        lines.append(f"- frozen assignment match: {block['frozen_assignment_matches']}/"
                     f"{block['frozen_assignment_total']}")
        lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {OUT_JSON}")


if __name__ == "__main__":
    main()
