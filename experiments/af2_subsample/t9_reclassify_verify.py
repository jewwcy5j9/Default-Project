#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T9: Re-classify from the PDBs now on disk and verify against the stored
classification JSONs (checks whether any downstream numbers used a stale
data version); then add the B1 (480) alignment-region sensitivity that
was previously blocked on the server-side PDBs.

  A. Main 840 ensemble: full-protein re-classification vs
     results/state_classifications.json (per-record match).
  B. B1 480 ensemble: full-protein re-classification vs
     output_independent_msa/results/state_classifications.json.
  C. B1 region sensitivity: N-lobe+activation (235-400) and alphaC-only
     (260-300) alignments, same argmin rule at 3.0 A.

Output: results/t9_reclassify_verify.json
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from Bio.PDB import PDBParser, Superimposer

import classify_states as CS

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
OUT = HERE / "results"
REGIONS = {"n_lobe_act": (235, 400), "alphaC_only": (260, 300)}
MUTANTS = ['WT', 'M290L', 'L301I', 'M290L_L301I', 'F382L', 'F382Y', 'F382V']


def load_refs():
    refs = {}
    for key, path in [("active", CS.DEFAULT_REF_ACTIVE),
                      ("i1", CS.DEFAULT_REF_I1),
                      ("i2", CS.DEFAULT_REF_I2)]:
        p = CS.resolve_path(path)
        s = PDBParser(QUIET=True).get_structure(key, str(p))
        ordered, by_resseq = CS.get_ca_atoms(s, chain_id="A", model_id=0)
        if not ordered:
            ordered, by_resseq = CS.get_ca_atoms(s)
        refs[key] = (ordered, by_resseq)
    return refs


def compute_region_offsets(refs):
    """Per-reference offset of the shared region frame.

    Region bounds in this module are stated in the 2HYY (I1 reference)
    numbering frame, which matches the project's canonical Abl1 residue
    numbering (E255V/T315I/M290L/F382 sites). The active/I2 references
    (6XR6/6XRG) number the same physical residues +19 higher, so each
    reference's filter bounds must be translated by its offset relative to
    the I1 frame before use.
    """
    offsets = {"i1": 0}
    for key in ("active", "i2"):
        offset = CS.detect_residue_offset(refs["i1"][0], refs[key][0])
        if offset is None:
            raise RuntimeError(f"cannot align reference {key} to the I1 frame")
        offsets[key] = int(offset)
    return offsets


def region_rmsd(pred_ordered, pred_by_resseq, ref_ordered, ref_by_resseq,
                region=(235, 400), region_offset=0):
    """Kabsch C-alpha RMSD restricted to a residue-number region.

    ``region`` is given in the I1-reference (2HYY) numbering frame;
    ``region_offset`` translates it into this reference's numbering
    (ref_resseq = i1_frame_resseq + region_offset)."""
    offset = CS.detect_residue_offset(pred_ordered, ref_ordered)
    if offset is None:
        return None, 0
    pred_mapped = {r + offset: atom for r, atom in pred_by_resseq.items()}
    lo = region[0] + region_offset
    hi = region[1] + region_offset
    common = sorted(set(pred_mapped.keys()) & set(ref_by_resseq.keys()))
    common = [r for r in common if lo <= r <= hi]
    if len(common) < 10:
        return None, len(common)
    fixed = [ref_by_resseq[r] for r in common]
    moving = [pred_mapped[r] for r in common]
    sup = Superimposer()
    sup.set_atoms(fixed, moving)
    return float(sup.rms), len(common)


def classify_one(pred_path, refs, region=None, region_offsets=None):
    s = PDBParser(QUIET=True).get_structure("p", str(pred_path))
    pred_ord, pred_by = CS.get_ca_atoms(s, chain_id="A", model_id=0)
    if not pred_ord:
        pred_ord, pred_by = CS.get_ca_atoms(s)
    rmsds = {}
    for key, (ro, rb) in refs.items():
        if region is None:
            rmsd, _ = CS.compute_ca_rmsd(pred_ord, pred_by, ro, rb)
        else:
            rmsd, _ = region_rmsd(pred_ord, pred_by, ro, rb, region,
                                  region_offsets.get(key, 0))
        rmsds[key] = rmsd
    valid = {k: v for k, v in rmsds.items() if v is not None}
    if not valid:
        return None, None
    best = min(valid, key=valid.get)
    state = {"active": "active", "i1": "I1", "i2": "I2"}.get(best, best) \
        if valid[best] < CS.RMSD_THRESHOLD else "unclassified"
    return state, valid


def reclassify_dir(output_dir, refs, region=None, region_offsets=None):
    preds = CS.find_predictions(str(output_dir))
    records = []
    for pdb_path, mutant, run, model, seed in preds:
        state, _ = classify_one(pdb_path, refs, region, region_offsets)
        records.append({"mutant": mutant, "run": run, "model": model,
                        "seed": seed, "state": state})
    return records


def compare(records_new, stored_json_path):
    stored = json.loads(stored_json_path.read_text(encoding="utf-8"))
    key = lambda r: (r["mutant"], r["run"], r["model"], r["seed"])
    old = {key(r): r["state"] for r in stored["classifications"]}
    n_match = n_diff = n_missing = 0
    diffs = []
    for r in records_new:
        k = key(r)
        if k not in old:
            n_missing += 1
            continue
        if old[k] == r["state"]:
            n_match += 1
        else:
            n_diff += 1
            diffs.append((k, old[k], r["state"]))
    extra = len(old) - (n_match + n_diff)
    return {"n_new": len(records_new), "n_stored": len(old),
            "n_match": n_match, "n_diff": n_diff, "n_missing_new": n_missing,
            "n_extra_stored": max(extra, 0), "diffs": diffs[:20],
            "n_diffs_shown": len(diffs)}


def summarize(records):
    c = Counter(r["state"] for r in records)
    return dict(c)


def main():
    t0 = time.time()
    print("=" * 90)
    print("T9: reclassify from local PDBs + verify vs stored JSON + B1 regions")
    print("=" * 90)
    refs = load_refs()
    region_offsets = compute_region_offsets(refs)
    out = {
        "region_frame": {
            "frame": "2HYY/I1 numbering (canonical Abl1 residue numbering)",
            "reference_region_offsets": region_offsets,
        },
    }

    # A. main 840
    main_dir = CS.resolve_path(CS.DEFAULT_OUTPUT_DIR)
    main_records = reclassify_dir(main_dir, refs)
    out["main_840"] = {
        "reclassified": summarize(main_records),
        "compare_vs_stored": compare(main_records, HERE / "results" / "state_classifications.json"),
    }
    print("\n[A] main 840 reclass:", summarize(main_records))
    cmp = out["main_840"]["compare_vs_stored"]
    print(f"    match={cmp['n_match']} diff={cmp['n_diff']} "
          f"missing={cmp['n_missing_new']} stored={cmp['n_stored']}")
    for d in cmp["diffs"][:10]:
        print("    DIFF", d)

    # B. B1 480 full-protein
    b1_dir = HERE / "output_independent_msa" / "output"
    b1_records = reclassify_dir(b1_dir, refs)
    out["b1_480_full_protein"] = {
        "reclassified": summarize(b1_records),
        "compare_vs_stored": compare(
            b1_records, HERE / "output_independent_msa" / "results" / "state_classifications.json"),
    }
    print("\n[B] B1 480 reclass:", summarize(b1_records))
    cmp = out["b1_480_full_protein"]["compare_vs_stored"]
    print(f"    match={cmp['n_match']} diff={cmp['n_diff']} "
          f"missing={cmp['n_missing_new']} stored={cmp['n_stored']}")
    for d in cmp["diffs"][:10]:
        print("    DIFF", d)

    # C. B1 region sensitivity
    b1_region = {}
    for rname, region in REGIONS.items():
        recs = reclassify_dir(b1_dir, refs, region, region_offsets)
        b1_region[rname] = {
            "region": f"residues {region[0]}-{region[1]}",
            "counts": summarize(recs),
            "i1i2_hits": sum(1 for r in recs if r["state"] in ("I1", "I2")),
        }
        print(f"\n[C] B1 {rname}: {summarize(recs)} -> I1/I2 hits "
              f"{b1_region[rname]['i1i2_hits']}")
    out["b1_region_sensitivity"] = b1_region
    out["full_protein_b1_i1i2"] = out["b1_480_full_protein"]["reclassified"].get("I1", 0) \
        + out["b1_480_full_protein"]["reclassified"].get("I2", 0)

    out["runtime_seconds"] = float(time.time() - t0)
    out_path = OUT / "t9_reclassify_verify.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
