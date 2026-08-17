#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T8: AF2 closing analyses (completes the pre-registered report list of
NEXT_TIER_EXECUTION_PLAN.md §6.4; all CPU-local, no GPU needed).

  A. B1 (independent-MSA, 480 predictions): per-mutant ICC over model /
     seed / run on the non-active indicator (same estimator as T5b), plus
     the 95% Clopper-Pearson upper bound for the 0/480 I1/I2 hit rate.
  B. Main 840-prediction ensemble: alignment-region sensitivity. Reclassify
     every prediction with Kabsch fit restricted to the alphaC-helix region
     (Abl1 residues 260-300) and compare against the full-protein
     classification (0/840 I1/I2 at 3.0 A, argmin rule).

Output: results/t8_af2_region_sensitivity.json
"""
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from Bio.PDB import PDBParser

import classify_states as CS
from t9_reclassify_verify import (
    compute_region_offsets,
    load_refs,
    region_rmsd,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
OUT = HERE / "results"
# region sensitivity: N-lobe + activation segment vs full protein. The tiny
# alphaC-only region (260-300) is reported separately as a discriminability
# check: local fits over ~40 Ca atoms leave RMSD with no state-discrimination
# power (the classifier collapses to a single reference).
REGIONS = {"n_lobe_act": (235, 400), "alphaC_only": (260, 300)}
# Region bounds above are stated in the 2HYY/I1 numbering frame (canonical
# Abl1 residue numbering); compute_region_offsets translates them per
# reference because 6XR6/6XRG number the same physical residues +19 higher.
THRESHOLD = 3.0
MUTANTS = ['WT', 'M290L', 'L301I', 'M290L_L301I', 'F382L', 'F382Y', 'F382V']


def clopper_pearson_upper(n_trials, n_hits, alpha=0.05):
    if n_hits == 0:
        return 1 - (alpha) ** (1.0 / n_trials)
    raise NotImplementedError("only zero-hit upper bound needed")


def icc(groups):
    grand = np.mean([v for g in groups for v in g])
    k = len(groups)
    n_j = np.array([len(g) for g in groups])
    ss_b = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    ss_w = sum(sum((v - np.mean(g)) ** 2 for v in g) for g in groups)
    n = int(n_j.sum())
    ms_b = ss_b / (k - 1)
    ms_w = ss_w / (n - k)
    n0 = (n - sum(n_j ** 2) / n) / (k - 1)
    return float((ms_b - ms_w) / (ms_b + (n0 - 1) * ms_w))


def main():
    t0 = time.time()
    print("=" * 90)
    print("T8: AF2 closing analyses (B1 ICC/upper bound + region sensitivity)")
    print("=" * 90)
    out = {}

    # ---------------- A. B1 ICC + upper bound ----------------
    b1 = json.loads((HERE / "output_independent_msa" / "results"
                     / "state_classifications.json").read_text(encoding="utf-8"))
    rows = []
    for r in b1["classifications"]:
        rows.append((r["mutant"], r["run"], r["model"], r["seed"],
                     1.0 if r["state"] != "active" else 0.0))
    by_mut = collections.defaultdict(list)
    for m, run, model, seed, v in rows:
        by_mut[m].append((run, model, seed, v))
    per_mut = {}
    for m, vals in by_mut.items():
        by_model = collections.defaultdict(list)
        by_seed = collections.defaultdict(list)
        by_run = collections.defaultdict(list)
        for run, model, seed, v in vals:
            by_model[model].append(v)
            by_seed[seed].append(v)
            by_run[run].append(v)
        per_mut[m] = {
            "icc_model": icc(list(by_model.values())),
            "icc_seed": icc(list(by_seed.values())),
            "icc_run": icc(list(by_run.values())),
            "coverage_non_active": float(np.mean([v for *_, v in vals])),
            "n_pdb": len(vals),
        }
    for m in per_mut:
        for k in ("icc_model", "icc_seed", "icc_run"):
            if per_mut[m][k] != per_mut[m][k]:  # NaN (zero variance group)
                per_mut[m][k] = None
    i1i2_total = sum(1 for r in b1["classifications"] if r["state"] in ("I1", "I2"))
    out["b1_icc"] = per_mut
    out["b1_upper_bound"] = {
        "n_trials": b1["n_classified"],
        "n_i1i2_hits": i1i2_total,
        "clopper_pearson_95_upper": float(clopper_pearson_upper(
            b1["n_classified"], i1i2_total)),
    }
    print("\n[A] B1 ICC (independent MSA, 480)")
    for m, v in per_mut.items():
        fmt = lambda x: "nan" if x is None else f"{x:.2f}"
        print(f"  {m}: cov={v['coverage_non_active']:.3f} "
              f"ICC(model)={fmt(v['icc_model'])} ICC(seed)={fmt(v['icc_seed'])} "
              f"ICC(run)={fmt(v['icc_run'])}")
    print(f"  0/480 Clopper-Pearson 95% upper bound = "
          f"{out['b1_upper_bound']['clopper_pearson_95_upper']:.4f}")

    # ---------------- B. region sensitivity on the 840 ensemble ----------------
    refs = load_refs()
    region_offsets = compute_region_offsets(refs)
    out["region_frame"] = {
        "frame": "2HYY/I1 numbering (canonical Abl1 residue numbering)",
        "reference_region_offsets": region_offsets,
    }
    print("\n[B] alignment-region sensitivity on the main 840 ensemble")
    main_cls = json.loads((HERE / "results" / "state_classifications.json")
                          .read_text(encoding="utf-8"))["classifications"]
    main_state = {(r["mutant"], r["run"], r["model"], r["seed"]): r["state"]
                  for r in main_cls}

    region_results = {}
    for region_name, region in REGIONS.items():
        region_state = collections.Counter()
        per_mut_region = collections.defaultdict(collections.Counter)
        n_missing = 0
        n_atoms_min = 999
        n_atoms_max = 0
        output_dir = CS.resolve_path(CS.DEFAULT_OUTPUT_DIR)
        for m in MUTANTS:
            for run in range(3):
                for model in range(1, 6):
                    for seed in range(8):
                        p = output_dir / m / f"run_{run}" / f"model_{model}_seed_{seed}.pdb"
                        if not p.exists():
                            n_missing += 1
                            continue
                        s = PDBParser().get_structure(m, str(p))
                        pred_ordered, pred_by = CS.get_ca_atoms(s)
                        rmsds = {}
                        for key, (ro, rb) in refs.items():
                            rmsd, na = region_rmsd(pred_ordered, pred_by, ro, rb,
                                                   region, region_offsets[key])
                            rmsds[key] = rmsd
                            if na:
                                n_atoms_min = min(n_atoms_min, na)
                                n_atoms_max = max(n_atoms_max, na)
                        valid = {k: v for k, v in rmsds.items() if v is not None}
                        if not valid:
                            n_missing += 1
                            continue
                        best = min(valid, key=valid.get)
                        state = {"active": "active", "i1": "I1", "i2": "I2"}[best] \
                            if valid[best] < THRESHOLD else "unclassified"
                        region_state[state] += 1
                        per_mut_region[m][state] += 1

        total = sum(region_state.values())
        region_results[region_name] = {
            "region": f"residues {region[0]}-{region[1]}",
            "n_classified": total,
            "n_missing": n_missing,
            "n_ca_atoms_min": n_atoms_min, "n_ca_atoms_max": n_atoms_max,
            "state_counts": dict(region_state),
            "coverage": {k: float(v / total) for k, v in region_state.items()},
            "per_mutant": {m: dict(c) for m, c in per_mut_region.items()},
            "i1i2_hits": int(region_state.get("I1", 0) + region_state.get("I2", 0)),
        }
        print(f"  {region_name} (res {region[0]}-{region[1]}): "
              f"{dict(region_state)} -> I1/I2 hits "
              f"{region_results[region_name]['i1i2_hits']}")

    out["region_sensitivity"] = {
        "regions": region_results,
        "full_protein_i1i2_hits": int(sum(
            1 for st in main_state.values() if st in ("I1", "I2"))),
    }
    print(f"  full-protein I1/I2 hits: "
          f"{out['region_sensitivity']['full_protein_i1i2_hits']}")

    out["runtime_seconds"] = float(time.time() - t0)
    out_path = OUT / "t8_af2_region_sensitivity.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
