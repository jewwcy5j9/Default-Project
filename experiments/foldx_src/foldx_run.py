"""Phase-1 FoldX runner: RepairPDB + 5x BuildModel per (state, mutation).

States: active (6XR6), i1 (2HYY), i2 (6XRG). Paper numbering maps per
results/abl1_state_mapping.json (2HYY offset 0; 6XR6/6XRG offset +19).

I2 special handling (see results/structure_review.md):
  - track 1 (default): restore construct mutation L309M via BuildModel, then run
    target mutations on the restored I2-WT reference (cells marked restored_wt).
  - track 2: no restore; mutations hitting residue 290 are identity (degenerate,
    recorded with mean_ddg = 0 and degenerate = true).

Usage:
  python foldx_run.py --foldx PATH --out results/foldx_abl1_ddg.json [--track 1|2] [--dry-run]

--dry-run: print the exact command plan + individual-list files without running.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
PREPARED = HERE / "prepared"
WORK = HERE / "foldx_work"
RESULTS = HERE / "results"

PREPARED_NAMES = {"active": "abl1_active_wt.pdb", "i1": "abl1_i1_wt.pdb", "i2": "abl1_i2_wt.pdb"}

MUTATIONS = ["M290L", "L301I", "M290L_L301I", "F382L", "F382Y", "F382V", "H396P"]
MUT_ATOMS = {
    "M290L": [(290, "M", "L")],
    "L301I": [(301, "L", "I")],
    "M290L_L301I": [(290, "M", "L"), (301, "L", "I")],
    "F382L": [(382, "F", "L")],
    "F382Y": [(382, "F", "Y")],
    "F382V": [(382, "F", "V")],
    "H396P": [(396, "H", "P")],
}
BETA = 1.677  # kcal^-1 at 300 K


def load_mapping():
    mp = json.loads((RESULTS / "abl1_state_mapping.json").read_text(encoding="utf-8"))
    return mp


def paper_to_pdb(mp, state, paper_pos):
    return paper_pos + mp["states"][state]["offset_paper_to_pdb"]


def mutation_string(mp, state, mutation, chain="A"):
    parts = [f"{wt}{chain}{paper_to_pdb(mp, state, p)}{mut}" for p, wt, mut in MUT_ATOMS[mutation]]
    return ",".join(parts) + ";"


def hits_paper_pos(mutation, paper_pos):
    return any(p == paper_pos for p, _, _ in MUT_ATOMS[mutation])


def run_cmd(foldx, args, cwd):
    cmd = [str(foldx), *args]
    print("RUN:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(f"FoldX failed ({r.returncode}): {r.stderr[-2000:]}")
    return r.stdout


def parse_total_energy(fxout: Path):
    text = fxout.read_text(encoding="utf-8", errors="replace")
    vals = [float(m.group(1)) for m in
            re.finditer(r"^\s*Total energy[^:]*:\s*([-+0-9.eE]+)", text, re.M)]
    if not vals:
        raise RuntimeError(f"no 'Total energy' line in {fxout.name}")
    return vals[-1]


def collect_ddgs(state_dir, prefix):
    """FoldX 5.1: parse Raw_<prefix>.fxout (one line per molecule:
    '<pdb> total energy ...'; mutant rows '<prefix>_N.pdb', WT rows
    'WT_<prefix>_N.pdb'). ddg = total(mutant) - total(WT) per run N.
    """
    raw = state_dir / f"Raw_{prefix}.fxout"
    text = raw.read_text(encoding="utf-8", errors="replace")
    totals = {}  # key: run N (int) -> {"mut": x, "wt": y}
    for line in text.splitlines():
        m = re.match(r"^\s*((?:WT_)?(?:[A-Za-z0-9._-]*?)_(\d+))\.pdb\s+([-+0-9.eE]+)", line)
        if not m:
            continue
        name, n, total = m.group(1), int(m.group(2)), float(m.group(3))
        key = (n, name.startswith("WT_"))
        totals[key] = total
    runs = {}
    for (n, is_wt), total in totals.items():
        runs.setdefault(n, {})["wt" if is_wt else "mut"] = total
    ddgs = []
    for n in sorted(runs):
        r = runs[n]
        if "mut" not in r or "wt" not in r:
            raise RuntimeError(f"incomplete Raw fxout for run {n}: {r}")
        ddgs.append(r["mut"] - r["wt"])
    if not ddgs:
        raise RuntimeError(f"no run rows parsed from {raw.name}")
    return ddgs


def find_output_pdb(state_dir, newer_than):
    """Find BuildModel mutant output pdb (name varies by FoldX version)."""
    cands = []
    for p in sorted(state_dir.glob("*.pdb")):
        if p.stat().st_mtime >= newer_than and not p.name.endswith("_Repair.pdb") and "WT_" not in p.name:
            cands.append(p)
    if not cands:
        raise RuntimeError("no BuildModel output pdb found")
    return cands[0] if len(cands) == 1 else cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--foldx", default=None, help="path to foldx binary")
    ap.add_argument("--out", default=str(RESULTS / "foldx_abl1_ddg.json"))
    ap.add_argument("--track", type=int, default=1, choices=[1, 2])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run and not args.foldx:
        sys.exit("--foldx PATH required (or --dry-run)")
    mp = load_mapping()
    beta = mp.get("beta", {"value": BETA, "unit": "kcal^-1", "note": "1/(RT) at 300 K"})

    results = {"schema": "registry v1 (per-fold/per-run)",
               "states": ["active", "i1", "i2"], "track": args.track,
               "beta": beta, "per_cell": [], "failures": [], "restored_i2_wt_used": False}

    work = WORK / f"track{args.track}"
    work.mkdir(parents=True, exist_ok=True)

    for state in ["active", "i1", "i2"]:
        src = PREPARED / PREPARED_NAMES[state]
        state_dir = work / state
        state_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, state_dir / src.name)

        if args.dry_run:
            print(f"--- {state}: RepairPDB")
            print(f"    foldx --command=RepairPDB --pdb={src.name}")
            continue

        print(f"\n=== {state}: RepairPDB ===")
        run_cmd(args.foldx, ["--command=RepairPDB", f"--pdb={src.name}"], state_dir)
        repaired = sorted(state_dir.glob("*_Repair.pdb"))
        if not repaired:
            results["failures"].append({"state": state, "stage": "RepairPDB",
                                        "message": "no *_Repair.pdb produced"})
            print(f"[FAIL] no repaired pdb for {state}")
            continue
        wt_pdb = repaired[0]

        # I2 restore step (track 1): L309M -> restored I2-WT reference
        i2_restore_done = False
        if state == "i2" and args.track == 1:
            print(f"  restore: BuildModel {wt_pdb.name} L309M")
            mfile = state_dir / "individual_list_restore.txt"
            mfile.write_text("LA309M;\n", encoding="utf-8")
            t0 = wt_pdb.stat().st_mtime
            run_cmd(args.foldx, ["--command=BuildModel", f"--pdb={wt_pdb.name}",
                                 f"--mutant-file={mfile.name}", "--numberOfRuns=1"], state_dir)
            wt_pdb = find_output_pdb(state_dir, t0)
            i2_restore_done = True

        for mutation in MUTATIONS:
            deg = state == "i2" and args.track == 2 and hits_paper_pos(mutation, 290)
            if deg:
                results["per_cell"].append({"state": state, "mutation": mutation,
                                            "mean_ddg": 0.0, "std_ddg": None, "runs": [],
                                            "n_runs": 0, "restored_wt": False,
                                            "degenerate": True, "error": None})
                print(f"  {state}/{mutation}: DEGENERATE (M290 already L in 6XRG)")
                continue

            mstring = mutation_string(mp, state, mutation)
            print(f"  {state}/{mutation}: BuildModel '{mstring}' x5")
            mfile = state_dir / f"individual_list_{state}_{mutation}.txt"
            mfile.write_text(mstring + "\n", encoding="utf-8")
            prefix = wt_pdb.stem
            t0 = wt_pdb.stat().st_mtime
            try:
                run_cmd(args.foldx, ["--command=BuildModel", f"--pdb={wt_pdb.name}",
                                     f"--mutant-file={mfile.name}", "--numberOfRuns=5"],
                        state_dir)
                ddgs = collect_ddgs(state_dir, prefix)
            except Exception as e:  # noqa: BLE001 - record and continue
                results["failures"].append({"state": state, "mutation": mutation,
                                            "stage": "BuildModel", "message": str(e)})
                print(f"[FAIL] {state}/{mutation}: {e}")
                continue
            results["per_cell"].append({"state": state, "mutation": mutation,
                                        "runs": ddgs, "n_runs": len(ddgs),
                                        "mean_ddg": float(sum(ddgs) / len(ddgs)),
                                        "std_ddg": float(__import__("numpy").std(ddgs)),
                                        "restored_wt": i2_restore_done,
                                        "degenerate": False, "error": None})
        if i2_restore_done:
            results["restored_i2_wt_used"] = True

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()
