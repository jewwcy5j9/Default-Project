"""Phase-1 FoldX runner: RepairPDB + 5x BuildModel per (state, mutation) for Src.

States: active (9NS0), e1 (9NS1), e2 (2SRC). PDB numbering == verified
structural numbering (see src_e2_structure_decision.md); mutation list follows
k3_data.SRC_K3 with structural positions (A311I -> pos 309, F405A -> pos 408).

Usage:
  python foldx_run_src.py --foldx PATH --out results/foldx_src_ddg.json [--dry-run]
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
WORK = HERE / "foldx_work_src"
RESULTS = HERE / "results"

PREPARED_NAMES = {"active": "src_active_9NS0.pdb", "e1": "src_e1_9NS1.pdb", "e2": "src_e2_2src.pdb"}

# structural PDB positions (verified against each state PDB)
MUTATIONS = ["L410A", "V332I", "L270F_V332I", "L325A", "A311I", "V380A", "V331A", "F405A"]
MUT_ATOMS = {
    "L410A": [(410, "L", "A")],
    "V332I": [(332, "V", "I")],
    "L270F_V332I": [(270, "L", "F"), (332, "V", "I")],
    "L325A": [(325, "L", "A")],
    "A311I": [(309, "A", "I")],
    "V380A": [(380, "V", "A")],
    "V331A": [(331, "V", "A")],
    "F405A": [(408, "F", "A")],
}
BETA = 1.677  # kcal^-1 at 300 K


def mutation_string(state, mutation, chain="A"):
    parts = [f"{wt}{chain}{p}{mut}" for p, wt, mut in MUT_ATOMS[mutation]]
    return ",".join(parts) + ";"


def run_cmd(foldx, args, cwd):
    cmd = [str(foldx), *args]
    print("RUN:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(f"FoldX failed ({r.returncode}): {r.stderr[-2000:]}")
    return r.stdout


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
    ap.add_argument("--out", default=str(RESULTS / "foldx_src_ddg.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run and not args.foldx:
        sys.exit("--foldx PATH required (or --dry-run)")

    results = {"schema": "registry v1 (per-fold/per-run)",
               "states": ["active", "e1", "e2"],
               "beta": {"value": BETA, "unit": "kcal^-1", "note": "1/(RT) at 300 K"},
               "per_cell": [], "failures": []}

    work = WORK
    work.mkdir(parents=True, exist_ok=True)

    for state in ["active", "e1", "e2"]:
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

        for mutation in MUTATIONS:
            mstring = mutation_string(state, mutation)
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
                                        "degenerate": False, "error": None})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()