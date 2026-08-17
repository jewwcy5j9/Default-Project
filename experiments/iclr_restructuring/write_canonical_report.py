import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "results" / "canonical_reproduction.json"
OUT = HERE / "results" / "canonical_reproduction_report.md"

d = json.loads(SRC.read_text(encoding="utf-8"))
local = [i for i in d["items"] if "PCA" not in i["key"]]
gpu = [i for i in d["items"] if "PCA" in i["key"]]

lines = ["# Canonical Reproduction Report (Phase-0 closure)", ""]
lines.append(f"- Date: {d['timestamp']}")
lines.append(f"- Tolerance: ±{d['tol']}")
lines.append(f"- Runtime: {d['runtime_seconds']}s")
lines.append("- Entry point: `python reproduce_canonical.py` (single entry, frozen code paths: `k3_data.py`, `k3_benchmark.py`, `k3_llr_proxy.py`, `p0_grouped_cv.py`, `t1_nested_cv.py`, `k3_weighted_loss.py` (Extended_w_pow05); L410A sensitivity embedded, same protocol; T7 items read live from `results/t7_fold_local_esm_pca_v2.json` (falls back to v1), PENDING_REMOTE only if absent; R4 artifact carries script/module/cache/env hashes and is verified against the current t7_fold_local_esm_pca.py; canonical input hashes carry source-data provenance because the server-layout T7 data-file fields are null)")
lines.append(f"- Input hashes: {json.dumps(d['input_hashes'])}")
if d.get("t7_input_hash"):
    lines.append(f"- T7 input hash: {d['t7_input_hash']}")
if d.get("provenance_notes"):
    lines.append(f"- Provenance notes: {json.dumps(d['provenance_notes'])}")
lines.append("- Source: results/canonical_reproduction.json")
lines.append("")
lines.append("## Locally recomputed items")
lines.append("")
lines.append("| key | registry target | recomputed | status |")
lines.append("|---|---|---|---|")
for i in local:
    lines.append(f"| {i['key']} | {i['target']} | {i['value']} | {i['status']} |")
lines.append("")
lines.append("## GPU-verification items")
lines.append("")
lines.append("| key | registry target | measured value | status |")
lines.append("|---|---|---|---|")
for i in gpu:
    lines.append(f"| {i['key']} | {i['target']} | {i['value']} | {i['status']} |")
lines.append("")
lines.append("T7 values are read live from `results/t7_fold_local_esm_pca_v2.json` at runtime;")
lines.append("the R4 rerun (GPU ESM-2 encoding, n_seeds=5 per-seed, run twice with")
lines.append("identical outputs, diff 0.000000) supersedes v1 (A4b) and the 8/4 values")
lines.append("(0.1459 / 0.3026), which are not reproducible in the current server venv.")
lines.append("K=2 fold-local PCA: abl1 0.21364, src 0.40763 (v2: 0.21360 / 0.40909).")
lines.append("")
lines.append("## Verdict")
lines.append("")
n_pass = sum(1 for i in d["items"] if i["status"] == "PASS")
failed = [i["key"] for i in d["items"] if i["status"] != "PASS"]
if n_pass == len(d["items"]):
    lines.append(f"All {len(d['items'])} canonical values PASS ({len(local)} local + "
                 f"{len(gpu)} GPU-verified T7) within ±{d['tol']} of the registry.")
    lines.append("Phase-0 closure: **GO**.")
else:
    lines.append(f"{n_pass}/{len(d['items'])} canonical values PASS; "
                 f"FAILED items: {failed}.")
    lines.append("Phase-0 closure: **NO-GO**.")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({len(d['items'])} items)")
