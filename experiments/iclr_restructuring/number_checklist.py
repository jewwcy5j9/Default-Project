"""Post-fix consistency checklist: result JSONs versus manuscript claims.

Historical checks still cover paper/main.tex. Frozen follow-up claims are
checked against the active manuscript (MS_TEX, default main_v3.tex).
Run: python number_checklist.py
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
R = Path(__file__).resolve().parent / "results"
TEX = ROOT / "paper" / "main.tex"
tex = TEX.read_text(encoding="utf-8")
ms_name = os.environ.get("MS_TEX", "main_v3.tex")
TEX_MS = ROOT / "paper" / ms_name
tex_ms = TEX_MS.read_text(encoding="utf-8")

def j(name):
    return json.loads((R / name).read_text(encoding="utf-8"))

def ok(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        sys.exit(f"FAILED: {label}")

def close(a, b, tol=0.0006):
    return abs(a - b) < tol

def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

# ---------------- core baselines (K=3, pooled) ----------------
cb = j("constant_baselines_core.json")
a = cb["abl1_k3_n6"]; s = cb["src_k3_n8"]
ok("Abl1 K3 constant-WT 0.3878", close(a["constant_WT"]["mean_mae"], 0.3878))
ok("Abl1 K3 training-mean 0.2329", close(a["training_mean_LOO"]["mean_mae"], 0.2329))
ok("Src K3 constant-WT 0.4600", close(s["constant_WT"]["mean_mae"], 0.4600))
tm_src = s["training_mean_LOO"]["mean_mae"]
print(f"    Src K3 training-mean json={tm_src:.4f} (paper: 0.2911)")
ok("Src K3 training-mean 0.2911", close(tm_src, 0.2911))
pooled = cb.get("abl1_pooled") or cb.get("abl1_pooled_n6") or {}
if "abl1_pooled_n6" in cb and "src_pooled_n8" in cb:
    ok("Abl1 pooled 0.5617", close(cb["abl1_pooled_n6"]["constant_WT"]["mean_mae"], 0.5617))
    ok("Abl1 pooled 0.3173", close(cb["abl1_pooled_n6"]["training_mean_LOO"]["mean_mae"], 0.3173))
    ok("Src pooled 0.5600", close(cb["src_pooled_n8"]["constant_WT"]["mean_mae"], 0.5600))
    ok("Src pooled 0.2529", close(cb["src_pooled_n8"]["training_mean_LOO"]["mean_mae"], 0.2529))

# ---------------- K=3 table numbers ----------------
k3 = j("k3_benchmark_results.json")
abl1 = k3["abl1_core"]; src = k3["src"]
ABL1_KEYS = ["Extended_10dim", "C_ddg_5dim", "pos_markers"]
SRC_KEYS = ["Extended_10dim", "pos_markers_4dim"]
# ESM2_20dim is NOT a k3_benchmark_results.json key: the ESM-2 rows are
# pinned from the T7/esm2 artifacts below, so its absence here is expected.
# Presence guard: the per-key checks below are conditional, so a renamed or
# removed JSON key would otherwise skip silently (vacuous pass).
ok("K3 benchmark keys present",
   all(k in abl1 for k in ABL1_KEYS) and all(k in src for k in SRC_KEYS))
for key, val in [("Extended_10dim", 0.3003), ("C_ddg_5dim", 0.0804),
                 ("ESM2_20dim", 0.3088), ("pos_markers", 0.2757)]:
    if key in abl1:
        ok(f"Abl1 {key}", close(abl1[key]["mae"], val), f"json={abl1[key]['mae']:.4f}")
for key, val in [("Extended_10dim", 0.3045), ("pos_markers_4dim", 0.3213),
                 ("ESM2_20dim", 0.3468)]:
    if key in src:
        ok(f"Src {key}", close(src[key]["mae"], val), f"json={src[key]['mae']:.4f}")

# ---------------- LLR proxy ----------------
llr = j("k3_llr_proxy_results.json")
ok("LLR fixed 0.163", close(llr["mae"], 0.1629))
ok("LLR direction 5/5", llr["direction"] == "5/5")

# ---------------- compositional baselines ----------------
p1 = j("p1_core_baselines.json")
ok("CLR-GP variantC 0.137", close(p1["abl1"]["variantC"]["CLR-GP"]["mae"], 0.1374))
ok("CLR-GP LLR 0.223", close(p1["abl1"]["LLR"]["CLR-GP"]["mae"], 0.2229))
ok("CLR-GP Src pos 0.256", close(p1["src"]["pos"]["CLR-GP"]["mae"], 0.2560))
ok("CLR-Ridge Src pos 0.2558", close(p1["src"]["pos"]["CLR-Ridge"]["mae"], 0.2558))
ok("Src CLR-GP Active direction 7/7", p1["src"]["pos"]["CLR-GP"]["direction"] == "7/7")
primary_gp = p1["_protocols"]["CLR-GP"]
ok("Primary GP protocol is locked",
   primary_gp["id"] == "GP-primary-0.05-v1"
   and primary_gp["kernel"] == "1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.05)"
   and primary_gp["alpha"] == 1e-4
   and primary_gp["normalize_y"] is True
   and primary_gp["n_restarts_optimizer"] == 1
   and primary_gp["random_state"] == 0)
f405_clrgp = p1["src"]["pos"]["CLR-GP"]["errors"]["SrcKD-F405A"]
print(f"    Src pos CLR-GP F405A err={f405_clrgp:.4f} (paper 0.55-0.56)")
ok("F405A CLR-GP err ~0.55-0.56", 0.55 <= f405_clrgp <= 0.56)

# ---------------- leave-site-out ----------------
lso = j("p0_grouped_cv.json")
ok("LSO variantC F382 0.2204", close(lso["variant_C"]["F382_family"]["group_mae"], 0.2204))
ok("LSO variantC 290/301 0.0801", close(lso["variant_C"]["290_301"]["group_mae"], 0.0801))
ok("LSO LLR F382 0.2058", close(lso["LLR_proxy"]["F382_family"]["group_mae"], 0.2058))
ok("LSO LLR 290/301 0.2637", close(lso["LLR_proxy"]["290_301"]["group_mae"], 0.2637))
ok("LSO pos F382 0.2862", close(lso["pos_markers"]["F382_family"]["group_mae"], 0.2862))
ok("LSO pos 290/301 0.1427", close(lso["pos_markers"]["290_301"]["group_mae"], 0.1427))

# ---------------- DDG provenance ----------------
ddg = j("p0_ddg_provenance.json")["summary"]
ok("ddg Pearson 0.91", close(ddg["pearson_reported_vs_implied"], 0.91, 0.005))
ok("ddg Spearman 0.90", close(ddg["spearman_reported_vs_implied"], 0.90))
ok("ddg mean|diff| 0.26", close(ddg["mean_abs_diff_all"], 0.26, 0.005))
ok("ddg max 0.87 (within 0.9)", ddg["max_abs_diff_all"] < 0.9)

# ---------------- nested CV ----------------
t1 = j("t1_nested_cv.json")
ok("nested MAE 0.2624", close(t1["nested_mae"], 0.2624))
ok("nested LLR scaling is fold-local",
   "training fold only" in t1.get("protocol", ""))
ok("fixed LLR 0.1629", close(t1["reference"]["LLR_fixed_mae"], 0.1629))
ok("training mean 0.2329", close(t1["reference"]["training_mean"], 0.2329))

# ---------------- T2 H396P ----------------
t2 = j("t2_i1_falsifiable.json")
vc = t2["variant_C"]["H396P"]; lr = t2["LLR"]["H396P"]
ok("H396P variantC dir wrong", vc["direction_ok"] is False)
ok("H396P LLR dir wrong", lr["direction_ok"] is False)
print(f"    H396P pred I2: variant_C={vc['pred'][2]:.4f}, LLR={lr['pred'][2]:.4f} (paper 0.116-0.177)")
ok("H396P I2 pred 0.116-0.177", 0.116 <= vc["pred"][2] <= 0.177 and 0.116 <= lr["pred"][2] <= 0.177)
print(f"    H396P MAE: variant_C={vc['mae']:.4f}, LLR={lr['mae']:.4f} (paper 0.0776-0.1176)")
ok("H396P MAE 0.0776-0.1176",
   round(vc["mae"], 4) == 0.1176 and round(lr["mae"], 4) == 0.0776)
ok("H396P const-WT 0.060", close(vc["constant_wt_mae"], 0.060))

# ---------------- T4 ----------------
t4 = j("t4_synthetic_support.json")
c01 = t4["collide_eps0.01"]
ok("T4 pair saturates ~0.13", all(0.115 <= v["pair_err"] <= 0.15 for v in c01.values()))
ok("T4 others decay <0.035", all(v["other_err"] < 0.035 for v in c01.values()))
deltas = [v["pair_sep_l1"] for v in c01.values()]
print(f"    T4 delta range: {min(deltas):.2f}-{max(deltas):.2f} (paper: ~0.7)")
ok("T4 delta ~0.6-0.85", 0.6 <= min(deltas) and max(deltas) <= 0.85)

# ---------------- p2c collision (L1) ----------------
p2c = j("p2c_feature_collision.json")
def count_coll(d, tau=0.25, delta=0.6):
    return sum(1 for v in d.values() if v["d_feat_l2_min"] < tau and v["d_dir_l1_at_min"] > delta)
src_pos = p2c["src_pos"]; abl1_pos = p2c["abl1_pos"]
abl1_ddg = p2c["abl1_ddg"]; abl1_llr = p2c["abl1_llr"]
ok("Src pos collisions 4/8", count_coll(src_pos) == 4)
ok("Abl1 pos collisions 3/6", count_coll(abl1_pos) == 3)
ok("Abl1 ddg collisions 0/6", count_coll(abl1_ddg) == 0)
ok("Abl1 llr collisions 0/6", count_coll(abl1_llr) == 0)
f405 = src_pos["SrcKD-F405A"]
ok("F405A nearest L410A", f405["nearest"] == "SrcKD-L410A")
ok("F405A d_feat 0.009", close(f405["d_feat_l2_min"], 0.0093, 0.0005))
ok("F405A d_dir L1 1.68", close(f405["d_dir_l1_at_min"], 1.68, 0.005))
l325 = src_pos["SrcKD-L325A"]
ok("L325A d_dir L1 1.10", close(l325["d_dir_l1_at_min"], 1.10, 0.005))
fl = p2c["floors"]["src_L410A_F405A"]["floor_per_state_mae"]
ok("floor L=5 ~0.272", close(fl["L=5.0"], 0.272, 0.003))
ok("floor L=20 ~0.249", close(fl["L=20.0"], 0.249, 0.003))

# ---------------- noise / risk ----------------
p2 = j("p2_theory_diagnostics.json")
nr = p2["noise_resolution"]["src"]
print(f"    noise flips: 5%={nr['lvl_5']['flipped_pairs_frac']:.5f}, 10%={nr['lvl_10']['flipped_pairs_frac']:.5f}")
ok("noise 5% ~0.0006", close(nr["lvl_5"]["flipped_pairs_frac"], 0.00064, 0.0001))
ok("noise 10% ~0.0164", close(nr["lvl_10"]["flipped_pairs_frac"], 0.0164, 0.0005))
ok("fine contrast Src 0.31", close(p2["risk_decomposition"]["src"]["fine_contrast_std_E1_share"], 0.31, 0.005))
ok("fine contrast Abl1 0.17", close(p2["risk_decomposition"]["abl1"]["fine_contrast_std_E1_share"], 0.17, 0.005))

# ---------------- zero-shot composition ----------------
comp = j("compositional_test.json")
print("    compositional_test:", json.dumps(comp, ensure_ascii=False)[:300])
kf = j("k3_followup_results.json")
print("    k3_followup keys:", list(kf.keys()))

# ---------------- T5 reviewer-response numbers ----------------
t5 = j("t5_review_responses.json")
t5b = j("t5b_review_responses.json")

db = t5["direction_baselines"]
ok("Src train-mean direction 8/8", db["src"]["summary"]["train_mean_dir_ok"] == 8)
ok("Src majority direction 8/8", db["src"]["summary"]["majority_dir_ok"] == 8)
ok("Abl1 train-mean direction 5/5", db["abl1"]["summary"]["train_mean_dir_ok"] == 5)
ok("Src majority MAE 0.3642", close(db["src"]["summary"]["majority_mae"], 0.3642, 0.001))

llr = t5["llr_only"]
ok("LLR-only CLR-Ridge 0.196", close(llr["CLR-Ridge"]["llr_only_mae"], 0.1958, 0.003))
ok("LLR+pos CLR-Ridge 0.262", close(llr["CLR-Ridge"]["llr_pos_mae"], 0.2616, 0.003))
ok("LLR-only CLR-GP 0.258", close(llr["CLR-GP"]["llr_only_mae"], 0.2584, 0.003))
ok("LLR+pos CLR-GP 0.222", close(llr["CLR-GP"]["llr_pos_mae"], 0.2222, 0.003))

cs = t5["contrast"]["clrgp_pos"]["summary"]
ok("CLR-GP u1 MAE 0.33", close(cs["u1_mae"], 0.326, 0.01))
ok("CLR-GP u2 MAE 0.60", close(cs["u2_mae"], 0.605, 0.01))
ok("CLR-GP u2 R2 < -1", cs["u2_r2"] < -1)
cm = t5["contrast"]["mlp_pos"]["summary"]
ok("Historical T5 artifact MLP u1 MAE 0.50", close(cm["u1_mae"], 0.501, 0.01))
ok("Historical T5 artifact MLP u2 MAE 0.71", close(cm["u2_mae"], 0.708, 0.01))
ok("K2 CLR-GP non-active 0.249", close(t5b["k2_clrgp_pos"]["non_active_mae"], 0.2489, 0.003))

sep = t5["collision_separation"]["models"]
ok("CLR-Ridge pair sep 0.003", close(sep["CLR-Ridge"]["output_separation_l1"], 0.003, 0.005))
ok("CLR-GP pair sep ~1.68", close(sep["CLR-GP"]["output_separation_l1"], 1.68, 0.02))
ok("kNN pair sep ~1.68", close(sep["kNN(1)"]["output_separation_l1"], 1.68, 0.02))

fl = t5["scale_sensitivity"]
ok("scale x1 d_feat 0.0093", close(fl["1.0"]["d_feat"], 0.0093, 0.0005))
ok("scale x0.01 still in range", fl["0.01"]["in_collision_range"])
ok("scale x100 out of range", not fl["100.0"]["in_collision_range"])

pt = t5["perturbation"]
ok("uniform pm10 flip 1.7%", close(pt["uniform_pm10"], 0.017, 0.002))
ok("dirichlet 50 flip 0.7%", close(pt["dirichlet_alpha50"], 0.0071, 0.002))
ok("lognormal flip 0.1%", close(pt["lognormal_sigma10"], 0.0011, 0.0005))

af2 = t5b["af2_icc"]
icc_ok = all(v["icc_model"] <= 0.16 and v["icc_msa"] <= 0.16 for v in af2.values())
ok("AF2 ICC(model)<=0.16 all mutants", icc_ok)
ml = t5b["af2_mutant_level"]
ok("AF2 mutant coverage std 0.015", close(ml["std_across_mutants"], 0.0152, 0.003))

# ---- T6: collision null calibration (independent-MSA-independent) ----
t6 = j("t6_collision_null.json")
ok("T6 dir-null abl1 exact 720 perms",
   t6["dir_null_abl1"]["n_exact"] == 720)
ok("T6 dir-null src exact 40320 perms",
   t6["dir_null_src"]["n_exact"] == 40320)
ok("T6 abl1 thresholded conflict count and null",
   t6["dir_null_abl1"]["thresholded_conflict_count_observed"] == 2
   and close(t6["dir_null_abl1"]["thresholded_conflict_count_null"]["p_perm_ge_obs"], 0.80, 0.02))
ok("T6 src thresholded conflict count and null",
   t6["dir_null_src"]["thresholded_conflict_count_observed"] == 6
   and close(t6["dir_null_src"]["thresholded_conflict_count_null"]["p_perm_ge_obs"], 0.86, 0.02))
ok(f"T6 thresholded conflict counts in {ms_name}",
   "thresholded conflict" in tex_ms and "0.80" in tex_ms and "0.86" in tex_ms)
sep_n = t6["sep_null_src_pos"]
ok("T6 CLR-Ridge sep 0.003",
   close(sep_n["CLR-Ridge"]["observed"], 0.003, 0.005))
ok("T6 CLR-GP sep 1.68",
   close(sep_n["CLR-GP"]["observed"], 1.68, 0.02))
ok("T6 kNN sep 1.68",
   close(sep_n["kNN(1)"]["observed"], 1.68, 0.02))
ok("T6 ridge LOO collision diff 0.20",
   close(t6["loo_error_null_src_pos"]["observed_diff"], 0.2004, 0.01))
ok("T6 loo null 10000 perms",
   t6["n_permutations"]["loo"] == 10000)
ok("T6 loo null p~0.09",
   close(t6["loo_error_null_src_pos"]["p_perm_diff_ge_obs"], 0.09, 0.01))
ok("T6 in main.tex (40320 exact enum)",
   "40320" in tex)

# ---- T7: fold-local ESM PCA (protocol cleanup) ----
# Explicit fallback: j() raises FileNotFoundError, so the previous
# `j(v2) or j(v1)` could never select v1.
try:
    t7 = j("t7_fold_local_esm_pca_v2.json")
except FileNotFoundError:
    t7 = j("t7_fold_local_esm_pca.json")
t7_repeat = j("t7_fold_local_esm_pca_v2b.json")
t7_cpu = j("t7_fold_local_esm_pca_cpu_check.json")
ok("T7 Abl1 fold-local pca20 MAE 0.2135995182",
   close(t7["abl1"]["results"]["pca_20dim"]["mae"], 0.2135995182, 5e-7))
ok("T7 Src fold-local pca20 MAE 0.4090941829",
   close(t7["src"]["results"]["pca_20dim"]["mae"], 0.4090941829, 5e-7))
ok("T7 Abl1 K3 fold-local PCA MAE 0.1477492904",
   close(t7["abl1"]["k3_pca_20dim"]["mae"], 0.1477492904, 5e-7))
ok("T7 Src K3 fold-local PCA MAE 0.3005821130",
   close(t7["src"]["k3_pca_20dim"]["mae"], 0.3005821130, 5e-7))

def active_direction(t7_system, wt_active):
    pred = t7_system["k3_pca_20dim"]["per_mutant_pred"]
    folds = t7_system["k3_pca_20dim"]["folds"]
    ok_count = total = 0
    for mutant, fold in folds.items():
        target_active = float(fold["target"][0])
        pred_active = float(pred[mutant][0])
        delta = target_active - wt_active
        if abs(delta) < 0.05:
            continue
        total += 1
        ok_count += int(math.copysign(1.0, pred_active - wt_active)
                        == math.copysign(1.0, delta))
    return f"{ok_count}/{total}"

ok("T7 K3 Active direction recomputed 5/5 and 7/7",
   active_direction(t7["abl1"], 0.88) == "5/5"
   and active_direction(t7["src"], 0.72) == "7/7")
ok("T7 artifact hash chain present",
   bool(t7.get("hashes", {}).get("script"))
   and bool(t7.get("hashes", {}).get("modules"))
   and bool(t7.get("hashes", {}).get("env"))
   and all(t7.get("hashes", {}).get("data_files", {}).values())
   and bool(t7.get("hashes", {}).get("model", {}).get("id"))
   and bool(t7.get("hashes", {}).get("model", {}).get("weights_sha256")))
ok("T7 NMR input hashes are non-empty and current",
   t7["hashes"]["data_files"]["xie2020_abl1_FINAL.json"]
       == file_sha256(ROOT / "data" / "nmr_populations" / "xie2020_abl1_FINAL.json")
   and t7["hashes"]["data_files"]["cui2025_src_kinase.json"]
       == file_sha256(ROOT / "data" / "nmr_populations" / "cui2025_src_kinase.json")
   and t7["hashes"]["data_files"]["esm2_encoding.json"]
       == file_sha256(R / "esm2_encoding.json"))
ok("T7 environment and model provenance complete",
   all(t7["hashes"]["env"].get(key) for key in
       ("python", "torch", "sklearn", "numpy", "transformers", "device"))
   and t7["hashes"]["model"]["id"] == "facebook/esm2_t33_650M_UR50D"
   and len(t7["hashes"]["model"]["weights_sha256"]) == 64)
ok("T7 independent GPU repeat agrees exactly",
   t7_repeat["abl1"]["k3_pca_20dim"]["mae"]
       == t7["abl1"]["k3_pca_20dim"]["mae"]
   and t7_repeat["src"]["k3_pca_20dim"]["mae"]
       == t7["src"]["k3_pca_20dim"]["mae"]
   and all(t7_repeat["hashes"]["data_files"].values())
   and all(t7_repeat["hashes"]["env"].get(key) for key in
           ("python", "torch", "sklearn", "numpy", "transformers", "device")))
ok("T7 CPU audit within frozen +/-0.003 system tolerance",
   abs(t7_cpu["abl1"]["k3_pca_20dim"]["mae"]
       - t7["abl1"]["k3_pca_20dim"]["mae"]) <= 0.003
   and abs(t7_cpu["src"]["k3_pca_20dim"]["mae"]
           - t7["src"]["k3_pca_20dim"]["mae"]) <= 0.003
   and all(t7_cpu["hashes"]["data_files"].values())
   and all(t7_cpu["hashes"]["env"].get(key) for key in
           ("python", "torch", "sklearn", "numpy", "transformers", "device")))
ok(f"T7 K3 measured numbers in {ms_name}", "0.1477" in tex_ms and "0.3006" in tex_ms)
tex_ms_flat = " ".join(tex_ms.split())
ok(f"T7 direction definition and denominators in {ms_name}",
    "Direction alignment is the sign of the predicted Active-state shift" in tex_ms_flat
    and "the denominator is the number of target non-ties" in tex_ms_flat
    and "reaches $5/5$ on Abl1 and $7/7$ on primary Src" in tex_ms_flat)
ok("Legacy main.tex retains the original T7 manuscript values",
    "0.1459" in tex and "0.3026" in tex
    and "positive shift alignment $5/5$ and $8/8$" in tex
    and "$0.2136$/$0.4096$" in tex)
ok("T7 fold-local in main.tex", "fold-local" in tex)

# ---- B1 independent-MSA control (0/480, in main.tex) ----
b1 = json.loads(Path(__file__).resolve().parent.parent.parent.joinpath(
    "experiments", "af2_subsample", "output_independent_msa", "results",
    "b1_comparison.json").read_text(encoding="utf-8"))
i1i2_total = sum(v.get("coverage_i1", 0) + v.get("coverage_i2", 0)
                 for k, v in b1["independent_msa_3A"].items())
ok("B1 0/480 I1/I2 (independent MSA)", i1i2_total == 0.0)
ok("B1 480 predictions", b1["n_predictions_total"] == 480)
ok("B1 in main.tex (0/480)", "0/480" in tex)

# ---- T8: AF2 closing analyses (B1 ICC + upper bound + region sensitivity) ----
t8 = json.loads(Path(__file__).resolve().parent.parent.parent.joinpath(
    "experiments", "af2_subsample", "results",
    "t8_af2_region_sensitivity.json").read_text(encoding="utf-8"))
ok("T8 B1 0/480 CP upper 0.62%",
   close(t8["b1_upper_bound"]["clopper_pearson_95_upper"], 0.0062, 0.0002))
ok("T8 B1 ICC(model)<=0.16",
   all((v["icc_model"] or 0) <= 0.16 and (v["icc_run"] or 0) <= 0.16
       for v in t8["b1_icc"].values()))
ok("T8 region n_lobe 0/840 I1/I2 (aligned region frame)",
   t8["region_frame"]["reference_region_offsets"] == {"i1": 0, "active": 19, "i2": 19}
   and t8["region_sensitivity"]["regions"]["n_lobe_act"]["i1i2_hits"] == 0)
ok("T8 region alphaC 451/840 I1",
   t8["region_sensitivity"]["regions"]["alphaC_only"]["i1i2_hits"] == 451
   and t8["region_sensitivity"]["regions"]["alphaC_only"]["state_counts"]["I1"] == 451)
ok("T8 full-protein 0/840",
   t8["region_sensitivity"]["full_protein_i1i2_hits"] == 0)
ok("T8 region sensitivity in main.tex", "Alignment-region sensitivity" in tex)

# ---- T9: reclassify from local PDBs + verify + B1 region sensitivity ----
t9 = json.loads(Path(__file__).resolve().parent.parent.parent.joinpath(
    "experiments", "af2_subsample", "results",
    "t9_reclassify_verify.json").read_text(encoding="utf-8"))
ok("T9 main 840 reclass matches stored 840/840",
   t9["main_840"]["compare_vs_stored"]["n_match"] == 840
   and t9["main_840"]["compare_vs_stored"]["n_diff"] == 0)
ok("T9 B1 480 reclass matches stored 480/480",
   t9["b1_480_full_protein"]["compare_vs_stored"]["n_match"] == 480
   and t9["b1_480_full_protein"]["compare_vs_stored"]["n_diff"] == 0)
ok("T9 B1 full-protein 0/480 I1/I2",
   t9["full_protein_b1_i1i2"] == 0)
ok("T9 B1 n_lobe 0/480 I1/I2 (aligned region frame)",
   t9["region_frame"]["reference_region_offsets"] == {"i1": 0, "active": 19, "i2": 19}
   and t9["b1_region_sensitivity"]["n_lobe_act"]["i1i2_hits"] == 0)
ok("T9 B1 alphaC 39/480 I1/I2",
   t9["b1_region_sensitivity"]["alphaC_only"]["i1i2_hits"] == 39)

# ---- P0 manuscript-improvement analyses: active manuscript (MS_TEX) ----
p0_nested = j("p2_k3_nested_pca_results.json")["systems"]
p0_nested_a = p0_nested["abl1"]
p0_nested_s = p0_nested["src"]
ok("P0-1 full nested MLP MAE 0.2625/0.3990",
   close(p0_nested_a["nested_mlp"]["mae"], 0.26254645444444447, 5e-10)
   and close(p0_nested_s["nested_mlp"]["mae"], 0.39904742025, 5e-10))
ok("P0-1 PCA selected 3/6 and 2/8",
   p0_nested_a["nested_mlp"]["selection_counts"]["pca20"] == 3
   and p0_nested_s["nested_mlp"]["selection_counts"]["pca20"] == 2)
ok("P0-1 primary nested loses to training mean",
   p0_nested_a["nested_mlp"]["mae"] > p0_nested_a["training_mean"]["mae"]
   and p0_nested_s["nested_mlp"]["mae"] > p0_nested_s["training_mean"]["mae"])
ok("P0-1 secondary model-selection MAE 0.3064/0.2581",
   close(p0_nested_a["nested_model_select"]["mae"], 0.3063989223888889, 5e-10)
   and close(p0_nested_s["nested_model_select"]["mae"], 0.25814414983333334, 5e-10))
ok("P0-1 primary and joint selectors use different scaling conventions",
   all(system["protocol"]["scaling_primary"]
           == "none (raw features; matches frozen fixed rows)"
           and system["protocol"]["scaling_secondary"]
           == "fold-local StandardScaler for all candidates"
           for system in (p0_nested_a, p0_nested_s)))

p0_stab = j("p2_k3_selector_stability.json")["systems"]
p0_stab_a = p0_stab["abl1"]["nested_mlp"]["summary"]
p0_stab_s = p0_stab["src"]["nested_mlp"]["summary"]
ok("P0-1 selector near-ties 4/6 and 5/8",
   p0_stab_a["n_near_ties"] == 4 and p0_stab_s["n_near_ties"] == 5)
ok("P0-1 selector margin medians 0.041/0.037",
   close(p0_stab_a["margin_median"], 0.041, 0.002)
   and close(p0_stab_s["margin_median"], 0.037, 0.002))
ok("P0-1 selector bootstrap stability 0.74/0.69",
   close(p0_stab_a["mean_bootstrap_p_selected_top"], 0.741, 0.01)
   and close(p0_stab_s["mean_bootstrap_p_selected_top"], 0.685, 0.01)
   and p0_stab_s["n_folds_unstable"] == 2)
ok("P0-1 selector regret vs oracle 0.165/0.194",
   close(p0_stab_a["regret_mean"], 0.165, 0.003)
   and close(p0_stab_s["regret_mean"], 0.194, 0.003))
ok(f"P0-1 selector stability in {ms_name}",
   "within-fold resampling audit" in tex_ms
   and "$0.165$" in tex_ms and "$0.194$" in tex_ms
   and "selector variance" in tex_ms)

p0_labels = j("p2_k3_src_label_sensitivity.json")["systems"]
probe = p0_labels["primary_probe"]
global_fit = p0_labels["l410a_global_fit_substitution"]
ok("P0-2 Src training means 0.2911/0.3186",
   close(probe["training_mean"]["mae"], 0.29107142857142854, 5e-10)
   and close(global_fit["training_mean"]["mae"], 0.31857142857142856, 5e-10))
ok("P0-2 Src position MLP 0.3169/0.3403",
   close(probe["fixed_k3"]["pos::LowRankCDST"]["mae"], 0.3169)
   and close(global_fit["fixed_k3"]["pos::LowRankCDST"]["mae"], 0.3403))
probe_pos_mlp_contrast = probe["fixed_k3"]["pos::LowRankCDST"]["u1_u2_contrast"]
ok("P0-2 current Src position MLP contrast 0.5207/0.6846",
   close(probe_pos_mlp_contrast["u1"], 0.5206792055330889, 5e-10)
   and close(probe_pos_mlp_contrast["u2"], 0.6845696578954912, 5e-10))
ok("P0-2 Src position CLR-GP 0.2560/0.2763",
   close(probe["fixed_k3"]["pos::CLR-GP"]["mae"], 0.2560)
   and close(global_fit["fixed_k3"]["pos::CLR-GP"]["mae"], 0.2763))
ok("P0-2 uses canonical primary and single-substitution protocols",
    p0_labels.keys() == {"primary_probe", "l410a_global_fit_substitution"}
    and global_fit["fixed_k3"]["pos::CLR-GP"]["direction"] == "7/8")
clr_robustness = j("p2_k3_src_clr_robustness.json")
clr_sensitivity = clr_robustness["pseudocount_sensitivity"]
clr_rows = clr_sensitivity["rows"]
ok("P0 Src CLR pseudocount grid is complete",
   clr_sensitivity["pseudocounts"] == [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
   and len(clr_rows) == 50)
ok("P0 Src CLR pseudocount conclusions are disclosed",
   clr_sensitivity["summary"]["mae_ranking_stable"] is False
   and clr_sensitivity["summary"]["all_rows_u2_gt_u1"] is False
   and clr_sensitivity["summary"]["f405a_vs_l410a_pattern_stable"] is True
   and sum(row["u2_gt_u1"] for row in clr_rows) == 49
   and sum(row["f405a_gt_l410a"] for row in clr_rows) == 50)
clr_interval = clr_robustness["digitization_interval_stress_test"]
ok("P0 Src bounded label stress is not called redigitization",
   clr_interval["status"] == "curator_interval_stress_test_not_independent_redigitization"
   and clr_interval["realizations"] == 200
   and close(clr_interval["proportions"]["gp_mae_le_ridge"], 0.10, 1e-12)
   and close(clr_interval["proportions"]["both_u2_gt_u1"], 0.605, 1e-12)
   and close(clr_interval["proportions"]["both_f405a_gt_l410a"], 0.42, 1e-12))
ok(f"P0-1/P0-2 results are in {ms_name}",
     all(s in tex_ms_flat for s in (
           "0.2625", "0.3990", "PCA frequency counts selected folds",
          "L410A label-substitution sensitivity", "0.3186", "$12/15\\ge0.94$",
          "Abl1 & 0.2329 & 0.1400 & 0.1477 & 0.2625 & 3/6",
          "Src & 0.2911 & 0.3442 & 0.3006 & 0.3990 & 2/8",
          "full 10-row MAE", "49/50", "60.5\\%", "independent redigitization",
          "50/50 rows (raw", "90.5\\% (raw", "13 of 15 rows overall")))
ok(f"P0-1/P0-2 stale caveats removed from {ms_name}",
   "ESM-2 PCA was not a candidate" not in tex_ms
   and "CLR-GP was not rerun" not in tex_ms
   and "contrast errors were not" not in tex_ms)
ok(f"Current Src MLP contrast and synthetic reference are in {ms_name}",
    "MLP (primary probe), $K=3$ & 0.319 & $-1.21$ & 0.484 & $-0.97$" in tex_ms_flat
    and "equal-prediction reference for a single shared prediction is $0.1475$" in tex_ms_flat
    and "not a lower bound on this two-fit LOO statistic" in tex_ms_flat
    and "12,000 generated datasets" in tex_ms_flat
    and "MLP, $K=3$ & 0.501" not in tex_ms_flat
    and "MLP, $K=3$ & 0.521" not in tex_ms_flat)
ok(f"Active {ms_name} float references and labels are clean",
   all(f"\\ref{{{label}}}" in tex_ms for label in (
       "fig:workflow", "fig:resolution", "fig:alignment", "tab:model-defs",
       "tab:unified-robustness", "tab:per-mut-errors"))
   and "\\label{app:additional}" not in tex_ms)
ok("Fixed-panel evidence terminology is consistent",
   tex_ms.count("Retrospective fixed-panel &") == 4
   and "Retrospective fixed &" not in tex_ms
   and "Fixed-panel &" not in tex_ms)
ok("Exploratory and confirmatory selectors are distinguished",
   "the candidate set, model set, and tie-break differ" in tex_ms_flat
   and "is not the confirmatory route's $0.3700$ estimate" in tex_ms_flat)

# ---- P0-3: exact paired mutation-level audit of the primary nested route ----
p0_paired = j("p2_k3_paired_exact.json")
ok("P0-3 paired artifact schema and hash chain",
   p0_paired.get("schema") == "p2_k3_paired_exact_v1"
   and bool(p0_paired.get("source_hashes", {}).get("nested_artifact_sha256"))
   and bool(p0_paired.get("source_hashes", {}).get("script_sha256")))
p0_pa = p0_paired["systems"]["abl1"]["paired"]
p0_ps = p0_paired["systems"]["src"]["paired"]
ok("P0-3 paired MAE and mutation counts",
    close(p0_pa["model_mae"], 0.26254645444444447, 5e-10)
   and close(p0_pa["training_mean_mae"], 0.23288888888888884, 5e-10)
   and p0_pa["n_mutations"] == 6
    and close(p0_ps["model_mae"], 0.39904742025, 5e-10)
   and close(p0_ps["training_mean_mae"], 0.29107142857142854, 5e-10)
   and p0_ps["n_mutations"] == 8)
ok("P0-3 conditional sign/permutation values",
    close(p0_pa["mean_delta_model_minus_mean"], 0.02965756555555558, 5e-10)
   and p0_pa["exact_sign_test"]["p_two_sided"] == 0.6875
   and p0_pa["exact_paired_permutation"]["p_two_sided"] == 0.71875
    and close(p0_ps["mean_delta_model_minus_mean"], 0.10797599167857146, 5e-10)
   and p0_ps["exact_sign_test"]["p_two_sided"] == 0.0703125
   and p0_ps["exact_paired_permutation"]["p_two_sided"] == 0.015625)
ok("P0-3 per-mutation table has 14 rows and conflict flags",
   len(p0_paired["systems"]["abl1"]["per_mutation"]) == 6
   and len(p0_paired["systems"]["src"]["per_mutation"]) == 8
   and sum(int(r["conflict"]) for r in p0_paired["systems"]["abl1"]["per_mutation"]) == 3
   and sum(int(r["conflict"]) for r in p0_paired["systems"]["src"]["per_mutation"]) == 4)
ok(f"P0-3 paired analysis is in {ms_name}",
    "Finite-panel paired sensitivity" in tex_ms
    and "Conditional sign-flip" in tex_ms
    and "overlapping-LOO" in tex_ms
   and "Per-mutation MAE" in tex_ms
   and "complete" in tex_ms
    and "Full shift vectors are released" in tex_ms
   and "tab:per-mut-errors" in tex_ms
   and "tab:model-defs" in tex_ms)

# ---- P0-4: empirical pairwise secant slopes (not global Lipschitz fits) ----
p0_slopes = j("p2c_empirical_local_slopes.json")
slopes = p0_slopes["models"]
ok("P0-4 empirical secant-slope artifact",
   p0_slopes.get("schema") == "p2c_empirical_local_slopes_v1"
   and bool(p0_slopes.get("source_hashes", {}).get("collision"))
   and bool(p0_slopes.get("source_hashes", {}).get("review"))
   and p0_slopes["exact_collision_control"]["slope"] is None)
ok("P0-4 secant slopes match frozen collision outputs",
   close(slopes["CLR-Ridge"]["secant_slope_output_l1_per_feature_l2"], 0.36298, 0.0001)
   and close(slopes["CLR-GP"]["secant_slope_output_l1_per_feature_l2"], 180.0177, 0.001)
   and close(slopes["kNN(1)"]["secant_slope_output_l1_per_feature_l2"], 180.0956, 0.001)
   and close(slopes["MLP(2 seeds)"]["secant_slope_output_l1_per_feature_l2"], 67.8419, 0.001))
ok(f"P0-4 secant-slope language is in {ms_name}",
   "Empirical secant-slope diagnostic" in tex_ms
   and "not global Lipschitz" in tex_ms
   and "0.01\\times" in tex_ms)

# ---- Frozen follow-up audits: active manuscript (MS_TEX) ----
p2_k3 = j("p2_k3_nested_results.json")["systems"]
p2_a = p2_k3["abl1"]
p2_s = p2_k3["src"]

ok("P2 Abl1 nested 0.445113476",
    close(p2_a["metrics"]["mae"], 0.44511347594444445, 5e-10))
ok("P2 Src nested 0.370023877",
    close(p2_s["metrics"]["mae"], 0.37002387679166665, 5e-10))
ok("P2 directions 4/5 and 6/7",
   p2_a["metrics"]["direction"] == "4/5"
   and p2_s["metrics"]["direction"] == "6/7")
ok("P2 marker controls 0.264898/0.255978",
    close(p2_a["analyses"]["marker_ctl"]["nested_mae"], 0.2648979437222222, 5e-10)
    and close(p2_s["analyses"]["marker_ctl"]["nested_mae"], 0.2559780198333333, 5e-10))
ok("P2 catastrophic folds candidate/control 3/2 and 1/1",
   len(p2_a["metrics"]["catastrophic_folds"]) == 3
   and len(p2_a["analyses"]["marker_ctl"]["catastrophic_folds"]) == 2
   and len(p2_s["metrics"]["catastrophic_folds"]) == 1
   and len(p2_s["analyses"]["marker_ctl"]["catastrophic_folds"]) == 1)

p2_a_lso = p2_a["metrics"]["leave_site_out"]
p2_s_lso = p2_s["metrics"]["leave_site_out"]
ok("P2 Abl1 same-route LSO exact",
   close(p2_a_lso["F382_family"]["group_mae"], 0.26724303855555553, 5e-10)
   and close(p2_a_lso["F382_family"]["comparator"], 0.27185185185185184, 5e-10)
    and close(p2_a_lso["290_301"]["group_mae"], 0.3323498596666667, 5e-10)
   and close(p2_a_lso["290_301"]["comparator"], 0.14296296296296296, 5e-10))
ok("P2 Src same-route LSO exact",
    close(p2_s_lso["N_lobe"]["group_mae"], 0.35370970780000005, 5e-10)
   and close(p2_s_lso["N_lobe"]["comparator"], 0.33466666666666667, 5e-10)
    and close(p2_s_lso["C_lobe"]["group_mae"], 0.4332842592222222, 5e-10)
   and close(p2_s_lso["C_lobe"]["comparator"], 0.3591111111111111, 5e-10))
ok("P2 alternative L410A 0.268931697",
    close(p2_s["metrics"]["alternative_label_sensitivity"]["alt_nested_mae"],
          0.2689316972916666, 5e-10))
ok("P2 hard verdict NO_GO in both systems",
   p2_a["hard_gates"]["verdict"] == "NO_GO"
   and p2_s["hard_gates"]["verdict"] == "NO_GO")

unified = j("p2_k3_unified_robustness.json")
unified_routes = unified["routes"]
unified_repr = unified_routes["representation_selection_audit"]["systems"]
unified_conf = unified_routes["candidate_model_confirmatory"]["systems"]
ok("Unified robustness schema, protocols, and source hashes",
   unified["schema_version"] == "p2_k3_unified_robustness_v1"
   and unified["protocol"] == {
       "biological_unit": "held-out mutation",
       "descriptive_deletion_refits_model": False,
       "leave_site_out_refits_confirmatory_route": True,
       "routes_must_remain_separate": True,
   }
   and unified["source_hashes"]["p2_k3_paired_exact_sha256"]
       == file_sha256(R / "p2_k3_paired_exact.json")
   and unified["source_hashes"]["p2_k3_nested_results_sha256"]
       == file_sha256(R / "p2_k3_nested_results.json"))
ok("Unified frozen-deletion robustness values",
   close(unified_repr["abl1"]["all_mutation_mae"], 0.2625464544444444, 5e-10)
   and close(unified_repr["abl1"]["leave_one_observation_out"]["minimum_mae"],
             0.19406584953333333, 5e-10)
   and close(unified_repr["abl1"]["leave_one_observation_out"]["maximum_mae"],
             0.3045419476666667, 5e-10)
   and close(unified_repr["abl1"]["double_mutant_exclusion"]["mae"],
             0.2726620573333333, 5e-10)
   and close(unified_repr["src"]["all_mutation_mae"], 0.39904742025, 5e-10)
   and close(unified_repr["src"]["leave_one_observation_out"]["minimum_mae"],
             0.37533716895238095, 5e-10)
   and close(unified_repr["src"]["leave_one_observation_out"]["maximum_mae"],
             0.4150331301904762, 5e-10)
   and close(unified_repr["src"]["double_mutant_exclusion"]["mae"],
             0.41046109823809523, 5e-10))
ok("Unified confirmatory robustness values",
   close(unified_conf["abl1"]["all_mutation_mae"], 0.44511347594444445, 5e-10)
   and close(unified_conf["abl1"]["leave_one_observation_out"]["minimum_mae"],
             0.4298732, 5e-10)
   and close(unified_conf["abl1"]["leave_one_observation_out"]["maximum_mae"],
             0.4888806, 5e-10)
   and close(unified_conf["abl1"]["double_mutant_exclusion"]["mae"],
             0.4489346, 5e-10)
   and close(unified_conf["src"]["all_mutation_mae"], 0.37002387679166665, 5e-10)
   and close(unified_conf["src"]["leave_one_observation_out"]["minimum_mae"],
             0.3309654285714286, 5e-10)
   and close(unified_conf["src"]["leave_one_observation_out"]["maximum_mae"],
             0.4227814285714286, 5e-10)
   and close(unified_conf["src"]["double_mutant_exclusion"]["mae"],
             0.36982585714285715, 5e-10))
ok(f"Unified robustness table is in {ms_name}",
    all(token in tex_ms_flat for token in (
        ".1941--.3045", ".3753--.4150", ".4299--.4889", ".3310--.4228",
        "without refitting", "the two routes are not pooled")))

fixed_a = min(v["mae"] for v in p2_a["analyses"]["fixed_loo"].values())
fixed_s = min(v["mae"] for v in p2_s["analyses"]["fixed_loo"].values())
ok("P2 fixed minima remain exploratory 0.1556/0.2560",
    close(fixed_a, 0.15564517242543016, 5e-10)
    and close(fixed_s, 0.25597586657460397, 5e-10))

foldx_r = ROOT / "experiments" / "foldx_src" / "results"
foldx_eval = json.loads((foldx_r / "foldx_clr_eval.json").read_text(encoding="utf-8"))
foldx_a_qc = json.loads((foldx_r / "foldx_abl1_qc.json").read_text(encoding="utf-8"))
foldx_s_qc = json.loads((foldx_r / "foldx_src_qc.json").read_text(encoding="utf-8"))
foldx_systems = foldx_eval["per_system"]
ok("FoldX Abl1 best fixed track1/track2",
   close(foldx_systems["abl1"]["track1"]["fixed_loo"]["FX_ddg::CLR-Ridge"]["mae"],
         0.19453120520758957, 5e-10)
   and close(foldx_systems["abl1"]["track2"]["fixed_loo"]["FX_ddg::CLR-Ridge"]["mae"],
             0.1827946666402258, 5e-10))
ok("FoldX Src best fixed 0.256030219",
   close(foldx_systems["src"]["src"]["fixed_loo"]["FX_ddg_llr::CLR-GP"]["mae"],
         0.25603021930009867, 5e-10))
ok("FoldX Abl1 coverage is 40 non-degenerate plus 2 degenerate",
   foldx_a_qc["cells_total"] == 42
   and foldx_a_qc["cells_ok"] == 42
   and foldx_a_qc["cells_non_degenerate"] == 40
   and foldx_a_qc["cells_degenerate"] == 2
   and foldx_a_qc["final_protocol_success"] is True
   and foldx_a_qc["final_5of5_success"] is False
   and foldx_a_qc["n_runs_per_non_degenerate_cell_min_5"] is True)
ok("FoldX Src coverage 24 cells x5",
   foldx_s_qc["cells_total"] == 24
   and foldx_s_qc["cells_ok"] == 24
   and foldx_s_qc["final_5of5_success"] is True
   and foldx_s_qc["n_runs_per_cell_min_5"] is True)
ok("FoldX frozen QC values",
   close(foldx_a_qc["state_energy_ordering_consistency"], 0.5714285714285714, 5e-10)
   and close(foldx_s_qc["state_energy_ordering_consistency"], 0.3125, 5e-10)
   and close(foldx_a_qc["repeat_perturbation_mae"], 0.09556272800000064, 5e-10)
   and close(foldx_s_qc["repeat_perturbation_mae"], 0.23961533333333304, 5e-10)
   and close(foldx_a_qc["spearman_foldx_vs_exp_ddg_abs"]["n6_all"]["rho"],
             0.6571428571428573, 5e-10))

ok(f"FoldX independent baseline is in {ms_name}",
   "FoldX mutation free energies" in tex_ms
   and "$0.1945$" in tex_ms and "$0.1828$" in tex_ms and "$0.2560$" in tex_ms
   and "$0.5714$/$0.3125$" in tex_ms
   and "$0.0956$/$0.2396$" in tex_ms)

p5_summary = j("p5_candidate_heterogeneity.json")["summary"]
p5_n6 = [r["selection_optimism_mean"] for r in p5_summary if r["n"] == 6]
ok("P5 candidate-quality spread optimism at n=6",
   p5_n6 and 0.02 <= min(p5_n6) and max(p5_n6) <= 0.05)
ok(f"P5 calibrated optimism is in {ms_name}",
   "$0.028$--$0.040$" in tex_ms and "$0.024$--$0.027$" in tex_ms
   and "$0.018$" in tex_ms)

canonical = j("canonical_reproduction.json")
ok("Canonical provenance notes clean", canonical["provenance_notes"] == ["ok"])
ok("Canonical reproduction 24/24 PASS",
   len(canonical["items"]) == 24
   and all(item["status"] == "PASS" for item in canonical["items"]))

ok(f"Core robustness claims are in {ms_name}",
       "fixed-to-selected gap is substantive" in tex_ms_flat
       and "not a complete second global-fit panel" in tex_ms_flat
       and "all ten CLR rows" in tex_ms_flat
       and "Reproducibility controls" in tex_ms_flat
       and "Two independent five-seed ESM-2 executions" in tex_ms_flat
       and "GP hyperparameters fitted on training folds only" in tex_ms_flat
       and "not directly comparable numerically with the fixed-MLP or nested-MLP rows" in tex_ms_flat)
ok(f"Core robustness numbers are in {ms_name}",
    all(s in tex_ms_flat for s in (
         "0.2581", "0.1477", "0.3006", "0.2625", "0.3990", "\\pm0.003")))
ok("Frozen follow-up text did not leak into legacy main.tex",
    "Selection and protocol robustness" not in tex
    and "protocol-compliant follow-up" not in tex
    and "0.1477" not in tex
    and "0.3006" not in tex
    and "$0.2136$/$0.4091$" not in tex)

# ---- Dynamic-range normalization and LLR protocol labels: active manuscript ----
p3_norm = j("p3_dynamic_range_normalization.json")
p3_a = p3_norm["systems"]["abl1"]
p3_s = p3_norm["systems"]["src"]
ok("P3 dynamic-range artifact schema and source hashes",
   p3_norm["schema_version"] == "p3-dynamic-range-normalization-v1"
   and all(p3_norm["source_hashes"].values()))
ok("P3 constant-WT scales 0.3878/0.4600",
   close(p3_a["raw_mae"]["constant_wt"], 0.3878)
   and close(p3_s["raw_mae"]["constant_wt"], 0.4600))
ok("P3 per-state ranges are frozen",
   all(close(a, b) for a, b in zip(p3_a["state_ranges"], [0.83, 0.10, 0.89]))
   and all(close(a, b) for a, b in zip(p3_s["state_ranges"], [0.73, 0.84, 0.84])))
ok("P3 normalized MAEs match manuscript values",
   close(p3_a["normalized_mae"]["loo_training_mean"], 0.6006)
    and close(p3_a["normalized_mae"]["fixed_llr_plus_position_fold_local"], 0.3611)
   and close(p3_a["normalized_mae"]["fixed_pca_gpu_canonical"], 0.3810)
    and close(p3_a["normalized_mae"]["full_nested_mlp"], 0.6771)
   and close(p3_s["normalized_mae"]["loo_training_mean"], 0.6328)
    and close(p3_s["normalized_mae"]["fixed_llr_plus_position_fold_local"], 0.7484)
   and close(p3_s["normalized_mae"]["fixed_pca_gpu_canonical"], 0.6534)
    and close(p3_s["normalized_mae"]["full_nested_mlp"], 0.8675)
   and "tab:normalized-errors" not in tex_ms)
ok(f"LLR global and fold-local protocols are distinguished in {ms_name}",
   "LLR$+$position (legacy scale)" in tex_ms
   and "Fixed LLR$+$pos" in tex_ms
   and "protocols are not interchangeable" in tex_ms
   and "$0.1629$ row" in tex_ms)

# ---- P4 frozen controlled support/resolution/selection experiment ----
p4 = j("p4_support_resolution_selection.json")
p4_manifest = j("p4_support_resolution_selection_manifest.json")
p4_summary = p4["summary"]

def p4_rows(**conditions):
    return [row for row in p4_summary
            if all(row[key] == value for key, value in conditions.items())]

def p4_mean(field, **conditions):
    rows = p4_rows(**conditions)
    return sum(row[field] for row in rows) / len(rows)

ok("P4 complete frozen factorial 72,000/360/200",
   p4_manifest["records"] == len(p4["records"]) == 72000
   and p4_manifest["settings"] == len(p4_summary) == 360
   and p4_manifest["repeats_per_setting"] == 200
   and p4_manifest["all_settings_retained"] is True
   and all(row["repeats"] == 200 for row in p4_summary))
ok("P4 figure hashes match current regenerated outputs",
   p4_manifest["figure_pdf_sha256"] == file_sha256(ROOT / "paper" / "figures_v2" / "fig4_synthetic_framework.pdf")
   and p4_manifest["figure_png_sha256"] == file_sha256(ROOT / "paper" / "figures_v2" / "fig4_synthetic_framework.png"))
ok("P4 simplex and nested-fold isolation",
   all(row["simplex_ok"] and row["nested_outer_fold_isolation"]
        for row in p4["records"]))
ok("P4 figure slices are explicit",
   p4["figure_slices"]["panel_A"]["fixed"]
       == {"n": 20, "m": 1, "resolution": "full_k3"}
   and p4["figure_slices"]["panel_B"]["fixed"] == {"n": 20, "m": 1}
   and p4["figure_slices"]["panel_B"]["averaged_over"] == ["epsilon"]
   and p4["figure_slices"]["panel_C"]["complete_factorial"] is True)

p4_collision = p4_rows(n=20, m=1, delta=1.2, epsilon=0.0,
                       resolution="full_k3")[0]
p4_separated = p4_rows(n=20, m=1, delta=1.2, epsilon=1.0,
                       resolution="full_k3")[0]
ok("P4 exact-collision and separated pair values",
   close(p4_collision["pair_clr_ridge_mae_mean"], 0.17489633163805834, 5e-10)
   and close(p4_collision["equal_prediction_pair_floor_mean"], 0.1475220213, 5e-10)
   and close(p4_separated["pair_clr_ridge_mae_mean"], 0.11727705899851619, 5e-10))
ok("P4 fine and shared contrast values",
   close(p4_mean("fine_contrast_clr_mae_mean", n=20, m=1, delta=0.0,
                 resolution="full_k3"), 0.0, 5e-10)
   and close(p4_mean("fine_contrast_clr_mae_mean", n=20, m=1, delta=1.2,
                      resolution="full_k3"), 0.05141263213491347, 5e-10)
   and close(p4_mean("shared_contrast_clr_mae_mean", n=20, m=1, delta=1.2,
                      resolution="full_k3"), 0.026350733050382873, 5e-10)
   and close(p4_mean("shared_contrast_clr_mae_mean", n=20, m=1, delta=1.2,
                      resolution="pooled_k2"), 0.026692759295392405, 5e-10))
ok("P4 candidate-count optimism values",
   close(p4_mean("selection_optimism_mean", m=5, resolution="full_k3"),
          0.0011067465250350482, 5e-10)
   and close(p4_mean("selection_optimism_mean", m=20, resolution="full_k3"),
              0.0033218984193121464, 5e-10)
   and close(p4_mean("selection_optimism_mean", m=5, resolution="pooled_k2"),
              0.0011404718597540354, 5e-10)
   and close(p4_mean("selection_optimism_mean", m=20, resolution="pooled_k2"),
              0.004151245698406193, 5e-10))
ok(f"P4 frozen claims are in {ms_name}",
   all(token in tex_ms for token in (
       "72,000", "$0.175$", "$0.148$", "$0.117$", "$0.051$",
        "$0.026$", "$0.027$", "$0.0033$", "$0.0042$")))

# ---- P6 audit detection benchmark (A1; derived from frozen P4/P5) ----
p6 = j("p6_audit_detection_benchmark.json")
p6_s2 = {tuple((r["n"], r["epsilon"], r["delta"])[:3]): r
         for r in p6["S2"] if r["epsilon"] == 0.0}
p6_r = [r for r in p6["R"] if abs(r["margin"] - math.sqrt(3.0)) < 1e-9]
p6_t = [r for r in p6["T"] if r["source"] == "P4" and r["tau_sel"] == 0.01]
ok("P6 regeneration cross-check is exact",
   p6["cross_check"]["repeats_verified"] == 12000
   and p6["cross_check"]["mismatches"] == 0)
ok("P6 S2 null-power values match report",
   close(p6_s2[(8, 0.0, 1.2)]["detection_rate"], 0.085, 0.005)
   and close(p6_s2[(6, 0.0, 1.2)]["detection_rate"], 0.0, 0.005)
   and close(p6_s2[(50, 0.0, 1.2)]["detection_rate"], 0.0, 0.005))
ok("P6 R contrast-detector values match report",
   close(next(r["detection_rate"] for r in p6_r
               if r["n"] == 6 and r["delta"] == 0.6), 0.93, 0.01)
   and close(next(r["detection_rate"] for r in p6_r
                   if r["n"] == 8 and r["delta"] == 1.2), 1.0, 0.01)
   and close(next(r["detection_rate"] for r in p6_r
                   if r["n"] == 20 and r["delta"] == 1.2), 0.525, 0.01)
   and close(next(r["detection_rate"] for r in p6_r
                   if r["n"] == 50 and r["delta"] == 1.2), 0.0, 0.01))
ok("P6 T optimism-flag noise values match report",
   close(next(r["detection_rate"] for r in p6_t
               if r["n"] == 6 and r["mechanism"] == 20), 0.372, 0.01)
   and close(next(r["detection_rate"] for r in p6_t
                   if r["n"] == 8 and r["mechanism"] == 20), 0.033, 0.01))
ok(f"P6 claims are in {ms_name}",
   all(token in tex_ms_flat for token in (
       "Audit detection benchmark", "$q_2>q_1$ detection rate",
       "calibrate the audit", "nomological network",
       "$0.93$--$1.0$", "$0.53$ at $n=20$", "$37\\%$")))

# ---- AF2 threshold/reference/ambiguity calibration ----
af2_calibration = json.loads((ROOT / "experiments" / "af2_subsample" / "results"
                              / "assignment_calibration.json").read_text(encoding="utf-8"))
af2_refs = af2_calibration["reference_calibration"]
af2_protocols = af2_calibration["protocols"]
ok("AF2 assignment calibration protocol and frozen reproduction",
   af2_calibration["schema_version"] == "af2_assignment_calibration_v1"
   and af2_calibration["method"]["thresholds_angstrom"]
       == [round(2.0 + 0.25 * i, 2) for i in range(13)]
   and af2_calibration["ambiguity_rule"]["margin_cutoff_angstrom"] == 0.5
   and af2_protocols["original"]["frozen_3A_full_protein_consistency"]["matching_records"] == 840
   and af2_protocols["original"]["frozen_3A_full_protein_consistency"]["mismatch_count"] == 0
   and af2_protocols["fresh_msa"]["frozen_3A_full_protein_consistency"]["matching_records"] == 480
   and af2_protocols["fresh_msa"]["frozen_3A_full_protein_consistency"]["mismatch_count"] == 0)
ok("AF2 reference geometry and ambiguity values",
   close(af2_refs["full_protein"]["pairwise_distances"]["active__I2"]["rmsd_angstrom"],
         8.893687880114713, 5e-10)
   and close(af2_refs["n_lobe_act"]["pairwise_distances"]["active__I2"]["rmsd_angstrom"],
             7.326877582309726, 5e-10)
   and close(af2_refs["alphaC_only"]["pairwise_distances"]["active__I2"]["rmsd_angstrom"],
             2.564432345596726, 5e-10)
   and close(af2_refs["alphaC_only"]["pairwise_distances"]["active__I1"]["rmsd_angstrom"],
             1.5686527987774919, 5e-10)
   and af2_protocols["original"]["regions"]["n_lobe_act"]
       ["ambiguity_at_frozen_threshold"]["counts"]["ambiguous"] == 0
   and af2_protocols["fresh_msa"]["regions"]["n_lobe_act"]
       ["ambiguity_at_frozen_threshold"]["counts"]["ambiguous"] == 0
   and all(token in tex_ms for token in (
       "$8.89$", "$7.33$", "$2.56$", "$451/840$", "$39/480$", "$1.57$")))

af2_align = json.loads((ROOT / "experiments" / "af2_subsample" / "results"
                        / "af2_alignment_mode_audit.json").read_text(encoding="utf-8"))
af2_align_p = af2_align["protocols"]
ok("AF2 alignment-mode audit: offset everywhere, 0 fallback",
   af2_align_p["original"]["n_structures"] == 840
   and af2_align_p["fresh_msa"]["n_structures"] == 480
   and af2_align_p["original"]["alignment_mode_counts"].get("fallback", 0) == 0
   and af2_align_p["fresh_msa"]["alignment_mode_counts"].get("fallback", 0) == 0
   and af2_align_p["original"]["alignment_mode_counts"]["offset"] == 2520
   and af2_align_p["fresh_msa"]["alignment_mode_counts"]["offset"] == 1440
   and af2_align_p["original"]["frozen_assignment_matches"] == 840
   and af2_align_p["fresh_msa"]["frozen_assignment_matches"] == 480)
ok(f"AF2 alignment-mode audit is in {ms_name}",
   "alignment-mode audit" in tex_ms and "none trigger the fallback" in tex_ms)

print("\nALL CHECKS PASSED")
