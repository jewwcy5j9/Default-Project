#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B1 分析: 独立 MSA 结果 vs Option B 原结果对比 + 预注册判定

流程:
  1. 调用 classify_states.py 对 output_independent_msa/output 分类 (3.0 Å 冻结口径)
  2. 阈值敏感性: 2.5 Å / 3.5 Å 重新分类 (零 GPU 成本, 预注册 §3.3)
  3. 计算 coverage + pLDDT, 与 experiments/af2_subsample/results/coverage_results.json 对比
  4. 按 b1_independent_msa_preregistration.md §3 判定规则输出结论

产物:
  output_independent_msa/results/state_classifications.json (3.0 Å)
  output_independent_msa/results_sens_25/state_classifications.json
  output_independent_msa/results_sens_35/state_classifications.json
  output_independent_msa/results/b1_comparison.json
  output_independent_msa/results/b1_comparison_report.md
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
OUT_ROOT = PROJECT_ROOT / "experiments" / "af2_subsample" / "output_independent_msa"
PDB_OUT_DIR = OUT_ROOT / "output"
RESULTS_DIR = OUT_ROOT / "results"
ORIGINAL_RESULTS = SCRIPT_DIR / "results" / "coverage_results.json"

B1_MUTANTS = ["WT", "L301I", "M290L_L301I", "F382V"]
THRESHOLDS = [3.0, 2.5, 3.5]  # 3.0 = 冻结口径; 2.5/3.5 = 敏感性


# ============================================================
# 分类 (复用预注册分类器)
# ============================================================

def run_classify(threshold: float, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SCRIPT_DIR / "classify_states.py"),
        "--output-dir", str(PDB_OUT_DIR),
        "--results-dir", str(results_dir),
        "--threshold", str(threshold),
    ]
    print(f"[classify] {threshold:.1f}A -> {results_dir.name}")
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"classify_states.py failed at {threshold:.1f}A")
    return results_dir / "state_classifications.json"


# ============================================================
# 统计
# ============================================================

def coverage_from_classifications(cls_path: Path):
    """从 state_classifications.json 计算 per-mutant coverage."""
    data = json.loads(cls_path.read_text(encoding="utf-8"))
    per_mutant = {}
    for e in data["classifications"]:
        m = e["mutant"]
        d = per_mutant.setdefault(m, Counter())
        d[e["state"]] += 1
    out = {}
    for m, c in per_mutant.items():
        n = sum(c.values())
        out[m] = {
            "N_total": n,
            "N_active": c.get("active", 0),
            "N_I1": c.get("I1", 0),
            "N_I2": c.get("I2", 0),
            "N_unclass": c.get("unclassified", 0),
            "coverage_active": c.get("active", 0) / n if n else 0.0,
            "coverage_i1": c.get("I1", 0) / n if n else 0.0,
            "coverage_i2": c.get("I2", 0) / n if n else 0.0,
            "coverage_unclass": c.get("unclassified", 0) / n if n else 0.0,
        }
    return out


def mean_plddt_per_mutant():
    """从预测 PDB 的 B-factor 列 (CA 原子) 提取平均 pLDDT."""
    out = {}
    for mutant in B1_MUTANTS:
        mut_dir = PDB_OUT_DIR / mutant
        if not mut_dir.exists():
            continue
        totals = []
        for pdb in sorted(mut_dir.rglob("*.pdb")):
            bfactors = []
            for line in pdb.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    try:
                        bfactors.append(float(line[60:66]))
                    except ValueError:
                        pass
            if bfactors:
                totals.append(sum(bfactors) / len(bfactors))
        if totals:
            out[mutant] = {
                "mean": float(sum(totals) / len(totals)),
                "n_structures": len(totals),
            }
    return out


# ============================================================
# 判定 (预注册 §3)
# ============================================================

def verdict(new_cov, orig_cov):
    """返回 (primary, secondary, details)."""
    i1i2 = {m: new_cov[m]["coverage_i1"] + new_cov[m]["coverage_i2"]
            for m in new_cov}
    max_i1i2 = max(i1i2.values())
    if max_i1i2 <= 0.02:
        primary = "ROBUST"
    elif max_i1i2 > 0.05:
        primary = "QUALIFY_REQUIRED"
    else:
        primary = "BORDERLINE"

    d_unclass = {}
    for m in new_cov:
        if m in orig_cov:
            d_unclass[m] = new_cov[m]["coverage_unclass"] - orig_cov[m]["coverage_unclass"]
    secondary_ok = all(abs(v) <= 0.08 for v in d_unclass.values()) if d_unclass else True

    return primary, secondary_ok, {"i1i2": i1i2, "max_i1i2": max_i1i2,
                                   "d_unclass": d_unclass}


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("B1 分析: 独立 MSA vs Option B 对比")
    print("=" * 70)

    # 1. 分类
    cls_paths = {}
    for th in THRESHOLDS:
        sub = "results" if th == 3.0 else f"results_sens_{int(th*10)}"
        cls_paths[th] = run_classify(th, OUT_ROOT / sub)

    new_cov = coverage_from_classifications(cls_paths[3.0])
    sens = {th: coverage_from_classifications(p) for th, p in cls_paths.items()}

    # 2. 原结果
    orig = json.loads(ORIGINAL_RESULTS.read_text(encoding="utf-8"))["per_mutant"]

    # 3. pLDDT
    plddt = mean_plddt_per_mutant()

    # 4. 判定
    primary, secondary_ok, det = verdict(new_cov, orig)

    # 5. 保存 JSON
    out_json = RESULTS_DIR / "b1_comparison.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "b1_independent_msa",
        "n_predictions_total": sum(v["N_total"] for v in new_cov.values()),
        "original": {m: orig[m] for m in B1_MUTANTS if m in orig},
        "independent_msa_3A": new_cov,
        "threshold_sensitivity": sens,
        "plddt": plddt,
        "verdict": {
            "primary": primary,
            "secondary_intermediate_consistent": secondary_ok,
            "max_i1i2_3A": det["max_i1i2"],
            "i1i2_per_mutant_3A": det["i1i2"],
            "d_unclassified_vs_original": det["d_unclass"],
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[json] {out_json}")

    # 6. 报告
    L = []
    L.append("# B1 对比报告: 独立 MSA vs Option B")
    L.append("")
    L.append("## 1. Coverage 对比 (3.0 Å 冻结口径)")
    L.append("")
    L.append("| Mutant | N | orig active | new active | orig I1+I2 | new I1+I2 | "
             "orig unclass | new unclass | Δunclass |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in B1_MUTANTS:
        if m not in new_cov:
            L.append(f"| {m} | 缺失 (检查 output_independent_msa) |")
            continue
        o = orig.get(m, {})
        n = new_cov[m]
        d = (n["coverage_unclass"] - o["coverage_unclass"]) if "coverage_unclass" in o else None
        L.append(f"| {m} | {n['N_total']} | "
                 f"{o.get('coverage_active', float('nan')):.3f} | {n['coverage_active']:.3f} | "
                 f"{o.get('coverage_i1', 0)+o.get('coverage_i2', 0):.3f} | "
                 f"{n['coverage_i1']+n['coverage_i2']:.3f} | "
                 f"{o.get('coverage_unclass', float('nan')):.3f} | {n['coverage_unclass']:.3f} | "
                 f"{(f'{d:+.3f}' if d is not None else '--')} |")
    L.append("")
    L.append("## 2. pLDDT (新预测)")
    L.append("")
    L.append("| Mutant | mean pLDDT | n |")
    L.append("|---|---:|---:|")
    for m, v in plddt.items():
        L.append(f"| {m} | {v['mean']:.1f} | {v['n_structures']} |")
    L.append("")
    L.append("## 3. 阈值敏感性 (I1+I2 coverage)")
    L.append("")
    L.append("| Mutant | 2.5Å | 3.0Å | 3.5Å |")
    L.append("|---|---:|---:|---:|")
    for m in B1_MUTANTS:
        if m not in new_cov:
            continue
        L.append(f"| {m} | "
                 f"{sens[2.5].get(m, {}).get('coverage_i1', 0)+sens[2.5].get(m, {}).get('coverage_i2', 0):.3f} | "
                 f"{sens[3.0].get(m, {}).get('coverage_i1', 0)+sens[3.0].get(m, {}).get('coverage_i2', 0):.3f} | "
                 f"{sens[3.5].get(m, {}).get('coverage_i1', 0)+sens[3.5].get(m, {}).get('coverage_i2', 0):.3f} |")
    L.append("")
    L.append("## 4. 判定 (预注册 §3)")
    L.append("")
    L.append(f"- **主问题 (I1/I2 命中)**: {primary} "
             f"(max I1+I2 = {det['max_i1i2']*100:.1f}%)")
    L.append(f"- **次问题 (中间态一致性)**: "
             f"{'一致' if secondary_ok else '不一致'} "
             f"(|Δunclass| ≤ 8pp 判定)")
    L.append("")
    L.append("### 论文动作")
    L.append("")
    if primary == "ROBUST":
        L.append("- 结论稳健: 0/840 I1/I2 主张不受 MSA 构建方式影响。"
                 "可选: 附录 B 加一句鲁棒性说明。正文数值不变。")
    elif primary == "QUALIFY_REQUIRED":
        L.append("- **投稿前必须改写**: 正文 0/840 表述需限定为"
                 " 'WT 派生 MSA 下 0/840' 或按 B1 结果调整; "
                 "18-23% 中间态表述同步检查。")
    else:
        L.append("- 边界情形: 正文保持原表述, 但附录/回复材料需报告双侧数值。")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"*脚本: analyze_independent_msa.py | 预注册: "
             f"b1_independent_msa_preregistration.md | 数据: {out_json.name}*")

    out_md = RESULTS_DIR / "b1_comparison_report.md"
    out_md.write_text("\n".join(L), encoding="utf-8")
    print(f"[md]   {out_md}")


if __name__ == "__main__":
    main()
