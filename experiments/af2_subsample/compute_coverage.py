#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_coverage.py — AF2 系综 Coverage 与准确率计算 + 最终报告（预注册 §5–§6）

读取 classify_states.py 的输出 state_classifications.json，计算：
  1. 各突变体的 coverage（active / I1 / I2 / unclassified 布居分数，§5.1）
  2. 与 Xie 2020 实验布居对比（§5.2）
  3. 单突变体准确率与全局准确率（§5.3，仅在 6 突变体上取均值，不含 WT）
  4. 95% 置信区间（bootstrap 1000 次重采样，样本量小时标注）
  5. 叙事分支判定（§6：≥80% 成功 / 40–80% 部分成功 / <40% 失败）

输出：
  experiments/af2_subsample/results/coverage_results.json   （机器可读）
  experiments/af2_subsample/results/af2_subsample_report.md  （人类可读报告）

用法：
  python compute_coverage.py
  python compute_coverage.py --classifications path/to/state_classifications.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ============================================================
# 预注册冻结参数（§5–§6，不可事后调整）
# ============================================================

# 6 个突变体（不含 WT）——全局准确率仅在此 6 个上取均值（§5.3，冻结）
MUTANTS_6 = ['M290L', 'L301I', 'M290L_L301I', 'F382L', 'F382Y', 'F382V']

# 全部 7 个序列（WT + 6 突变体）
MUTANTS_ALL = ['WT'] + MUTANTS_6

# 实验布居来源（§5.2 / §8）
DEFAULT_EXP_DATA = 'data/nmr_populations/xie2020_abl1_FINAL.json'

# 默认输入输出路径
DEFAULT_CLASSIFICATIONS = 'experiments/af2_subsample/results/state_classifications.json'
DEFAULT_RESULTS_DIR = 'experiments/af2_subsample/results'

# 叙事分支阈值（§6，冻结）
SUCCESS_THRESHOLD = 0.80   # Global Accuracy ≥ 80% → 成功档
PARTIAL_THRESHOLD = 0.40   # 40% ≤ GA < 80% → 部分成功档；< 40% → 失败档

# MdS2024 对标基准
MDS2024_BENCHMARK = 0.80   # MdS2024 报告 Abl1 同靶点 >80% 全局准确率

# Bootstrap 参数
N_BOOTSTRAP = 1000         # 重采样次数
BOOTSTRAP_SEED = 20260730  # 随机种子（可复现）
SMALL_SAMPLE_THRESHOLD = 100  # 样本量 < 此值视为"小样本"，需标注 CI

# 突变体→确定性种子的偏移量（避免 Python hash 随机化导致不可复现）
_MUTANT_SEED_OFFSET = {mut: i * 7919 for i, mut in enumerate(MUTANTS_ALL)}

# 项目根目录（本脚本位于 experiments/af2_subsample/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_path(path_str):
    """将相对路径解析为基于项目根目录的绝对路径。"""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


# ============================================================
# 数据加载
# ============================================================

def load_classifications(path):
    """加载 state_classifications.json，返回分类列表。

    兼容两种格式：
      - 纯列表 [{...}, ...]（旧格式）
      - {classifications: [...], ...}（新格式，含元数据）
    """
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f'分类结果文件不存在: {path}\n'
            f'请先运行 classify_states.py 生成该文件。'
        )
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and 'classifications' in data:
        return data['classifications']
    raise ValueError(f'无法识别的分类结果格式: {path}')


def load_experimental_populations(path):
    """加载 Xie 2020 实验布居数据，返回 {mutant: P_exp(active)} 字典。

    数据来源：data/nmr_populations/xie2020_abl1_FINAL.json（gold-tier 13C CEST）
    仅提取 gold-tier 突变体的 Active 布居。
    """
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f'实验布居数据不存在: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pops = {}
    for mutant, info in data['populations'].items():
        active = info.get('Active')
        if active is not None:
            pops[mutant] = float(active)
    return pops


# ============================================================
# Coverage 计算（§5.1）
# ============================================================

def compute_coverage(states_list):
    """计算单个突变体的 coverage。

    参数:
        states_list: 该突变体所有预测的 state 标签列表

    返回:
        dict: N_total, N_active, N_I1, N_I2, N_unclass,
              coverage_active, coverage_i1, coverage_i2, coverage_unclass
    """
    n_total = len(states_list)
    if n_total == 0:
        return {
            'N_total': 0, 'N_active': 0, 'N_I1': 0, 'N_I2': 0, 'N_unclass': 0,
            'coverage_active': 0.0, 'coverage_i1': 0.0,
            'coverage_i2': 0.0, 'coverage_unclass': 0.0,
        }
    n_active = sum(1 for s in states_list if s == 'active')
    n_i1 = sum(1 for s in states_list if s == 'I1')
    n_i2 = sum(1 for s in states_list if s == 'I2')
    n_unclass = sum(1 for s in states_list if s == 'unclassified')
    return {
        'N_total': n_total,
        'N_active': n_active,
        'N_I1': n_i1,
        'N_I2': n_i2,
        'N_unclass': n_unclass,
        'coverage_active': n_active / n_total,
        'coverage_i1': n_i1 / n_total,
        'coverage_i2': n_i2 / n_total,
        'coverage_unclass': n_unclass / n_total,
    }


# ============================================================
# 准确率计算（§5.3）
# ============================================================

def compute_accuracy(coverage_active, p_exp_active):
    """计算单突变体准确率（§5.3，冻结）。

    err(m) = |P_AF2(active)(m) - P_exp(active)(m)| / P_exp(active)(m)
    acc(m) = 1 - err(m)

    注：P_exp(active)=0 时无法定义相对误差，返回 None。
    """
    if p_exp_active is None or p_exp_active == 0:
        return None, None
    err = abs(coverage_active - p_exp_active) / p_exp_active
    acc = 1.0 - err
    return acc, err


# ============================================================
# Bootstrap 95% 置信区间
# ============================================================

def bootstrap_coverage_ci(states_list, n_bootstrap=N_BOOTSTRAP,
                          seed=BOOTSTRAP_SEED):
    """对 coverage_active 做 bootstrap 95% CI。

    对预测状态列表做有放回重采样（n_bootstrap 次），每次计算 active 布居分数，
    取 2.5%–97.5% 分位数作为 95% CI。

    返回: (ci_low, ci_high) — coverage_active 的 95% CI
    """
    n = len(states_list)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    states_arr = np.array(states_list)
    boot_fracs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(states_arr, size=n, replace=True)
        boot_fracs[b] = np.sum(sample == 'active') / n
    ci_low = float(np.percentile(boot_fracs, 2.5))
    ci_high = float(np.percentile(boot_fracs, 97.5))
    return ci_low, ci_high


def bootstrap_accuracy_ci(states_list, p_exp_active,
                          n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """对单突变体准确率做 bootstrap 95% CI。

    每次重采样后计算 coverage_active → acc = 1 - |cov - P_exp|/P_exp。

    返回: (acc_ci_low, acc_ci_high)
    """
    n = len(states_list)
    if n == 0 or p_exp_active is None or p_exp_active == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    states_arr = np.array(states_list)
    boot_acc = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(states_arr, size=n, replace=True)
        cov_active = np.sum(sample == 'active') / n
        err = abs(cov_active - p_exp_active) / p_exp_active
        boot_acc[b] = 1.0 - err
    ci_low = float(np.percentile(boot_acc, 2.5))
    ci_high = float(np.percentile(boot_acc, 97.5))
    return ci_low, ci_high


def bootstrap_global_accuracy_ci(mutant_states, mutant_p_exp,
                                 n_bootstrap=N_BOOTSTRAP,
                                 seed=BOOTSTRAP_SEED):
    """对全局准确率做 bootstrap 95% CI。

    每次迭代：对每个突变体分别重采样状态、计算单突变体 acc，再取 6 突变体均值。
    仅在 6 突变体（不含 WT）上计算。

    参数:
        mutant_states: {mutant: states_list}
        mutant_p_exp: {mutant: P_exp(active)}

    返回: (ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    mutants = [m for m in MUTANTS_6 if m in mutant_states and m in mutant_p_exp]
    if not mutants:
        return (None, None)

    boot_global = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        accs = []
        for m in mutants:
            states_arr = np.array(mutant_states[m])
            n = len(states_arr)
            if n == 0:
                continue
            sample = rng.choice(states_arr, size=n, replace=True)
            cov_active = np.sum(sample == 'active') / n
            p_exp = mutant_p_exp[m]
            if p_exp == 0:
                continue
            err = abs(cov_active - p_exp) / p_exp
            accs.append(1.0 - err)
        boot_global[b] = np.mean(accs) if accs else np.nan

    valid = boot_global[~np.isnan(boot_global)]
    if len(valid) == 0:
        return (None, None)
    ci_low = float(np.percentile(valid, 2.5))
    ci_high = float(np.percentile(valid, 97.5))
    return ci_low, ci_high


# ============================================================
# 叙事分支判定（§6，冻结）
# ============================================================

def determine_narrative_branch(global_accuracy):
    """根据全局准确率判定叙事分支（§6，冻结，由数值自动决定）。

    返回: (branch_key, branch_name, description)
    """
    if global_accuracy >= SUCCESS_THRESHOLD:
        return (
            'success',
            '成功档（≥80%）',
            'AF2-Subsample 全局准确率 ≥ 80%，复现 MdS2024 的 >80% 声明。'
            '触发分层基线 fallback 叙事：\n'
            '  - 第一层：原生生成模型（AF3 vanilla / BioEmu）在少样本设定下失败，'
            'CDST 核心对比仍成立\n'
            '  - 第二层：AF2 达到 80% 但需 480 predictions/蛋白，数据效率极低\n'
            '  - 第三层：CDST 在 n=6–8 少样本下具数据效率与可解释性优势'
        )
    elif global_accuracy >= PARTIAL_THRESHOLD:
        return (
            'partial',
            '部分成功档（40%–80%）',
            'AF2-Subsample 全局准确率处于 40%–80% 之间。'
            '报告结果并分析失败原因：MSA 多样性、突变类型影响等；'
            '按突变体逐个报告 acc(m)，识别 CDST 在哪些突变上仍占优。'
            '保留"CDST 在部分突变上更优"的次级叙事。'
        )
    else:
        return (
            'failure',
            '失败档（<40%）',
            'AF2-Subsample 全局准确率 < 40%，无法复现 MdS2024 的 80%。'
            '强化"生成模型失败"叙事，但诚实讨论：'
            '失败可能源于配置不当（MSA 深度、subsampling 比例、参考结构选取）'
            '而非 AF2 方法本身失效。'
        )


# ============================================================
# 报告生成
# ============================================================

def generate_report(results, global_accuracy, global_acc_ci, branch_info,
                    exp_data, n_total_predictions):
    """生成人类可读的 Markdown 报告（§5–§6）。"""
    branch_key, branch_name, branch_desc = branch_info
    lines = []
    a = lines.append

    a('# AF2 MSA-Subsampling 系综基线 — 结果报告')
    a('')
    a('> 预注册协议 v1.0（CDST Task 6 / SubTask 6.1）')
    a('> 阈值与判定标准在实验前冻结，不可事后调整')
    a('')
    a('---')
    a('')

    # 摘要
    a('## 1. 摘要')
    a('')
    a(f'- **全局准确率**（6 突变体均值，不含 WT）: **{global_accuracy:.1%}**')
    if global_acc_ci[0] is not None:
        a(f'  - 95% CI: [{global_acc_ci[0]:.1%}, {global_acc_ci[1]:.1%}]')
    a(f'- **MdS2024 对标基准**: {MDS2024_BENCHMARK:.0%}')
    a(f'- **叙事分支**: {branch_name}')
    a(f'- **总预测数**: {n_total_predictions}')
    a('')

    # 约束检查
    a('### 约束验证')
    a('')
    a('| 突变体 | coverage_active+I1+I2+unclass | 是否=1.0 |')
    a('|--------|-------------------------------|----------|')
    all_valid = True
    for mut in MUTANTS_ALL:
        if mut not in results:
            continue
        r = results[mut]
        total = (r['coverage_active'] + r['coverage_i1']
                 + r['coverage_i2'] + r['coverage_unclass'])
        ok = '✓' if abs(total - 1.0) < 1e-6 else '✗'
        if abs(total - 1.0) >= 1e-6:
            all_valid = False
        a(f'| {mut} | {total:.6f} | {ok} |')
    a('')
    if not all_valid:
        a('> ⚠️ 存在 coverage 之和不为 1.0 的突变体，请检查数据完整性。')
    a('')

    a('---')
    a('')

    # 各突变体 coverage 表
    a('## 2. 各突变体 Coverage（§5.1）')
    a('')
    a('| 突变体 | N_total | N_active | N_I1 | N_I2 | N_unclass | '
      'cov_active | cov_I1 | cov_I2 | cov_unclass | 小样本 |')
    a('|--------|---------|----------|------|------|-----------|'
      '------------|--------|--------|-------------|--------|')
    for mut in MUTANTS_ALL:
        if mut not in results:
            continue
        r = results[mut]
        small = '是 ⚠' if r['N_total'] < SMALL_SAMPLE_THRESHOLD else '否'
        a(
            f'| {mut} | {r["N_total"]} | {r["N_active"]} | {r["N_I1"]} | '
            f'{r["N_I2"]} | {r["N_unclass"]} | '
            f'{r["coverage_active"]:.4f} | {r["coverage_i1"]:.4f} | '
            f'{r["coverage_i2"]:.4f} | {r["coverage_unclass"]:.4f} | {small} |'
        )
    a('')
    a(f'> 小样本定义: N_total < {SMALL_SAMPLE_THRESHOLD}（降级配置 20 predictions/'
      '突变时触发，需标注 95% CI）')
    a('')

    a('---')
    a('')

    # 准确率对比
    a('## 3. 准确率对比（§5.2–§5.3）')
    a('')
    a('### 3.1 六突变体准确率（进入全局准确率）')
    a('')
    a('| 突变体 | P_AF2(active) | P_exp(active) | err(m) | acc(m) | '
      'acc 95% CI | 小样本 |')
    a('|--------|---------------|---------------|--------|--------|'
      '-------------|--------|')
    for mut in MUTANTS_6:
        if mut not in results:
            continue
        r = results[mut]
        p_exp = exp_data.get(mut)
        if p_exp is None:
            a(f'| {mut} | {r["coverage_active"]:.4f} | N/A | N/A | N/A | '
              f'N/A | - |')
            continue
        acc = r.get('accuracy')
        err = r.get('error')
        ci = r.get('acc_ci')
        small = '是 ⚠' if r['N_total'] < SMALL_SAMPLE_THRESHOLD else '否'
        acc_str = f'{acc:.4f}' if acc is not None else 'N/A'
        err_str = f'{err:.4f}' if err is not None else 'N/A'
        if ci and ci[0] is not None:
            ci_str = f'[{ci[0]:.4f}, {ci[1]:.4f}]'
        else:
            ci_str = 'N/A'
        a(
            f'| {mut} | {r["coverage_active"]:.4f} | {p_exp:.2f} | '
            f'{err_str} | {acc_str} | {ci_str} | {small} |'
        )
    a('')
    a(f'**全局准确率 = mean(acc(m), m ∈ 6 突变体) = {global_accuracy:.1%}**')
    if global_acc_ci[0] is not None:
        a(f'  - 95% CI: [{global_acc_ci[0]:.1%}, {global_acc_ci[1]:.1%}] '
          f'(bootstrap {N_BOOTSTRAP} 次)')
    a('')

    # WT 单独报告
    if 'WT' in results:
        a('### 3.2 WT（单独报告，不进入全局准确率）')
        a('')
        wt = results['WT']
        wt_p_exp = exp_data.get('WT')
        a(f'- N_total = {wt["N_total"]}')
        a(f'- P_AF2(active) = {wt["coverage_active"]:.4f}')
        if wt_p_exp is not None:
            a(f'- P_exp(active) = {wt_p_exp:.2f}')
            wt_acc, wt_err = compute_accuracy(
                wt['coverage_active'], wt_p_exp)
            if wt_acc is not None:
                a(f'- acc(WT) = {wt_acc:.4f} (err = {wt_err:.4f})')
        a('> WT 不进入全局准确率，避免 88% active 主导均值（§5.3）。')
        a('')

    a('---')
    a('')

    # 与 MdS2024 对标
    a('## 4. 与 MdS2024 80% 基准对比')
    a('')
    a(f'| 指标 | 本实验 (AF2-Subsample) | MdS2024 基准 | 差距 |')
    a(f'|------|------------------------|--------------|------|')
    diff = global_accuracy - MDS2024_BENCHMARK
    a(f'| 全局准确率 | {global_accuracy:.1%} | {MDS2024_BENCHMARK:.0%} | '
      f'{diff:+.1%} |')
    a('')
    if global_accuracy >= MDS2024_BENCHMARK:
        a('AF2-Subsample **达到** MdS2024 报告的 >80% 全局准确率。')
    else:
        gap = MDS2024_BENCHMARK - global_accuracy
        a(f'AF2-Subsample **未达到** MdS2024 基准，差距 {gap:.1%}。')
    a('')

    a('---')
    a('')

    # 叙事分支
    a('## 5. 叙事分支判定（§6，由数值自动决定）')
    a('')
    a(f'**判定结果: {branch_name}**')
    a('')
    a(f'触发条件: 全局准确率 = {global_accuracy:.1%}')
    a('')
    a('```')
    a(branch_desc)
    a('```')
    a('')

    a('---')
    a('')

    # 诚实局限性讨论
    a('## 6. 诚实局限性讨论')
    a('')
    a('### 6.1 方法局限')
    a('')
    a('- **RMSD 阈值固定 3.0 Å**：预注册冻结，不论实际 RMSD 分布如何均不调整。'
      '该阈值可能对某些突变体偏严或偏松，但为避免事后挑选，不做修改。')
    a('- **全蛋白 Cα RMSD**：与 MdS2024 口径一致，但不使用 aC-helix / A-loop '
      '子结构 RMSD。子结构差异可能被全局对齐平均掉。')
    a('- **unclassified 桶**：按"未分到 active"处理（§5.2），对 AF2 形成惩罚。'
      '若 unclassified 比例过高（>30%），说明参考结构对齐口径或采样存在问题，'
      '但不调阈值，如实报告。')
    a('')
    a('### 6.2 配置局限')
    a('')
    # 根据样本量动态生成
    small_mutants = [m for m in MUTANTS_ALL
                     if m in results and results[m]['N_total'] < SMALL_SAMPLE_THRESHOLD]
    if small_mutants:
        a(f'- **小样本警告**：以下突变体 N_total < {SMALL_SAMPLE_THRESHOLD}，'
          f'统计噪声较大：{", ".join(small_mutants)}')
        a(f'  - 已标注 95% bootstrap CI（{N_BOOTSTRAP} 次重采样）')
        a('  - 若为降级配置（5×4×1=20 predictions/突变），结果须与满配结果分开标注')
    else:
        a('- 满配（480 predictions/蛋白）下统计噪声较小。')
    a('')
    a('- **MSA 来源**：MSA 是否为各突变体独立构建或复用 WT MSA，'
      '影响 MSA subsampling 的有效性。')
    a('- **参考结构选取**：I1 参考 2HYY、I2 参考 6RXG 在实验前锁定，不更换。'
      '参考结构的残基编号与 AF2 预测可能不同，脚本通过共有 Cα 原子对齐处理。')
    a('')
    a('### 6.3 因果推断局限')
    a('')
    a('- AF2-Subsample 的成功/失败**不能**简单归因于 AF2 方法本身。'
      '配置因素（MSA 深度、subsampling 比例 1/4、recycle 次数、dropout）'
      '均可能影响结果。')
    a('- 本协议**只复现基线**，不改进 AF2；不引入 CDST 自身组件。'
      'CDST 的优势需通过独立的少样本实验验证，不可仅凭此基线结果声称。')
    a('- 实验布居来自 Xie 2020 13C CEST（gold-tier），但原文无误差棒，'
      '绝对值可能存在测量不确定度。')
    a('')
    a('### 6.4 多重比较')
    a('')
    a('- 6 个突变体逐个报告 acc(m) 存在多重比较风险，'
      '个别突变体的高/低 acc 可能由随机波动引起，'
      '应关注全局趋势而非单个突变体。')
    a('')

    a('---')
    a('')
    a('## 附录：预注册冻结参数一览')
    a('')
    a('| 冻结项 | 值 |')
    a('|--------|----|')
    a(f'| 状态判定 RMSD 阈值 | {3.0} Å（全 Cα, Kabsch） |')
    a(f'| 多态冲突解决 | argmin RMSD |')
    a(f'| 全归不下 | unclassified（独立桶，不分摊） |')
    a(f'| 全局准确率 | mean over 6 mutants（不含 WT） |')
    a(f'| 成功阈值 | Global Accuracy ≥ 80% |')
    a(f'| 部分成功阈值 | 40% ≤ GA < 80% |')
    a(f'| 失败阈值 | GA < 40% |')
    a(f'| Bootstrap 次数 | {N_BOOTSTRAP} |')
    a(f'| 小样本阈值 | N_total < {SMALL_SAMPLE_THRESHOLD} |')
    a('')
    a('---')
    a('')
    a('*本报告由 compute_coverage.py 自动生成，遵循预注册协议 v1.0。*')
    a('')

    return '\n'.join(lines)


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='AF2 系综 Coverage 与准确率计算 + 最终报告（预注册 §5–§6）'
    )
    parser.add_argument(
        '--classifications', default=DEFAULT_CLASSIFICATIONS,
        help=f'state_classifications.json 路径'
             f'（默认: {DEFAULT_CLASSIFICATIONS}）'
    )
    parser.add_argument(
        '--results-dir', default=DEFAULT_RESULTS_DIR,
        help=f'结果输出目录（默认: {DEFAULT_RESULTS_DIR}）'
    )
    parser.add_argument(
        '--exp-data', default=DEFAULT_EXP_DATA,
        help=f'实验布居 JSON 路径（默认: {DEFAULT_EXP_DATA}）'
    )
    parser.add_argument(
        '--n-bootstrap', type=int, default=N_BOOTSTRAP,
        help=f'bootstrap 重采样次数（默认: {N_BOOTSTRAP}）'
    )
    args = parser.parse_args()

    results_dir = resolve_path(args.results_dir)

    print('=' * 70)
    print('AF2 系综 Coverage 与准确率计算（预注册 §5–§6）')
    print('=' * 70)

    # 加载分类结果
    print('\n加载分类结果...')
    classifications = load_classifications(args.classifications)
    print(f'  共 {len(classifications)} 条分类记录')

    if len(classifications) == 0:
        print('\n警告: 分类结果为空，无法计算 coverage。')
        print('请先运行 classify_states.py 生成有效分类结果。')
        results_dir.mkdir(parents=True, exist_ok=True)
        # 写入空结果
        empty_output = {
            'protocol': 'af2_subsample_preregistration_v1.0',
            'global_accuracy': None,
            'per_mutant': {},
            'error': 'No classifications found',
        }
        with open(results_dir / 'coverage_results.json', 'w',
                  encoding='utf-8') as f:
            json.dump(empty_output, f, indent=2, ensure_ascii=False)
        print(f'已写入空结果: {results_dir / "coverage_results.json"}')
        return

    # 加载实验布居
    print('\n加载实验布居数据...')
    exp_data = load_experimental_populations(args.exp_data)
    print(f'  实验布居 (Active):')
    for mut in MUTANTS_ALL:
        if mut in exp_data:
            print(f'    {mut:<20s}: {exp_data[mut]:.2f}')

    # 按突变体分组状态
    mutant_states = {}
    for c in classifications:
        mut = c['mutant']
        if mut not in mutant_states:
            mutant_states[mut] = []
        mutant_states[mut].append(c['state'])

    # 逐突变体计算 coverage + 准确率 + CI
    print('\n计算各突变体 coverage 与准确率...')
    results = {}
    for mut in MUTANTS_ALL:
        if mut not in mutant_states:
            print(f'  {mut}: 无预测数据，跳过')
            continue
        states_list = mutant_states[mut]
        cov = compute_coverage(states_list)
        p_exp = exp_data.get(mut)
        acc, err = compute_accuracy(cov['coverage_active'], p_exp)
        cov['P_exp_active'] = p_exp
        cov['accuracy'] = acc
        cov['error'] = err

        # Bootstrap CI（使用确定性种子，确保可复现）
        is_small = cov['N_total'] < SMALL_SAMPLE_THRESHOLD
        mut_seed = BOOTSTRAP_SEED + _MUTANT_SEED_OFFSET.get(mut, 0)
        cov_ci = bootstrap_coverage_ci(
            states_list, n_bootstrap=args.n_bootstrap,
            seed=mut_seed)
        cov['coverage_active_ci'] = cov_ci
        acc_ci = bootstrap_accuracy_ci(
            states_list, p_exp, n_bootstrap=args.n_bootstrap,
            seed=mut_seed)
        cov['acc_ci'] = acc_ci
        cov['is_small_sample'] = is_small

        results[mut] = cov

        acc_str = f'{acc:.4f}' if acc is not None else 'N/A'
        print(f'  {mut:<20s}: N={cov["N_total"]}, '
              f'cov_active={cov["coverage_active"]:.4f}, '
              f'P_exp={p_exp if p_exp else "N/A"}, '
              f'acc={acc_str}'
              f'{" (小样本)" if is_small else ""}')

    # 全局准确率（仅在 6 突变体上取均值，不含 WT）
    accs_6 = []
    for mut in MUTANTS_6:
        if mut in results and results[mut]['accuracy'] is not None:
            accs_6.append(results[mut]['accuracy'])

    if accs_6:
        global_accuracy = float(np.mean(accs_6))
        global_acc_ci = bootstrap_global_accuracy_ci(
            mutant_states, exp_data, n_bootstrap=args.n_bootstrap,
            seed=BOOTSTRAP_SEED)
    else:
        global_accuracy = None
        global_acc_ci = (None, None)

    print(f'\n全局准确率（6 突变体均值）: '
          f'{global_accuracy:.1%}' if global_accuracy is not None
          else '\n全局准确率: N/A')
    if global_acc_ci[0] is not None:
        print(f'  95% CI: [{global_acc_ci[0]:.1%}, {global_acc_ci[1]:.1%}]')

    # 叙事分支判定
    if global_accuracy is not None:
        branch_info = determine_narrative_branch(global_accuracy)
        print(f'\n叙事分支: {branch_info[1]}')
    else:
        branch_info = ('unknown', '未知', '无法判定（缺少有效准确率数据）')
        print('\n叙事分支: 无法判定')

    # 写入 coverage_results.json
    results_dir.mkdir(parents=True, exist_ok=True)

    coverage_output = {
        'protocol': 'af2_subsample_preregistration_v1.0',
        'sections': '§5 Coverage + §6 叙事分支',
        'n_total_predictions': len(classifications),
        'mutants_in_global_accuracy': MUTANTS_6,
        'wt_reported_separately': True,
        'global_accuracy': global_accuracy,
        'global_accuracy_ci': list(global_acc_ci)
                              if global_acc_ci[0] is not None else None,
        'mds2024_benchmark': MDS2024_BENCHMARK,
        'narrative_branch': {
            'key': branch_info[0],
            'name': branch_info[1],
        },
        'thresholds': {
            'rmsd': 3.0,
            'success': SUCCESS_THRESHOLD,
            'partial': PARTIAL_THRESHOLD,
        },
        'bootstrap': {
            'n_resamples': args.n_bootstrap,
            'small_sample_threshold': SMALL_SAMPLE_THRESHOLD,
        },
        'per_mutant': {},
    }

    for mut in MUTANTS_ALL:
        if mut not in results:
            continue
        r = results[mut]
        coverage_output['per_mutant'][mut] = {
            'N_total': r['N_total'],
            'N_active': r['N_active'],
            'N_I1': r['N_I1'],
            'N_I2': r['N_I2'],
            'N_unclass': r['N_unclass'],
            'coverage_active': r['coverage_active'],
            'coverage_i1': r['coverage_i1'],
            'coverage_i2': r['coverage_i2'],
            'coverage_unclass': r['coverage_unclass'],
            'P_exp_active': r['P_exp_active'],
            'accuracy': r['accuracy'],
            'error': r['error'],
            'coverage_active_ci': list(r['coverage_active_ci']),
            'acc_ci': list(r['acc_ci']) if r['acc_ci'][0] is not None else None,
            'is_small_sample': r['is_small_sample'],
        }

    coverage_path = results_dir / 'coverage_results.json'
    with open(coverage_path, 'w', encoding='utf-8') as f:
        json.dump(coverage_output, f, indent=2, ensure_ascii=False)
    print(f'\n机器可读结果已写入: {coverage_path}')

    # 生成人类可读报告
    ga_for_report = global_accuracy if global_accuracy is not None else 0.0
    report = generate_report(
        results, ga_for_report, global_acc_ci, branch_info,
        exp_data, len(classifications))
    report_path = results_dir / 'af2_subsample_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'人类可读报告已写入: {report_path}')

    # 终端摘要
    print('\n' + '=' * 70)
    print('摘要')
    print('=' * 70)
    if global_accuracy is not None:
        print(f'  全局准确率: {global_accuracy:.1%} '
              f'(MdS2024 基准: {MDS2024_BENCHMARK:.0%})')
        print(f'  叙事分支: {branch_info[1]}')
    print(f'  总预测数: {len(classifications)}')
    print(f'  结果文件: {coverage_path}')
    print(f'  报告文件: {report_path}')


if __name__ == '__main__':
    main()
