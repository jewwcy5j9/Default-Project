#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
classify_states.py — AF2 系综预测构象态分类（预注册 §4）

对每个 AF2 预测 PDB 计算其与三个参考结构的全蛋白 Cα-RMSD（Kabsch 对齐），
按预注册冻结阈值 3.0 Å 归类为 active / I1 / I2 / unclassified。

参考结构（§4.1，冻结；I2 路径修正见 protocol_deviation_log.md DEV-002）：
  active : data/bioemu_abl1/ref_6XR6_active.pdb   (PDB 6XR6, DFG-in)
  I1     : data/bioemu_abl1/ref_2HYY_i1.pdb        (PDB 2HYY, 中间态)
  I2     : data/bioemu_abl1/ref_6XRG_i2.pdb        (PDB 6XRG, DFG-out)
  注：预注册原文误写为 6RXG（Bifidobacterium longum 磷酸酶），实际应为 6XRG（Abl1 I2）

归类规则（§4.2，冻结，顺序无关）：
  1. 对每个预测分别计算 RMSD_active / RMSD_I1 / RMSD_I2
  2. 若与多个状态 RMSD 同时 < 3.0 Å → 归入 RMSD 最小的状态（argmin）
  3. 若所有状态 RMSD ≥ 3.0 Å → 归入 unclassified

RMSD 口径（§4.3，冻结）：
  - 对齐域：全蛋白 Cα（kinase domain，与 MdS2024 一致）
  - Kabsch 最优对齐后的 Cα-RMSD
  - 阈值 3.0 Å 在实验前冻结，不论实际分布如何均不调整

输出：experiments/af2_subsample/results/state_classifications.json

用法：
  python classify_states.py
  python classify_states.py --output-dir path/to/predictions --results-dir path/to/results
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

# 中文与 Å 输出在非 UTF-8 Windows 控制台会 UnicodeEncodeError;兄弟脚本
# 均使用同一守卫。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# ============================================================
# 预注册冻结参数（§4，不可事后调整）
# ============================================================
RMSD_THRESHOLD = 3.0  # Å — 状态判定 RMSD 阈值（§4.1，冻结）

# AF2 预测残基编号偏移量：预测残基1 = Abl1 残基235
# (a3m query 从 W235 开始，见 run_af2_subsample.py 中 ABL1A_OFFSET=227 + FASTA偏移)
# 参考结构使用 PDB 残基号(248-534 或 235-498)，预测使用 1-263
# 匹配时需将预测残基号 + PRED_RESEQ_OFFSET 映射到 Abl1 编号
PRED_RESEQ_OFFSET = 234  # 预测残基1 → Abl1 235 (235-1=234)

# 7 个突变体（§5.4，冻结）：WT + 6 突变体
MUTANTS = ['WT', 'M290L', 'L301I', 'M290L_L301I', 'F382L', 'F382Y', 'F382V']

# 默认参考结构路径（§4.1，冻结）
DEFAULT_REF_ACTIVE = 'data/bioemu_abl1/ref_6XR6_active.pdb'
DEFAULT_REF_I1 = 'data/bioemu_abl1/ref_2HYY_i1.pdb'
DEFAULT_REF_I2 = 'data/bioemu_abl1/ref_6XRG_i2.pdb'  # 修正：6RXG→6XRG（DEV-002）

# 默认预测输出目录与结果目录
DEFAULT_OUTPUT_DIR = 'experiments/af2_subsample/output'
DEFAULT_RESULTS_DIR = 'experiments/af2_subsample/results'

# 预测文件名模式：run_R/model_M_seed_S.pdb
FILENAME_RE = re.compile(r'run_(\d+)[\\/]model_(\d+)_seed_(\d+)\.pdb$', re.IGNORECASE)

# 项目根目录（本脚本位于 experiments/af2_subsample/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_path(path_str):
    """将相对路径解析为基于项目根目录的绝对路径。"""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def get_ca_atoms(structure, chain_id=None, model_id=0):
    """从结构中提取 Cα 原子。

    参数:
        structure: BioPython Structure 对象
        chain_id: 指定链 ID；None 则自动取第一条链
        model_id: 模型编号（NMR 多模型结构取第 0 个）

    返回:
        ordered: [(resseq, Atom), ...] 按残基顺序排列
        by_resseq: {resseq: Atom} 字典，用于按残基号匹配
    """
    model = structure[model_id]
    # 选择链：优先指定链，否则取第一条
    if chain_id is not None and chain_id in model:
        chain = model[chain_id]
    else:
        chain = next(model.get_chains())

    ordered = []
    by_resseq = {}
    for residue in chain:
        # 只取标准氨基酸残基（id[0]==' ' 表示非异质）
        if residue.id[0] != ' ':
            continue
        if 'CA' in residue:
            resseq = residue.id[1]
            atom = residue['CA']
            ordered.append((resseq, atom))
            by_resseq[resseq] = atom
    return ordered, by_resseq


def compute_ca_rmsd(pred_ordered, pred_by_resseq, ref_ordered, ref_by_resseq):
    """计算两个结构之间的 Cα-RMSD（Kabsch 最优对齐）。

    匹配策略（§4.3 全蛋白 Cα）：
      1. 自动检测参考结构与预测的残基编号偏移量（通过序列比对）
      2. 按偏移量映射残基号，使用共有 Cα 原子对齐
      3. 若序列比对失败，回退到按顺序取前 N 个 Cα

    使用 Bio.PDB.Superimposer 执行 Kabsch SVD 对齐。

    返回:
        rmsd: 对齐后 Cα-RMSD（Å）
        n_atoms: 参与对齐的 Cα 原子数
    """
    # 自动检测残基编号偏移量：通过比较前几个残基的序列
    offset = detect_residue_offset(pred_ordered, ref_ordered)
    if offset is not None:
        # 将预测残基号映射到参考结构的编号体系
        pred_mapped = {r + offset: atom for r, atom in pred_by_resseq.items()}
        common_resseqs = sorted(set(pred_mapped.keys()) & set(ref_by_resseq.keys()))
        if len(common_resseqs) >= max(10, int(0.3 * min(len(pred_ordered), len(ref_ordered)))):
            fixed_atoms = [ref_by_resseq[r] for r in common_resseqs]
            moving_atoms = [pred_mapped[r] for r in common_resseqs]
        else:
            n = min(len(pred_ordered), len(ref_ordered))
            fixed_atoms = [atom for _, atom in ref_ordered[:n]]
            moving_atoms = [atom for _, atom in pred_ordered[:n]]
    else:
        # 序列比对失败，按顺序取前 N 个
        n = min(len(pred_ordered), len(ref_ordered))
        fixed_atoms = [atom for _, atom in ref_ordered[:n]]
        moving_atoms = [atom for _, atom in pred_ordered[:n]]

    if len(fixed_atoms) < 3:
        raise ValueError(
            f"共有 Cα 原子数过少（{len(fixed_atoms)}），无法计算 RMSD"
        )

    from Bio.PDB import Superimposer

    # Kabsch 对齐（fixed=参考，moving=预测）
    sup = Superimposer()
    sup.set_atoms(fixed_atoms, moving_atoms)
    return float(sup.rms), len(fixed_atoms)


# 氨基酸三字母→单字母映射
_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def get_sequence_from_ordered(ordered):
    """从有序 Cα 列表提取单字母序列字符串。"""
    return ''.join(_THREE_TO_ONE.get(atom.get_parent().get_resname(), 'X')
                   for _, atom in ordered)


def detect_residue_offset(pred_ordered, ref_ordered, k=15):
    """通过序列比对检测参考结构与预测的残基编号偏移量。

    策略：取参考结构前 k 个残基的序列，在预测序列中搜索，
    找到匹配位置后计算偏移量。

    返回:
        offset: 参考结构残基号 = 预测残基号 + offset
        None: 无法确定偏移量
    """
    if len(pred_ordered) < k or len(ref_ordered) < k:
        return None

    pred_seq = get_sequence_from_ordered(pred_ordered)
    ref_seq = get_sequence_from_ordered(ref_ordered)

    # 用参考结构前 k 个残基在预测序列中搜索
    prefix = ref_seq[:k]
    idx = pred_seq.find(prefix)
    if idx >= 0:
        # 预测第(idx+1)个残基 = 参考结构第1个残基
        # 参考残基号 ref_resseq[0] = 预测残基号 pred_resseq[idx]
        # offset = ref_resseq[0] - pred_resseq[idx]
        offset = ref_ordered[0][0] - pred_ordered[idx][0]
        return offset

    # 尝试用预测前 k 个在参考序列中搜索
    prefix = pred_seq[:k]
    idx = ref_seq.find(prefix)
    if idx >= 0:
        offset = ref_ordered[idx][0] - pred_ordered[0][0]
        return offset

    return None


def classify_state(rmsd_active, rmsd_i1, rmsd_i2, threshold=RMSD_THRESHOLD):
    """按 §4.2 归类构象态（顺序无关，预注册冻结）。

    - 若与多个状态 RMSD 同时 < 3.0 Å → 归入 RMSD 最小的状态（argmin）
    - 若所有状态 RMSD ≥ 3.0 Å → 归入 unclassified
    """
    rmsds = {'active': rmsd_active, 'I1': rmsd_i1, 'I2': rmsd_i2}
    # 筛选 RMSD < 阈值的状态
    below = {k: v for k, v in rmsds.items() if v < threshold}
    if not below:
        return 'unclassified'
    # 多态冲突时取 RMSD 最小的状态（argmin）
    return min(below, key=below.get)


def load_reference(parser, path, label):
    """加载参考结构，返回 Cα 有序列表与字典。

    对 NMR 多模型结构（如 6XR6 有 20 个模型）取第 0 个模型；
    对多链结构（如 2HYY 有 A/B/C/D 链）取链 A。
    """
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"参考结构文件不存在: {path}")
    structure = parser.get_structure(label, str(path))
    ordered, by_resseq = get_ca_atoms(structure, chain_id='A', model_id=0)
    if len(ordered) == 0:
        # 链 A 无 Cα，回退到第一条可用链
        ordered, by_resseq = get_ca_atoms(structure, chain_id=None, model_id=0)
    if len(ordered) == 0:
        raise ValueError(f"参考结构 {path} 未找到任何 Cα 原子")
    return ordered, by_resseq, len(ordered)


def find_predictions(output_dir):
    """扫描 output_dir 下所有预测 PDB。

    目录结构：output/<mutant>/run_R/model_M_seed_S.pdb

    返回: [(pdb_path, mutant, run, model, seed), ...]
    """
    results = []
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return results

    for pdb_path in sorted(output_dir.rglob('*.pdb')):
        # 匹配文件名模式 run_R/model_M_seed_S.pdb
        m = FILENAME_RE.search(str(pdb_path))
        if not m:
            continue
        run, model, seed = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # 突变体名 = 相对路径的第一级目录
        try:
            rel = pdb_path.relative_to(output_dir)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 2:
            continue
        mutant = parts[0]
        if mutant not in MUTANTS:
            continue
        results.append((str(pdb_path), mutant, run, model, seed))
    return results


def main():
    from Bio.PDB import PDBParser

    parser = argparse.ArgumentParser(
        description='AF2 系综预测构象态分类（预注册 §4）'
    )
    parser.add_argument(
        '--output-dir', default=DEFAULT_OUTPUT_DIR,
        help=f'预测 PDB 所在目录（默认: {DEFAULT_OUTPUT_DIR}）'
    )
    parser.add_argument(
        '--results-dir', default=DEFAULT_RESULTS_DIR,
        help=f'结果输出目录（默认: {DEFAULT_RESULTS_DIR}）'
    )
    parser.add_argument(
        '--ref-active', default=DEFAULT_REF_ACTIVE,
        help=f'active 参考结构路径（默认: {DEFAULT_REF_ACTIVE}）'
    )
    parser.add_argument(
        '--ref-i1', default=DEFAULT_REF_I1,
        help=f'I1 参考结构路径（默认: {DEFAULT_REF_I1}）'
    )
    parser.add_argument(
        '--ref-i2', default=DEFAULT_REF_I2,
        help=f'I2 参考结构路径（默认: {DEFAULT_REF_I2}）'
    )
    parser.add_argument(
        '--threshold', type=float, default=RMSD_THRESHOLD,
        help=f'RMSD 阈值 Å（预注册冻结 {RMSD_THRESHOLD}，仅供记录不得更改）'
    )
    args = parser.parse_args()

    # 解析路径
    output_dir = resolve_path(args.output_dir)
    results_dir = resolve_path(args.results_dir)

    print('=' * 70)
    print('AF2 系综预测构象态分类（预注册 §4）')
    print('=' * 70)
    print(f'预测目录: {output_dir}')
    print(f'结果目录: {results_dir}')
    print(f'参考结构:')
    print(f'  active: {args.ref_active}')
    print(f'  I1    : {args.ref_i1}')
    print(f'  I2    : {args.ref_i2}')
    print(f'RMSD 阈值: {args.threshold} Å（预注册冻结）')
    print()

    # 加载参考结构
    parser_pdb = PDBParser(QUIET=True)
    print('加载参考结构...')
    ref_active_ord, ref_active_dict, n_active = load_reference(
        parser_pdb, args.ref_active, 'active')
    ref_i1_ord, ref_i1_dict, n_i1 = load_reference(
        parser_pdb, args.ref_i1, 'i1')
    ref_i2_ord, ref_i2_dict, n_i2 = load_reference(
        parser_pdb, args.ref_i2, 'i2')
    print(f'  active: {n_active} Cα 原子')
    print(f'  I1    : {n_i1} Cα 原子')
    print(f'  I2    : {n_i2} Cα 原子')
    print()

    # 扫描预测 PDB
    print('扫描预测 PDB 文件...')
    predictions = find_predictions(output_dir)
    print(f'共找到 {len(predictions)} 个预测文件')
    if not predictions:
        print()
        print('警告: 未找到任何预测 PDB 文件。')
        print(f'请确认预测目录结构为: {output_dir}/<mutant>/run_R/model_M_seed_S.pdb')
        print('输出空结果文件以便后续脚本正常运行。')
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / 'state_classifications.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        print(f'已写入空结果: {output_path}')
        return

    # 按突变体统计
    mutant_counts = {}
    for _, mutant, _, _, _ in predictions:
        mutant_counts[mutant] = mutant_counts.get(mutant, 0) + 1
    print('各突变体预测数:')
    for mut in MUTANTS:
        print(f'  {mut:<20s}: {mutant_counts.get(mut, 0)}')
    print()

    # 逐个预测计算 RMSD 并归类
    print('开始计算 Cα-RMSD 并归类...')
    classifications = []
    t0 = time.time()
    n_total = len(predictions)
    warnings = []

    for idx, (pdb_path, mutant, run, model, seed) in enumerate(predictions):
        if (idx + 1) % 100 == 0 or idx == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (n_total - idx - 1) / rate if rate > 0 else 0
            print(f'  [{idx+1}/{n_total}] {mutant} run_{run} model_{model} '
                  f'seed_{seed}  ({rate:.1f}/s, ETA {eta:.0f}s)')

        try:
            struct = parser_pdb.get_structure('pred', pdb_path)
            pred_ord, pred_dict = get_ca_atoms(struct, chain_id='A', model_id=0)
            if len(pred_ord) == 0:
                pred_ord, pred_dict = get_ca_atoms(struct, chain_id=None, model_id=0)
            if len(pred_ord) == 0:
                msg = f'{pdb_path}: 未找到 Cα 原子，跳过'
                warnings.append(msg)
                print(f'  警告: {msg}')
                continue
        except Exception as e:
            msg = f'{pdb_path}: 解析失败 ({e})，跳过'
            warnings.append(msg)
            print(f'  警告: {msg}')
            continue

        # 计算与三个参考的 Cα-RMSD
        try:
            rmsd_active, n_a = compute_ca_rmsd(
                pred_ord, pred_dict, ref_active_ord, ref_active_dict)
            rmsd_i1, n_1 = compute_ca_rmsd(
                pred_ord, pred_dict, ref_i1_ord, ref_i1_dict)
            rmsd_i2, n_2 = compute_ca_rmsd(
                pred_ord, pred_dict, ref_i2_ord, ref_i2_dict)
        except Exception as e:
            msg = f'{pdb_path}: RMSD 计算失败 ({e})，跳过'
            warnings.append(msg)
            print(f'  警告: {msg}')
            continue

        state = classify_state(rmsd_active, rmsd_i1, rmsd_i2, args.threshold)

        classifications.append({
            'mutant': mutant,
            'run': run,
            'model': model,
            'seed': seed,
            'pdb_path': pdb_path,
            'rmsd_active': round(rmsd_active, 4),
            'rmsd_i1': round(rmsd_i1, 4),
            'rmsd_i2': round(rmsd_i2, 4),
            'state': state,
        })

    elapsed = time.time() - t0
    print(f'\n完成: {len(classifications)}/{n_total} 个预测已归类 ({elapsed:.1f}s)')

    if warnings:
        print(f'\n警告 ({len(warnings)} 条):')
        for w in warnings[:10]:
            print(f'  - {w}')
        if len(warnings) > 10:
            print(f'  ... 还有 {len(warnings) - 10} 条')

    # 写入结果
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / 'state_classifications.json'
    output_data = {
        'protocol': 'af2_subsample_preregistration_v1.0',
        'section': '§4 构象态判定',
        'rmsd_threshold': args.threshold,
        'references': {
            'active': args.ref_active,
            'I1': args.ref_i1,
            'I2': args.ref_i2,
        },
        'rmsd_method': 'full-protein Cα, Kabsch (Bio.PDB.Superimposer)',
        'classification_rule': f'argmin among states with RMSD < {args.threshold:g} Å; '
                               f'unclassified if all >= {args.threshold:g} Å',
        'n_predictions_found': n_total,
        'n_classified': len(classifications),
        'n_warnings': len(warnings),
        'classifications': classifications,
    }
    if warnings:
        output_data['warnings'] = warnings

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f'\n结果已写入: {output_path}')

    # 简要统计
    print('\n归类统计:')
    state_counts = {}
    for c in classifications:
        key = (c['mutant'], c['state'])
        state_counts[key] = state_counts.get(key, 0) + 1
    print(f'  {"突变体":<20s} {"active":>8s} {"I1":>8s} {"I2":>8s} '
          f'{"unclass":>8s} {"总计":>8s}')
    print('  ' + '-' * 62)
    for mut in MUTANTS:
        n_a = state_counts.get((mut, 'active'), 0)
        n_1 = state_counts.get((mut, 'I1'), 0)
        n_2 = state_counts.get((mut, 'I2'), 0)
        n_u = state_counts.get((mut, 'unclassified'), 0)
        n_t = n_a + n_1 + n_2 + n_u
        print(f'  {mut:<20s} {n_a:>8d} {n_1:>8d} {n_2:>8d} '
              f'{n_u:>8d} {n_t:>8d}')


if __name__ == '__main__':
    main()
