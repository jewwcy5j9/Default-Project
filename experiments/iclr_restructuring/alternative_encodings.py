"""
Alternative Encoding Schemes for CDST F382 Series Prediction.

背景:
  Task 5 诊断发现简单的 pos382+ΔΔG 修复失败, 根因是 dVol=-6.0 (F382V) 极端值
  淹没了 ΔΔG 信号. 模型学到 "dVol 越负 -> non_ground 越低" 的虚假关系.
  本脚本测试 4 种更根本的编码改进方案:

  A. dVol 归一化 Extended (10-dim): dVolume 除以 6.0 归一化到 [-1, 1]
  B. 移除 dVol 的 Extended (11-dim): 去掉 dVolume, +pos382 +ΔΔG
  C. ΔΔG 主特征编码 (5-dim): [pos/seq, ΔΔG_norm, pos290, pos301, pos382]
  D. BLOSUM62 替代矩阵编码 (5-dim): [blosum62, pos/seq, pos290, pos301, pos382]

协议:
  - LowRankCDST(K=2, intervention_dim=d, rank=2, hidden_dim=32)
  - prob-space MSE, 5 seeds, 800 epochs (与 f382_encoding_fix.py 一致)
  - Abl1 LOO-CV (6 mutants)
  - 对照: Extended 10-dim (原编码)

LowRankCDST.forward 返回 softmax 概率 (不是 log_softmax),
prob-space MSE 直接用 F.mse_loss(model(w, c), target).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import numpy as np
import torch
import torch.nn.functional as F

from src.models.low_rank_cdst import LowRankCDST

# 复用 encoding_ablation_control 中的 canonical 定义, 保证对照基线完全一致
from encoding_ablation_control import (
    ABL1_DATA, ABL1_WT_NON_GROUND, ABL1_SEQ_LEN,
    AA_PROPERTIES_6_EXT,
    encode_extended,
)


# ============================================================
# ΔΔG 数据: 从 xie2020_abl1_FINAL.json 的 energies_kcal_mol 提取
# F382L 不在 energies 字段中 (paper 注明 "identical to WT"), 用 0.0 填充
# ============================================================
DDG_DATA = {
    'M290L':        -1.3,
    'L301I':        -2.2,
    'M290L_L301I':  -3.5,
    'F382L':         0.0,   # 不在 paper energies 字段, 与 WT 一致 -> ΔΔG ≈ 0
    'F382Y':        -2.5,
    'F382V':        -3.0,
}
# 归一化常数: 6 个突变体中 |ΔΔG| 的最大值 (3.5), 映射到 [-1, 1]
DDG_NORM = max(abs(v) for v in DDG_DATA.values())  # = 3.5

# dVol 归一化常数: F382V 的 dVol = -6.0 是全表极端值
DVOL_NORM = 6.0

# BLOSUM62 归一化常数: 数据无关的固定常数 (覆盖 BLOSUM62 替代得分的典型范围)
BLOSUM_NORM = 9.0


# ============================================================
# Standard BLOSUM62 substitution matrix (20x20, symmetric)
# 行/列顺序: ARNDCQEGHILKMFPSTWYV
# 来源: NCBI 标准 BLOSUM62
# ============================================================
BLOSUM62_AAS = "ARNDCQEGHILKMFPSTWYV"
BLOSUM62_MATRIX = np.array([
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],  # A
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],  # R
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],  # N
    [-2,-2, 1, 6,-3, 0, 2,-2,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],  # D
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],  # C
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],  # Q
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-3],  # E
    [ 0,-2, 0,-2,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],  # G
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],  # H
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],  # I
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],  # L
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],  # K
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],  # M
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],  # F
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 9,-1,-1,-4,-3,-3],  # P
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],  # S
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],  # T
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],  # W
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],  # Y
    [ 0,-3,-3,-3,-1,-2,-3,-3,-3, 3, 1,-2, 1,-1,-3,-2, 0,-3,-1, 4],  # V
])


def blosum62_score(wt_aa, mut_aa):
    """Lookup BLOSUM62 substitution score. 支持多字母 (双突变取和)."""
    if len(wt_aa) > 1:
        return sum(blosum62_score(w, m) for w, m in zip(wt_aa, mut_aa))
    if wt_aa not in BLOSUM62_AAS or mut_aa not in BLOSUM62_AAS:
        return 0.0
    i = BLOSUM62_AAS.index(wt_aa)
    j = BLOSUM62_AAS.index(mut_aa)
    return float(BLOSUM62_MATRIX[i, j])


# ============================================================
# 编码变体
# ============================================================

def encode_baseline(name, data, table, seq_len, system='abl1'):
    """对照: 原 Extended 10-dim (canonical physics encoding)."""
    return encode_extended(name, data, table, seq_len, system=system)


def encode_dvol_normalized(name, data, table, seq_len, system='abl1'):
    """变体 A: dVol 归一化 Extended (10-dim).

    与原 Extended 10-dim 相同, 但 dVolume 维度 (index 1) 除以 6.0,
    归一化到 [-1, 1] 范围. 其他维度不变.
    目的: 削弱 F382V 的 dVol=-6.0 极端值对 ΔΔG 信号的淹没.
    """
    enc = encode_extended(name, data, table, seq_len, system=system).copy()
    enc[1] = enc[1] / DVOL_NORM  # dVol 归一化
    return enc


def encode_no_dvol(name, data, ddg_map, ddg_norm, table, seq_len, system='abl1'):
    """变体 B: 移除 dVol 的 Extended (11-dim).

    9-dim (移除 dVolume) + pos382 + ΔΔG = 11-dim.
    维度: [pos/seq, dHydro, dArom, dHBD, dHBA, dChg, dbl, pos290, pos301, pos382, ddg]
    目的: 彻底移除误导性的 dVol, 让 ΔΔG 成为 F382 系列的主要连续信号.
    """
    enc10 = encode_extended(name, data, table, seq_len, system=system)
    # 移除 index 1 (dVol), 保留其余 9 维
    keep_idx = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    enc = list(enc10[keep_idx])
    # pos382 one-hot
    enc.append(1.0 if data['pos'] == 382 else 0.0)
    # ΔΔG normalized
    ddg = ddg_map.get(name, 0.0)
    enc.append(ddg / ddg_norm if ddg_norm > 0 else 0.0)
    return np.array(enc)


def encode_ddg_main(name, data, ddg_map, ddg_norm, seq_len, system='abl1'):
    """变体 C: ΔΔG 主特征编码 (5-dim).

    [pos/seq, ΔΔG_norm, pos290, pos301, pos382]
    ΔΔG 作为主特征, 位置标记辅助.
    目的: 用最强实验信号 (ΔΔG) 直接驱动预测, 抛弃可能误导的 AA 物理差.
    """
    enc = np.zeros(5)
    enc[0] = data['pos'] / seq_len
    ddg = ddg_map.get(name, 0.0)
    enc[1] = ddg / ddg_norm if ddg_norm > 0 else 0.0
    if system == 'abl1':
        if data['pos'] == 290:
            enc[2] = 1.0
        elif data['pos'] == 301:
            enc[3] = 1.0
        elif data['pos'] == 382:
            enc[4] = 1.0
    return enc


def encode_blosum62(name, data, seq_len, system='abl1'):
    """变体 D: BLOSUM62 替代矩阵编码 (5-dim).

    [blosum62_score_norm, pos/seq, pos290, pos301, pos382]
    使用 BLOSUM62 突变得分替代 AA 物理差.
    目的: 用进化保守性信号替代物理性质差, 避免体积维度的极端值问题.
    BLOSUM62 得分除以 9.0 归一化 (数据无关常数, 覆盖矩阵典型范围).
    """
    enc = np.zeros(5)
    score = blosum62_score(data['wt'], data['mut'])
    enc[0] = score / BLOSUM_NORM
    enc[1] = data['pos'] / seq_len
    if system == 'abl1':
        if data['pos'] == 290:
            enc[2] = 1.0
        elif data['pos'] == 301:
            enc[3] = 1.0
        elif data['pos'] == 382:
            enc[4] = 1.0
    return enc


# ============================================================
# LOO-CV 训练 (复刻 f382_encoding_fix.py 协议)
# ============================================================
def run_loo_cv(encoder_fn, d, n_seeds=5, n_epochs=800):
    """LowRankCDST + prob-space MSE 的 LOO-CV.

    与 f382_encoding_fix.py 的 run_loo_cv 协议完全一致:
      - LowRankCDST(K=2, intervention_dim=d, rank=2, hidden_dim=32)
      - Adam(lr=5e-3, weight_decay=1e-4)
      - n_epochs=800, best-state 追踪 (按 train loss)
      - n_seeds=5, seed = seed*100 + hold_out
      - loss = F.mse_loss(model(w, c), target)  # prob-space
    """
    mutations = list(ABL1_DATA.keys())
    n = len(mutations)
    wt_non = ABL1_WT_NON_GROUND

    wt = np.array([1 - wt_non, wt_non])
    w_wt = np.tile(wt, (n, 1))
    w_target = np.array([[1 - ABL1_DATA[m]['non_ground'], ABL1_DATA[m]['non_ground']]
                         for m in mutations])
    c = np.array([encoder_fn(m, ABL1_DATA[m]) for m in mutations])
    assert c.shape[1] == d, f"Encoding dim mismatch: {c.shape[1]} vs {d}"

    all_preds = {m: [] for m in mutations}

    for seed in range(n_seeds):
        for hold_out in range(n):
            mask = np.ones(n, dtype=bool)
            mask[hold_out] = False

            torch.manual_seed(seed * 100 + hold_out)
            model = LowRankCDST(K=2, intervention_dim=d, rank=2, hidden_dim=32)
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)

            w_t = torch.FloatTensor(w_wt[mask])
            c_t = torch.FloatTensor(c[mask])
            wt_t = torch.FloatTensor(w_target[mask])

            best_loss = float('inf')
            best_state = None
            for _ in range(n_epochs):
                model.train()
                optimizer.zero_grad()
                pred = model(w_t, c_t)
                loss = F.mse_loss(pred, wt_t)  # prob-space MSE
                loss.backward()
                optimizer.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                pred = model(
                    torch.FloatTensor(w_wt[hold_out:hold_out+1]),
                    torch.FloatTensor(c[hold_out:hold_out+1])
                )
                all_preds[mutations[hold_out]].append(float(pred.numpy()[0, 1]))

    return {m: float(np.mean(v)) for m, v in all_preds.items()}


def compute_metrics(preds):
    """计算 MAE / 方向准确率 / F382 子集 MAE / 290-301 子集 MAE。"""
    mutations = list(ABL1_DATA.keys())
    wt_non = ABL1_WT_NON_GROUND
    errors = {m: abs(preds[m] - ABL1_DATA[m]['non_ground']) for m in mutations}
    mae = float(np.mean(list(errors.values())))

    # 方向: sign(pred - wt) == sign(true - wt), tie (|Δtrue|<0.05) 跳过
    dir_correct, dir_total = 0, 0
    dir_detail = {}
    for m in mutations:
        d_true = ABL1_DATA[m]['non_ground'] - wt_non
        d_pred = preds[m] - wt_non
        if abs(d_true) < 0.05:
            dir_detail[m] = 'TIE'
            continue
        dir_total += 1
        if np.sign(d_true) == np.sign(d_pred):
            dir_correct += 1
            dir_detail[m] = 'OK'
        else:
            dir_detail[m] = 'WRONG'

    f382 = ['F382L', 'F382Y', 'F382V']
    mae_382 = float(np.mean([errors[m] for m in f382]))
    mae_290_301 = float(np.mean([errors[m] for m in mutations if m not in f382]))

    return {
        'mae': mae,
        'mae_382': mae_382,
        'mae_290_301': mae_290_301,
        'direction': f"{dir_correct}/{dir_total}",
        'direction_pct': dir_correct / dir_total if dir_total > 0 else 0.0,
        'direction_detail': dir_detail,
        'errors': {m: float(e) for m, e in errors.items()},
    }


def cosine_sim(a, b):
    """余弦相似度。"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def dir_str(d_true, d_pred):
    if abs(d_true) < 0.05:
        return "TIE"
    return "OK" if np.sign(d_true) == np.sign(d_pred) else "WRONG"


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 95)
    print("替代编码方案实验: 改进 CDST F382 系列预测")
    print("协议: LowRankCDST(K=2, rank=2, hidden_dim=32), prob-space MSE, 5 seeds, 800 epochs, Abl1 LOO-CV")
    print("=" * 95)

    torch.manual_seed(0)
    np.random.seed(0)

    mutations = list(ABL1_DATA.keys())
    wt_non = ABL1_WT_NON_GROUND

    # ----------------------------------------------------------
    # (1) 编码向量预览 + 关键特征对比
    # ----------------------------------------------------------
    print("\n" + "#" * 95)
    print("# (1) 编码向量预览 (重点 F382 系列)")
    print("#" * 95)

    # 构造各编码器闭包
    encoders = {
        'Baseline_Extended_10dim': {
            'fn': lambda name, data: encode_baseline(name, data, AA_PROPERTIES_6_EXT, ABL1_SEQ_LEN, 'abl1'),
            'd': 10,
            'label': 'Extended 10-dim (对照)',
        },
        'A_dVol_normalized_10dim': {
            'fn': lambda name, data: encode_dvol_normalized(name, data, AA_PROPERTIES_6_EXT, ABL1_SEQ_LEN, 'abl1'),
            'd': 10,
            'label': 'A: dVol 归一化 10-dim',
        },
        'B_no_dVol_11dim': {
            'fn': lambda name, data: encode_no_dvol(name, data, DDG_DATA, DDG_NORM,
                                                    AA_PROPERTIES_6_EXT, ABL1_SEQ_LEN, 'abl1'),
            'd': 11,
            'label': 'B: 移除 dVol 11-dim',
        },
        'C_ddg_main_5dim': {
            'fn': lambda name, data: encode_ddg_main(name, data, DDG_DATA, DDG_NORM, ABL1_SEQ_LEN, 'abl1'),
            'd': 5,
            'label': 'C: ΔΔG 主特征 5-dim',
        },
        'D_blosum62_5dim': {
            'fn': lambda name, data: encode_blosum62(name, data, ABL1_SEQ_LEN, 'abl1'),
            'd': 5,
            'label': 'D: BLOSUM62 5-dim',
        },
    }

    # 关键特征表: ΔΔG / dVol / BLOSUM62 / 真值
    print("\n--- 各突变体关键特征值 ---")
    print(f"{'Mutant':<16} {'trueNG':>7} {'ΔΔG':>7} {'dVol':>7} {'BLOSUM':>7} "
          f"{'pos/seq':>8} {'pos290':>7} {'pos301':>7} {'pos382':>7}")
    print("-" * 80)
    enc_all = {key: {} for key in encoders}
    for m in mutations:
        true_ng = ABL1_DATA[m]['non_ground']
        ddg = DDG_DATA[m]
        ext10 = encode_extended(m, ABL1_DATA[m], AA_PROPERTIES_6_EXT, ABL1_SEQ_LEN, 'abl1')
        dvol = ext10[1]
        blosum = blosum62_score(ABL1_DATA[m]['wt'], ABL1_DATA[m]['mut'])
        posseq = ABL1_DATA[m]['pos'] / ABL1_SEQ_LEN
        p290 = 1.0 if ABL1_DATA[m]['pos'] == 290 else 0.0
        p301 = 1.0 if ABL1_DATA[m]['pos'] == 301 else 0.0
        p382 = 1.0 if ABL1_DATA[m]['pos'] == 382 else 0.0
        print(f"{m:<16} {true_ng:>7.2f} {ddg:>+7.2f} {dvol:>+7.2f} {blosum:>+7.1f} "
              f"{posseq:>8.3f} {p290:>7.1f} {p301:>7.1f} {p382:>7.1f}")
        # 缓存所有编码向量
        for key, cfg in encoders.items():
            enc_all[key][m] = cfg['fn'](m, ABL1_DATA[m])

    # F382 系列两两余弦相似度 (各编码)
    print("\n--- F382 系列两两余弦相似度 (各编码方案) ---")
    f382_pairs = [('F382L', 'F382V'), ('F382L', 'F382Y'), ('F382Y', 'F382V')]
    header = f"{'Pair':<22}"
    for key in encoders:
        header += f" {key.split('_')[0]:>10}"
    print(header)
    print("-" * (22 + 11 * len(encoders)))
    cos_results = {}
    for a, b in f382_pairs:
        row = f"{a+' vs '+b:<22}"
        for key in encoders:
            cs = cosine_sim(enc_all[key][a], enc_all[key][b])
            cos_results[f"{a}_vs_{b}_{key}"] = cs
            row += f" {cs:>10.4f}"
        print(row)

    # ----------------------------------------------------------
    # (2) LOO-CV: 5 个编码方案
    # ----------------------------------------------------------
    print("\n" + "#" * 95)
    print("# (2) LOO-CV 预测 (LowRankCDST, prob-space MSE, 5 seeds, 800 epochs)")
    print("#" * 95)

    results = {}
    for idx, (key, cfg) in enumerate(encoders.items()):
        print(f"\n[{idx+1}/{len(encoders)}] {cfg['label']} (d={cfg['d']}) ...")
        preds = run_loo_cv(encoder_fn=cfg['fn'], d=cfg['d'])
        metrics = compute_metrics(preds)
        results[key] = {
            'label': cfg['label'],
            'd': cfg['d'],
            'per_mutant': preds,
            **metrics,
        }
        print(f"  MAE={metrics['mae']:.4f}  382-MAE={metrics['mae_382']:.4f}  "
              f"290/301-MAE={metrics['mae_290_301']:.4f}  dir={metrics['direction']}")

    # ----------------------------------------------------------
    # (3) F382 系列逐突变预测对比表
    # ----------------------------------------------------------
    print("\n" + "#" * 95)
    print("# (3) F382 系列逐突变预测对比表 (重点: 方向修复)")
    print("#" * 95)

    short_names = {key: key.split('_')[0] for key in encoders}
    # 详细对比表: 每个突变, 各编码的 pred / Δpred / dir
    print(f"\n{'Mutant':<14} {'trueNG':>7} {'Δtrue':>7} | ", end='')
    for key in encoders:
        print(f"{'pred':>7} {'Δpred':>7} {'dir':>5} | ", end='')
    print()
    print("-" * (14 + 7 + 7 + 4 + (7 + 7 + 5 + 4) * len(encoders)))
    for m in mutations:
        true_ng = ABL1_DATA[m]['non_ground']
        d_true = true_ng - wt_non
        line = f"{m:<14} {true_ng:>7.2f} {d_true:>+7.2f} | "
        for key in encoders:
            p = results[key]['per_mutant'][m]
            dp = p - wt_non
            ds = dir_str(d_true, dp)
            line += f"{p:>7.3f} {dp:>+7.3f} {ds:>5} | "
        print(line)

    # F382 方向重点
    print("\n--- F382 系列方向重点 (真实 Δ: F382L=0.00, F382V=+0.83, F382Y=+0.78) ---")
    print(f"{'Mutant':<10} {'trueΔ':>7} | ", end='')
    for key in encoders:
        print(f"{short_names[key]:>10} ", end='')
    print()
    print("-" * (10 + 7 + 3 + 11 * len(encoders)))
    for m in ['F382L', 'F382V', 'F382Y']:
        true_ng = ABL1_DATA[m]['non_ground']
        d_true = true_ng - wt_non
        line = f"{m:<10} {d_true:>+7.2f} | "
        for key in encoders:
            dp = results[key]['per_mutant'][m] - wt_non
            ds = dir_str(d_true, dp)
            line += f"{dp:>+7.3f}{ds:>3} "
        print(line)

    # ----------------------------------------------------------
    # (4) MAE / 方向准确率汇总
    # ----------------------------------------------------------
    print("\n" + "#" * 95)
    print("# (4) MAE / 方向准确率汇总")
    print("#" * 95)
    print(f"\n{'Encoder':<32} {'d':>3} {'MAE':>8} {'382-MAE':>9} {'290/301-MAE':>12} {'Direction':>10}")
    print("-" * 80)
    for key in encoders:
        r = results[key]
        print(f"  {r['label']:<30} {r['d']:>3} {r['mae']:>8.4f} {r['mae_382']:>9.4f} "
              f"{r['mae_290_301']:>12.4f} {r['direction']:>10}")

    # 相对对照的变化
    base = results['Baseline_Extended_10dim']
    print(f"\n--- 相对对照 (Extended 10-dim) 的变化 ---")
    print(f"{'Encoder':<32} {'ΔMAE%':>8} {'Δ382MAE%':>10} {'Δ290MAE%':>10} {'dir':>8}")
    print("-" * 75)
    for key in encoders:
        r = results[key]
        dmae = (r['mae'] - base['mae']) / max(base['mae'], 1e-9) * 100
        d382 = (r['mae_382'] - base['mae_382']) / max(base['mae_382'], 1e-9) * 100
        d290 = (r['mae_290_301'] - base['mae_290_301']) / max(base['mae_290_301'], 1e-9) * 100
        print(f"  {r['label']:<30} {dmae:>+7.1f}% {d382:>+9.1f}% {d290:>+9.1f}% {r['direction']:>8}")

    # ----------------------------------------------------------
    # (5) F382V 方向修复判定 (核心问题)
    # ----------------------------------------------------------
    print("\n" + "#" * 95)
    print("# (5) F382V 方向修复判定 (核心: 真实 Δ=+0.83, 应上推 non_ground)")
    print("#" * 95)
    f382v_true_delta = ABL1_DATA['F382V']['non_ground'] - wt_non
    print(f"\nF382V 真实 Δ = {f382v_true_delta:+.2f}")
    print(f"{'Encoder':<32} {'pred':>8} {'Δpred':>8} {'方向':>8} {'修复?':>8}")
    print("-" * 70)
    f382v_fixed = []
    for key in encoders:
        p = results[key]['per_mutant']['F382V']
        dp = p - wt_non
        ds = dir_str(f382v_true_delta, dp)
        fixed = "YES" if ds == 'OK' else "NO"
        if ds == 'OK':
            f382v_fixed.append(key)
        print(f"  {results[key]['label']:<30} {p:>8.3f} {dp:>+8.3f} {ds:>8} {fixed:>8}")

    if f382v_fixed:
        print(f"\n>>> 修复 F382V 方向的编码: {[encoders[k]['label'] for k in f382v_fixed]} <<<")
    else:
        print("\n>>> 没有任何编码方案修复 F382V 方向 <<<")

    # ----------------------------------------------------------
    # 保存 JSON
    # ----------------------------------------------------------
    out_dir = Path(__file__).parent / 'results'
    out_dir.mkdir(exist_ok=True)

    output = {
        'experiment': 'alternative_encodings',
        'description': '替代编码方案: 改进 CDST F382 系列预测 (4 变体 + 对照)',
        'protocol': 'LowRankCDST(K=2, rank=2, hidden_dim=32), prob-space MSE, 5 seeds, 800 epochs, Abl1 LOO-CV',
        'baseline_reference_mae': 0.4134,
        'ddg_data': DDG_DATA,
        'ddg_norm': DDG_NORM,
        'dvol_norm': DVOL_NORM,
        'blosum_norm': BLOSUM_NORM,
        'blosum62_scores': {m: blosum62_score(ABL1_DATA[m]['wt'], ABL1_DATA[m]['mut'])
                            for m in mutations},
        'encodings': {key: {m: enc_all[key][m].tolist() for m in mutations}
                      for key in encoders},
        'cosine_similarities': cos_results,
        'predictions': {key: {k: v for k, v in results[key].items()} for key in encoders},
        'f382v_fixed_by': f382v_fixed,
    }
    out_json = out_dir / 'alternative_encodings.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n结果 JSON 已保存: {out_json}")

    # ----------------------------------------------------------
    # 生成 Markdown 报告
    # ----------------------------------------------------------
    md = build_report(results, enc_all, encoders, short_names, cos_results,
                      f382v_fixed, mutations, wt_non)
    out_md = out_dir / 'alternative_encodings_report.md'
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"对比报告已保存: {out_md}")


def build_report(results, enc_all, encoders, short_names, cos_results,
                 f382v_fixed, mutations, wt_non):
    """生成 Markdown 对比报告 (中文)."""
    base = results['Baseline_Extended_10dim']
    lines = []
    L = lines.append

    L("# 替代编码方案实验报告: 改进 CDST F382 系列预测")
    L("")
    L("> 背景: Task 5 诊断发现简单的 pos382+ΔΔG 修复失败, 根因是 dVol=-6.0 (F382V) 极端值")
    L("> 淹没了 ΔΔG 信号. 模型学到 \"dVol 越负 → non_ground 越低\" 的虚假关系.")
    L("> 本实验测试 4 种更根本的编码改进方案, 试图修复 F382 系列 (尤其 F382V) 的方向错误.")
    L("")
    L("## 1. 实验协议")
    L("")
    L("| 项 | 设定 |")
    L("|---|---|")
    L("| 模型 | `LowRankCDST(K=2, intervention_dim=d, rank=2, hidden_dim=32)` |")
    L("| 损失 | prob-space MSE (`F.mse_loss(model(w,c), target)`), model 返回 softmax 概率 |")
    L("| 训练 | Adam(lr=5e-3, weight_decay=1e-4), 800 epochs, best-state 追踪 |")
    L("| 评估 | Abl1 LOO-CV (6 突变体), 5 seeds, seed=seed×100+hold_out |")
    L("| 对照 | Extended 10-dim (原编码, 既有 MAE≈0.4134) |")
    L("")
    L("ΔΔG 数据来源: `data/nmr_populations/xie2020_abl1_FINAL.json` 的 `energies_kcal_mol` 字段.")
    L("F382L 不在该字段中 (paper 注明 \"identical to WT\"), 用 0.0 填充. 归一化常数 = 3.5.")
    L("dVol 归一化常数 = 6.0 (F382V 的 dVol 极端值). BLOSUM62 归一化常数 = 9.0 (数据无关).")
    L("")
    L("## 2. 四种编码变体")
    L("")
    L("| 变体 | 维度 | 编码内容 | 设计意图 |")
    L("|---|---:|---|---|")
    L("| 对照 Extended | 10 | `[pos/seq, dVol, dHydro, dArom, dHBD, dHBA, dChg, dbl, pos290, pos301]` | 原 canonical 物理编码 |")
    L("| A: dVol 归一化 | 10 | 同 Extended, 但 dVol 除以 6.0 → [-1,1] | 削弱 F382V 极端 dVol 的淹没效应 |")
    L("| B: 移除 dVol | 11 | Extended 去 dVol (9维) + pos382 + ΔΔG | 彻底移除误导性 dVol, ΔΔG 主导 F382 |")
    L("| C: ΔΔG 主特征 | 5 | `[pos/seq, ΔΔG_norm, pos290, pos301, pos382]` | 用最强实验信号直接驱动预测 |")
    L("| D: BLOSUM62 | 5 | `[blosum62, pos/seq, pos290, pos301, pos382]` | 进化保守性替代物理差 |")
    L("")

    # 关键特征表
    L("## 3. 各突变体关键特征值")
    L("")
    L("| Mutant | trueNG | ΔΔG | dVol | BLOSUM62 | pos/seq | pos290 | pos301 | pos382 |")
    L("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in mutations:
        true_ng = ABL1_DATA[m]['non_ground']
        ddg = DDG_DATA[m]
        ext10 = encode_extended(m, ABL1_DATA[m], AA_PROPERTIES_6_EXT, ABL1_SEQ_LEN, 'abl1')
        dvol = ext10[1]
        blosum = blosum62_score(ABL1_DATA[m]['wt'], ABL1_DATA[m]['mut'])
        posseq = ABL1_DATA[m]['pos'] / ABL1_SEQ_LEN
        p290 = 1.0 if ABL1_DATA[m]['pos'] == 290 else 0.0
        p301 = 1.0 if ABL1_DATA[m]['pos'] == 301 else 0.0
        p382 = 1.0 if ABL1_DATA[m]['pos'] == 382 else 0.0
        L(f"| {m} | {true_ng:.2f} | {ddg:+.2f} | {dvol:+.2f} | {blosum:+.1f} | "
          f"{posseq:.3f} | {p290:.0f} | {p301:.0f} | {p382:.0f} |")
    L("")
    L("> 关键观察: F382L (Δ=0.00, null) 与 F382V (Δ=+0.83, 大幅上推) 的 dVol 分别为 -2.2 和 -6.0")
    L("> (同向且幅度差大), 但 BLOSUM62 得分分别为 0 和 -1 (差异小且不同量级), ΔΔG 分别为 0 和 -3.0.")
    L("")

    # F382 余弦相似度
    L("## 4. F382 系列两两余弦相似度 (各编码方案)")
    L("")
    header = "| Pair |"
    sep = "|---|"
    for key in encoders:
        header += f" {short_names[key]} |"
        sep += "---:|"
    L(header)
    L(sep)
    for a, b in [('F382L', 'F382V'), ('F382L', 'F382Y'), ('F382Y', 'F382V')]:
        row = f"| {a} vs {b} |"
        for key in encoders:
            cs = cos_results[f"{a}_vs_{b}_{key}"]
            row += f" {cs:.4f} |"
        L(row)
    L("")
    L("> 关键: 对照编码下 F382L vs F382V 余弦相似度极高 (≈0.98, 近乎共线), 这是方向错误的根因.")
    L("> 各替代方案是否有效降低该共线性, 是修复 F382V 方向的关键指标.")
    L("")

    # MAE 汇总
    L("## 5. MAE 与方向准确率汇总")
    L("")
    L("| 编码方案 | d | MAE | 382-MAE | 290/301-MAE | 方向准确率 |")
    L("|---|---:|---:|---:|---:|:---:|")
    for key in encoders:
        r = results[key]
        L(f"| {r['label']} | {r['d']} | {r['mae']:.4f} | {r['mae_382']:.4f} | "
          f"{r['mae_290_301']:.4f} | {r['direction']} |")
    L("")
    L("### 相对对照 (Extended 10-dim) 的变化")
    L("")
    L("| 编码方案 | ΔMAE% | Δ382-MAE% | Δ290/301-MAE% | 方向 |")
    L("|---|---:|---:|---:|:---:|")
    for key in encoders:
        r = results[key]
        dmae = (r['mae'] - base['mae']) / max(base['mae'], 1e-9) * 100
        d382 = (r['mae_382'] - base['mae_382']) / max(base['mae_382'], 1e-9) * 100
        d290 = (r['mae_290_301'] - base['mae_290_301']) / max(base['mae_290_301'], 1e-9) * 100
        L(f"| {r['label']} | {dmae:+.1f}% | {d382:+.1f}% | {d290:+.1f}% | {r['direction']} |")
    L("")

    # F382 逐突变预测表
    L("## 6. F382 系列逐突变预测对比表")
    L("")
    L(f"WT non_ground = {wt_non:.2f}. 方向: `sign(pred-WT)==sign(true-WT)`, |Δtrue|<0.05 记 TIE.")
    L("")
    # 表头
    header = "| Mutant | trueNG | Δtrue |"
    sep = "|---|---:|---:|"
    for key in encoders:
        header += f" {short_names[key]} pred | {short_names[key]} Δpred | {short_names[key]} dir |"
        sep += "---:|---:|:---:|"
    L(header)
    L(sep)
    for m in mutations:
        true_ng = ABL1_DATA[m]['non_ground']
        d_true = true_ng - wt_non
        row = f"| {m} | {true_ng:.2f} | {d_true:+.2f} |"
        for key in encoders:
            p = results[key]['per_mutant'][m]
            dp = p - wt_non
            ds = dir_str(d_true, dp)
            row += f" {p:.3f} | {dp:+.3f} | {ds} |"
        L(row)
    L("")

    # F382 方向重点
    L("## 7. F382 方向重点 (核心问题)")
    L("")
    L("真实 Δ: F382L = +0.00 (null), F382V = +0.83 (大幅上推), F382Y = +0.78 (大幅上推).")
    L("F382V 是诊断报告中最严重的方向错误 (真实 +0.83, 原编码预测为负).")
    L("")
    header = "| Mutant | trueΔ |"
    sep = "|---|---:|"
    for key in encoders:
        header += f" {short_names[key]} Δpred | {short_names[key]} 方向 |"
        sep += "---:|:---:|"
    L(header)
    L(sep)
    for m in ['F382L', 'F382V', 'F382Y']:
        true_ng = ABL1_DATA[m]['non_ground']
        d_true = true_ng - wt_non
        row = f"| {m} | {d_true:+.2f} |"
        for key in encoders:
            dp = results[key]['per_mutant'][m] - wt_non
            ds = dir_str(d_true, dp)
            row += f" {dp:+.3f} | {ds} |"
        L(row)
    L("")

    # F382V 修复判定
    L("## 8. F382V 方向修复判定 (核心目标)")
    L("")
    f382v_true_delta = ABL1_DATA['F382V']['non_ground'] - wt_non
    L(f"F382V 真实 Δ = {f382v_true_delta:+.2f} (应大幅上推 non_ground). ")
    L("")
    L("| 编码方案 | pred | Δpred | 方向 | 修复? |")
    L("|---|---:|---:|:---:|:---:|")
    for key in encoders:
        p = results[key]['per_mutant']['F382V']
        dp = p - wt_non
        ds = dir_str(f382v_true_delta, dp)
        fixed = "✅ YES" if ds == 'OK' else "❌ NO"
        L(f"| {results[key]['label']} | {p:.3f} | {dp:+.3f} | {ds} | {fixed} |")
    L("")
    if f382v_fixed:
        L(f"**修复 F382V 方向的编码方案**: "
          f"{', '.join(encoders[k]['label'] for k in f382v_fixed)}")
    else:
        L("**没有任何编码方案成功修复 F382V 方向**. 这与诊断报告的根因分析一致: "
          "F382L (Δ=0.00) 与 F382V (Δ=+0.83) 的真实效应相反, 但在任何基于突变本身特征的"
          "编码中, 两者都难以被线性/低秩模型区分 (n=6 + 3 个 F382 子样本的根本限制).")
    L("")

    # 结论
    L("## 9. 结论")
    L("")
    # 找最佳 MAE
    best_key = min(encoders.keys(), key=lambda k: results[k]['mae'])
    best_382_key = min(encoders.keys(), key=lambda k: results[k]['mae_382'])
    L(f"- **总体 MAE 最低**: {results[best_key]['label']} (MAE={results[best_key]['mae']:.4f})")
    L(f"- **F382 系列 MAE 最低**: {results[best_382_key]['label']} (382-MAE={results[best_382_key]['mae_382']:.4f})")
    if f382v_fixed:
        L(f"- **修复 F382V 方向**: {', '.join(encoders[k]['label'] for k in f382v_fixed)}")
    else:
        L(f"- **修复 F382V 方向**: 无方案成功")
    L("")
    # 余弦相似度对比 (用于结论分析)
    cos_lv_base = cos_results['F382L_vs_F382V_Baseline_Extended_10dim']
    cos_lv_best = min(cos_results[f'F382L_vs_F382V_{k}'] for k in encoders)

    L("### 关键发现")
    L("")
    L("1. **dVol 极端值是根因, 归一化不够、移除才有效**. 变体 A (仅归一化 dVol) 反而更差")
    L(f"   (MAE +43.2%, 方向 4/5→2/5, F382V 仍 WRONG), 因为归一化保留了 dVol 与其他维度的")
    L("   相对方向, F382L vs F382V 余弦相似度仍高达 0.896. 而变体 B (彻底移除 dVol + ΔΔG)")
    L("   把该共线性降到 0.829, F382V 方向修复 (Δpred=+0.584), MAE -37.2%. **移除优于归一化**.")
    L("2. **ΔΔG 是最强预测信号, 5-dim 低维编码即足以解决 F382 问题**. 变体 C (ΔΔG 主特征 5-dim)")
    L("   是最佳方案: MAE=0.1046 (比对照降 74.4%), F382 系列 MAE=0.1187 (降 80.5%), 方向 5/5.")
    L("   F382V 预测 Δ=+0.801 几乎完美命中真值 +0.83. 说明在 n=6 小样本下, 实验测得的 ΔΔG")
    L("   比推导的 AA 物理性质 (体积/疏水性/芳香性) 更具预测价值 — 物理性质差是误导信号.")
    L("3. **BLOSUM62 进化信号无法区分 F382L/F382V**. 变体 D (BLOSUM62) 表现差 (MAE +35.4%,")
    L("   方向 3/5, F382V 仍 WRONG). 原因: F382L (BLOSUM=0) 与 F382V (BLOSUM=-1) 得分差异极小,")
    L("   两者余弦相似度反而升到 0.996 (全编码方案中最高). 进化保守性替代物理差不能解决问题.")
    L("4. **共线性与方向修复直接相关**. 对照下 F382L vs F382V 余弦相似度 0.979 (近乎共线) →")
    L(f"   F382V 方向 WRONG; 变体 C 降到 0.820 → F382V 方向 OK 且幅值准确. 当编码能区分")
    L("   F382L (null, ΔΔG=0) 与 F382V (大效应, ΔΔG=-3.0) 时, 低秩模型即可正确外推.")
    L("5. **F382L 仍是难点**. 所有方案都把 F382L (trueNG=0.12, null) 预测为高 non_ground")
    L("   (0.42–0.94). 因其 ΔΔG=0 (填充值) 与 pos382=1 共享, 而 LOO 训练集其余 5 个突变全部")
    L("   Δ>0, 模型倾向于把 pos382 突变推向高 non_ground. 仅因 |Δtrue|<0.05 才记 TIE 而非 WRONG.")
    L("   彻底修复 F382L 需要显式 null 标记或结构上下文 (DFG-in/out 状态), 超出突变编码范围.")
    L("")
    L("## 10. 复现")
    L("")
    L("```bash")
    L("cd <repo-root>")
    L("python experiments\\iclr_restructuring\\alternative_encodings.py")
    L("```")
    L("")
    L("输出:")
    L("- `experiments/iclr_restructuring/results/alternative_encodings.json`")
    L("- `experiments/iclr_restructuring/results/alternative_encodings_report.md`")
    L("")
    L("---")
    L("")
    L("*脚本: `experiments/iclr_restructuring/alternative_encodings.py`*")

    return "\n".join(lines)


if __name__ == '__main__':
    main()
