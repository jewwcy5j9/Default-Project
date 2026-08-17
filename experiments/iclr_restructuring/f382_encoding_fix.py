"""
F382 编码泛化失败修复实验。

背景:
  诊断发现 F382 系列方向全错的根因是编码泛化失败:
  1. F382L 与 F382V 的 6-dim AA 物理差余弦相似度 0.997 (几乎共线),
     但真实效应完全相反 (F382L Δ=0.00, F382V Δ=+0.83)
  2. F382 系列没有位置标记 (dim[8]/[9] 是 290/301 的 one-hot, F382 全为 0)
  3. F382Y 物理方向与 F382L/V 相反但效应相同

修复方案:
  在原 Extended 10-dim 基础上增加 2 个维度:
  - dim[10] = pos382 one-hot (突变位置在 382 时为 1, 否则 0)
  - dim[11] = ΔΔG 实验值 (归一化到 [-1, 1] 范围)

协议:
  - LowRankCDST(K=2, intervention_dim=d, rank=2, hidden_dim=32)
  - prob-space MSE, 5 seeds, 800 epochs (与 encoding_ablation.py 一致)
  - Abl1 LOO-CV (6 mutants)
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

# 归一化常数: 用 6 个突变体中 |ΔΔG| 的最大值 (3.5), 映射到 [-1, 1]
DDG_NORM = max(abs(v) for v in DDG_DATA.values())  # = 3.5


def encode_extended_v2(name, data, ddg_map, ddg_norm, table, seq_len, system='abl1'):
    """Extended v2 12-dim 编码 (在原 10-dim 基础上增加 pos382 + ΔΔG).

    [0-9]   原 Extended 10-dim (position, AA phys delta, double flag, pos290, pos301)
    [10]    pos382 one-hot (突变位置在 382 时为 1, 否则 0)
    [11]    ΔΔG 实验值 / ddg_norm (归一化到 [-1, 1])

    注意: 不修改原 encode_extended 函数, 而是调用它并追加 2 维。
    """
    enc10 = encode_extended(name, data, table, seq_len, system=system)  # (10,)
    enc12 = np.zeros(12)
    enc12[:10] = enc10

    # dim[10]: pos382 one-hot
    if data['pos'] == 382:
        enc12[10] = 1.0

    # dim[11]: ΔΔG 归一化到 [-1, 1]
    ddg = ddg_map.get(name, 0.0)
    enc12[11] = ddg / ddg_norm if ddg_norm > 0 else 0.0

    return enc12


# ============================================================
# LOO-CV 训练 (复刻 encoding_ablation.py 的 run_experiment 协议)
# ============================================================
def run_loo_cv(encoder_fn, d, n_seeds=5, n_epochs=800):
    """LowRankCDST + prob-space MSE 的 LOO-CV。

    与 encoding_ablation.py 的 run_experiment 协议完全一致:
      - LowRankCDST(K=2, intervention_dim=d, rank=2, hidden_dim=32)
      - Adam(lr=5e-3, weight_decay=1e-4)
      - n_epochs=800, best-state 追踪 (按 train loss)
      - n_seeds=5, seed = seed*100 + hold_out
      - loss = F.mse_loss(model(w,c), target)  # prob-space
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

    cdst_mean = {m: float(np.mean(v)) for m, v in all_preds.items()}
    return cdst_mean, c


def compute_metrics(preds):
    """计算 MAE / 方向准确率 / F382 子集 MAE。"""
    mutations = list(ABL1_DATA.keys())
    wt_non = ABL1_WT_NON_GROUND
    errors = {m: abs(preds[m] - ABL1_DATA[m]['non_ground']) for m in mutations}
    mae = float(np.mean(list(errors.values())))

    # 方向: sign(pred - wt) == sign(true - wt), tie (|Δtrue|<0.05) 跳过
    dir_correct, dir_total = 0, 0
    for m in mutations:
        d_true = ABL1_DATA[m]['non_ground'] - wt_non
        d_pred = preds[m] - wt_non
        if abs(d_true) < 0.05:
            continue
        dir_total += 1
        if np.sign(d_true) == np.sign(d_pred):
            dir_correct += 1

    f382 = ['F382L', 'F382Y', 'F382V']
    mae_382 = float(np.mean([errors[m] for m in f382]))
    mae_290_301 = float(np.mean([errors[m] for m in mutations if m not in f382]))

    return {
        'mae': mae,
        'mae_382': mae_382,
        'mae_290_301': mae_290_301,
        'direction': f"{dir_correct}/{dir_total}",
        'direction_pct': dir_correct / dir_total if dir_total > 0 else 0.0,
        'errors': errors,
    }


def cosine_sim(a, b):
    """余弦相似度。"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 90)
    print("F382 编码泛化失败修复实验")
    print("方案: Extended 10-dim + pos382 one-hot + ΔΔG  ->  12-dim (v2)")
    print("协议: LowRankCDST, prob-space MSE, 5 seeds, 800 epochs, Abl1 LOO-CV")
    print("=" * 90)

    torch.manual_seed(0)
    np.random.seed(0)

    mutations = list(ABL1_DATA.keys())

    # ----------------------------------------------------------
    # (1) 编码向量对比: 10-dim vs 12-dim, 重点 F382L/V/Y
    # ----------------------------------------------------------
    print("\n" + "#" * 90)
    print("# (1) 编码向量对比: Extended 10-dim  vs  Extended v2 12-dim")
    print("#" * 90)

    enc10_all = {m: encode_extended(m, ABL1_DATA[m], AA_PROPERTIES_6_EXT, ABL1_SEQ_LEN, 'abl1')
                 for m in mutations}
    enc12_all = {m: encode_extended_v2(m, ABL1_DATA[m], DDG_DATA, DDG_NORM,
                                       AA_PROPERTIES_6_EXT, ABL1_SEQ_LEN, 'abl1')
                 for m in mutations}

    print("\n--- 各突变体 ΔΔG (kcal/mol) 与归一化值 ---")
    print(f"{'Mutant':<16} {'ΔΔG':>8} {'ΔΔG/3.5':>10}")
    print("-" * 36)
    for m in mutations:
        ddg = DDG_DATA[m]
        print(f"{m:<16} {ddg:>+8.2f} {ddg/DDG_NORM:>+10.4f}")

    print("\n--- 编码向量 (10-dim -> 12-dim) ---")
    feat10 = ['pos/seq', 'dVol', 'dHydro', 'dArom', 'dHBD', 'dHBA', 'dChg',
              'dbl', 'pos290', 'pos301']
    feat12 = feat10 + ['pos382', 'ddg']
    print(f"{'Mutant':<16} {'trueNG':>7} | {'--- 10-dim ---':>40} | {'+pos382':>7} {'+ddg':>7}")
    print("-" * 95)
    for m in mutations:
        true_ng = ABL1_DATA[m]['non_ground']
        e10 = enc10_all[m]
        e12 = enc12_all[m]
        e10_str = " ".join(f"{x:>6.3f}" for x in e10)
        print(f"{m:<16} {true_ng:>7.2f} | {e10_str} | {e12[10]:>7.3f} {e12[11]:>+7.3f}")

    # 余弦相似度: F382 系列两两 (10-dim 全维 / 6-dim AA 物理差 / 12-dim 全维)
    print("\n--- F382 系列两两余弦相似度 (修复前 vs 修复后) ---")
    f382 = ['F382L', 'F382Y', 'F382V']
    pairs = [('F382L', 'F382V'), ('F382L', 'F382Y'), ('F382Y', 'F382V')]
    print(f"{'Pair':<22} {'cos(10-dim)':>14} {'cos(6-dim phys)':>16} {'cos(12-dim v2)':>16}")
    print("-" * 70)
    cos_results = {}
    for a, b in pairs:
        c10 = cosine_sim(enc10_all[a], enc10_all[b])
        c6 = cosine_sim(enc10_all[a][1:7], enc10_all[b][1:7])
        c12 = cosine_sim(enc12_all[a], enc12_all[b])
        cos_results[f"{a}_vs_{b}"] = {'cos_10dim': c10, 'cos_6dim_phys': c6, 'cos_12dim': c12}
        print(f"{a+' vs '+b:<22} {c10:>14.4f} {c6:>16.4f} {c12:>16.4f}")

    # ----------------------------------------------------------
    # (2) LOO-CV: 10-dim (对照) vs 12-dim (修复)
    # ----------------------------------------------------------
    print("\n" + "#" * 90)
    print("# (2) LOO-CV 预测: Extended 10-dim (对照)  vs  Extended v2 12-dim (修复)")
    print("#" * 90)

    print("\n[1/2] Extended 10-dim (对照) ...")
    preds10, _ = run_loo_cv(
        encoder_fn=lambda name, data: encode_extended(name, data, AA_PROPERTIES_6_EXT, ABL1_SEQ_LEN, 'abl1'),
        d=10,
    )
    metrics10 = compute_metrics(preds10)
    print(f"  MAE={metrics10['mae']:.4f}  382-MAE={metrics10['mae_382']:.4f}  "
          f"290/301-MAE={metrics10['mae_290_301']:.4f}  dir={metrics10['direction']}")

    print("\n[2/2] Extended v2 12-dim (修复: +pos382 +ΔΔG) ...")
    preds12, _ = run_loo_cv(
        encoder_fn=lambda name, data: encode_extended_v2(name, data, DDG_DATA, DDG_NORM,
                                                        AA_PROPERTIES_6_EXT, ABL1_SEQ_LEN, 'abl1'),
        d=12,
    )
    metrics12 = compute_metrics(preds12)
    print(f"  MAE={metrics12['mae']:.4f}  382-MAE={metrics12['mae_382']:.4f}  "
          f"290/301-MAE={metrics12['mae_290_301']:.4f}  dir={metrics12['direction']}")

    # ----------------------------------------------------------
    # (3) 预测对比表 (重点 F382L/V/Y 方向)
    # ----------------------------------------------------------
    print("\n" + "#" * 90)
    print("# (3) 预测对比表 (重点 F382L/V/Y 方向)")
    print("#" * 90)

    wt_non = ABL1_WT_NON_GROUND
    print(f"\n{'Mutant':<16} {'trueNG':>7} {'Δtrue':>7} | "
          f"{'pred10':>8} {'Δpred10':>8} {'dir10':>6} | "
          f"{'pred12':>8} {'Δpred12':>8} {'dir12':>6}")
    print("-" * 95)
    for m in mutations:
        true_ng = ABL1_DATA[m]['non_ground']
        d_true = true_ng - wt_non
        p10 = preds10[m]
        p12 = preds12[m]
        d10 = p10 - wt_non
        d12 = p12 - wt_non

        def dir_str(d_true, d_pred):
            if abs(d_true) < 0.05:
                return "TIE"
            return "OK" if np.sign(d_true) == np.sign(d_pred) else "WRONG"

        print(f"{m:<16} {true_ng:>7.2f} {d_true:>+7.2f} | "
              f"{p10:>8.3f} {d10:>+8.3f} {dir_str(d_true, d10):>6} | "
              f"{p12:>8.3f} {d12:>+8.3f} {dir_str(d_true, d12):>6}")

    # F382 系列方向重点
    print("\n--- F382 系列方向重点 ---")
    for m in ['F382L', 'F382V', 'F382Y']:
        true_ng = ABL1_DATA[m]['non_ground']
        d_true = true_ng - wt_non
        d10 = preds10[m] - wt_non
        d12 = preds12[m] - wt_non
        print(f"  {m}: trueΔ={d_true:+.2f}  "
              f"10-dim predΔ={d10:+.3f} ({'OK' if (abs(d_true)<0.05 or np.sign(d_true)==np.sign(d10)) else 'WRONG'})  "
              f"12-dim predΔ={d12:+.3f} ({'OK' if (abs(d_true)<0.05 or np.sign(d_true)==np.sign(d12)) else 'WRONG'})")

    # ----------------------------------------------------------
    # (4) MAE / 方向准确率汇总
    # ----------------------------------------------------------
    print("\n" + "#" * 90)
    print("# (4) MAE / 方向准确率汇总")
    print("#" * 90)

    print(f"\n{'Encoder':<28} {'MAE':>8} {'382-MAE':>9} {'290/301-MAE':>12} {'Direction':>10}")
    print("-" * 75)
    print(f"  {'Extended 10-dim (对照)':<26} {metrics10['mae']:>8.4f} {metrics10['mae_382']:>9.4f} "
          f"{metrics10['mae_290_301']:>12.4f} {metrics10['direction']:>10}")
    print(f"  {'Extended v2 12-dim (修复)':<26} {metrics12['mae']:>8.4f} {metrics12['mae_382']:>9.4f} "
          f"{metrics12['mae_290_301']:>12.4f} {metrics12['direction']:>10}")

    print(f"\nMAE 变化:    总体 {metrics10['mae']:.4f} -> {metrics12['mae']:.4f} "
          f"({(metrics12['mae']-metrics10['mae'])/metrics10['mae']*100:+.1f}%)")
    print(f"            F382 {metrics10['mae_382']:.4f} -> {metrics12['mae_382']:.4f} "
          f"({(metrics12['mae_382']-metrics10['mae_382'])/max(metrics10['mae_382'],1e-9)*100:+.1f}%)")
    print(f"方向准确率:  {metrics10['direction']} -> {metrics12['direction']}")

    # ----------------------------------------------------------
    # 保存结果
    # ----------------------------------------------------------
    out_dir = Path(__file__).parent / 'results'
    out_dir.mkdir(exist_ok=True)

    output = {
        'experiment': 'F382_encoding_fix',
        'description': '修复 F382 系列编码泛化失败: 增加 pos382 one-hot + ΔΔG 维度',
        'protocol': 'LowRankCDST(K=2, rank=2, hidden_dim=32), prob-space MSE, 5 seeds, 800 epochs, LOO-CV',
        'ddg_data': DDG_DATA,
        'ddg_norm': DDG_NORM,
        'encodings': {
            'ext_10dim': {m: enc10_all[m].tolist() for m in mutations},
            'ext_v2_12dim': {m: enc12_all[m].tolist() for m in mutations},
            'feature_names_10': feat10,
            'feature_names_12': feat12,
        },
        'cosine_similarities': cos_results,
        'predictions': {
            'ext_10dim': {'per_mutant': preds10, **metrics10},
            'ext_v2_12dim': {'per_mutant': preds12, **metrics12},
        },
    }
    out_json = out_dir / 'f382_encoding_fix.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n结果已保存: {out_json}")


if __name__ == '__main__':
    main()
