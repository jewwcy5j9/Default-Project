"""
F382 系列突变体方向全错根因诊断脚本。

背景:
  Abl1 激酶 6 个突变体 LOO-CV 中, F382L / F382Y / F382V 三个突变体方向全部预测反了,
  而 M290L / L301I / M290L_L301I 方向基本正确。

本脚本做四件事:
  a. 提取并对比 6 个突变体的 Extended 10-dim 编码向量
  b. 对每个留一突变 (重点 F382L/V/Y) 训练 SimpleCDST 并保存学到的 T 矩阵
  c. 分析 F382 系列编码向量与 M290L/L301I 的差异 (欧氏距离 / 余弦相似度)
  d. 分析 T 矩阵在留出 F382 突变时学到了什么 (是否过拟合到 M290/L301 模式)

协议: prob-space MSE (与 canonical_results.json 一致), SimpleCDST 纯线性模型。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 encoding_ablation_control 中的定义, 保证与 canonical 完全一致
from encoding_ablation_control import (
    SimpleCDST,
    ABL1_DATA, ABL1_WT_NON_GROUND, ABL1_SEQ_LEN,
    AA_PROPERTIES_6_EXT,
    encode_extended,
)


# ============================================================
# 工具: 训练并返回 T 矩阵 (复刻 encoding_ablation_control.train_predict 协议)
# ============================================================
def train_with_T(w_wt_train, c_train, w_target_train,
                 d, n_seeds=5, n_epochs=800, seed_base=0, loss_space='prob'):
    """LOO-CV 训练, 额外返回学到的 T 矩阵 (跨 seed 平均)。

    完全复刻 encoding_ablation_control.train_predict 的训练协议:
      - SimpleCDST(K=2, intervention_dim=d)
      - Adam(lr=5e-3, weight_decay=1e-4)
      - n_epochs=800, best-state 追踪 (按 train loss)
      - n_seeds=5, seed = seed*100 + seed_base
    """
    T_list = []
    for seed in range(n_seeds):
        torch.manual_seed(seed * 100 + seed_base)
        np.random.seed(seed * 100 + seed_base)

        model = SimpleCDST(K=2, intervention_dim=d)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)

        w_t = torch.FloatTensor(w_wt_train)
        c_t = torch.FloatTensor(c_train)
        wt_t = torch.FloatTensor(w_target_train)
        target_log = torch.FloatTensor(np.log(np.clip(w_target_train, 1e-5, None)))

        best_loss = float('inf')
        best_state = None
        for _ in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            pred_log = model(w_t, c_t)
            if loss_space == 'log':
                loss = F.mse_loss(pred_log, target_log)
            else:
                pred = torch.exp(pred_log)
                loss = F.mse_loss(pred, wt_t)
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        T_list.append(model.T.detach().numpy().copy())  # shape (K=2, d)

    T_avg = np.mean(T_list, axis=0)  # (2, d)
    return T_avg, T_list


def predict_with_T(T, w_wt_test, c_test):
    """用给定 T 矩阵做预测, 返回 non_ground 概率。"""
    with torch.no_grad():
        model = SimpleCDST(K=2, intervention_dim=T.shape[1])
        model.T.data = torch.FloatTensor(T)
        model.eval()
        pred_log = model(
            torch.FloatTensor(w_wt_test),
            torch.FloatTensor(c_test)
        )
        pred = torch.exp(pred_log).numpy()
    return float(pred[0, 1])  # non_ground


# ============================================================
# 主诊断流程
# ============================================================
def main():
    np.random.seed(0)
    torch.manual_seed(0)

    mutations = ABL1_DATA
    names = list(mutations.keys())
    n = len(names)
    target_key = 'non_ground'
    wt_non_target = ABL1_WT_NON_GROUND
    seq_len = ABL1_SEQ_LEN

    # WT 分布: [ground, non_ground]
    wt_dist = np.array([1 - wt_non_target, wt_non_target])
    w_wt = np.tile(wt_dist, (n, 1))
    targets = np.array([[1 - mutations[m][target_key], mutations[m][target_key]] for m in names])
    encodings = np.array([encode_extended(m, mutations[m], AA_PROPERTIES_6_EXT, seq_len, system='abl1')
                          for m in names])

    print("=" * 90)
    print("F382 系列突变体方向全错根因诊断")
    print("=" * 90)
    print(f"WT non_ground = {wt_non_target}")
    print(f"序列长度 = {seq_len}, 编码维度 = {encodings.shape[1]}")
    print(f"突变体数 = {n}: {names}")

    # ----------------------------------------------------------
    # (a) 编码向量对比表
    # ----------------------------------------------------------
    print("\n" + "#" * 90)
    print("# (a) 6 个突变体 Extended 10-dim 编码向量对比")
    print("#" * 90)

    feat_names = ['pos/seq', 'dVolume', 'dHydroph', 'dAromat', 'dHBD', 'dHBA', 'dCharge',
                  'dblFlag', 'pos290', 'pos301']
    header = f"{'Mutant':<16} {'trueNG':>7} {'dNG':>6}  " + " ".join(f"{fn:>8}" for fn in feat_names)
    print(header)
    print("-" * len(header))
    for i, m in enumerate(names):
        true_ng = mutations[m][target_key]
        d_ng = true_ng - wt_non_target
        enc_str = " ".join(f"{encodings[i, j]:>8.3f}" for j in range(10))
        print(f"{m:<16} {true_ng:>7.2f} {d_ng:>+6.2f}  {enc_str}")

    print("\n编码维度说明:")
    print("  [0] pos/seq    = 位置 / 序列长度 (连续值, 382/534=0.715)")
    print("  [1-6] AA 物理量差 / 5.0 (volume, hydrophobicity, aromaticity, HBD, HBA, charge)")
    print("  [7] dblFlag    = 双突变标记 (仅 M290L_L301I = 1)")
    print("  [8] pos290     = 位置 290 标记 (仅 M290L = 1)")
    print("  [9] pos301     = 位置 301 标记 (仅 L301I / M290L_L301I = 1)")
    print("  >>> 关键: F382 系列在 [8][9] 全为 0, 没有位置专属标记!")

    # ----------------------------------------------------------
    # (b) LOO-CV: 对每个留一突变训练 SimpleCDST, 保存 T 矩阵 + 预测
    # ----------------------------------------------------------
    print("\n" + "#" * 90)
    print("# (b) LOO-CV: 每个留一突变训练 SimpleCDST, 保存 T 矩阵")
    print("#" * 90)

    loo_results = {}
    print(f"\n{'留出突变':<16} {'predNG':>8} {'trueNG':>8} {'方向':>6} {'|err|':>8} {'trainMSE':>10}")
    print("-" * 70)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        held = names[i]
        T_avg, T_list = train_with_T(
            w_wt[mask], encodings[mask], targets[mask],
            d=10, n_seeds=5, seed_base=i, loss_space='prob',
        )
        pred = predict_with_T(T_avg, w_wt[i:i+1], encodings[i:i+1])
        true_ng = mutations[held][target_key]
        d_true = true_ng - wt_non_target
        d_pred = pred - wt_non_target
        # 方向正确性: pred 与 true 相对 WT 同号
        if abs(d_true) < 0.05:
            dir_str = "TIE"
        elif np.sign(d_true) == np.sign(d_pred):
            dir_str = "OK"
        else:
            dir_str = "WRONG"

        # 训练集 MSE (in-sample fit)
        train_preds = []
        for j in np.where(mask)[0]:
            train_preds.append(predict_with_T(T_avg, w_wt[j:j+1], encodings[j:j+1]))
        train_errs = [abs(p - mutations[names[j]][target_key]) for p, j in zip(train_preds, np.where(mask)[0])]
        train_mae = float(np.mean(train_errs))

        loo_results[held] = {
            'T_avg': T_avg, 'T_list': T_list, 'pred': pred, 'true': true_ng,
            'd_true': d_true, 'd_pred': d_pred, 'dir': dir_str,
            'train_mae': train_mae,
            'train_preds': {names[j]: p for p, j in zip(train_preds, np.where(mask)[0])},
        }
        print(f"{held:<16} {pred:>8.3f} {true_ng:>8.2f} {dir_str:>6} {abs(pred-true_ng):>8.3f} {train_mae:>10.3f}")

    # ----------------------------------------------------------
    # (c) F382 系列编码向量与 M290L/L301I 的差异
    # ----------------------------------------------------------
    print("\n" + "#" * 90)
    print("# (c) 编码向量差异分析 (欧氏距离 / 余弦相似度)")
    print("#" * 90)

    print("\n--- 编码向量全貌 (10-dim) ---")
    for i, m in enumerate(names):
        print(f"  {m:<16}: {np.array2string(encodings[i], precision=3, separator=', ')}")

    print("\n--- 两两欧氏距离 (全部 10 维) ---")
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_mat[i, j] = np.linalg.norm(encodings[i] - encodings[j])
    print(f"{'':<16}", end="")
    for m in names:
        print(f"{m:>16}", end="")
    print()
    for i, m in enumerate(names):
        print(f"{m:<16}", end="")
        for j in range(n):
            print(f"{dist_mat[i, j]:>16.3f}", end="")
        print()

    print("\n--- 两两余弦相似度 (全部 10 维) ---")
    cos_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_mat[i, j] = np.dot(encodings[i], encodings[j]) / (
                np.linalg.norm(encodings[i]) * np.linalg.norm(encodings[j]) + 1e-12)
    print(f"{'':<16}", end="")
    for m in names:
        print(f"{m:>16}", end="")
    print()
    for i, m in enumerate(names):
        print(f"{m:<16}", end="")
        for j in range(n):
            print(f"{cos_mat[i, j]:>16.3f}", end="")
        print()

    # 只看 AA 物理差 (维度 1-6), 排除位置/标记干扰
    print("\n--- 仅 AA 物理差 (dim 1-6) 的余弦相似度 ---")
    enc_phys = encodings[:, 1:7]
    cos_phys = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ni = np.linalg.norm(enc_phys[i]) + 1e-12
            nj = np.linalg.norm(enc_phys[j]) + 1e-12
            cos_phys[i, j] = np.dot(enc_phys[i], enc_phys[j]) / (ni * nj)
    print(f"{'':<16}", end="")
    for m in names:
        print(f"{m:>16}", end="")
    print()
    for i, m in enumerate(names):
        print(f"{m:<16}", end="")
        for j in range(n):
            print(f"{cos_phys[i, j]:>16.3f}", end="")
        print()

    print("\n--- F382 系列与 M290L/L301I 的关键差异 (全 10 维距离) ---")
    ref_mutants = ['M290L', 'L301I', 'M290L_L301I']
    f382_mutants = ['F382L', 'F382Y', 'F382V']
    for f in f382_mutants:
        i = names.index(f)
        print(f"  {f} (true Δ={loo_results[f]['d_true']:+.2f}):")
        for r in ref_mutants:
            j = names.index(r)
            print(f"    vs {r:<14} (true Δ={loo_results[r]['d_true']:+.2f}):  "
                  f"dist={dist_mat[i, j]:.3f}  cos={cos_mat[i, j]:.3f}  cos_phys={cos_phys[i, j]:.3f}")

    # ----------------------------------------------------------
    # (d) T 矩阵分析: 留出 F382 突变时学到了什么
    # ----------------------------------------------------------
    print("\n" + "#" * 90)
    print("# (d) T 矩阵分析: 留出各突变时 T 学到了什么")
    print("#" * 90)

    print("\n说明: T 形状 (2, 10), T[1, :] 影响 non_ground 状态 (越大越推高 non_ground)")
    print("      delta_non_ground = c · T[1, :] - c · T[0, :] (相对 ground 的推力)")

    # 对每个留出突变, 打印 T 矩阵和它对 held-out 编码的 delta
    print(f"\n{'留出突变':<16} {'c·T[1]':>8} {'c·T[0]':>8} {'Δpush':>8} {'pred':>8} {'true':>6} {'dir':>6}")
    print("-" * 70)
    for held in names:
        T = loo_results[held]['T_avg']
        i = names.index(held)
        c = encodings[i]
        c_T1 = float(np.dot(c, T[1, :]))
        c_T0 = float(np.dot(c, T[0, :]))
        push = c_T1 - c_T0
        print(f"{held:<16} {c_T1:>8.3f} {c_T0:>8.3f} {push:>+8.3f} "
              f"{loo_results[held]['pred']:>8.3f} {loo_results[held]['true']:>6.2f} "
              f"{loo_results[held]['dir']:>6}")

    # 重点: 留出 F382L/V/Y 时的 T 矩阵明细
    print("\n--- 留出 F382L 时的 T 矩阵 (训练集 = M290L, L301I, M290L_L301I, F382Y, F382V) ---")
    for held in f382_mutants:
        T = loo_results[held]['T_avg']
        print(f"\n  留出 {held} (训练集 5 个突变):")
        print(f"    T[0, ground ] = {np.array2string(T[0], precision=4, separator=', ')}")
        print(f"    T[1, nonGnd ] = {np.array2string(T[1], precision=4, separator=', ')}")
        # 训练集中 4 个非 held-out 突变的 delta 推力
        print(f"    训练集推力 (c·T[1] - c·T[0]):")
        for m2 in names:
            if m2 == held:
                continue
            j = names.index(m2)
            c = encodings[j]
            push = float(np.dot(c, T[1, :]) - np.dot(c, T[0, :]))
            true_d = loo_results[m2]['d_true']
            print(f"      {m2:<14} push={push:>+7.3f}  trueΔ={true_d:>+5.2f}  "
                  f"pred={loo_results[held]['train_preds'][m2]:.3f}")

    # 分析: 留出 F382L 时训练集的 true Δ 分布
    print("\n--- 留出各 F382 突变时, 训练集 true Δ 分布 (解释 T 为何学到特定方向) ---")
    for held in f382_mutants:
        train_deltas = [loo_results[m]['d_true'] for m in names if m != held]
        print(f"  留出 {held}: 训练集 true Δ = {train_deltas}, "
              f"均值={np.mean(train_deltas):+.3f}, "
              f"全部向上? {all(d > 0 for d in train_deltas)}")

    # 关键诊断: F382L 是唯一的 NULL 突变 (true Δ ≈ 0)
    print("\n--- 关键发现: F382L 是唯一的 NULL 突变 (true non_ground == WT) ---")
    print(f"  WT non_ground        = {wt_non_target}")
    for m in names:
        true_ng = mutations[m][target_key]
        d = true_ng - wt_non_target
        marker = " <<< NULL (唯一不偏移)" if abs(d) < 0.05 else ""
        print(f"  {m:<16} trueNG={true_ng:.2f}  Δ={d:+.2f}{marker}")

    # 编码维度对 non_ground 的"虚假"相关性
    print("\n--- 各编码维度与 true Δ 的相关性 (训练集内, LOO 视角) ---")
    true_deltas = np.array([mutations[m][target_key] - wt_non_target for m in names])
    for j, fn in enumerate(feat_names):
        corr = np.corrcoef(encodings[:, j], true_deltas)[0, 1]
        print(f"  dim[{j}] {fn:<10}: corr(Δ) = {corr:+.3f}")

    # ----------------------------------------------------------
    # 保存诊断结果到 JSON
    # ----------------------------------------------------------
    out_dir = Path(__file__).parent / 'results'
    out_dir.mkdir(exist_ok=True)

    # 编码向量 + T 矩阵的序列化
    diag_data = {
        'experiment': 'F382_direction_failure_diagnosis',
        'protocol': 'prob-space MSE, SimpleCDST linear, LOO-CV, n_seeds=5',
        'wt_non_ground': wt_non_target,
        'encodings': {m: encodings[i].tolist() for i, m in enumerate(names)},
        'feature_names': feat_names,
        'loo_results': {
            held: {
                'pred': loo_results[held]['pred'],
                'true': loo_results[held]['true'],
                'd_true': loo_results[held]['d_true'],
                'd_pred': loo_results[held]['d_pred'],
                'direction': loo_results[held]['dir'],
                'train_mae': loo_results[held]['train_mae'],
                'T_avg': loo_results[held]['T_avg'].tolist(),
                'train_preds': loo_results[held]['train_preds'],
            } for held in names
        },
        'distance_matrix': dist_mat.tolist(),
        'cosine_matrix': cos_mat.tolist(),
        'cosine_phys_matrix': cos_phys.tolist(),
    }
    out_json = out_dir / 'f382_diagnosis.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(diag_data, f, indent=2, default=float)
    print(f"\n诊断数据已保存: {out_json}")


if __name__ == '__main__':
    main()
