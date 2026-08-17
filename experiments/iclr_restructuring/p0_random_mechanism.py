"""
P0 诊断脚本：解析 Random Gaussian 编码"反优"机制
=============================================

背景：
P0 双协议实验中 Random Gaussian (10-dim) 在 Abl1 和 Src 两个体系下 MAE
均优于 Extended (物理编码)，导致 narrative 失败。

假设：
Random Gaussian 编码在 LOO-CV 下退化为"训练集均值预测器"。其低 MAE 并
非来自编码质量，而是因为：
  1. Src 中 6/8 突变 non_active=1.0，训练均值天然接近 1.0
  2. Abl1 中 5/6 突变 non_ground ≥ 0.45，训练均值天然偏高
  3. 4/5 (Abl1) 和 7/7 (Src) 的 direction 也只是"高于 WT=0.12/0.28"
     这一平凡方向，并不携带区分信息

诊断指标（每个体系、每种编码）：
  - 各留一突变的预测值表
  - 预测值方差 vs 真实值方差（方差越小 → 越接近常数预测）
  - |预测 - 训练集均值| 的均值（接近 0 → 退化为均值预测器）
  - 预测值 vs 训练均值的 Pearson r 与 R²（接近 1 → 完全跟随训练均值）
  - "常数预测器" baseline：直接预测训练集均值时的 MAE
  - Random MAE / 常数预测器 MAE 比值（接近 1 → Random 没有额外信号）

协议：prob-space MSE（与 canonical 训练协议一致）
模型：SimpleCDST (linear, K=2)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import numpy as np
import torch

# 复用原 P0 实验代码以保证一致性
from experiments.iclr_restructuring.encoding_ablation_control import (
    SimpleCDST,
    ABL1_DATA, SRC_DATA,
    ABL1_WT_NON_GROUND, SRC_WT_NON_ACTIVE,
    ABL1_SEQ_LEN, SRC_SEQ_LEN,
    AA_PROPERTIES_6_EXT,
    encode_extended,
    encode_random_gaussian,
    train_predict,
)


# ============================================================
# 诊断版 LOO-CV：返回预测值 + 训练均值（排除留一突变）
# ============================================================
def run_loo_with_diagnostics(
    mutations, target_key, wt_non_target, seq_len,
    encoder_fn, d, n_seeds=5, loss_space='prob',
):
    """运行 LOO-CV，记录每个留一突变的预测值与训练集均值。

    返回 dict，包含 per-mutant 真实值、预测值、训练均值、训练 WT 等。
    """
    n = len(mutations)
    names = list(mutations.keys())

    # WT 分布：[ground/active, non_ground/non_active]
    wt_dist = np.array([1 - wt_non_target, wt_non_target])
    w_wt = np.tile(wt_dist, (n, 1))

    targets = np.array([
        [1 - mutations[m][target_key], mutations[m][target_key]] for m in names
    ])
    # 只取 non_ground/non_active 维度（即预测输出 index=1）
    true_vals = np.array([mutations[m][target_key] for m in names], dtype=float)

    encodings = np.array([encoder_fn(m, mutations[m]) for m in names])
    assert encodings.shape[1] == d, f"Encoding dim mismatch: {encodings.shape[1]} vs {d}"

    preds = {}
    train_means = {}   # 排除留一突变后训练集的均值
    train_stds = {}
    train_wt_vals = {}  # 训练集 WT (始终是同一 WT，但记录以便对照)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        pred = train_predict(
            w_wt[mask], encodings[mask], targets[mask],
            encodings[i:i + 1], w_wt[i:i + 1],
            d=d, n_seeds=n_seeds, seed_base=i, loss_space=loss_space,
        )
        preds[names[i]] = float(pred)
        train_means[names[i]] = float(np.mean(true_vals[mask]))
        train_stds[names[i]] = float(np.std(true_vals[mask]))
        train_wt_vals[names[i]] = float(wt_non_target)

    errors = {m: float(abs(preds[m] - true_vals[i]))
              for i, m in enumerate(names)}

    return {
        'names': names,
        'true_vals': true_vals.tolist(),
        'preds': preds,
        'train_means': train_means,
        'train_stds': train_stds,
        'train_wt': train_wt_vals,
        'errors': errors,
        'wt_non_target': float(wt_non_target),
    }


# ============================================================
# 统计分析：方差、相关性、常数预测器 baseline
# ============================================================
def analyze(result):
    """对单个 (system, encoder) 结果计算诊断统计量。"""
    names = result['names']
    true_arr = np.array(result['true_vals'])
    pred_arr = np.array([result['preds'][m] for m in names])
    mean_arr = np.array([result['train_means'][m] for m in names])
    wt = result['wt_non_target']

    # 预测值方差 vs 真实值方差
    var_pred = float(np.var(pred_arr))
    var_true = float(np.var(true_arr))
    var_ratio = var_pred / var_true if var_true > 0 else float('nan')

    # |pred - train_mean| 均值
    abs_pred_minus_mean = float(np.mean(np.abs(pred_arr - mean_arr)))

    # pred 与 train_mean 的 Pearson r 与 R²
    if np.std(pred_arr) > 1e-12 and np.std(mean_arr) > 1e-12:
        r = float(np.corrcoef(pred_arr, mean_arr)[0, 1])
    else:
        r = float('nan')
    r2 = r * r if not np.isnan(r) else float('nan')

    # pred 与 true 的相关性（衡量预测能力）
    if np.std(pred_arr) > 1e-12 and np.std(true_arr) > 1e-12:
        r_pred_true = float(np.corrcoef(pred_arr, true_arr)[0, 1])
    else:
        r_pred_true = float('nan')

    # 常数预测器 baseline：每个留一突变都预测训练集均值
    const_errors = np.abs(mean_arr - true_arr)
    const_mae = float(np.mean(const_errors))
    actual_mae = float(np.mean(np.abs(pred_arr - true_arr)))
    mae_ratio = actual_mae / const_mae if const_mae > 0 else float('nan')

    # 方向准确率 (与原协议一致，0.05 阈值)
    dir_correct = 0
    dir_total = 0
    for i, m in enumerate(names):
        true_delta = true_arr[i] - wt
        pred_delta = pred_arr[i] - wt
        if abs(true_delta) < 0.05:
            continue
        if np.sign(true_delta) == np.sign(pred_delta):
            dir_correct += 1
        dir_total += 1
    direction = f"{dir_correct}/{dir_total}"
    direction_pct = dir_correct / dir_total if dir_total > 0 else 0.0

    # 预测范围 (max - min)
    pred_range = float(pred_arr.max() - pred_arr.min())
    true_range = float(true_arr.max() - true_arr.min())

    return {
        'var_pred': var_pred,
        'var_true': var_true,
        'var_ratio_pred_over_true': var_ratio,
        'pred_range': pred_range,
        'true_range': true_range,
        'abs_pred_minus_train_mean_mean': abs_pred_minus_mean,
        'r_pred_vs_train_mean': r,
        'r2_pred_vs_train_mean': r2,
        'r_pred_vs_true': r_pred_true,
        'constant_predictor_mae': const_mae,
        'actual_mae': actual_mae,
        'mae_ratio_actual_over_constant': mae_ratio,
        'direction': direction,
        'direction_pct': direction_pct,
        'per_mutant_table': [
            {
                'mutant': m,
                'true': float(true_arr[i]),
                'pred': float(pred_arr[i]),
                'train_mean_excluding': float(mean_arr[i]),
                'train_std_excluding': float(result['train_stds'][m]),
                'wt_non_target': float(wt),
                'abs_err_pred': float(abs(pred_arr[i] - true_arr[i])),
                'abs_err_const': float(abs(mean_arr[i] - true_arr[i])),
                'abs_pred_minus_train_mean': float(abs(pred_arr[i] - mean_arr[i])),
            }
            for i, m in enumerate(names)
        ],
    }


# ============================================================
# 主入口
# ============================================================
def main():
    print("=" * 90)
    print("P0 诊断：Random Gaussian 编码反优机制分析 (prob-space MSE)")
    print("=" * 90)

    torch.manual_seed(0)
    np.random.seed(0)

    # Random Gaussian：固定 seed (=1，与原实验 gs=0 一致)
    # 并额外跑 5 seeds 平均以匹配原协议的稳健性
    RANDOM_FIXED_SEED = 1
    N_TRAIN_SEEDS = 5
    N_GAUSSIAN_DRAWS = 5

    systems = [
        {
            'name': 'Abl1',
            'mutations': ABL1_DATA,
            'target_key': 'non_ground',
            'wt_non_target': ABL1_WT_NON_GROUND,
            'seq_len': ABL1_SEQ_LEN,
            'system': 'abl1',
        },
        {
            'name': 'Src',
            'mutations': SRC_DATA,
            'target_key': 'non_active',
            'wt_non_target': SRC_WT_NON_ACTIVE,
            'seq_len': SRC_SEQ_LEN,
            'system': 'src',
        },
    ]

    summary = {}

    for sys_cfg in systems:
        name = sys_cfg['name']
        print(f"\n{'='*90}")
        print(f"SYSTEM: {name}")
        print(f"{'='*90}")

        # ---------- Extended (physics) ----------
        print(f"\n[Extended / {name}] Running LOO-CV ...")
        ext_loo = run_loo_with_diagnostics(
            sys_cfg['mutations'], sys_cfg['target_key'], sys_cfg['wt_non_target'],
            sys_cfg['seq_len'],
            encoder_fn=lambda n, d, cfg=sys_cfg: encode_extended(
                n, d, AA_PROPERTIES_6_EXT, cfg['seq_len'], system=cfg['system']),
            d=10, n_seeds=N_TRAIN_SEEDS, loss_space='prob',
        )
        ext_stats = analyze(ext_loo)

        # ---------- Random Gaussian (固定单 seed) ----------
        print(f"[Random Gaussian (fixed seed={RANDOM_FIXED_SEED}) / {name}] Running LOO-CV ...")
        rand_fixed_loo = run_loo_with_diagnostics(
            sys_cfg['mutations'], sys_cfg['target_key'], sys_cfg['wt_non_target'],
            sys_cfg['seq_len'],
            encoder_fn=lambda n, d, cfg=sys_cfg: encode_random_gaussian(
                n, d, seed=RANDOM_FIXED_SEED, seq_len=cfg['seq_len']),
            d=10, n_seeds=N_TRAIN_SEEDS, loss_space='prob',
        )
        rand_fixed_stats = analyze(rand_fixed_loo)

        # ---------- Random Gaussian (5 draws 平均，匹配原协议) ----------
        print(f"[Random Gaussian (avg over {N_GAUSSIAN_DRAWS} draws) / {name}] Running LOO-CV ...")
        rand_avg_loo_list = []
        for gs in range(N_GAUSSIAN_DRAWS):
            loo = run_loo_with_diagnostics(
                sys_cfg['mutations'], sys_cfg['target_key'], sys_cfg['wt_non_target'],
                sys_cfg['seq_len'],
                encoder_fn=lambda n, d, gs=gs, cfg=sys_cfg: encode_random_gaussian(
                    n, d, seed=gs * 97 + 1, seq_len=cfg['seq_len']),
                d=10, n_seeds=N_TRAIN_SEEDS, loss_space='prob',
            )
            rand_avg_loo_list.append(loo)

        # 平均每突变预测
        rand_avg_preds = {m: float(np.mean([loo['preds'][m] for loo in rand_avg_loo_list]))
                          for m in rand_avg_loo_list[0]['preds']}
        rand_avg_loo = dict(rand_avg_loo_list[0])
        rand_avg_loo['preds'] = rand_avg_preds
        rand_avg_stats = analyze(rand_avg_loo)

        # ---------- 打印 per-mutant 表 ----------
        print(f"\n--- {name} Per-Mutant 表 ---")
        print(f"{'Mutant':<25} {'true':>7} {'ExtPred':>9} {'RandPred':>9} "
              f"{'TrainMean':>10} {'WT':>6} {'ExtErr':>7} {'RandErr':>8} {'ConstErr':>9}")
        print("-" * 95)
        for i, m in enumerate(ext_loo['names']):
            true_v = ext_loo['true_vals'][i]
            ext_p = ext_loo['preds'][m]
            rand_p = rand_avg_loo['preds'][m]
            tmean = ext_loo['train_means'][m]
            wt = ext_loo['wt_non_target']
            ext_e = abs(ext_p - true_v)
            rand_e = abs(rand_p - true_v)
            const_e = abs(tmean - true_v)
            print(f"{m:<25} {true_v:>7.3f} {ext_p:>9.4f} {rand_p:>9.4f} "
                  f"{tmean:>10.4f} {wt:>6.3f} {ext_e:>7.4f} {rand_e:>8.4f} {const_e:>9.4f}")

        # ---------- 打印统计 ----------
        def fmt_stats(s, label):
            print(f"\n--- {label} 统计 ---")
            print(f"  预测值方差          var(pred)            = {s['var_pred']:.6f}")
            print(f"  真实值方差          var(true)            = {s['var_true']:.6f}")
            print(f"  方差比 pred/true                         = {s['var_ratio_pred_over_true']:.4f}")
            print(f"  预测范围 [max-min]                       = {s['pred_range']:.4f}")
            print(f"  真实范围 [max-min]                       = {s['true_range']:.4f}")
            print(f"  |pred - train_mean| 均值                 = {s['abs_pred_minus_train_mean_mean']:.4f}")
            print(f"  r(pred, train_mean)                      = {s['r_pred_vs_train_mean']:.4f}")
            print(f"  R²(pred, train_mean)                     = {s['r2_pred_vs_train_mean']:.4f}")
            print(f"  r(pred, true)                            = {s['r_pred_vs_true']:.4f}")
            print(f"  常数预测器 MAE (predict train mean)      = {s['constant_predictor_mae']:.4f}")
            print(f"  实际 MAE                                 = {s['actual_mae']:.4f}")
            print(f"  MAE 比值 actual/constant                 = {s['mae_ratio_actual_over_constant']:.4f}")
            print(f"  direction                                = {s['direction']} ({s['direction_pct']*100:.0f}%)")

        fmt_stats(ext_stats, f"{name} / Extended")
        fmt_stats(rand_avg_stats, f"{name} / Random Gaussian (5-draw avg)")
        fmt_stats(rand_fixed_stats, f"{name} / Random Gaussian (fixed seed={RANDOM_FIXED_SEED})")

        summary[name] = {
            'Extended': {
                'stats': ext_stats,
                'per_mutant': ext_loo,
            },
            'Random_Gaussian_avg5': {
                'stats': rand_avg_stats,
                'per_mutant': rand_avg_loo,
            },
            'Random_Gaussian_fixed_seed': {
                'seed': RANDOM_FIXED_SEED,
                'stats': rand_fixed_stats,
                'per_mutant': rand_fixed_loo,
            },
        }

    # ---------- 保存 JSON ----------
    out_dir = Path(__file__).parent / 'results'
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / 'p0_random_mechanism.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'p0_random_mechanism_diagnosis',
            'protocol': 'prob-space MSE, SimpleCDST (linear, K=2)',
            'description': 'Diagnose whether Random Gaussian encoding collapses to training-mean predictor',
            'n_train_seeds': N_TRAIN_SEEDS,
            'n_gaussian_draws_avg': N_GAUSSIAN_DRAWS,
            'random_fixed_seed': RANDOM_FIXED_SEED,
            'summary': summary,
        }, f, indent=2, default=float)
    print(f"\n诊断结果已保存到: {out_json}")

    # ---------- 总结论 ----------
    print("\n" + "=" * 90)
    print("总结论")
    print("=" * 90)
    for sys_name, sys_res in summary.items():
        ext = sys_res['Extended']['stats']
        rnd = sys_res['Random_Gaussian_avg5']['stats']
        print(f"\n[{sys_name}]")
        print(f"  Extended  MAE={ext['actual_mae']:.4f}  var(pred)={ext['var_pred']:.4f}  "
              f"r(pred,train_mean)={ext['r_pred_vs_train_mean']:.3f}  "
              f"const_ratio={ext['mae_ratio_actual_over_constant']:.3f}")
        print(f"  Random    MAE={rnd['actual_mae']:.4f}  var(pred)={rnd['var_pred']:.4f}  "
              f"r(pred,train_mean)={rnd['r_pred_vs_train_mean']:.3f}  "
              f"const_ratio={rnd['mae_ratio_actual_over_constant']:.3f}")
        print(f"  常数预测器 MAE={ext['constant_predictor_mae']:.4f}")
        verdict = "退化（≈均值预测器）" if (
            rnd['mae_ratio_actual_over_constant'] > 0.85 and
            rnd['var_ratio_pred_over_true'] < 0.5
        ) else "携带额外信号"
        print(f"  -> Random 判定: {verdict}")


if __name__ == '__main__':
    main()
