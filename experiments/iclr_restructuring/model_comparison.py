"""
SimpleCDST (linear) vs LowRankCDST (MLP) 公平对比脚本。

目的：在完全相同条件下（相同编码 Extended 10-dim、相同数据 Abl1 6-mutant、
相同损失函数 prob-space MSE、相同 LOO-CV 协议）对比两个模型变体，
以判定哪个是权威结果。

背景矛盾：
  - canonical_results.json  (encoding_ablation.py, LowRankCDST, 800 epochs): Abl1 MAE=0.4134
  - headline_FINAL.json     (headline_FINAL.py,   LowRankCDST, 1000 epochs): Abl1 MAE=0.5680
  两个结果都声称来自 LowRankCDST，但 MAE 差距 0.155。
  同时 encoding_ablation_control.py 引入 SimpleCDST（纯线性）声称更适合 n<10 的小样本。

本脚本测试矩阵：
  模型:  SimpleCDST (linear)  vs  LowRankCDST (MLP, rank=2)
  编码:  Extended 10-dim (canonical, src.canonical_encoding)
  损失:  prob-space MSE
  Epochs: 800 / 1000 / 1500  (检验 epoch 数对 LowRankCDST 稳定性的影响)
  Seeds: 5 seeds/hold_out, LOO-CV
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from src.models.low_rank_cdst import LowRankCDST
from src.canonical_encoding import encode_mutation


# ============================================================
# SimpleCDST: 纯线性模型 w' = softmax(log w + T @ c)
# 与 encoding_ablation_control.py 中的 SimpleCDST 完全一致
# ============================================================
class SimpleCDST(nn.Module):
    """w' = softmax(log(w) + T @ c). 纯线性，无隐藏层。"""
    def __init__(self, K: int, intervention_dim: int, **kwargs):
        super().__init__()
        self.T = nn.Parameter(torch.zeros(K, intervention_dim))

    def forward(self, w, c):
        log_w = torch.log(w.clamp(min=1e-8))
        delta = c @ self.T.T  # (batch, K)
        return F.log_softmax(log_w + delta, dim=-1)


# ============================================================
# NMR FINAL 数据（与 canonical_results.json / encoding_ablation.py 一致）
# ============================================================
NMR_DATA = {
    'M290L':       {'non_ground': 0.45},
    'L301I':       {'non_ground': 0.75},
    'M290L_L301I': {'non_ground': 0.92},
    'F382L':       {'non_ground': 0.12},
    'F382Y':       {'non_ground': 0.90},
    'F382V':       {'non_ground': 0.95},
}
WT_NON_GROUND = 0.12
MUTATIONS = list(NMR_DATA.keys())


def prepare_data():
    """准备 LOO-CV 数据，使用 canonical Extended 10-dim 编码。"""
    n = len(MUTATIONS)
    wt = np.array([1 - WT_NON_GROUND, WT_NON_GROUND])
    w_wt = np.tile(wt, (n, 1))
    w_target = np.array([[1 - NMR_DATA[m]['non_ground'],
                          NMR_DATA[m]['non_ground']] for m in MUTATIONS])
    c = np.array([encode_mutation(m) for m in MUTATIONS])
    return w_wt, w_target, c


# ============================================================
# 训练函数
# ============================================================
def train_loo(model_type, w_wt, w_target, c, n_epochs=800, n_seeds=5):
    """LOO-CV 训练，返回 per-mutant 预测（non_ground 概率）和 MAE。

    model_type: 'simple' 或 'lowrank'
    使用 prob-space MSE 损失，与 encoding_ablation.py / headline_FINAL.py 一致。
    """
    n = len(MUTATIONS)
    d = c.shape[1]
    all_preds = {m: [] for m in MUTATIONS}

    for hold_out in range(n):
        mask = np.ones(n, dtype=bool)
        mask[hold_out] = False
        for seed in range(n_seeds):
            torch.manual_seed(seed * 100 + hold_out)
            np.random.seed(seed * 100 + hold_out)

            if model_type == 'simple':
                model = SimpleCDST(K=2, intervention_dim=d)
            else:
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
                if model_type == 'simple':
                    pred_log = model(w_t, c_t)
                    pred = torch.exp(pred_log)
                else:
                    pred = model(w_t, c_t)
                loss = F.mse_loss(pred, wt_t)
                loss.backward()
                optimizer.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                w_test = torch.FloatTensor(w_wt[hold_out:hold_out + 1])
                c_test = torch.FloatTensor(c[hold_out:hold_out + 1])
                if model_type == 'simple':
                    pred = torch.exp(model(w_test, c_test)).numpy()[0, 1]
                else:
                    pred = model(w_test, c_test).numpy()[0, 1]
            all_preds[MUTATIONS[hold_out]].append(float(pred))

    mean_preds = {m: float(np.mean(v)) for m, v in all_preds.items()}
    errors = [abs(mean_preds[m] - NMR_DATA[m]['non_ground']) for m in MUTATIONS]
    mae = float(np.mean(errors))

    # Direction accuracy
    dir_correct = 0
    dir_total = 0
    for m in MUTATIONS:
        true_delta = NMR_DATA[m]['non_ground'] - WT_NON_GROUND
        pred_delta = mean_preds[m] - WT_NON_GROUND
        if abs(true_delta) < 0.05:
            continue
        if np.sign(true_delta) == np.sign(pred_delta):
            dir_correct += 1
        dir_total += 1

    return {
        'per_mutant': mean_preds,
        'mae': mae,
        'median': float(np.median(errors)),
        'direction': f"{dir_correct}/{dir_total}",
        'direction_pct': dir_correct / dir_total if dir_total > 0 else 0.0,
        'errors': {m: float(abs(mean_preds[m] - NMR_DATA[m]['non_ground'])) for m in MUTATIONS},
    }


def main():
    print("=" * 90)
    print("SimpleCDST (linear) vs LowRankCDST (MLP) 公平对比")
    print("编码: Extended 10-dim (canonical) | 损失: prob-space MSE | 协议: LOO-CV, 5 seeds")
    print("=" * 90)

    torch.manual_seed(0)
    np.random.seed(0)

    w_wt, w_target, c = prepare_data()
    print(f"\nMutations ({len(MUTATIONS)}): {MUTATIONS}")
    print(f"Encoding dim: {c.shape[1]}")
    print(f"WT non_ground: {WT_NON_GROUND}")

    # 验证 canonical 编码与 npz 数据一致
    print("\n编码矩阵 (Extended 10-dim):")
    for i, m in enumerate(MUTATIONS):
        print(f"  {m:<15} {c[i]}")

    # 测试矩阵
    configs = [
        ('SimpleCDST',  'simple',   800),
        ('SimpleCDST',  'simple',   1000),
        ('LowRankCDST', 'lowrank',  800),
        ('LowRankCDST', 'lowrank',  1000),
        ('LowRankCDST', 'lowrank',  1500),
    ]

    results = {}
    print("\n" + "=" * 90)
    print("运行对比 (每配置 LOO-CV 6×5=30 次训练)...")
    print("=" * 90)

    for label, mtype, epochs in configs:
        key = f"{label}_{epochs}ep"
        print(f"\n>>> {label} ({epochs} epochs) ...")
        r = train_loo(mtype, w_wt, w_target, c, n_epochs=epochs, n_seeds=5)
        results[key] = r
        print(f"    MAE={r['mae']:.4f}  median={r['median']:.4f}  "
              f"dir={r['direction']} ({r['direction_pct']*100:.0f}%)")
        for m in MUTATIONS:
            print(f"      {m:<15} pred={r['per_mutant'][m]:.4f}  "
                  f"true={NMR_DATA[m]['non_ground']:.2f}  err={r['errors'][m]:.4f}")

    # 汇总表
    print("\n\n" + "=" * 90)
    print("汇总对比表")
    print("=" * 90)
    print(f"{'Model':<16} {'Epochs':>7} {'MAE':>8} {'Median':>8} {'Direction':>12}")
    print("-" * 90)
    for label, mtype, epochs in configs:
        key = f"{label}_{epochs}ep"
        r = results[key]
        print(f"  {label:<14} {epochs:>7} {r['mae']:>8.4f} {r['median']:>8.4f} "
              f"{r['direction']:>6} ({r['direction_pct']*100:>3.0f}%)")

    # 参考值
    print("\n参考值 (历史结果文件):")
    print(f"  canonical_results.json  (encoding_ablation.py, LowRankCDST, 800ep): MAE=0.4134")
    print(f"  headline_FINAL.json     (headline_FINAL.py,   LowRankCDST, 1000ep): MAE=0.5680")

    # 稳定性分析
    print("\n\n" + "=" * 90)
    print("稳定性分析 (epoch 敏感性)")
    print("=" * 90)
    lr_800 = results['LowRankCDST_800ep']['mae']
    lr_1000 = results['LowRankCDST_1000ep']['mae']
    lr_1500 = results['LowRankCDST_1500ep']['mae']
    s_800 = results['SimpleCDST_800ep']['mae']
    s_1000 = results['SimpleCDST_1000ep']['mae']

    print(f"\nLowRankCDST MAE 随 epochs 变化:")
    print(f"  800ep:  {lr_800:.4f}")
    print(f"  1000ep: {lr_1000:.4f}")
    print(f"  1500ep: {lr_1500:.4f}")
    print(f"  波动范围: {max(lr_800, lr_1000, lr_1500) - min(lr_800, lr_1000, lr_1500):.4f}")

    print(f"\nSimpleCDST MAE 随 epochs 变化:")
    print(f"  800ep:  {s_800:.4f}")
    print(f"  1000ep: {s_1000:.4f}")
    print(f"  波动范围: {abs(s_800 - s_1000):.4f}")

    stable_lr = (max(lr_800, lr_1000, lr_1500) - min(lr_800, lr_1000, lr_1500)) < 0.05
    stable_s = abs(s_800 - s_1000) < 0.05
    print(f"\n  LowRankCDST 稳定 (波动<0.05): {'是' if stable_lr else '否'}")
    print(f"  SimpleCDST  稳定 (波动<0.05): {'是' if stable_s else '否'}")

    # 保存结果
    out_path = Path(__file__).parent / 'results'
    out_path.mkdir(exist_ok=True)
    output = {
        'experiment': 'SimpleCDST vs LowRankCDST fair comparison',
        'encoding': 'Extended 10-dim (canonical, src.canonical_encoding)',
        'loss': 'prob-space MSE',
        'protocol': 'LOO-CV, 5 seeds/hold_out',
        'data': 'Abl1 6-mutant NMR FINAL (2-state)',
        'wt_non_ground': WT_NON_GROUND,
        'results': results,
        'reference': {
            'canonical_results_json': {'mae': 0.4134, 'model': 'LowRankCDST', 'epochs': 800,
                                        'source': 'encoding_ablation.py'},
            'headline_FINAL_json':    {'mae': 0.5680, 'model': 'LowRankCDST', 'epochs': 1000,
                                        'source': 'headline_FINAL.py'},
        },
    }
    out_file = out_path / 'model_comparison.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n结果已保存到 {out_file}")


if __name__ == '__main__':
    main()
