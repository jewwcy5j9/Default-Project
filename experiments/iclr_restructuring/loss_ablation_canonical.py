"""
C: Loss ablation on the CANONICAL Abl1 benchmark (6 losses).

Motivation: the paper's §3.2 presents the Fisher-Rao/Hellinger loss as a
geometric alternative, but the experiments train with prob-space MSE. To make
the presentation honest and current (the old p1_loss_comparison.json ran a
different, 3-state protocol), this script evaluates MSE / KL / Fisher-Rao
(Hellinger) / Hellinger / JSD / natural-parameter-L2 under the exact
canonical protocol (LowRankCDST, 800 epochs, 5 seeds, LOO, seed=seed*100+hold_out).

Output: results/loss_ablation_canonical.json + _report.md
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn.functional as F

from encoding_ablation_control import (
    ABL1_DATA, ABL1_WT_NON_GROUND, ABL1_SEQ_LEN,
)
from alternative_encodings import DDG_DATA, DDG_NORM, encode_ddg_main
from src.models.low_rank_cdst import LowRankCDST
from src.models.losses import (
    FisherRaoLoss, HellingerLoss, SymmetricKLLoss, NaturalParameterLoss,
)


def train_one_seed(w_train, c_train, target_train, d, seed, loss_name,
                   n_epochs=800):
    """Canonical training loop with pluggable loss (default prob-space MSE)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LowRankCDST(K=2, intervention_dim=d, rank=2, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)

    w_t = torch.FloatTensor(w_train)
    c_t = torch.FloatTensor(c_train)
    target_t = torch.FloatTensor(target_train)

    losses = {
        'mse': lambda p, t: F.mse_loss(p, t),
        'kl': lambda p, t: F.kl_div(torch.log(p.clamp(min=1e-8)), t,
                                    reduction='batchmean'),
        'fisher_rao': FisherRaoLoss(mode='hellinger'),
        'hellinger': HellingerLoss(),
        'jsd': SymmetricKLLoss(),
        'natural_l2': NaturalParameterLoss(),
    }
    loss_fn = losses[loss_name]

    best_loss = float('inf')
    best_state = None
    for _ in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(w_t, c_t)
        loss = loss_fn(pred, target_t)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model


def run_loo(loss_name, n_seeds=5, n_epochs=800):
    """LOO-CV under canonical protocol for one loss."""
    names = list(ABL1_DATA.keys())
    n = len(names)
    wt_dist = np.array([1 - ABL1_WT_NON_GROUND, ABL1_WT_NON_GROUND])
    w_wt = np.tile(wt_dist, (n, 1))
    targets = np.array([[1 - ABL1_DATA[m]['non_ground'], ABL1_DATA[m]['non_ground']]
                        for m in names])
    encodings = np.array([encode_ddg_main(m, ABL1_DATA[m], DDG_DATA, DDG_NORM,
                                          ABL1_SEQ_LEN, system='abl1')
                          for m in names])

    mean_preds = {}
    for hold_out in range(n):
        mask = np.ones(n, dtype=bool)
        mask[hold_out] = False
        seed_preds = []
        for seed in range(n_seeds):
            model = train_one_seed(w_wt[mask], encodings[mask], targets[mask],
                                   d=5, seed=seed * 100 + hold_out,
                                   loss_name=loss_name, n_epochs=n_epochs)
            with torch.no_grad():
                p = model(torch.FloatTensor(w_wt[hold_out:hold_out + 1]),
                          torch.FloatTensor(encodings[hold_out:hold_out + 1]))[0, 1]
            seed_preds.append(float(p))
        mean_preds[names[hold_out]] = float(np.mean(seed_preds))

    errors = {m: abs(mean_preds[m] - ABL1_DATA[m]['non_ground']) for m in names}
    mae = float(np.mean(list(errors.values())))
    dir_correct, dir_total = 0, 0
    dir_detail = {}
    for m in names:
        d_true = ABL1_DATA[m]['non_ground'] - ABL1_WT_NON_GROUND
        d_pred = mean_preds[m] - ABL1_WT_NON_GROUND
        if abs(d_true) < 0.05:
            dir_detail[m] = 'TIE'
            continue
        dir_total += 1
        if np.sign(d_true) == np.sign(d_pred):
            dir_correct += 1
            dir_detail[m] = 'OK'
        else:
            dir_detail[m] = 'WRONG'
    return {'mae': mae, 'direction': f'{dir_correct}/{dir_total}',
            'per_mutant': mean_preds, 'errors': errors,
            'direction_detail': dir_detail}


def main():
    t0 = time.time()
    print("=" * 80)
    print("Loss ablation on canonical Abl1 benchmark (variant C encoding)")
    print("=" * 80)
    losses = ['mse', 'kl', 'fisher_rao', 'hellinger', 'jsd', 'natural_l2']
    results = {}
    for loss in losses:
        print(f"[{loss}] running LOO (5 seeds x 6 folds, 800 epochs) ...",
              flush=True)
        results[loss] = run_loo(loss)
        print(f"  MAE={results[loss]['mae']:.4f}  dir={results[loss]['direction']}",
              flush=True)

    base = results['mse']['mae']
    for loss in losses:
        r = results[loss]
        r['delta_vs_mse_pct'] = (r['mae'] - base) / base * 100

    out_dir = Path(__file__).parent / 'results'
    out_dir.mkdir(exist_ok=True)
    payload = {
        'experiment': 'loss_ablation_canonical',
        'protocol': 'LowRankCDST(K=2, rank=2, hidden_dim=32), variant C encoding, '
                    '800 epochs, 5 seeds, LOO-CV (seed=seed*100+hold_out)',
        'results': results,
        'mse_reference_mae': base,
        'max_abs_delta_pct': max(abs(r['delta_vs_mse_pct']) for r in results.values()),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'note': 'Replaces p1_loss_comparison.json (old 3-state protocol, not used)',
    }
    out_json = out_dir / 'loss_ablation_canonical.json'
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding='utf-8')
    print(f"[OK] {out_json}")

    L = []
    L.append("# Loss 消融（canonical Abl1 协议）")
    L.append("")
    L.append(f"> 协议: {payload['protocol']} | 耗时 {time.time()-t0:.0f}s")
    L.append("")
    L.append("| Loss | MAE | Δ vs MSE | 方向 |")
    L.append("|---|---:|---:|:---:|")
    for loss in losses:
        r = results[loss]
        L.append(f"| {loss} | {r['mae']:.4f} | {r['delta_vs_mse_pct']:+.2f}% | "
                 f"{r['direction']} |")
    L.append("")
    L.append(f"最大偏差: {payload['max_abs_delta_pct']:.2f}% (MSE 为基准)")
    L.append("")
    L.append("论文写法建议: '几何损失（FR/Hellinger、JSD、自然参数 L2）与 MSE "
             "差异 < X%（附录 B），正文采用 MSE'")
    L.append("")
    L.append("---")
    L.append("*脚本: experiments/iclr_restructuring/loss_ablation_canonical.py*")
    out_md = out_dir / 'loss_ablation_canonical_report.md'
    out_md.write_text("\n".join(L), encoding='utf-8')
    print(f"[OK] {out_md}")


if __name__ == '__main__':
    main()
