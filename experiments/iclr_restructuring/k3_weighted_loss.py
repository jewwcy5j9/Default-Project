"""Inverse-frequency weighted loss at Src K=3: can rare-state (E2) weighting
fix the E2-direction failure? Tests pos-markers / Extended / ESM-2 encodings.
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

from k3_data import SRC_K3, SRC_K3_WT_POP, SRC_SEQ_LEN
from k3_benchmark import metrics
from src.models.low_rank_cdst import LowRankCDST
from k3_src_deepdive import enc_src_pos_anchored, enc_src_ext_anchored

SRC_MUT = {m: SRC_K3[m] for m in SRC_K3 if m != 'SrcKD-WT'}


def loo_weighted(mutations, wt_pop, encoder_fn, d, n_seeds=5, n_epochs=800,
                 weight_power=0.5):
    """LOO with per-state inverse-frequency loss weighting."""
    names = list(mutations.keys())
    n = len(names)
    w_wt = np.tile(np.array(wt_pop, dtype=float), (n, 1))
    targets = np.array([mutations[m]['pop'] for m in names])
    encodings = np.array([encoder_fn(m, mutations[m]) for m in names])

    # per-state mean fraction -> weights (computed on FULL data; LOO-stable)
    mean_frac = targets.mean(axis=0) + 1e-3
    weights = mean_frac ** (-weight_power)
    weights = weights / weights.mean()

    mean_preds = {}
    for hold_out in range(n):
        mask = np.ones(n, dtype=bool)
        mask[hold_out] = False
        seed_preds = []
        for seed in range(n_seeds):
            torch.manual_seed(seed * 100 + hold_out)
            np.random.seed(seed * 100 + hold_out)
            # FIXED 2026-08-17: start the weighted-optimization loop from a
            # FRESH model (same class/seed handling as k3_benchmark.train_one_seed)
            # instead of continuing from MSE-trained weights. Previously
            # train_one_seed(...) pre-trained with MSE and the weighted loop
            # below fine-tuned those weights, contradicting the "from scratch"
            # claim in the old comment. Stored k3_weighted_loss_results.json
            # outputs predate this fix.
            model = LowRankCDST(K=3, intervention_dim=d, rank=2, hidden_dim=32)
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-3,
                                         weight_decay=1e-4)
            # weighted loss training from the fresh initialization
            w_t = torch.FloatTensor(w_wt[mask])
            c_t = torch.FloatTensor(encodings[mask])
            t_t = torch.FloatTensor(targets[mask])
            best_loss, best_state = float('inf'), None
            for _ in range(n_epochs):
                optimizer.zero_grad()
                pred = model(w_t, c_t)
                loss = ((pred - t_t) ** 2 * torch.FloatTensor(weights)).mean()
                loss.backward()
                optimizer.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                p = model(torch.FloatTensor([wt_pop]),
                          torch.FloatTensor([encoder_fn(names[hold_out],
                                                        mutations[names[hold_out]])]))
            seed_preds.append(p.numpy()[0])
        mean_preds[names[hold_out]] = np.mean(seed_preds, axis=0)
    return mean_preds, weights


def main():
    t0 = time.time()
    print("=" * 90)
    print("Rare-state weighted loss at Src K=3")
    print("=" * 90)
    results = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}

    for enc_name, fn, d in [('pos_markers', enc_src_pos_anchored, 4),
                            ('Extended', enc_src_ext_anchored, 10)]:
        for power in [0.5, 1.0]:
            preds, w = loo_weighted(SRC_MUT, SRC_K3_WT_POP, fn, d,
                                    weight_power=power)
            met = metrics(preds, {m: SRC_MUT[m]['pop'] for m in SRC_MUT},
                          SRC_K3_WT_POP)
            key = f'{enc_name}_pow{power}'
            results[key] = {'mae': met['mae'], 'direction': met['direction'],
                            'errors': met['mae_per_mutant'],
                            'weights': w.tolist()}
            print(f"  {key:<20} MAE={met['mae']:.4f} dir={met['direction']} "
                  f"(weights {np.round(w,2)})")
            for m in ['SrcKD-F405A', 'SrcKD-V380A', 'SrcKD-V331A']:
                print(f"      {m:<14} true={np.round(np.array(SRC_MUT[m]['pop']),2)} "
                      f"pred={np.round(preds[m],2)} "
                      f"err={met['mae_per_mutant'][m]:.4f}")

    # (ESM-2 weighted branch removed 2026-08-17: encodings were never
    #  persisted, so the old branch loaded a JSON it never used.)

    out_dir = Path(__file__).parent / 'results'
    out_json = out_dir / 'k3_weighted_loss_results.json'
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False,
                                   default=float), encoding='utf-8')
    print(f"\n[OK] {out_json}  (total {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
