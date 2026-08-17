# Legacy exploratory script: direction/tie conventions and AF2 reference values differ from the canonical ADR-002 protocol; see k3_benchmark.metrics and canonical_results.py.
"""
P0-1: Headline Comparison Table
CDST vs AF2 frequency vs Majority vs Linear, against NMR ground truth

Key: 2-state collapse (Active vs Non-Ground) for fair comparison with AF2 Fig 6B
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import json

from src.models.cdst import CDST
from src.models.low_rank_cdst import LowRankCDST


def load_data():
    """Load NMR gold data and AF2 baseline."""
    data_dir = Path(__file__).parent.parent.parent / 'data'
    
    # NMR 2-state data
    nmr = np.load(data_dir / 'nmr_populations' / 'cdst_training_nmr_2state.npz', allow_pickle=True)
    
    # AF2 Fig 6B values (x/480 fractions, %non-ground)
    # From Monteiro 2024 Figure 6B
    af2_non_ground = {
        'M290L': 140/480,      # 29.2%
        'L301I': 109/480,      # 22.7%
        'M290L_L301I': 79/480, # 16.5%
        'E255V': 51/480,       # 10.6%
        'T315I': 55/480,       # 11.5%
        'F382L': 63/480,       # 13.1%
        'F382Y': 64/480,       # 13.3%
        'F382V': 74/480,       # 15.4%
    }
    
    # WT non-ground from NMR
    wt_non_ground = 0.06 + 0.06  # I1 + I2 = 12%
    
    return nmr, af2_non_ground, wt_non_ground


def train_cdst(w_train, c_train, wt_train, K=2, d=5, model_type='lowrank'):
    """Train CDST model."""
    if model_type == 'lowrank':
        model = LowRankCDST(K=K, intervention_dim=d, rank=2, hidden_dim=32)
    else:
        model = CDST(K=K, intervention_dim=d, hidden_dim=32, latent_dim=16)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    
    w_t = torch.FloatTensor(w_train)
    c_t = torch.FloatTensor(c_train)
    wt_t = torch.FloatTensor(wt_train)
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        pred = model(w_t, c_t)
        loss = F.mse_loss(pred, wt_t)
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    return model


def compute_headline_table():
    """Compute the headline comparison table."""
    
    print("=" * 70)
    print("P0-1: Headline Comparison Table")
    print("CDST vs AF2 frequency vs Majority vs Linear")
    print("Against NMR ground truth (2-state: Active vs Non-Ground)")
    print("=" * 70)
    
    nmr, af2_non_ground, wt_non_ground = load_data()
    
    mutations = nmr['mutations'].tolist()
    w_wt = nmr['w']        # [n, 2] WT populations
    w_mut = nmr['w_target'] # [n, 2] mutant populations (NMR truth)
    c = nmr['c']           # [n, 5] mutation encoding
    tiers = nmr['tiers'].tolist()
    
    n = len(mutations)
    print(f"\nMutations: {mutations}")
    print(f"Tiers: {tiers}")
    print(f"WT non-ground: {wt_non_ground:.1%}")
    
    # Find intersection (mutations with both NMR and AF2 data)
    intersection = [m for m in mutations if m in af2_non_ground]
    print(f"\nIntersection (NMR + AF2): {intersection}")
    
    # Prepare data for intersection only
    idx_intersect = [i for i, m in enumerate(mutations) if m in intersection]
    
    # NMR ground truth (2-state)
    nmr_truth = w_mut[idx_intersect]  # [n_intersect, 2]
    
    # AF2 predictions (2-state)
    af2_pred = np.array([[1 - af2_non_ground[m], af2_non_ground[m]] for m in intersection])
    
    # Majority baseline (predict WT)
    majority_pred = np.tile(w_wt[0], (len(intersection), 1))
    
    # Linear baseline (mean delta)
    deltas = w_mut - w_wt
    mean_delta = deltas.mean(axis=0)
    linear_pred = w_wt[idx_intersect] + mean_delta
    linear_pred = np.clip(linear_pred, 0, 1)
    linear_pred = linear_pred / linear_pred.sum(axis=1, keepdims=True)
    
    # CDST predictions (LOMO)
    print("\n--- Training CDST (LOMO) ---")
    cdst_preds = np.zeros_like(nmr_truth)
    
    for hold_out in range(n):
        mask = np.ones(n, dtype=bool)
        mask[hold_out] = False
        
        w_train = w_wt[mask]
        c_train = c[mask]
        wt_train = w_mut[mask]
        
        w_test = w_wt[hold_out:hold_out+1]
        c_test = c[hold_out:hold_out+1]
        
        # Train 3 seeds and average
        seed_preds = []
        for seed in range(3):
            torch.manual_seed(seed * 100 + hold_out)
            model = train_cdst(w_train, c_train, wt_train)
            model.eval()
            with torch.no_grad():
                pred = model(torch.FloatTensor(w_test), torch.FloatTensor(c_test))
                seed_preds.append(pred.numpy()[0])
        
        cdst_preds[hold_out] = np.mean(seed_preds, axis=0)
    
    # Extract intersection predictions
    cdst_intersect = cdst_preds[idx_intersect]
    
    # Compute metrics
    print("\n" + "=" * 70)
    print("RESULTS: MAE against NMR ground truth (2-state)")
    print("=" * 70)
    
    methods = {
        'AF2 frequency': af2_pred,
        'Majority (WT)': majority_pred,
        'Linear (mean delta)': linear_pred,
        'CDST (LowRank-r2)': cdst_intersect,
    }
    
    print(f"\n{'Method':<25} {'MAE':>8} {'Dir Acc':>10} {'vs NMR':>10}")
    print("-" * 55)
    
    results = {}
    for name, pred in methods.items():
        mae = np.abs(pred - nmr_truth).mean()
        
        # Direction accuracy (did non-ground increase/decrease correctly?)
        true_delta = nmr_truth[:, 1] - w_wt[idx_intersect, 1]
        pred_delta = pred[:, 1] - w_wt[idx_intersect, 1]
        direction = ((true_delta * pred_delta) > 0).mean()
        
        results[name] = {'mae': mae, 'direction': direction}
        print(f"  {name:<23} {mae:>8.4f} {direction*100:>9.1f}%")
    
    # Per-mutant breakdown
    print("\n" + "=" * 70)
    print("Per-mutant breakdown (Non-Ground population)")
    print("=" * 70)
    print(f"{'Mutant':<15} {'NMR':>8} {'AF2':>8} {'Maj':>8} {'Lin':>8} {'CDST':>8}")
    print("-" * 60)
    
    for i, m in enumerate(intersection):
        nmr_val = nmr_truth[i, 1]
        af2_val = af2_pred[i, 1]
        maj_val = majority_pred[i, 1]
        lin_val = linear_pred[i, 1]
        cdst_val = cdst_intersect[i, 1]
        
        print(f"  {m:<13} {nmr_val:>7.1%} {af2_val:>7.1%} {maj_val:>7.1%} {lin_val:>7.1%} {cdst_val:>7.1%}")
    
    # Key finding
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    
    af2_mae = results['AF2 frequency']['mae']
    cdst_mae = results['CDST (LowRank-r2)']['mae']
    majority_mae = results['Majority (WT)']['mae']
    
    print(f"""
AF2 frequency MAE:  {af2_mae:.4f}
CDST MAE:           {cdst_mae:.4f}
Majority MAE:       {majority_mae:.4f}

CDST vs AF2:     {(af2_mae - cdst_mae)/af2_mae*100:+.1f}% improvement
CDST vs Majority: {(majority_mae - cdst_mae)/majority_mae*100:+.1f}% improvement

AF2 systematically UNDERESTIMATES population shifts:
  - NMR shows large shifts (e.g., L301I: 71% non-ground)
  - AF2 predicts compressed shifts (e.g., L301I: 23% non-ground)
  - This is the core evidence for "AF2 compresses amplitudes"
""")
    
    # Save results
    out_path = Path(__file__).parent / 'results'
    out_path.mkdir(exist_ok=True)
    
    output = {
        'methods': {k: {'mae': float(v['mae']), 'direction': float(v['direction'])} 
                   for k, v in results.items()},
        'per_mutant': {m: {'nmr': float(nmr_truth[i, 1]), 
                          'af2': float(af2_pred[i, 1]),
                          'cdst': float(cdst_intersect[i, 1])}
                      for i, m in enumerate(intersection)},
        'intersection': intersection,
    }
    
    with open(out_path / 'headline_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to {out_path / 'headline_comparison.json'}")
    
    return results


if __name__ == '__main__':
    compute_headline_table()
