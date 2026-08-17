# Legacy exploratory script: direction/tie conventions and AF2 reference values differ from the canonical ADR-002 protocol; see k3_benchmark.metrics and canonical_results.py.
"""
Headline Comparison with FINAL NMR data
CDST vs AF2 frequency vs Majority vs Linear
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


def load_final_data():
    """Load FINAL NMR data."""
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'nmr_populations'
    
    # Load FINAL 2-state data
    nmr = np.load(data_dir / 'cdst_training_nmr_FINAL_2state.npz', allow_pickle=True)
    
    # AF2 Fig 6B values (x/480 fractions, %non-ground)
    af2_non_ground = {
        'M290L': 140/480,
        'L301I': 109/480,
        'M290L_L301I': 79/480,
        'F382L': 63/480,
        'F382Y': 64/480,
        'F382V': 74/480,
    }
    
    return nmr, af2_non_ground


def train_cdst(w_train, c_train, wt_train, K=2, d=10):
    """Train LowRank CDST."""
    model = LowRankCDST(K=K, intervention_dim=d, rank=2, hidden_dim=32)
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


def main():
    print("=" * 70)
    print("Headline Comparison - FINAL NMR Data")
    print("=" * 70)
    
    nmr, af2_non_ground = load_final_data()
    
    mutations = nmr['mutations'].tolist()
    w_wt = nmr['w']
    w_mut = nmr['w_target']
    c = nmr['c']
    
    n = len(mutations)
    print(f"\nMutations ({n}): {mutations}")
    
    # Find intersection with AF2
    intersection = [m for m in mutations if m in af2_non_ground]
    idx_intersect = [i for i, m in enumerate(mutations) if m in intersection]
    print(f"Intersection with AF2 ({len(intersection)}): {intersection}")
    
    # Prepare data
    nmr_truth = w_mut[idx_intersect]
    af2_pred = np.array([[1 - af2_non_ground[m], af2_non_ground[m]] for m in intersection])
    majority_pred = np.tile(w_wt[0], (len(intersection), 1))
    
    # Linear baseline
    deltas = w_mut - w_wt
    mean_delta = deltas.mean(axis=0)
    linear_pred = w_wt[idx_intersect] + mean_delta
    linear_pred = np.clip(linear_pred, 0, 1)
    linear_pred = linear_pred / linear_pred.sum(axis=1, keepdims=True)
    
    # CDST LOMO
    print("\n--- Training CDST (LOMO, 5 seeds) ---")
    cdst_preds = np.zeros((n, 2))
    
    for hold_out in range(n):
        mask = np.ones(n, dtype=bool)
        mask[hold_out] = False
        
        seed_preds = []
        for seed in range(5):
            torch.manual_seed(seed * 100 + hold_out)
            model = train_cdst(w_wt[mask], c[mask], w_mut[mask])
            model.eval()
            with torch.no_grad():
                pred = model(
                    torch.FloatTensor(w_wt[hold_out:hold_out+1]),
                    torch.FloatTensor(c[hold_out:hold_out+1])
                )
                seed_preds.append(pred.numpy()[0])
        
        cdst_preds[hold_out] = np.mean(seed_preds, axis=0)
    
    cdst_intersect = cdst_preds[idx_intersect]
    
    # Compute metrics
    print("\n" + "=" * 70)
    print("RESULTS: MAE against NMR ground truth (2-state)")
    print("=" * 70)
    
    methods = {
        'AF2 frequency': af2_pred,
        'Majority (WT)': majority_pred,
        'Linear': linear_pred,
        'CDST (LowRank-r2)': cdst_intersect,
    }
    
    print(f"\n{'Method':<25} {'MAE':>8} {'Dir Acc':>10}")
    print("-" * 45)
    
    results = {}
    for name, pred in methods.items():
        mae = np.abs(pred - nmr_truth).mean()
        
        true_delta = nmr_truth[:, 1] - w_wt[idx_intersect, 1]
        pred_delta = pred[:, 1] - w_wt[idx_intersect, 1]
        direction = ((true_delta * pred_delta) > 0).mean()
        
        results[name] = {'mae': float(mae), 'direction': float(direction)}
        print(f"  {name:<23} {mae:>8.4f} {direction*100:>9.1f}%")
    
    # Per-mutant
    print("\n" + "=" * 70)
    print("Per-mutant (Non-Ground %)")
    print("=" * 70)
    print(f"{'Mutant':<15} {'NMR':>8} {'AF2':>8} {'CDST':>8} {'AF2 err':>8} {'CDST err':>8}")
    print("-" * 60)
    
    for i, m in enumerate(intersection):
        nmr_val = nmr_truth[i, 1]
        af2_val = af2_pred[i, 1]
        cdst_val = cdst_intersect[i, 1]
        af2_err = af2_val - nmr_val
        cdst_err = cdst_val - nmr_val
        
        print(f"  {m:<13} {nmr_val:>7.1%} {af2_val:>7.1%} {cdst_val:>7.1%} {af2_err:>+7.1%} {cdst_err:>+7.1%}")
    
    # Key stats
    print("\n" + "=" * 70)
    print("KEY STATISTICS")
    print("=" * 70)
    
    af2_mae = results['AF2 frequency']['mae']
    cdst_mae = results['CDST (LowRank-r2)']['mae']
    maj_mae = results['Majority (WT)']['mae']
    
    print(f"""
AF2 MAE:      {af2_mae:.4f}
CDST MAE:     {cdst_mae:.4f}
Majority MAE: {maj_mae:.4f}

CDST vs AF2:      {(af2_mae - cdst_mae)/af2_mae*100:+.1f}%
CDST vs Majority: {(maj_mae - cdst_mae)/maj_mae*100:+.1f}%

Direction accuracy:
  AF2:  {results['AF2 frequency']['direction']*100:.1f}%
  CDST: {results['CDST (LowRank-r2)']['direction']*100:.1f}%
""")
    
    # Save
    out_path = Path(__file__).parent / 'results'
    out_path.mkdir(exist_ok=True)
    
    with open(out_path / 'headline_FINAL.json', 'w') as f:
        json.dump({
            'methods': results,
            'per_mutant': {m: {'nmr': float(nmr_truth[i, 1]), 
                              'af2': float(af2_pred[i, 1]),
                              'cdst': float(cdst_intersect[i, 1])}
                          for i, m in enumerate(intersection)},
            'n_mutations': len(intersection),
        }, f, indent=2)
    
    print(f"Saved to {out_path / 'headline_FINAL.json'}")


if __name__ == '__main__':
    main()
