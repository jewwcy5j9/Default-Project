"""
CDST Retraining with NMR Gold-Standard Labels.

Key changes from previous version:
- Labels from Xie 2020 NMR CEST (not AF2 frequency)
- K=3 states (Active, I1, I2)
- AF2 frequency as baseline comparison (not training label)
- Must beat Majority/Linear baselines

Evaluation:
- Leave-one-mutant-out (LOMO) cross-validation
- Metrics: MAE, direction accuracy, vs baselines
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from src.models.cdst import CDST
from src.models.low_rank_cdst import LowRankCDST, AdaptiveRankCDST
from src.models.compositional_cdst import CompositionalCDST
from src.models.losses import FisherRaoLoss, HellingerLoss


def load_nmr_data():
    """Load NMR training data."""
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'nmr_populations'
    
    # 3-state data
    d3 = np.load(data_dir / 'cdst_training_nmr.npz', allow_pickle=True)
    
    return {
        'w_wt': d3['w_wt'],        # [n, 3]
        'w_mut': d3['w_mut'],      # [n, 3]
        'c': d3['c'],              # [n, 5]
        'mutations': d3['mutations'].tolist(),
        'tiers': d3['tiers'].tolist(),
        'K': 3,
    }


def load_af2_baseline():
    """Load AF2 frequency baseline for comparison."""
    # AF2 Fig 6B values (x/480 fractions)
    af2_pops = {
        'M290L': {'non_ground': 140/480},      # 29.2%
        'L301I': {'non_ground': 109/480},      # 22.7%
        'M290L_L301I': {'non_ground': 79/480}, # 16.5% - wait, this seems wrong
        'E255V': {'non_ground': 51/480},       # 10.6%
        'T315I': {'non_ground': 55/480},       # 11.5%
        'F382L': {'non_ground': 63/480},       # 13.1%
        'F382Y': {'non_ground': 64/480},       # 13.3%
        'F382V': {'non_ground': 74/480},       # 15.4%
    }
    return af2_pops


def train_model(model, w_train, c_train, wt_train, loss_fn=None, 
                n_epochs=1000, lr=5e-3):
    """Train model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    w_t = torch.FloatTensor(w_train)
    c_t = torch.FloatTensor(c_train)
    wt_t = torch.FloatTensor(wt_train)
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        pred = model(w_t, c_t)
        
        if loss_fn is not None:
            loss = loss_fn(pred, wt_t)
        else:
            loss = F.mse_loss(pred, wt_t)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model


def evaluate_model(model, w_test, c_test, wt_test):
    """Evaluate model."""
    model.eval()
    with torch.no_grad():
        w_t = torch.FloatTensor(w_test)
        c_t = torch.FloatTensor(c_test)
        wt_t = torch.FloatTensor(wt_test)
        
        pred = model(w_t, c_t)
        
        mae = (pred - wt_t).abs().mean().item()
        
        # Direction accuracy (for each state)
        true_delta = wt_t - w_t
        pred_delta = pred - w_t
        direction = ((true_delta * pred_delta) > 0).float().mean().item()
    
    return {'mae': mae, 'direction': direction, 'pred': pred.numpy()}


def run_lomo(data, model_fn, loss_fn=None, n_seeds=5):
    """Leave-one-mutant-out CV."""
    n = len(data['w_wt'])
    results = []
    
    for hold_out in range(n):
        mask = np.ones(n, dtype=bool)
        mask[hold_out] = False
        
        w_train = data['w_wt'][mask]
        c_train = data['c'][mask]
        wt_train = data['w_mut'][mask]
        
        w_test = data['w_wt'][hold_out:hold_out+1]
        c_test = data['c'][hold_out:hold_out+1]
        wt_test = data['w_mut'][hold_out:hold_out+1]
        
        seed_results = []
        for seed in range(n_seeds):
            torch.manual_seed(seed * 100 + hold_out)
            np.random.seed(seed * 100 + hold_out)
            
            model = model_fn()
            model = train_model(model, w_train, c_train, wt_train, loss_fn=loss_fn)
            result = evaluate_model(model, w_test, c_test, wt_test)
            seed_results.append(result)
        
        avg_mae = np.mean([r['mae'] for r in seed_results])
        avg_dir = np.mean([r['direction'] for r in seed_results])
        
        results.append({
            'mutation': data['mutations'][hold_out],
            'tier': data['tiers'][hold_out],
            'mae': avg_mae,
            'direction': avg_dir,
            'true_pop': wt_test[0].tolist(),
        })
    
    overall_mae = np.mean([r['mae'] for r in results])
    overall_dir = np.mean([r['direction'] for r in results])
    
    return {'per_mutant': results, 'overall_mae': overall_mae, 'overall_direction': overall_dir}


def compute_baselines(data):
    """Compute Majority and Linear baselines."""
    w_wt = data['w_wt']
    w_mut = data['w_mut']
    n = len(w_wt)
    
    # Majority: predict WT
    majority_mae = np.abs(w_mut - w_wt).mean()
    
    # Linear: predict mean delta
    deltas = w_mut - w_wt
    mean_delta = deltas.mean(axis=0)
    linear_pred = w_wt + mean_delta
    linear_pred = np.clip(linear_pred, 0, 1)
    linear_pred = linear_pred / linear_pred.sum(axis=1, keepdims=True)
    linear_mae = np.abs(linear_pred - w_mut).mean()
    
    return {
        # placeholder removed 2026-08-17; direction not computed for these baselines
        'Majority': {'mae': float(majority_mae), 'direction': None},
        'Linear': {'mae': float(linear_mae), 'direction': None},
    }


def main():
    print("=" * 70)
    print("CDST Retraining with NMR Gold-Standard Labels")
    print("=" * 70)
    
    data = load_nmr_data()
    K = data['K']
    d = data['c'].shape[1]
    n = len(data['w_wt'])
    
    print(f"\nData: {n} pairs, K={K} states, d={d} encoding dim")
    print(f"Mutations: {data['mutations']}")
    print(f"Tiers: {data['tiers']}")
    print(f"WT: {data['w_wt'][0]}")
    
    # Baselines
    print("\n--- Baselines ---")
    baselines = compute_baselines(data)
    for name, result in baselines.items():
        print(f"  {name}: MAE={result['mae']:.4f}")
    
    # Models
    models = {
        'CDST-vanilla': (lambda: CDST(K=K, intervention_dim=d, hidden_dim=32, latent_dim=16), None),
        'CDST-FR': (lambda: CDST(K=K, intervention_dim=d, hidden_dim=32, latent_dim=16), FisherRaoLoss()),
        'LowRank-r1': (lambda: LowRankCDST(K=K, intervention_dim=d, rank=1, hidden_dim=32), None),
        'LowRank-r2': (lambda: LowRankCDST(K=K, intervention_dim=d, rank=2, hidden_dim=32), None),
        'Compositional': (lambda: CompositionalCDST(K=K, d=d, rank=2), None),
    }
    
    print("\n--- LOMO Cross-Validation ---")
    print(f"{'Model':<20} {'MAE':>8} {'Dir%':>8} {'vs Majority':>12}")
    print("-" * 52)
    
    results = baselines.copy()
    majority_mae = baselines['Majority']['mae']
    
    for name, (model_fn, loss_fn) in models.items():
        try:
            cv_result = run_lomo(data, model_fn, loss_fn=loss_fn, n_seeds=3)
            results[name] = cv_result
            
            improvement = (majority_mae - cv_result['overall_mae']) / majority_mae * 100
            sign = "+" if improvement > 0 else ""
            
            print(f"  {name:<18} {cv_result['overall_mae']:>8.4f} "
                  f"{cv_result['overall_direction']*100:>7.1f}% "
                  f"{sign}{improvement:>10.1f}%")
        except Exception as e:
            print(f"  {name:<18} ERROR: {e}")
    
    # Full data fit
    print("\n--- Full Data Fit ---")
    for name, (model_fn, loss_fn) in models.items():
        try:
            torch.manual_seed(42)
            model = model_fn()
            model = train_model(model, data['w_wt'], data['c'], data['w_mut'], loss_fn=loss_fn)
            result = evaluate_model(model, data['w_wt'], data['c'], data['w_mut'])
            print(f"  {name:<18} Train MAE={result['mae']:.4f}")
        except Exception as e:
            print(f"  {name:<18} ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Find best model
    model_results = {k: v for k, v in results.items() if 'overall_mae' in v and 'per_mutant' in v}
    if model_results:
        best_name = min(model_results, key=lambda k: model_results[k]['overall_mae'])
        best = model_results[best_name]
        
        print(f"\nBest model: {best_name}")
        print(f"  MAE = {best['overall_mae']:.4f}")
        print(f"  Direction = {best['overall_direction']*100:.1f}%")
        print(f"  vs Majority: {(majority_mae - best['overall_mae'])/majority_mae*100:+.1f}%")
        
        # Check if beats Majority
        if best['overall_mae'] < majority_mae:
            print("\n  [PASS] Beats Majority baseline!")
        else:
            print("\n  [FAIL] Does NOT beat Majority baseline")
        
        print(f"\nPer-mutant breakdown:")
        for r in best['per_mutant']:
            tier_mark = "G" if r['tier'].startswith('gold') else "S"
            print(f"    [{tier_mark}] {r['mutation']:<15} MAE={r['mae']:.4f} "
                  f"Dir={'Y' if r['direction'] > 0.5 else 'N'} "
                  f"True={r['true_pop']}")
    
    # Save results
    out_path = Path(__file__).parent / 'results'
    out_path.mkdir(exist_ok=True)
    
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(x) for x in obj]
        return obj
    
    with open(out_path / 'nmr_training_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    
    print(f"\nResults saved to {out_path / 'nmr_training_results.json'}")


if __name__ == '__main__':
    main()
