"""
P5: Protein Conformational Shift Experiments.

Uses existing AF2 subsampling data (Abl1 kinase, 7 mutant pairs, K=2).
Evaluates all CDST variants with leave-one-mutant-out cross-validation.

Data: Monteiro da Silva et al. (2024) AF2 subsampling predictions
- 7 Abl1 mutants with known population shifts
- K=2 states (Ground/Active vs Non-Ground/Inactive)
- d=5 intervention encoding (physicochemical properties)

Evaluation:
- Leave-one-mutant-out (LOMO) cross-validation
- Metrics: MAE, direction accuracy, rank correlation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import Dict, List

from src.models.cdst import CDST
from src.models.low_rank_cdst import LowRankCDST, AdaptiveRankCDST
from src.models.compositional_cdst import CompositionalCDST
from src.models.losses import FisherRaoLoss, HellingerLoss, NaturalParameterLoss


def load_af2_data():
    """Load AF2 2-state training data."""
    data_path = Path(__file__).parent.parent.parent / 'data' / 'af2_populations'
    d = np.load(data_path / 'cdst_training_2state.npz', allow_pickle=True)
    return {
        'w': d['w'],           # [7, 2] WT populations
        'c': d['c'],           # [7, 5] mutation encodings
        'w_target': d['w_target'],  # [7, 2] mutant populations
        'mutations': d['mutations'].tolist()
    }


def train_model(model, w_train, c_train, wt_train, 
                loss_fn=None, n_epochs=1000, lr=5e-3, weight_decay=1e-4):
    """Train a model on given data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    w_t = torch.FloatTensor(w_train)
    c_t = torch.FloatTensor(c_train)
    wt_t = torch.FloatTensor(wt_train)
    
    best_loss = float('inf')
    best_state = None
    patience = 200
    no_improve = 0
    
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
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > patience:
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model


def evaluate_model(model, w_test, c_test, wt_test):
    """Evaluate model on test data."""
    model.eval()
    with torch.no_grad():
        w_t = torch.FloatTensor(w_test)
        c_t = torch.FloatTensor(c_test)
        wt_t = torch.FloatTensor(wt_test)
        
        pred = model(w_t, c_t)
        
        # MAE
        mae = (pred - wt_t).abs().mean().item()
        
        # Direction accuracy (did population increase/decrease correctly?)
        true_delta = wt_t[:, 0] - w_t[:, 0]
        pred_delta = pred[:, 0] - w_t[:, 0]
        direction = ((true_delta * pred_delta) > 0).float().mean().item()
        
        # Per-sample errors
        per_sample_mae = (pred - wt_t).abs().mean(dim=1).numpy()
    
    return {
        'mae': mae,
        'direction_accuracy': direction,
        'per_sample_mae': per_sample_mae.tolist(),
        'predictions': pred.numpy().tolist()
    }


def run_lomo_cv(data, model_fn, loss_fn=None, n_seeds=5, **train_kwargs):
    """Leave-one-mutant-out cross-validation.
    
    Args:
        data: dict with w, c, w_target, mutations
        model_fn: callable() -> model
        loss_fn: loss function or None
        n_seeds: number of random seeds per fold
    """
    n = len(data['w'])
    all_results = []
    
    for hold_out in range(n):
        # Split
        mask = np.ones(n, dtype=bool)
        mask[hold_out] = False
        
        w_train = data['w'][mask]
        c_train = data['c'][mask]
        wt_train = data['w_target'][mask]
        
        w_test = data['w'][hold_out:hold_out+1]
        c_test = data['c'][hold_out:hold_out+1]
        wt_test = data['w_target'][hold_out:hold_out+1]
        
        seed_results = []
        for seed in range(n_seeds):
            torch.manual_seed(seed * 100 + hold_out)
            np.random.seed(seed * 100 + hold_out)
            
            model = model_fn()
            model = train_model(model, w_train, c_train, wt_train,
                              loss_fn=loss_fn, **train_kwargs)
            result = evaluate_model(model, w_test, c_test, wt_test)
            seed_results.append(result)
        
        # Average over seeds
        avg_mae = np.mean([r['mae'] for r in seed_results])
        avg_dir = np.mean([r['direction_accuracy'] for r in seed_results])
        
        all_results.append({
            'mutation': data['mutations'][hold_out],
            'mae': avg_mae,
            'direction': avg_dir,
            'true_population': wt_test[0].tolist(),
        })
    
    # Aggregate
    overall_mae = np.mean([r['mae'] for r in all_results])
    overall_dir = np.mean([r['direction'] for r in all_results])
    
    return {
        'per_mutant': all_results,
        'overall_mae': overall_mae,
        'overall_direction': overall_dir,
    }


def run_protein_experiments():
    """Run full protein experiment suite."""
    print("=" * 70)
    print("P5: Protein Conformational Shift Experiments (Abl1 Kinase)")
    print("=" * 70)
    
    data = load_af2_data()
    K = 2
    d = 5
    n = len(data['w'])
    
    print(f"\nData: {n} mutant pairs, K={K} states, d={d} encoding dim")
    print(f"Mutations: {data['mutations']}")
    print(f"WT populations: Ground={data['w'][0,0]:.3f}, Non-Ground={data['w'][0,1]:.3f}")
    
    results = {}
    
    # Define model factories
    models = {
        'CDST-vanilla': (lambda: CDST(K=K, intervention_dim=d, hidden_dim=32, latent_dim=16), None),
        'CDST-FR': (lambda: CDST(K=K, intervention_dim=d, hidden_dim=32, latent_dim=16), FisherRaoLoss()),
        'CDST-Hellinger': (lambda: CDST(K=K, intervention_dim=d, hidden_dim=32, latent_dim=16), HellingerLoss()),
        'CDST-Natural': (lambda: CDST(K=K, intervention_dim=d, hidden_dim=32, latent_dim=16), NaturalParameterLoss()),
        'LowRank-r1': (lambda: LowRankCDST(K=K, intervention_dim=d, rank=1, hidden_dim=32), None),
        'LowRank-r2': (lambda: LowRankCDST(K=K, intervention_dim=d, rank=2, hidden_dim=32), None),
        'AdaptiveRank': (lambda: AdaptiveRankCDST(K=K, intervention_dim=d, hidden_dim=32), None),
        'Compositional': (lambda: CompositionalCDST(K=K, d=d, rank=2), None),
    }
    
    # Baselines (non-parametric)
    print("\n--- Baselines ---")
    
    # Majority predictor (always predict WT)
    majority_mae = np.abs(data['w_target'] - data['w']).mean()
    print(f"  Majority (predict WT): MAE={majority_mae:.4f}, Dir=50.0%")
    # placeholder removed 2026-08-17; direction not computed for these baselines
    results['Majority'] = {'overall_mae': float(majority_mae), 'overall_direction': None}
    
    # Linear response (log-space linear regression)
    log_w = np.log(data['w'].clip(1e-10))
    log_wt = np.log(data['w_target'].clip(1e-10))
    delta_true = log_wt - log_w  # [7, 2]
    # Simple: predict mean delta
    mean_delta = delta_true.mean(axis=0)
    linear_pred = np.exp(log_w + mean_delta)
    linear_pred = linear_pred / linear_pred.sum(axis=1, keepdims=True)
    linear_mae = np.abs(linear_pred - data['w_target']).mean()
    print(f"  Linear (mean shift): MAE={linear_mae:.4f}")
    # placeholder removed 2026-08-17; direction not computed for these baselines
    results['Linear-mean'] = {'overall_mae': float(linear_mae), 'overall_direction': None}
    
    # Run each model with LOMO-CV
    print("\n--- Model Comparison (Leave-One-Mutant-Out) ---")
    print(f"{'Model':<20} {'MAE':>8} {'Dir%':>8}")
    print("-" * 40)
    
    for name, (model_fn, loss_fn) in models.items():
        try:
            cv_result = run_lomo_cv(data, model_fn, loss_fn=loss_fn, 
                                   n_seeds=3, n_epochs=800, lr=3e-3)
            results[name] = cv_result
            print(f"  {name:<18} {cv_result['overall_mae']:>8.4f} "
                  f"{cv_result['overall_direction']*100:>7.1f}%")
        except Exception as e:
            print(f"  {name:<18} ERROR: {e}")
            results[name] = {'error': str(e)}
    
    # Full-data training (no CV, just fit)
    print("\n--- Full Data Fit (all 7 pairs) ---")
    print(f"{'Model':<20} {'Train MAE':>10}")
    print("-" * 35)
    
    for name, (model_fn, loss_fn) in models.items():
        try:
            torch.manual_seed(42)
            model = model_fn()
            model = train_model(model, data['w'], data['c'], data['w_target'],
                              loss_fn=loss_fn, n_epochs=1000, lr=3e-3)
            result = evaluate_model(model, data['w'], data['c'], data['w_target'])
            print(f"  {name:<18} {result['mae']:>10.4f}")
            results[f'{name}_fullfit'] = result
        except Exception as e:
            print(f"  {name:<18} ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Find best LOMO model
    lomo_results = {k: v for k, v in results.items() 
                    if isinstance(v, dict) and 'overall_mae' in v and 'per_mutant' in v}
    if lomo_results:
        best_name = min(lomo_results, key=lambda k: lomo_results[k]['overall_mae'])
        best = lomo_results[best_name]
        print(f"\nBest LOMO model: {best_name}")
        print(f"  MAE = {best['overall_mae']:.4f}")
        print(f"  Direction = {best['overall_direction']*100:.1f}%")
        print(f"\nPer-mutant breakdown:")
        for r in best['per_mutant']:
            print(f"    {r['mutation']:<15} MAE={r['mae']:.4f} "
                  f"Dir={'Y' if r['direction'] > 0.5 else 'N'}")
    
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
    
    with open(out_path / 'protein_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    
    print(f"\nResults saved to {out_path / 'protein_results.json'}")
    return results


if __name__ == '__main__':
    run_protein_experiments()
