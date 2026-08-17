"""
P1 Validation: Compare loss functions and model variants on Abl1 AF2 data.

Tests:
1. CDST-vanilla (KL loss) - existing baseline
2. CDST + Fisher-Rao loss
3. CDST + Hellinger loss
4. CDST + Natural parameter loss
5. LowRankCDST (rank=1, rank=2)
6. AdaptiveRankCDST
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import json

from src.models.cdst import CDST
from src.models.low_rank_cdst import LowRankCDST, AdaptiveRankCDST
from src.models.losses import (
    FisherRaoLoss, HellingerLoss, SymmetricKLLoss, 
    NaturalParameterLoss, compute_all_losses
)


def train_and_eval(model, w_train, c_train, wt_train, w_test, c_test, wt_test,
                   loss_fn=None, n_epochs=500, lr=1e-3, reg_fn=None):
    """Train model and evaluate."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    w_t = torch.FloatTensor(w_train)
    c_t = torch.FloatTensor(c_train)
    wt_t = torch.FloatTensor(wt_train)
    
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = model(w_t, c_t)
        
        if loss_fn is not None:
            loss = loss_fn(pred, wt_t)
        else:
            loss = F.kl_div(pred.log(), wt_t, reduction='batchmean')
        
        if reg_fn is not None:
            loss = loss + reg_fn()
        
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        pred_test = model(torch.FloatTensor(w_test), torch.FloatTensor(c_test)).numpy()
    
    mae = np.mean(np.abs(pred_test - wt_test))
    delta_pred = pred_test - w_test
    delta_true = wt_test - w_test
    dir_acc = np.mean(np.sign(delta_pred[:, 0]) == np.sign(delta_true[:, 0]))
    
    return mae, dir_acc


def main():
    print("="*70)
    print("P1 Validation: Loss Functions & Model Variants on Abl1")
    print("="*70)
    
    # Load AF2 data
    data = np.load('data/af2_populations/cdst_training_2state.npz', allow_pickle=True)
    w_all = data['w']
    c_all = data['c']
    wt_all = data['w_target']
    mutations = data['mutations']
    
    K = 2
    c_dim = c_all.shape[1]
    n = len(w_all)
    
    print(f"\nData: {n} pairs, K={K}, c_dim={c_dim}")
    print(f"Mutations: {list(mutations)}")
    
    # Leave-one-out evaluation
    results = {}
    
    configs = [
        ('CDST-vanilla (KL)', 'cdst', 'kl', None),
        ('CDST + Fisher-Rao', 'cdst', 'fr', None),
        ('CDST + Hellinger', 'cdst', 'hellinger', None),
        ('CDST + Natural-L2', 'cdst', 'natural', None),
        ('CDST + JSD', 'cdst', 'jsd', None),
        ('LowRank-CDST (r=1)', 'lowrank1', 'fr', None),
        ('LowRank-CDST (r=2)', 'lowrank2', 'fr', None),
        ('AdaptiveRank-CDST', 'adaptive', 'fr', 'nuclear'),
    ]
    
    print(f"\n{'Method':<28} {'LOO-MAE':<12} {'LOO-Dir':<10} {'Train-MAE':<12}")
    print("-"*62)
    
    for name, model_type, loss_type, reg_type in configs:
        loo_maes = []
        loo_dirs = []
        
        for i in range(n):
            # LOO split
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            
            w_train, c_train, wt_train = w_all[mask], c_all[mask], wt_all[mask]
            w_test, c_test, wt_test = w_all[i:i+1], c_all[i:i+1], wt_all[i:i+1]
            
            # Create model
            torch.manual_seed(42)
            if model_type == 'cdst':
                model = CDST(K=K, intervention_dim=c_dim, hidden_dim=64, latent_dim=32)
            elif model_type == 'lowrank1':
                model = LowRankCDST(K=K, intervention_dim=c_dim, rank=1, hidden_dim=32)
            elif model_type == 'lowrank2':
                model = LowRankCDST(K=K, intervention_dim=c_dim, rank=2, hidden_dim=32)
            elif model_type == 'adaptive':
                model = AdaptiveRankCDST(K=K, intervention_dim=c_dim, hidden_dim=32,
                                         lambda_nuclear=0.01)
            
            # Select loss
            if loss_type == 'kl':
                loss_fn = None  # Use default KL in train_and_eval
            elif loss_type == 'fr':
                loss_fn = FisherRaoLoss(mode='hellinger')
            elif loss_type == 'hellinger':
                loss_fn = HellingerLoss(squared=True)
            elif loss_type == 'natural':
                loss_fn = NaturalParameterLoss()
            elif loss_type == 'jsd':
                loss_fn = SymmetricKLLoss()
            
            # Regularization
            reg_fn = None
            if reg_type == 'nuclear' and hasattr(model, 'regularization_loss'):
                reg_fn = model.regularization_loss
            
            mae, dir_acc = train_and_eval(
                model, w_train, c_train, wt_train,
                w_test, c_test, wt_test,
                loss_fn=loss_fn, n_epochs=500, lr=1e-3, reg_fn=reg_fn
            )
            loo_maes.append(mae)
            loo_dirs.append(dir_acc)
        
        # Also train on full data for train MAE
        torch.manual_seed(42)
        if model_type == 'cdst':
            model_full = CDST(K=K, intervention_dim=c_dim, hidden_dim=64, latent_dim=32)
        elif model_type == 'lowrank1':
            model_full = LowRankCDST(K=K, intervention_dim=c_dim, rank=1, hidden_dim=32)
        elif model_type == 'lowrank2':
            model_full = LowRankCDST(K=K, intervention_dim=c_dim, rank=2, hidden_dim=32)
        elif model_type == 'adaptive':
            model_full = AdaptiveRankCDST(K=K, intervention_dim=c_dim, hidden_dim=32)
        
        if loss_type == 'kl':
            loss_fn_full = None
        elif loss_type == 'fr':
            loss_fn_full = FisherRaoLoss(mode='hellinger')
        elif loss_type == 'hellinger':
            loss_fn_full = HellingerLoss(squared=True)
        elif loss_type == 'natural':
            loss_fn_full = NaturalParameterLoss()
        elif loss_type == 'jsd':
            loss_fn_full = SymmetricKLLoss()
        
        reg_fn_full = None
        if reg_type == 'nuclear' and hasattr(model_full, 'regularization_loss'):
            reg_fn_full = model_full.regularization_loss
        
        train_mae, train_dir = train_and_eval(
            model_full, w_all, c_all, wt_all, w_all, c_all, wt_all,
            loss_fn=loss_fn_full, n_epochs=500, lr=1e-3, reg_fn=reg_fn_full
        )
        
        loo_mae_mean = np.mean(loo_maes)
        loo_dir_mean = np.mean(loo_dirs)
        
        print(f"{name:<28} {loo_mae_mean:<12.4f} {loo_dir_mean*100:<10.1f} {train_mae:<12.4f}")
        
        results[name] = {
            'loo_mae': float(loo_mae_mean),
            'loo_mae_std': float(np.std(loo_maes)),
            'loo_dir': float(loo_dir_mean),
            'train_mae': float(train_mae),
        }
    
    # Save
    output_dir = Path('experiments/iclr_restructuring/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'p1_loss_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {output_dir / 'p1_loss_comparison.json'}")
    
    # Summary
    print(f"\n{'='*70}")
    print("Key Findings:")
    best_loo = min(results.items(), key=lambda x: x[1]['loo_mae'])
    print(f"  Best LOO-MAE: {best_loo[0]} ({best_loo[1]['loo_mae']:.4f})")
    best_dir = max(results.items(), key=lambda x: x[1]['loo_dir'])
    print(f"  Best LOO-Dir: {best_dir[0]} ({best_dir[1]['loo_dir']*100:.1f}%)")


if __name__ == '__main__':
    main()
