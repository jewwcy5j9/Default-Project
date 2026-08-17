"""
Training Pipeline for Protein CDST

Handles training on protein mutation data with proper validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json
from tqdm import tqdm

from ..models.protein_cdst import ProteinCDST, ProteinCDSTLoss, create_protein_cdst
from ..models.baselines import ConditionalFlowMatching, ConditionalVAE, DirectMLP, LinearResponseTheory


class ProteinDataLoader:
    """Load and preprocess protein mutation data for CDST training."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
    
    def load_abl1(self) -> Dict[str, np.ndarray]:
        """Load Abl1 dataset."""
        samples = np.load(self.data_dir / 'abl1_samples.npz', allow_pickle=True)
        return {
            'w': samples['w'],
            'c': samples['c'],
            'w_target': samples['w_target'],
            'mutations': samples['mutations'],
        }
    
    def create_splits(
        self,
        data: Dict[str, np.ndarray],
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> Tuple[Dict, Dict]:
        """Create train/test splits."""
        n = len(data['w'])
        indices = np.random.RandomState(seed).permutation(n)
        n_train = int(n * train_ratio)
        
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        train_data = {k: v[train_idx] for k, v in data.items()}
        test_data = {k: v[test_idx] for k, v in data.items()}
        
        return train_data, test_data


def train_protein_cdst(
    model: ProteinCDST,
    train_data: Dict[str, np.ndarray],
    val_data: Optional[Dict[str, np.ndarray]] = None,
    n_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = 'cpu',
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Train Protein CDST model.
    
    Args:
        model: ProteinCDST instance
        train_data: Training data dict with 'w', 'c', 'w_target'
        val_data: Optional validation data
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        verbose: Whether to print progress
    
    Returns:
        history: Dict of loss histories
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = ProteinCDSTLoss()
    
    # Convert to tensors
    w_train = torch.FloatTensor(train_data['w']).to(device)
    c_train = torch.FloatTensor(train_data['c']).to(device)
    w_target_train = torch.FloatTensor(train_data['w_target']).to(device)
    
    train_dataset = TensorDataset(w_train, c_train, w_target_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    history = {'train_loss': [], 'train_kl': [], 'val_kl': []}
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_kl = 0
        
        for w_batch, c_batch, wt_batch in train_loader:
            optimizer.zero_grad()
            loss, loss_dict = criterion(model, w_batch, c_batch, wt_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss_dict['total']
            epoch_kl += loss_dict['kl']
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_kl = epoch_kl / len(train_loader)
        history['train_loss'].append(avg_loss)
        history['train_kl'].append(avg_kl)
        
        # Validation
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                w_val = torch.FloatTensor(val_data['w']).to(device)
                c_val = torch.FloatTensor(val_data['c']).to(device)
                wt_val = torch.FloatTensor(val_data['w_target']).to(device)
                
                w_pred = model(w_val, c_val)
                val_kl = F.kl_div(torch.log(w_pred + 1e-8), wt_val, reduction='batchmean')
                history['val_kl'].append(val_kl.item())
        
        if verbose and (epoch + 1) % 50 == 0:
            msg = f"Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, kl={avg_kl:.4f}"
            if val_data is not None:
                msg += f", val_kl={history['val_kl'][-1]:.4f}"
            print(msg)
    
    return history


def evaluate_protein_model(
    model: nn.Module,
    test_data: Dict[str, np.ndarray],
    device: str = 'cpu',
) -> Dict[str, float]:
    """Evaluate model on test data.
    
    Returns:
        metrics: Dict with KL, PCC, direction accuracy, MAE
    """
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        w = torch.FloatTensor(test_data['w']).to(device)
        c = torch.FloatTensor(test_data['c']).to(device)
        w_true = torch.FloatTensor(test_data['w_target']).to(device)
        
        if isinstance(model, LinearResponseTheory):
            w_pred = torch.FloatTensor(model.predict(test_data['w'], test_data['c'])).to(device)
        else:
            w_pred = model(w, c)
        
        # KL divergence
        kl = F.kl_div(torch.log(w_pred + 1e-8), w_true, reduction='batchmean').item()
        
        # Per-state PCC
        K = w_true.shape[1]
        pccs = []
        for k in range(K):
            pred_k = w_pred[:, k].cpu().numpy()
            true_k = w_true[:, k].cpu().numpy()
            if np.std(pred_k) > 1e-8 and np.std(true_k) > 1e-8:
                pcc = np.corrcoef(pred_k, true_k)[0, 1]
            else:
                pcc = 1.0 if np.allclose(pred_k, true_k, atol=1e-4) else 0.0
            pccs.append(pcc)
        mean_pcc = np.mean(pccs)
        
        # Direction accuracy (elementwise sign agreement over all states).
        # NOTE: this exploratory convention counts a true tie (delta == 0) as
        # correct only when the prediction is exactly zero, and differs from
        # the frozen canonical per-mutant definition (|NMR - WT| < 0.05 ties
        # excluded from numerator and denominator, canonical_encoding.py).
        # Do not quote the two side by side.
        w_base = test_data['w'].mean(axis=0)
        delta_pred = (w_pred - torch.FloatTensor(w_base).to(device)).cpu().numpy()
        delta_true = (w_true - torch.FloatTensor(w_base).to(device)).cpu().numpy()
        direction_acc = np.mean(np.sign(delta_pred) == np.sign(delta_true))
        
        # MAE
        mae = torch.abs(w_pred - w_true).mean().item()
    
    return {
        'kl': kl,
        'mean_pcc': mean_pcc,
        'pcc_per_state': pccs,
        'direction_acc': direction_acc,
        'mae': mae,
    }


def run_protein_experiment(
    data_dir: Path,
    output_dir: Path,
    K: int = 3,
    device: str = 'cpu',
    n_seeds: int = 3,
) -> Dict[str, Dict]:
    """Run full protein mutation experiment.
    
    Trains CDST and baselines, evaluates on test set.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    loader = ProteinDataLoader(data_dir)
    data = loader.load_abl1()
    
    print(f"Loaded {len(data['w'])} samples")
    print(f"K={K}, mutation_dim={data['c'].shape[1]}")
    
    results = {}
    
    for seed in range(n_seeds):
        print(f"\n=== Seed {seed+1}/{n_seeds} ===")

        # Seed torch as well as the numpy split: without this, model init,
        # DataLoader shuffling and the flow/VAE stochastic terms all varied
        # run-to-run, so the reported per-seed aggregates were irreproducible.
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Split data
        train_data, test_data = loader.create_splits(data, train_ratio=0.75, seed=seed)

        # CDST
        print("Training CDST...")
        mutation_dim = train_data['c'].shape[1]
        cdst = ProteinCDST(
            K=K,
            mutation_input_dim=mutation_dim,
            hidden_dim=128,
            latent_dim=64,
        )
        # NOTE: the "validation" history is computed on the final test set
        # (monitoring only). It must never gate early stopping or model
        # selection — that would be test-set selection.
        history = train_protein_cdst(
            cdst, train_data, val_data=test_data,
            n_epochs=200, device=device, verbose=False
        )
        cdst_metrics = evaluate_protein_model(cdst, test_data, device)
        print(f"  CDST: KL={cdst_metrics['kl']:.4f}, PCC={cdst_metrics['mean_pcc']:.4f}")
        
        # cFlow baseline
        print("Training cFlow...")
        cflow = ConditionalFlowMatching(K=K, intervention_dim=data['c'].shape[1], hidden_dim=64)
        # Train cFlow
        cflow = cflow.to(device)
        optimizer = torch.optim.Adam(cflow.parameters(), lr=1e-3)
        w_train = torch.FloatTensor(train_data['w']).to(device)
        c_train = torch.FloatTensor(train_data['c']).to(device)
        wt_train = torch.FloatTensor(train_data['w_target']).to(device)
        
        for epoch in range(200):
            optimizer.zero_grad()
            loss = cflow.compute_loss(w_train, c_train, wt_train)
            loss.backward()
            optimizer.step()
        
        cflow_metrics = evaluate_protein_model(cflow, test_data, device)
        print(f"  cFlow: KL={cflow_metrics['kl']:.4f}, PCC={cflow_metrics['mean_pcc']:.4f}")
        
        # DirectMLP baseline
        print("Training DirectMLP...")
        direct = DirectMLP(K=K, intervention_dim=data['c'].shape[1], hidden_dim=64)
        direct = direct.to(device)
        optimizer = torch.optim.Adam(direct.parameters(), lr=1e-3)
        
        for epoch in range(200):
            optimizer.zero_grad()
            loss = direct.compute_loss(w_train, c_train, wt_train)
            loss.backward()
            optimizer.step()
        
        direct_metrics = evaluate_protein_model(direct, test_data, device)
        print(f"  DirectMLP: KL={direct_metrics['kl']:.4f}, PCC={direct_metrics['mean_pcc']:.4f}")
        
        results[f'seed_{seed}'] = {
            'CDST': cdst_metrics,
            'cFlow': cflow_metrics,
            'DirectMLP': direct_metrics,
        }
    
    # Aggregate results
    print("\n=== Aggregate Results ===")
    for method in ['CDST', 'cFlow', 'DirectMLP']:
        kls = [results[f'seed_{s}'][method]['kl'] for s in range(n_seeds)]
        pccs = [results[f'seed_{s}'][method]['mean_pcc'] for s in range(n_seeds)]
        print(f"{method}: KL={np.mean(kls):.4f}+-{np.std(kls):.4f}, PCC={np.mean(pccs):.4f}+-{np.std(pccs):.4f}")
    
    # Save results
    with open(output_dir / 'protein_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/abl1')
    parser.add_argument('--output_dir', type=str, default='experiments/protein/results')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_seeds', type=int, default=3)
    args = parser.parse_args()
    
    run_protein_experiment(
        Path(args.data_dir),
        Path(args.output_dir),
        device=args.device,
        n_seeds=args.n_seeds,
    )
