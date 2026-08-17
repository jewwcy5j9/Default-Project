"""
Meta-Learning CDST: Cross-System Transfer via MAML/Reptile.

Key insight: Each protein/system provides only 5-8 training pairs,
but multiple related systems share common "transition structure".
Meta-learning finds an initialization that adapts quickly to new systems.

Approach:
- Treat each protein system as a "task" T_i = {(w_j, c_j, w'_j)}
- Inner loop: adapt model to task T_i with few gradient steps
- Outer loop: optimize initialization across all tasks
- Test: leave-one-protein-out cross-validation

Variants:
- MAML: second-order gradient through adaptation
- Reptile: first-order approximation (more stable, cheaper)
- ANIL: only adapt head layers (faster)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from copy import deepcopy


class MetaCDST(nn.Module):
    """Base CDST model designed for meta-learning.
    
    Architecture is kept simple to enable fast adaptation:
    - Linear intervention encoder
    - Low-rank transition
    """
    
    def __init__(self, K: int, d: int, rank: int = 2, hidden: int = 32):
        super().__init__()
        self.K = K
        self.d = d
        self.rank = rank
        
        # Intervention encoder (adaptable)
        self.encoder = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, rank)
        )
        
        # State-space directions (shared across tasks)
        self.U = nn.Parameter(torch.randn(K, rank) * 0.1)
        
        # Bias term (task-specific)
        self.bias = nn.Parameter(torch.zeros(K))
        
    def forward(self, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Predict shifted distribution."""
        log_w = torch.log(w.clamp(min=1e-10))
        g_c = self.encoder(c)  # [batch, rank]
        delta = g_c @ self.U.T + self.bias  # [batch, K]
        return F.softmax(log_w + delta, dim=-1)
    
    def get_adaptable_params(self) -> List[nn.Parameter]:
        """Parameters to adapt in inner loop."""
        return list(self.encoder.parameters()) + [self.bias]
    
    def get_shared_params(self) -> List[nn.Parameter]:
        """Parameters shared across tasks (outer loop only)."""
        return [self.U]


class MAMLTrainer:
    """MAML (Model-Agnostic Meta-Learning) trainer.
    
    Finds initialization theta such that after k gradient steps
    on a new task, performance is optimal.
    
    Algorithm:
        1. Sample batch of tasks {T_i}
        2. For each task:
           a. Split into support (train) and query (val) sets
           b. Adapt: theta'_i = theta - alpha * grad(L_Ti(theta))
           c. Evaluate: L_query(T_i, theta'_i)
        3. Update: theta -= beta * sum_i grad(L_query(T_i, theta'_i))
    """
    
    def __init__(self, model: MetaCDST, 
                 inner_lr: float = 0.01,
                 outer_lr: float = 0.001,
                 inner_steps: int = 5,
                 first_order: bool = False):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        
        self.meta_optimizer = torch.optim.Adam(
            model.parameters(), lr=outer_lr
        )
    
    def adapt(self, model: MetaCDST,
              w_support: torch.Tensor,
              c_support: torch.Tensor,
              wt_support: torch.Tensor,
              n_steps: Optional[int] = None) -> MetaCDST:
        """Inner loop: adapt model to a single task.

        n_steps overrides the trainer's inner_steps (used by
        evaluate_new_task's test-time adaptation budget).

        Returns adapted model (differentiable if not first_order).
        """
        adapted_model = deepcopy(model)
        inner_opt = torch.optim.SGD(
            adapted_model.get_adaptable_params(),
            lr=self.inner_lr
        )

        for _ in range(n_steps or self.inner_steps):
            inner_opt.zero_grad()
            pred = adapted_model(w_support, c_support)
            loss = F.mse_loss(pred, wt_support)
            loss.backward()
            inner_opt.step()
            
            if self.first_order:
                # Detach for first-order approximation
                for p in adapted_model.parameters():
                    p.data = p.data.detach()
                    p.requires_grad_(True)
        
        return adapted_model
    
    def meta_step(self, tasks: List[Dict[str, np.ndarray]]) -> float:
        """One meta-learning step across a batch of tasks.
        
        Args:
            tasks: list of dicts with 'w', 'c', 'wt' arrays
                   Each is split 50/50 into support/query
                   
        Returns:
            Meta-loss (average query loss after adaptation)
        """
        self.meta_optimizer.zero_grad()
        total_meta_loss = 0.0
        
        for task in tasks:
            w = torch.FloatTensor(task['w'])
            c = torch.FloatTensor(task['c'])
            wt = torch.FloatTensor(task['wt'])
            
            n = len(w)
            n_support = max(1, n // 2)
            
            # Split support/query
            w_s, c_s, wt_s = w[:n_support], c[:n_support], wt[:n_support]
            w_q, c_q, wt_q = w[n_support:], c[n_support:], wt[n_support:]
            
            if len(w_q) == 0:
                w_q, c_q, wt_q = w_s, c_s, wt_s
            
            # Inner loop: adapt
            adapted = self.adapt(self.model, w_s, c_s, wt_s)
            
            # Outer loop: evaluate on query
            pred_q = adapted(w_q, c_q)
            query_loss = F.mse_loss(pred_q, wt_q)
            total_meta_loss += query_loss
        
        # Average and backprop
        meta_loss = total_meta_loss / len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def evaluate_new_task(self, task: Dict[str, np.ndarray],
                          n_adapt_steps: Optional[int] = None) -> Dict:
        """Evaluate on a new task with adaptation.
        
        This is the test-time procedure:
        1. Start from meta-learned initialization
        2. Adapt with support set
        3. Evaluate on query set
        """
        steps = n_adapt_steps or self.inner_steps * 2

        w = torch.FloatTensor(task['w'])
        c = torch.FloatTensor(task['c'])
        wt = torch.FloatTensor(task['wt'])

        n = len(w)
        n_support = max(1, n // 2)

        w_s, c_s, wt_s = w[:n_support], c[:n_support], wt[:n_support]
        w_q, c_q, wt_q = w[n_support:], c[n_support:], wt[n_support:]

        if len(w_q) == 0:
            w_q, c_q, wt_q = w_s, c_s, wt_s

        # Adapt from meta-init, honouring the requested adaptation budget
        adapted = self.adapt(self.model, w_s, c_s, wt_s, n_steps=steps)
        
        # Evaluate
        adapted.eval()
        with torch.no_grad():
            pred = adapted(w_q, c_q)
            mae = (pred - wt_q).abs().mean().item()
            
            # Direction accuracy (for K=2)
            if self.model.K == 2:
                true_dir = (wt_q[:, 0] > w_q[:, 0]).float()
                pred_dir = (pred[:, 0] > w_q[:, 0]).float()
                dir_acc = (true_dir == pred_dir).float().mean().item()
            else:
                dir_acc = None
        
        return {'mae': mae, 'direction_accuracy': dir_acc}


class ReptileTrainer:
    """Reptile meta-learning (first-order, more stable).
    
    Algorithm:
        1. Sample task T_i
        2. Adapt: theta'_i = SGD(theta, T_i, k steps)
        3. Update: theta += epsilon * (theta'_i - theta)
    
    Simpler than MAML (no second-order gradients), often works better.
    """
    
    def __init__(self, model: MetaCDST,
                 inner_lr: float = 0.01,
                 outer_lr: float = 0.1,
                 inner_steps: int = 10,
                 seed: int = 0):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        # Dedicated seeded RNG: sampling from numpy's global RNG made
        # leave-one-system-out runs irreproducible.
        self.rng = np.random.default_rng(seed)
    
    def reptile_step(self, task: Dict[str, np.ndarray]) -> float:
        """One Reptile update step.
        
        Args:
            task: dict with 'w', 'c', 'wt' arrays
            
        Returns:
            Task loss after adaptation
        """
        w = torch.FloatTensor(task['w'])
        c = torch.FloatTensor(task['c'])
        wt = torch.FloatTensor(task['wt'])
        
        # Save initial parameters
        initial_params = {n: p.clone() for n, p in self.model.named_parameters()}
        
        # Inner loop: SGD adaptation
        inner_opt = torch.optim.SGD(
            self.model.parameters(), lr=self.inner_lr
        )
        
        for _ in range(self.inner_steps):
            inner_opt.zero_grad()
            pred = self.model(w, c)
            loss = F.mse_loss(pred, wt)
            loss.backward()
            inner_opt.step()
        
        # Compute final loss
        self.model.eval()
        with torch.no_grad():
            final_loss = F.mse_loss(self.model(w, c), wt).item()
        self.model.train()
        
        # Reptile update: move initial params toward adapted params
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                adapted = param.data
                original = initial_params[name]
                param.data = original + self.outer_lr * (adapted - original)
        
        return final_loss
    
    def train(self, tasks: List[Dict[str, np.ndarray]], 
              n_epochs: int = 1000,
              tasks_per_step: int = 4) -> List[float]:
        """Full Reptile training loop.
        
        Args:
            tasks: list of task dicts
            n_epochs: number of meta-updates
            tasks_per_step: tasks sampled per update
            
        Returns:
            List of average losses per epoch
        """
        losses = []

        for epoch in range(n_epochs):
            # Sample task batch from the trainer's seeded RNG
            indices = self.rng.choice(len(tasks),
                                      min(tasks_per_step, len(tasks)),
                                      replace=False)
            
            epoch_losses = []
            for idx in indices:
                loss = self.reptile_step(tasks[idx])
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}")
        
        return losses
    
    def evaluate_new_task(self, task: Dict[str, np.ndarray],
                          n_adapt_steps: Optional[int] = None) -> Dict:
        """Evaluate on new task with adaptation from meta-init."""
        steps = n_adapt_steps or self.inner_steps * 2
        
        # Save meta-params
        meta_params = {n: p.clone() for n, p in self.model.named_parameters()}
        
        w = torch.FloatTensor(task['w'])
        c = torch.FloatTensor(task['c'])
        wt = torch.FloatTensor(task['wt'])
        
        n = len(w)
        n_support = max(1, n // 2)
        w_s, c_s, wt_s = w[:n_support], c[:n_support], wt[:n_support]
        w_q, c_q, wt_q = w[n_support:], c[n_support:], wt[n_support:]
        
        if len(w_q) == 0:
            w_q, c_q, wt_q = w_s, c_s, wt_s
        
        # Adapt
        inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        for _ in range(steps):
            inner_opt.zero_grad()
            pred = self.model(w_s, c_s)
            loss = F.mse_loss(pred, wt_s)
            loss.backward()
            inner_opt.step()
        
        # Evaluate
        self.model.eval()
        with torch.no_grad():
            pred = self.model(w_q, c_q)
            mae = (pred - wt_q).abs().mean().item()
        
        # Restore meta-params
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = meta_params[name]
        
        return {'mae': mae}


def leave_one_system_out(tasks: List[Dict[str, np.ndarray]],
                         K: int, d: int, rank: int = 2,
                         method: str = 'reptile',
                         **kwargs) -> Dict:
    """Leave-one-system-out cross-validation.
    
    For each system i:
        1. Train meta-model on all systems except i
        2. Adapt to system i with its data
        3. Evaluate on held-out portion of system i
    
    Args:
        tasks: list of task dicts (one per system)
        K, d, rank: model dimensions
        method: 'maml' or 'reptile'
        
    Returns:
        Dict with per-system and average metrics
    """
    n_systems = len(tasks)
    results = []
    
    for hold_out in range(n_systems):
        print(f"  Hold-out system {hold_out+1}/{n_systems}")
        
        # Create fresh model
        model = MetaCDST(K=K, d=d, rank=rank)
        
        # Training tasks (all except held-out)
        train_tasks = [t for i, t in enumerate(tasks) if i != hold_out]
        test_task = tasks[hold_out]
        
        if method == 'reptile':
            trainer = ReptileTrainer(model, **kwargs)
            trainer.train(train_tasks, n_epochs=500)
        else:
            trainer = MAMLTrainer(model, **kwargs)
            for epoch in range(500):
                trainer.meta_step(train_tasks)
        
        # Evaluate on held-out system
        result = trainer.evaluate_new_task(test_task)
        result['system'] = hold_out
        results.append(result)
        print(f"    MAE={result['mae']:.4f}")
    
    # Aggregate
    avg_mae = np.mean([r['mae'] for r in results])
    
    return {
        'per_system': results,
        'avg_mae': avg_mae,
        'method': method
    }
