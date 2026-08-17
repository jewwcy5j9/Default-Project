"""
Compositional CDST: Exploiting Additivity of Intervention Effects.

Key insight: In many systems, combined interventions have approximately
additive effects on the log-odds:
    Delta_logits(c1 + c2) ≈ Delta_logits(c1) + Delta_logits(c2)

This enables:
1. Data augmentation: n single interventions -> C(n,2) composed pairs
2. Better generalization: compositional structure constrains the function space
3. Zero-shot composition: predict combined effects without seeing them

Architecture variants:
- Strict additive: Delta(c1+c2) = Delta(c1) + Delta(c2) exactly
- Relaxed additive: Delta(c1,c2) = Delta(c1) + Delta(c2) + epsilon(c1,c2)
- Gated interaction: epsilon is learned but regularized to be small
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
from itertools import combinations


class CompositionalCDST(nn.Module):
    """CDST with strict additivity constraint.
    
    The transition is decomposed as:
        Delta_logits(c) = U @ g(c)
    where g is linear: g(c1 + c2) = g(c1) + g(c2)
    
    This means: Delta(c1+c2) = U @ (g(c1) + g(c2)) = Delta(c1) + Delta(c2)
    
    Parameters:
        K: number of states
        d: intervention dimension
        rank: rank of the transition (default: min(K, d))
    """
    
    def __init__(self, K: int, d: int, rank: Optional[int] = None):
        super().__init__()
        self.K = K
        self.d = d
        self.rank = rank or min(K, d)
        
        # Linear intervention encoder (enforces additivity)
        self.g = nn.Linear(d, self.rank, bias=False)
        
        # State-space directions
        self.U = nn.Parameter(torch.randn(K, self.rank) * 0.1)
        
    def delta_logits(self, c: torch.Tensor) -> torch.Tensor:
        """Compute Delta_logits with guaranteed additivity."""
        return self.g(c) @ self.U.T  # [batch, K]
    
    def forward(self, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Predict shifted distribution.
        
        Args:
            w: current distribution [batch, K]
            c: intervention descriptor [batch, d]
            
        Returns:
            w': predicted shifted distribution [batch, K]
        """
        log_w = torch.log(w.clamp(min=1e-10))
        delta = self.delta_logits(c)
        log_w_prime = log_w + delta
        return F.softmax(log_w_prime, dim=-1)
    
    def compose(self, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        """Predict effect of combined intervention (zero-shot).
        
        Due to linearity: Delta(c1+c2) = Delta(c1) + Delta(c2)
        """
        return self.delta_logits(c1) + self.delta_logits(c2)
    
    def forward_composed(self, w: torch.Tensor, c1: torch.Tensor, 
                         c2: torch.Tensor) -> torch.Tensor:
        """Predict distribution under combined intervention."""
        log_w = torch.log(w.clamp(min=1e-10))
        delta = self.compose(c1, c2)
        return F.softmax(log_w + delta, dim=-1)


class InteractionCDST(nn.Module):
    """CDST with relaxed additivity: additive + learned interaction term.
    
    Delta(c1, c2) = Delta(c1) + Delta(c2) + epsilon(c1, c2)
    
    The interaction term epsilon is regularized to be small,
    allowing slight deviations from pure additivity.
    """
    
    def __init__(self, K: int, d: int, rank: Optional[int] = None,
                 interaction_strength: float = 0.1):
        super().__init__()
        self.K = K
        self.d = d
        self.rank = rank or min(K, d)
        self.interaction_strength = interaction_strength
        
        # Additive components
        self.g = nn.Linear(d, self.rank, bias=False)
        self.U = nn.Parameter(torch.randn(K, self.rank) * 0.1)
        
        # Interaction network (small output)
        self.interaction_net = nn.Sequential(
            nn.Linear(2 * d, 64),
            nn.ReLU(),
            nn.Linear(64, K),
            nn.Tanh()  # Bounded interaction
        )
        
        # Initialize interaction to near-zero
        for p in self.interaction_net.parameters():
            p.data *= 0.01
            
    def delta_single(self, c: torch.Tensor) -> torch.Tensor:
        """Single intervention effect (additive part)."""
        return self.g(c) @ self.U.T
    
    def delta_interaction(self, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        """Interaction term for combined interventions."""
        combined = torch.cat([c1, c2], dim=-1)
        return self.interaction_strength * self.interaction_net(combined)
    
    def forward(self, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Single intervention prediction."""
        log_w = torch.log(w.clamp(min=1e-10))
        delta = self.delta_single(c)
        return F.softmax(log_w + delta, dim=-1)
    
    def forward_composed(self, w: torch.Tensor, c1: torch.Tensor,
                         c2: torch.Tensor) -> torch.Tensor:
        """Composed intervention with interaction."""
        log_w = torch.log(w.clamp(min=1e-10))
        delta = (self.delta_single(c1) + self.delta_single(c2) + 
                 self.delta_interaction(c1, c2))
        return F.softmax(log_w + delta, dim=-1)
    
    def interaction_magnitude(self, c1: torch.Tensor, c2: torch.Tensor) -> float:
        """Measure interaction strength (for monitoring)."""
        with torch.no_grad():
            return self.delta_interaction(c1, c2).abs().mean().item()


class CompositionalDataAugmenter:
    """Generate composed training pairs from single-intervention data.
    
    Given n single-intervention pairs (w, c_i, w'_i), generates
    composed pairs (w, c_i + c_j, w''_ij) where w''_ij is computed
    by applying both shifts sequentially.
    
    This provides C(n,2) additional training pairs for free.
    """
    
    def __init__(self, assume_commutative: bool = True):
        self.assume_commutative = assume_commutative
    
    def augment(self, w: np.ndarray, c: np.ndarray, 
                w_prime: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate augmented dataset.
        
        Args:
            w: base distributions [n, K]
            c: intervention descriptors [n, d]
            w_prime: shifted distributions [n, K]
            
        Returns:
            Tuple of (w_aug, c_aug, w_prime_aug) including original + composed
        """
        n, K = w.shape
        d = c.shape[1]
        
        # Compute log-ratios (Delta_logits) for each intervention
        delta_logits = np.log(w_prime.clip(1e-10)) - np.log(w.clip(1e-10))
        # Center (remove mean shift which is absorbed by softmax)
        delta_logits = delta_logits - delta_logits.mean(axis=1, keepdims=True)
        
        # Original data
        all_w = [w]
        all_c = [c]
        all_w_prime = [w_prime]
        
        # Generate composed pairs
        for i, j in combinations(range(n), 2):
            # Composed intervention, emitted as a CONCATENATION [c_i, c_j]
            # (dim 2d) so the rows feed CompositionalCDST.forward_composed
            # through train_compositional's half-split unchanged. (The former
            # element-wise sum had dimension d and crashed that path.)
            c_composed = np.concatenate([c[i], c[j]])

            # Composed effect (additive in log-space)
            delta_composed = delta_logits[i] + delta_logits[j]

            # Apply to base distribution (use average of w[i] and w[j] as base)
            w_base = (w[i] + w[j]) / 2
            log_w_new = np.log(w_base.clip(1e-10)) + delta_composed
            w_composed = np.exp(log_w_new)
            w_composed = w_composed / w_composed.sum()

            all_w.append(w_base[None])
            all_c.append(c_composed[None])
            all_w_prime.append(w_composed[None])

            # NOTE: no reverse-order rows. Additive log-space composition is
            # symmetric by construction, so (j, i) would duplicate (i, j)
            # exactly; assume_commutative=False cannot create order
            # information that this representation does not carry.
        
        return (np.concatenate(all_w, axis=0),
                np.concatenate(all_c, axis=0),
                np.concatenate(all_w_prime, axis=0))


def train_compositional(model, w_train, c_train, wt_train,
                        w_test, c_test, wt_test,
                        n_epochs=500, lr=1e-3,
                        composed_w=None, composed_c=None, composed_wt=None,
                        interaction_reg=0.01):
    """Train compositional model with optional augmented data.
    
    Args:
        model: CompositionalCDST or InteractionCDST
        w_train, c_train, wt_train: single-intervention training data
        w_test, c_test, wt_test: test data (may include composed)
        composed_w/c/wt: augmented composed training pairs
        interaction_reg: regularization strength for interaction term
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Convert to tensors
    w_t = torch.FloatTensor(w_train)
    c_t = torch.FloatTensor(c_train)
    wt_t = torch.FloatTensor(wt_train)
    
    # Add composed data if available
    if composed_w is not None:
        cw_t = torch.FloatTensor(composed_w)
        cc_t = torch.FloatTensor(composed_c)
        cwt_t = torch.FloatTensor(composed_wt)
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Single-intervention loss
        pred = model(w_t, c_t)
        loss = F.mse_loss(pred, wt_t)
        
        # Composed data loss
        if composed_w is not None and hasattr(model, 'forward_composed'):
            # For composed data, split c into c1, c2 (assume equal halves)
            d_half = cc_t.shape[1] // 2
            c1 = cc_t[:, :d_half]
            c2 = cc_t[:, d_half:]
            pred_composed = model.forward_composed(cw_t, c1, c2)
            loss = loss + F.mse_loss(pred_composed, cwt_t)
        
        # Interaction regularization
        if hasattr(model, 'interaction_net') and interaction_reg > 0:
            for p in model.interaction_net.parameters():
                loss = loss + interaction_reg * p.pow(2).sum()
        
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        w_te = torch.FloatTensor(w_test)
        c_te = torch.FloatTensor(c_test)
        wt_te = torch.FloatTensor(wt_test)
        pred_te = model(w_te, c_te)
        mae = (pred_te - wt_te).abs().mean().item()
        
    return {'mae': mae, 'train_loss': best_loss}
