"""
Low-Rank CDST: Distributional Shift Prediction with Structured Transitions.

Key insight: In real systems, interventions typically affect only a few
"directions" on the simplex (e.g., kinases have DFG-in/out as main DoF).
The Delta_logits vector is therefore low-rank when viewed as a linear
operator from intervention space to state space.

By explicitly constraining the transition to be low-rank, we reduce
sample complexity from O(K*d) to O(r*d) where r << K is the effective rank.

Architecture:
    Delta_logits = U @ g(c)
    where U in R^{K x r} (state-space directions)
          g: R^d -> R^r (intervention projection)
    
    w' = softmax(log(w) + U @ g(c))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class LowRankTransition(nn.Module):
    """Low-rank factorization of the transition operator.
    
    Instead of predicting Delta_logits in R^K directly (K parameters per sample),
    we predict in a reduced r-dimensional space and project back.
    
    This is equivalent to assuming the transition lies in an r-dimensional
    subspace of R^K, which is natural when the system has r dominant
    conformational degrees of freedom.
    """
    
    def __init__(self, K: int, intervention_dim: int, rank: int = 2,
                 hidden_dim: int = 64):
        """
        Args:
            K: number of states (simplex dimension)
            intervention_dim: dimension of intervention encoding
            rank: rank of the transition (number of active directions)
            hidden_dim: hidden layer width for g(c)
        """
        super().__init__()
        if rank > K:
            raise ValueError(
                f"rank={rank} exceeds K={K}; QR of a (K, rank) factor would "
                "silently truncate and train a full-rank model instead")
        self.K = K
        self.rank = rank
        self.intervention_dim = intervention_dim

        # U: state-space directions (K x r)
        # Initialize with orthogonal vectors for numerical stability
        U_init = torch.randn(K, rank)
        U_init, _ = torch.linalg.qr(U_init)
        self.U = nn.Parameter(U_init[:, :rank])
        
        # g: intervention -> reduced space (d -> r)
        self.g = nn.Sequential(
            nn.Linear(intervention_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, rank),
        )
        
        # Initialize g output near zero (zero perturbation = identity)
        nn.init.zeros_(self.g[-1].weight)
        nn.init.zeros_(self.g[-1].bias)
    
    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c: (batch, intervention_dim)
        Returns:
            delta_logits: (batch, K)
        """
        # g(c) in R^r, then project to R^K via U
        reduced = self.g(c)  # (batch, r)
        delta_logits = reduced @ self.U.T  # (batch, K)
        return delta_logits
    
    def get_effective_rank(self, threshold: float = 0.01) -> int:
        """Compute effective rank based on singular value decay."""
        with torch.no_grad():
            # The effective transition matrix is U (K x r)
            # Its rank is at most r, but could be less if U is degenerate
            svs = torch.linalg.svdvals(self.U)
            svs_normalized = svs / svs[0].clamp(min=1e-10)
            return int((svs_normalized > threshold).sum().item())
    
    def get_principal_directions(self) -> torch.Tensor:
        """Return the principal transition directions (columns of U)."""
        return self.U.detach()


class LowRankCDST(nn.Module):
    """CDST with low-rank transition structure.
    
    w' = softmax(log(w) + U @ g(c))
    
    where U in R^{K x r} captures the r dominant conformational change directions,
    and g: R^d -> R^r maps interventions to their projection along these directions.
    
    Theoretical guarantee: If the true transition has rank r*, then LowRankCDST
    with rank >= r* achieves sample complexity O(r*d / eps^2) vs O(K*d / eps^2)
    for unstructured models.
    """
    
    def __init__(self, K: int, intervention_dim: int, rank: int = 2,
                 hidden_dim: int = 64, use_state_dependence: bool = True):
        """
        Args:
            K: number of states
            intervention_dim: intervention encoding dimension
            rank: transition rank (number of active directions)
            hidden_dim: hidden layer width
            use_state_dependence: if True, transition depends on current state w
        """
        super().__init__()
        self.K = K
        self.rank = rank
        self.use_state_dependence = use_state_dependence
        
        # Low-rank transition operator
        if use_state_dependence:
            # State-dependent: g takes (c, log_w) as input
            self.transition = LowRankTransition(
                K=K, 
                intervention_dim=intervention_dim + K,  # c + log_w
                rank=rank,
                hidden_dim=hidden_dim
            )
        else:
            # State-independent: g takes only c
            self.transition = LowRankTransition(
                K=K,
                intervention_dim=intervention_dim,
                rank=rank,
                hidden_dim=hidden_dim
            )
    
    def forward(self, w: torch.Tensor, c: torch.Tensor,
                return_delta: bool = False) -> torch.Tensor:
        """
        Args:
            w: (batch, K) current distribution
            c: (batch, intervention_dim) intervention
        Returns:
            w_pred: (batch, K) predicted post-intervention distribution
        """
        log_w = torch.log(w.clamp(min=1e-8))
        
        if self.use_state_dependence:
            # Concatenate intervention with state info
            g_input = torch.cat([c, log_w], dim=-1)
        else:
            g_input = c
        
        delta_logits = self.transition(g_input)
        
        # Boltzmann update: w' = softmax(log_w + delta)
        w_pred = F.softmax(log_w + delta_logits, dim=-1)
        
        if return_delta:
            return w_pred, delta_logits
        return w_pred
    
    def get_transition_matrix(self, c: torch.Tensor) -> torch.Tensor:
        """Get the effective K x K transition matrix for a given intervention.

        For linear analysis: w' approx T(c) @ w (in probability space)
        """
        # The Jacobian requires autograd through the forward pass, so this
        # must NOT run under torch.no_grad().
        # Compute Jacobian of output w.r.t. input w at w = uniform
        w_uniform = torch.ones(1, self.K, device=c.device) / self.K
        w_uniform.requires_grad_(True)
        w_pred = self.forward(w_uniform, c.unsqueeze(0) if c.dim() == 1 else c)
        # Jacobian: d(w_pred_k) / d(w_j)
        J = torch.zeros(self.K, self.K, device=w_pred.device)
        for k in range(self.K):
            grad = torch.autograd.grad(w_pred[0, k], w_uniform,
                                       retain_graph=(k < self.K - 1))[0]
            J[k] = grad[0].detach()
        return J


class NuclearNormRegularizer(nn.Module):
    """Nuclear norm regularization for adaptive rank selection.
    
    Instead of fixing the rank r a priori, we use a full-rank parameterization
    with nuclear norm penalty: L_reg = lambda * ||W||_*
    
    This encourages the learned transition to be low-rank without specifying
    the rank in advance. The effective rank is determined by the data.
    """
    
    def __init__(self, K: int, intervention_dim: int, hidden_dim: int = 64,
                 lambda_nuclear: float = 0.01):
        super().__init__()
        self.K = K
        self.lambda_nuclear = lambda_nuclear
        
        # Full-rank transition (K x K) with nuclear norm penalty
        self.W = nn.Parameter(torch.randn(K, K) * 0.01)
        
        # Intervention encoder
        self.encoder = nn.Sequential(
            nn.Linear(intervention_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, K),
        )
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)
    
    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Compute delta_logits with full-rank W."""
        h = self.encoder(c)  # (batch, K)
        delta_logits = h @ self.W.T  # (batch, K)
        return delta_logits
    
    def nuclear_norm(self) -> torch.Tensor:
        """Compute nuclear norm of W for regularization."""
        return torch.linalg.svdvals(self.W).sum()
    
    def effective_rank(self, threshold: float = 0.05) -> int:
        """Get effective rank based on normalized singular values."""
        with torch.no_grad():
            svs = torch.linalg.svdvals(self.W)
            svs_norm = svs / svs[0].clamp(min=1e-10)
            return int((svs_norm > threshold).sum().item())


class AdaptiveRankCDST(nn.Module):
    """CDST with automatic rank selection via nuclear norm regularization.
    
    Uses a full-rank parameterization but penalizes the nuclear norm of the
    transition matrix, encouraging low-rank solutions. The effective rank
    adapts to the complexity of the data.
    
    Training objective:
        L = L_task(pred, target) + lambda * ||W||_*
    """
    
    def __init__(self, K: int, intervention_dim: int, hidden_dim: int = 64,
                 lambda_nuclear: float = 0.01):
        super().__init__()
        self.K = K
        self.lambda_nuclear = lambda_nuclear
        
        self.regularizer = NuclearNormRegularizer(
            K=K, intervention_dim=intervention_dim,
            hidden_dim=hidden_dim, lambda_nuclear=lambda_nuclear
        )
    
    def forward(self, w: torch.Tensor, c: torch.Tensor,
                return_delta: bool = False) -> torch.Tensor:
        log_w = torch.log(w.clamp(min=1e-8))
        delta_logits = self.regularizer(c)
        w_pred = F.softmax(log_w + delta_logits, dim=-1)
        
        if return_delta:
            return w_pred, delta_logits
        return w_pred
    
    def regularization_loss(self) -> torch.Tensor:
        """Nuclear norm penalty to add to task loss."""
        return self.lambda_nuclear * self.regularizer.nuclear_norm()
    
    def effective_rank(self) -> int:
        return self.regularizer.effective_rank()


# =============================================================================
# Sparse Transition (alternative to low-rank)
# =============================================================================

class SparseTransitionCDST(nn.Module):
    """CDST with sparse transition (L1 regularization on delta_logits).
    
    Alternative to low-rank: assumes interventions affect only a few states
    directly (sparsity in state space rather than low-rank structure).
    
    Useful when K is large but interventions are local (e.g., mutation at
    one position primarily affects nearby conformational states).
    """
    
    def __init__(self, K: int, intervention_dim: int, hidden_dim: int = 64,
                 lambda_sparse: float = 0.01):
        super().__init__()
        self.K = K
        self.lambda_sparse = lambda_sparse
        
        self.net = nn.Sequential(
            nn.Linear(intervention_dim + K, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, K),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, w: torch.Tensor, c: torch.Tensor,
                return_delta: bool = False) -> torch.Tensor:
        log_w = torch.log(w.clamp(min=1e-8))
        combined = torch.cat([c, log_w], dim=-1)
        delta_logits = self.net(combined)
        w_pred = F.softmax(log_w + delta_logits, dim=-1)
        
        if return_delta:
            return w_pred, delta_logits
        return w_pred
    
    def sparsity_loss(self, delta_logits: torch.Tensor) -> torch.Tensor:
        """L1 penalty on delta_logits for sparsity."""
        return self.lambda_sparse * delta_logits.abs().mean()
