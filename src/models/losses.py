"""
Information-Geometric Loss Functions for Distributional Shift Prediction.

Implements losses that respect the Fisher-Rao Riemannian structure of the
probability simplex, enabling better few-shot generalization.

Key insight: The K-simplex is a Riemannian manifold with the Fisher-Rao metric.
Standard KL divergence is asymmetric and doesn't respect this geometry.
Fisher-Rao distance (equivalent to Hellinger in closed form) is the natural
geodesic distance on this manifold.

Reference:
    - Amari, S. (2016). Information Geometry and Its Applications.
    - Nielsen, F. (2020). On a generalization of the Jensen-Shannon divergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class FisherRaoLoss(nn.Module):
    """Fisher-Rao geodesic distance on the probability simplex.
    
    The Fisher-Rao distance between two distributions p, q on the K-simplex is:
        d_FR(p, q) = 2 * arccos(sum_k sqrt(p_k * q_k))
    
    This is the natural geodesic distance under the Fisher information metric.
    It is symmetric, bounded in [0, pi], and invariant to sufficient statistics.
    
    For training, we use the squared FR distance (smooth, differentiable):
        L_FR(p, q) = 4 * arccos^2(sum_k sqrt(p_k * q_k))
    
    Or the linearized version (Hellinger-like, faster to compute):
        L_H(p, q) = 1 - sum_k sqrt(p_k * q_k)  (Bhattacharyya coefficient)
    """
    
    def __init__(self, mode: str = 'hellinger', eps: float = 1e-8):
        """
        Args:
            mode: 'full' (arccos form), 'hellinger' (linearized), or 'squared'
            eps: numerical stability constant
        """
        super().__init__()
        self.mode = mode
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (batch, K) predicted distribution on simplex
            target: (batch, K) target distribution on simplex
        Returns:
            loss: scalar, mean FR distance over batch
        """
        # Clamp for numerical stability
        pred = pred.clamp(min=self.eps)
        target = target.clamp(min=self.eps)
        
        # Renormalize after clamping
        pred = pred / pred.sum(dim=-1, keepdim=True)
        target = target / target.sum(dim=-1, keepdim=True)
        
        # Bhattacharyya coefficient: BC = sum sqrt(p*q)
        bc = (pred.sqrt() * target.sqrt()).sum(dim=-1)
        bc = bc.clamp(min=self.eps, max=1.0 - self.eps)
        
        if self.mode == 'full':
            # Full FR distance: 2*arccos(BC)
            loss = 2.0 * torch.arccos(bc)
        elif self.mode == 'squared':
            # Squared FR distance: 4*arccos^2(BC)
            loss = 4.0 * torch.arccos(bc) ** 2
        elif self.mode == 'hellinger':
            # Hellinger-like: 1 - BC (linearized, bounded in [0,1])
            loss = 1.0 - bc
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return loss.mean()


class HellingerLoss(nn.Module):
    """Hellinger distance between distributions.
    
    H^2(p, q) = 1 - sum_k sqrt(p_k * q_k)
              = (1/2) * sum_k (sqrt(p_k) - sqrt(q_k))^2
    
    Properties:
        - Symmetric: H(p,q) = H(q,p)
        - Bounded: H in [0, 1]
        - Metric: satisfies triangle inequality
        - Equivalent to L2 distance in sqrt-space
    """
    
    def __init__(self, squared: bool = True, eps: float = 1e-8):
        super().__init__()
        self.squared = squared
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(min=self.eps)
        target = target.clamp(min=self.eps)
        pred = pred / pred.sum(dim=-1, keepdim=True)
        target = target / target.sum(dim=-1, keepdim=True)
        
        # H^2 = (1/2) * ||sqrt(p) - sqrt(q)||^2
        diff = pred.sqrt() - target.sqrt()
        h_sq = 0.5 * (diff ** 2).sum(dim=-1)
        
        if self.squared:
            return h_sq.mean()
        else:
            return h_sq.sqrt().mean()


class SymmetricKLLoss(nn.Module):
    """Symmetric KL divergence (Jensen-Shannon divergence).
    
    JSD(p, q) = (1/2) * KL(p || m) + (1/2) * KL(q || m)
    where m = (p + q) / 2
    
    Properties:
        - Symmetric
        - Bounded: JSD in [0, log 2]
        - sqrt(JSD) is a metric
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(min=self.eps)
        target = target.clamp(min=self.eps)
        pred = pred / pred.sum(dim=-1, keepdim=True)
        target = target / target.sum(dim=-1, keepdim=True)
        
        m = 0.5 * (pred + target)
        
        kl_pm = (pred * (pred.log() - m.log())).sum(dim=-1)
        kl_qm = (target * (target.log() - m.log())).sum(dim=-1)
        
        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd.mean()


class NaturalParameterLoss(nn.Module):
    """Loss in natural parameter (log) space.
    
    Key insight: In natural parameter space eta = log(w), the CDST transition
    is LINEAR: Delta_eta = eta' - eta = Delta_logits(c).
    
    This loss directly penalizes the error in natural parameter space:
        L = ||log(w_pred) - log(w_target)||^2
    
    This is equivalent to the Fisher-Rao metric locally (first-order approximation).
    """
    
    def __init__(self, normalize: bool = True, eps: float = 1e-8):
        """
        Args:
            normalize: if True, subtract mean (work with centered log-ratios)
            eps: numerical stability
        """
        super().__init__()
        self.normalize = normalize
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(min=self.eps)
        target = target.clamp(min=self.eps)
        
        # Map to natural parameter space
        eta_pred = pred.log()
        eta_target = target.log()
        
        if self.normalize:
            # Center (remove gauge freedom: softmax is invariant to constant shift)
            eta_pred = eta_pred - eta_pred.mean(dim=-1, keepdim=True)
            eta_target = eta_target - eta_target.mean(dim=-1, keepdim=True)
        
        # L2 in natural parameter space
        loss = ((eta_pred - eta_target) ** 2).sum(dim=-1)
        return loss.mean()


class AdaptiveLoss(nn.Module):
    """Adaptive loss that combines multiple geometric losses.
    
    Dynamically weights KL, FR, and natural parameter losses based on
    the current prediction quality (curriculum-style).
    
    Early training: use KL (strong gradients for bad predictions)
    Late training: use FR/natural (precise geometric refinement)
    """
    
    def __init__(self, warmup_epochs: int = 50):
        super().__init__()
        self.fr_loss = FisherRaoLoss(mode='hellinger')
        self.nat_loss = NaturalParameterLoss()
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Progress factor: 0 at start, 1 after warmup
        alpha = min(1.0, self.current_epoch / max(1, self.warmup_epochs))

        # KL loss (need log input); clamp like every other loss in this
        # module so a padded exact-zero column cannot produce -inf.
        kl = F.kl_div(pred.clamp(min=1e-8).log(), target, reduction='batchmean')

        # Geometric losses
        fr = self.fr_loss(pred, target)
        nat = self.nat_loss(pred, target)

        # Interpolate: KL -> geometric
        loss = (1 - alpha) * kl + alpha * (0.5 * fr + 0.5 * nat)
        return loss


# NOTE (2026-08-17): the former NaturalGradientOptimizer class was removed.
# Its step() claimed Fisher preconditioning but only executed a ~1.0001
# gradient rescaling (the p-weighted projection was computed and discarded),
# so any experiment labelled "natural gradient" was silently plain
# SGD/Adam. The class had no users in this repository; reintroduce it only
# together with a test that its update differs from the raw optimizer.


# =============================================================================
# Utility: Loss comparison for ablation studies
# =============================================================================

def compute_all_losses(pred: torch.Tensor, target: torch.Tensor, 
                       eps: float = 1e-8) -> dict:
    """Compute all loss variants for comparison.
    
    Returns dict with: kl, reverse_kl, jsd, hellinger, fisher_rao, natural_l2
    """
    pred = pred.clamp(min=eps)
    target = target.clamp(min=eps)
    pred = pred / pred.sum(dim=-1, keepdim=True)
    target = target / target.sum(dim=-1, keepdim=True)
    
    results = {}
    
    # Forward KL: KL(target || pred)
    results['kl_forward'] = (target * (target.log() - pred.log())).sum(-1).mean()
    
    # Reverse KL: KL(pred || target)
    results['kl_reverse'] = (pred * (pred.log() - target.log())).sum(-1).mean()
    
    # Jensen-Shannon
    m = 0.5 * (pred + target)
    results['jsd'] = 0.5 * ((pred * (pred.log() - m.log())).sum(-1) + 
                            (target * (target.log() - m.log())).sum(-1)).mean()
    
    # Hellinger
    results['hellinger'] = (1 - (pred.sqrt() * target.sqrt()).sum(-1)).mean()
    
    # Fisher-Rao (arccos form)
    bc = (pred.sqrt() * target.sqrt()).sum(-1).clamp(min=eps, max=1-eps)
    results['fisher_rao'] = (2 * torch.arccos(bc)).mean()
    
    # Natural parameter L2
    eta_pred = pred.log() - pred.log().mean(-1, keepdim=True)
    eta_target = target.log() - target.log().mean(-1, keepdim=True)
    results['natural_l2'] = ((eta_pred - eta_target)**2).sum(-1).mean()
    
    # MAE (probability space)
    results['mae'] = (pred - target).abs().sum(-1).mean()
    
    return {k: v.item() for k, v in results.items()}
