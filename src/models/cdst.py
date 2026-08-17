"""
CDST: Conditional Distributional State Transition
Core model implementation.

Architecture:
    State Encoder: w + {x_k} → h_state ∈ R^D
    Intervention Encoder: c → z_int ∈ R^D
    Transition Predictor: (h_state, z_int) → Δlogits ∈ R^K
    Output: w'_k = softmax(log w_k + Δlogits_k)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class StateEncoder(nn.Module):
    """Encodes the current state distribution w into a latent representation.
    
    For synthetic experiments, the state is simply the K-dimensional distribution w.
    For protein experiments, this would encode structural features of each state.
    """
    
    def __init__(self, K: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(K, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w: (batch, K) current state distribution (simplex)
        Returns:
            h_state: (batch, output_dim) state representation
        """
        # Use log w as input (more informative for small probabilities)
        log_w = torch.log(w + 1e-8)
        return self.net(log_w)


class InterventionEncoder(nn.Module):
    """Encodes the perturbation/intervention c into a latent representation.
    
    For synthetic experiments, c is a continuous vector describing the perturbation.
    For protein mutations, c would encode mutation type, position, etc.
    """
    
    def __init__(self, intervention_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(intervention_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c: (batch, intervention_dim) perturbation description
        Returns:
            z_int: (batch, output_dim) intervention representation
        """
        return self.net(c)


class TransitionPredictor(nn.Module):
    """Predicts Δlogits from state and intervention representations.
    
    This is the core of CDST: predicts the free energy change vector ΔG(c)/kT
    which directly gives the logit shift.
    """
    
    def __init__(self, state_dim: int, intervention_dim: int, K: int, hidden_dim: int = 128):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(state_dim + intervention_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, K),
        )
        # Initialize last layer near zero (zero perturbation = identity)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, h_state: torch.Tensor, z_int: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_state: (batch, state_dim)
            z_int: (batch, intervention_dim)
        Returns:
            delta_logits: (batch, K) predicted logit shifts
        """
        combined = torch.cat([h_state, z_int], dim=-1)
        return self.net(combined)


class CDST(nn.Module):
    """Conditional Distributional State Transition model.
    
    Core architecture:
        w' = softmax(log w + Δlogits(c))
    
    where Δlogits(c) is predicted by a neural network from the current state
    distribution w and the perturbation description c.
    
    Key properties:
        - Zero perturbation → identity (architectural guarantee)
        - Boltzmann structure (softmax = Boltzmann distribution)
        - Compositional: Δlogits(c1+c2) ≈ Δlogits(c1) + Δlogits(c2) for independent perturbations
    """
    
    def __init__(
        self,
        K: int,
        intervention_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        use_thermo_consistency: bool = True,
    ):
        super().__init__()
        self.K = K
        self.intervention_dim = intervention_dim
        self.use_thermo_consistency = use_thermo_consistency
        
        self.state_encoder = StateEncoder(K, hidden_dim, latent_dim)
        self.intervention_encoder = InterventionEncoder(intervention_dim, hidden_dim, latent_dim)
        self.transition_predictor = TransitionPredictor(latent_dim, latent_dim, K, hidden_dim)
    
    def forward(
        self, 
        w: torch.Tensor, 
        c: torch.Tensor,
        return_delta_logits: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            w: (batch, K) current equilibrium distribution
            c: (batch, intervention_dim) perturbation description
            return_delta_logits: if True, also return Δlogits
        Returns:
            w_pred: (batch, K) predicted post-perturbation distribution
            [delta_logits: (batch, K)] optional
        """
        h_state = self.state_encoder(w)
        z_int = self.intervention_encoder(c)
        delta_logits = self.transition_predictor(h_state, z_int)
        
        # Boltzmann softmax: w' = softmax(log w + Δlogits)
        log_w = torch.log(w + 1e-8)
        w_pred = F.softmax(log_w + delta_logits, dim=-1)
        
        if return_delta_logits:
            return w_pred, delta_logits
        return w_pred
    
    def predict_delta_G(self, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Predict the free energy change vector ΔG(c)/kT."""
        _, delta_logits = self.forward(w, c, return_delta_logits=True)
        return delta_logits


class CDSTLoss(nn.Module):
    """Training loss for CDST.

    L = KL(w_true || w_pred) + λ₁·L_sigreg + λ₂·L_thermo + λ₃·L_recon

    (F.kl_div(log w_pred, w_true) computes KL of the target against the
    prediction, i.e. KL(w_true ‖ w_pred).)
    """
    
    def __init__(
        self,
        lambda_sigreg: float = 0.01,
        lambda_thermo: float = 0.1,
        lambda_recon: float = 0.0,
    ):
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
        self.lambda_thermo = lambda_thermo
        self.lambda_recon = lambda_recon
    
    def kl_divergence(self, w_pred: torch.Tensor, w_true: torch.Tensor) -> torch.Tensor:
        """KL(w_true || w_pred) - measures prediction accuracy."""
        return F.kl_div(
            torch.log(w_pred + 1e-8), 
            w_true, 
            reduction='batchmean'
        )
    
    def sigreg_loss(self, delta_logits: torch.Tensor) -> torch.Tensor:
        """Sigmoid regularization: prevent Δlogits from being too large."""
        return torch.mean(torch.sigmoid(delta_logits.abs() - 5.0))
    
    def thermo_consistency_loss(
        self, 
        model: CDST,
        w: torch.Tensor, 
        c1: torch.Tensor, 
        c2: torch.Tensor,
    ) -> torch.Tensor:
        """Thermodynamic cycle consistency.
        
        For independent perturbations: Δlogits(c1+c2) ≈ Δlogits(c1) + Δlogits(c2)
        """
        if not model.use_thermo_consistency:
            return torch.tensor(0.0, device=w.device)
        
        # Get individual Δlogits
        _, dl1 = model(w, c1, return_delta_logits=True)
        _, dl2 = model(w, c2, return_delta_logits=True)
        
        # Get combined Δlogits
        c_combined = c1 + c2
        _, dl_combined = model(w, c_combined, return_delta_logits=True)
        
        # Cycle consistency: dl(c1+c2) ≈ dl(c1) + dl(c2)
        return F.mse_loss(dl_combined, dl1 + dl2)
    
    def forward(
        self,
        model: CDST,
        w: torch.Tensor,
        c: torch.Tensor,
        w_true: torch.Tensor,
        c1: Optional[torch.Tensor] = None,
        c2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: scalar
            loss_dict: breakdown of losses
        """
        w_pred, delta_logits = model(w, c, return_delta_logits=True)
        
        # Main loss: KL divergence
        kl_loss = self.kl_divergence(w_pred, w_true)
        
        # Regularization
        sigreg = self.sigreg_loss(delta_logits)
        
        # Thermodynamic consistency (if pairs available)
        thermo_loss = torch.tensor(0.0, device=w.device)
        if c1 is not None and c2 is not None:
            thermo_loss = self.thermo_consistency_loss(model, w, c1, c2)
        
        total = kl_loss + self.lambda_sigreg * sigreg + self.lambda_thermo * thermo_loss
        
        loss_dict = {
            'total': total.item(),
            'kl': kl_loss.item(),
            'sigreg': sigreg.item(),
            'thermo': thermo_loss.item(),
        }
        
        return total, loss_dict
