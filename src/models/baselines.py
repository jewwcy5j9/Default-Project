"""
Baseline models for comparison with CDST.

Baselines:
    1. cFlow: Conditional Flow Matching (conditional generative SOTA)
    2. cVAE: Conditional Variational Autoencoder
    3. LRT: Linear Response Theory (analytical)
    4. DirectMLP: Direct prediction without Boltzmann structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# =============================================================================
# Baseline 1: Conditional Flow Matching (simplified for distributions)
# =============================================================================

class ConditionalFlowMatching(nn.Module):
    """Conditional Flow Matching for distribution prediction.
    
    Learns a velocity field v(x, t, c) that transports samples from 
    a simple prior to the target conditional distribution.
    
    For K-dimensional simplex, we work in log-space and project back.
    """
    
    def __init__(self, K: int, intervention_dim: int, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        self.K = K
        self.intervention_dim = intervention_dim
        
        # Velocity field: (x, t, c) → v
        layers = []
        input_dim = K + 1 + intervention_dim  # x + t + c
        for i in range(n_layers):
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
        layers.append(nn.Linear(hidden_dim, K))
        self.velocity_net = nn.Sequential(*layers)
    
    def velocity(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Compute velocity field at (x, t) conditioned on c."""
        t_expanded = t.unsqueeze(-1) if t.dim() == 1 else t
        inp = torch.cat([x, t_expanded, c], dim=-1)
        return self.velocity_net(inp)
    
    def forward(self, w: torch.Tensor, c: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        """Generate prediction by integrating the flow.
        
        Args:
            w: (batch, K) starting distribution
            c: (batch, intervention_dim) condition
            n_steps: number of Euler integration steps
        Returns:
            w_pred: (batch, K) predicted distribution
        """
        # Start from log-space representation of w
        x = torch.log(w + 1e-8)
        dt = 1.0 / n_steps
        
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            v = self.velocity(x, t, c)
            x = x + v * dt
        
        # Project back to simplex
        w_pred = F.softmax(x, dim=-1)
        return w_pred
    
    def compute_loss(self, w: torch.Tensor, c: torch.Tensor, w_target: torch.Tensor) -> torch.Tensor:
        """Flow matching loss: ||v(x_t, t, c) - (x_1 - x_0)||^2
        
        Using optimal transport conditional flow matching.
        """
        batch_size = w.shape[0]
        
        # Source and target in log-space
        x_0 = torch.log(w + 1e-8)
        x_1 = torch.log(w_target + 1e-8)
        
        # Sample random time
        t = torch.rand(batch_size, device=w.device)
        
        # Interpolate: x_t = (1-t)*x_0 + t*x_1
        t_exp = t.unsqueeze(-1)
        x_t = (1 - t_exp) * x_0 + t_exp * x_1
        
        # Target velocity (optimal transport)
        u_t = x_1 - x_0
        
        # Predicted velocity
        v_pred = self.velocity(x_t, t, c)
        
        # MSE loss
        return F.mse_loss(v_pred, u_t)


# =============================================================================
# Baseline 2: Conditional VAE
# =============================================================================

class ConditionalVAE(nn.Module):
    """Conditional VAE for distribution prediction.
    
    Encoder: (w, w', c) → z (during training)
    Decoder: (w, z, c) → w' (during inference, z ~ N(0,I))
    """
    
    def __init__(self, K: int, intervention_dim: int, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.K = K
        self.latent_dim = latent_dim
        
        # Encoder: (w, w', c) → μ, logvar
        enc_input = 2 * K + intervention_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: (w, z, c) → w'
        dec_input = K + latent_dim + intervention_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, K),
        )
    
    def encode(self, w: torch.Tensor, w_target: torch.Tensor, c: torch.Tensor):
        inp = torch.cat([w, w_target, c], dim=-1)
        h = self.encoder(inp)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, w: torch.Tensor, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([w, z, c], dim=-1)
        logits = self.decoder(inp)
        return F.softmax(logits, dim=-1)
    
    def forward(self, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Inference: sample z from prior and decode."""
        z = torch.randn(w.shape[0], self.latent_dim, device=w.device)
        return self.decode(w, z, c)
    
    def compute_loss(self, w: torch.Tensor, c: torch.Tensor, w_target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """VAE ELBO loss."""
        mu, logvar = self.encode(w, w_target, c)
        z = self.reparameterize(mu, logvar)
        w_pred = self.decode(w, z, c)
        
        # Reconstruction loss (KL between predicted and true)
        recon_loss = F.kl_div(torch.log(w_pred + 1e-8), w_target, reduction='batchmean')
        
        # KL regularization
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        total = recon_loss + 0.01 * kl_loss
        return total, {'recon': recon_loss.item(), 'kl_reg': kl_loss.item()}


# =============================================================================
# Baseline 3: Linear Response Theory (LRT)
# =============================================================================

class LinearResponseTheory:
    """Linear Response Theory baseline.
    
    For small perturbations: w' ≈ w + J·c
    where J is the Jacobian (susceptibility matrix).
    
    This is the analytical limit of CDST when Δlogits → 0.
    """
    
    def __init__(self, K: int, intervention_dim: int):
        self.K = K
        self.intervention_dim = intervention_dim
        self.J = None  # Susceptibility matrix (K × intervention_dim)
    
    def fit(self, w_data: np.ndarray, c_data: np.ndarray, w_target_data: np.ndarray):
        """Fit the linear response matrix J from data.
        
        w' - w ≈ J · c
        Solve via least squares.
        """
        # Δw = w' - w
        delta_w = w_target_data - w_data  # (N, K)
        
        # Solve: delta_w = c @ J^T → J^T = (c^T c)^{-1} c^T delta_w
        # Using pseudo-inverse for stability
        self.J = np.linalg.lstsq(c_data, delta_w, rcond=None)[0]  # (intervention_dim, K)
    
    def predict(self, w: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Predict w' using linear response."""
        delta_w = c @ self.J  # (batch, K)
        w_pred = w + delta_w
        
        # Project back to simplex (clip + normalize)
        w_pred = np.clip(w_pred, 1e-8, None)
        w_pred = w_pred / w_pred.sum(axis=-1, keepdims=True)
        return w_pred


# =============================================================================
# Baseline 4: Direct MLP (no Boltzmann structure)
# =============================================================================

class DirectMLP(nn.Module):
    """Direct MLP prediction without Boltzmann softmax structure.
    
    Ablation baseline: shows the importance of the Boltzmann architecture.
    Simply predicts w' directly from (w, c).
    """
    
    def __init__(self, K: int, intervention_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(K + intervention_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, K),
        )
    
    def forward(self, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Direct prediction: (w, c) → w'"""
        inp = torch.cat([w, c], dim=-1)
        logits = self.net(inp)
        return F.softmax(logits, dim=-1)
    
    def compute_loss(self, w: torch.Tensor, c: torch.Tensor, w_target: torch.Tensor) -> torch.Tensor:
        w_pred = self.forward(w, c)
        return F.kl_div(torch.log(w_pred + 1e-8), w_target, reduction='batchmean')
