"""
Protein CDST Model

CDST architecture adapted for protein mutation perturbation prediction.

Architecture:
    State Encoder: w (+ structural features) → h_state
    Mutation Encoder: mutation → z_mut
    Transition Predictor: (h_state, z_mut) → Δlogits ∈ R^K
    Output: w' = softmax(log w + Δlogits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .state_encoder import create_state_encoder, PopulationStateEncoder
from .mutation_encoder import MutationEncoder


class ProteinCDST(nn.Module):
    """CDST model for protein mutation perturbation prediction.
    
    Key features:
    - Boltzmann softmax architecture (w' = softmax(log w + Δlogits))
    - Learnable mutation encoder
    - Optional structural state features
    - Thermodynamic consistency regularization
    """
    
    def __init__(
        self,
        K: int,
        mutation_input_dim: int = 6,
        state_feature_dim: int = 0,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        state_encoder_type: str = "population",
        use_thermo_consistency: bool = True,
    ):
        """
        Args:
            K: Number of conformational states
            mutation_input_dim: Dimension of mutation encoding
            state_feature_dim: Dimension of structural features per state (0 for population-only)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent representation dimension
            state_encoder_type: "population", "structural", or "hybrid"
            use_thermo_consistency: Whether to use thermodynamic cycle regularization
        """
        super().__init__()
        self.K = K
        self.use_thermo_consistency = use_thermo_consistency
        
        # State encoder
        self.state_encoder = create_state_encoder(
            K=K,
            encoder_type=state_encoder_type,
            feature_dim=state_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
        )
        
        # Mutation encoder
        self.mutation_encoder = MutationEncoder(
            input_dim=mutation_input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
        )
        
        # Transition predictor
        self.transition_predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, K),
        )
        
        # Initialize last layer near zero (zero perturbation = identity)
        nn.init.zeros_(self.transition_predictor[-1].weight)
        nn.init.zeros_(self.transition_predictor[-1].bias)
    
    def forward(
        self,
        w: torch.Tensor,
        c: torch.Tensor,
        state_features: Optional[torch.Tensor] = None,
        return_delta_logits: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            w: (batch, K) current state distribution
            c: (batch, mutation_dim) mutation encoding
            state_features: (batch, K, feature_dim) optional structural features
            return_delta_logits: Whether to return Δlogits
        
        Returns:
            w_pred: (batch, K) predicted post-mutation distribution
            [delta_logits: (batch, K)] optional
        """
        # Encode state
        if isinstance(self.state_encoder, PopulationStateEncoder):
            h_state = self.state_encoder(w)
        else:
            h_state = self.state_encoder(w, state_features)
        
        # Encode mutation
        z_mut = self.mutation_encoder(c)
        
        # Predict Δlogits
        combined = torch.cat([h_state, z_mut], dim=-1)
        delta_logits = self.transition_predictor(combined)
        
        # Boltzmann softmax
        log_w = torch.log(w + 1e-8)
        w_pred = F.softmax(log_w + delta_logits, dim=-1)
        
        if return_delta_logits:
            return w_pred, delta_logits
        return w_pred
    
    def predict_delta_G(self, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Predict free energy change vector ΔG/kT."""
        _, delta_logits = self.forward(w, c, return_delta_logits=True)
        return delta_logits


class ProteinCDSTLoss(nn.Module):
    """Training loss for Protein CDST.

    L = KL(w_true || w_pred) + λ1*L_sigreg

    Note: despite the constructor argument, no thermodynamic-consistency
    term is implemented in this loss (unlike CDSTLoss); lambda_thermo is
    stored but unused. (F.kl_div(log w_pred, w_true) computes
    KL(w_true ‖ w_pred).)
    """
    
    def __init__(
        self,
        lambda_sigreg: float = 0.01,
        lambda_thermo: float = 0.1,
    ):
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
        self.lambda_thermo = lambda_thermo
    
    def forward(
        self,
        model: ProteinCDST,
        w: torch.Tensor,
        c: torch.Tensor,
        w_true: torch.Tensor,
        state_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss: scalar
            loss_dict: breakdown
        """
        w_pred, delta_logits = model(w, c, state_features, return_delta_logits=True)
        
        # Main loss: KL divergence
        kl_loss = F.kl_div(torch.log(w_pred + 1e-8), w_true, reduction='batchmean')
        
        # Sigmoid regularization (prevent extreme Δlogits)
        sigreg = torch.mean(torch.sigmoid(delta_logits.abs() - 5.0))
        
        total = kl_loss + self.lambda_sigreg * sigreg
        
        loss_dict = {
            'total': total.item(),
            'kl': kl_loss.item(),
            'sigreg': sigreg.item(),
        }
        
        return total, loss_dict


class ProteinBaselines:
    """Baseline methods for protein mutation perturbation prediction."""
    
    @staticmethod
    def ddg_boltzmann_reweight(
        w: torch.Tensor,
        ddg: torch.Tensor,
        kT: float = 0.6,
    ) -> torch.Tensor:
        """ΔΔG-based Boltzmann reweighting baseline.
        
        Assumes mutation affects all states equally except one.
        w'_k ∝ w_k * exp(-ΔΔG_k / kT)
        
        Args:
            w: (batch, K) initial distribution
            ddg: (batch,) predicted ΔΔG (scalar per mutation)
            kT: thermal energy
        
        Returns:
            w_pred: (batch, K) predicted distribution
        """
        # Simple model: ΔΔG affects state 0 (ground state) only
        batch_size, K = w.shape
        delta_G = torch.zeros(batch_size, K, device=w.device)
        delta_G[:, 0] = ddg  # Destabilize ground state
        
        log_w = torch.log(w + 1e-8)
        w_pred = F.softmax(log_w - delta_G / kT, dim=-1)
        return w_pred
    
    @staticmethod
    def linear_response(
        w: torch.Tensor,
        c: torch.Tensor,
        J: torch.Tensor,
    ) -> torch.Tensor:
        """Linear response theory baseline.
        
        w' = w + J @ c (projected to simplex)
        
        Args:
            w: (batch, K) initial distribution
            c: (batch, p) mutation encoding
            J: (K, p) susceptibility matrix (learned)
        
        Returns:
            w_pred: (batch, K) predicted distribution
        """
        delta_w = c @ J.T  # (batch, K)
        w_pred = w + delta_w
        w_pred = torch.clamp(w_pred, min=1e-8)
        w_pred = w_pred / w_pred.sum(dim=-1, keepdim=True)
        return w_pred


def create_protein_cdst(
    K: int,
    mutation_encoder_type: str = "properties",
    state_encoder_type: str = "population",
    state_feature_dim: int = 0,
    hidden_dim: int = 128,
    latent_dim: int = 64,
) -> ProteinCDST:
    """Factory function to create Protein CDST model.
    
    Args:
        K: Number of states
        mutation_encoder_type: "properties" (6-dim) or "onehot" (41-dim)
        state_encoder_type: "population" or "hybrid"
        state_feature_dim: Structural feature dimension (for hybrid)
        hidden_dim: Hidden dimension
        latent_dim: Latent dimension
    
    Returns:
        model: ProteinCDST instance
    """
    # Determine mutation input dimension
    if mutation_encoder_type == "properties":
        mutation_input_dim = 6
    elif mutation_encoder_type == "onehot":
        mutation_input_dim = 41
    else:
        mutation_input_dim = 6
    
    return ProteinCDST(
        K=K,
        mutation_input_dim=mutation_input_dim,
        state_feature_dim=state_feature_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        state_encoder_type=state_encoder_type,
    )
