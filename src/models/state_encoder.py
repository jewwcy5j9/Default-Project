"""
Protein State Encoder

Encode protein conformational states for CDST.

Options:
1. Population-only encoder (simple, for when only populations known)
2. Structural feature encoder (RMSD, contacts, dihedrals)
3. Hybrid encoder (populations + structural features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class PopulationStateEncoder(nn.Module):
    """Encode state distribution only (lightweight).
    
    Used when only population data is available (e.g., from NMR).
    Input: w ∈ Δ^K (simplex)
    """
    
    def __init__(self, K: int, hidden_dim: int = 64, output_dim: int = 32):
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
            w: (batch, K) state distribution
        Returns:
            h: (batch, output_dim) state representation
        """
        # Use log for better gradient flow with small probabilities
        log_w = torch.log(w + 1e-8)
        return self.net(log_w)


class StructuralStateEncoder(nn.Module):
    """Encode structural features of conformational ensemble.
    
    Input: Structural features per state (RMSD, contacts, etc.)
    """
    
    def __init__(
        self,
        K: int,
        feature_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ):
        """
        Args:
            K: Number of states
            feature_dim: Dimension of structural features per state
        """
        super().__init__()
        self.K = K
        self.feature_dim = feature_dim
        
        # Per-state feature encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        # Attention-weighted aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 1, 1),  # +1 for population weight
            nn.Softmax(dim=1),
        )
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(
        self,
        w: torch.Tensor,
        state_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            w: (batch, K) state populations
            state_features: (batch, K, feature_dim) structural features per state
        Returns:
            h: (batch, output_dim) ensemble representation
        """
        batch_size = w.shape[0]
        
        # Encode each state's features
        state_encoded = self.state_encoder(state_features)  # (batch, K, hidden//2)
        
        # Attention weights (population-informed)
        w_expanded = w.unsqueeze(-1)  # (batch, K, 1)
        attn_input = torch.cat([state_encoded, w_expanded], dim=-1)  # (batch, K, hidden//2+1)
        attn_weights = self.attention(attn_input)  # (batch, K, 1)
        
        # Weighted sum
        ensemble_repr = (state_encoded * attn_weights).sum(dim=1)  # (batch, hidden//2)
        
        return self.output_proj(ensemble_repr)


class HybridStateEncoder(nn.Module):
    """Hybrid encoder combining populations and structural features.
    
    Uses both w (populations) and structural features for richer representation.
    """
    
    def __init__(
        self,
        K: int,
        feature_dim: int = 0,
        hidden_dim: int = 128,
        output_dim: int = 64,
        use_structural: bool = True,
    ):
        super().__init__()
        self.K = K
        self.use_structural = use_structural and feature_dim > 0
        
        # Population encoder
        self.pop_encoder = nn.Sequential(
            nn.Linear(K, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        if self.use_structural:
            # Structural encoder
            self.struct_encoder = nn.Sequential(
                nn.Linear(K * feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
            )
            fusion_dim = hidden_dim
        else:
            fusion_dim = hidden_dim // 2
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(
        self,
        w: torch.Tensor,
        state_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            w: (batch, K) state populations
            state_features: (batch, K, feature_dim) optional structural features
        Returns:
            h: (batch, output_dim) state representation
        """
        # Population encoding
        log_w = torch.log(w + 1e-8)
        pop_encoded = self.pop_encoder(log_w)
        
        if self.use_structural and state_features is not None:
            # Flatten structural features
            batch_size = state_features.shape[0]
            struct_flat = state_features.reshape(batch_size, -1)
            struct_encoded = self.struct_encoder(struct_flat)
            combined = torch.cat([pop_encoded, struct_encoded], dim=-1)
        else:
            combined = pop_encoded
        
        return self.fusion(combined)


def compute_structural_features(
    coordinates: np.ndarray,
    reference_coords: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute structural features from coordinates.
    
    Args:
        coordinates: (K, n_atoms, 3) coordinates per state
        reference_coords: (n_atoms, 3) reference for RMSD calculation
    
    Returns:
        features: (K, feature_dim) structural features
    """
    K, n_atoms, _ = coordinates.shape
    
    if reference_coords is None:
        reference_coords = coordinates.mean(axis=0)
    
    features = []
    
    for k in range(K):
        coords = coordinates[k]
        
        # RMSD to reference
        rmsd = np.sqrt(((coords - reference_coords) ** 2).sum(axis=1).mean())
        
        # Radius of gyration
        center = coords.mean(axis=0)
        rg = np.sqrt(((coords - center) ** 2).sum(axis=1).mean())
        
        # Contact count (atom pairs within 8 A); count ordered pairs above
        # the diagonal only — the full matrix includes the zero-distance
        # diagonal, which inflated counts by K/2 self-contacts.
        if n_atoms > 1:
            dist_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1))
            contacts = np.triu(dist_matrix < 8.0, k=1).sum()
        else:
            contacts = 0
        
        features.append([rmsd, rg, contacts])
    
    return np.array(features)


def create_state_encoder(
    K: int,
    encoder_type: str = "population",
    feature_dim: int = 0,
    hidden_dim: int = 128,
    output_dim: int = 64,
) -> nn.Module:
    """Factory function to create state encoder.
    
    Args:
        K: Number of states
        encoder_type: "population", "structural", or "hybrid"
        feature_dim: Dimension of structural features (for structural/hybrid)
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
    
    Returns:
        encoder: State encoder module
    """
    if encoder_type == "population":
        return PopulationStateEncoder(K, hidden_dim, output_dim)
    elif encoder_type == "structural":
        return StructuralStateEncoder(K, feature_dim, hidden_dim, output_dim)
    elif encoder_type == "hybrid":
        return HybridStateEncoder(K, feature_dim, hidden_dim, output_dim, use_structural=True)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
