"""
Protein Mutation Encoders

Encode mutations into vectors for CDST input.

Options:
1. Amino acid properties (5-dim, lightweight)
2. ESM-2 embeddings (1280-dim, requires fair-esm)
3. One-hot + properties (hybrid)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List


# Amino acid properties (Kyte-Doolittle hydrophobicity, Zamyatnin volume, etc.)
AA_PROPERTIES = {
    'A': {'hydro': 1.8, 'volume': 88.6, 'charge': 0, 'polarity': 0.0, 'aromatic': 0},
    'R': {'hydro': -4.5, 'volume': 173.4, 'charge': 1, 'polarity': 1.0, 'aromatic': 0},
    'N': {'hydro': -3.5, 'volume': 114.1, 'charge': 0, 'polarity': 1.0, 'aromatic': 0},
    'D': {'hydro': -3.5, 'volume': 111.1, 'charge': -1, 'polarity': 1.0, 'aromatic': 0},
    'C': {'hydro': 2.5, 'volume': 108.5, 'charge': 0, 'polarity': 0.0, 'aromatic': 0},
    'E': {'hydro': -3.5, 'volume': 138.4, 'charge': -1, 'polarity': 1.0, 'aromatic': 0},
    'Q': {'hydro': -3.5, 'volume': 143.8, 'charge': 0, 'polarity': 1.0, 'aromatic': 0},
    'G': {'hydro': -0.4, 'volume': 60.1, 'charge': 0, 'polarity': 0.0, 'aromatic': 0},
    'H': {'hydro': -3.2, 'volume': 153.2, 'charge': 0.5, 'polarity': 1.0, 'aromatic': 1},
    'I': {'hydro': 4.5, 'volume': 166.7, 'charge': 0, 'polarity': 0.0, 'aromatic': 0},
    'L': {'hydro': 3.8, 'volume': 166.7, 'charge': 0, 'polarity': 0.0, 'aromatic': 0},
    'K': {'hydro': -3.9, 'volume': 168.6, 'charge': 1, 'polarity': 1.0, 'aromatic': 0},
    'M': {'hydro': 1.9, 'volume': 162.9, 'charge': 0, 'polarity': 0.0, 'aromatic': 0},
    'F': {'hydro': 2.8, 'volume': 189.9, 'charge': 0, 'polarity': 0.0, 'aromatic': 1},
    'P': {'hydro': -1.6, 'volume': 112.7, 'charge': 0, 'polarity': 0.0, 'aromatic': 0},
    'S': {'hydro': -0.8, 'volume': 89.0, 'charge': 0, 'polarity': 1.0, 'aromatic': 0},
    'T': {'hydro': -0.7, 'volume': 116.1, 'charge': 0, 'polarity': 1.0, 'aromatic': 0},
    'W': {'hydro': -0.9, 'volume': 227.8, 'charge': 0, 'polarity': 0.0, 'aromatic': 1},
    'Y': {'hydro': -1.3, 'volume': 193.6, 'charge': 0, 'polarity': 1.0, 'aromatic': 1},
    'V': {'hydro': 4.2, 'volume': 140.0, 'charge': 0, 'polarity': 0.0, 'aromatic': 0},
}

AA_LIST = list(AA_PROPERTIES.keys())
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


class AminoAcidPropertyEncoder:
    """Encode mutations using amino acid physicochemical properties.
    
    Output: 6-dimensional vector
    - Δhydrophobicity (Kyte-Doolittle)
    - Δvolume (Zamyatnin)
    - Δcharge
    - Δpolarity
    - Δaromaticity
    - Normalized position
    """
    
    def __init__(self, seq_length: int = 300):
        self.seq_length = seq_length
        self.output_dim = 6
    
    def encode(self, mutation: str, position: int) -> np.ndarray:
        """Encode a single mutation.
        
        Args:
            mutation: Mutation string like "T315I"
            position: Mutation position (1-indexed)
        
        Returns:
            c: (6,) encoding vector
        """
        # Parse mutation
        wt_aa = mutation[0]
        mut_aa = mutation[-1]
        
        wt_props = AA_PROPERTIES.get(wt_aa, {'hydro': 0, 'volume': 0, 'charge': 0, 'polarity': 0, 'aromatic': 0})
        mut_props = AA_PROPERTIES.get(mut_aa, {'hydro': 0, 'volume': 0, 'charge': 0, 'polarity': 0, 'aromatic': 0})
        
        c = np.array([
            (mut_props['hydro'] - wt_props['hydro']) / 5.0,  # Normalized Δhydrophobicity
            (mut_props['volume'] - wt_props['volume']) / 100.0,  # Normalized Δvolume
            mut_props['charge'] - wt_props['charge'],  # Δcharge
            mut_props['polarity'] - wt_props['polarity'],  # Δpolarity
            mut_props['aromatic'] - wt_props['aromatic'],  # Δaromaticity
            position / self.seq_length,  # Normalized position
        ])
        
        return c
    
    def encode_batch(self, mutations: List[str], positions: List[int]) -> np.ndarray:
        """Encode a batch of mutations."""
        return np.array([self.encode(m, p) for m, p in zip(mutations, positions)])


class OneHotMutationEncoder:
    """Encode mutations using one-hot representation.
    
    Output: 20 (WT) + 20 (Mut) + 1 (position) = 41 dimensions
    """
    
    def __init__(self, seq_length: int = 300):
        self.seq_length = seq_length
        self.output_dim = 41  # 20 + 20 + 1
    
    def encode(self, mutation: str, position: int) -> np.ndarray:
        """Encode a single mutation."""
        wt_aa = mutation[0]
        mut_aa = mutation[-1]
        
        wt_onehot = np.zeros(20)
        mut_onehot = np.zeros(20)
        
        if wt_aa in AA_TO_IDX:
            wt_onehot[AA_TO_IDX[wt_aa]] = 1
        if mut_aa in AA_TO_IDX:
            mut_onehot[AA_TO_IDX[mut_aa]] = 1
        
        pos_normalized = position / self.seq_length
        
        return np.concatenate([wt_onehot, mut_onehot, [pos_normalized]])
    
    def encode_batch(self, mutations: List[str], positions: List[int]) -> np.ndarray:
        """Encode a batch of mutations."""
        return np.array([self.encode(m, p) for m, p in zip(mutations, positions)])


class ESM2MutationEncoder(nn.Module):
    """Encode mutations using ESM-2 embeddings (if available).
    
    Output: 1280-dimensional embedding difference
    Requires: fair-esm package
    
    Falls back to property encoder if ESM-2 not available.
    """
    
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model_name = model_name
        # Representation width per ESM-2 checkpoint family; the constructor
        # previously ignored model_name and always loaded the 650M model.
        width = {
            "esm2_t6_8M_UR50D": 320,
            "esm2_t12_35M_UR50D": 480,
            "esm2_t30_150M_UR50D": 640,
            "esm2_t33_650M_UR50D": 1280,
            "esm2_t36_3B_UR50D": 2560,
        }
        if model_name not in width:
            raise ValueError(
                f"unsupported ESM-2 checkpoint {model_name!r}; "
                f"choose from {sorted(width)}")
        self.output_dim = width[model_name]
        self.esm_available = False

        try:
            import esm
            loader = getattr(esm.pretrained, model_name)
            self.model, self.alphabet = loader()
            self.batch_converter = self.alphabet.get_batch_converter()
            self.model = self.model.to(device)
            self.model.eval()
            self.esm_available = True
            print(f"ESM-2 ({model_name}) loaded successfully")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            # ImportError: fair-esm missing; the others cover weight
            # download/load failures (offline reviewer machines included).
            print(f"ESM-2 unavailable ({type(exc).__name__}: {exc}); "
                  "falling back to property encoder")
            self.fallback_encoder = AminoAcidPropertyEncoder()
            self.output_dim = 6
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Get ESM-2 embedding for a sequence."""
        if not self.esm_available:
            raise RuntimeError("ESM-2 not available")
        
        data = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33])
        
        # Mean pooling over sequence length
        embeddings = results["representations"][33][0, 1:len(sequence)+1]
        return embeddings.mean(dim=0)
    
    def encode_mutation(self, wt_seq: str, mut_seq: str) -> torch.Tensor:
        """Encode mutation as embedding difference."""
        if not self.esm_available:
            # Fallback: extract mutation info and use property encoder
            # Find mutation position
            for i, (wt, mut) in enumerate(zip(wt_seq, mut_seq)):
                if wt != mut:
                    mutation = f"{wt}{i+1}{mut}"
                    return torch.FloatTensor(self.fallback_encoder.encode(mutation, i+1))
            return torch.zeros(6)
        
        wt_embed = self.encode_sequence(wt_seq)
        mut_embed = self.encode_sequence(mut_seq)
        return mut_embed - wt_embed


class MutationEncoder(nn.Module):
    """Unified mutation encoder with learnable projection.
    
    Takes raw mutation encoding and projects to latent space.
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        output_dim: int = 32,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
            c: (batch, input_dim) raw mutation encoding
        Returns:
            z: (batch, output_dim) latent mutation representation
        """
        return self.net(c)


def create_mutation_encoder(
    encoder_type: str = "properties",
    seq_length: int = 300,
    **kwargs
) -> Tuple[object, int]:
    """Factory function to create mutation encoder.
    
    Args:
        encoder_type: "properties", "onehot", or "esm2"
        seq_length: Protein sequence length
    
    Returns:
        encoder: Encoder instance
        output_dim: Dimension of encoding
    """
    if encoder_type == "properties":
        encoder = AminoAcidPropertyEncoder(seq_length)
        return encoder, encoder.output_dim
    elif encoder_type == "onehot":
        encoder = OneHotMutationEncoder(seq_length)
        return encoder, encoder.output_dim
    elif encoder_type == "esm2":
        encoder = ESM2MutationEncoder(**kwargs)
        return encoder, encoder.output_dim
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
