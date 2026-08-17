"""
Synthetic Data Generator for ICLR Experiments.

Generates distributional shift data with controlled properties:
- K: number of states (simplex dimension)
- rank: effective rank of the transition operator
- noise: observation noise level
- compositional: whether interventions compose linearly

Ground truth: w' = softmax(log(w) + U @ g(c))
where U in R^{K x rank}, g: R^d -> R^rank
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    K: int = 10              # Number of states
    d: int = 5               # Intervention dimension
    rank: int = 2            # True rank of transition
    n_train: int = 50        # Training samples
    n_test: int = 200        # Test samples
    noise: float = 0.0       # Observation noise (std in log-space)
    seed: int = 42
    compositional: bool = False  # If True, interventions are additive
    base_concentration: float = 1.0  # Dirichlet concentration for base w


class SyntheticShiftGenerator:
    """Generates synthetic distributional shift data with known ground truth.
    
    The generative model:
        1. Sample base distribution: w ~ Dirichlet(alpha)
        2. Sample intervention: c ~ N(0, I_d)
        3. Compute shift: Delta = U @ g(c)  where U in R^{K x r}, g is nonlinear
        4. Apply shift: w' = softmax(log(w) + Delta + noise)
    
    This matches the CDST model class exactly when g is linear.
    """
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        
        # Generate ground truth parameters
        self._generate_ground_truth()
    
    def _generate_ground_truth(self):
        """Generate the true transition operator."""
        K, d, r = self.config.K, self.config.d, self.config.rank

        if r > K:
            raise ValueError(
                f"rank={r} exceeds K={K}; QR of a (K, rank) factor would "
                "silently truncate and yield a full-rank ground truth "
                "instead of the requested low-rank one")

        # U: state-space directions (K x r), orthonormal columns
        U_raw = self.rng.standard_normal((K, r))
        U, _ = np.linalg.qr(U_raw)
        self.U_true = U[:, :r]  # (K, r)
        
        # g: intervention -> reduced space
        # Use a random linear map + nonlinearity for realism
        self.W_g = self.rng.standard_normal((r, d)) * 0.5
        self.b_g = self.rng.standard_normal(r) * 0.1
        
        # Scale factor for transition magnitude
        self.scale = 2.0  # Controls how large the shifts are
    
    def _g(self, c: np.ndarray) -> np.ndarray:
        """Ground truth g function: R^d -> R^r."""
        # Linear + tanh nonlinearity
        linear = c @ self.W_g.T + self.b_g  # (batch, r)
        return np.tanh(linear) * self.scale
    
    def _compute_delta(self, c: np.ndarray) -> np.ndarray:
        """Compute true Delta_logits = U @ g(c)."""
        reduced = self._g(c)  # (batch, r)
        return reduced @ self.U_true.T  # (batch, K)
    
    def _sample_base_distribution(self, n: int) -> np.ndarray:
        """Sample base distributions from Dirichlet."""
        alpha = np.ones(self.config.K) * self.config.base_concentration
        return self.rng.dirichlet(alpha, size=n)
    
    def _apply_shift(self, w: np.ndarray, delta: np.ndarray, 
                     noise: float = 0.0) -> np.ndarray:
        """Apply shift in log-space: w' = softmax(log(w) + delta + noise)."""
        log_w = np.log(w + 1e-10)
        noisy_delta = delta
        if noise > 0:
            noisy_delta = delta + self.rng.standard_normal(delta.shape) * noise
        
        # Softmax
        logits = log_w + noisy_delta
        logits = logits - logits.max(axis=-1, keepdims=True)  # Stability
        exp_logits = np.exp(logits)
        w_prime = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        return w_prime
    
    def generate_dataset(self, n: Optional[int] = None, 
                         seed_offset: int = 0) -> dict:
        """Generate a dataset of (w, c, w') triplets.
        
        Returns:
            dict with keys: w, c, w_target, delta_logits, config
        """
        if n is None:
            n = self.config.n_train + self.config.n_test
        
        # Use different seed for data generation
        rng_data = np.random.default_rng(self.config.seed + seed_offset + 1000)
        
        # Sample base distributions
        alpha = np.ones(self.config.K) * self.config.base_concentration
        w = rng_data.dirichlet(alpha, size=n)
        
        # Sample interventions
        c = rng_data.standard_normal((n, self.config.d))
        
        # Compute true shifts
        delta = self._compute_delta(c)
        
        # Apply shifts with noise
        w_target = self._apply_shift(w, delta, noise=self.config.noise)
        
        return {
            'w': w,
            'c': c,
            'w_target': w_target,
            'delta_logits': delta,
            'U_true': self.U_true,
            'config': {
                'K': self.config.K,
                'd': self.config.d,
                'rank': self.config.rank,
                'noise': self.config.noise,
                'n': n,
            }
        }
    
    def generate_compositional_dataset(self, n_single: int = 30,
                                        seed_offset: int = 0) -> dict:
        """Generate dataset with single AND composed interventions.
        
        Single interventions: c_i -> delta_i
        Composed: c_i + c_j -> delta_i + delta_j (additivity)
        
        This tests whether models can exploit compositional structure.
        """
        rng_comp = np.random.default_rng(self.config.seed + seed_offset + 2000)
        
        # Generate single interventions
        alpha = np.ones(self.config.K) * self.config.base_concentration
        w_base = rng_comp.dirichlet(alpha, size=1)[0]  # Single base state
        
        c_singles = rng_comp.standard_normal((n_single, self.config.d))
        delta_singles = self._compute_delta(c_singles)
        w_singles = self._apply_shift(
            np.tile(w_base, (n_single, 1)), delta_singles, self.config.noise)
        
        # Generate composed interventions (all pairs)
        n_pairs = min(n_single * (n_single - 1) // 2, 200)
        pairs = []
        for i in range(n_single):
            for j in range(i+1, n_single):
                pairs.append((i, j))
                if len(pairs) >= n_pairs:
                    break
            if len(pairs) >= n_pairs:
                break
        
        c_composed = np.array([c_singles[i] + c_singles[j] for i, j in pairs])
        delta_composed = np.array([delta_singles[i] + delta_singles[j] 
                                   for i, j in pairs])
        w_composed = self._apply_shift(
            np.tile(w_base, (len(pairs), 1)), delta_composed, self.config.noise)
        
        return {
            'single': {
                'w': np.tile(w_base, (n_single, 1)),
                'c': c_singles,
                'w_target': w_singles,
                'delta_logits': delta_singles,
            },
            'composed': {
                'w': np.tile(w_base, (len(pairs), 1)),
                'c': c_composed,
                'w_target': w_composed,
                'delta_logits': delta_composed,
            },
            'w_base': w_base,
            'pairs': pairs,
        }
    
    def get_optimal_predictor(self, w: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Return the Bayes-optimal prediction (no noise)."""
        delta = self._compute_delta(c)
        return self._apply_shift(w, delta, noise=0.0)


def generate_sweep_configs():
    """Generate all configurations for the full parameter sweep."""
    configs = []
    
    # Sweep 1: Sample efficiency (vary n, fix K=10, r=2)
    for n in [5, 10, 20, 50, 100, 200, 500]:
        configs.append(SyntheticConfig(
            K=10, d=5, rank=2, n_train=n, n_test=200,
            noise=0.02, seed=42
        ))
    
    # Sweep 2: State dimension (vary K, fix n=50, r=2)
    for K in [3, 5, 10, 20, 50]:
        configs.append(SyntheticConfig(
            K=K, d=5, rank=2, n_train=50, n_test=200,
            noise=0.02, seed=42
        ))
    
    # Sweep 3: True rank (vary r, fix K=20, n=50)
    for r in [1, 2, 3, 5, 10, 20]:  # r=20 is full rank for K=20
        configs.append(SyntheticConfig(
            K=20, d=5, rank=r, n_train=50, n_test=200,
            noise=0.02, seed=42
        ))
    
    # Sweep 4: Noise robustness (vary noise, fix K=10, n=50, r=2)
    for noise in [0.0, 0.01, 0.05, 0.1, 0.2]:
        configs.append(SyntheticConfig(
            K=10, d=5, rank=2, n_train=50, n_test=200,
            noise=noise, seed=42
        ))
    
    return configs
