"""
Synthetic data generators for CDST experiments.

Experiment 1a: Double-well potential (analytical ground truth)
Experiment 1b: Multi-modal GMM (d/p/K sweep)
Experiment 1c: Coupled oscillators (many-body, non-analytical)
Experiment 1d: New state emergence (failure mode)
Experiment 1e: Compositional perturbation (Proposition 3)
"""

import numpy as np
from scipy.special import softmax
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class SyntheticDataset:
    """Container for synthetic experiment data."""
    w: np.ndarray           # (N, K) initial distributions
    c: np.ndarray           # (N, intervention_dim) perturbations
    w_target: np.ndarray    # (N, K) target distributions
    K: int                  # number of states
    intervention_dim: int   # dimension of perturbation
    name: str               # experiment name
    metadata: Dict = None   # additional info


# =============================================================================
# Experiment 1a: Double-Well Potential
# =============================================================================

class DoubleWellSystem:
    """Double-well potential system with analytical ground truth.
    
    Potential: V(x) = a*(x^2 - b^2)^2 + perturbation
    Two states: left well (x < 0) and right well (x > 0)
    
    Perturbation shifts the relative depth of the two wells.
    Ground truth: Boltzmann distribution over the two wells.
    """
    
    def __init__(self, a: float = 1.0, b: float = 1.0, kT: float = 1.0):
        self.a = a
        self.b = b
        self.kT = kT
        self.K = 2
    
    def potential(self, x: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """Double-well potential with linear perturbation.
        
        V(x) = a*(x^2 - b^2)^2 - epsilon*x
        epsilon > 0 shifts population to right well.
        """
        return self.a * (x**2 - self.b**2)**2 - epsilon * x
    
    def compute_free_energies(self, epsilon: float, n_samples: int = 10000) -> np.ndarray:
        """Compute free energies of the two wells via numerical integration.
        
        F_k = -kT * log(Z_k) where Z_k = ∫_well_k exp(-V(x)/kT) dx
        """
        x = np.linspace(-3*self.b, 3*self.b, n_samples)
        dx = x[1] - x[0]
        
        boltzmann = np.exp(-self.potential(x, epsilon) / self.kT)
        
        # Left well: x < 0, Right well: x > 0
        Z_left = np.sum(boltzmann[x < 0]) * dx
        Z_right = np.sum(boltzmann[x >= 0]) * dx
        
        F_left = -self.kT * np.log(Z_left + 1e-30)
        F_right = -self.kT * np.log(Z_right + 1e-30)
        
        return np.array([F_left, F_right])
    
    def equilibrium_distribution(self, epsilon: float) -> np.ndarray:
        """Compute equilibrium Boltzmann distribution over the two wells."""
        F = self.compute_free_energies(epsilon)
        return softmax(-F / self.kT)
    
    def generate_dataset(self, N: int = 1000, epsilon_range: float = 2.0, 
                         seed: int = 42) -> SyntheticDataset:
        """Generate training/test data.
        
        Perturbation c = [epsilon] (scalar shift of potential)
        Initial state: equilibrium at epsilon=0
        Target: equilibrium at epsilon=c
        """
        rng = np.random.default_rng(seed)
        
        # Base distribution (epsilon=0, symmetric)
        w_base = self.equilibrium_distribution(0.0)
        
        # Sample perturbations
        epsilons = rng.uniform(-epsilon_range, epsilon_range, size=(N, 1))
        
        # Compute target distributions
        w_targets = np.array([self.equilibrium_distribution(eps[0]) for eps in epsilons])
        
        # Initial distributions (with small noise for diversity)
        w_init = np.tile(w_base, (N, 1))
        noise = rng.dirichlet(np.ones(self.K) * 50, size=N)
        w_init = 0.9 * w_init + 0.1 * noise
        w_init = w_init / w_init.sum(axis=1, keepdims=True)
        
        return SyntheticDataset(
            w=w_init,
            c=epsilons,
            w_target=w_targets,
            K=self.K,
            intervention_dim=1,
            name="double_well",
            metadata={'a': self.a, 'b': self.b, 'kT': self.kT, 'w_base': w_base}
        )
    
    def analytical_delta_logits(self, epsilon: float) -> np.ndarray:
        """Analytical Δlogits for verification."""
        w0 = self.equilibrium_distribution(0.0)
        w_eps = self.equilibrium_distribution(epsilon)
        return np.log(w_eps + 1e-8) - np.log(w0 + 1e-8)


# =============================================================================
# Experiment 1b: Multi-Modal GMM (d/p/K sweep)
# =============================================================================

class MultiModalGMMSystem:
    """Multi-modal Gaussian Mixture Model system.
    
    K states in d-dimensional space, each a Gaussian blob.
    Perturbation shifts the means/energies of the states.
    
    This is the "killer figure" experiment: sweep d, p (perturbation dim), K
    to show CDST's sample complexity advantage.
    """
    
    def __init__(self, K: int = 4, d: int = 10, p: int = 2, kT: float = 1.0, seed: int = 42):
        """
        Args:
            K: number of states (modes)
            d: dimension of state space
            p: dimension of perturbation space
            kT: temperature
        """
        self.K = K
        self.d = d
        self.p = p
        self.kT = kT
        self.seed = seed
        
        rng = np.random.default_rng(seed)
        
        # Generate state centers (well-separated)
        self.centers = rng.standard_normal((K, d)) * 3.0
        
        # Base energies for each state
        self.base_energies = rng.uniform(-1, 1, size=K)
        
        # Perturbation coupling matrix: how perturbation affects each state's energy
        # ΔG_k(c) = M_k · c (linear coupling)
        self.coupling_matrix = rng.standard_normal((K, p)) * 0.5
    
    def state_energies(self, c: np.ndarray) -> np.ndarray:
        """Compute energies of all states under perturbation c.
        
        G_k(c) = G_k^0 + M_k · c
        """
        # c: (..., p) → ΔG: (..., K)
        delta_G = c @ self.coupling_matrix.T  # (..., K)
        return self.base_energies + delta_G
    
    def equilibrium_distribution(self, c: np.ndarray) -> np.ndarray:
        """Compute Boltzmann distribution over K states."""
        G = self.state_energies(c)
        return softmax(-G / self.kT, axis=-1)
    
    def generate_dataset(self, N: int = 1000, c_scale: float = 1.0,
                         seed: Optional[int] = None) -> SyntheticDataset:
        """Generate dataset for training/testing."""
        rng = np.random.default_rng(seed if seed is not None else self.seed + 100)
        
        # Base distribution
        w_base = self.equilibrium_distribution(np.zeros(self.p))
        
        # Sample perturbations
        c = rng.normal(0, c_scale, size=(N, self.p))
        
        # Target distributions
        w_targets = self.equilibrium_distribution(c)
        
        # Initial distributions (base + noise)
        w_init = np.tile(w_base, (N, 1))
        noise = rng.dirichlet(np.ones(self.K) * 30, size=N)
        w_init = 0.95 * w_init + 0.05 * noise
        w_init = w_init / w_init.sum(axis=1, keepdims=True)
        
        return SyntheticDataset(
            w=w_init,
            c=c,
            w_target=w_targets,
            K=self.K,
            intervention_dim=self.p,
            name=f"gmm_K{self.K}_d{self.d}_p{self.p}",
            metadata={
                'centers': self.centers,
                'base_energies': self.base_energies,
                'coupling_matrix': self.coupling_matrix,
                'w_base': w_base,
            }
        )
    
    def generate_sweep_datasets(self, N: int = 500, 
                                K_values: list = None,
                                d_values: list = None,
                                p_values: list = None) -> Dict[str, SyntheticDataset]:
        """Generate datasets for d/p/K sweep (killer figure)."""
        if K_values is None:
            K_values = [2, 4, 8, 16]
        if d_values is None:
            d_values = [2, 5, 10, 20, 50]
        if p_values is None:
            p_values = [1, 2, 5, 10]
        
        datasets = {}
        for K in K_values:
            for d in d_values:
                for p in p_values:
                    if p > d:
                        continue
                    system = MultiModalGMMSystem(K=K, d=d, p=p, seed=self.seed)
                    key = f"K{K}_d{d}_p{p}"
                    datasets[key] = system.generate_dataset(N=N)
        
        return datasets


# =============================================================================
# Experiment 1c: Coupled Oscillators
# =============================================================================

class CoupledOscillatorSystem:
    """Coupled oscillator system (many-body, non-analytical).
    
    N oscillators with nonlinear coupling. States correspond to 
    different collective modes of the system.
    """
    
    def __init__(self, n_oscillators: int = 4, K: int = 3, kT: float = 1.0, seed: int = 42):
        self.n = n_oscillators
        self.K = K
        self.kT = kT
        self.seed = seed
        
        rng = np.random.default_rng(seed)
        
        # Coupling matrix between oscillators
        self.coupling = rng.standard_normal((n_oscillators, n_oscillators)) * 0.3
        self.coupling = (self.coupling + self.coupling.T) / 2  # symmetric
        
        # Natural frequencies
        self.omega = rng.uniform(0.5, 2.0, size=n_oscillators)
        
        # State definitions: collective mode patterns
        self.state_patterns = rng.standard_normal((K, n_oscillators))
        self.state_patterns /= np.linalg.norm(self.state_patterns, axis=1, keepdims=True)
        
        # Base energies
        self.base_energies = rng.uniform(-0.5, 0.5, size=K)
        
        # Perturbation coupling (affects oscillator frequencies)
        self.perturbation_dim = n_oscillators
        self.pert_coupling = rng.standard_normal((K, n_oscillators)) * 0.3
    
    def compute_state_energy(self, c: np.ndarray) -> np.ndarray:
        """Compute effective energies including perturbation effects.
        
        Nonlinear: includes coupling effects between oscillators.
        """
        # Linear part
        delta_G_linear = c @ self.pert_coupling.T
        
        # Nonlinear correction (coupling effects)
        c_effective = c + 0.1 * np.tanh(c @ self.coupling)
        delta_G_nonlinear = c_effective @ self.pert_coupling.T * 0.3
        
        return self.base_energies + delta_G_linear + delta_G_nonlinear
    
    def equilibrium_distribution(self, c: np.ndarray) -> np.ndarray:
        G = self.compute_state_energy(c)
        return softmax(-G / self.kT, axis=-1)
    
    def generate_dataset(self, N: int = 1000, c_scale: float = 0.8,
                         seed: Optional[int] = None) -> SyntheticDataset:
        rng = np.random.default_rng(seed if seed is not None else self.seed + 200)
        
        w_base = self.equilibrium_distribution(np.zeros(self.perturbation_dim))
        c = rng.normal(0, c_scale, size=(N, self.perturbation_dim))
        w_targets = self.equilibrium_distribution(c)
        
        w_init = np.tile(w_base, (N, 1))
        noise = rng.dirichlet(np.ones(self.K) * 30, size=N)
        w_init = 0.95 * w_init + 0.05 * noise
        w_init = w_init / w_init.sum(axis=1, keepdims=True)
        
        return SyntheticDataset(
            w=w_init,
            c=c,
            w_target=w_targets,
            K=self.K,
            intervention_dim=self.perturbation_dim,
            name="coupled_oscillators",
            metadata={'n_oscillators': self.n, 'w_base': w_base}
        )


# =============================================================================
# Experiment 1d: New State Emergence (Failure Mode)
# =============================================================================

class NewStateEmergenceSystem:
    """System where perturbation creates a NEW state (induced fit).
    
    This demonstrates CDST's failure mode: when K-stable assumption is violated.
    For small perturbations: K states remain (CDST works)
    For large perturbations: new state K+1 appears (CDST fails)
    """
    
    def __init__(self, K_base: int = 3, p: int = 2, kT: float = 1.0, 
                 emergence_threshold: float = 1.5, seed: int = 42):
        self.K_base = K_base
        self.p = p
        self.kT = kT
        self.emergence_threshold = emergence_threshold
        self.seed = seed
        
        rng = np.random.default_rng(seed)
        
        # Base state energies
        self.base_energies = rng.uniform(-1, 0.5, size=K_base)
        self.coupling = rng.standard_normal((K_base, p)) * 0.4
        
        # New state: appears when ||c|| > threshold
        self.new_state_energy_base = 2.0  # high energy (unpopulated) initially
        self.new_state_coupling = rng.standard_normal(p) * 1.5  # strongly coupled to perturbation
    
    def has_new_state(self, c: np.ndarray) -> np.ndarray:
        """Check if new state emerges for each perturbation."""
        return np.linalg.norm(c, axis=-1) > self.emergence_threshold
    
    def equilibrium_distribution(self, c: np.ndarray) -> np.ndarray:
        """Compute distribution, including new state if emerged."""
        batch_size = c.shape[0] if c.ndim > 1 else 1
        c = np.atleast_2d(c)
        
        results = []
        for i in range(batch_size):
            ci = c[i]
            
            # Base states
            G_base = self.base_energies + ci @ self.coupling.T
            
            if np.linalg.norm(ci) > self.emergence_threshold:
                # New state emerges with energy decreasing as ||c|| grows
                G_new = self.new_state_energy_base - np.linalg.norm(ci) * 0.8
                G_new += ci @ self.new_state_coupling * 0.3
                G_all = np.append(G_base, G_new)
            else:
                G_all = G_base
            
            w = softmax(-G_all / self.kT)
            results.append(w)

        # ndarray like every sibling system (a plain list broke .shape and
        # vectorized consumers); normalized rows guaranteed by softmax.
        return np.asarray(results)
    
    def generate_dataset(self, N: int = 1000, c_scale: float = 1.0,
                         seed: Optional[int] = None) -> SyntheticDataset:
        """Generate dataset with mixed K-stable and induced-fit cases."""
        rng = np.random.default_rng(seed if seed is not None else self.seed + 300)
        
        # Use K_base + 1 states (pad with 0 for non-emerged cases)
        K_full = self.K_base + 1
        
        c = rng.normal(0, c_scale, size=(N, self.p))
        
        # Compute targets (variable K, pad to K_full)
        w_targets = np.zeros((N, K_full))
        emerged = np.zeros(N, dtype=bool)
        
        for i in range(N):
            dists = self.equilibrium_distribution(c[i:i+1])[0]
            w_targets[i, :len(dists)] = dists
            if len(dists) > self.K_base:
                emerged[i] = True
        
        # Base distribution (no new state)
        w_base_full = np.zeros(K_full)
        w_base_full[:self.K_base] = softmax(-self.base_energies / self.kT)
        
        w_init = np.tile(w_base_full, (N, 1))
        
        # Compute State Coverage (SC) for each sample
        # SC = fraction of "holo" states covered by "apo" states
        # In this synthetic system: SC = 1.0 if no new state, SC = K_base/(K_base+1) if new state emerged
        sc_values = np.where(emerged, self.K_base / K_full, 1.0)
        
        return SyntheticDataset(
            w=w_init,
            c=c,
            w_target=w_targets,
            K=K_full,
            intervention_dim=self.p,
            name="new_state_emergence",
            metadata={
                'K_base': self.K_base,
                'emerged': emerged,
                'emergence_threshold': self.emergence_threshold,
                'w_base': w_base_full,
                'sc_values': sc_values,  # State Coverage for each sample
            }
        )


# =============================================================================
# Experiment 1e: Compositional Perturbation (Proposition 3)
# =============================================================================

class CompositionalPerturbationSystem:
    """System for verifying compositional property (Proposition 3).
    
    If perturbations c1, c2 act independently on different states:
        Δlogits(c1 + c2) = Δlogits(c1) + Δlogits(c2)
    
    This is an algebraic property of the softmax parameterization.
    """
    
    def __init__(self, K: int = 4, p: int = 4, kT: float = 1.0, seed: int = 42):
        self.K = K
        self.p = p
        self.kT = kT
        self.seed = seed
        
        rng = np.random.default_rng(seed)
        
        # Diagonal coupling: each perturbation dimension affects one state
        # This ensures exact compositionality
        self.coupling = np.zeros((K, p))
        for i in range(min(K, p)):
            self.coupling[i, i] = rng.uniform(0.5, 1.5)
        
        # Base energies
        self.base_energies = rng.uniform(-0.5, 0.5, size=K)
    
    def equilibrium_distribution(self, c: np.ndarray) -> np.ndarray:
        G = self.base_energies + c @ self.coupling.T
        return softmax(-G / self.kT, axis=-1)
    
    def generate_dataset(self, N: int = 1000, c_scale: float = 1.0,
                         seed: Optional[int] = None) -> SyntheticDataset:
        rng = np.random.default_rng(seed if seed is not None else self.seed + 400)
        
        w_base = self.equilibrium_distribution(np.zeros(self.p))
        c = rng.normal(0, c_scale, size=(N, self.p))
        w_targets = self.equilibrium_distribution(c)
        
        w_init = np.tile(w_base, (N, 1))
        
        return SyntheticDataset(
            w=w_init,
            c=c,
            w_target=w_targets,
            K=self.K,
            intervention_dim=self.p,
            name="compositional",
            metadata={'coupling': self.coupling, 'w_base': w_base}
        )
    
    def generate_composition_pairs(self, N: int = 500, c_scale: float = 0.8,
                                   seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate pairs (c1, c2) for testing compositionality."""
        rng = np.random.default_rng(seed if seed is not None else self.seed + 500)
        
        c1 = rng.normal(0, c_scale, size=(N, self.p))
        c2 = rng.normal(0, c_scale, size=(N, self.p))
        
        return c1, c2, c1 + c2
