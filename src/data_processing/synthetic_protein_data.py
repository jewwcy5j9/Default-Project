"""
Physics-based synthetic data generator for CDST training.

This module generates realistic protein mutation data based on:
1. Boltzmann distribution for state populations
2. Physical mutation effects (hydrophobicity, volume, charge)
3. Free energy perturbation theory

Used to expand training data when experimental data is limited.
"""

import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class ProteinState:
    """A conformational state with physical properties."""
    name: str
    energy: float  # kcal/mol relative to ground state
    hydrophobicity_preference: float  # Preference for hydrophobic residues
    volume_tolerance: float  # Tolerance for large residues
    charge_preference: float  # Preference for charged residues


class PhysicsBasedGenerator:
    """Generate synthetic mutation data based on physical principles."""
    
    # Amino acid properties
    AA_PROPERTIES = {
        'A': {'hydro': 1.8, 'vol': 88.6, 'charge': 0, 'polarity': 0},
        'R': {'hydro': -4.5, 'vol': 173.4, 'charge': 1, 'polarity': 1},
        'N': {'hydro': -3.5, 'vol': 114.1, 'charge': 0, 'polarity': 1},
        'D': {'hydro': -3.5, 'vol': 111.1, 'charge': -1, 'polarity': 1},
        'C': {'hydro': 2.5, 'vol': 108.5, 'charge': 0, 'polarity': 0},
        'E': {'hydro': -3.5, 'vol': 138.4, 'charge': -1, 'polarity': 1},
        'Q': {'hydro': -3.5, 'vol': 143.8, 'charge': 0, 'polarity': 1},
        'G': {'hydro': -0.4, 'vol': 60.1, 'charge': 0, 'polarity': 0},
        'H': {'hydro': -3.2, 'vol': 153.2, 'charge': 0.5, 'polarity': 1},
        'I': {'hydro': 4.5, 'vol': 166.7, 'charge': 0, 'polarity': 0},
        'L': {'hydro': 3.8, 'vol': 166.7, 'charge': 0, 'polarity': 0},
        'K': {'hydro': -3.9, 'vol': 168.6, 'charge': 1, 'polarity': 1},
        'M': {'hydro': 1.9, 'vol': 162.9, 'charge': 0, 'polarity': 0},
        'F': {'hydro': 2.8, 'vol': 189.9, 'charge': 0, 'polarity': 0},
        'P': {'hydro': -1.6, 'vol': 112.7, 'charge': 0, 'polarity': 0},
        'S': {'hydro': -0.8, 'vol': 89.0, 'charge': 0, 'polarity': 1},
        'T': {'hydro': -0.7, 'vol': 116.1, 'charge': 0, 'polarity': 1},
        'W': {'hydro': -0.9, 'vol': 227.8, 'charge': 0, 'polarity': 0},
        'Y': {'hydro': -1.3, 'vol': 193.6, 'charge': 0, 'polarity': 1},
        'V': {'hydro': 4.2, 'vol': 140.0, 'charge': 0, 'polarity': 0},
    }
    
    def __init__(
        self,
        n_states: int = 3,
        temperature: float = 300.0,
        seed: int = 42,
    ):
        """Initialize generator.
        
        Args:
            n_states: Number of conformational states
            temperature: Temperature in Kelvin
            seed: Random seed
        """
        self.n_states = n_states
        self.T = temperature
        self.R = 1.987e-3  # kcal/(mol·K)
        self.RT = self.R * self.T
        self.rng = np.random.default_rng(seed)
        
        # Define default states (Abl1-like)
        self.states = self._create_default_states()
    
    def _create_default_states(self) -> List[ProteinState]:
        """Create default conformational states."""
        return [
            ProteinState(
                name='Active',
                energy=0.0,  # Ground state
                hydrophobicity_preference=0.5,
                volume_tolerance=0.5,
                charge_preference=0.0,
            ),
            ProteinState(
                name='Inactive1',
                energy=1.5,  # ~1.5 kcal/mol higher
                hydrophobicity_preference=-0.3,
                volume_tolerance=0.3,
                charge_preference=0.2,
            ),
            ProteinState(
                name='Inactive2',
                energy=2.5,  # ~2.5 kcal/mol higher
                hydrophobicity_preference=-0.5,
                volume_tolerance=-0.2,
                charge_preference=0.5,
            ),
        ]
    
    def boltzmann_populations(self, energies: np.ndarray) -> np.ndarray:
        """Calculate Boltzmann populations from energies.
        
        Args:
            energies: State energies (K,) in kcal/mol
        
        Returns:
            Populations (K,) summing to 1
        """
        # Shift for numerical stability
        energies_shifted = energies - energies.min()
        
        # Boltzmann factors
        boltzmann = np.exp(-energies_shifted / self.RT)
        
        # Normalize
        populations = boltzmann / boltzmann.sum()
        
        return populations
    
    def mutation_energy_effect(
        self,
        wt_aa: str,
        mut_aa: str,
        state: ProteinState,
        position_factor: float = 1.0,
    ) -> float:
        """Calculate mutation effect on state energy.
        
        Args:
            wt_aa: Wild-type amino acid
            mut_aa: Mutant amino acid
            state: Target conformational state
            position_factor: Position-dependent scaling (0-1)
        
        Returns:
            ΔΔG contribution (kcal/mol)
        """
        wt_props = self.AA_PROPERTIES.get(wt_aa, {'hydro': 0, 'vol': 0, 'charge': 0, 'polarity': 0})
        mut_props = self.AA_PROPERTIES.get(mut_aa, {'hydro': 0, 'vol': 0, 'charge': 0, 'polarity': 0})
        
        # Hydrophobicity effect
        delta_hydro = mut_props['hydro'] - wt_props['hydro']
        hydro_effect = -delta_hydro * state.hydrophobicity_preference * 0.1
        
        # Volume effect (steric clashes)
        delta_vol = mut_props['vol'] - wt_props['vol']
        volume_effect = delta_vol * (1 - state.volume_tolerance) * 0.005
        
        # Charge effect
        delta_charge = mut_props['charge'] - wt_props['charge']
        charge_effect = delta_charge * state.charge_preference * 0.3
        
        # Polarity effect
        delta_polarity = mut_props['polarity'] - wt_props['polarity']
        polarity_effect = delta_polarity * 0.1
        
        # Total effect with position scaling
        total_effect = (hydro_effect + volume_effect + charge_effect + polarity_effect) * position_factor
        
        return total_effect
    
    def generate_mutation_sample(
        self,
        wt_aa: str,
        mut_aa: str,
        position: int,
        sequence_length: int = 300,
    ) -> Dict:
        """Generate a single mutation sample.
        
        Args:
            wt_aa: Wild-type amino acid
            mut_aa: Mutant amino acid
            position: Mutation position
            sequence_length: Protein sequence length
        
        Returns:
            Dictionary with w, c, w_target
        """
        # Position factor (core vs surface)
        # Assume positions 50-250 are core, others surface
        if 50 <= position <= 250:
            position_factor = 1.0  # Core
        else:
            position_factor = 0.5  # Surface
        
        # Calculate WT populations
        wt_energies = np.array([s.energy for s in self.states])
        w_wt = self.boltzmann_populations(wt_energies)
        
        # Calculate mutation effects on each state
        mut_energies = wt_energies.copy()
        for i, state in enumerate(self.states):
            delta_g = self.mutation_energy_effect(wt_aa, mut_aa, state, position_factor)
            mut_energies[i] += delta_g
        
        # Calculate mutant populations
        w_mut = self.boltzmann_populations(mut_energies)
        
        # Encode mutation
        wt_props = self.AA_PROPERTIES.get(wt_aa, {'hydro': 0, 'vol': 0, 'charge': 0})
        mut_props = self.AA_PROPERTIES.get(mut_aa, {'hydro': 0, 'vol': 0, 'charge': 0})
        
        c = np.array([
            (mut_props['hydro'] - wt_props['hydro']) / 5.0,
            (mut_props['vol'] - wt_props['vol']) / 100.0,
            mut_props['charge'] - wt_props['charge'],
            position / sequence_length,
            0.0,  # Polarity placeholder
        ])
        
        return {
            'w': w_wt,
            'c': c,
            'w_target': w_mut,
            'mutation': f'{wt_aa}{position}{mut_aa}',
            'ddg': mut_energies[0] - wt_energies[0],  # ΔΔG for ground state
        }
    
    def generate_dataset(
        self,
        n_samples: int = 1000,
        sequence_length: int = 300,
    ) -> Dict:
        """Generate a synthetic dataset.
        
        Args:
            n_samples: Number of samples to generate
            sequence_length: Protein sequence length
        
        Returns:
            Dataset dictionary
        """
        amino_acids = list(self.AA_PROPERTIES.keys())
        
        w_list = []
        c_list = []
        w_target_list = []
        mutation_list = []
        ddg_list = []
        
        for _ in range(n_samples):
            # Random mutation
            wt_aa = self.rng.choice(amino_acids)
            mut_aa = self.rng.choice([aa for aa in amino_acids if aa != wt_aa])
            position = self.rng.integers(1, sequence_length + 1)
            
            sample = self.generate_mutation_sample(
                wt_aa, mut_aa, position, sequence_length
            )
            
            w_list.append(sample['w'])
            c_list.append(sample['c'])
            w_target_list.append(sample['w_target'])
            mutation_list.append(sample['mutation'])
            ddg_list.append(sample['ddg'])
        
        return {
            'w': np.array(w_list),
            'c': np.array(c_list),
            'w_target': np.array(w_target_list),
            'mutations': np.array(mutation_list),
            'ddg': np.array(ddg_list),
            'n_samples': n_samples,
            'n_states': self.n_states,
            'mutation_dim': 5,
        }
    
    def generate_abl1_like_dataset(
        self,
        n_samples: int = 100,
    ) -> Dict:
        """Generate Abl1-like dataset with realistic mutations.
        
        Focuses on kinase-relevant positions and mutations.
        """
        # Abl1-like mutation hotspots
        hotspots = [
            (255, 'E', ['V', 'K', 'G']),  # P-loop
            (290, 'M', ['L', 'I', 'V']),  # αC-helix
            (301, 'L', ['I', 'M', 'V']),  # αC-helix
            (315, 'T', ['I', 'M', 'A']),  # Gatekeeper
            (360, 'I', ['M', 'V', 'L']),  # αC-helix
            (382, 'F', ['L', 'V', 'Y']),  # C-lobe
            (472, 'M', ['I', 'L', 'V']),  # C-lobe
            (486, 'F', ['S', 'L', 'V']),  # C-lobe
        ]
        
        w_list = []
        c_list = []
        w_target_list = []
        mutation_list = []
        ddg_list = []
        
        for _ in range(n_samples):
            # Pick random hotspot
            pos, wt_aa, mut_choices = hotspots[self.rng.integers(len(hotspots))]
            mut_aa = self.rng.choice(mut_choices)
            
            # Hotspot positions are absolute Abl1b numbering (290-486), so
            # normalize by the full isoform length 534; the former 287
            # pushed the 382/472/486 hotspots outside [0, 1].
            sample = self.generate_mutation_sample(wt_aa, mut_aa, pos, 534)
            
            w_list.append(sample['w'])
            c_list.append(sample['c'])
            w_target_list.append(sample['w_target'])
            mutation_list.append(sample['mutation'])
            ddg_list.append(sample['ddg'])
        
        return {
            'w': np.array(w_list),
            'c': np.array(c_list),
            'w_target': np.array(w_target_list),
            'mutations': np.array(mutation_list),
            'ddg': np.array(ddg_list),
            'n_samples': n_samples,
            'n_states': self.n_states,
            'mutation_dim': 5,
        }


def main():
    """Generate and save synthetic datasets."""
    from pathlib import Path
    
    output_dir = Path('data/synthetic')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Physics-Based Synthetic Data Generation")
    print("="*60)
    
    # Initialize generator
    generator = PhysicsBasedGenerator(n_states=3, seed=42)
    
    # Generate general dataset
    print("\nGenerating general dataset (1000 samples)...")
    general_data = generator.generate_dataset(n_samples=1000)
    np.savez(output_dir / 'synthetic_general.npz', **general_data)
    print(f"  Saved: {output_dir / 'synthetic_general.npz'}")
    print(f"  Samples: {general_data['n_samples']}")
    print(f"  ΔΔG range: [{general_data['ddg'].min():.2f}, {general_data['ddg'].max():.2f}]")
    
    # Generate Abl1-like dataset
    print("\nGenerating Abl1-like dataset (100 samples)...")
    abl1_data = generator.generate_abl1_like_dataset(n_samples=100)
    np.savez(output_dir / 'synthetic_abl1.npz', **abl1_data)
    print(f"  Saved: {output_dir / 'synthetic_abl1.npz'}")
    print(f"  Samples: {abl1_data['n_samples']}")
    print(f"  ΔΔG range: [{abl1_data['ddg'].min():.2f}, {abl1_data['ddg'].max():.2f}]")
    
    # Statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    
    print("\nGeneral Dataset:")
    print(f"  Ground state: {general_data['w_target'][:, 0].mean():.3f} +/- {general_data['w_target'][:, 0].std():.3f}")
    print(f"  I1 state: {general_data['w_target'][:, 1].mean():.3f} +/- {general_data['w_target'][:, 1].std():.3f}")
    print(f"  I2 state: {general_data['w_target'][:, 2].mean():.3f} +/- {general_data['w_target'][:, 2].std():.3f}")
    
    print("\nAbl1-like Dataset:")
    print(f"  Ground state: {abl1_data['w_target'][:, 0].mean():.3f} +/- {abl1_data['w_target'][:, 0].std():.3f}")
    print(f"  I1 state: {abl1_data['w_target'][:, 1].mean():.3f} +/- {abl1_data['w_target'][:, 1].std():.3f}")
    print(f"  I2 state: {abl1_data['w_target'][:, 2].mean():.3f} +/- {abl1_data['w_target'][:, 2].std():.3f}")
    
    print("\n" + "="*60)
    print("Usage")
    print("="*60)
    print("""
These synthetic datasets can be used to:
1. Pre-train CDST on diverse mutations
2. Fine-tune on experimental Abl1 data
3. Validate model architecture
4. Generate learning curves

Note: Synthetic data is based on physical approximations.
Always validate on experimental data before drawing conclusions.
""")


if __name__ == '__main__':
    main()
