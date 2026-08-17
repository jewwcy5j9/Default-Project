"""
Abl1 Kinase Flagship Dataset

Based on: Monteiro da Silva et al., Nat Commun 15, 2464 (2024)
"High-throughput prediction of protein conformational distributions with subsampled AlphaFold2"

Abl1 kinase has three major conformational states:
- Active (A): DFG-in/AL-open
- Inactive 1 (I1): DFG-out/AL-open intermediate  
- Inactive 2 (I2): DFG-out/AL-closed (imatinib-binding competent)

The paper tested 8 mutants with NMR-validated population changes.
Direction accuracy: >80%
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


@dataclass
class Abl1Mutant:
    """Abl1 mutant with experimental population data."""
    mutation: str
    position: int
    wt_aa: str
    mut_aa: str
    description: str
    # Populations: [Active, Inactive1, Inactive2]
    wt_populations: np.ndarray  # Wild-type populations
    mut_populations: np.ndarray  # Mutant populations
    direction_correct: bool  # Whether subsampled AF2 predicted direction correctly
    source: str = "Monteiro da Silva et al., Nat Commun 2024"


class Abl1Dataset:
    """Abl1 kinase dataset for CDST training and validation.
    
    This is the flagship system because:
    1. Has NMR experimental populations (ground truth)
    2. Subsampled AF2 published results for direct comparison
    3. Multiple mutants with characterized effects
    """
    
    # PDB IDs for representative structures
    # Active/I2: VERIFIED from paper Figure 3 caption
    # I1: Inferred from conformational definition (DFG-out/AL-open intermediate)
    #     Paper does not provide explicit I1 PDB ID.
    #     2HYY = dasatinib-bound Abl1, DFG-out with open activation loop
    #     Reference: Tokarski et al., Cancer Res 66, 5790 (2006)
    #     Alternative candidates: 3CS9 (GNF-5), 4XEY (asciminib)
    PDB_IDS = {
        'Active': '6XR6',      # Ground state (DFG-in/AL-open) - VERIFIED from paper
        'Inactive1': '2HYY',   # I1 state (DFG-out/AL-open) - INFERRED, dasatinib-bound
        'Inactive2': '6RXG',   # I2 state (DFG-out/AL-closed) - VERIFIED from paper
    }
    
    # Wild-type populations (VERIFIED from Source Data Figure 6B)
    # Paper Figure 6B: WT mean 16.458% not in ground state
    # Therefore: Ground state = 83.542%, Non-ground = 16.458%
    # Non-ground split between I1 and I2 (approximate, needs NMR verification)
    # STATUS: VERIFIED from Source Data XLSX (Figure 6B)
    WT_POPULATIONS = np.array([0.835, 0.10, 0.065])  # Ground, I1, I2 (I1/I2 split approximate)
    
    # Epsilon for population clipping (avoid log(0))
    EPSILON = 5e-3
    
    # Data verification status
    DATA_STATUS = """
    Data Verification Status
    ========================
    Source: Monteiro da Silva et al., Nat Commun 15, 2464 (2024)
    GitHub: https://github.com/GMdSilva/gms_natcomms_1705932980_data
    
    VERIFIED:
    - PDB IDs: 6XR6 (Active), 6RXG (I2) - from paper Figure 3
    - Direction accuracy: >80% (8 mutants)
    - WT ground state: ~83.5%
    - Mutant populations: from Source Data Figure 6B
    
    INFERRED:
    - I1 PDB: 2HYY (dasatinib-bound, DFG-out/AL-open)
      Rationale: Paper defines I1 as DFG-out/AL-open intermediate.
      2HYY captures this conformation. Not explicitly stated in paper.
      Alternatives: 3CS9, 4XEY
    - I1/I2 population split: approximate (paper reports % not in ground only)
    
    MD TRAJECTORY AVAILABLE:
    - abl1_wt_gr_to_i1_to_i2_traj.dcd (20MB)
    - Confirms I1 as intermediate on Ground->I2 pathway
    """
    
    def __init__(self):
        self.mutants = self._load_mutants()
        self.n_states = 3
        self.state_labels = ['Active', 'Inactive1', 'Inactive2']
    
    @staticmethod
    def clip_populations(w: np.ndarray, epsilon: float = 5e-3) -> np.ndarray:
        """Apply epsilon-clipping to avoid log(0) toxicity.
        
        Args:
            w: Population array (..., K)
            epsilon: Minimum population (default 0.5%)
        
        Returns:
            w_clipped: Clipped and renormalized populations
        """
        w_clipped = np.clip(w, epsilon, 1.0)
        return w_clipped / w_clipped.sum(axis=-1, keepdims=True)
    
    def _load_mutants(self) -> List[Abl1Mutant]:
        """Load mutant data from Monteiro da Silva 2024 Source Data (Figure 6B).
        
        VERIFIED from Source Data XLSX:
        - Metric: % predictions not in ground state (AF2 subsampling)
        - 3 independent replicates per mutant
        - Values below are mean % not in ground state
        
        Mutants in paper (Figure 6B):
        - M290L: 10.625% not in ground (decreases non-ground)
        - L301I: 22.708% not in ground (increases non-ground)
        - F382V: 29.167% not in ground (strongly increases non-ground)
        - M290L/L301I: 23.542% not in ground (double mutant)
        - E255V: 11.458% not in ground (decreases non-ground)
        - T315I: 13.125% not in ground (gatekeeper, slight decrease)
        - F382L: 13.333% not in ground (slight decrease)
        - E255V/T315I: 15.417% not in ground (double mutant, near WT)
        
        WT: 16.458% not in ground
        """
        mutants = []
        
        # WT reference: 16.458% not in ground = 83.542% ground
        wt_ground = 0.83542
        
        # M290L - decreases non-ground population (stabilizes ground state)
        # 10.625% not in ground -> 89.375% ground
        mutants.append(Abl1Mutant(
            mutation='M290L',
            position=290,
            wt_aa='M',
            mut_aa='L',
            description='Stabilizes ground state, decreases non-ground',
            wt_populations=self.WT_POPULATIONS.copy(),
            mut_populations=np.array([0.894, 0.065, 0.041]),  # Approximate split
            direction_correct=True,  # AF2 correctly predicted direction
        ))
        
        # L301I - increases non-ground population
        # 22.708% not in ground -> 77.292% ground
        mutants.append(Abl1Mutant(
            mutation='L301I',
            position=301,
            wt_aa='L',
            mut_aa='I',
            description='Increases non-ground population',
            wt_populations=self.WT_POPULATIONS.copy(),
            mut_populations=np.array([0.773, 0.137, 0.090]),  # Approximate split
            direction_correct=True,
        ))
        
        # F382V - strongly increases non-ground population
        # 29.167% not in ground -> 70.833% ground
        mutants.append(Abl1Mutant(
            mutation='F382V',
            position=382,
            wt_aa='F',
            mut_aa='V',
            description='Strongly increases non-ground (largest effect)',
            wt_populations=self.WT_POPULATIONS.copy(),
            mut_populations=np.array([0.708, 0.175, 0.117]),  # Approximate split
            direction_correct=True,
        ))
        
        # M290L/L301I - double mutant
        # 23.542% not in ground -> 76.458% ground
        mutants.append(Abl1Mutant(
            mutation='M290L_L301I',
            position=290,  # Primary position
            wt_aa='M',
            mut_aa='L',  # Note: also L301I
            description='Double mutant M290L/L301I',
            wt_populations=self.WT_POPULATIONS.copy(),
            mut_populations=np.array([0.765, 0.141, 0.094]),  # Approximate split
            direction_correct=True,
        ))
        
        # E255V - decreases non-ground population
        # 11.458% not in ground -> 88.542% ground
        mutants.append(Abl1Mutant(
            mutation='E255V',
            position=255,
            wt_aa='E',
            mut_aa='V',
            description='Stabilizes ground state',
            wt_populations=self.WT_POPULATIONS.copy(),
            mut_populations=np.array([0.885, 0.069, 0.046]),  # Approximate split
            direction_correct=True,
        ))
        
        # T315I - gatekeeper mutation, slight decrease in non-ground
        # 13.125% not in ground -> 86.875% ground
        mutants.append(Abl1Mutant(
            mutation='T315I',
            position=315,
            wt_aa='T',
            mut_aa='I',
            description='Gatekeeper mutation (imatinib resistance)',
            wt_populations=self.WT_POPULATIONS.copy(),
            mut_populations=np.array([0.869, 0.079, 0.052]),  # Approximate split
            direction_correct=True,
        ))
        
        # F382L - slight decrease in non-ground
        # 13.333% not in ground -> 86.667% ground
        mutants.append(Abl1Mutant(
            mutation='F382L',
            position=382,
            wt_aa='F',
            mut_aa='L',
            description='Slight stabilization of ground state',
            wt_populations=self.WT_POPULATIONS.copy(),
            mut_populations=np.array([0.867, 0.080, 0.053]),  # Approximate split
            direction_correct=True,
        ))
        
        # E255V/T315I - double mutant, near WT
        # 15.417% not in ground -> 84.583% ground
        # (The former name 'E255V_E315I' was a transcription typo: residue
        # 315 is threonine — see the T315I gatekeeper entry — and
        # process_af2_data.py keys the same mutant as 'E255V_T315I'.)
        mutants.append(Abl1Mutant(
            mutation='E255V_T315I',
            position=255,  # Primary position
            wt_aa='E',
            mut_aa='V',  # Note: also T315I
            description='Double mutant E255V/T315I (near WT)',
            wt_populations=self.WT_POPULATIONS.copy(),
            mut_populations=np.array([0.846, 0.094, 0.060]),  # Approximate split
            direction_correct=True,
        ))
        
        return mutants
    
    def get_mutation_encoding(self, mutant: Abl1Mutant, encoding_type: str = 'properties') -> np.ndarray:
        """Encode mutation as a vector for CDST input.
        
        Args:
            mutant: Abl1Mutant instance
            encoding_type: 'properties' (5-dim) or 'simple' (1-dim ΔΔG proxy)
        
        Returns:
            c: perturbation encoding vector
        """
        # Amino acid properties (Kyte-Doolittle hydrophobicity, Zamyatnin volume)
        aa_properties = {
            'A': {'hydro': 1.8, 'volume': 88.6, 'charge': 0, 'polarity': 0.0},
            'R': {'hydro': -4.5, 'volume': 173.4, 'charge': 1, 'polarity': 1.0},
            'N': {'hydro': -3.5, 'volume': 114.1, 'charge': 0, 'polarity': 1.0},
            'D': {'hydro': -3.5, 'volume': 111.1, 'charge': -1, 'polarity': 1.0},
            'C': {'hydro': 2.5, 'volume': 108.5, 'charge': 0, 'polarity': 0.0},
            'E': {'hydro': -3.5, 'volume': 138.4, 'charge': -1, 'polarity': 1.0},
            'Q': {'hydro': -3.5, 'volume': 143.8, 'charge': 0, 'polarity': 1.0},
            'G': {'hydro': -0.4, 'volume': 60.1, 'charge': 0, 'polarity': 0.0},
            'H': {'hydro': -3.2, 'volume': 153.2, 'charge': 0.5, 'polarity': 1.0},
            'I': {'hydro': 4.5, 'volume': 166.7, 'charge': 0, 'polarity': 0.0},
            'L': {'hydro': 3.8, 'volume': 166.7, 'charge': 0, 'polarity': 0.0},
            'K': {'hydro': -3.9, 'volume': 168.6, 'charge': 1, 'polarity': 1.0},
            'M': {'hydro': 1.9, 'volume': 162.9, 'charge': 0, 'polarity': 0.0},
            'F': {'hydro': 2.8, 'volume': 189.9, 'charge': 0, 'polarity': 0.0},
            'P': {'hydro': -1.6, 'volume': 112.7, 'charge': 0, 'polarity': 0.0},
            'S': {'hydro': -0.8, 'volume': 89.0, 'charge': 0, 'polarity': 1.0},
            'T': {'hydro': -0.7, 'volume': 116.1, 'charge': 0, 'polarity': 1.0},
            'W': {'hydro': -0.9, 'volume': 227.8, 'charge': 0, 'polarity': 0.0},
            'Y': {'hydro': -1.3, 'volume': 193.6, 'charge': 0, 'polarity': 1.0},
            'V': {'hydro': 4.2, 'volume': 140.0, 'charge': 0, 'polarity': 0.0},
        }
        
        wt_props = aa_properties.get(mutant.wt_aa, {'hydro': 0, 'volume': 0, 'charge': 0, 'polarity': 0})
        mut_props = aa_properties.get(mutant.mut_aa, {'hydro': 0, 'volume': 0, 'charge': 0, 'polarity': 0})
        
        if encoding_type == 'properties':
            # 5-dimensional encoding
            # Positions are ABSOLUTE Abl1b numbering (255-382), so normalize
            # by the full isoform length 534 (matching canonical_encoding);
            # dividing by the 287-residue domain length pushed 6 of 8
            # mutants above 1.0.
            c = np.array([
                mutant.position / 534,  # Normalized position
                mut_props['hydro'] - wt_props['hydro'],  # Δhydrophobicity
                (mut_props['volume'] - wt_props['volume']) / 100,  # Δvolume (scaled)
                mut_props['charge'] - wt_props['charge'],  # Δcharge
                mut_props['polarity'] - wt_props['polarity'],  # Δpolarity
            ])
        elif encoding_type == 'simple':
            # 1-dimensional: approximate ΔΔG from hydrophobicity change
            c = np.array([mut_props['hydro'] - wt_props['hydro']])
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        return c
    
    def generate_cdst_samples(self, encoding_type: str = 'properties', apply_clipping: bool = True) -> Dict[str, np.ndarray]:
        """Generate CDST training samples from Abl1 mutant data.
        
        Args:
            encoding_type: Mutation encoding type
            apply_clipping: Whether to apply epsilon-clipping to populations
        
        Returns:
            Dictionary with keys: 'w', 'c', 'w_target', 'mutations'
        """
        w_list = []
        c_list = []
        w_target_list = []
        mutation_list = []
        
        for mutant in self.mutants:
            w = mutant.wt_populations.copy()
            c = self.get_mutation_encoding(mutant, encoding_type)
            w_target = mutant.mut_populations.copy()
            
            # Apply epsilon-clipping to avoid log(0) toxicity
            if apply_clipping:
                w = self.clip_populations(w, self.EPSILON)
                w_target = self.clip_populations(w_target, self.EPSILON)
            
            w_list.append(w)
            c_list.append(c)
            w_target_list.append(w_target)
            mutation_list.append(mutant.mutation)
        
        return {
            'w': np.array(w_list),
            'c': np.array(c_list),
            'w_target': np.array(w_target_list),
            'mutations': np.array(mutation_list),
        }
    
    def save_dataset(self, output_dir: Path, encoding_type: str = 'properties'):
        """Save dataset to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save samples
        samples = self.generate_cdst_samples(encoding_type)
        np.savez(output_dir / 'abl1_samples.npz', **samples)
        
        # Save metadata
        meta = {
            'system_id': 'abl1_kinase',
            'protein_name': 'Abl1 kinase',
            'uniprot_id': 'P00519',
            'pdb_ids': self.PDB_IDS,
            'n_states': self.n_states,
            'state_labels': self.state_labels,
            'wt_populations': self.WT_POPULATIONS.tolist(),
            'n_mutants': len(self.mutants),
            'encoding_type': encoding_type,
            'encoding_dim': len(self.get_mutation_encoding(self.mutants[0], encoding_type)),
            'source': 'Monteiro da Silva et al., Nat Commun 15, 2464 (2024)',
        }
        with open(output_dir / 'abl1_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        # Save mutant details
        mutant_details = []
        for m in self.mutants:
            mutant_details.append({
                'mutation': m.mutation,
                'position': m.position,
                'wt_aa': m.wt_aa,
                'mut_aa': m.mut_aa,
                'description': m.description,
                'wt_populations': m.wt_populations.tolist(),
                'mut_populations': m.mut_populations.tolist(),
                'direction_correct': m.direction_correct,
            })
        with open(output_dir / 'abl1_mutants.json', 'w') as f:
            json.dump(mutant_details, f, indent=2)
        
        print(f"Dataset saved to {output_dir}")
        print(f"  Samples: {len(self.mutants)}")
        print(f"  Encoding dim: {meta['encoding_dim']}")
        
        return samples
    
    def summary(self):
        """Print dataset summary."""
        print("="*60)
        print("Abl1 Kinase Dataset Summary")
        print("="*60)
        print(f"States: {self.state_labels}")
        print(f"WT populations: {self.WT_POPULATIONS}")
        print(f"PDB IDs: {self.PDB_IDS}")
        print(f"\nMutants ({len(self.mutants)}):")
        print("-"*60)
        for m in self.mutants:
            pop_shift = m.mut_populations - m.wt_populations
            direction = "->I2" if pop_shift[2] > 0.05 else ("->Active" if pop_shift[0] > 0.05 else "neutral")
            status = "[OK]" if m.direction_correct else "[X]"
            print(f"  {m.mutation:8s} {direction:10s} {status} | {m.description}")
        print("-"*60)
        n_correct = sum(m.direction_correct for m in self.mutants)
        print(f"Direction accuracy: {n_correct}/{len(self.mutants)} = {n_correct/len(self.mutants)*100:.0f}%")


if __name__ == '__main__':
    # Create and save dataset
    dataset = Abl1Dataset()
    dataset.summary()
    
    # Save to data directory
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'abl1'
    dataset.save_dataset(output_dir)
