"""
Process AF2 prediction data from Monteiro da Silva 2024 GitHub repository.

This script:
1. Loads AF2 prediction CSV files
2. Clusters predictions into states based on RMSD
3. Calculates state populations (AF2 frequencies)
4. Creates expanded dataset for CDST

IMPORTANT: AF2 frequencies are NOT ground truth populations.
They are used as:
- AF2-frequency baseline (to be beaten by CDST)
- State enumeration (what conformations exist)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from typing import Dict, Tuple, List
import json


def load_af2_predictions(csv_path: Path) -> pd.DataFrame:
    """Load AF2 prediction results from CSV."""
    df = pd.read_csv(csv_path)
    return df


def assign_states_by_rmsd(
    df: pd.DataFrame,
    rmsd_threshold: float = 3.0,
    i2_threshold: float = 10.0,
) -> np.ndarray:
    """Assign predictions to states based on RMSD to reference structures.

    States:
    - 0: Ground state (close to 6XR6)
    - 1: I2 state (close to 6RXG)
    - 2: Intermediate/other

    Args:
        df: DataFrame with RMSD columns
        rmsd_threshold: RMSD cutoff for the ground-state assignment (Angstrom)
        i2_threshold: looser RMSD cutoff for the I2 assignment (Angstrom);
            the I2 reference sits ~18 A from ground, so this cutoff is
            intentionally larger than rmsd_threshold
    
    Returns:
        state_assignments: (N,) array of state indices
    """
    # Get RMSD columns
    rmsd_ground_col = 'aC-Helix Backbone RMSD vs. 6XR6 (ground) aC-Helix Backbone'
    rmsd_i2_col = 'aC-Helix Backbone RMSD vs. 6XRG (I2) aC-Helix Backbone'
    
    rmsd_ground = df[rmsd_ground_col].values
    rmsd_i2 = df[rmsd_i2_col].values
    
    # Assign states based on RMSD
    states = np.full(len(df), 2, dtype=int)  # Default: intermediate
    
    # Ground state: low RMSD to ground, high RMSD to I2
    ground_mask = (rmsd_ground < rmsd_threshold) & (rmsd_ground < rmsd_i2)
    states[ground_mask] = 0
    
    # I2 state: low RMSD to I2, high RMSD to ground. The I2 cutoff is
    # intentionally looser than rmsd_threshold (the I2 reference sits ~18 A
    # from ground, so tight cutoffs assign no I2 at all); it was previously
    # a silent 10.0 hardcode, now an explicit parameter with the same
    # default so existing behaviour is unchanged.
    i2_mask = (rmsd_i2 < i2_threshold) & (rmsd_i2 < rmsd_ground)
    states[i2_mask] = 1
    
    return states


def calculate_populations(states: np.ndarray, n_states: int = 3) -> np.ndarray:
    """Calculate state populations from assignments."""
    populations = np.zeros(n_states)
    for k in range(n_states):
        populations[k] = (states == k).sum() / len(states)
    return populations


def process_mutant_series(csv_path: Path) -> Dict:
    """Process the Abl1 mutant series AF2 predictions.
    
    Returns:
        Dictionary with mutant data including AF2 frequencies
    """
    df = load_af2_predictions(csv_path)
    
    # Get unique mutants
    mutants = df['Trial'].unique()
    
    results = {}
    
    for mutant in mutants:
        mutant_df = df[df['Trial'] == mutant]
        
        # Calculate populations for each replicate
        replicates = mutant_df['Subsampling Level'].unique()
        rep_populations = []
        
        for rep in replicates:
            rep_df = mutant_df[mutant_df['Subsampling Level'] == rep]
            states = assign_states_by_rmsd(rep_df)
            pops = calculate_populations(states)
            rep_populations.append(pops)
        
        # Average across replicates
        mean_pops = np.mean(rep_populations, axis=0)
        std_pops = np.std(rep_populations, axis=0)
        
        # Parse mutation info
        mutant_clean = mutant.replace('Abl1 ', '').replace(' + ', '_')
        if mutant_clean == '':
            mutant_clean = 'WT'
        
        results[mutant_clean] = {
            'name': mutant,
            'n_predictions': len(mutant_df),
            'n_replicates': len(replicates),
            'af2_populations': mean_pops.tolist(),
            'af2_populations_std': std_pops.tolist(),
            'replicate_populations': [p.tolist() for p in rep_populations],
        }
    
    return results


def create_expanded_dataset(
    af2_results: Dict,
    output_dir: Path,
) -> Dict[str, np.ndarray]:
    """Create expanded CDST dataset from AF2 results.
    
    NOTE: This uses AF2 frequencies as a proxy for populations.
    For actual training, experimental NMR populations should be used.
    This dataset is primarily for:
    1. AF2-frequency baseline comparison
    2. Testing data pipeline
    """
    # Mutation encoding (simplified)
    mutation_info = {
        'WT': {'pos': 0, 'wt': 'X', 'mut': 'X'},
        'M290L': {'pos': 290, 'wt': 'M', 'mut': 'L'},
        'L301I': {'pos': 301, 'wt': 'L', 'mut': 'I'},
        'F382Y': {'pos': 382, 'wt': 'F', 'mut': 'Y'},
        'F382L': {'pos': 382, 'wt': 'F', 'mut': 'L'},
        'F382V': {'pos': 382, 'wt': 'F', 'mut': 'V'},
        'E255V': {'pos': 255, 'wt': 'E', 'mut': 'V'},
        'T315I': {'pos': 315, 'wt': 'T', 'mut': 'I'},
        # Double mutants encode their FIRST site only (frozen exploratory
        # convention kept for backward compatibility of stored artifacts);
        # the canonical pipeline sums per-site deltas instead.
        'M290L_L301I': {'pos': 290, 'wt': 'M', 'mut': 'L'},  # Double
        'E255V_T315I': {'pos': 255, 'wt': 'E', 'mut': 'V'},  # Double
    }
    
    # Amino acid properties
    aa_props = {
        'A': {'hydro': 1.8, 'vol': 88.6}, 'R': {'hydro': -4.5, 'vol': 173.4},
        'N': {'hydro': -3.5, 'vol': 114.1}, 'D': {'hydro': -3.5, 'vol': 111.1},
        'C': {'hydro': 2.5, 'vol': 108.5}, 'E': {'hydro': -3.5, 'vol': 138.4},
        'Q': {'hydro': -3.5, 'vol': 143.8}, 'G': {'hydro': -0.4, 'vol': 60.1},
        'H': {'hydro': -3.2, 'vol': 153.2}, 'I': {'hydro': 4.5, 'vol': 166.7},
        'L': {'hydro': 3.8, 'vol': 166.7}, 'K': {'hydro': -3.9, 'vol': 168.6},
        'M': {'hydro': 1.9, 'vol': 162.9}, 'F': {'hydro': 2.8, 'vol': 189.9},
        'P': {'hydro': -1.6, 'vol': 112.7}, 'S': {'hydro': -0.8, 'vol': 89.0},
        'T': {'hydro': -0.7, 'vol': 116.1}, 'W': {'hydro': -0.9, 'vol': 227.8},
        'Y': {'hydro': -1.3, 'vol': 193.6}, 'V': {'hydro': 4.2, 'vol': 140.0},
        'X': {'hydro': 0.0, 'vol': 0.0},
    }
    
    w_list = []
    c_list = []
    w_target_list = []
    mutant_names = []
    
    # WT reference
    wt_pops = np.array(af2_results.get('WT', {}).get('af2_populations', [0.835, 0.10, 0.065]))
    
    for mutant_name, data in af2_results.items():
        if mutant_name == 'WT':
            continue
        
        # Get mutation info
        mut_info = mutation_info.get(mutant_name, {'pos': 0, 'wt': 'X', 'mut': 'X'})
        
        # Encode mutation
        wt_aa = aa_props.get(mut_info['wt'], {'hydro': 0, 'vol': 0})
        mut_aa = aa_props.get(mut_info['mut'], {'hydro': 0, 'vol': 0})
        
        c = np.array([
            (mut_aa['hydro'] - wt_aa['hydro']) / 5.0,
            (mut_aa['vol'] - wt_aa['vol']) / 100.0,
            mut_info['pos'] / 500.0,  # Normalized position
            0.0,  # Charge change (simplified)
            0.0,  # Polarity change (simplified)
        ])
        
        # Use AF2 populations as target (NOTE: this is AF2 frequency, not experimental GT)
        w_target = np.array(data['af2_populations'])
        
        # Apply epsilon clipping
        eps = 5e-3
        w_target = np.clip(w_target, eps, 1.0)
        w_target = w_target / w_target.sum()
        
        wt_clipped = np.clip(wt_pops, eps, 1.0)
        wt_clipped = wt_clipped / wt_clipped.sum()
        
        w_list.append(wt_clipped)
        c_list.append(c)
        w_target_list.append(w_target)
        mutant_names.append(mutant_name)
    
    return {
        'w': np.array(w_list),
        'c': np.array(c_list),
        'w_target': np.array(w_target_list),
        'mutations': np.array(mutant_names),
    }


def main():
    # Paths
    data_dir = Path('data/external/github/gms_data/prediction_results')
    output_dir = Path('data/abl1_expanded')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process mutant series
    csv_path = data_dir / 'abl_mut_series_predictions_maxseq256extraseq256.csv'
    print(f"Processing: {csv_path}")
    
    af2_results = process_mutant_series(csv_path)
    
    # Print summary
    print("\n" + "="*60)
    print("AF2 Prediction Summary (Abl1 Mutant Series)")
    print("="*60)
    print(f"{'Mutant':<20} {'Ground':>10} {'I1':>10} {'I2':>10} {'N':>6}")
    print("-"*60)
    
    for name, data in af2_results.items():
        pops = data['af2_populations']
        print(f"{name:<20} {pops[0]:>10.3f} {pops[1]:>10.3f} {pops[2]:>10.3f} {data['n_predictions']:>6}")
    
    # Create expanded dataset
    print("\nCreating expanded dataset...")
    dataset = create_expanded_dataset(af2_results, output_dir)
    
    # Save
    np.savez(output_dir / 'abl1_af2_samples.npz', **dataset)
    
    # Save AF2 results as JSON
    with open(output_dir / 'af2_results.json', 'w') as f:
        json.dump(af2_results, f, indent=2)
    
    print(f"\nDataset saved to: {output_dir}")
    print(f"  Samples: {len(dataset['w'])}")
    print(f"  Encoding dim: {dataset['c'].shape[1]}")
    
    # Important note
    print("\n" + "="*60)
    print("IMPORTANT NOTE")
    print("="*60)
    print("These populations are AF2 SAMPLING FREQUENCIES, not experimental GT.")
    print("Use for:")
    print("  - AF2-frequency baseline comparison")
    print("  - State enumeration")
    print("DO NOT use as ground truth for CDST training!")
    print("="*60)


if __name__ == '__main__':
    main()
