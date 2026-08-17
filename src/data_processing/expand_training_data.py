"""
Expand CDST training data using AF2 prediction ensembles.

This script:
1. Loads AF2 prediction data from GitHub (480 predictions per mutant)
2. Clusters predictions into conformational states
3. Calculates state populations
4. Creates expanded training dataset

Note: AF2 frequencies are used as training labels here,
with the understanding that they approximate Boltzmann populations
(Monteiro da Silva et al., 2024 validated r > 0.9 correlation).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple
import json


def load_af2_predictions(csv_path: Path) -> pd.DataFrame:
    """Load AF2 prediction results."""
    return pd.read_csv(csv_path)


def cluster_by_rmsd(
    df: pd.DataFrame,
    n_clusters: int = 3,
    rmsd_threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster AF2 predictions by RMSD to reference states.
    
    Returns:
        state_assignments: (N,) array of state indices
        populations: (K,) array of state populations
    """
    # Get RMSD columns
    rmsd_ground = df['aC-Helix Backbone RMSD vs. 6XR6 (ground) aC-Helix Backbone'].values
    rmsd_i2 = df['aC-Helix Backbone RMSD vs. 6XRG (I2) aC-Helix Backbone'].values
    
    # Feature matrix for clustering
    features = np.column_stack([rmsd_ground, rmsd_i2])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # Calculate populations
    populations = np.zeros(n_clusters)
    for k in range(n_clusters):
        populations[k] = (labels == k).sum() / len(labels)
    
    # Sort by population (largest first = ground state)
    sort_idx = np.argsort(-populations)
    populations = populations[sort_idx]
    
    # Remap labels
    label_map = {old: new for new, old in enumerate(sort_idx)}
    labels_sorted = np.array([label_map[l] for l in labels])
    
    return labels_sorted, populations


def encode_mutation(
    mutation_name: str,
    sequence_length: int = 287,
) -> np.ndarray:
    """Encode mutation as feature vector.
    
    Args:
        mutation_name: e.g., "M290L", "L301I"
        sequence_length: Length of protein sequence
    
    Returns:
        Encoding vector (5,)
    """
    # Amino acid properties
    aa_props = {
        'A': {'hydro': 1.8, 'vol': 88.6, 'charge': 0},
        'R': {'hydro': -4.5, 'vol': 173.4, 'charge': 1},
        'N': {'hydro': -3.5, 'vol': 114.1, 'charge': 0},
        'D': {'hydro': -3.5, 'vol': 111.1, 'charge': -1},
        'C': {'hydro': 2.5, 'vol': 108.5, 'charge': 0},
        'E': {'hydro': -3.5, 'vol': 138.4, 'charge': -1},
        'Q': {'hydro': -3.5, 'vol': 143.8, 'charge': 0},
        'G': {'hydro': -0.4, 'vol': 60.1, 'charge': 0},
        'H': {'hydro': -3.2, 'vol': 153.2, 'charge': 0.5},
        'I': {'hydro': 4.5, 'vol': 166.7, 'charge': 0},
        'L': {'hydro': 3.8, 'vol': 166.7, 'charge': 0},
        'K': {'hydro': -3.9, 'vol': 168.6, 'charge': 1},
        'M': {'hydro': 1.9, 'vol': 162.9, 'charge': 0},
        'F': {'hydro': 2.8, 'vol': 189.9, 'charge': 0},
        'P': {'hydro': -1.6, 'vol': 112.7, 'charge': 0},
        'S': {'hydro': -0.8, 'vol': 89.0, 'charge': 0},
        'T': {'hydro': -0.7, 'vol': 116.1, 'charge': 0},
        'W': {'hydro': -0.9, 'vol': 227.8, 'charge': 0},
        'Y': {'hydro': -1.3, 'vol': 193.6, 'charge': 0},
        'V': {'hydro': 4.2, 'vol': 140.0, 'charge': 0},
    }
    
    # Parse mutation; multi-site names contribute the SUM of per-site
    # property deltas (matching the canonical encoder) instead of silently
    # discarding every site after the first. Unparseable names raise rather
    # than silently returning a zero vector.
    sites = mutation_name.split('_') if '_' in mutation_name else [mutation_name]
    parsed = []
    for site in sites:
        if len(site) >= 3 and site[1:-1].isdigit():
            parsed.append((site[0], int(site[1:-1]), site[-1]))
        else:
            raise ValueError(f"unparseable mutation name: {mutation_name!r}")
    if not parsed:
        raise ValueError(f"unparseable mutation name: {mutation_name!r}")
    wt_aa, pos, mut_aa = parsed[0]
    wt_props = {'hydro': 0.0, 'vol': 0.0, 'charge': 0.0}
    mut_props = {'hydro': 0.0, 'vol': 0.0, 'charge': 0.0}
    for site_wt, _site_pos, site_mut in parsed:
        w = aa_props.get(site_wt, {'hydro': 0, 'vol': 0, 'charge': 0})
        m = aa_props.get(site_mut, {'hydro': 0, 'vol': 0, 'charge': 0})
        for key in wt_props:
            wt_props[key] += w[key]
            mut_props[key] += m[key]
    
    encoding = np.array([
        (mut_props['hydro'] - wt_props['hydro']) / 5.0,
        (mut_props['vol'] - wt_props['vol']) / 100.0,
        mut_props['charge'] - wt_props['charge'],
        pos / sequence_length,
        0.0,  # Polarity placeholder
    ])
    
    return encoding


def create_expanded_training_data(
    csv_path: Path,
    output_dir: Path,
    n_clusters: int = 3,
) -> Dict:
    """Create expanded training dataset from AF2 predictions.
    
    Args:
        csv_path: Path to AF2 predictions CSV
        output_dir: Output directory
        n_clusters: Number of conformational states
    
    Returns:
        Training data dictionary
    """
    df = load_af2_predictions(csv_path)
    
    # Get unique mutants
    mutants = df['Trial'].unique()
    
    w_list = []
    c_list = []
    w_target_list = []
    mutant_names = []
    
    # WT reference (first entry without mutation name)
    wt_name = 'Abl1'
    wt_df = df[df['Trial'] == wt_name]
    
    if len(wt_df) > 0:
        _, wt_pops = cluster_by_rmsd(wt_df, n_clusters)
    else:
        wt_pops = np.array([0.835, 0.10, 0.065])  # From Source Data
    
    print(f"WT populations: {wt_pops}")
    
    # Process each mutant
    for mutant in mutants:
        if mutant == wt_name:
            continue
        
        mutant_df = df[df['Trial'] == mutant]
        
        # Cluster and get populations
        _, pops = cluster_by_rmsd(mutant_df, n_clusters)
        
        # Parse mutation name
        mut_clean = mutant.replace('Abl1 ', '').replace(' + ', '_')
        
        # Encode mutation
        c = encode_mutation(mut_clean)
        
        # Apply epsilon clipping
        eps = 5e-3
        pops_clipped = np.clip(pops, eps, 1.0)
        pops_clipped = pops_clipped / pops_clipped.sum()
        
        wt_clipped = np.clip(wt_pops, eps, 1.0)
        wt_clipped = wt_clipped / wt_clipped.sum()
        
        w_list.append(wt_clipped)
        c_list.append(c)
        w_target_list.append(pops_clipped)
        mutant_names.append(mut_clean)
        
        print(f"  {mut_clean}: {pops_clipped}")
    
    # Create dataset
    dataset = {
        'w': np.array(w_list),
        'c': np.array(c_list),
        'w_target': np.array(w_target_list),
        'mutations': np.array(mutant_names),
        'n_samples': len(w_list),
        'n_states': n_clusters,
        'mutation_dim': 5,
    }
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / 'abl1_expanded_train.npz', **dataset)
    
    # Save metadata
    meta = {
        'source': 'AF2 predictions (Monteiro da Silva 2024)',
        'n_samples': len(w_list),
        'n_states': n_clusters,
        'mutants': mutant_names,
        'wt_populations': wt_pops.tolist(),
        'note': 'AF2 frequencies used as population proxy',
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    return dataset


def main():
    # Paths
    csv_path = Path('data/external/github/gms_data/prediction_results/'
                    'abl_mut_series_predictions_maxseq256extraseq256.csv')
    output_dir = Path('data/abl1_expanded')
    
    print("="*60)
    print("Expanded Training Data Creation")
    print("="*60)
    print(f"\nSource: {csv_path}")
    
    # Create dataset
    dataset = create_expanded_training_data(csv_path, output_dir, n_clusters=3)
    
    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)
    print(f"Samples: {dataset['n_samples']}")
    print(f"States: {dataset['n_states']}")
    print(f"Mutation dim: {dataset['mutation_dim']}")
    print(f"\nMutants: {dataset['mutations']}")
    
    print(f"\nSaved to: {output_dir}")
    print("  - abl1_expanded_train.npz")
    print("  - metadata.json")
    
    # Statistics
    print("\n" + "="*60)
    print("Population Statistics")
    print("="*60)
    w_targets = dataset['w_target']
    print(f"Ground state: {w_targets[:, 0].mean():.3f} +/- {w_targets[:, 0].std():.3f}")
    print(f"I1 state: {w_targets[:, 1].mean():.3f} +/- {w_targets[:, 1].std():.3f}")
    print(f"I2 state: {w_targets[:, 2].mean():.3f} +/- {w_targets[:, 2].std():.3f}")


if __name__ == '__main__':
    main()
