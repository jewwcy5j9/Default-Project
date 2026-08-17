"""
ΔΔG Validation Pipeline for CDST

This script:
1. Loads experimental ΔΔG data from BioEmu-benchmarks
2. Uses CDST to predict population changes
3. Converts population changes to ΔΔG
4. Compares predictions with experimental values

Key formula:
    ΔΔG = -RT ln(K_mut / K_wt)
    where K = w_active / w_inactive (equilibrium constant)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from scipy import stats
import json


# Constants
R = 1.987e-3  # kcal/(mol·K)
T = 300.0  # K (physiological temperature)
RT = R * T  # ~0.596 kcal/mol


def load_bioemu_ddg_data(csv_path: Path) -> pd.DataFrame:
    """Load BioEmu experimental ΔΔG data.
    
    Args:
        csv_path: Path to system_info.csv
    
    Returns:
        DataFrame with columns: name, name_wt, mutant, dg_exp, ddg_exp, sequence
    """
    df = pd.read_csv(csv_path)
    return df


def populations_to_ddg(
    w_wt: np.ndarray,
    w_mut: np.ndarray,
    active_state: int = 0,
    inactive_state: int = 1,
) -> float:
    """Convert population changes to ΔΔG.
    
    ΔΔG = -RT ln(K_mut / K_wt)
    K = w_active / w_inactive
    
    Args:
        w_wt: Wild-type populations (K,)
        w_mut: Mutant populations (K,)
        active_state: Index of active/ground state
        inactive_state: Index of inactive state
    
    Returns:
        ΔΔG in kcal/mol
    """
    # Equilibrium constants
    K_wt = w_wt[active_state] / (w_wt[inactive_state] + 1e-10)
    K_mut = w_mut[active_state] / (w_mut[inactive_state] + 1e-10)
    
    # ΔΔG
    ddg = -RT * np.log(K_mut / (K_wt + 1e-10))
    return ddg


def ddg_to_populations(
    ddg: float,
    w_wt: np.ndarray,
    active_state: int = 0,
    inactive_state: int = 1,
) -> np.ndarray:
    """Convert ΔΔG to mutant populations (inverse of above).
    
    Given ΔΔG and WT populations, estimate mutant populations.
    
    Args:
        ddg: Experimental ΔΔG (kcal/mol)
        w_wt: Wild-type populations (K,)
        active_state: Index of active state
        inactive_state: Index of inactive state
    
    Returns:
        Estimated mutant populations (K,)
    """
    # K_mut = K_wt * exp(-ΔΔG / RT)
    K_wt = w_wt[active_state] / (w_wt[inactive_state] + 1e-10)
    K_mut = K_wt * np.exp(-ddg / RT)
    
    # Solve for w_mut given K_mut and sum(w) = 1
    # w_active / w_inactive = K_mut
    # w_active + w_inactive + w_other = 1
    # For simplicity, assume 2-state system
    w_active = K_mut / (1 + K_mut)
    w_inactive = 1 / (1 + K_mut)
    
    # Distribute remaining population to other states
    w_mut = np.zeros_like(w_wt)
    w_mut[active_state] = w_active
    w_mut[inactive_state] = w_inactive
    
    # Redistribute other states proportionally
    other_states = [i for i in range(len(w_wt)) if i not in [active_state, inactive_state]]
    if other_states:
        w_other_total = 0.0  # Assume no other states in 2-state approximation
        for i in other_states:
            w_mut[i] = w_other_total / len(other_states)
    
    return w_mut


def encode_mutation_from_sequence(
    seq_wt: str,
    seq_mut: str,
) -> np.ndarray:
    """Encode mutation from sequence difference.
    
    Args:
        seq_wt: Wild-type sequence
        seq_mut: Mutant sequence
    
    Returns:
        Mutation encoding (5,)
    """
    # Collect ALL differing residues (a multi-point mutant used to encode
    # only the first difference, silently mislabelling it as single-site);
    # property deltas are summed over the sites, as in the canonical encoder.
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(seq_wt, seq_mut))
             if a != b]

    if not diffs:
        return np.zeros(5)

    pos = diffs[0][0]
    
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
        'X': {'hydro': 0.0, 'vol': 0.0, 'charge': 0},
    }
    
    wt_h = wt_v = wt_c = 0.0
    mut_h = mut_v = mut_c = 0.0
    for _i, a, b in diffs:
        wp = aa_props.get(a, {'hydro': 0, 'vol': 0, 'charge': 0})
        mp = aa_props.get(b, {'hydro': 0, 'vol': 0, 'charge': 0})
        wt_h += wp['hydro']; wt_v += wp['vol']; wt_c += wp['charge']
        mut_h += mp['hydro']; mut_v += mp['vol']; mut_c += mp['charge']

    encoding = np.array([
        (mut_h - wt_h) / 5.0,
        (mut_v - wt_v) / 100.0,
        mut_c - wt_c,
        pos / len(seq_wt),  # Normalized position
        0.0,  # Polarity (placeholder)
    ])

    return encoding


def create_ddg_validation_dataset(
    bioemu_csv: Path,
    output_path: Path,
) -> Dict:
    """Create validation dataset from BioEmu ΔΔG data.
    
    Args:
        bioemu_csv: Path to system_info.csv
        output_path: Path to save processed data
    
    Returns:
        Dictionary with validation data
    """
    df = load_bioemu_ddg_data(bioemu_csv)
    
    # Filter for mutants only
    mutants_df = df[df['mutant'] == True].copy()
    
    # Get wild-type sequences
    wt_sequences = {}
    for _, row in df[df['mutant'] == False].iterrows():
        wt_sequences[row['name']] = row['sequence']
    
    # Process each mutant
    records = []
    n_missing_wt = 0

    for _, row in mutants_df.iterrows():
        name = row['name']
        name_wt = row['name_wt']
        ddg_exp = row['ddg_exp']
        seq_mut = row['sequence']

        # Get WT sequence
        seq_wt = wt_sequences.get(name_wt, '')
        if not seq_wt:
            n_missing_wt += 1
            continue
        
        # Encode mutation
        c = encode_mutation_from_sequence(seq_wt, seq_mut)
        
        # Estimate populations from ΔΔG (for validation)
        # Assume 2-state: Active (ground) and Inactive
        # WT: assume 90% active, 10% inactive (typical for stable proteins)
        w_wt = np.array([0.9, 0.1])
        w_mut_est = ddg_to_populations(ddg_exp, w_wt)
        
        records.append({
            'name': name,
            'name_wt': name_wt,
            'ddg_exp': ddg_exp,
            'dg_exp': row['dg_exp'],
            'sequence': seq_mut,
            'mutation_encoding': c.tolist(),
            'w_wt': w_wt.tolist(),
            'w_mut_est': w_mut_est.tolist(),
        })
    
    result = {
        'records': records,
        'n_samples': len(records),
        'n_skipped_missing_wt': n_missing_wt,
        'ddg_values': [r['ddg_exp'] for r in records],
    }
    if n_missing_wt:
        print(f"[ddg_validation] skipped {n_missing_wt} mutant(s) with no "
              "matching WT sequence; n_samples covers survivors only")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    return result


def evaluate_ddg_predictions(
    ddg_exp: np.ndarray,
    ddg_pred: np.ndarray,
) -> Dict:
    """Evaluate ΔΔG predictions.
    
    NOTE: This is a BASELINE IMPLEMENTATION exercise, NOT the core CDST task.
    CDST's core task is K-dimensional population shift prediction.
    ΔΔG (scalar stability) ≠ ΔG_k (per-state free energy changes).
    
    Args:
        ddg_exp: Experimental ΔΔG values
        ddg_pred: Predicted ΔΔG values
    
    Returns:
        Dictionary with evaluation metrics
    """
    # MAE
    mae = np.mean(np.abs(ddg_exp - ddg_pred))
    
    # RMSE
    rmse = np.sqrt(np.mean((ddg_exp - ddg_pred) ** 2))
    
    # Pearson correlation
    if len(ddg_exp) > 2 and np.std(ddg_pred) > 1e-8:
        pearson_r, pearson_p = stats.pearsonr(ddg_exp, ddg_pred)
    else:
        pearson_r, pearson_p = 0.0, 1.0
    
    # Spearman correlation
    if len(ddg_exp) > 2:
        spearman_r, spearman_p = stats.spearmanr(ddg_exp, ddg_pred)
    else:
        spearman_r, spearman_p = 0.0, 1.0
    
    # Direction accuracy (sign agreement)
    direction_correct = np.sign(ddg_exp) == np.sign(ddg_pred)
    direction_acc = np.mean(direction_correct)
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'direction_accuracy': float(direction_acc),
        'n_samples': len(ddg_exp),
    }


def compute_trivial_baselines(ddg_exp: np.ndarray) -> Dict:
    """Compute trivial baseline metrics for comparison.
    
    Baselines:
    1. Predict mean: always predict the mean ΔΔG
    2. Predict zero: always predict 0 (no change)
    3. Predict majority class: always predict the more common direction
    
    Args:
        ddg_exp: Experimental ΔΔG values
    
    Returns:
        Dictionary with baseline metrics
    """
    n = len(ddg_exp)
    mean_ddg = np.mean(ddg_exp)
    
    # Baseline 1: Predict mean
    pred_mean = np.full(n, mean_ddg)
    mae_mean = np.mean(np.abs(ddg_exp - pred_mean))
    
    # Baseline 2: Predict zero
    pred_zero = np.zeros(n)
    mae_zero = np.mean(np.abs(ddg_exp - pred_zero))
    
    # Baseline 3: Predict majority class direction
    n_stabilizing = np.sum(ddg_exp < 0)
    n_destabilizing = np.sum(ddg_exp > 0)
    majority_direction = 1 if n_destabilizing > n_stabilizing else -1
    pred_majority = np.full(n, majority_direction * np.mean(np.abs(ddg_exp)))
    mae_majority = np.mean(np.abs(ddg_exp - pred_majority))
    
    # Direction accuracy for majority class predictor
    majority_dir_acc = max(n_stabilizing, n_destabilizing) / n
    
    return {
        'predict_mean': {
            'mae': float(mae_mean),
            'description': f'Always predict mean ({mean_ddg:.3f})',
        },
        'predict_zero': {
            'mae': float(mae_zero),
            'description': 'Always predict 0 (no change)',
        },
        'predict_majority': {
            'mae': float(mae_majority),
            'direction_accuracy': float(majority_dir_acc),
            'description': f'Always predict majority direction ({"destabilizing" if majority_direction > 0 else "stabilizing"})',
            'n_stabilizing': int(n_stabilizing),
            'n_destabilizing': int(n_destabilizing),
        },
    }


def main():
    # Paths
    bioemu_csv = Path('data/external/bioemu-benchmarks/bioemu_benchmarks/assets/'
                      'folding_free_energies_benchmark_0.1/folding_free_energies/system_info.csv')
    output_path = Path('data/ddg_validation/ddg_dataset.json')
    
    print("="*60)
    print("ΔΔG Validation Pipeline (BASELINE IMPLEMENTATION)")
    print("="*60)
    print("\nNOTE: This is NOT the core CDST task.")
    print("CDST's core task is K-dimensional population shift prediction.")
    print("ΔΔG (scalar stability) ≠ ΔG_k (per-state free energy changes).")
    
    # Create validation dataset
    print(f"\nLoading BioEmu ΔΔG data from: {bioemu_csv}")
    result = create_ddg_validation_dataset(bioemu_csv, output_path)
    
    print(f"\nDataset created:")
    print(f"  Samples: {result['n_samples']}")
    print(f"  ΔΔG range: [{min(result['ddg_values']):.2f}, {max(result['ddg_values']):.2f}] kcal/mol")
    print(f"  Mean ΔΔG: {np.mean(result['ddg_values']):.2f} kcal/mol")
    
    # Statistics
    ddg_values = np.array(result['ddg_values'])
    print(f"\nΔΔG Statistics:")
    print(f"  Stabilizing (ΔΔG < 0): {np.sum(ddg_values < 0)}")
    print(f"  Destabilizing (ΔΔG > 0): {np.sum(ddg_values > 0)}")
    print(f"  Neutral (|ΔΔG| < 0.5): {np.sum(np.abs(ddg_values) < 0.5)}")
    
    print(f"\nDataset saved to: {output_path}")
    
    # Compute trivial baselines
    print("\n" + "="*60)
    print("Trivial Baselines (MUST report for context)")
    print("="*60)
    
    baselines = compute_trivial_baselines(ddg_values)
    
    print(f"\n  {'Baseline':<25} {'MAE':<15} {'Dir Acc':<15}")
    print(f"  {'-'*55}")
    print(f"  {'Predict mean':<25} {baselines['predict_mean']['mae']:<15.3f} {'-':<15}")
    print(f"  {'Predict zero':<25} {baselines['predict_zero']['mae']:<15.3f} {'-':<15}")
    print(f"  {'Predict majority':<25} {baselines['predict_majority']['mae']:<15.3f} {baselines['predict_majority']['direction_accuracy']:<15.1%}")
    
    print(f"\n  Majority class: {baselines['predict_majority']['description']}")
    print(f"    Stabilizing: {baselines['predict_majority']['n_stabilizing']}")
    print(f"    Destabilizing: {baselines['predict_majority']['n_destabilizing']}")
    
    # Example: Evaluate a simple baseline (predict 0 for all)
    print("\n" + "="*60)
    print("Model Evaluation (for reference only)")
    print("="*60)
    
    ddg_pred_baseline = np.zeros_like(ddg_values)
    metrics = evaluate_ddg_predictions(ddg_values, ddg_pred_baseline)
    
    print(f"\n  Predict 0 Metrics:")
    print(f"    MAE: {metrics['mae']:.3f} kcal/mol")
    print(f"    RMSE: {metrics['rmse']:.3f} kcal/mol")
    print(f"    Direction Accuracy: {metrics['direction_accuracy']:.1%}")
    
    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)
    print("1. Train CDST on Abl1 population data (CORE TASK)")
    print("2. Evaluate on NMR populations (Figure 6B), NOT ΔΔG")
    print("3. Use ΔΔG only as baseline comparison")
    print("4. BioEmu populations = silver standard (training only)")
    print("5. NMR populations = gold standard (evaluation only)")


if __name__ == '__main__':
    main()
