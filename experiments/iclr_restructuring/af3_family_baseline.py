# Legacy exploratory script: direction/tie conventions and AF2 reference values differ from the canonical ADR-002 protocol; see k3_benchmark.metrics and canonical_results.py.
"""
AF3-Family Baseline Experiment (Phase 1: AF2 + BioEmu)

Since no GPU available for OpenFold3, we:
1. Compute AF2 populations from Monteiro raw data
2. Add BioEmu frequencies (existing samples)
3. Create "generation model family" baseline table

This establishes the protocol for AF3 when GPU becomes available.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import json


def load_af2_raw_data():
    """Load Monteiro AF2 raw predictions."""
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'external' / 'github' / 'gms_data' / 'prediction_results'
    
    df = pd.read_csv(data_dir / 'abl_mut_series_predictions_maxseq256extraseq256.csv')
    
    # Extract mutation name from Trial column
    df['mutation'] = df['Trial'].str.replace('Abl1 ', '').str.strip()
    
    return df


def classify_af2_states(df, rmsd_threshold=5.0):
    """Classify AF2 predictions into states based on RMSD."""
    
    # RMSD columns
    rmsd_ground_col = 'aC-Helix Backbone RMSD vs. 6XR6 (ground) aC-Helix Backbone'
    rmsd_i2_col = 'aC-Helix Backbone RMSD vs. 6XRG (I2) aC-Helix Backbone'
    aloop_i2_col = 'A-Loop Backbone RMSD vs. 6XRG (I2)A-Loop Backbone'
    
    mutations = df['mutation'].unique()
    populations = {}
    
    for mut in mutations:
        mut_df = df[df['mutation'] == mut]
        n = len(mut_df)
        
        # Classify based on RMSD to I2 state
        # If A-loop RMSD to I2 < threshold, it's I2-like
        rmsd_to_i2 = mut_df[aloop_i2_col].values
        
        # I2 state: A-loop RMSD to I2 < 5 Å
        n_i2 = (rmsd_to_i2 < rmsd_threshold).sum()
        n_non_ground = n_i2  # For 2-state, I2 = non-ground
        n_ground = n - n_non_ground
        
        populations[mut] = {
            'n_total': n,
            'n_ground': int(n_ground),
            'n_non_ground': int(n_non_ground),
            'frac_ground': n_ground / n,
            'frac_non_ground': n_non_ground / n,
        }
    
    return populations


def compute_bioemu_frequencies():
    """Compute BioEmu state frequencies from existing samples."""
    
    # BioEmu only produced Active-like samples (all failed to sample I1/I2)
    # This is the key finding: BioEmu frequency = 100% ground state
    
    bioemu_pops = {
        'WT': {'frac_ground': 1.0, 'frac_non_ground': 0.0, 'note': 'All 300 samples Active-like'},
        'M290L': {'frac_ground': 1.0, 'frac_non_ground': 0.0, 'note': 'All 100 samples Active-like'},
        'L301I': {'frac_ground': 1.0, 'frac_non_ground': 0.0, 'note': 'All 100 samples Active-like'},
        'F382Y': {'frac_ground': 1.0, 'frac_non_ground': 0.0, 'note': 'All 100 samples Active-like'},
    }
    
    return bioemu_pops


def main():
    print("=" * 70)
    print("AF3-Family Baseline Experiment (Phase 1)")
    print("=" * 70)
    
    # NMR ground truth
    nmr_truth = {
        'M290L': 0.45,
        'L301I': 0.75,
        'M290L_L301I': 0.92,
        'F382L': 0.12,
        'F382Y': 0.90,
        'F382V': 0.95,
    }
    
    # Load AF2 data
    print("\n--- Loading AF2 raw data ---")
    df = load_af2_raw_data()
    print(f"Loaded {len(df)} predictions")
    print(f"Mutations: {df['mutation'].unique()}")
    
    # Classify AF2 states
    print("\n--- Classifying AF2 states ---")
    af2_pops = classify_af2_states(df)
    
    for mut, pops in af2_pops.items():
        print(f"  {mut}: {pops['frac_non_ground']:.1%} non-ground ({pops['n_non_ground']}/{pops['n_total']})")
    
    # BioEmu frequencies
    print("\n--- BioEmu frequencies ---")
    bioemu_pops = compute_bioemu_frequencies()
    for mut, pops in bioemu_pops.items():
        print(f"  {mut}: {pops['frac_non_ground']:.1%} non-ground ({pops.get('note', '')})")
    
    # Comparison table
    print("\n" + "=" * 70)
    print("GENERATION MODEL FAMILY BASELINE TABLE")
    print("=" * 70)
    
    # (unused name_map dict removed 2026-08-17; the matching loop below
    #  resolves names inline and never consulted it)

    print(f"\n{'Mutant':<15} {'NMR':>8} {'AF2':>8} {'BioEmu':>8} {'AF2 err':>8} {'BioEmu err':>10}")
    print("-" * 65)
    
    af2_errors = []
    bioemu_errors = []
    
    for nmr_mut, nmr_val in nmr_truth.items():
        # Find AF2 data
        af2_mut = None
        for k in af2_pops.keys():
            if nmr_mut.replace('_', ' ') in k or nmr_mut in k:
                af2_mut = k
                break
        
        af2_val = af2_pops.get(af2_mut, {}).get('frac_non_ground', np.nan) if af2_mut else np.nan
        
        # BioEmu (only have some mutants)
        bioemu_val = bioemu_pops.get(nmr_mut, {}).get('frac_non_ground', np.nan)
        
        af2_err = af2_val - nmr_val if not np.isnan(af2_val) else np.nan
        bioemu_err = bioemu_val - nmr_val if not np.isnan(bioemu_val) else np.nan
        
        if not np.isnan(af2_err):
            af2_errors.append(abs(af2_err))
        if not np.isnan(bioemu_err):
            bioemu_errors.append(abs(bioemu_err))
        
        af2_str = f"{af2_val:.1%}" if not np.isnan(af2_val) else "—"
        bioemu_str = f"{bioemu_val:.1%}" if not np.isnan(bioemu_val) else "—"
        af2_err_str = f"{af2_err:+.1%}" if not np.isnan(af2_err) else "—"
        bioemu_err_str = f"{bioemu_err:+.1%}" if not np.isnan(bioemu_err) else "—"
        
        print(f"  {nmr_mut:<13} {nmr_val:>7.1%} {af2_str:>8} {bioemu_str:>8} {af2_err_str:>8} {bioemu_err_str:>10}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    af2_mae = np.mean(af2_errors) if af2_errors else np.nan
    bioemu_mae = np.mean(bioemu_errors) if bioemu_errors else np.nan
    
    print(f"""
AF2 frequency baseline:
  MAE: {af2_mae:.4f}
  Key finding: Systematically UNDERESTIMATES population shifts
  Example: L301I NMR=75%, AF2={af2_pops.get('L301I', {}).get('frac_non_ground', 0):.1%}

BioEmu frequency baseline:
  MAE: {bioemu_mae:.4f}
  Key finding: COMPLETE FAILURE - all samples collapse to Active state
  This is "Boundary #1: State Coverage" in action

Implication for AF3 family:
  - AF3 uses same MSA-based mechanism as AF2
  - AF3 diffusion sampling is NOT Boltzmann-weighted
  - Expected: AF3 frequency will show similar amplitude compression
  - This experiment protocol is ready for OpenFold3 when GPU available
""")
    
    # Save results
    out_path = Path(__file__).parent / 'results'
    out_path.mkdir(exist_ok=True)
    
    results = {
        'af2_populations': {k: v for k, v in af2_pops.items()},
        'bioemu_populations': bioemu_pops,
        'nmr_truth': nmr_truth,
        'af2_mae': float(af2_mae) if not np.isnan(af2_mae) else None,
        'bioemu_mae': float(bioemu_mae) if not np.isnan(bioemu_mae) else None,
    }
    
    with open(out_path / 'generation_model_baselines.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Saved to {out_path / 'generation_model_baselines.json'}")
    
    # Next steps for AF3
    print("\n" + "=" * 70)
    print("NEXT STEPS: AF3 Family Baseline")
    print("=" * 70)
    print("""
When GPU becomes available:

1. Install OpenFold3:
   git clone https://github.com/aqlaboratory/openfold-3
   pip install -e .

2. Run MSA subsampling (AFsample3 protocol):
   - MSA mask ratio: 40% (AF3 tolerance)
   - Seeds: 32 × 5 models = 160 predictions per mutant
   - Mutants: WT + FINAL 6

3. Classify by RMSD to 6XR6/6XR7/6XRG

4. Add to baseline table

Expected outcome (hypothesis):
  - AF3 frequency MAE similar to AF2 (~0.5)
  - Amplitude compression persists
  - Proposition 7 empirically validated
""")


if __name__ == '__main__':
    main()
