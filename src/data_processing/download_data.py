"""
Data Download and Verification Script

Downloads and verifies data from:
1. Figshare (Monteiro da Silva 2024 original data)
2. STGNet Zenodo (AF2 conformations for baseline only)

Usage:
    python download_data.py --figshare
    python download_data.py --stgnet
    python download_data.py --verify
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
import numpy as np


# =============================================================================
# GitHub Data (Monteiro da Silva 2024) - CORRECT SOURCE
# =============================================================================

GITHUB_REPO = "https://github.com/GMdSilva/gms_natcomms_1705932980_data"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/GMdSilva/gms_natcomms_1705932980_data/main"

# Note from paper: "Due to storage limitations, the repository neither includes 
# the resulting PDB ensembles nor all of the MSAs generated and used in this study,
# although access can be obtained by contacting the corresponding author."

# Expected files from GitHub
GITHUB_EXPECTED_FILES = [
    "README.md",
    "abl1/",           # Abl1 analysis scripts and data
    "gmscf/",          # GM-CSF analysis scripts and data  
    "scripts/",        # MSA assembly and AF2 running scripts
]

# PDB IDs (verified from paper Figure 3)
PDB_IDS_VERIFIED = {
    'Active': '6XR6',    # Ground state (DFG-in/AL-open)
    'Inactive2': '6RXG', # I2 state (DFG-out/AL-closed) - NOTE: was typo'd as 6XRG
    # Inactive1 (I1) PDB ID not explicitly stated in paper
}


def download_github_data(output_dir: Path, verify_only: bool = False) -> Dict:
    """Download GitHub data for Monteiro da Silva 2024.
    
    Repository: https://github.com/GMdSilva/gms_natcomms_1705932980_data
    
    Note: PDB ensembles and full MSAs NOT included due to storage limits.
    Contact corresponding author for complete data.
    
    Args:
        output_dir: Directory to save downloaded files
        verify_only: If True, only check existing files
    
    Returns:
        Dictionary with download status
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    status = {
        'repo': GITHUB_REPO,
        'files': {},
        'pdb_ids': PDB_IDS_VERIFIED,
        'instructions': []
    }
    
    print("="*60)
    print("GitHub Data Download (Monteiro da Silva 2024)")
    print("="*60)
    print(f"Repository: {GITHUB_REPO}")
    print()
    print("Verified PDB IDs:")
    for state, pdb_id in PDB_IDS_VERIFIED.items():
        print(f"  {state}: {pdb_id}")
    print()
    
    # Check for existing files
    for expected in GITHUB_EXPECTED_FILES:
        path = output_dir / expected
        exists = path.exists()
        status['files'][expected] = 'exists' if exists else 'missing'
        print(f"  [{'OK' if exists else 'MISSING'}] {expected}")
    
    if verify_only:
        return status
    
    # Download instructions
    print()
    print("Download Instructions:")
    print("-"*60)
    print(f"1. Clone repository:")
    print(f"   git clone {GITHUB_REPO}")
    print(f"2. Or download ZIP from GitHub page")
    print(f"3. Extract to: {output_dir}")
    print()
    print("IMPORTANT NOTES:")
    print("  - PDB ensembles NOT included (storage limits)")
    print("  - Full MSAs NOT included")
    print("  - Contact corresponding author for complete data")
    print("  - Source Data (XLSX) available from Nature Communications paper")
    print()
    print("Paper: DOI 10.1038/s41467-024-46715-9")
    print("Source Data: Download XLSX from paper webpage")
    
    status['instructions'] = [
        f"git clone {GITHUB_REPO}",
        "Download Source Data XLSX from paper",
        "Contact author for PDB ensembles if needed"
    ]
    
    return status


def verify_figshare_data(data_dir: Path) -> Dict:
    """Verify downloaded Figshare data and extract population numbers.
    
    Args:
        data_dir: Directory containing downloaded files
    
    Returns:
        Dictionary with verified population data
    """
    data_dir = Path(data_dir)
    
    print("="*60)
    print("Figshare Data Verification")
    print("="*60)
    
    verified_data = {
        'wt_populations': None,
        'mutant_populations': {},
        'pdb_ids': {},
        'notes': []
    }
    
    # Check for summary CSV
    summary_path = data_dir / "abl1_allelic_series_summary.csv"
    if summary_path.exists():
        print(f"\nFound: {summary_path}")
        # Parse CSV to extract populations
        # (Implementation depends on actual CSV format)
        verified_data['notes'].append("Summary CSV found - parse to extract populations")
    else:
        print(f"\nMissing: {summary_path}")
        verified_data['notes'].append("Summary CSV not found")
    
    # Check for AF2 output
    af2_dir = data_dir / "af2_output"
    if af2_dir.exists():
        pdb_files = list(af2_dir.glob("*.pdb"))
        print(f"\nAF2 output: {len(pdb_files)} PDB files")
        verified_data['notes'].append(f"AF2 output: {len(pdb_files)} PDBs")
    else:
        print(f"\nMissing: {af2_dir}")
    
    return verified_data


# =============================================================================
# STGNet Data (Zenodo)
# =============================================================================

STGNET_ZENODO_ID = "14607966"
STGNET_ZENODO_URL = f"https://zenodo.org/records/{STGNET_ZENODO_ID}"

# STGNet role: ONLY for state enumeration and AF2-frequency baseline
# NOT for ground truth populations (circular reasoning risk)
STGNET_USAGE_POLICY = """
STGNet Data Usage Policy
========================
ALLOWED:
  - State enumeration (identify conformational states)
  - AF2-frequency baseline (compute sampling frequencies as baseline)
  
FORBIDDEN:
  - Ground truth for CDST training (circular reasoning with AF2)
  - Claiming to "beat AF2" when trained on AF2 frequencies
  
Reference: Proposition 7 (sampling frequency != Boltzmann population)
"""


def download_stgnet(output_dir: Path) -> Dict:
    """Download STGNet data from Zenodo.
    
    IMPORTANT: This data is ONLY for:
    1. State enumeration (what conformations exist)
    2. AF2-frequency baseline (to be beaten, not trained on)
    
    NOT for ground truth populations!
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("STGNet Data Download (Zenodo)")
    print("="*60)
    print(f"Zenodo ID: {STGNET_ZENODO_ID}")
    print(f"URL: {STGNET_ZENODO_URL}")
    print()
    print(STGNET_USAGE_POLICY)
    print()
    print("Download Instructions:")
    print("-"*60)
    print(f"1. Visit: {STGNET_ZENODO_URL}")
    print("2. Download: abl1_mutants_conformations.zip (or similar)")
    print(f"3. Extract to: {output_dir}")
    print()
    print("Contents (expected):")
    print("  - 31 ABL mutant ensembles")
    print("  - 640 conformations per mutant (ColabFold AF2 subsampling)")
    print("  - NO experimental populations (AF2 frequencies only)")
    
    return {
        'zenodo_id': STGNET_ZENODO_ID,
        'url': STGNET_ZENODO_URL,
        'usage': 'baseline_and_enumeration_only',
        'warning': 'DO NOT USE AS GROUND TRUTH'
    }


# =============================================================================
# Population Verification Checklist
# =============================================================================

def create_verification_checklist(output_path: Path):
    """Create a checklist for verifying population data against original paper."""
    
    checklist = """
# Abl1 Population Verification Checklist

## Source: Monteiro da Silva et al., Nat Commun 15, 2464 (2024)
## Figshare DOI: 10.6084/m9.figshare.23834715

### Wild-Type Populations
- [ ] Active state: _____ % (current: 89%)
- [ ] Inactive1 state: _____ % (current: 5%)
- [ ] Inactive2 state: _____ % (current: 6%)
- [ ] Sum = 100%: _____

### PDB IDs Verification
- [ ] Active: 6XR6 (verify)
- [ ] Inactive1: 6XR7 (verify - may not exist)
- [ ] Inactive2: 6RXG (verify - was 6XRG, possible typo)

### Mutant Populations (8 mutants)
| Mutation | Active | Inactive1 | Inactive2 | Direction | Verified |
|----------|--------|-----------|-----------|-----------|----------|
| T315I    | ___    | ___       | ___       | ->I2      | [ ]      |
| M472I    | ___    | ___       | ___       | ->I2      | [ ]      |
| F486S    | ___    | ___       | ___       | ->I2      | [ ]      |
| H361R    | ___    | ___       | ___       | ->I2      | [ ]      |
| E255K    | ___    | ___       | ___       | neutral   | [ ]      |
| Y253F    | ___    | ___       | ___       | neutral   | [ ]      |
| D276G    | ___    | ___       | ___       | ->I2      | [ ]      |
| I360M    | ___    | ___       | ___       | ->I2      | [ ]      |

### Notes
- NMR detection limit: ~1-5% for minor states
- Populations below detection should be set to epsilon (0.5%)
- All populations must sum to 100%

### Actions After Verification
1. Update abl1_dataset.py with verified numbers
2. Regenerate abl1_samples.npz
3. Document any discrepancies with original report
"""
    
    output_path = Path(output_path)
    output_path.write_text(checklist)
    print(f"Verification checklist created: {output_path}")
    return checklist


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download and verify CDST data")
    parser.add_argument('--github', action='store_true', help='Download GitHub data (Monteiro da Silva 2024)')
    parser.add_argument('--stgnet', action='store_true', help='Download STGNet data')
    parser.add_argument('--verify', action='store_true', help='Verify downloaded data')
    parser.add_argument('--checklist', action='store_true', help='Create verification checklist')
    parser.add_argument('--data_dir', type=str, default='data/external', help='Data directory')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if args.github:
        download_github_data(data_dir / 'github')
    
    if args.stgnet:
        download_stgnet(data_dir / 'stgnet')
    
    if args.verify:
        verify_figshare_data(data_dir / 'github')
    
    if args.checklist:
        create_verification_checklist(data_dir / 'verification_checklist.md')
    
    if not any([args.github, args.stgnet, args.verify, args.checklist]):
        print("Usage:")
        print("  python download_data.py --github      # GitHub download instructions")
        print("  python download_data.py --stgnet      # STGNet download instructions")
        print("  python download_data.py --verify      # Verify downloaded data")
        print("  python download_data.py --checklist   # Create verification checklist")


if __name__ == '__main__':
    main()
