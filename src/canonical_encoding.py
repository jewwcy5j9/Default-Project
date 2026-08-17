"""
Canonical mutation encoding for CDST project.

This is the SINGLE SOURCE OF TRUTH for mutation encoding.
All scripts should import from here.

Encoding: Extended 10-dim (from encoding_ablation.py)
- Validated MAE: 0.413 on FINAL 6-mutant dataset
"""

import numpy as np

# Amino acid properties: [volume, hydrophobicity, aromaticity, h_bond_donor, h_bond_acceptor, charge]
# Sources: AAIndex, Kyte-Doolittle, standard biochemistry
AA_PROPERTIES = {
    'F': [135, 2.8, 1.0, 0.0, 0.0, 0.0],   # Phe: large, hydrophobic, aromatic
    'L': [124, 3.8, 0.0, 0.0, 0.0, 0.0],   # Leu: large, hydrophobic, aliphatic
    'Y': [141, -1.3, 1.0, 1.0, 1.0, 0.0],  # Tyr: large, aromatic, H-bond
    'V': [105, 4.2, 0.0, 0.0, 0.0, 0.0],   # Val: medium, hydrophobic, aliphatic
    'M': [124, 1.9, 0.0, 0.0, 0.0, 0.0],   # Met: large, sulfur
    'I': [126, 4.5, 0.0, 0.0, 0.0, 0.0],   # Ile: large, hydrophobic
}

# Mutation definitions (position in Abl1b numbering, WT AA, Mutant AA)
MUTATION_DEFS = {
    'M290L': {'pos': 290, 'wt': 'M', 'mut': 'L'},
    'L301I': {'pos': 301, 'wt': 'L', 'mut': 'I'},
    'M290L_L301I': {'pos': 301, 'wt': 'ML', 'mut': 'LI'},  # Double mutant
    'F382L': {'pos': 382, 'wt': 'F', 'mut': 'L'},
    'F382Y': {'pos': 382, 'wt': 'F', 'mut': 'Y'},
    'F382V': {'pos': 382, 'wt': 'F', 'mut': 'V'},
}


def encode_mutation(name):
    """
    Encode mutation using Extended 10-dim scheme.
    
    Features:
    [0]: position (normalized by 534)
    [1-6]: AA property deltas (volume, hydrophobicity, aromaticity, hbd, hba, charge) / 5.0
    [7]: double mutant flag
    [8]: position 290 flag
    [9]: position 301 flag
    
    Args:
        name: Mutation name (e.g., 'M290L', 'F382Y', 'M290L_L301I')
    
    Returns:
        np.ndarray of shape (10,)
    """
    if name not in MUTATION_DEFS:
        raise ValueError(f"Unknown mutation: {name}. Available: {list(MUTATION_DEFS.keys())}")
    
    data = MUTATION_DEFS[name]
    enc = np.zeros(10)
    
    # Position
    enc[0] = data['pos'] / 534
    
    # AA property deltas: multi-site names carry concatenated wt/mut strings
    # ('ML'/'LI'); sum the per-residue deltas exactly like the canonical
    # pipeline encoder (encoding_ablation_control._aa_delta), so this module
    # agrees with the frozen canonical numbers instead of silently zeroing
    # the double mutant's chemistry.
    delta = np.zeros(6)
    for w, m in zip(data['wt'], data['mut']):
        if w in AA_PROPERTIES and m in AA_PROPERTIES:
            delta += (np.array(AA_PROPERTIES[m]) - np.array(AA_PROPERTIES[w]))
    enc[1:7] = delta / 5.0  # Normalize

    # Double mutant flag
    if '_' in name:
        enc[7] = 1.0

    # Position-specific flags
    if data['pos'] == 290:
        enc[8] = 1.0
    elif data['pos'] == 301:
        enc[9] = 1.0

    return enc


def encode_all_mutations():
    """Encode all defined mutations."""
    return {name: encode_mutation(name) for name in MUTATION_DEFS}


# Canonical CDST predictions (Extended encoding, from encoding_ablation.json)
CDST_EXTENDED_PREDICTIONS = {
    'M290L': 0.20486879348754883,
    'L301I': 0.9287528991699219,
    'M290L_L301I': 0.671317994594574,
    'F382L': 0.8859054446220398,
    'F382Y': 0.7288229465484619,
    'F382V': 0.07946029305458069,
}

# Canonical metrics (aligned with data/canonical_results.json, the SINGLE authority)
# Direction excludes ties: F382L is a tie (|NMR-WT| < 5%), scored over 5 non-tie mutants.
CDST_EXTENDED_MAE = 0.41336471935113267
CDST_EXTENDED_DIRECTION = 0.8  # 4/5 = 80% (M290L, L301I, M290L_L301I, F382Y correct; F382V wrong; F382L tie excluded)


if __name__ == '__main__':
    print("Canonical Mutation Encoding")
    print("=" * 50)
    
    for name in MUTATION_DEFS:
        enc = encode_mutation(name)
        print(f"\n{name}:")
        print(f"  Encoding: {enc}")
        print(f"  CDST prediction: {CDST_EXTENDED_PREDICTIONS.get(name, 'N/A'):.3f}")
    
    print(f"\nCanonical MAE: {CDST_EXTENDED_MAE:.4f}")
    print(f"Canonical Direction: {CDST_EXTENDED_DIRECTION:.1%}")
