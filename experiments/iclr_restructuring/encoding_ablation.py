"""
Encoding Ablation: Test if richer amino acid features help F382 series

Compare:
1. Baseline encoding (position + simple type)
2. Extended encoding (position + volume + hydrophobicity + aromaticity + H-bond)
3. AAIndex encoding (literature-derived physicochemical scales)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import json

from src.models.low_rank_cdst import LowRankCDST


# Amino acid properties (AAIndex-derived, normalized)
AA_PROPERTIES = {
    # [volume, hydrophobicity, aromaticity, h_bond_donor, h_bond_acceptor, charge]
    'F': [135, 2.8, 1.0, 0.0, 0.0, 0.0],   # Phe: large, hydrophobic, aromatic
    'L': [124, 3.8, 0.0, 0.0, 0.0, 0.0],   # Leu: large, hydrophobic, aliphatic
    'Y': [141, -1.3, 1.0, 1.0, 1.0, 0.0],  # Tyr: large, aromatic, H-bond
    'V': [105, 4.2, 0.0, 0.0, 0.0, 0.0],   # Val: medium, hydrophobic, aliphatic
    'M': [124, 1.9, 0.0, 0.0, 0.0, 0.0],   # Met: large, sulfur
    'I': [126, 4.5, 0.0, 0.0, 0.0, 0.0],   # Ile: large, hydrophobic
}

# NMR data
NMR_DATA = {
    'M290L': {'wt': 'M', 'mut': 'L', 'pos': 290, 'non_ground': 0.45},
    'L301I': {'wt': 'L', 'mut': 'I', 'pos': 301, 'non_ground': 0.75},
    'M290L_L301I': {'wt': 'ML', 'mut': 'LI', 'pos': 301, 'non_ground': 0.92},
    'F382L': {'wt': 'F', 'mut': 'L', 'pos': 382, 'non_ground': 0.12},
    'F382Y': {'wt': 'F', 'mut': 'Y', 'pos': 382, 'non_ground': 0.90},
    'F382V': {'wt': 'F', 'mut': 'V', 'pos': 382, 'non_ground': 0.95},
}

WT_NON_GROUND = 0.12
AF2_NON_GROUND = {
    'M290L': 140/480, 'L301I': 109/480, 'M290L_L301I': 79/480,
    'F382L': 63/480, 'F382Y': 64/480, 'F382V': 74/480,
}


def encode_baseline(name, data):
    """Baseline: position + simple flags."""
    enc = np.zeros(5)
    enc[0] = data['pos'] / 534
    if 'M290L' in name:
        enc[1] = 1.0
    if 'L301I' in name:
        enc[2] = 1.0
    if 'F382' in name:
        enc[3] = 1.0
        if 'Y' in name:
            enc[4] = 1.0
        elif 'V' in name:
            enc[4] = -1.0
    return enc


def encode_extended(name, data):
    """Extended: position + AA property deltas."""
    enc = np.zeros(10)
    enc[0] = data['pos'] / 534
    
    # Get AA properties
    wt_aa = data['wt'][0] if len(data['wt']) == 1 else data['wt']
    mut_aa = data['mut'][0] if len(data['mut']) == 1 else data['mut']
    
    if wt_aa in AA_PROPERTIES and mut_aa in AA_PROPERTIES:
        wt_props = np.array(AA_PROPERTIES[wt_aa])
        mut_props = np.array(AA_PROPERTIES[mut_aa])
        delta = mut_props - wt_props  # [volume, hydrophobicity, aromaticity, hbd, hba, charge]
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


def encode_aaindex(name, data):
    """AAIndex: literature-derived scales (simplified)."""
    # Use 3 key scales: size, hydrophobicity, polarity
    enc = np.zeros(8)
    enc[0] = data['pos'] / 534
    
    wt_aa = data['wt'][0]
    mut_aa = data['mut'][0]
    
    if wt_aa in AA_PROPERTIES and mut_aa in AA_PROPERTIES:
        wt = AA_PROPERTIES[wt_aa]
        mut = AA_PROPERTIES[mut_aa]
        
        # Size difference (volume)
        enc[1] = (mut[0] - wt[0]) / 50.0
        # Hydrophobicity difference (Kyte-Doolittle)
        enc[2] = (mut[1] - wt[1]) / 5.0
        # Aromaticity change
        enc[3] = mut[2] - wt[2]
        # H-bond capacity change
        enc[4] = (mut[3] + mut[4]) - (wt[3] + wt[4])
        # Polarity (derived)
        enc[5] = enc[4] * 0.5 + enc[3] * 0.3
    
    if '_' in name:
        enc[6] = 1.0
    
    return enc


def run_experiment(encoder, encoder_name, n_seeds=5):
    """Run LOSO experiment with given encoder."""
    mutations = list(NMR_DATA.keys())
    n = len(mutations)
    
    # Prepare data
    wt = np.array([1 - WT_NON_GROUND, WT_NON_GROUND])
    w_wt = np.tile(wt, (n, 1))
    w_target = np.array([[1 - NMR_DATA[m]['non_ground'], NMR_DATA[m]['non_ground']] for m in mutations])
    c = np.array([encoder(m, NMR_DATA[m]) for m in mutations])
    d = c.shape[1]
    
    all_preds = {m: [] for m in mutations}
    
    for seed in range(n_seeds):
        for hold_out in range(n):
            mask = np.ones(n, dtype=bool)
            mask[hold_out] = False
            
            torch.manual_seed(seed * 100 + hold_out)
            model = LowRankCDST(K=2, intervention_dim=d, rank=2, hidden_dim=32)
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
            
            w_t = torch.FloatTensor(w_wt[mask])
            c_t = torch.FloatTensor(c[mask])
            wt_t = torch.FloatTensor(w_target[mask])
            
            best_loss = float('inf')
            best_state = None
            
            for epoch in range(800):
                model.train()
                optimizer.zero_grad()
                pred = model(w_t, c_t)
                loss = F.mse_loss(pred, wt_t)
                loss.backward()
                optimizer.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                pred = model(
                    torch.FloatTensor(w_wt[hold_out:hold_out+1]),
                    torch.FloatTensor(c[hold_out:hold_out+1])
                )
                all_preds[mutations[hold_out]].append(pred.numpy()[0, 1])
    
    # Compute metrics
    cdst_mean = {m: np.mean(v) for m, v in all_preds.items()}
    
    errors = [abs(cdst_mean[m] - NMR_DATA[m]['non_ground']) for m in mutations]
    mae = np.mean(errors)
    median = np.median(errors)
    
    # Per-position
    pos_290_301 = ['M290L', 'L301I', 'M290L_L301I']
    pos_382 = ['F382L', 'F382Y', 'F382V']
    
    mae_290_301 = np.mean([abs(cdst_mean[m] - NMR_DATA[m]['non_ground']) for m in pos_290_301])
    mae_382 = np.mean([abs(cdst_mean[m] - NMR_DATA[m]['non_ground']) for m in pos_382])
    
    # Direction
    dir_correct = sum(1 for m in mutations 
                     if (NMR_DATA[m]['non_ground'] - WT_NON_GROUND) * (cdst_mean[m] - WT_NON_GROUND) > 0
                     or abs(NMR_DATA[m]['non_ground'] - WT_NON_GROUND) < 0.05)
    
    return {
        'encoder': encoder_name,
        'mae': mae,
        'median': median,
        'mae_290_301': mae_290_301,
        'mae_382': mae_382,
        'direction': dir_correct / n,
        'per_mutant': {m: cdst_mean[m] for m in mutations},
    }


def main():
    print("=" * 70)
    print("Encoding Ablation Study")
    print("=" * 70)
    
    encoders = [
        (encode_baseline, "Baseline (5-dim)"),
        (encode_extended, "Extended (10-dim)"),
        (encode_aaindex, "AAIndex (8-dim)"),
    ]
    
    results = []
    
    for encoder, name in encoders:
        print(f"\n--- {name} ---")
        result = run_experiment(encoder, name)
        results.append(result)
        
        print(f"  Overall MAE: {result['mae']:.4f}")
        print(f"  Median MAE:  {result['median']:.4f}")
        print(f"  Position 290/301 MAE: {result['mae_290_301']:.4f}")
        print(f"  Position 382 MAE:     {result['mae_382']:.4f}")
        print(f"  Direction: {result['direction']*100:.1f}%")
    
    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Encoder':<20} {'MAE':>8} {'Median':>8} {'290/301':>8} {'382':>8} {'Dir':>8}")
    print("-" * 60)
    
    for r in results:
        print(f"  {r['encoder']:<18} {r['mae']:>8.4f} {r['median']:>8.4f} {r['mae_290_301']:>8.4f} {r['mae_382']:>8.4f} {r['direction']*100:>7.1f}%")
    
    # AF2 reference
    af2_errors = [abs(AF2_NON_GROUND[m] - NMR_DATA[m]['non_ground']) for m in NMR_DATA]
    af2_mae = np.mean(af2_errors)
    af2_290_301 = np.mean([abs(AF2_NON_GROUND[m] - NMR_DATA[m]['non_ground']) for m in ['M290L', 'L301I', 'M290L_L301I']])
    af2_382 = np.mean([abs(AF2_NON_GROUND[m] - NMR_DATA[m]['non_ground']) for m in ['F382L', 'F382Y', 'F382V']])
    
    print(f"  {'AF2 frequency':<18} {af2_mae:>8.4f} {np.median(af2_errors):>8.4f} {af2_290_301:>8.4f} {af2_382:>8.4f} {'100.0%':>8}")
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    best = min(results, key=lambda x: x['mae'])
    print(f"""
Best encoder: {best['encoder']}
  Overall MAE: {best['mae']:.4f}
  Position 382 MAE: {best['mae_382']:.4f}

Key question: Does richer encoding help F382 series?
  Baseline 382 MAE: {results[0]['mae_382']:.4f}
  Best 382 MAE:     {best['mae_382']:.4f}
  Improvement:      {(results[0]['mae_382'] - best['mae_382'])/results[0]['mae_382']*100:+.1f}%

Expected: Marginal improvement (encoding is not the bottleneck, data is)
""")
    
    # Save
    out_path = Path(__file__).parent / 'results'
    out_path.mkdir(exist_ok=True)
    
    with open(out_path / 'encoding_ablation.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"Saved to {out_path / 'encoding_ablation.json'}")


if __name__ == '__main__':
    main()
