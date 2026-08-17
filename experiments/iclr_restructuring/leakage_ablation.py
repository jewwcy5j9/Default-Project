"""
Leakage Ablation: Position flags (dim 7/8/9) contribution assessment.

Three encoding variants:
  A: Full 10-dim (Extended) — includes position flags
  B: 7-dim (Chemistry+Position) — remove dim7(double flag), dim8(pos290), dim9(pos301)
  C: 6-dim (Pure Chemistry) — only AA property deltas, no position info at all

Protocol: LOSO (Leave-One-Subject-Out) × 10 seeds, same as encoding_ablation.py
AF2 reference uses CORRECTED values from raw CSV.

Decision criterion:
  - If B still beats AF2 (MAE < 0.415) → headline stands
  - If B collapses to ≥ 0.415 → position memory was the main contributor
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import json

from src.models.low_rank_cdst import LowRankCDST


# ============================================================
# DATA (verified against canonical_results.json)
# ============================================================

AA_PROPERTIES = {
    'F': [135, 2.8, 1.0, 0.0, 0.0, 0.0],
    'L': [124, 3.8, 0.0, 0.0, 0.0, 0.0],
    'Y': [141, -1.3, 1.0, 1.0, 1.0, 0.0],
    'V': [105, 4.2, 0.0, 0.0, 0.0, 0.0],
    'M': [124, 1.9, 0.0, 0.0, 0.0, 0.0],
    'I': [126, 4.5, 0.0, 0.0, 0.0, 0.0],
}

NMR_DATA = {
    'M290L': {'wt': 'M', 'mut': 'L', 'pos': 290, 'non_ground': 0.45},
    'L301I': {'wt': 'L', 'mut': 'I', 'pos': 301, 'non_ground': 0.75},
    'M290L_L301I': {'wt': 'ML', 'mut': 'LI', 'pos': 301, 'non_ground': 0.92},
    'F382L': {'wt': 'F', 'mut': 'L', 'pos': 382, 'non_ground': 0.12},
    'F382Y': {'wt': 'F', 'mut': 'Y', 'pos': 382, 'non_ground': 0.90},
    'F382V': {'wt': 'F', 'mut': 'V', 'pos': 382, 'non_ground': 0.95},
}

WT_NON_GROUND = 0.12

# CORRECTED AF2 values (from raw CSV, aC-helix RMSD > 2.5A)
AF2_CORRECTED = {
    'M290L': 35/480,        # 7.3%
    'L301I': 145/480,       # 30.2%
    'M290L_L301I': 139/480, # 29.0%
    'F382L': 63/480,        # 13.1%
    'F382Y': 139/480,       # 29.0%
    # F382V: NOT in Monteiro dataset
}


# ============================================================
# ENCODERS
# ============================================================

def get_aa_delta(data):
    """Get 6-dim AA property delta.

    FIXED 2026-08-17: multi-letter wt/mut ('ML'/'LI', the double mutant
    M290L_L301I) now sums per-residue deltas, mirroring
    encoding_ablation_control._aa_delta; previously they fell through to
    zeros because 'ML'/'LI' are not table keys. Stored leakage_ablation
    outputs in results/ predate this fix.
    """
    from encoding_ablation_control import _aa_delta
    wt_aa = data['wt'][0] if len(data['wt']) == 1 else data['wt']
    mut_aa = data['mut'][0] if len(data['mut']) == 1 else data['mut']
    return _aa_delta(wt_aa, mut_aa, AA_PROPERTIES)


def encode_full_10dim(name, data):
    """A: Full Extended 10-dim (with position flags)."""
    enc = np.zeros(10)
    enc[0] = data['pos'] / 534
    enc[1:7] = get_aa_delta(data)
    if '_' in name:
        enc[7] = 1.0
    if data['pos'] == 290:
        enc[8] = 1.0
    elif data['pos'] == 301:
        enc[9] = 1.0
    return enc


def encode_no_flags_7dim(name, data):
    """B: Chemistry + Position (7-dim). No identity flags."""
    enc = np.zeros(7)
    enc[0] = data['pos'] / 534
    enc[1:7] = get_aa_delta(data)
    # NO dim7 (double mutant flag)
    # NO dim8 (position 290 flag)
    # NO dim9 (position 301 flag)
    return enc


def encode_pure_chem_6dim(name, data):
    """C: Pure Chemistry (6-dim). Only AA property deltas."""
    # NO position information at all
    return get_aa_delta(data)


# ============================================================
# EXPERIMENT
# ============================================================

def run_loso(encoder, encoder_name, n_seeds=10):
    """Run LOSO with multiple seeds."""
    mutations = list(NMR_DATA.keys())
    n = len(mutations)
    
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
            np.random.seed(seed * 100 + hold_out)
            
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
                all_preds[mutations[hold_out]].append(float(pred.numpy()[0, 1]))
    
    # Aggregate
    cdst_mean = {m: np.mean(v) for m, v in all_preds.items()}
    cdst_std = {m: np.std(v) for m, v in all_preds.items()}
    
    # Metrics (6 mutants)
    errors_6 = [abs(cdst_mean[m] - NMR_DATA[m]['non_ground']) for m in mutations]
    mae_6 = np.mean(errors_6)
    
    # Metrics (5 mutants, fair vs AF2 — exclude F382V)
    mutations_5 = ['M290L', 'L301I', 'M290L_L301I', 'F382L', 'F382Y']
    errors_5 = [abs(cdst_mean[m] - NMR_DATA[m]['non_ground']) for m in mutations_5]
    mae_5 = np.mean(errors_5)
    
    # Per-position
    pos_290_301 = ['M290L', 'L301I', 'M290L_L301I']
    pos_382 = ['F382L', 'F382Y', 'F382V']
    mae_290_301 = np.mean([abs(cdst_mean[m] - NMR_DATA[m]['non_ground']) for m in pos_290_301])
    mae_382 = np.mean([abs(cdst_mean[m] - NMR_DATA[m]['non_ground']) for m in pos_382])
    
    # Direction (5 mutants, exclude F382L as tie)
    dir_mutants = ['M290L', 'L301I', 'M290L_L301I', 'F382Y', 'F382V']
    dir_correct = 0
    dir_total = 0
    for m in dir_mutants:
        true_delta = NMR_DATA[m]['non_ground'] - WT_NON_GROUND
        pred_delta = cdst_mean[m] - WT_NON_GROUND
        if abs(true_delta) < 0.05:
            continue
        dir_total += 1
        if true_delta * pred_delta > 0:
            dir_correct += 1
    
    return {
        'encoder': encoder_name,
        'dim': d,
        'n_seeds': n_seeds,
        'mae_6mut': float(mae_6),
        'mae_5mut': float(mae_5),
        'mae_290_301': float(mae_290_301),
        'mae_382': float(mae_382),
        'direction_correct': dir_correct,
        'direction_total': dir_total,
        'direction_acc': dir_correct / dir_total if dir_total > 0 else None,
        'per_mutant_mean': cdst_mean,
        'per_mutant_std': cdst_std,
    }


def main():
    print("=" * 70)
    print("LEAKAGE ABLATION: Position Flags Contribution")
    print("=" * 70)
    print("\nQuestion: Are CDST's wins on 290/301 due to learning or memorization?")
    print("Method: Compare 10-dim (with flags) vs 7-dim (no flags) vs 6-dim (pure chem)")
    print("Protocol: LOSO × 10 seeds, LowRank-r2, K=2")
    
    encoders = [
        (encode_full_10dim, "A: Full 10-dim (with flags)"),
        (encode_no_flags_7dim, "B: No-flags 7-dim (chem+pos)"),
        (encode_pure_chem_6dim, "C: Pure chemistry 6-dim"),
    ]
    
    results = []
    
    for encoder, name in encoders:
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"{'='*50}")
        result = run_loso(encoder, name, n_seeds=10)
        results.append(result)
        
        print(f"  MAE (6mut): {result['mae_6mut']:.4f}")
        print(f"  MAE (5mut, fair): {result['mae_5mut']:.4f}")
        print(f"  MAE 290/301: {result['mae_290_301']:.4f}")
        print(f"  MAE 382:     {result['mae_382']:.4f}")
        print(f"  Direction:   {result['direction_correct']}/{result['direction_total']}")
        print(f"  Per-mutant predictions:")
        for m in NMR_DATA:
            pred = result['per_mutant_mean'][m]
            std = result['per_mutant_std'][m]
            nmr = NMR_DATA[m]['non_ground']
            print(f"    {m:<15} pred={pred:.3f}±{std:.3f}  NMR={nmr:.2f}  err={abs(pred-nmr):.3f}")
    
    # AF2 reference (CORRECTED, 5 mutants only)
    mutations_5 = ['M290L', 'L301I', 'M290L_L301I', 'F382L', 'F382Y']
    af2_errors_5 = [abs(AF2_CORRECTED[m] - NMR_DATA[m]['non_ground']) for m in mutations_5]
    af2_mae_5 = np.mean(af2_errors_5)
    
    af2_errors_6 = af2_errors_5 + [abs(0.0 - NMR_DATA['F382V']['non_ground'])]  # no AF2 data → 0
    # Actually AF2 has no F382V, so for 6-mutant comparison we can't include it fairly
    # Just report 5-mutant AF2
    
    # AF2 direction (4 mutants, exclude F382V which has no data, exclude F382L tie)
    af2_dir_mutants = ['M290L', 'L301I', 'M290L_L301I', 'F382Y']
    af2_dir_correct = 0
    for m in af2_dir_mutants:
        true_delta = NMR_DATA[m]['non_ground'] - WT_NON_GROUND
        pred_delta = AF2_CORRECTED[m] - WT_NON_GROUND
        if true_delta * pred_delta > 0:
            af2_dir_correct += 1
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY (vs AF2 CORRECTED)")
    print("=" * 70)
    print(f"\n{'Method':<30} {'MAE(5)':>8} {'MAE(6)':>8} {'290/301':>8} {'382':>8} {'Dir':>8}")
    print("-" * 75)
    
    for r in results:
        dir_str = f"{r['direction_correct']}/{r['direction_total']}"
        print(f"  {r['encoder']:<28} {r['mae_5mut']:>8.4f} {r['mae_6mut']:>8.4f} {r['mae_290_301']:>8.4f} {r['mae_382']:>8.4f} {dir_str:>8}")
    
    print(f"  {'AF2 (corrected, 5mut)':<28} {af2_mae_5:>8.4f} {'N/A':>8} "
          f"{np.mean([abs(AF2_CORRECTED[m]-NMR_DATA[m]['non_ground']) for m in ['M290L','L301I','M290L_L301I']]):>8.4f} "
          f"{np.mean([abs(AF2_CORRECTED[m]-NMR_DATA[m]['non_ground']) for m in ['F382L','F382Y']]):>8.4f} "
          f"{af2_dir_correct}/4{'':>4}")
    
    # Decision
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)
    
    mae_b = results[1]['mae_5mut']  # No-flags 7-dim
    mae_a = results[0]['mae_5mut']  # Full 10-dim
    
    print(f"\n  AF2 baseline (5mut):  {af2_mae_5:.4f}")
    print(f"  CDST Full 10-dim:     {mae_a:.4f}  {'WINS' if mae_a < af2_mae_5 else 'LOSES'} vs AF2")
    print(f"  CDST No-flags 7-dim:  {mae_b:.4f}  {'WINS' if mae_b < af2_mae_5 else 'LOSES'} vs AF2")
    print(f"  CDST Pure chem 6-dim: {results[2]['mae_5mut']:.4f}  {'WINS' if results[2]['mae_5mut'] < af2_mae_5 else 'LOSES'} vs AF2")
    
    print(f"\n  Flags contribution:   {mae_b - mae_a:+.4f} (negative = flags help)")
    
    if mae_b < af2_mae_5:
        print("\n  >>> VERDICT: Headline STANDS. Chemistry+position encoding beats AF2 without identity flags.")
        print("  >>> Paper claim: 'CDST learns from physicochemical features, not position memorization.'")
    else:
        print("\n  >>> VERDICT: Headline COLLAPSES. Position flags were the main contributor.")
        print("  >>> Paper must pivot to boundary/POC narrative.")
    
    # Save
    out_path = Path(__file__).parent / 'results'
    out_path.mkdir(exist_ok=True)
    
    output = {
        'experiment': 'leakage_ablation',
        'date': '2026-07-24',
        'protocol': 'LOSO x 10 seeds, LowRank-r2, K=2, 800 epochs',
        'af2_reference': {
            'source': 'CORRECTED from raw CSV (aC-helix RMSD > 2.5A)',
            'values': AF2_CORRECTED,
            'mae_5mut': float(af2_mae_5),
            'direction': f"{af2_dir_correct}/4",
        },
        'results': results,
        'decision': {
            'af2_mae_5mut': float(af2_mae_5),
            'full_10dim_mae_5mut': float(mae_a),
            'no_flags_7dim_mae_5mut': float(mae_b),
            'pure_chem_6dim_mae_5mut': float(results[2]['mae_5mut']),
            'flags_contribution': float(mae_b - mae_a),
            'headline_stands': bool(mae_b < af2_mae_5),
        }
    }
    
    with open(out_path / 'leakage_ablation.json', 'w') as f:
        json.dump(output, f, indent=2, default=float)
    
    print(f"\nSaved to {out_path / 'leakage_ablation.json'}")


if __name__ == '__main__':
    main()
