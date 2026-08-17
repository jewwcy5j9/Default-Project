"""
CDST on Abl1 - EXPANDED dataset (11 data points).
Cautious approach:
- Tier 1 (train+LOO): 7 gold + H396P = 8 mutants (same background, same conditions)
- Tier 2 (held-out validation): I2M, T315I (different background)
- Excluded: M290L_H396P (pH 6.5, different condition)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path

# ============================================================
# Data: Abl1 extended
# ============================================================
EPS = 5e-3  # epsilon clipping (same as canonical CDST)

WT = np.array([0.88, 0.06, 0.06])

# AA properties: [volume, hydrophobicity, aromaticity, HBD, HBA, charge]
AA_PROPS = {
    'F': [135, 2.8, 1.0, 0.0, 0.0, 0.0],
    'L': [124, 3.8, 0.0, 0.0, 0.0, 0.0],
    'Y': [141, -1.3, 1.0, 1.0, 1.0, 0.0],
    'V': [105, 4.2, 0.0, 0.0, 0.0, 0.0],
    'M': [124, 1.9, 0.0, 0.0, 0.0, 0.0],
    'I': [126, 4.5, 0.0, 0.0, 0.0, 0.0],
    'H': [118, -3.2, 0.5, 1.0, 0.0, 0.5],
    'P': [90, -1.6, 0.0, 0.0, 0.0, 0.0],
    'T': [93, -0.7, 0.0, 1.0, 1.0, 0.0],
}

def encode_mutation(wt_aa, mut_aa, position, seq_len=534):
    """10-dim Extended encoding (matching canonical_encoding.py)."""
    wt = np.array(AA_PROPS.get(wt_aa, [0]*6))
    mut = np.array(AA_PROPS.get(mut_aa, [0]*6))
    delta = (mut - wt) / 5.0
    
    enc = np.zeros(10)
    enc[0] = position / seq_len
    enc[1:7] = delta
    # enc[7] = double mutant flag (0 for singles)
    # enc[8] = position 290 flag
    # enc[9] = position 301 flag
    if position == 290: enc[8] = 1.0
    elif position == 301: enc[9] = 1.0
    return enc

def encode_double(pos1, wt1, mut1, pos2, wt2, mut2, seq_len=534):
    """Double mutant: sum of singles + double flag."""
    enc1 = encode_mutation(wt1, mut1, pos1, seq_len)
    enc2 = encode_mutation(wt2, mut2, pos2, seq_len)
    enc = enc1 + enc2
    enc[7] = 1.0  # double mutant flag
    return enc

# Tier 1: Same background, same conditions (train + LOO)
TIER1 = [
    {'name': 'M290L', 'enc': encode_mutation('M', 'L', 290), 'pop': [0.55, 0.10, 0.35]},
    {'name': 'L301I', 'enc': encode_mutation('L', 'I', 301), 'pop': [0.25, 0.10, 0.65]},
    {'name': 'M290L_L301I', 'enc': encode_double(290, 'M', 'L', 301, 'L', 'I'), 'pop': [0.08, 0.10, 0.82]},
    {'name': 'F382L', 'enc': encode_mutation('F', 'L', 382), 'pop': [0.88, 0.06, 0.06]},
    {'name': 'F382Y', 'enc': encode_mutation('F', 'Y', 382), 'pop': [0.10, 0.00, 0.90]},
    {'name': 'F382V', 'enc': encode_mutation('F', 'V', 382), 'pop': [0.05, 0.00, 0.95]},
    {'name': 'H396P', 'enc': encode_mutation('H', 'P', 396), 'pop': [0.85, 0.15, 0.00]},  # NEW: silver
]

# Tier 2: Different background (held-out validation)
TIER2 = [
    {'name': 'I2M_background', 'enc': np.zeros(10), 'pop': [0.10, 0.00, 0.90], 'note': 'triple mutant background'},
    {'name': 'T315I_in_I2M', 'enc': encode_mutation('T', 'I', 315), 'pop': [0.93, 0.00, 0.07], 'note': 'in I2M background'},
]

# Clip populations
for m in TIER1 + TIER2:
    pop = np.clip(np.array(m['pop']), EPS, None)
    m['target'] = pop / pop.sum()

WT_CLIP = np.clip(WT, EPS, None)
WT_CLIP = WT_CLIP / WT_CLIP.sum()

# ============================================================
# Model
# ============================================================
class SimpleCDST(nn.Module):
    def __init__(self, K, d):
        super().__init__()
        self.T = nn.Parameter(torch.zeros(K, d))
    
    def forward(self, log_w, c):
        delta = c @ self.T.T
        return F.log_softmax(log_w + delta, dim=-1)

# ============================================================
# LOO on Tier 1 (8 mutants)
# ============================================================
print("=" * 70)
print("CDST ON Abl1 - EXPANDED (8 mutants, LOO)")
print("=" * 70)
print(f"\nTier 1 (train+LOO): {len(TIER1)} mutants")
print(f"Tier 2 (held-out): {len(TIER2)} constructs (different background)")
print(f"Encoding: 10-dim Extended")

K = 3
d = 10
N_SEEDS = 10

print(f"\n--- LOO Results (Tier 1, {N_SEEDS} seeds) ---")
print(f"{'Mutant':>15} {'Pred [A,I1,I2]':>22} {'True [A,I1,I2]':>22} {'MAE':>7}")
print("-" * 70)

loo_maes = []
loo_preds = {}

for i in range(len(TIER1)):
    train = [TIER1[j] for j in range(len(TIER1)) if j != i]
    test = TIER1[i]
    
    preds_seeds = []
    for seed in range(N_SEEDS):
        torch.manual_seed(seed * 11 + 42)
        
        log_wt = torch.FloatTensor(np.log(WT_CLIP)).unsqueeze(0).expand(len(train), -1)
        c_train = torch.FloatTensor(np.array([m['enc'] for m in train]))
        target_log = torch.FloatTensor(np.log(np.array([m['target'] for m in train])))
        
        model = SimpleCDST(K, d)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        
        for epoch in range(800):
            optimizer.zero_grad()
            pred = model(log_wt, c_train)
            loss = F.mse_loss(pred, target_log)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            c_test = torch.FloatTensor(test['enc']).unsqueeze(0)
            log_w = torch.FloatTensor(np.log(WT_CLIP)).unsqueeze(0)
            pred = torch.exp(model(log_w, c_test)).numpy()[0]
        preds_seeds.append(pred)
    
    pred_mean = np.mean(preds_seeds, axis=0)
    pred_mean = pred_mean / pred_mean.sum()
    true = test['target']
    mae = np.abs(pred_mean - true).mean()
    loo_maes.append(mae)
    loo_preds[test['name']] = pred_mean.tolist()
    
    marker = " *NEW*" if test['name'] == 'H396P' else ""
    print(f"  {test['name']:>13} [{pred_mean[0]:.3f},{pred_mean[1]:.3f},{pred_mean[2]:.3f}]"
          f"  [{true[0]:.3f},{true[1]:.3f},{true[2]:.3f}]  {mae:.4f}{marker}")

# ============================================================
# Tier 2: Held-out validation (train on ALL Tier 1, predict Tier 2)
# ============================================================
print(f"\n--- Tier 2: Held-out validation (I2M background) ---")
print(f"  Training on ALL {len(TIER1)} Tier 1 mutants, predicting Tier 2")

tier2_maes = []
for test in TIER2:
    preds_seeds = []
    for seed in range(N_SEEDS):
        torch.manual_seed(seed * 11 + 42)
        
        log_wt = torch.FloatTensor(np.log(WT_CLIP)).unsqueeze(0).expand(len(TIER1), -1)
        c_train = torch.FloatTensor(np.array([m['enc'] for m in TIER1]))
        target_log = torch.FloatTensor(np.log(np.array([m['target'] for m in TIER1])))
        
        model = SimpleCDST(K, d)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        
        for epoch in range(800):
            optimizer.zero_grad()
            pred = model(log_wt, c_train)
            loss = F.mse_loss(pred, target_log)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            c_test = torch.FloatTensor(test['enc']).unsqueeze(0)
            log_w = torch.FloatTensor(np.log(WT_CLIP)).unsqueeze(0)
            pred = torch.exp(model(log_w, c_test)).numpy()[0]
        preds_seeds.append(pred)
    
    pred_mean = np.mean(preds_seeds, axis=0)
    pred_mean = pred_mean / pred_mean.sum()
    true = test['target']
    mae = np.abs(pred_mean - true).mean()
    tier2_maes.append(mae)
    
    print(f"  {test['name']:>15} [{pred_mean[0]:.3f},{pred_mean[1]:.3f},{pred_mean[2]:.3f}]"
          f"  [{true[0]:.3f},{true[1]:.3f},{true[2]:.3f}]  {mae:.4f}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"  Tier 1 LOO-MAE (8 mut): {np.mean(loo_maes):.4f} ± {np.std(loo_maes):.4f}")
print(f"  Tier 1 Median MAE:      {np.median(loo_maes):.4f}")
print(f"  Tier 2 held-out MAE:    {np.mean(tier2_maes):.4f}")

# Compare with original 7-mutant result
print(f"\n  vs original 7-mutant (MAE=0.322):")
orig7_maes = [loo_maes[i] for i in range(6)]  # first 6 are original (excluding H396P)
print(f"    Original 6 single+double (excl H396P): {np.mean(orig7_maes):.4f}")
print(f"    With H396P added: {np.mean(loo_maes):.4f}")

# Baseline
baseline = np.mean([np.abs(WT_CLIP - m['target']).mean() for m in TIER1])
print(f"\n  Baseline (predict WT): {baseline:.4f}")
print(f"  Improvement: {(baseline - np.mean(loo_maes))/baseline*100:.1f}%")

# Direction
dir_correct = 0
for i, m in enumerate(TIER1):
    pred = np.array(loo_preds[m['name']])
    true = m['target']
    true_shift = true - WT_CLIP
    pred_shift = pred - WT_CLIP
    if np.sign(true_shift[np.argmax(np.abs(true_shift))]) == np.sign(pred_shift[np.argmax(np.abs(true_shift))]):
        dir_correct += 1
print(f"  Direction: {dir_correct}/{len(TIER1)}")

# Save
results = {
    'system': 'Abl1_expanded',
    'tier1_n': len(TIER1),
    'tier1_loo_mae': float(np.mean(loo_maes)),
    'tier1_median_mae': float(np.median(loo_maes)),
    'tier2_heldout_mae': float(np.mean(tier2_maes)),
    'baseline_mae': float(baseline),
    'improvement_pct': float((baseline - np.mean(loo_maes))/baseline*100),
    'direction': f"{dir_correct}/{len(TIER1)}",
    'per_mutant_tier1': {m['name']: {'pred': loo_preds[m['name']], 'true': m['target'].tolist(), 'mae': loo_maes[i]} 
                          for i, m in enumerate(TIER1)},
    'note': 'H396P is silver tier (from SI). I2M/T315I are different background (held-out only).',
}
out_path = Path(__file__).resolve().parent / 'results'
with open(out_path / 'cdst_abl1_expanded.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path / 'cdst_abl1_expanded.json'}")
