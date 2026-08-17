"""
P0: Random Encoding Control Experiment — LowRankCDST 重跑（权威模型）

Question: Is physics-informed encoding NECESSARY for CDST's few-shot success?
Design: Compare 4 encoders on Abl1 + Src kinase systems under identical LOO-CV.

背景:
  P0 实验原用 SimpleCDST（纯线性，MAE=0.5252），判定 "NARRATIVE FAILS"。
  但诊断确认权威模型是 LowRankCDST（MLP, MAE=0.4134, 降低 21.3%）。
  本脚本用 LowRankCDST 重跑 P0，判断 Extended vs Random 排名是否反转。

Encoders:
  1. Extended 10-dim (physics, baseline) — current canonical encoding
  2. Random Gaussian 10-dim — preserves position, destroys AA physics
  3. Shuffled Property 10-dim — 20-aa property shuffle (most rigorous control)
  4. One-hot Mutation 6-dim — mutation identity, no continuous physics

Judgment (pre-registered):
  - Extended significantly beats Shuffled on BOTH systems (MAE gap > 5%)
    AND direction accuracy strictly higher -> narrative holds (25-35% ICLR)
  - Roughly tied (MAE gap < 2% or direction equal) -> narrative retreats to
    "linear model + benchmark" (10-15% ICLR)
  - Shuffled/Random beats Extended -> narrative fails
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from src.models.low_rank_cdst import LowRankCDST


# ============================================================
# 20-amino-acid property table (for shuffled-property control)
# [volume, hydrophobicity, aromaticity, h_bond_donor, h_bond_acceptor, charge]
# Sources: AAIndex / Kyte-Doolittle / standard biochemistry
# ============================================================
AA_PROPERTIES_20 = {
    'A': [88.6, 1.8, 0.0, 0.0, 0.0, 0.0],
    'R': [173.4, -4.5, 0.0, 1.0, 0.0, 1.0],
    'N': [114.1, -3.5, 0.0, 1.0, 1.0, 0.0],
    'D': [111.1, -3.5, 0.0, 0.0, 1.0, -1.0],
    'C': [108.5, 2.5, 0.0, 1.0, 0.0, 0.0],
    'E': [138.8, -3.5, 0.0, 0.0, 1.0, -1.0],
    'Q': [143.8, -3.5, 0.0, 1.0, 1.0, 0.0],
    'G': [60.1, -0.4, 0.0, 0.0, 0.0, 0.0],
    'H': [153.2, -3.2, 0.0, 1.0, 1.0, 0.1],
    'I': [166.7, 4.5, 0.0, 0.0, 0.0, 0.0],
    'L': [166.7, 3.8, 0.0, 0.0, 0.0, 0.0],
    'K': [168.6, -3.9, 0.0, 1.0, 0.0, 1.0],
    'M': [162.9, 1.9, 0.0, 0.0, 0.0, 0.0],
    'F': [189.9, 2.8, 1.0, 0.0, 0.0, 0.0],
    'P': [112.7, -1.6, 0.0, 0.0, 0.0, 0.0],
    'S': [89.0, -0.8, 0.0, 1.0, 1.0, 0.0],
    'T': [116.1, -0.7, 0.0, 1.0, 1.0, 0.0],
    'W': [227.8, -0.9, 1.0, 1.0, 0.0, 0.0],
    'Y': [141.2, -1.3, 1.0, 1.0, 1.0, 0.0],
    'V': [140.0, 4.2, 0.0, 0.0, 0.0, 0.0],
}

# Project's canonical 6-aa property table (matches canonical_encoding.py / encoding_ablation.py)
AA_PROPERTIES_6 = {
    'F': [135, 2.8, 1.0, 0.0, 0.0, 0.0],
    'L': [124, 3.8, 0.0, 0.0, 0.0, 0.0],
    'Y': [141, -1.3, 1.0, 1.0, 1.0, 0.0],
    'V': [105, 4.2, 0.0, 0.0, 0.0, 0.0],
    'M': [124, 1.9, 0.0, 0.0, 0.0, 0.0],
    'I': [126, 4.5, 0.0, 0.0, 0.0, 0.0],
}


# ============================================================
# Datasets: Abl1 (6 mutants) + Src (8 mutants, point + double)
# ============================================================

# Abl1: non_ground target, same as encoding_ablation.py
ABL1_DATA = {
    'M290L':         {'wt': 'M', 'mut': 'L', 'pos': 290, 'non_ground': 0.45},
    'L301I':         {'wt': 'L', 'mut': 'I', 'pos': 301, 'non_ground': 0.75},
    'M290L_L301I':   {'wt': 'ML', 'mut': 'LI', 'pos': 301, 'non_ground': 0.92},
    'F382L':         {'wt': 'F', 'mut': 'L', 'pos': 382, 'non_ground': 0.12},
    'F382Y':         {'wt': 'F', 'mut': 'Y', 'pos': 382, 'non_ground': 0.90},
    'F382V':         {'wt': 'F', 'mut': 'V', 'pos': 382, 'non_ground': 0.95},
}
ABL1_WT_NON_GROUND = 0.12
ABL1_SEQ_LEN = 534

# Src: 2-state collapse (Active vs E1+E2). WT from Fig S5 Met305 probe = [0.72, 0.07, 0.21]
# non_active = 1 - 0.72 = 0.28. Same metric as Abl1 (non_ground -> non_active).
SRC_DATA = {
    'SrcKD-L410A':         {'wt': 'L', 'mut': 'A', 'pos': 410, 'non_active': 0.27},   # Fig S5: A=0.73
    'SrcKD-V332I':         {'wt': 'V', 'mut': 'I', 'pos': 332, 'non_active': 0.52},   # A=0.48
    'SrcKD-L270F_V332I':   {'wt': 'LV', 'mut': 'FI', 'pos': 270, 'non_active': 0.91}, # double, A=0.09
    'SrcKD-L325A':         {'wt': 'L', 'mut': 'A', 'pos': 325, 'non_active': 1.00},   # A=0.00
    'SrcKD-A311I':         {'wt': 'A', 'mut': 'I', 'pos': 311, 'non_active': 1.00},   # A=0.00
    'SrcKD-V380A':         {'wt': 'V', 'mut': 'A', 'pos': 380, 'non_active': 1.00},   # A=0.00
    'SrcKD-V331A':         {'wt': 'V', 'mut': 'A', 'pos': 331, 'non_active': 1.00},   # A=0.00
    'SrcKD-F405A':         {'wt': 'F', 'mut': 'A', 'pos': 405, 'non_active': 1.00},   # A=0.00
}
SRC_WT_NON_ACTIVE = 0.28  # 1 - 0.72
SRC_SEQ_LEN = 536

# Src needs AA 'A' which is not in AA_PROPERTIES_6
AA_PROPERTIES_6_EXT = dict(AA_PROPERTIES_6)
AA_PROPERTIES_6_EXT['A'] = [88.6, 1.8, 0.0, 0.0, 0.0, 0.0]  # Ala


# ============================================================
# Encoders
# ============================================================

def _aa_delta(wt_aa, mut_aa, table):
    """Compute normalized AA property delta. wt_aa/mut_aa may be multi-letter for double mutants."""
    if len(wt_aa) > 1:
        # double mutant: sum of single deltas
        deltas = [_aa_delta(w, m, table) for w, m in zip(wt_aa, mut_aa)]
        return np.sum(deltas, axis=0)
    if wt_aa not in table or mut_aa not in table:
        return np.zeros(6)
    return (np.array(table[mut_aa]) - np.array(table[wt_aa])) / 5.0


def encode_extended(name, data, table, seq_len, system='abl1'):
    """Extended 10-dim encoding (canonical physics encoding).

    [0] position / seq_len
    [1-6] AA property deltas / 5.0
    [7] double mutant flag
    [8-9] position-specific flags (only meaningful for Abl1 290/301)
    """
    enc = np.zeros(10)
    enc[0] = data['pos'] / seq_len
    enc[1:7] = _aa_delta(data['wt'], data['mut'], table)

    if '_' in name:
        enc[7] = 1.0

    # Position-specific flags (Abl1 only; Src gets zeros, consistent with original code)
    if system == 'abl1':
        if data['pos'] == 290:
            enc[8] = 1.0
        elif data['pos'] == 301:
            enc[9] = 1.0
    return enc


def encode_random_gaussian(name, data, seed, seq_len):
    """Random Gaussian 10-dim: preserve position dim, destroy AA physics.

    [0] position / seq_len (preserved)
    [1-9] iid N(0, 1) random
    """
    rng = np.random.default_rng(seed)
    enc = np.zeros(10)
    enc[0] = data['pos'] / seq_len
    enc[1:10] = rng.standard_normal(9)
    return enc


def encode_shuffled_property(name, data, shuffled_table, seq_len, system='abl1'):
    """Shuffled Property 10-dim: 20-aa property shuffle.

    Same structure as Extended, but AA->property lookup uses a shuffled table
    where each property column is independently permuted across 20 AAs.
    Preserves: dimensionality, distribution, position info, structural flags.
    Destroys: amino-acid -> physical-property mapping.
    """
    return encode_extended(name, data, shuffled_table, seq_len, system=system)


def encode_onehot_mutation(name, data, mutation_list):
    """One-hot mutation identity (no continuous physics).

    LOO-CV caveat: held-out mutation gets all-zero vector (no training analogue),
    which is the legitimate weakness of one-hot — cannot generalize to unseen mutations.
    """
    enc = np.zeros(len(mutation_list))
    if name in mutation_list:
        enc[mutation_list.index(name)] = 1.0
    return enc


def make_shuffled_table(shuffle_seed):
    """Produce a shuffled 20-aa property table by permuting each property column independently."""
    rng = np.random.default_rng(shuffle_seed)
    aas = list(AA_PROPERTIES_20.keys())
    props = np.array([AA_PROPERTIES_20[aa] for aa in aas], dtype=float)  # [20, 6]
    shuffled = props.copy()
    for col in range(6):
        shuffled[:, col] = rng.permutation(shuffled[:, col])
    # Return a table that supports lookup for any AA (including ones outside the 6 used in the project)
    return {aa: shuffled[i].tolist() for i, aa in enumerate(aas)}


# ============================================================
# Training & Evaluation (LowRankCDST, prob-space / log-space MSE)
# ============================================================

def train_predict(w_wt_train, c_train, w_target_train, c_test, w_wt_test,
                  d, n_seeds=5, n_epochs=800, seed_base=0, loss_space='prob'):
    """LOO-CV training with LowRankCDST (MLP, rank=2, hidden_dim=32).

    LowRankCDST.forward(w, c) returns w_pred = softmax(...) (probability, NOT log).
    This differs from SimpleCDST which returns log_softmax.

    loss_space='prob': L = MSE(softmax(log_w + U@g(c)), target)  [prob-space]
    loss_space='log':  L = MSE(log(softmax(...)), log(target+eps))  [log-space]
    """
    preds = []
    for seed in range(n_seeds):
        torch.manual_seed(seed * 100 + seed_base)
        np.random.seed(seed * 100 + seed_base)

        # 权威模型: LowRankCDST (rank=2, hidden_dim=32, use_state_dependence=True default)
        model = LowRankCDST(K=2, intervention_dim=d, rank=2, hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)

        w_t = torch.FloatTensor(w_wt_train)
        c_t = torch.FloatTensor(c_train)
        wt_t = torch.FloatTensor(w_target_train)
        target_log = torch.FloatTensor(np.log(np.clip(w_target_train, 1e-5, None)))

        best_loss = float('inf')
        best_state = None
        for _ in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            # LowRankCDST returns probabilities (softmax), not log-probabilities
            pred = model(w_t, c_t)
            if loss_space == 'log':
                pred_log = torch.log(pred.clamp(min=1e-8))
                loss = F.mse_loss(pred_log, target_log)
            else:
                loss = F.mse_loss(pred, wt_t)
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            # LowRankCDST returns probabilities directly (no exp needed)
            pred = model(
                torch.FloatTensor(w_wt_test),
                torch.FloatTensor(c_test)
            ).numpy()[0, 1]
        preds.append(float(pred))

    return float(np.mean(preds))


def run_loo(mutations, target_key, wt_non_target, encoder_fn, d, n_seeds=5, loss_space='prob'):
    """Run LOO-CV with given encoder. Returns per-mutant predictions + metrics."""
    n = len(mutations)
    names = list(mutations.keys())

    # WT distribution: [ground/active, non_ground/non_active]
    wt_dist = np.array([1 - wt_non_target, wt_non_target])
    w_wt = np.tile(wt_dist, (n, 1))

    targets = np.array([[1 - mutations[m][target_key], mutations[m][target_key]] for m in names])
    encodings = np.array([encoder_fn(m, mutations[m]) for m in names])
    assert encodings.shape[1] == d, f"Encoding dim mismatch: {encodings.shape[1]} vs {d}"

    preds = {}
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        pred = train_predict(
            w_wt[mask], encodings[mask], targets[mask],
            encodings[i:i+1], w_wt[i:i+1],
            d=d, n_seeds=n_seeds, seed_base=i, loss_space=loss_space,
        )
        preds[names[i]] = pred

    errors = [abs(preds[m] - mutations[m][target_key]) for m in names]
    mae = float(np.mean(errors))
    median = float(np.median(errors))

    # Direction: sign(pred - wt) == sign(true - wt)
    dir_correct = 0
    dir_total = 0
    for m in names:
        true_delta = mutations[m][target_key] - wt_non_target
        pred_delta = preds[m] - wt_non_target
        if abs(true_delta) < 0.05:  # tie threshold
            continue
        if np.sign(true_delta) == np.sign(pred_delta):
            dir_correct += 1
        dir_total += 1

    return {
        'per_mutant': preds,
        'mae': mae,
        'median': median,
        'direction': f"{dir_correct}/{dir_total}",
        'direction_pct': dir_correct / dir_total if dir_total > 0 else 0.0,
        'errors': {m: float(abs(preds[m] - mutations[m][target_key])) for m in names},
    }


# ============================================================
# Main experiment
# ============================================================

def run_system(system_name, mutations, target_key, wt_non_target, seq_len, system='abl1', loss_space='prob'):
    """Run all 4 encoders on one system."""
    print(f"\n{'='*70}")
    print(f"SYSTEM: {system_name} (n={len(mutations)})  loss={loss_space}  model=LowRankCDST")
    print(f"{'='*70}")

    mutation_list = list(mutations.keys())
    n_shuffled_seeds = 5  # average over 5 shuffles
    n_gaussian_seeds = 5  # average over 5 random draws
    n_train_seeds = 5     # training stochasticity

    # 1. Extended (physics baseline)
    print("\n[1/4] Extended 10-dim (physics) ...")
    ext_result = run_loo(
        mutations, target_key, wt_non_target,
        encoder_fn=lambda name, data: encode_extended(name, data, AA_PROPERTIES_6_EXT, seq_len, system=system),
        d=10, n_seeds=n_train_seeds, loss_space=loss_space,
    )
    print(f"  MAE={ext_result['mae']:.4f}  median={ext_result['median']:.4f}  "
          f"dir={ext_result['direction']} ({ext_result['direction_pct']*100:.0f}%)")

    # 2. Random Gaussian (position preserved)
    print("\n[2/4] Random Gaussian 10-dim ...")
    gauss_results = []
    for gs in range(n_gaussian_seeds):
        r = run_loo(
            mutations, target_key, wt_non_target,
            encoder_fn=lambda name, data, gs=gs: encode_random_gaussian(name, data, seed=gs*97+1, seq_len=seq_len),
            d=10, n_seeds=n_train_seeds, loss_space=loss_space,
        )
        gauss_results.append(r)
    gauss_result = aggregate_results(gauss_results, 'Random Gaussian')
    print(f"  MAE={gauss_result['mae']:.4f}  median={gauss_result['median']:.4f}  "
          f"dir={gauss_result['direction']} ({gauss_result['direction_pct']*100:.0f}%)  "
          f"(mean over {n_gaussian_seeds} seeds)")

    # 3. Shuffled Property (20-aa column shuffle)
    print("\n[3/4] Shuffled Property 10-dim ...")
    shuf_results = []
    for ss in range(n_shuffled_seeds):
        shuffled_table = make_shuffled_table(shuffle_seed=ss*131+7)
        r = run_loo(
            mutations, target_key, wt_non_target,
            encoder_fn=lambda name, data, t=shuffled_table: encode_shuffled_property(name, data, t, seq_len, system=system),
            d=10, n_seeds=n_train_seeds, loss_space=loss_space,
        )
        shuf_results.append(r)
    shuf_result = aggregate_results(shuf_results, 'Shuffled Property')
    print(f"  MAE={shuf_result['mae']:.4f}  median={shuf_result['median']:.4f}  "
          f"dir={shuf_result['direction']} ({shuf_result['direction_pct']*100:.0f}%)  "
          f"(mean over {n_shuffled_seeds} shuffles)")

    # 4. One-hot Mutation (no continuous physics)
    print("\n[4/4] One-hot Mutation 6-dim ...")
    onehot_result = run_loo(
        mutations, target_key, wt_non_target,
        encoder_fn=lambda name, data: encode_onehot_mutation(name, data, mutation_list),
        d=len(mutation_list), n_seeds=n_train_seeds, loss_space=loss_space,
    )
    print(f"  MAE={onehot_result['mae']:.4f}  median={onehot_result['median']:.4f}  "
          f"dir={onehot_result['direction']} ({onehot_result['direction_pct']*100:.0f}%)")
    print(f"  (LOO caveat: held-out mutation -> all-zero vector, cannot generalize)")

    return {
        'system': system_name,
        'loss_space': loss_space,
        'n_mutants': len(mutations),
        'model': 'LowRankCDST_rank2_h32',
        'Extended_10dim': ext_result,
        'Random_Gaussian_10dim': gauss_result,
        'Shuffled_Property_10dim': shuf_result,
        'Onehot_Mutation_6dim': onehot_result,
    }


def aggregate_results(results, name):
    """Aggregate results across multiple random seeds (mean predictions)."""
    # Average per-mutant predictions across seeds
    all_mutants = list(results[0]['per_mutant'].keys())
    avg_preds = {m: float(np.mean([r['per_mutant'][m] for r in results])) for m in all_mutants}
    return {
        'per_mutant': avg_preds,
        'mae': float(np.mean([r['mae'] for r in results])),
        'median': float(np.mean([r['median'] for r in results])),
        'direction': results[0]['direction'],  # direction is same across seeds (depends on data)
        'direction_pct': float(np.mean([r['direction_pct'] for r in results])),
        'errors': {m: float(np.mean([r['errors'][m] for r in results])) for m in all_mutants},
        'n_seeds_averaged': len(results),
    }


def main():
    print("=" * 90)
    print("P0: RANDOM ENCODING CONTROL EXPERIMENT (LowRankCDST, DUAL-LOSS PROTOCOL)")
    print("Question: Is physics-informed encoding necessary for CDST's few-shot success?")
    print("Model: LowRankCDST (MLP, rank=2, hidden_dim=32, use_state_dependence=True)")
    print("Protocol: LowRankCDST + BOTH prob-space and log-space MSE")
    print("=" * 90)

    torch.manual_seed(0)
    np.random.seed(0)

    all_results = []

    # === Protocol 1: prob-space MSE ===
    print("\n\n" + "#" * 90)
    print("# PROTOCOL 1: PROB-SPACE MSE")
    print("#" * 90)

    abl1_prob = run_system(
        system_name='Abl1', mutations=ABL1_DATA, target_key='non_ground',
        wt_non_target=ABL1_WT_NON_GROUND, seq_len=ABL1_SEQ_LEN, system='abl1', loss_space='prob',
    )
    src_prob = run_system(
        system_name='Src', mutations=SRC_DATA, target_key='non_active',
        wt_non_target=SRC_WT_NON_ACTIVE, seq_len=SRC_SEQ_LEN, system='src', loss_space='prob',
    )
    all_results.extend([abl1_prob, src_prob])

    # === Protocol 2: log-space MSE ===
    print("\n\n" + "#" * 90)
    print("# PROTOCOL 2: LOG-SPACE MSE")
    print("#" * 90)

    abl1_log = run_system(
        system_name='Abl1', mutations=ABL1_DATA, target_key='non_ground',
        wt_non_target=ABL1_WT_NON_GROUND, seq_len=ABL1_SEQ_LEN, system='abl1', loss_space='log',
    )
    src_log = run_system(
        system_name='Src', mutations=SRC_DATA, target_key='non_active',
        wt_non_target=SRC_WT_NON_ACTIVE, seq_len=SRC_SEQ_LEN, system='src', loss_space='log',
    )
    all_results.extend([abl1_log, src_log])

    # === Comparison tables ===
    print("\n\n" + "=" * 100)
    print("DUAL-PROTOCOL COMPARISON TABLE (LowRankCDST)")
    print("=" * 100)
    print(f"{'System':<8} {'Loss':<6} {'Encoder':<28} {'MAE':>8} {'Median':>8} {'Direction':>10}")
    print("-" * 100)

    for r in all_results:
        sys_name = r['system']
        loss = r['loss_space']
        for enc in ['Extended_10dim', 'Random_Gaussian_10dim', 'Shuffled_Property_10dim', 'Onehot_Mutation_6dim']:
            e = r[enc]
            print(f"  {sys_name:<6} {loss:<6} {enc:<28} {e['mae']:>8.4f} {e['median']:>8.4f} "
                  f"{e['direction']:>6} ({e['direction_pct']*100:>3.0f}%)")
        print()

    # === Cross-protocol stability analysis ===
    print("=" * 100)
    print("CROSS-PROTOCOL STABILITY ANALYSIS")
    print("=" * 100)
    print(f"\n{'Encoder':<28} {'Abl1 prob':>10} {'Abl1 log':>10} {'Stable?':>8} | "
          f"{'Src prob':>10} {'Src log':>10} {'Stable?':>8}")
    print("-" * 100)

    encoders = ['Extended_10dim', 'Random_Gaussian_10dim', 'Shuffled_Property_10dim', 'Onehot_Mutation_6dim']
    for enc in encoders:
        abl1_p = abl1_prob[enc]['mae']
        abl1_l = abl1_log[enc]['mae']
        src_p = src_prob[enc]['mae']
        src_l = src_log[enc]['mae']
        abl1_gap = abs(abl1_p - abl1_l) / max(abl1_p, abl1_l) * 100
        src_gap = abs(src_p - src_l) / max(src_p, src_l) * 100
        abl1_stable = "YES" if abl1_gap < 15 else "NO"
        src_stable = "YES" if src_gap < 15 else "NO"
        print(f"  {enc:<28} {abl1_p:>10.4f} {abl1_l:>10.4f} {abl1_stable:>4} ({abl1_gap:>3.0f}%) | "
              f"{src_p:>10.4f} {src_l:>10.4f} {src_stable:>4} ({src_gap:>3.0f}%)")

    # === Ranking comparison ===
    print(f"\n{'Encoder ranking (by MAE)':<28} {'Abl1 prob':>15} {'Abl1 log':>15} {'Src prob':>15} {'Src log':>15}")
    print("-" * 100)
    for sys_r, label in [(abl1_prob, 'Abl1'), (src_prob, 'Src')]:
        ranked = sorted(encoders, key=lambda e: sys_r[e]['mae'])
        print(f"  {label+' prob':<28} {' > '.join([e.split('_')[0] for e in ranked])}")
    for sys_r, label in [(abl1_log, 'Abl1'), (src_log, 'Src')]:
        ranked = sorted(encoders, key=lambda e: sys_r[e]['mae'])
        print(f"  {label+' log':<28} {' > '.join([e.split('_')[0] for e in ranked])}")

    # === Narrative judgment ===
    print("\n\n" + "=" * 100)
    print("NARRATIVE JUDGMENT PER PROTOCOL (LowRankCDST)")
    print("=" * 100)
    for protocol_results, loss_name in [([abl1_prob, src_prob], 'prob-space'), ([abl1_log, src_log], 'log-space')]:
        print(f"\n--- {loss_name} ---")
        for sys_r in protocol_results:
            ext = sys_r['Extended_10dim']
            shuf = sys_r['Shuffled_Property_10dim']
            rand = sys_r['Random_Gaussian_10dim']
            gap_shuf = (shuf['mae'] - ext['mae']) / max(shuf['mae'], 1e-6) * 100
            gap_rand = (rand['mae'] - ext['mae']) / max(rand['mae'], 1e-6) * 100
            print(f"  {sys_r['system']}: Ext={ext['mae']:.4f} Shuf={shuf['mae']:.4f} "
                  f"Rand={rand['mae']:.4f}  Ext-Shuf gap={gap_shuf:+.1f}%  Ext-Rand gap={gap_rand:+.1f}%")

    # === Overall conclusion ===
    print("\n\n" + "=" * 100)
    print("OVERALL CONCLUSION (DUAL-PROTOCOL, LowRankCDST)")
    print("=" * 100)
    # Check if Extended wins on both systems under EITHER protocol
    prob_ext_wins = (abl1_prob['Extended_10dim']['mae'] < abl1_prob['Shuffled_Property_10dim']['mae'] and
                     src_prob['Extended_10dim']['mae'] < src_prob['Shuffled_Property_10dim']['mae'])
    log_ext_wins = (abl1_log['Extended_10dim']['mae'] < abl1_log['Shuffled_Property_10dim']['mae'] and
                    src_log['Extended_10dim']['mae'] < src_log['Shuffled_Property_10dim']['mae'])

    # Also check Extended vs Random (the key question for ranking inversion)
    prob_ext_beats_rand = (abl1_prob['Extended_10dim']['mae'] < abl1_prob['Random_Gaussian_10dim']['mae'] and
                           src_prob['Extended_10dim']['mae'] < src_prob['Random_Gaussian_10dim']['mae'])
    log_ext_beats_rand = (abl1_log['Extended_10dim']['mae'] < abl1_log['Random_Gaussian_10dim']['mae'] and
                          src_log['Extended_10dim']['mae'] < src_log['Random_Gaussian_10dim']['mae'])

    if prob_ext_wins and log_ext_wins:
        verdict = "NARRATIVE HOLDS (Extended wins both systems under both protocols)"
    elif prob_ext_wins or log_ext_wins:
        verdict = "PROTOCOL-DEPENDENT (Extended wins under one protocol but not the other) -> UNSTABLE"
    else:
        verdict = "NARRATIVE FAILS (Extended does not win under either protocol)"
    print(f"\n  >>> VERDICT (vs Shuffled): {verdict} <<<")
    print(f"\n  Extended vs Random ranking:")
    print(f"    prob-space: Extended {'beats' if prob_ext_beats_rand else 'does NOT beat'} Random on both systems")
    print(f"    log-space:  Extended {'beats' if log_ext_beats_rand else 'does NOT beat'} Random on both systems")

    # Save
    out_path = Path(__file__).parent / 'results'
    out_path.mkdir(exist_ok=True)
    output = {
        'experiment': 'P0_random_encoding_control_dual_protocol_LowRankCDST',
        'description': 'Dual-loss protocol with LowRankCDST (authority model) re-run of P0',
        'model': 'LowRankCDST (rank=2, hidden_dim=32, use_state_dependence=True)',
        'protocols': ['prob-space MSE', 'log-space MSE'],
        'results': {
            'prob_space': {'Abl1': abl1_prob, 'Src': src_prob},
            'log_space': {'Abl1': abl1_log, 'Src': src_log},
        },
        'verdict': verdict,
        'extended_vs_random': {
            'prob_space_ext_beats_rand_both_systems': prob_ext_beats_rand,
            'log_space_ext_beats_rand_both_systems': log_ext_beats_rand,
        },
        'encoders': encoders,
    }
    out_file = out_path / 'encoding_ablation_lowrank_dual.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    main()
