"""
Src K=3 negative-finding deep dive:

Exp 1: phospho anchors (pY530 = E2 reference [0,0,100], pY419 = active
       [0.90,0.05,0.05]) added as TRAINING-ONLY examples -> does E2 evidence
       in training fix F405A / V380A / V331A?
Exp 2: capacity/rank sensitivity at Src K=3 (rank 3, epochs 1500, hidden 64,
       lr 1e-3) on pos-markers and Extended.
Exp 4: canonical 2-state Src sensitivity to the L410A ground truth
       (Fig S5 probe 0.27 vs Table S2 global 0.04).
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn.functional as F

from k3_data import (
    SRC_K3, SRC_K3_WT_POP, SRC_SEQ_LEN, ABL1_SEQ_LEN,
    enc_src_extended, enc_src_pos_markers,
)
from k3_benchmark import train_one_seed, metrics
from src.models.low_rank_cdst import LowRankCDST

SRC_MUT = {m: SRC_K3[m] for m in SRC_K3 if m != 'SrcKD-WT'}

# Anchors (training-only): audited values from cui2025 file mutants array
ANCHORS = {
    'SrcKD-pY419': {'wt': 'Y', 'mut': 'pY', 'pos': 419, 'pop': [0.90, 0.05, 0.05],
                    'tier': 'low'},
    'SrcpY530':    {'wt': 'Y', 'mut': 'pY', 'pos': 530, 'pop': [0.00, 0.00, 1.00],
                    'tier': 'medium'},
}


def enc_src_pos_anchored(name, data):
    enc = np.zeros(4)
    if data['pos']:
        enc[0] = data['pos'] / SRC_SEQ_LEN
    for i, p in enumerate([311, 332, 380]):
        if data['pos'] == p:
            enc[i + 1] = 1.0
    return enc


def enc_src_ext_anchored(name, data):
    """Extended encoding; pY treated via Y properties (delta = 0) + position."""
    from encoding_ablation_control import AA_PROPERTIES_6_EXT, encode_extended
    d = dict(data)
    if data['wt'] == 'Y' and data['mut'] == 'pY':
        d['mut'] = 'Y'
    return encode_extended(name, d, AA_PROPERTIES_6_EXT, SRC_SEQ_LEN, system='src')


def loo_with_anchors(mutations, anchors, wt_pop, encoder_fn, d, n_seeds=5,
                     n_epochs=800, rank=2, hidden=32, lr=5e-3):
    """LOO over mutations; anchors always in the training set."""
    names = list(mutations.keys())
    n = len(names)
    w_wt = np.tile(np.array(wt_pop, dtype=float), (n, 1))
    targets = np.array([mutations[m]['pop'] for m in names])
    encodings = np.array([encoder_fn(m, mutations[m]) for m in names])

    w_anc = np.tile(np.array(wt_pop, dtype=float), (len(anchors), 1))
    if anchors:
        t_anc = np.array([anchors[m]['pop'] for m in anchors])
        e_anc = np.array([encoder_fn(m, anchors[m]) for m in anchors])
    else:
        t_anc = np.zeros((0, 3))
        e_anc = np.zeros((0, d))

    mean_preds = {}
    for hold_out in range(n):
        mask = np.ones(n, dtype=bool)
        mask[hold_out] = False
        seed_preds = []
        for seed in range(n_seeds):
            torch.manual_seed(seed * 100 + hold_out)
            np.random.seed(seed * 100 + hold_out)
            model = LowRankCDST(K=3, intervention_dim=d, rank=rank,
                                hidden_dim=hidden)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=1e-4)
            w_tr = torch.FloatTensor(np.concatenate([w_wt[mask], w_anc]))
            c_tr = torch.FloatTensor(np.concatenate([encodings[mask], e_anc]))
            t_tr = torch.FloatTensor(np.concatenate([targets[mask], t_anc]))
            best_loss, best_state = float('inf'), None
            for _ in range(n_epochs):
                optimizer.zero_grad()
                pred = model(w_tr, c_tr)
                loss = F.mse_loss(pred, t_tr)
                loss.backward()
                optimizer.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                p = model(torch.FloatTensor([wt_pop]),
                          torch.FloatTensor([encoder_fn(names[hold_out],
                                                        mutations[names[hold_out]])]))
            seed_preds.append(p.numpy()[0])
        mean_preds[names[hold_out]] = np.mean(seed_preds, axis=0)
    return mean_preds


def run_exp1():
    print("\n[Exp 1] anchors (pY419, pY530) training-only, Src K=3")
    out = {}
    for enc_name, fn, d in [('pos_markers', enc_src_pos_anchored, 4),
                            ('Extended', enc_src_ext_anchored, 10)]:
        preds = loo_with_anchors(SRC_MUT, ANCHORS, SRC_K3_WT_POP, fn, d)
        met = metrics(preds, {m: SRC_MUT[m]['pop'] for m in SRC_MUT}, SRC_K3_WT_POP)
        out[enc_name] = {'mae': met['mae'], 'direction': met['direction'],
                         'errors': met['mae_per_mutant'],
                         'preds': {m: preds[m].tolist() for m in preds}}
        print(f"  {enc_name:<12} MAE={met['mae']:.4f} dir={met['direction']}")
        for m in ['SrcKD-F405A', 'SrcKD-V380A', 'SrcKD-V331A']:
            print(f"      {m:<14} true={np.round(np.array(SRC_MUT[m]['pop']),2)} "
                  f"pred={np.round(preds[m],2)} err={met['mae_per_mutant'][m]:.4f}")
    return out


def run_exp2():
    print("\n[Exp 2] capacity/rank sensitivity, Src K=3 pos-markers")
    out = {}
    for cfg in [dict(rank=3), dict(rank=2, hidden=64), dict(rank=2, n_epochs=1500),
                dict(rank=2, lr=1e-3), dict(rank=3, hidden=64, n_epochs=1500)]:
        preds = loo_with_anchors(SRC_MUT, {}, SRC_K3_WT_POP, enc_src_pos_markers, 4,
                                 n_epochs=cfg.get('n_epochs', 800), rank=cfg.get('rank', 2),
                                 hidden=cfg.get('hidden', 32), lr=cfg.get('lr', 5e-3))
        met = metrics(preds, {m: SRC_MUT[m]['pop'] for m in SRC_MUT}, SRC_K3_WT_POP)
        label = str(cfg)
        out[label] = {'mae': met['mae'], 'direction': met['direction'],
                      'errors': met['mae_per_mutant']}
        print(f"  {label:<40} MAE={met['mae']:.4f} dir={met['direction']}")
        print(f"      F405A err={met['mae_per_mutant']['SrcKD-F405A']:.4f} "
              f"V380A err={met['mae_per_mutant']['SrcKD-V380A']:.4f}")
    return out


def run_exp4():
    """Canonical 2-state Src: L410A probe 0.27 vs global 0.04."""
    print("\n[Exp 4] 2-state Src sensitivity to L410A ground truth")
    from src_validation_and_robustness import (
        run_loo_cv, encode_src_pos_markers as pos2, encode_src_extended as ext2,
    )
    from encoding_ablation_control import SRC_DATA, SRC_WT_NON_ACTIVE, SRC_SEQ_LEN
    out = {}
    for l410a_val, label in [(0.27, 'probe_0.27'), (0.04, 'global_0.04')]:
        data = {k: dict(v) for k, v in SRC_DATA.items()}
        data['SrcKD-L410A']['non_active'] = l410a_val
        r_pos = run_loo_cv(data, 'non_active', SRC_WT_NON_ACTIVE, SRC_SEQ_LEN,
                           pos2, 4)
        r_ext = run_loo_cv(data, 'non_active', SRC_WT_NON_ACTIVE, SRC_SEQ_LEN,
                           ext2, 10)
        out[label] = {'pos_markers_mae': r_pos['mae'],
                      'extended_mae': r_ext['mae'],
                      'pos_dir': r_pos['direction'], 'ext_dir': r_ext['direction']}
        print(f"  L410A={l410a_val:<8} pos={r_pos['mae']:.4f} ({r_pos['direction']}) "
              f"ext={r_ext['mae']:.4f} ({r_ext['direction']})")
    return out


def main():
    t0 = time.time()
    print("=" * 90)
    print("Src K=3 negative-finding deep dive")
    print("=" * 90)
    results = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    results['exp1_anchors'] = run_exp1()
    results['exp2_capacity'] = run_exp2()
    results['exp4_2state_l410a'] = run_exp4()

    out_dir = Path(__file__).parent / 'results'
    out_json = out_dir / 'k3_src_deepdive_results.json'
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False,
                                   default=float), encoding='utf-8')
    print(f"\n[OK] {out_json}  (total {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
