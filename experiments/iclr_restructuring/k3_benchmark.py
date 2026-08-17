"""
K=3 benchmark: full re-run of the canonical pipeline at 3-state resolution.

Scope (Phase-1 revamp, P1.1/P1.2):
  - Abl1 core (n=6): Extended / A(dVol-rescaled) / B(no-dVol+ddG) / C(ddG) /
    D(BLOSUM62) / pos-markers / One-hot / Random(5-draw) / Shuffled(5-draw)
  - Abl1 extended (n=8, +H396P, +M290L_H396P): same minus C (no ddG for new mutants)
  - Src (n=8): Extended / no-dVol / pos-markers / One-hot / Random / Shuffled
  - Loss ablation (K=3): variant C (Abl1 core) and pos-markers (Src), 6 losses
  - Paired tests (K=3): C vs Extended (Abl1 core), pos-markers vs Extended (Src)

Protocol identical to canonical (LowRankCDST(K=3, rank=2, hidden_dim=32),
prob-space MSE, 800 epochs, 5 seeds, seed=seed*100+hold_out, LOO-CV).
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
    ABL1_K3, ABL1_K3_WT_POP, ABL1_K3_CORE, ABL1_K3_EXT, ABL1_SEQ_LEN,
    SRC_K3, SRC_K3_WT_POP, SRC_SEQ_LEN,
    enc_abl1_extended, enc_abl1_dvol_normalized, enc_abl1_no_dvol,
    enc_abl1_ddg_main, enc_abl1_blosum62, enc_abl1_pos_markers,
    enc_abl1_onehot, enc_src_extended, enc_src_no_dvol, enc_src_pos_markers,
    enc_src_onehot, shuffled_factory, random_factory,
    ABL1_ONEHOT_NAMES, SRC_ONEHOT_NAMES,
)
from src.models.low_rank_cdst import LowRankCDST
from src.models.losses import (
    FisherRaoLoss, HellingerLoss, SymmetricKLLoss, NaturalParameterLoss,
)
from scipy import stats


def train_one_seed(w_train, c_train, target_train, d, seed, loss_name='mse',
                   n_epochs=800, K=3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LowRankCDST(K=K, intervention_dim=d, rank=2, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)

    w_t = torch.FloatTensor(w_train)
    c_t = torch.FloatTensor(c_train)
    target_t = torch.FloatTensor(target_train)

    losses = {
        'mse': lambda p, t: F.mse_loss(p, t),
        'kl': lambda p, t: F.kl_div(torch.log(p.clamp(min=1e-8)), t,
                                    reduction='batchmean'),
        'fisher_rao': FisherRaoLoss(mode='hellinger'),
        'hellinger': HellingerLoss(),
        'jsd': SymmetricKLLoss(),
        'natural_l2': NaturalParameterLoss(),
    }
    loss_fn = losses[loss_name]

    best_loss = float('inf')
    best_state = None
    for _ in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(w_t, c_t)
        loss = loss_fn(pred, target_t)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model


def run_loo(mutations, wt_pop, encoder_fn, d, loss_name='mse', n_seeds=5,
            n_epochs=800, K=3):
    """LOO-CV for K-state populations. Returns mean preds + per-seed preds."""
    names = list(mutations.keys())
    n = len(names)
    w_wt = np.tile(np.array(wt_pop, dtype=float), (n, 1))
    targets = np.array([mutations[m]['pop'] for m in names])
    encodings = np.array([encoder_fn(m, mutations[m]) for m in names])
    assert encodings.shape[1] == d, f"{encodings.shape[1]} != {d}"

    mean_preds = {}
    per_seed_preds = {m: [] for m in names}
    for hold_out in range(n):
        mask = np.ones(n, dtype=bool)
        mask[hold_out] = False
        seed_preds = []
        for seed in range(n_seeds):
            model = train_one_seed(w_wt[mask], encodings[mask], targets[mask],
                                   d=d, seed=seed * 100 + hold_out,
                                   loss_name=loss_name, n_epochs=n_epochs, K=K)
            with torch.no_grad():
                p = model(torch.FloatTensor(w_wt[hold_out:hold_out + 1]),
                          torch.FloatTensor(encodings[hold_out:hold_out + 1]))
            seed_preds.append(p.numpy()[0])
            per_seed_preds[names[hold_out]].append(p.numpy()[0])
        mean_preds[names[hold_out]] = np.mean(seed_preds, axis=0)

    return {'per_mutant': mean_preds, 'per_seed': per_seed_preds,
            'targets': {m: mutations[m]['pop'] for m in names}}


def metrics(preds, targets, wt_pop, tie_delta=0.05):
    """Per-state MAE and ACTIVE-state direction agreement (ADR-002)."""
    names = list(preds.keys())
    mae = {}
    dir_ok, dir_total = 0, 0
    dir_detail = {}
    for m in names:
        p = np.array(preds[m])
        t = np.array(targets[m])
        e = float(np.abs(p - t).mean())
        mae[m] = e
        true_active_delta = float(t[0] - wt_pop[0])
        pred_active_delta = float(p[0] - wt_pop[0])
        if abs(true_active_delta) < tie_delta:
            dir_detail[m] = 'TIE'
            continue
        dir_total += 1
        if pred_active_delta * true_active_delta > 0:
            dir_ok += 1
            dir_detail[m] = 'OK'
        else:
            dir_detail[m] = 'WRONG'
    return {'mae_per_mutant': mae, 'mae': float(np.mean(list(mae.values()))),
            'direction': f'{dir_ok}/{dir_total}',
            'direction_detail': dir_detail}


def paired_tests(err_a, err_b, n_boot=10000, seed=42):
    names = list(err_a.keys())
    a = np.array([err_a[m] for m in names])
    b = np.array([err_b[m] for m in names])
    diff = a - b
    n = len(names)
    w_p = stats.wilcoxon(diff, zero_method='wilcox').pvalue
    t_stat, t_p = stats.ttest_rel(a, b)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(diff, size=n, replace=True).mean()
                     for _ in range(n_boot)])
    d_paired = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float('nan')
    return {
        'mean_err_a': float(a.mean()), 'mean_err_b': float(b.mean()),
        'mean_diff_a_minus_b': float(diff.mean()),
        'mae_ratio': float(a.mean() / b.mean()) if b.mean() > 0 else None,
        'n_pairs': n,
        'a_worse_folds': int((diff > 0).sum()),
        'b_worse_folds': int((diff < 0).sum()),
        'wilcoxon_p': float(w_p),
        't_paired_p': float(t_p),
        'bootstrap_ci_95': [float(np.percentile(boot, 2.5)),
                            float(np.percentile(boot, 97.5))],
        'cohen_d_paired': d_paired,
    }


def block_bootstrap(preds_a, preds_b, targets, names, wt_pop, n_boot=10000, seed=42):
    n_seeds = len(preds_a[names[0]])
    err_a = {m: [float(np.abs(np.array(preds_a[m][s]) - np.array(targets[m])).mean())
                 for s in range(n_seeds)] for m in names}
    err_b = {m: [float(np.abs(np.array(preds_b[m][s]) - np.array(targets[m])).mean())
                 for s in range(n_seeds)] for m in names}
    rng = np.random.default_rng(seed)
    n = len(names)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs = []
        for i in idx:
            m = names[i]
            for s in range(n_seeds):
                diffs.append(err_a[m][s] - err_b[m][s])
        boot.append(np.mean(diffs))
    boot = np.array(boot)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


# ============================================================
# Encoder registry
# ============================================================

def build_abl1_encoders(names):
    onehot_names = [m for m in names if m != 'WT']
    rnd = random_factory()
    shf = shuffled_factory('abl1')
    return {
        'Extended_10dim': {'fn': enc_abl1_extended, 'd': 10},
        'A_dVol_rescaled_10dim': {'fn': enc_abl1_dvol_normalized, 'd': 10},
        'B_no_dVol_ddg_11dim': {'fn': enc_abl1_no_dvol, 'd': 11},
        'C_ddg_5dim': {'fn': enc_abl1_ddg_main, 'd': 5},
        'D_blosum62_5dim': {'fn': enc_abl1_blosum62, 'd': 5},
        'pos_markers': {'fn': lambda m, d: enc_abl1_pos_markers(m, d), 'd': 4},
        'Onehot': {'fn': lambda m, d: enc_abl1_onehot(m, d, onehot_names),
                   'd': len(onehot_names)},
        'Random_10dim': {'fn': lambda m, d: rnd(m, d, ABL1_SEQ_LEN), 'd': 10},
        'Shuffled_10dim': {'fn': shf, 'd': 10},
    }


def build_abl1_ext_encoders(names):
    """Extended set (n=8): no C (missing ddG for H396P/M290L_H396P)."""
    encoders = build_abl1_encoders(names)
    encoders['pos_markers_ext'] = {
        'fn': lambda m, d: enc_abl1_pos_markers(m, d, extra_positions=(396,)),
        'd': 5}
    encoders.pop('C_ddg_5dim')
    return encoders


def build_src_encoders(names):
    onehot_names = [m for m in names if m != 'SrcKD-WT']
    rnd = random_factory()
    shf = shuffled_factory('src')
    return {
        'Extended_10dim': {'fn': enc_src_extended, 'd': 10},
        'no_dVol_9dim': {'fn': enc_src_no_dvol, 'd': 9},
        'pos_markers_4dim': {'fn': enc_src_pos_markers, 'd': 4},
        'Onehot': {'fn': lambda m, d: enc_src_onehot(m, d, onehot_names),
                   'd': len(onehot_names)},
        'Random_10dim': {'fn': lambda m, d: rnd(m, d, SRC_SEQ_LEN), 'd': 10},
        'Shuffled_10dim': {'fn': shf, 'd': 10},
    }


# ============================================================
# Main
# ============================================================

def run_encoding_set(name, mutations, wt_pop, encoders, loss='mse'):
    out = {}
    for ek, cfg in encoders.items():
        t0 = time.time()
        res = run_loo(mutations, wt_pop, cfg['fn'], cfg['d'], loss_name=loss)
        met = metrics(res['per_mutant'], res['targets'], wt_pop)
        out[ek] = {'d': cfg['d'], **met,
                   'per_mutant_preds': {m: res['per_mutant'][m].tolist()
                                        for m in res['per_mutant']},
                   'per_seed': {m: [p.tolist() for p in res['per_seed'][m]]
                                for m in res['per_seed']},
                   'seconds': round(time.time() - t0, 1)}
        print(f"  [{name}] {ek:<22} MAE={met['mae']:.4f} "
              f"dir={met['direction']} ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    t0 = time.time()
    print("=" * 90)
    print("K=3 benchmark (Phase-1 revamp)")
    print("=" * 90)
    results = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
               'protocol': 'LowRankCDST(K=3, rank=2, hidden_dim=32), 800 epochs, '
                           '5 seeds, LOO, seed=seed*100+hold_out; K=3 targets'}

    # ---- Abl1 core (n=6) ----
    abl1_core = {m: ABL1_K3[m] for m in ABL1_K3_CORE if m != 'WT'}
    print("\n[Abl1 core n=6, K=3]")
    results['abl1_core'] = run_encoding_set('abl1_core', abl1_core,
                                            ABL1_K3_WT_POP,
                                            build_abl1_encoders(ABL1_K3_CORE))

    # ---- Abl1 extended (n=8) ----
    abl1_ext = {m: ABL1_K3[m] for m in ABL1_K3_EXT if m != 'WT'}
    print("\n[Abl1 extended n=8, K=3]")
    results['abl1_ext'] = run_encoding_set('abl1_ext', abl1_ext,
                                           ABL1_K3_WT_POP,
                                           build_abl1_ext_encoders(ABL1_K3_EXT))

    # ---- Src (n=8) ----
    src = {m: SRC_K3[m] for m in SRC_K3 if m != 'SrcKD-WT'}
    print("\n[Src n=8, K=3]")
    results['src'] = run_encoding_set('src', src, SRC_K3_WT_POP,
                                      build_src_encoders(list(SRC_K3.keys())))

    # ---- Loss ablation (K=3) ----
    print("\n[Loss ablation] Abl1 core variant C + Src pos-markers")
    src_no_wt = {m: SRC_K3[m] for m in SRC_K3 if m != 'SrcKD-WT'}
    loss_results = {}
    for loss in ['mse', 'kl', 'fisher_rao', 'hellinger', 'jsd', 'natural_l2']:
        r_a = run_loo(abl1_core, ABL1_K3_WT_POP, enc_abl1_ddg_main, 5,
                      loss_name=loss)
        r_s = run_loo(src_no_wt, SRC_K3_WT_POP, enc_src_pos_markers, 4,
                      loss_name=loss)
        loss_results[loss] = {
            'abl1_core_C': metrics(r_a['per_mutant'], r_a['targets'], ABL1_K3_WT_POP),
            'src_pos': metrics(r_s['per_mutant'], r_s['targets'], SRC_K3_WT_POP),
        }
        print(f"  {loss:<12} Abl1-C MAE={loss_results[loss]['abl1_core_C']['mae']:.4f} "
              f"| Src-pos MAE={loss_results[loss]['src_pos']['mae']:.4f}", flush=True)
    results['loss_ablation'] = loss_results

    # ---- Paired tests (K=3) ----
    print("\n[Paired tests]")
    t_abl1 = paired_tests(results['abl1_core']['Extended_10dim']['mae_per_mutant'],
                          results['abl1_core']['C_ddg_5dim']['mae_per_mutant'])
    bb_abl1 = block_bootstrap(results['abl1_core']['Extended_10dim']['per_seed'],
                              results['abl1_core']['C_ddg_5dim']['per_seed'],
                              {m: abl1_core[m]['pop'] for m in abl1_core},
                              list(abl1_core.keys()), ABL1_K3_WT_POP)
    t_src = paired_tests(results['src']['Extended_10dim']['mae_per_mutant'],
                         results['src']['pos_markers_4dim']['mae_per_mutant'])
    bb_src = block_bootstrap(results['src']['Extended_10dim']['per_seed'],
                             results['src']['pos_markers_4dim']['per_seed'],
                             {m: SRC_K3[m]['pop'] for m in src},
                             list(src.keys()), SRC_K3_WT_POP)
    results['paired_tests'] = {
        'abl1_C_vs_Extended': {'test': t_abl1, 'block_bootstrap_ci': bb_abl1},
        'src_pos_vs_Extended': {'test': t_src, 'block_bootstrap_ci': bb_src},
    }
    print(f"  Abl1 C vs Ext: Wilcoxon p={t_abl1['wilcoxon_p']:.4f}, "
          f"CI {bb_abl1[0]:.3f}-{bb_abl1[1]:.3f}")
    print(f"  Src  pos vs Ext: Wilcoxon p={t_src['wilcoxon_p']:.4f}, "
          f"CI {bb_src[0]:.3f}-{bb_src[1]:.3f}")

    # ---- Save ----
    out_dir = Path(__file__).parent / 'results'
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / 'k3_benchmark_results.json'
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False,
                                   default=float), encoding='utf-8')
    print(f"\n[OK] {out_json}  (total {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
