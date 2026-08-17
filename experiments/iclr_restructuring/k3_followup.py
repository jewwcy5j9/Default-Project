"""
K=3 follow-ups (Phase-1):
  A. Scalarized collapse re-check at K=3 (faithful K=2 analog: non-active fraction)
  B. Discriminative baselines at K=3 (P1.3): Ridge/Lasso/Logistic/kNN/
     probability-GP diagnostic/RF/SimpleCDST(linear), on variant C (Abl1 core)
     and pos-markers (Src). The probability-GP diagnostic reuses the locked
     primary GP factory but is not the manuscript's CLR-GP baseline.
  C. Grouped CV by position (P1.5): hold out F382 family / 290-301 group (Abl1)
  D. Compositional zero-shot holdout (P1.6): train on singles, predict double
     directly vs via logit composition
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
    ABL1_K3, ABL1_K3_WT_POP, ABL1_SEQ_LEN, SRC_K3, SRC_K3_WT_POP, SRC_SEQ_LEN,
    enc_abl1_ddg_main, enc_abl1_extended, enc_src_pos_markers,
    enc_src_extended,
)
from k3_benchmark import train_one_seed
from gp_protocols import PRIMARY_GP_PROTOCOL, make_primary_gp
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

TGT_A = {m: ABL1_K3[m]['pop'] for m in ABL1_K3 if m != 'WT'}
TGT_S = {m: SRC_K3[m]['pop'] for m in SRC_K3 if m != 'SrcKD-WT'}


def mae_metric(preds, targets, names):
    errs = {m: float(np.abs(np.array(preds[m]) - np.array(targets[m])).mean())
            for m in names}
    return errs, float(np.mean(list(errs.values())))


def run_loo_sklearn(est_fn, mutations, encoder_fn, d, targets, n_seeds=5):
    """LOO with sklearn-style estimator factory; returns mean preds."""
    names = list(mutations.keys())
    X = np.array([encoder_fn(m, mutations[m]) for m in names])
    Y = np.array([targets[m] for m in names])
    mean_preds = {}
    for hold_out in range(len(names)):
        mask = np.ones(len(names), dtype=bool)
        mask[hold_out] = False
        seed_preds = []
        for seed in range(n_seeds):
            est = est_fn(seed)
            if isinstance(est, tuple):
                est, scaler = est
                X_train = scaler.fit_transform(X[mask])
                X_test = scaler.transform(X[hold_out:hold_out + 1])
            else:
                X_train = X[mask]
                X_test = X[hold_out:hold_out + 1]
            est.fit(X_train, Y[mask])
            p = est.predict(X_test)[0]
            p = np.clip(p, 0, 1)
            if p.sum() > 0:
                p = p / p.sum()
            seed_preds.append(p)
        mean_preds[names[hold_out]] = np.mean(seed_preds, axis=0)
    return mean_preds


def simple_cdst_loo(mutations, wt_pop, encoder_fn, d, targets, n_seeds=5,
                    n_epochs=800):
    """Linear CDST: w' = softmax(log w + cG), K=3. Returns mean preds + errors."""
    names = list(mutations.keys())
    n = len(names)
    w_wt = np.tile(np.array(wt_pop, dtype=float), (n, 1))
    encodings = np.array([encoder_fn(m, mutations[m]) for m in names])
    Y = np.array([targets[m] for m in names])
    mean_preds = {}
    errors = {}

    for hold_out in range(n):
        mask = np.ones(n, dtype=bool)
        mask[hold_out] = False
        seed_preds = []
        for seed in range(n_seeds):
            torch.manual_seed(seed * 100 + hold_out)
            np.random.seed(seed * 100 + hold_out)
            G = torch.zeros(d, 3, requires_grad=True)
            opt = torch.optim.Adam([G], lr=5e-3, weight_decay=1e-4)
            w_t = torch.FloatTensor(w_wt[mask])
            c_t = torch.FloatTensor(encodings[mask])
            y_t = torch.FloatTensor(Y[mask])
            best_loss, best_G = float('inf'), None
            for _ in range(n_epochs):
                opt.zero_grad()
                dl = c_t @ G
                pred = F.softmax(torch.log(w_t.clamp(min=1e-8)) + dl, dim=-1)
                loss = F.mse_loss(pred, y_t)
                loss.backward()
                opt.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_G = G.detach().clone()
            with torch.no_grad():
                dl = torch.FloatTensor(encodings[hold_out:hold_out + 1]) @ best_G
                p = F.softmax(torch.log(torch.FloatTensor(
                    [wt_pop])) + dl, dim=-1)
            seed_preds.append(p.numpy()[0])
        mp = np.mean(seed_preds, axis=0)
        mean_preds[names[hold_out]] = mp
        errors[names[hold_out]] = float(np.abs(mp - np.array(targets[names[hold_out]])).mean())
    return mean_preds, errors


def run_baselines(system, mutations, wt_pop, enc_fn, d, targets, names):
    ests = {
        'Ridge': lambda s: MultiOutputRegressor(Ridge(alpha=1.0)),
        'Lasso': lambda s: MultiOutputRegressor(Lasso(alpha=1e-3)),
        # Logistic regression was removed: the panel targets are continuous
        # three-state population vectors, which LogisticRegression can never
        # fit (it raised on every call and the try/except wrote a dead row).
        'kNN': lambda s: KNeighborsRegressor(n_neighbors=min(3, len(names) - 1)),
        'GPR-probability-diagnostic': lambda s: (
            make_primary_gp(), StandardScaler()),
        'RandomForest': lambda s: MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, random_state=s)),
    }
    out = {}
    for name, fn in ests.items():
        try:
            preds = run_loo_sklearn(fn, mutations, enc_fn, d, targets)
            errs, m = mae_metric(preds, targets, names)
            out[name] = {'mae': m, 'errors': errs}
            print(f"  [{system}] {name:<14} MAE={m:.4f}", flush=True)
        except Exception as e:
            print(f"  [{system}] {name}: FAILED ({e})")
            out[name] = {'mae': None, 'error': str(e)}
    return out


def grouped_cv(mutations, wt_pop, encoder_fn, d, targets, groups, n_seeds=5):
    """Leave-site-out: train on all mutants OUTSIDE the group, predict the group.

    Same-group peers are never in the training rows; the whole group is held
    out, so one training per (group, seed) predicts every group member.
    """
    names = list(mutations.keys())
    out = {}
    for g_idx, (gname, group) in enumerate(groups.items()):
        train = [m for m in names if m not in group]
        if not train:
            raise ValueError(
                f"group {gname} covers the whole panel; no external training rows")
        w_tr = np.tile(np.array(wt_pop, dtype=float), (len(train), 1))
        c_tr = np.array([encoder_fn(x, mutations[x]) for x in train])
        y_tr = np.array([targets[x] for x in train])
        seed_models = [
            train_one_seed(w_tr, c_tr, y_tr, d=d, seed=seed * 100 + g_idx)
            for seed in range(n_seeds)
        ]
        g_preds = {}
        g_errs = {}
        for m in group:
            with torch.no_grad():
                c_m = torch.FloatTensor([encoder_fn(m, mutations[m])])
                w_m = torch.FloatTensor([wt_pop])
                seed_preds = [model(w_m, c_m).numpy()[0] for model in seed_models]
            g_preds[m] = np.mean(seed_preds, axis=0)
            g_errs[m] = float(np.abs(g_preds[m] - np.array(targets[m])).mean())
        out[gname] = {'members': group,
                      'train_members': train,
                      'preds': {m: g_preds[m].tolist() for m in group},
                      'errors': g_errs,
                      'group_mae': float(np.mean(list(g_errs.values())))}
        print(f"  [group={gname}] members={group} group_mae={out[gname]['group_mae']:.4f}")
        for m in group:
            print(f"      {m:<14} true={np.round(np.array(targets[m]),2)} "
                  f"pred={np.round(g_preds[m],2)} err={g_errs[m]:.4f}")
    return out


def compositional_holdout(mutations, wt_pop, encoder_fn, d, targets, double_name,
                          singles, n_seeds=5):
    """Train on singles (excl. double), predict double directly and by
    logit-space composition of the single predictions."""
    train = [m for m in mutations if m != double_name]
    names_all = list(mutations.keys())
    w_tr = np.tile(np.array(wt_pop, dtype=float), (len(train), 1))
    c_tr = np.array([encoder_fn(x, mutations[x]) for x in train])
    y_tr = np.array([targets[x] for x in train])
    w_t = np.array(wt_pop, dtype=float)
    c_db = np.array([encoder_fn(double_name, mutations[double_name])])

    direct_preds, comp_preds = [], []
    single_preds = {m: [] for m in singles}
    for seed in range(n_seeds):
        model = train_one_seed(w_tr, c_tr, y_tr, d=d,
                               seed=seed * 100 + names_all.index(double_name))
        with torch.no_grad():
            p_db = model(torch.FloatTensor([w_t]), torch.FloatTensor(c_db))
            p_singles = {m: model(torch.FloatTensor([w_t]),
                                  torch.FloatTensor([encoder_fn(m, mutations[m])]))
                         for m in singles}
        direct_preds.append(p_db.numpy()[0])
        for m in singles:
            single_preds[m].append(p_singles[m].numpy()[0])
        # logit composition: log p_db = log p1 + log p2 - log w (then renormalize)
        log_p1 = np.log(np.clip(p_singles[singles[0]].numpy()[0], 1e-8, 1))
        log_p2 = np.log(np.clip(p_singles[singles[1]].numpy()[0], 1e-8, 1))
        log_w = np.log(np.clip(np.array(wt_pop), 1e-8, 1))
        log_comp = log_p1 + log_p2 - log_w
        log_comp -= log_comp.max()
        comp_preds.append(np.exp(log_comp) / np.exp(log_comp).sum())

    d_direct = np.mean(direct_preds, axis=0)
    d_comp = np.mean(comp_preds, axis=0)
    true = np.array(targets[double_name])
    out = {
        'double': double_name, 'true': true.tolist(),
        'direct_pred': d_direct.tolist(),
        'composed_pred': d_comp.tolist(),
        'direct_mae': float(np.abs(d_direct - true).mean()),
        'composed_mae': float(np.abs(d_comp - true).mean()),
        'single_preds': {m: np.mean(v, axis=0).tolist() for m, v in single_preds.items()},
    }
    print(f"  [comp] {double_name}: true={np.round(true,3)} "
          f"direct={np.round(d_direct,3)} (MAE {out['direct_mae']:.4f}) "
          f"composed={np.round(d_comp,3)} (MAE {out['composed_mae']:.4f})")
    return out


def scalarized_collapse(enc_result, targets, wt_pop):
    """K=2-faithful analog: non-active fraction = 1 - p[0]."""
    names = list(enc_result['per_mutant_preds'].keys())
    t = np.array([1 - targets[m][0] for m in names])
    p = np.array([1 - enc_result['per_mutant_preds'][m][0] for m in names])
    var_t = float(np.var(t))
    var_p = float(np.var(p))
    return var_t, var_p, (var_p / var_t * 100 if var_t > 0 else None)


def main():
    t0 = time.time()
    print("=" * 90)
    print("K=3 follow-ups: collapse recheck / baselines / grouped CV / composition")
    print("=" * 90)
    results = {
        'paired_vs_baselines': {},
        '_protocols': {
            'GPR-probability-diagnostic': {
                **PRIMARY_GP_PROTOCOL,
                'target_coordinates': 'raw probability components',
                'postprocess': 'clip to [0,1] and renormalize',
                'role': 'legacy Phase-1 diagnostic; not manuscript CLR-GP',
            }
        },
    }

    # A. Scalarized collapse at K=3
    print("\n[A] scalarized collapse (non-active fraction variance ratio)")
    k3 = json.loads((Path(__file__).parent / 'results' / 'k3_benchmark_results.json')
                    .read_text(encoding='utf-8'))
    results['collapse'] = {'abl1_core': {}, 'src': {}}
    for ek in ['Extended_10dim', 'C_ddg_5dim', 'Random_10dim', 'Shuffled_10dim',
               'Onehot', 'pos_markers']:
        vt, vp, r = scalarized_collapse(k3['abl1_core'][ek], TGT_A, ABL1_K3_WT_POP)
        results['collapse']['abl1_core'][ek] = {'ratio_pct': r}
        print(f"  abl1 {ek:<20} ratio={r:6.1f}%")
    for ek in ['Extended_10dim', 'pos_markers_4dim', 'Random_10dim',
               'Shuffled_10dim', 'Onehot']:
        vt, vp, r = scalarized_collapse(k3['src'][ek], TGT_S, SRC_K3_WT_POP)
        results['collapse']['src'][ek] = {'ratio_pct': r}
        print(f"  src  {ek:<20} ratio={r:6.1f}%")

    # B. Baselines (P1.3)
    abl1_core = {m: ABL1_K3[m] for m in ABL1_K3 if m not in ('WT', 'H396P', 'M290L_H396P')}
    src = {m: SRC_K3[m] for m in SRC_K3 if m != 'SrcKD-WT'}
    tgt_a = {m: abl1_core[m]['pop'] for m in abl1_core}
    tgt_s = {m: src[m]['pop'] for m in src}

    print("\n[B] baselines at K=3 (variant C / pos-markers)")
    results['baselines'] = {
        'abl1_core_C': run_baselines('abl1', abl1_core, ABL1_K3_WT_POP,
                                     enc_abl1_ddg_main, 5, tgt_a, list(abl1_core)),
        'src_pos': run_baselines('src', src, SRC_K3_WT_POP,
                                 enc_src_pos_markers, 4, tgt_s, list(src)),
    }
    sc_a, errs_sc_a = simple_cdst_loo(abl1_core, ABL1_K3_WT_POP, enc_abl1_ddg_main, 5, tgt_a)
    sc_s, errs_sc_s = simple_cdst_loo(src, SRC_K3_WT_POP, enc_src_pos_markers, 4, tgt_s)
    _, ma_sc_a = mae_metric(sc_a, tgt_a, list(abl1_core))
    _, ma_sc_s = mae_metric(sc_s, tgt_s, list(src))
    results['baselines']['abl1_core_C']['SimpleCDST_linear'] = {'mae': ma_sc_a,
                                                                'errors': errs_sc_a}
    results['baselines']['src_pos']['SimpleCDST_linear'] = {'mae': ma_sc_s,
                                                            'errors': errs_sc_s}
    print(f"  abl1 SimpleCDST MAE={ma_sc_a:.4f} | src SimpleCDST MAE={ma_sc_s:.4f}")

    # paired tests: LowRankCDST vs best baselines (Abl1, variant C)
    k3 = json.loads((Path(__file__).parent / 'results' / 'k3_benchmark_results.json')
                    .read_text(encoding='utf-8'))
    cdst_errs = k3['abl1_core']['C_ddg_5dim']['mae_per_mutant']
    from scipy import stats as sps
    for name, errs in [('SimpleCDST', errs_sc_a),
                       ('Lasso', results['baselines']['abl1_core_C']['Lasso']['errors'])]:
        w_p = sps.wilcoxon([cdst_errs[m] - errs[m] for m in tgt_a],
                           zero_method='wilcox').pvalue
        print(f"  paired: LowRankCDST vs {name}: "
              f"Wilcoxon p={w_p:.4f} (CDST {np.mean(list(cdst_errs.values())):.4f} "
              f"vs {name} {np.mean(list(errs.values())):.4f})")
        results['paired_vs_baselines'][name] = {
            'cdst_mae': float(np.mean(list(cdst_errs.values()))),
            'baseline_mae': float(np.mean(list(errs.values()))),
            'wilcoxon_p': float(w_p)}

    # C. Grouped CV (P1.5)
    print("\n[C] grouped CV by position (Abl1, variant C)")
    groups = {
        'F382_family': ['F382L', 'F382Y', 'F382V'],
        '290_301': ['M290L', 'L301I', 'M290L_L301I'],
    }
    results['grouped_cv'] = grouped_cv(abl1_core, ABL1_K3_WT_POP,
                                       enc_abl1_ddg_main, 5, tgt_a, groups)

    # D. Compositional holdout (P1.6)
    print("\n[D] compositional zero-shot holdout (Abl1, variant C)")
    results['compositional_holdout'] = compositional_holdout(
        abl1_core, ABL1_K3_WT_POP, enc_abl1_ddg_main, 5, tgt_a,
        'M290L_L301I', ['M290L', 'L301I'])

    # Save
    out_dir = Path(__file__).parent / 'results'
    out_json = out_dir / 'k3_followup_results.json'
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False,
                                   default=float), encoding='utf-8')
    print(f"\n[OK] {out_json}  (total {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
