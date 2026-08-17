"""
Paired significance tests for the headline encoding comparisons.

Motivation (reviewer-critical gap): the paper's central claim is that
variant C (ddG 5-dim) massively reduces Abl1 LOO MAE vs the Extended
10-dim control (0.1046 vs 0.4086, -74.4%). With n=6, reviewers will ask
whether this is statistically distinguishable from noise. The per-mutant
LOO errors are PAIRED (same mutants, same folds, same seeds), so a paired
test is the correct analysis.

Tests (primary = seed-averaged per-mutant errors, n=6/8 pairs):
  1. Wilcoxon signed-rank (nonparametric; skewed errors at small n)
  2. Paired t-test (reported for completeness)
  3. Paired bootstrap 95% CI of the mean difference (10k resamples)
  4. Cohen's d (paired) effect size
  5. Number of folds where the better encoding wins

Sensitivity (per-seed, 5 seeds x n mutants):
  6. Block bootstrap over mutants (cluster-aware) 95% CI of mean difference

Comparisons:
  - Abl1: variant C (ddG+positions, 5-dim)  vs  Extended (10-dim)
  - Src : position markers (4-dim)          vs  Extended (10-dim)

Protocol: identical to alternative_encodings.py / src_validation_and_robustness.py
(LowRankCDST, prob-space MSE, 5 seeds, seed=seed*100+hold_out, 800 epochs, LOO).
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from scipy import stats

from encoding_ablation_control import (
    ABL1_DATA, ABL1_WT_NON_GROUND, ABL1_SEQ_LEN,
    SRC_DATA, SRC_WT_NON_ACTIVE, SRC_SEQ_LEN,
    AA_PROPERTIES_6_EXT, encode_extended,
)
from alternative_encodings import DDG_DATA, DDG_NORM, encode_ddg_main
from src_validation_and_robustness import (
    run_loo_cv, encode_src_pos_markers,
)

N_BOOT = 10_000
RNG_SEED = 42


def encode_abl1_extended(name, data):
    return encode_extended(name, data, AA_PROPERTIES_6_EXT, ABL1_SEQ_LEN,
                           system='abl1')


def encode_abl1_variant_c(name, data):
    return encode_ddg_main(name, data, DDG_DATA, DDG_NORM, ABL1_SEQ_LEN,
                           system='abl1')


def paired_tests(err_a, err_b, label_a, label_b, n_boot=N_BOOT):
    """err_a / err_b: dict {mutant: abs error}. Positive diff = A worse than B."""
    names = list(err_a.keys())
    a = np.array([err_a[m] for m in names])
    b = np.array([err_b[m] for m in names])
    diff = a - b  # >0 means A worse
    n = len(names)

    n_wins_a_worse = int((diff > 0).sum())
    n_wins_b_worse = int((diff < 0).sum())

    # 1. Wilcoxon signed-rank (ties: zero_method='wilcox' handles zeros)
    try:
        w_p = stats.wilcoxon(diff, zero_method='wilcox').pvalue
    except ValueError:
        w_p = float('nan')

    # 2. Paired t-test
    t_stat, t_p = stats.ttest_rel(a, b)

    # 3. Paired bootstrap CI of mean difference (resample pairs)
    rng = np.random.default_rng(RNG_SEED)
    boot_means = np.array([
        rng.choice(diff, size=n, replace=True).mean() for _ in range(n_boot)
    ])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    # 4. Cohen's d (paired): mean(diff) / sd(diff)
    d_paired = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float('nan')

    return {
        'n_pairs': n,
        'mean_err_a': float(a.mean()),
        'mean_err_b': float(b.mean()),
        'mean_diff_a_minus_b': float(diff.mean()),
        'mae_ratio': float(a.mean() / b.mean()) if b.mean() > 0 else None,
        'folds_a_worse': n_wins_a_worse,
        'folds_b_worse': n_wins_b_worse,
        'folds_tie': n - n_wins_a_worse - n_wins_b_worse,
        'wilcoxon_p': float(w_p),
        't_paired_p': float(t_p),
        't_paired_stat': float(t_stat),
        'bootstrap_ci_95_mean_diff': [float(ci_lo), float(ci_hi)],
        'cohen_d_paired': d_paired,
    }


# (dead block_bootstrap_ci stub that returned None removed 2026-08-17;
#  the real implementation used downstream is block_bootstrap_ci_errors below)


def block_bootstrap_ci_errors(preds_a, preds_b, targets, names, wt=None,
                              n_boot=N_BOOT, seed=RNG_SEED):
    """preds: {mutant: [per-seed preds]}; targets: {mutant: true value}."""
    n_seeds = len(preds_a[names[0]])
    err_a = {m: [abs(preds_a[m][s] - targets[m]) for s in range(n_seeds)]
             for m in names}
    err_b = {m: [abs(preds_b[m][s] - targets[m]) for s in range(n_seeds)]
             for m in names}
    rng = np.random.default_rng(seed)
    n = len(names)
    boot_means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs = []
        for i in idx:
            m = names[i]
            for s in range(n_seeds):
                diffs.append(err_a[m][s] - err_b[m][s])
        boot_means.append(np.mean(diffs))
    boot_means = np.array(boot_means)
    return {
        'n_clusters': n,
        'n_seeds': n_seeds,
        'ci_95_mean_diff': [float(np.percentile(boot_means, 2.5)),
                            float(np.percentile(boot_means, 97.5))],
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("Paired significance: variant C vs Extended (Abl1) / pos-markers vs Extended (Src)")
    print("=" * 80)

    results = {'protocol': 'LowRankCDST, prob-space MSE, 5 seeds, LOO-CV, '
                           'paired per-mutant LOO errors; Wilcoxon signed-rank, '
                           'paired t, 10k paired bootstrap, Cohen d',
               'n_bootstrap': N_BOOT, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}

    # ---------------- Abl1 ----------------
    print("\n[Abl1] running LOO-CV: Extended vs variant C ...")
    abl1_ext = run_loo_cv(ABL1_DATA, 'non_ground', ABL1_WT_NON_GROUND,
                          ABL1_SEQ_LEN, encode_abl1_extended, 10)
    abl1_c = run_loo_cv(ABL1_DATA, 'non_ground', ABL1_WT_NON_GROUND,
                        ABL1_SEQ_LEN, encode_abl1_variant_c, 5)
    print(f"  Extended MAE={abl1_ext['mae']:.4f} | variant C MAE={abl1_c['mae']:.4f}")

    abl1_test = paired_tests(abl1_ext['errors'], abl1_c['errors'],
                             'Extended', 'variant C')
    abl1_block = block_bootstrap_ci_errors(
        abl1_ext['per_seed'], abl1_c['per_seed'],
        {m: ABL1_DATA[m]['non_ground'] for m in ABL1_DATA}, list(ABL1_DATA.keys()))
    results['abl1_variantC_vs_extended'] = {
        'test': abl1_test,
        'block_bootstrap_per_seed': abl1_block,
        'per_mutant_errors': {
            'Extended': abl1_ext['errors'],
            'variant_C': abl1_c['errors'],
        },
    }

    # ---------------- Src ----------------
    print("[Src] running LOO-CV: Extended vs pos-markers ...")
    from src_validation_and_robustness import encode_src_extended
    src_ext = run_loo_cv(SRC_DATA, 'non_active', SRC_WT_NON_ACTIVE,
                         SRC_SEQ_LEN, encode_src_extended, 10)
    src_pos = run_loo_cv(SRC_DATA, 'non_active', SRC_WT_NON_ACTIVE,
                         SRC_SEQ_LEN, encode_src_pos_markers, 4)
    print(f"  Extended MAE={src_ext['mae']:.4f} | pos-markers MAE={src_pos['mae']:.4f}")

    src_test = paired_tests(src_ext['errors'], src_pos['errors'],
                            'Extended', 'pos-markers')
    src_block = block_bootstrap_ci_errors(
        src_ext['per_seed'], src_pos['per_seed'],
        {m: SRC_DATA[m]['non_active'] for m in SRC_DATA}, list(SRC_DATA.keys()))
    results['src_posmarkers_vs_extended'] = {
        'test': src_test,
        'block_bootstrap_per_seed': src_block,
        'per_mutant_errors': {
            'Extended': src_ext['errors'],
            'pos_markers': src_pos['errors'],
        },
    }

    # ---------------- Save ----------------
    out_dir = Path(__file__).parent / 'results'
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / 'paired_significance.json'
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                        encoding='utf-8')
    print(f"\n[OK] {out_json}")

    # ---------------- Report ----------------
    L = []
    L.append("# 配对显著性检验报告 (variant C vs Extended)")
    L.append("")
    L.append(f"> 协议: {results['protocol']} | 时间: {results['timestamp']} | "
             f"耗时 {time.time()-t0:.0f}s")
    L.append("")
    L.append("## 1. Abl1: variant C vs Extended (n=6 LOO 折, 种子平均)")
    L.append("")
    r = results['abl1_variantC_vs_extended']['test']
    L.append(f"- MAE: Extended {r['mean_err_a']:.4f} vs variant C {r['mean_err_b']:.4f} "
             f"(比值 {r['mae_ratio']:.2f}x)")
    L.append(f"- 均值差 (Extended − C): **{r['mean_diff_a_minus_b']:.4f}** "
             f"[bootstrap 95% CI {r['bootstrap_ci_95_mean_diff'][0]:.3f}, "
             f"{r['bootstrap_ci_95_mean_diff'][1]:.3f}]")
    n_b_better = r['n_pairs'] - r['folds_b_worse'] - r['folds_tie']
    L.append(f"- 折内胜负: C 更优 {n_b_better}/{r['n_pairs']} 折 "
             f"(C 更差 {r['folds_b_worse']} 折)")
    L.append(f"- **Wilcoxon signed-rank p = {r['wilcoxon_p']:.4f}**")
    L.append(f"- 配对 t-test p = {r['t_paired_p']:.4f} (t={r['t_paired_stat']:.2f})")
    L.append(f"- 配对 Cohen's d = {r['cohen_d_paired']:.2f}")
    L.append(f"- 逐折误差 (Ext | C): " +
             ", ".join(f"{results['abl1_variantC_vs_extended']['per_mutant_errors']['Extended'][m]:.3f}|{results['abl1_variantC_vs_extended']['per_mutant_errors']['variant_C'][m]:.3f}" for m in ABL1_DATA))
    bb = results['abl1_variantC_vs_extended']['block_bootstrap_per_seed']
    L.append(f"- per-seed 分块 bootstrap (5 seeds x 6 突变体, 按突变体重采样): "
             f"95% CI [{bb['ci_95_mean_diff'][0]:.3f}, {bb['ci_95_mean_diff'][1]:.3f}]")
    L.append("")
    L.append("## 2. Src: pos-markers vs Extended (n=8 LOO 折, 种子平均)")
    L.append("")
    r = results['src_posmarkers_vs_extended']['test']
    L.append(f"- MAE: Extended {r['mean_err_a']:.4f} vs pos-markers {r['mean_err_b']:.4f} "
             f"(比值 {r['mae_ratio']:.2f}x)")
    L.append(f"- 均值差 (Extended − pos): **{r['mean_diff_a_minus_b']:.4f}** "
             f"[bootstrap 95% CI {r['bootstrap_ci_95_mean_diff'][0]:.3f}, "
             f"{r['bootstrap_ci_95_mean_diff'][1]:.3f}]")
    n_b_better = r['n_pairs'] - r['folds_b_worse'] - r['folds_tie']
    L.append(f"- 折内胜负: pos 更优 {n_b_better}/{r['n_pairs']} 折 "
             f"(pos 更差 {r['folds_b_worse']} 折)")
    L.append(f"- **Wilcoxon signed-rank p = {r['wilcoxon_p']:.4f}**")
    L.append(f"- 配对 t-test p = {r['t_paired_p']:.4f} (t={r['t_paired_stat']:.2f})")
    L.append(f"- 配对 Cohen's d = {r['cohen_d_paired']:.2f}")
    bb = results['src_posmarkers_vs_extended']['block_bootstrap_per_seed']
    L.append(f"- per-seed 分块 bootstrap: 95% CI "
             f"[{bb['ci_95_mean_diff'][0]:.3f}, {bb['ci_95_mean_diff'][1]:.3f}]")
    L.append("")
    L.append("## 3. 解读与论文建议")
    L.append("")
    L.append("- n=6/8 下非参数检验是主口径 (误差偏斜); t 检验仅作完整性报告")
    L.append("- 若 Wilcoxon p < 0.05: 论文在 §5.4/Table 4 报"
             " 'paired Wilcoxon p=X, all N folds improved, "
             "bootstrap 95% CI of mean MAE reduction [a,b]'")
    L.append("- 若 p 边缘: 诚实报告 + 强调 6/6 折一致改善与效应量")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*脚本: experiments/iclr_restructuring/paired_significance.py | "
             "数据: results/paired_significance.json*")

    out_md = out_dir / 'paired_significance_report.md'
    out_md.write_text("\n".join(L), encoding='utf-8')
    print(f"[OK] {out_md}")


if __name__ == '__main__':
    main()
