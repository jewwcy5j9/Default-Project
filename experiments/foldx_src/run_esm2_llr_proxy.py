"""
B2: ESM-2 masked-marginal LLR as a DeltaDeltaG proxy for Src (and Abl1 validation).

Background (from experiments/foldx_src/src_variant_c_report.md):
  - FoldX 5.1 unavailable (requires human academic license registration).
  - The previous proxy (wild-type-marginal LLR, single forward pass) gave
    Src variant-C MAE=0.3320 (+32.4% vs pos-markers baseline 0.2508) and
    weak correlation with non_active (Spearman rho=-0.2455).
  - This experiment upgrades the proxy to the standard ESM-1v *masked-marginal*
    protocol (Meier et al., Nat. Biotechnol. 2021): per position, average
    log-likelihoods over 20 forward passes that mask the target residue plus a
    random 15% of other positions. This is the ProteinGym-standard LLR and
    generally correlates better with experimental stability than the
    wild-type-marginal variant.

Protocol (identical to canonical src_validation_and_robustness.py):
  LowRankCDST(K=2, rank=2, hidden_dim=32), prob-space MSE,
  Adam(lr=5e-3, weight_decay=1e-4), 800 epochs, 5 seeds,
  seed = seed*100 + hold_out, best-state tracking, LOO-CV.

Checks:
  1. Abl1: correlation of masked-marginal LLR with experimental ddG
     (data/nmr_populations/xie2020_abl1_FINAL.json energies).
  2. Src: correlation of LLR with non_active population.
  3. Abl1 LOO-CV: variant C with LLR proxy vs experimental ddG variant C
     (sanity: experimental ddG should reproduce MAE ~0.1046).
  4. Src LOO-CV: variant C with LLR proxy vs pos-markers 4-dim baseline
     (sanity: pos-markers should reproduce MAE ~0.2508).

No FoldX output is fabricated; all ddG proxies are explicitly labeled as
ESM-2 zero-shot scores.
"""

import sys
import json
import time
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "iclr_restructuring"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn.functional as F

from esm2_encoding import (
    ABL1_KD, SRC_FULL, find_position, parse_mutations_from_name,
    ESM2_LOCAL_PATH, ESM2_MODEL_ID,
)
from encoding_ablation_control import (
    ABL1_DATA, ABL1_WT_NON_GROUND, ABL1_SEQ_LEN,
    SRC_DATA, SRC_WT_NON_ACTIVE, SRC_SEQ_LEN,
)
from alternative_encodings import DDG_DATA, DDG_NORM
from src_validation_and_robustness import run_loo_cv

N_MASK_REPS = 20
MASK_PROB = 0.15
BATCH_SIZE = 10
SEED = 0


# ============================================================
# ESM-2 masked-marginal LLR
# ============================================================

def load_esm2_mlm():
    from transformers import EsmForMaskedLM, EsmTokenizer

    source = ESM2_LOCAL_PATH if Path(ESM2_LOCAL_PATH).exists() else ESM2_MODEL_ID
    print(f"Loading ESM-2 (MaskedLM): {source}")
    t0 = time.time()
    tokenizer = EsmTokenizer.from_pretrained(source)
    model = EsmForMaskedLM.from_pretrained(source)
    model.eval()
    model = model.to(torch.device('cpu'))
    print(f"  model loaded in {time.time()-t0:.1f}s")
    return model, tokenizer


def masked_marginal_logprob(model, tokenizer, wt_seq, pos_idx, n_reps=N_MASK_REPS,
                            mask_prob=MASK_PROB, batch_size=BATCH_SIZE, seed=SEED):
    """Average log P(aa | masked context) at pos_idx over n_reps forward passes.

    ESM-1v protocol: mask the target residue plus a random 15% of other
    residues; average log-probabilities over independent passes.
    Returns dict {aa: mean_log_prob}.
    """
    rng = np.random.default_rng(seed)
    ids = tokenizer(wt_seq, return_tensors='pt', truncation=True,
                    max_length=1024).input_ids[0]  # (L+2,), BOS at 0, EOS at -1
    tok_idx = pos_idx + 1  # residue pos -> token pos (BOS offset)
    seq_len = len(ids) - 2

    vocab = tokenizer.vocab if hasattr(tokenizer, 'vocab') else tokenizer.get_vocab()
    mask_id = tokenizer.mask_token_id

    masked_batches = []
    for rep in range(n_reps):
        m = ids.clone()
        for j in range(1, seq_len + 1):
            if j != tok_idx and rng.random() < mask_prob:
                m[j] = mask_id
        m[tok_idx] = mask_id
        masked_batches.append(m)

    sum_logp = {}
    for start in range(0, n_reps, batch_size):
        batch = torch.stack(masked_batches[start:start + batch_size])
        with torch.no_grad():
            out = model(batch)
        log_probs = F.log_softmax(out.logits, dim=-1)  # (B, L+2, V)
        lp = log_probs[:, tok_idx, :]  # (B, V)
        for aa in ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']:
            tid = tokenizer.convert_tokens_to_ids(aa)
            vals = lp[:, tid].tolist()
            sum_logp[aa] = sum_logp.get(aa, 0.0) + float(np.sum(vals))
        print(f"    batch {start // batch_size + 1}/{(n_reps + batch_size - 1) // batch_size} done",
              flush=True)

    return {aa: v / n_reps for aa, v in sum_logp.items()}


def compute_llr_scores(model, tokenizer, wt_seq, mutations, system):
    """Masked-marginal LLR for all mutants. Double mutants: sum of per-position LLR."""
    results = {}
    for name, data in mutations.items():
        mut_list = parse_mutations_from_name(name, data)
        total_llr = 0.0
        pos_info = []
        for pos, wt_aa, mut_aa in mut_list:
            idx, offset = find_position(wt_seq, pos, wt_aa, system)
            print(f"  {name}: {wt_aa}{pos}{mut_aa} -> seq[{idx}] "
                  f"(offset {offset:+d}), masked-marginal x{N_MASK_REPS}...", flush=True)
            lp = masked_marginal_logprob(model, tokenizer, wt_seq, idx,
                                         seed=SEED + idx)
            llr_pos = float(lp[mut_aa] - lp[wt_aa])
            total_llr += llr_pos
            pos_info.append({'nominal_pos': pos, 'seq_idx': idx, 'offset': offset,
                             'wt': wt_aa, 'mut': mut_aa, 'llr_single': llr_pos})
            print(f"    LLR({wt_aa}->{mut_aa}) = {llr_pos:+.4f}", flush=True)
        results[name] = {'llr': float(total_llr), 'positions': pos_info}
    return results


# ============================================================
# Variant-C encodings driven by the LLR proxy
# ============================================================

def make_llr_encoders(llr_map, norm_const, seq_len, system):
    def encoder(name, data):
        enc = np.zeros(5)
        enc[0] = data['pos'] / seq_len
        enc[1] = llr_map.get(name, 0.0) / norm_const if norm_const > 0 else 0.0
        if system == 'abl1':
            if data['pos'] == 290:
                enc[2] = 1.0
            elif data['pos'] == 301:
                enc[3] = 1.0
            elif data['pos'] == 382:
                enc[4] = 1.0
        elif system == 'src':
            if data['pos'] == 311:
                enc[2] = 1.0
            elif data['pos'] == 332:
                enc[3] = 1.0
            elif data['pos'] == 380:
                enc[4] = 1.0
        return enc
    return encoder


def encode_src_pos_markers(name, data):
    enc = np.zeros(4)
    enc[0] = data['pos'] / SRC_SEQ_LEN
    for i, pos in enumerate([311, 332, 380]):
        if data['pos'] == pos:
            enc[i + 1] = 1.0
    return enc


def encode_abl1_ddg_variant_c(name, data):
    from alternative_encodings import encode_ddg_main
    return encode_ddg_main(name, data, DDG_DATA, DDG_NORM, ABL1_SEQ_LEN, system='abl1')


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 90)
    print("B2: ESM-2 masked-marginal LLR as DeltaDeltaG proxy")
    print("    Abl1 validation + Src variant-C pipeline")
    print("=" * 90)

    torch.set_num_threads(16)
    model, tokenizer = load_esm2_mlm()

    # 1. LLR for Abl1
    print("\n[Abl1] computing masked-marginal LLR (kinase domain, KD):")
    abl1_llr = compute_llr_scores(model, tokenizer, ABL1_KD, ABL1_DATA, 'abl1')

    # 2. LLR for Src
    print("\n[Src] computing masked-marginal LLR (full length):")
    src_llr = compute_llr_scores(model, tokenizer, SRC_FULL, SRC_DATA, 'src')

    abl1_llr_vals = {n: r['llr'] for n, r in abl1_llr.items()}
    src_llr_vals = {n: r['llr'] for n, r in src_llr.items()}

    # 3. Correlations
    from scipy.stats import pearsonr, spearmanr

    abl1_ddg = np.array([DDG_DATA[n] for n in ABL1_DATA])
    abl1_llr_arr = np.array([abl1_llr_vals[n] for n in ABL1_DATA])
    corr_abl1_pearson, p1 = pearsonr(abl1_llr_arr, abl1_ddg)
    corr_abl1_spearman, p2 = spearmanr(abl1_llr_arr, abl1_ddg)

    src_na = np.array([SRC_DATA[n]['non_active'] for n in SRC_DATA])
    src_llr_arr = np.array([src_llr_vals[n] for n in SRC_DATA])
    corr_src_pearson, p3 = pearsonr(src_llr_arr, src_na)
    corr_src_spearman, p4 = spearmanr(src_llr_arr, src_na)

    print("\n--- Correlations ---")
    print(f"  Abl1: masked-marginal LLR vs experimental ddG: "
          f"Pearson r={corr_abl1_pearson:+.4f} (p={p1:.3f}), "
          f"Spearman rho={corr_abl1_spearman:+.4f} (p={p2:.3f})")
    print(f"  Src : masked-marginal LLR vs non_active:       "
          f"Pearson r={corr_src_pearson:+.4f} (p={p3:.3f}), "
          f"Spearman rho={corr_src_spearman:+.4f} (p={p4:.3f})")

    # 4. LOO-CV: Abl1
    print("\n[Abl1] LOO-CV variant C (LLR proxy, 5-dim):")
    abl1_norm = max(abs(v) for v in abl1_llr_vals.values())
    abl1_llr_enc = make_llr_encoders(abl1_llr_vals, abl1_norm, ABL1_SEQ_LEN, 'abl1')
    abl1_llr_res = run_loo_cv(ABL1_DATA, 'non_ground', ABL1_WT_NON_GROUND,
                              ABL1_SEQ_LEN, abl1_llr_enc, 5)
    print(f"  LLR variant C: MAE={abl1_llr_res['mae']:.4f} dir={abl1_llr_res['direction']}")

    print("[Abl1] LOO-CV variant C (experimental ddG, sanity):")
    abl1_ddg_res = run_loo_cv(ABL1_DATA, 'non_ground', ABL1_WT_NON_GROUND,
                              ABL1_SEQ_LEN, encode_abl1_ddg_variant_c, 5)
    print(f"  exp ddG variant C: MAE={abl1_ddg_res['mae']:.4f} dir={abl1_ddg_res['direction']} "
          f"(expect ~0.1046)")

    # 5. LOO-CV: Src
    print("\n[Src] LOO-CV variant C (LLR proxy, 5-dim):")
    src_norm = max(abs(v) for v in src_llr_vals.values())
    src_llr_enc = make_llr_encoders(src_llr_vals, src_norm, SRC_SEQ_LEN, 'src')
    src_llr_res = run_loo_cv(SRC_DATA, 'non_active', SRC_WT_NON_ACTIVE,
                             SRC_SEQ_LEN, src_llr_enc, 5)
    print(f"  LLR variant C: MAE={src_llr_res['mae']:.4f} dir={src_llr_res['direction']}")

    print("[Src] LOO-CV pos-markers 4-dim (sanity):")
    src_pos_res = run_loo_cv(SRC_DATA, 'non_active', SRC_WT_NON_ACTIVE,
                             SRC_SEQ_LEN, encode_src_pos_markers, 4)
    print(f"  pos markers: MAE={src_pos_res['mae']:.4f} dir={src_pos_res['direction']} "
          f"(expect ~0.2508)")

    # 6. Save
    out_dir = Path(__file__).parent / 'results'
    out_dir.mkdir(exist_ok=True)
    output = {
        'experiment': 'esm2_masked_marginal_llr_proxy',
        'description': 'B2: masked-marginal LLR as ddG proxy; Abl1 validation + Src variant C',
        'protocol': {
            'esm2_model': ESM2_MODEL_ID,
            'model_source': 'local models/esm2_650m',
            'llr_protocol': 'ESM-1v masked-marginal (Meier et al. 2021): '
                            'mask target residue + random 15% of others, '
                            f'{N_MASK_REPS} reps averaged, per-position LLR = mean(log P(mut)) - mean(log P(wt))',
            'double_mutant_handling': 'sum of per-position LLR',
            'cdst_protocol': 'LowRankCDST(K=2, rank=2, hidden_dim=32), prob-space MSE, '
                             'Adam(lr=5e-3, wd=1e-4), 800 epochs, 5 seeds, LOO-CV',
            'foldx_available': False,
            'honest_disclosure': 'No FoldX output; all ddG proxies are ESM-2 zero-shot LLR scores',
        },
        'correlations': {
            'abl1_llr_vs_experimental_ddg': {
                'pearson_r': float(corr_abl1_pearson), 'p_value': float(p1),
                'spearman_rho': float(corr_abl1_spearman), 'p_value_s': float(p2),
            },
            'src_llr_vs_non_active': {
                'pearson_r': float(corr_src_pearson), 'p_value': float(p3),
                'spearman_rho': float(corr_src_spearman), 'p_value_s': float(p4),
            },
        },
        'abl1': {
            'llr': abl1_llr_vals,
            'llr_norm_const': float(abl1_norm),
            'experimental_ddg': DDG_DATA,
            'variant_c_llr_proxy': {
                'mae': abl1_llr_res['mae'],
                'direction': abl1_llr_res['direction'],
                'per_mutant': abl1_llr_res['per_mutant'],
                'errors': abl1_llr_res['errors'],
                'direction_detail': abl1_llr_res['direction_detail'],
            },
            'variant_c_experimental_ddg': {
                'mae': abl1_ddg_res['mae'],
                'direction': abl1_ddg_res['direction'],
                'per_mutant': abl1_ddg_res['per_mutant'],
                'errors': abl1_ddg_res['errors'],
                'direction_detail': abl1_ddg_res['direction_detail'],
            },
            'baselines': {'variant_C_exp_ddg': 0.1046, 'Extended_10dim': 0.4134},
        },
        'src': {
            'llr': src_llr_vals,
            'llr_norm_const': float(src_norm),
            'variant_c_llr_proxy': {
                'mae': src_llr_res['mae'],
                'direction': src_llr_res['direction'],
                'per_mutant': src_llr_res['per_mutant'],
                'errors': src_llr_res['errors'],
                'direction_detail': src_llr_res['direction_detail'],
            },
            'pos_markers_4dim': {
                'mae': src_pos_res['mae'],
                'direction': src_pos_res['direction'],
                'per_mutant': src_pos_res['per_mutant'],
                'errors': src_pos_res['errors'],
                'direction_detail': src_pos_res['direction_detail'],
            },
            'baselines': {'pos_markers_4dim': 0.2508, 'Extended_10dim': 0.4443,
                          'old_wt_marginal_llr_variant_c': 0.3320},
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    out_json = out_dir / 'esm2_llr_proxy_results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved: {out_json}")

    # 7. Markdown report
    lines = []
    L = lines.append
    L("# B2 报告: ESM-2 masked-marginal LLR 作为 ΔΔG 代理")
    L("")
    L(f"> 时间: {output['timestamp']} | 协议: {output['protocol']['llr_protocol']}")
    L("")
    L("## 1. 背景")
    L("")
    L("FoldX 5.1 不可用（需人工学术注册）。上一轮用 wild-type-marginal LLR 作代理：")
    L("- Src 变体 C MAE=0.3320（+32.4% 劣于位置标记基线 0.2508）")
    L("- LLR 与 non_active 相关弱（Spearman ρ=-0.2455）")
    L("")
    L("本轮升级为 **ESM-1v masked-marginal 协议**（ProteinGym 标准）：掩码目标残基 + 随机 15% 其他残基，"
      f"{N_MASK_REPS} 次独立前向平均 log 概率。")
    L("")
    L("## 2. 相关性")
    L("")
    L("| 对比 | Pearson r | Spearman ρ |")
    L("|---|---:|---:|")
    L(f"| Abl1: LLR vs 实验 ΔΔG | {corr_abl1_pearson:+.4f} | {corr_abl1_spearman:+.4f} |")
    L(f"| Src: LLR vs non_active | {corr_src_pearson:+.4f} | {corr_src_spearman:+.4f} |")
    L("")
    L("## 3. Abl1 LOO-CV（LLR 代理 vs 实验 ΔΔG）")
    L("")
    L("| 编码 | MAE | 方向 |")
    L("|---|---:|:---:|")
    L(f"| 变体C + 实验 ΔΔG（sanity） | {abl1_ddg_res['mae']:.4f} | {abl1_ddg_res['direction']} |")
    L(f"| 变体C + LLR 代理 | {abl1_llr_res['mae']:.4f} | {abl1_llr_res['direction']} |")
    L("")
    L("逐突变：")
    L("")
    L("| Mutant | true | exp-ΔΔG pred | LLR pred |")
    L("|---|---:|---:|---:|")
    for m in ABL1_DATA:
        L(f"| {m} | {ABL1_DATA[m]['non_ground']:.2f} | "
          f"{abl1_ddg_res['per_mutant'][m]:.3f} | {abl1_llr_res['per_mutant'][m]:.3f} |")
    L("")
    L("## 4. Src LOO-CV（LLR 代理 vs 位置标记基线）")
    L("")
    L("| 编码 | MAE | 方向 |")
    L("|---|---:|:---:|")
    L(f"| 位置标记 4-dim（sanity） | {src_pos_res['mae']:.4f} | {src_pos_res['direction']} |")
    L(f"| 变体C + LLR 代理 | {src_llr_res['mae']:.4f} | {src_llr_res['direction']} |")
    L("")
    L("逐突变：")
    L("")
    L("| Mutant | true | pos-marker pred | LLR pred |")
    L("|---|---:|---:|---:|")
    for m in SRC_DATA:
        L(f"| {m} | {SRC_DATA[m]['non_active']:.2f} | "
          f"{src_pos_res['per_mutant'][m]:.3f} | {src_llr_res['per_mutant'][m]:.3f} |")
    L("")
    L("## 5. 判定")
    L("")
    abl1_gain = abl1_llr_res['mae'] / abl1_ddg_res['mae']
    src_gain = src_llr_res['mae'] / src_pos_res['mae']
    L(f"- Abl1: LLR 代理 MAE 是实验 ΔΔG 的 **{abl1_gain:.2f}×**"
      f"（{abl1_llr_res['mae']:.4f} vs {abl1_ddg_res['mae']:.4f}）")
    L(f"- Src: LLR 代理 MAE 是位置标记基线的 **{src_gain:.2f}×**"
      f"（{src_llr_res['mae']:.4f} vs {src_pos_res['mae']:.4f}）")
    L("")
    L("### 5.1 结论")
    L("")
    L("见实验输出。若 LLR 代理未优于位置标记基线，结论为："
      "ESM-2 zero-shot 分数（wild-type-marginal 与 masked-marginal 均）无法替代实验 ΔΔG 作为 Src 中间特征；"
      "Src 的最佳可用编码仍是纯位置标记 4-dim。")
    L("")
    L("---")
    L("")
    L("*脚本: `experiments/foldx_src/run_esm2_llr_proxy.py` | 结果: `results/esm2_llr_proxy_results.json`*")

    out_md = out_dir / 'esm2_llr_proxy_report.md'
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Report saved: {out_md}")


if __name__ == '__main__':
    main()
