"""
SubTask 5.1-5.4: 计算 Src 8 突变体 ΔΔG 替代值.

诚实声明 (HONEST DISCLOSURE):
=============================
FoldX 5.1 在本自动化环境中不可用:
  1. FoldX 二进制未安装于系统 (Get-Command foldx 无输出)
  2. FoldX 学术许可证需在 https://foldxsuite.crg.eu/ 注册, 经人工审核后
     邮件发送下载链接 — 自动化环境无法完成此人工流程
  3. 任务明确要求 "若注册失败或邮件未到达, 立即停止并报告失败",
     "不要伪造任何 FoldX 输出"

依据任务备选方案 2, 采用 ESM-2 zero-shot LLR (log-likelihood ratio) 标量
作为 ΔΔG 替代. 该方法在 Task 1 (experiments/iclr_restructuring/esm2_encoding.py)
中已验证可行, 且 ESM-2 650M 模型已本地存在于 models/esm2_650m/.

方法 (ESM-2 zero-shot mutation effect prediction, 标准 ESM-1v/ESM-2 协议):
  LLR(mut) = log P(mut_aa | WT context) - log P(wt_aa | WT context)
  ΔΔG_proxy = -LLR   (正值 = 去稳定, 与 FoldX 约定一致)
  双突变: 取两个位置 LLR 之和 (与项目 _aa_delta 求和约定一致)

参考文献:
  - Meier et al., Nat. Biotechnol. 2021 (ESM-1v zero-shot)
  - Lin et al., Science 2023 (ESM-2)
  - 任务说明: "使用 ESM-2 zero-shot LLR 标量作为 ΔΔG 替代 (已在 Task 1 中验证可行)"

PDB 1Y57 已从 RCSB 公开下载 (experiments/foldx_src/1y57.pdb), 作为结构参考.
未运行 FoldX RepairPDB / BuildModel (binary 不可用).
"""
import sys
import os
import json
import time
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Src 序列 (UniProt P12931 SRC_HUMAN SV=3, 536 残基)
# 与 experiments/iclr_restructuring/esm2_encoding.py 完全一致
# ============================================================
SRC_FULL = (
    "MGSNKSKPKDASQRRRSLEPAENVHGAGGGAFPASQTPSKPASADGHRGPSAAFAPAAAEPKLFGGFNSSD"
    "TVTSPQRAGPLAGGVTTFVALYDYESRTETDLSFKKGERLQIVNNTEGDWWLAHSLSTGQTGYIPSNYVAP"
    "SDSIQAEEWYFGKITRRESERLLLNAENPRGTFLVRESETTKGAYCLSVSDFDNAKGLNVKHYKIRKLDSG"
    "GFYITSRTQFNSLQQLVAYYSKHADGLCHRLTTVCPTSKPQTQGLAKDAWEIPRESLRLEVKLGQGCFGEV"
    "WMGTWNGTTRVAIKTLKPGTMSPEAFLQEAQVMKKLRHEKLVQLYAVVSEEPIYIVTEYMSKGSLLDFLKG"
    "ETGKYLRLPQLVDMAAQIASGMAYVERMNYVHRDLRAANILVGENLVCKVADFGLARLIEDNEYTARQGAK"
    "FPIKWTAPEAALYGRFTIKSDVWSFGILLTELTTKGRVPYPGMVNREVLDQVERGYRMPCPPECPESLHDL"
    "MCQCWRKEPEERPTFEYLQAFLEDYFTSTEPQYQPGENL"
)
SRC_SEQ_LEN = len(SRC_FULL)  # 536

# Src 8 突变体 (来自 encoding_ablation_control.py 的 SRC_DATA)
# non_active = 1 - Active population (来自 Fig S5 Met305 probe)
SRC_DATA = {
    'SrcKD-L410A':         {'wt': 'L', 'mut': 'A', 'pos': 410, 'non_active': 0.27},
    'SrcKD-V332I':         {'wt': 'V', 'mut': 'I', 'pos': 332, 'non_active': 0.52},
    'SrcKD-L270F_V332I':   {'wt': 'LV', 'mut': 'FI', 'pos': 270, 'non_active': 0.91},
    'SrcKD-L325A':         {'wt': 'L', 'mut': 'A', 'pos': 325, 'non_active': 1.00},
    'SrcKD-A311I':         {'wt': 'A', 'mut': 'I', 'pos': 311, 'non_active': 1.00},
    'SrcKD-V380A':         {'wt': 'V', 'mut': 'A', 'pos': 380, 'non_active': 1.00},
    'SrcKD-V331A':         {'wt': 'V', 'mut': 'A', 'pos': 331, 'non_active': 1.00},
    'SrcKD-F405A':         {'wt': 'F', 'mut': 'A', 'pos': 405, 'non_active': 1.00},
}
SRC_WT_NON_ACTIVE = 0.28  # 1 - 0.72

# 本地 ESM-2 650M 模型路径
ESM2_LOCAL_PATH = str(Path(__file__).parent.parent.parent / "models" / "esm2_650m")
ESM2_MODEL_ID = "facebook/esm2_t33_650M_UR50D"


# ============================================================
# 位置映射 (复用 esm2_encoding.py 的 find_position 逻辑)
# ============================================================
def find_position(seq, nominal_pos, wt_aa, system='src'):
    """在 seq 中找到 wt_aa 的 0-indexed 位置, 最接近 nominal_pos (1-indexed).

    Src 位置映射: 项目 pos p → UniProt P12931 1-indexed → 0-indexed (p-1).
    若 nominal 位置不匹配, 搜索 ±10 范围内最近的 wt_aa.
    返回 (idx, offset).
    """
    nominal_idx = nominal_pos - 1

    if 0 <= nominal_idx < len(seq) and seq[nominal_idx] == wt_aa:
        return nominal_idx, 0

    best_idx, best_dist = None, float('inf')
    for offset in range(-10, 11):
        idx = nominal_idx + offset
        if 0 <= idx < len(seq) and seq[idx] == wt_aa:
            if abs(offset) < best_dist:
                best_dist = abs(offset)
                best_idx = idx

    if best_idx is not None:
        return best_idx, best_idx - nominal_idx

    raise ValueError(
        f"Cannot find {wt_aa} near position {nominal_pos} (idx {nominal_idx}) "
        f"in Src sequence (len {len(seq)})"
    )


def parse_mutations_from_name(name, data):
    """从突变名解析突变列表 [(pos, wt, mut), ...].

    单突变: 用 data 中的 pos/wt/mut.
    双突变: 从名称解析 (如 'SrcKD-L270F_V332I' → [(270,'L','F'), (332,'V','I')]).
    """
    clean = name.replace('SrcKD-', '')
    if '_' not in clean:
        return [(data['pos'], data['wt'], data['mut'])]

    parts = clean.split('_')
    muts = []
    for part in parts:
        m = re.match(r'([A-Z])(\d+)([A-Z])', part)
        if m:
            muts.append((int(m.group(2)), m.group(1), m.group(3)))
    if len(muts) != 2:
        raise ValueError(f"Cannot parse double mutant from name: {name}")
    return muts


# ============================================================
# ESM-2 zero-shot LLR 计算
# ============================================================
def load_esm2_mlm():
    """加载 ESM-2 650M 带 MaskedLM head (用于 zero-shot LLR)."""
    from transformers import EsmForMaskedLM, EsmTokenizer

    source = ESM2_LOCAL_PATH if Path(ESM2_LOCAL_PATH).exists() else ESM2_MODEL_ID
    print(f"加载 ESM-2 (MaskedLM): {source}")
    t0 = time.time()
    tokenizer = EsmTokenizer.from_pretrained(source)
    model = EsmForMaskedLM.from_pretrained(source)
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)
    print(f"  模型加载完成, 耗时 {time.time()-t0:.1f}s")
    return model, tokenizer, device


def compute_llr_scores(model, tokenizer, wt_seq, mutations, device='cpu'):
    """计算每个突变体的 zero-shot LLR.

    LLR(mut) = log P(mut_aa | WT context) - log P(wt_aa | WT context)
    ΔΔG_proxy = -LLR  (正值 = 去稳定, 与 FoldX 约定一致)

    双突变: 取两个位置 LLR 之和.

    返回:
      results: {name: {'llr': float, 'ddg_proxy': float, 'positions': [...]}}
    """
    print(f"\n计算 WT 序列 logits (len={len(wt_seq)})...")
    t0 = time.time()
    inputs = tokenizer(wt_seq, return_tensors='pt', truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # logits: (1, seq_len+2, vocab_size), 包含 BOS/EOS
    logits = outputs.logits[0]  # (seq_len+2, vocab_size)
    # log_softmax 得到 log 概率
    log_probs = F.log_softmax(logits, dim=-1)
    print(f"  logits shape: {logits.shape}, 耗时 {time.time()-t0:.1f}s")

    # 氨基酸 -> token id (ESM tokenizer 用 ' <aa>' 形式, 首个特殊 token 后)
    # 标准 ESM tokenizer: 氨基酸 token 即单字母大写
    aa_tokens = {}
    for aa in set(sum([list(d['wt']) + list(d['mut']) for d in mutations.values()], [])):
        if aa not in aa_tokens:
            # ESM tokenizer: 直接用单字母
            tid = tokenizer.convert_tokens_to_ids(aa)
            aa_tokens[aa] = tid

    print(f"\n氨基酸 token IDs: {aa_tokens}")

    results = {}
    print(f"\n{'Mutant':<25} {'pos':>6} {'wt→mut':>8} {'idx':>5} {'LLR':>8} {'ΔΔG_proxy':>10}")
    print("-" * 75)

    for name, data in mutations.items():
        mut_list = parse_mutations_from_name(name, data)
        total_llr = 0.0
        pos_info = []

        for pos, wt_aa, mut_aa in mut_list:
            idx, offset = find_position(wt_seq, pos, wt_aa, 'src')
            # logits 中位置: idx + 1 (因为 BOS token 在位置 0)
            logit_idx = idx + 1

            wt_tok = aa_tokens[wt_aa]
            mut_tok = aa_tokens[mut_aa]

            llr_pos = float(log_probs[logit_idx, mut_tok] - log_probs[logit_idx, wt_tok])
            total_llr += llr_pos
            pos_info.append({
                'nominal_pos': pos,
                'seq_idx': idx,
                'offset': offset,
                'wt': wt_aa,
                'mut': mut_aa,
                'llr_single': llr_pos,
            })

        ddg_proxy = -total_llr  # ΔΔG_proxy = -LLR
        results[name] = {
            'llr': float(total_llr),
            'ddg_proxy': float(ddg_proxy),
            'positions': pos_info,
            'non_active': data['non_active'],
        }
        pos_str = '+'.join(str(p['nominal_pos']) for p in pos_info)
        wt_mut_str = '+'.join(f"{p['wt']}→{p['mut']}" for p in pos_info)
        idx_str = '+'.join(str(p['seq_idx']) for p in pos_info)
        print(f"  {name:<23} {pos_str:>6} {wt_mut_str:>8} {idx_str:>5} "
              f"{total_llr:>+8.4f} {ddg_proxy:>+10.4f}")

    return results


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 80)
    print("SubTask 5.1-5.4: Src ΔΔG 计算 (ESM-2 zero-shot LLR 备选方案)")
    print("=" * 80)
    print("FoldX 不可用 (未安装 + 需学术注册), 采用 ESM-2 LLR 作为 ΔΔG 替代.")
    print("PDB 1Y57 已下载作为结构参考 (未运行 FoldX RepairPDB/BuildModel).")
    print()

    # 1. 加载 ESM-2
    try:
        model, tokenizer, device = load_esm2_mlm()
    except Exception as e:
        print(f"\n!!! ESM-2 模型加载失败: {e}")
        print("实验终止. 未生成结果文件.")
        return

    # 2. 位置验证
    print("\n[1/3] 位置验证与突变解析:")
    pos_map = {}
    for name, data in SRC_DATA.items():
        mut_list = parse_mutations_from_name(name, data)
        entries = []
        for pos, wt_aa, mut_aa in mut_list:
            idx, offset = find_position(SRC_FULL, pos, wt_aa, 'src')
            entries.append((idx, offset, wt_aa, mut_aa, pos))
            status = "OK" if offset == 0 else f"adjusted ({offset:+d})"
            print(f"  {name}: {wt_aa}{pos}{mut_aa} → seq[{idx}]={SRC_FULL[idx]} ({status})")
        pos_map[name] = entries

    # 3. 计算 LLR
    print("\n[2/3] ESM-2 zero-shot LLR 计算:")
    results = compute_llr_scores(model, tokenizer, SRC_FULL, SRC_DATA, device)

    # 4. 归一化 (用于变体 C 编码)
    ddg_values = {n: r['ddg_proxy'] for n, r in results.items()}
    ddg_norm = max(abs(v) for v in ddg_values.values())
    print(f"\n  ΔΔG_proxy 归一化常数 (max|ΔΔG|) = {ddg_norm:.4f}")
    print(f"  归一化后 ΔΔG_norm ∈ [-1, 1]:")

    # 计算与 non_active 的相关性
    non_active_vals = [SRC_DATA[n]['non_active'] for n in SRC_DATA]
    ddg_vals = [ddg_values[n] for n in SRC_DATA]
    wt_non = SRC_WT_NON_ACTIVE
    true_delta = [SRC_DATA[n]['non_active'] - wt_non for n in SRC_DATA]

    # Pearson 相关系数
    ddg_arr = np.array(ddg_vals)
    target_arr = np.array(non_active_vals)
    if np.std(ddg_arr) > 1e-9 and np.std(target_arr) > 1e-9:
        corr_pearson = float(np.corrcoef(ddg_arr, target_arr)[0, 1])
    else:
        corr_pearson = 0.0

    # Spearman (rank) 相关
    from scipy.stats import spearmanr
    corr_spearman, _ = spearmanr(ddg_vals, non_active_vals)

    print(f"\n  相关性分析:")
    print(f"    ΔΔG_proxy vs non_active:  Pearson  r = {corr_pearson:+.4f}")
    print(f"    ΔΔG_proxy vs non_active:  Spearman ρ = {corr_spearman:+.4f}")
    print(f"    (负相关合理: 去稳定突变 → ΔΔG↑ → non_active↑, 但 LLR 与 active↑ 关系复杂)")

    # 5. 保存 JSON
    print("\n[3/3] 保存结果至 src_ddg_results.json:")
    output = {
        'experiment': 'src_ddg_computation',
        'method': 'ESM-2 zero-shot LLR (FoldX unavailable - backup option 2)',
        'honest_disclosure': {
            'foldx_available': False,
            'foldx_unavailable_reason': (
                'FoldX binary not installed; academic license registration at '
                'https://foldxsuite.crg.eu/ requires human review + email delivery, '
                'which cannot be completed in automated environment.'
            ),
            'backup_method': 'ESM-2 zero-shot log-likelihood ratio (LLR) as ΔΔG proxy',
            'backup_validation': 'Validated feasible in Task 1 (esm2_encoding.py)',
            'no_fake_foldx_output': True,
        },
        'foldx_version_intended': '5.1 (not used)',
        'esm2_model': ESM2_MODEL_ID,
        'esm2_model_source': ESM2_LOCAL_PATH if Path(ESM2_LOCAL_PATH).exists() else ESM2_MODEL_ID,
        'esm2_embed_dim': 1280,
        'pdb_reference': {
            'id': '1Y57',
            'source': 'RCSB https://www.rcsb.org/structure/1Y57',
            'description': 'Unphosphorylated c-Src in complex with inhibitor (SH3+SH2+kinase)',
            'local_path': 'experiments/foldx_src/1y57.pdb',
            'foldx_repair_run': False,
            'foldx_buildmodel_run': False,
        },
        'sequence': {
            'source': 'UniProt P12931 SRC_HUMAN SV=3',
            'length': SRC_SEQ_LEN,
        },
        'method_details': {
            'llr_formula': 'LLR = log P(mut_aa | WT context) - log P(wt_aa | WT context)',
            'ddg_proxy_formula': 'ΔΔG_proxy = -LLR (positive = destabilizing, matches FoldX sign convention)',
            'double_mutant_handling': 'sum of per-position LLR (consistent with project _aa_delta sum convention)',
            'normalization': 'ΔΔG_norm = ΔΔG_proxy / max(|ΔΔG_proxy|) over 8 mutants',
        },
        'src_data': {n: {**d} for n, d in SRC_DATA.items()},
        'src_wt_non_active': SRC_WT_NON_ACTIVE,
        'ddg_results': {},
        'ddg_norm_constant': float(ddg_norm),
        'correlation_with_non_active': {
            'pearson_r': corr_pearson,
            'spearman_rho': float(corr_spearman),
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    for name, r in results.items():
        output['ddg_results'][name] = {
            'llr': r['llr'],
            'ddg_proxy': r['ddg_proxy'],
            'ddg_norm': r['ddg_proxy'] / ddg_norm if ddg_norm > 0 else 0.0,
            'non_active': r['non_active'],
            'true_delta_non_active': r['non_active'] - SRC_WT_NON_ACTIVE,
            'positions': r['positions'],
        }

    out_path = Path(__file__).parent / 'src_ddg_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"  结果已保存: {out_path}")

    # 6. 汇总表
    print(f"\n{'='*80}")
    print("汇总: Src 8 突变体 ΔΔG_proxy (ESM-2 LLR)")
    print(f"{'='*80}")
    print(f"{'Mutant':<25} {'LLR':>8} {'ΔΔG_proxy':>10} {'ΔΔG_norm':>9} "
          f"{'non_active':>10} {'trueΔ':>7}")
    print("-" * 80)
    for name in SRC_DATA:
        r = output['ddg_results'][name]
        print(f"  {name:<23} {r['llr']:>+8.4f} {r['ddg_proxy']:>+10.4f} "
              f"{r['ddg_norm']:>+9.4f} {r['non_active']:>10.2f} "
              f"{r['true_delta_non_active']:>+7.2f}")
    print(f"\n  Pearson r = {corr_pearson:+.4f}")
    print(f"  Spearman ρ = {corr_spearman:+.4f}")

    return output


if __name__ == '__main__':
    main()
