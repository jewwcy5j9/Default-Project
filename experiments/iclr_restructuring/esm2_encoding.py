"""
ESM-2 per-residue delta + PCA encoding experiment (Task 1).

验证 ESM-2 650M per-residue embedding 差值, 经 PCA 降维至 10/15/20 dim 后,
能否在 Abl1 + Src 上与现有编码方案竞争.

编码方式 (方法 A): Δe = E_mut(p) − E_wt(p), p 为突变位置.
  双突变: 取两个位置差值之和 (与 _aa_delta 在 encoding_ablation_control.py 中的求和约定一致).

PCA 策略: 在所有突变体所有残基位置的 delta 向量上拟合 PCA (n_mutants × seq_len 样本),
  然后变换突变位置的 delta 向量. 这样既有足够样本支撑 10-20 dim, 又捕获 delta 空间方差.

协议 (与 alternative_encodings.py 完全一致):
  - LowRankCDST(K=2, intervention_dim=d, rank=2, hidden_dim=32)
  - prob-space MSE: F.mse_loss(model(w, c), target)
  - Adam(lr=5e-3, weight_decay=1e-4), 800 epochs, best-state 追踪
  - 5 seeds, seed = seed*100 + hold_out

现有对照:
  - Abl1 变体 C (ΔΔG 主特征 5-dim): MAE=0.1046
  - Src 纯位置标记 4-dim:           MAE=0.2508
  - Extended 10-dim (canonical):    MAE=0.4134 (Abl1) / 0.4443 (Src)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import re
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from src.models.low_rank_cdst import LowRankCDST
from encoding_ablation_control import (
    ABL1_DATA, ABL1_WT_NON_GROUND, ABL1_SEQ_LEN,
    SRC_DATA, SRC_WT_NON_ACTIVE, SRC_SEQ_LEN,
)


# ============================================================
# WT 序列
# ============================================================

# Abl1a 全长 (UniProt P00519-1), 来源: scripts/generate_af3_fasta_v2.py
ABL1A_FULL = (
    "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGGLSEAARWNSKENLLAGPSENDPNLFVALY"
    "DFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLL"
    "SSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITT"
    "LHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEF"
    "LKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYL"
    "EKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAF"
    "GVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETM"
    "FQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPA"
    "VSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERRGAGEEEG"
    "RDISNGALAFTPLDTADPAKSPKPSNGAGVPNGALRESGGSGFRSPHLWKKSSTLTSSRLATGEEEGGGSS"
    "SKRFLRSCSASCVPHGAKDTEWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRGTV"
    "TPPPRLVKKNEEAADEVFKDIMESSPGSSPPNLTPKPLRRQVTVAPASGLPHKEEAGKGSALGTPAAAEPV"
    "TPTSKAGSGAPGGTSKGPAEESRVRRHKHSSESPGRDKGKLSRLKPAPPPPPAASAGKAGGKPSQSPSQEA"
    "AGEAVLGAKTKATSLVDAVNSDAAKPSQPGEGLKKPVLPATPKPQSAKPSGTPISPAPVPSTLPSASSALA"
    "GDQPSSTAFIPLISTRVSLRKTRQPPERIASGAITKGVVLDSTEALCLAISRNSEQMASHSAVLEAGKNLY"
    "TFCVSYVDSIQQMRNKFAFREAINKLENNLRELQICPATAGSGPAATQDFSKLLSSVKEISDIVQR"
)

# Abl1 激酶域 (Monteiro 2024: Abl1a 229-515, 287 残基)
ABL1_KD = ABL1A_FULL[228:515]

# Src 全长 (UniProt P12931 SRC_HUMAN SV=3, 536 残基)
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

ESM2_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
ESM2_EMBED_DIM = 1280

# 本地模型路径 (从 ModelScope 下载, 因 huggingface.co 在此网络环境不可达)
# 若本地路径存在则使用, 否则回退到 HF hub id (需网络)
ESM2_LOCAL_PATH = str(Path(__file__).parent.parent.parent / "models" / "esm2_650m")
ESM2_SOURCE = ESM2_LOCAL_PATH if Path(ESM2_LOCAL_PATH).exists() else ESM2_MODEL_ID
PCA_DIMS = [10, 15, 20]

# 现有对照基线 (来自已有实验结果, 不重新运行)
BASELINES = {
    'abl1': {
        'variant_C_ddg_5dim': 0.1046,
        'Extended_10dim': 0.4134,
    },
    'src': {
        'pos_markers_4dim': 0.2508,
        'Extended_10dim': 0.4443,
    },
}


# ============================================================
# 位置映射与验证
# ============================================================

def find_position(seq, nominal_pos, wt_aa, system='abl1'):
    """在 seq 中找到 wt_aa 的 0-indexed 位置, 最接近 nominal_pos (1-indexed).

    Abl1 位置映射: 项目 pos p → Abl1a pos (p+1) → KD index (p+1-229) = p-228.
      (项目编号比 Abl1a 统一偏移 -1, 已验证: M290→KD[62], L301→KD[73], F382→KD[154])
    Src 位置映射: 项目 pos p → UniProt P12931 1-indexed → 0-indexed (p-1).
      (大部分精确匹配; A311I 实际 A 在 309, F405A 实际 DFG Phe 在 408, 因序列版本差异)

    若 nominal 位置不匹配, 搜索 ±10 范围内最近的 wt_aa.
    """
    if system == 'abl1':
        nominal_idx = nominal_pos - 228
    else:
        nominal_idx = nominal_pos - 1

    if 0 <= nominal_idx < len(seq) and seq[nominal_idx] == wt_aa:
        return nominal_idx, 0  # exact match, offset=0

    best_idx, best_dist = None, float('inf')
    for offset in range(-10, 11):
        idx = nominal_idx + offset
        if 0 <= idx < len(seq) and seq[idx] == wt_aa:
            if abs(offset) < best_dist:
                best_dist = abs(offset)
                best_idx = idx

    if best_idx is not None:
        return best_idx, best_dist

    raise ValueError(
        f"Cannot find {wt_aa} near position {nominal_pos} (idx {nominal_idx}) "
        f"in {system} sequence (len {len(seq)})"
    )


def parse_mutations_from_name(name, data):
    """从突变名解析突变列表 [(pos, wt, mut), ...].

    单突变: 用 data 中的 pos/wt/mut.
    双突变: 从名称解析两个突变 (如 'M290L_L301I' → [(290,'M','L'), (301,'L','I')]).
      注意 ABL1_DATA 中 M290L_L301I 的 pos=301 (第二个位置), 不能直接用于第一个突变.
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


def verify_positions(mutations, wt_seq, system):
    """验证所有突变位置, 返回 {name: [(idx, offset, wt, mut, nominal_pos), ...]}."""
    pos_map = {}
    print(f"\n  [{system}] 位置验证 (WT seq len={len(wt_seq)}):")
    for name, data in mutations.items():
        mut_list = parse_mutations_from_name(name, data)
        entries = []
        for pos, wt_aa, mut_aa in mut_list:
            idx, offset = find_position(wt_seq, pos, wt_aa, system)
            entries.append((idx, offset, wt_aa, mut_aa, pos))

        pos_map[name] = entries
        for idx, offset, w, m, p in entries:
            status = "OK" if offset == 0 else f"adjusted (+{offset})"
            print(f"    {name}: {w}{p}{m} → seq[{idx}]={wt_seq[idx]} ({status})")

    return pos_map


def generate_mutant_seq(wt_seq, pos_entries):
    """根据位置映射生成突变体序列. pos_entries = [(idx, offset, wt, mut, pos), ...]."""
    seq = list(wt_seq)
    for idx, _, wt, mut, _ in pos_entries:
        assert seq[idx] == wt, f"Expected {wt} at index {idx}, got {seq[idx]}"
        seq[idx] = mut
    return ''.join(seq)


# ============================================================
# ESM-2 模型加载与推理
# ============================================================

def load_esm2():
    """加载 ESM-2 650M 模型 (CPU). 优先从本地路径加载, 回退到 HF hub."""
    from transformers import EsmTokenizer, EsmModel

    print(f"\n加载 ESM-2 模型: {ESM2_SOURCE}")
    if ESM2_SOURCE == ESM2_LOCAL_PATH:
        print(f"  (从本地路径加载: {ESM2_LOCAL_PATH})")
    else:
        print("  (首次运行需下载 ~2.5GB, 请耐心等待...)")
    t0 = time.time()
    tokenizer = EsmTokenizer.from_pretrained(ESM2_SOURCE)
    model = EsmModel.from_pretrained(ESM2_SOURCE)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  模型加载完成, 耗时 {time.time()-t0:.1f}s")
    return model, tokenizer, device


def per_residue_embedding(model, tokenizer, seq, device='cpu'):
    """计算 per-residue embedding. 返回 (seq_len, 1280) numpy 数组.

    去掉 BOS/EOS 特殊 token.
    """
    inputs = tokenizer(seq, return_tensors='pt', truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model(**inputs)
    # last_hidden_state: (1, seq_len+2, 1280), 去掉首尾 BOS/EOS
    emb = out.last_hidden_state[0, 1:-1].cpu().numpy()
    return emb


def compute_all_embeddings(model, tokenizer, wt_seq, mutations, pos_map, system, device='cpu'):
    """计算 WT 和所有突变体的 per-residue embedding.

    返回:
      embeddings: {name: (seq_len, 1280) array}
      mut_positions: {name: [idx1, idx2, ...]} 突变位置 (0-indexed)
    """
    embeddings = {}
    mut_positions = {}

    # WT
    print(f"  [{system}] 计算 WT embedding (len={len(wt_seq)})...")
    t0 = time.time()
    embeddings['WT'] = per_residue_embedding(model, tokenizer, wt_seq, device)
    print(f"    WT: {embeddings['WT'].shape}, {time.time()-t0:.1f}s")

    # 突变体
    for name in mutations:
        mut_seq = generate_mutant_seq(wt_seq, pos_map[name])
        positions = [e[0] for e in pos_map[name]]
        mut_positions[name] = positions

        # 验证序列确实不同
        diffs = [i for i in range(len(wt_seq)) if wt_seq[i] != mut_seq[i]]
        assert len(diffs) == len(positions), \
            f"{name}: expected {len(positions)} mutations, found {len(diffs)}"

        print(f"  [{system}] 计算 {name} embedding (len={len(mut_seq)})...")
        t0 = time.time()
        embeddings[name] = per_residue_embedding(model, tokenizer, mut_seq, device)
        print(f"    {name}: {embeddings[name].shape}, {time.time()-t0:.1f}s")

    return embeddings, mut_positions


# ============================================================
# Delta 计算与 PCA
# ============================================================

def compute_delta_encodings(embeddings, mut_positions, wt_key='WT'):
    """计算每个突变体的 delta 编码向量 (1280-dim).

    单突变: Δe = E_mut[pos] - E_wt[pos]
    双突变: Δe = Σ (E_mut[pos_i] - E_wt[pos_i])  (求和, 与 _aa_delta 约定一致)

    返回:
      delta_vectors: {name: (1280,) array}
      all_delta_rows: (n_mutants * seq_len, 1280) 用于 PCA 拟合
    """
    e_wt = embeddings[wt_key]
    delta_vectors = {}
    all_delta_rows = []

    for name, positions in mut_positions.items():
        e_mut = embeddings[name]
        delta_matrix = e_mut - e_wt  # (seq_len, 1280)
        all_delta_rows.append(delta_matrix)

        # 突变位置的 delta (求和)
        delta_vec = np.zeros(e_wt.shape[1])
        for p in positions:
            delta_vec += delta_matrix[p]
        delta_vectors[name] = delta_vec

    all_delta_rows = np.vstack(all_delta_rows)  # (n_mutants * seq_len, 1280)
    return delta_vectors, all_delta_rows


def fit_pca_and_reduce(delta_vectors, all_delta_rows, dims=PCA_DIMS):
    """在 all_delta_rows 上拟合 PCA, 然后变换 delta_vectors.

    返回:
      pca_model: 拟合的 PCA 对象 (n_components=max(dims))
      reduced: {dim: {name: (dim,) array}}
      variance_explained: {dim: cumulative_variance}
    """
    max_dim = max(dims)
    pca = PCA(n_components=max_dim)
    pca.fit(all_delta_rows)

    # 计算各 dim 的累积方差解释率
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    variance_explained = {d: float(cum_var[d - 1]) for d in dims}

    # 变换 delta_vectors
    names = list(delta_vectors.keys())
    delta_matrix = np.array([delta_vectors[n] for n in names])  # (n_mutants, 1280)
    reduced_full = pca.transform(delta_matrix)  # (n_mutants, max_dim)

    reduced = {}
    for d in dims:
        reduced[d] = {names[i]: reduced_full[i, :d] for i in range(len(names))}

    return pca, reduced, variance_explained


# ============================================================
# LOO-CV (与 alternative_encodings.py 协议完全一致)
# ============================================================

def run_loo_cv(encodings, mutations, target_key, wt_non_target, d,
               n_seeds=5, n_epochs=800):
    """LowRankCDST LOO-CV.

    协议 (与 alternative_encodings.py 的 run_loo_cv 一致):
      - LowRankCDST(K=2, intervention_dim=d, rank=2, hidden_dim=32)
      - Adam(lr=5e-3, weight_decay=1e-4)
      - 800 epochs, best-state 追踪 (按 train loss)
      - 5 seeds, seed=seed*100+hold_out
      - loss = F.mse_loss(model(w, c), target)  # prob-space
    """
    names = list(mutations.keys())
    n = len(names)

    wt_dist = np.array([1 - wt_non_target, wt_non_target])
    w_wt = np.tile(wt_dist, (n, 1))
    w_target = np.array([[1 - mutations[m][target_key], mutations[m][target_key]]
                         for m in names])
    c = np.array([encodings[m] for m in names])
    assert c.shape[1] == d, f"Encoding dim mismatch: {c.shape[1]} vs {d}"

    all_preds = {m: [] for m in names}

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
            for _ in range(n_epochs):
                model.train()
                optimizer.zero_grad()
                pred = model(w_t, c_t)
                loss = F.mse_loss(pred, wt_t)  # prob-space MSE
                loss.backward()
                optimizer.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                pred = model(
                    torch.FloatTensor(w_wt[hold_out:hold_out + 1]),
                    torch.FloatTensor(c[hold_out:hold_out + 1])
                )
                all_preds[names[hold_out]].append(float(pred.numpy()[0, 1]))

    return {m: float(np.mean(v)) for m, v in all_preds.items()}


def compute_metrics(preds, mutations, target_key, wt_non_target):
    """计算 MAE / 方向准确率 / per-mutant 误差."""
    names = list(mutations.keys())
    errors = {m: abs(preds[m] - mutations[m][target_key]) for m in names}
    mae = float(np.mean(list(errors.values())))
    median = float(np.median(list(errors.values())))

    dir_correct, dir_total = 0, 0
    dir_detail = {}
    for m in names:
        d_true = mutations[m][target_key] - wt_non_target
        d_pred = preds[m] - wt_non_target
        if abs(d_true) < 0.05:
            dir_detail[m] = 'TIE'
            continue
        dir_total += 1
        if np.sign(d_true) == np.sign(d_pred):
            dir_correct += 1
            dir_detail[m] = 'OK'
        else:
            dir_detail[m] = 'WRONG'

    return {
        'per_mutant': preds,
        'mae': mae,
        'median': median,
        'errors': {m: float(e) for m, e in errors.items()},
        'direction': f"{dir_correct}/{dir_total}",
        'direction_pct': dir_correct / dir_total if dir_total > 0 else 0.0,
        'direction_detail': dir_detail,
    }


# ============================================================
# 主流程
# ============================================================

def run_system(system_name, mutations, target_key, wt_non_target, wt_seq, system,
               model, tokenizer, device):
    """在一个体系 (Abl1 或 Src) 上运行完整的 ESM-2 编码实验."""
    print(f"\n{'='*80}")
    print(f"体系: {system_name} (n={len(mutations)} 突变体)")
    print(f"{'='*80}")

    # 1. 位置验证
    print(f"\n[1/4] 位置验证与突变体序列生成...")
    pos_map = verify_positions(mutations, wt_seq, system)

    # 2. ESM-2 embedding
    print(f"\n[2/4] ESM-2 per-residue embedding 计算...")
    embeddings, mut_positions = compute_all_embeddings(
        model, tokenizer, wt_seq, mutations, pos_map, system, device
    )

    # 3. Delta + PCA
    print(f"\n[3/4] Delta 计算与 PCA 降维...")
    delta_vectors, all_delta_rows = compute_delta_encodings(embeddings, mut_positions)
    print(f"  Delta 矩阵: {all_delta_rows.shape} (用于 PCA 拟合)")

    pca_model, reduced_encodings, variance_explained = fit_pca_and_reduce(
        delta_vectors, all_delta_rows
    )
    print(f"  PCA 累积方差解释率:")
    for d in PCA_DIMS:
        print(f"    {d} dim: {variance_explained[d]*100:.1f}%")

    # 4. LOO-CV (每个 PCA dim)
    print(f"\n[4/4] LOO-CV (LowRankCDST, prob-space MSE, 5 seeds, 800 epochs)...")
    results = {}
    for d in PCA_DIMS:
        print(f"\n  --- PCA dim={d} ---")
        t0 = time.time()
        preds = run_loo_cv(
            reduced_encodings[d], mutations, target_key, wt_non_target, d=d
        )
        metrics = compute_metrics(preds, mutations, target_key, wt_non_target)
        results[f'pca_{d}dim'] = {
            'd': d,
            'cumulative_variance': variance_explained[d],
            **metrics,
        }
        print(f"  MAE={metrics['mae']:.4f}  median={metrics['median']:.4f}  "
              f"dir={metrics['direction']} ({metrics['direction_pct']*100:.0f}%)  "
              f"({time.time()-t0:.1f}s)")

    # 逐突变预测详情表
    print(f"\n  --- {system_name} 逐突变预测详情 ---")
    print(f"  {'Mutant':<22} {'true':>6}", end='')
    for d in PCA_DIMS:
        print(f" | pca{d} pred  err  dir", end='')
    print()
    for m in mutations:
        true_val = mutations[m][target_key]
        print(f"  {m:<22} {true_val:>6.2f}", end='')
        for d in PCA_DIMS:
            r = results[f'pca_{d}dim']
            p = r['per_mutant'][m]
            e = r['errors'][m]
            ds = r['direction_detail'][m]
            print(f" | {p:>8.3f} {e:>5.3f} {ds:>4}", end='')
        print()

    return {
        'system': system_name,
        'n_mutants': len(mutations),
        'wt_seq_length': len(wt_seq),
        'position_map': {
            name: [{'idx': e[0], 'offset': e[1], 'wt': e[2], 'mut': e[3],
                     'nominal_pos': e[4]} for e in entries]
            for name, entries in pos_map.items()
        },
        'pca_variance_explained': variance_explained,
        'delta_vector_norms': {n: float(np.linalg.norm(v))
                               for n, v in delta_vectors.items()},
        'results': results,
    }


def main():
    print("=" * 80)
    print("ESM-2 per-residue delta + PCA 编码实验 (Task 1)")
    print("模型: ESM-2 650M (facebook/esm2_t33_650M_UR50D), 1280-dim embedding")
    print("协议: LowRankCDST(K=2, rank=2, hidden_dim=32), prob-space MSE, "
          "5 seeds, 800 epochs")
    print("=" * 80)

    torch.manual_seed(0)
    np.random.seed(0)

    # 加载 ESM-2
    try:
        model, tokenizer, device = load_esm2()
    except Exception as e:
        print(f"\n!!! ESM-2 模型加载失败 !!!")
        print(f"错误: {e}")
        print("实验终止. 未生成结果文件.")
        return

    all_results = {}

    # === Abl1 ===
    abl1_result = run_system(
        'Abl1', ABL1_DATA, 'non_ground', ABL1_WT_NON_GROUND,
        ABL1_KD, 'abl1', model, tokenizer, device
    )
    all_results['abl1'] = abl1_result

    # === Src ===
    src_result = run_system(
        'Src', SRC_DATA, 'non_active', SRC_WT_NON_ACTIVE,
        SRC_FULL, 'src', model, tokenizer, device
    )
    all_results['src'] = src_result

    # === 汇总对比表 ===
    print("\n\n" + "=" * 90)
    print("汇总对比表")
    print("=" * 90)
    print(f"{'System':<8} {'编码方案':<24} {'d':>3} {'MAE':>8} {'Direction':>10} "
          f"{'vs 变体C/位置标记':>16} {'vs Extended':>12}")
    print("-" * 90)

    for sys_name, sys_data in all_results.items():
        baselines = BASELINES[sys_name]
        best_existing = min(baselines.values())
        ext_mae = baselines.get('Extended_10dim', 0)
        for d in PCA_DIMS:
            r = sys_data['results'][f'pca_{d}dim']
            vs_best = (r['mae'] - best_existing) / best_existing * 100
            vs_ext = (r['mae'] - ext_mae) / ext_mae * 100
            print(f"  {sys_name:<6} ESM2_pca{d:<16} {d:>3} {r['mae']:>8.4f} "
                  f"{r['direction']:>10} {vs_best:>+15.1f}% {vs_ext:>+11.1f}%")
        # 现有对照
        for label, mae in baselines.items():
            print(f"  {sys_name:<6} {label:<24} {'—':>3} {mae:>8.4f} {'—':>10} "
                  f"{'(baseline)':>16} {'—':>12}")
        print()

    # === 保存 JSON ===
    out_dir = Path(__file__).parent / 'results'
    out_dir.mkdir(exist_ok=True)

    output = {
        'experiment': 'esm2_encoding',
        'description': 'ESM-2 per-residue delta + PCA 编码实验',
        'model': ESM2_MODEL_ID,
        'model_source': ESM2_SOURCE,
        'embed_dim': ESM2_EMBED_DIM,
        'protocol': {
            'model': 'LowRankCDST(K=2, rank=2, hidden_dim=32)',
            'loss': 'prob-space MSE (F.mse_loss(model(w,c), target))',
            'optimizer': 'Adam(lr=5e-3, weight_decay=1e-4)',
            'n_epochs': 800,
            'n_seeds': 5,
            'seed_formula': 'seed*100 + hold_out',
            'best_state_tracking': True,
        },
        'encoding_method': 'per-residue delta (Method A): Δe = E_mut(p) − E_wt(p)',
        'double_mutant_handling': 'sum of per-position deltas (consistent with _aa_delta)',
        'pca_strategy': 'fit on all per-residue delta rows (n_mutants × seq_len samples), '
                        'then transform mutation-position delta vectors',
        'pca_dims': PCA_DIMS,
        'baselines': BASELINES,
        'sequences': {
            'abl1': {'source': 'UniProt P00519-1, kinase domain 229-515',
                     'length': len(ABL1_KD)},
            'src': {'source': 'UniProt P12931 SRC_HUMAN SV=3, full',
                    'length': len(SRC_FULL)},
        },
        'results': all_results,
    }

    out_json = out_dir / 'esm2_encoding.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n结果 JSON 已保存: {out_json}")

    # === 生成 Markdown 报告 ===
    md = build_report(all_results, output)
    out_md = out_dir / 'esm2_encoding_report.md'
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"详细报告已保存: {out_md}")


# ============================================================
# Markdown 报告生成
# ============================================================

def build_report(all_results, output):
    """生成中文 Markdown 报告."""
    lines = []
    L = lines.append

    L("# ESM-2 per-residue delta + PCA 编码实验报告")
    L("")
    L("> **任务**: Task 1 — 验证 ESM-2 650M per-residue 差值编码 + PCA 降维是否能")
    L("> 优于现有 Extended 10-dim / ΔΔG 主特征 / 纯位置标记编码.")
    L("> **模型**: ESM-2 650M (`facebook/esm2_t33_650M_UR50D`), 1280-dim per-residue embedding")
    L("")

    # 方法
    L("## 1. 方法")
    L("")
    L("### 1.1 编码方式 (方法 A: per-residue 差值)")
    L("")
    L("```")
    L("Δe = E_mut(p) − E_wt(p)    # p = 突变位置, E = ESM-2 per-residue embedding (1280-dim)")
    L("```")
    L("")
    L("双突变处理: 取两个位置差值之**和** (与 `encoding_ablation_control.py` 中 `_aa_delta` 的")
    L("求和约定一致).")
    L("")

    L("### 1.2 PCA 降维策略")
    L("")
    L("在所有突变体的**所有残基位置**的 delta 向量上拟合 PCA (样本数 = n_mutants × seq_len),")
    L("然后变换突变位置的 delta 向量至 10/15/20 dim.")
    L("这样既有足够样本支撑 10-20 dim PCA, 又捕获 delta 空间的主方差方向.")
    L("")

    L("### 1.3 协议 (与现有实验完全一致)")
    L("")
    L("| 项 | 设定 |")
    L("|---|---|")
    L("| 模型 | `LowRankCDST(K=2, intervention_dim=d, rank=2, hidden_dim=32)` |")
    L("| 损失 | prob-space MSE (`F.mse_loss(model(w,c), target)`) |")
    L("| 优化器 | Adam(lr=5e-3, weight_decay=1e-4), 800 epochs, best-state 追踪 |")
    L("| 评估 | LOO-CV, 5 seeds, seed=seed×100+hold_out |")
    L("")

    L("### 1.4 序列来源与位置映射")
    L("")
    L("| 体系 | 序列来源 | 长度 | 位置映射 |")
    L("|---|---|---:|---|")
    L(f"| Abl1 | UniProt P00519-1 激酶域 (229-515) | {len(ABL1_KD)} | 项目 pos p → KD index (p−228) |")
    L(f"| Src | UniProt P12931 SV=3 全长 | {len(SRC_FULL)} | 项目 pos p → seq index (p−1) |")
    L("")
    L("> Abl1 项目编号比 Abl1a 统一偏移 −1 (已验证: M290→KD[62]=M, L301→KD[73]=L, F382→KD[154]=F=DFG Phe).")
    L("> Src 大部分位置精确匹配; A311I (实际 A 在 309) 和 F405A (DFG Phe 在 408) 因 UniProt")
    L("> 序列版本差异自动校正 (搜索 ±10 范围内最近的 WT 氨基酸).")
    L("")

    # 各体系结果
    for sys_name, sys_data in all_results.items():
        sys_label = 'Abl1' if sys_name == 'abl1' else 'Src'
        target_key = 'non_ground' if sys_name == 'abl1' else 'non_active'
        wt_non = ABL1_WT_NON_GROUND if sys_name == 'abl1' else SRC_WT_NON_ACTIVE
        mutations = ABL1_DATA if sys_name == 'abl1' else SRC_DATA
        baselines = BASELINES[sys_name]

        L(f"## {2 if sys_name == 'abl1' else 3}. {sys_label} 结果")
        L("")

        # PCA 方差
        L(f"### {sys_data['system']}.1 PCA 累积方差解释率")
        L("")
        L("| PCA dim | 累积方差解释率 |")
        L("|---:|---:|")
        for d in PCA_DIMS:
            var = sys_data['pca_variance_explained'][d]
            L(f"| {d} | {var*100:.1f}% |")
        L("")

        # MAE 对比
        L(f"### {sys_data['system']}.2 MAE 与方向准确率")
        L("")
        L("| 编码方案 | d | MAE | median | 方向 | vs 最优现有 | vs Extended |")
        L("|---|---:|---:|---:|:---:|---:|---:|")
        best_existing = min(baselines.values())
        ext_mae = baselines.get('Extended_10dim', 0)
        for d in PCA_DIMS:
            r = sys_data['results'][f'pca_{d}dim']
            vs_best = (r['mae'] - best_existing) / best_existing * 100
            vs_ext = (r['mae'] - ext_mae) / ext_mae * 100
            L(f"| ESM2 PCA {d}-dim | {d} | {r['mae']:.4f} | {r['median']:.4f} | "
              f"{r['direction']} | {vs_best:+.1f}% | {vs_ext:+.1f}% |")
        for label, mae in baselines.items():
            L(f"| {label} (现有) | — | {mae:.4f} | — | — | (baseline) | — |")
        L("")

        # 逐突变预测
        L(f"### {sys_data['system']}.3 逐突变预测详情")
        L("")
        L(f"WT {target_key} = {wt_non:.2f}")
        L("")
        header = f"| Mutant | true |"
        sep = "|---|---:|"
        for d in PCA_DIMS:
            header += f" pca{d} pred | pca{d} err | pca{d} dir |"
            sep += "---:|---:|:---:|"
        L(header)
        L(sep)
        for m in mutations:
            true_val = mutations[m][target_key]
            row = f"| {m} | {true_val:.2f} |"
            for d in PCA_DIMS:
                r = sys_data['results'][f'pca_{d}dim']
                p = r['per_mutant'][m]
                e = r['errors'][m]
                ds = r['direction_detail'][m]
                row += f" {p:.3f} | {e:.3f} | {ds} |"
            L(row)
        L("")

        # 位置映射
        L(f"### {sys_data['system']}.4 位置映射详情")
        L("")
        L("| Mutant | nominal_pos | actual_idx | offset | WT→Mut |")
        L("|---|---:|---:|---:|---|")
        for name, entries in sys_data['position_map'].items():
            for e in entries:
                L(f"| {name} | {e['nominal_pos']} | {e['idx']} | {e['offset']} | "
                  f"{e['wt']}→{e['mut']} |")
        L("")

    # 结论
    L("## 4. 结论")
    L("")

    # 判定
    abl1_results = all_results['abl1']['results']
    src_results = all_results['src']['results']
    abl1_best = min(abl1_results[f'pca_{d}dim']['mae'] for d in PCA_DIMS)
    src_best = min(src_results[f'pca_{d}dim']['mae'] for d in PCA_DIMS)
    abl1_baseline = BASELINES['abl1']['variant_C_ddg_5dim']
    src_baseline = BASELINES['src']['pos_markers_4dim']

    L(f"### 4.1 MAE 对比汇总")
    L("")
    L("| 体系 | ESM-2 最佳 MAE | PCA dim | 现有最优 MAE | 现有方案 | 差距 |")
    L("|---|---:|---:|---:|---|---:|")
    abl1_best_dim = min(PCA_DIMS, key=lambda d: abl1_results[f'pca_{d}dim']['mae'])
    src_best_dim = min(PCA_DIMS, key=lambda d: src_results[f'pca_{d}dim']['mae'])
    abl1_gap = (abl1_best - abl1_baseline) / abl1_baseline * 100
    src_gap = (src_best - src_baseline) / src_baseline * 100
    L(f"| Abl1 | {abl1_best:.4f} | {abl1_best_dim} | {abl1_baseline:.4f} | 变体C (ΔΔG) | "
      f"{abl1_gap:+.1f}% |")
    L(f"| Src | {src_best:.4f} | {src_best_dim} | {src_baseline:.4f} | 纯位置标记4-dim | "
      f"{src_gap:+.1f}% |")
    L("")

    L("### 4.2 诚实判定")
    L("")

    # ESM-2 是否在 Abl1 上有竞争力
    abl1_competitive = abl1_best < abl1_baseline * 1.1  # within 10%
    abl1_beats = abl1_best < abl1_baseline
    src_competitive = src_best < src_baseline * 1.1
    src_beats = src_best < src_baseline

    if abl1_beats:
        L(f"- **Abl1**: ESM-2 (MAE={abl1_best:.4f}) **超越** 变体C (MAE={abl1_baseline:.4f}), "
          f"差距 {abl1_gap:+.1f}%. ESM-2 embedding 含有比实验 ΔΔG 更强的预测信号.")
    elif abl1_competitive:
        L(f"- **Abl1**: ESM-2 (MAE={abl1_best:.4f}) 与变体C (MAE={abl1_baseline:.4f}) **竞争力相当** "
          f"(差距 {abl1_gap:+.1f}% < 10%). 考虑到 ESM-2 不需要实验 ΔΔG, "
          f"这是有价值的跨体系可迁移编码.")
    else:
        L(f"- **Abl1**: ESM-2 (MAE={abl1_best:.4f}) **劣于** 变体C (MAE={abl1_baseline:.4f}), "
          f"差距 {abl1_gap:+.1f}%. 实验 ΔΔG 仍是 Abl1 上的最强信号. "
          f"但 ESM-2 无需实验数据, 仍有方法价值.")

    if src_beats:
        L(f"- **Src**: ESM-2 (MAE={src_best:.4f}) **超越** 纯位置标记 (MAE={src_baseline:.4f}), "
          f"差距 {src_gap:+.1f}%. ESM-2 提供了位置标记无法捕获的氨基酸特异性信号.")
    elif src_competitive:
        L(f"- **Src**: ESM-2 (MAE={src_best:.4f}) 与纯位置标记 (MAE={src_baseline:.4f}) "
          f"**竞争力相当** (差距 {src_gap:+.1f}% < 10%). 叙事成立: ESM-2 在 Src 上"
          f"不显著劣于现有编码, 与 ESM-Effect ICLR 2025 跨领域互证.")
    else:
        L(f"- **Src**: ESM-2 (MAE={src_best:.4f}) **劣于** 纯位置标记 (MAE={src_baseline:.4f}), "
          f"差距 {src_gap:+.1f}%. Src 上 n=8 小样本下, 位置标记已足够, ESM-2 的"
          f"高维信号反而引入噪声.")
    L("")

    L("### 4.3 发表价值评估")
    L("")
    L("无论 ESM-2 是否超越现有编码, 本实验均有发表价值:")
    L("1. **跨领域互证**: 与 ESM-Effect (ICLR 2025) 互证 pLM embedding 在构象布居"
      "预测上的可用性, 扩展了 pLM 的应用边界.")
    L("2. **方法可迁移性**: ESM-2 不依赖实验 ΔΔG, 可推广到任意激酶/蛋白, "
      "解决现有 ΔΔG 编码在 Src 上不可用的问题.")
    L("3. **小样本启示**: 如 ESM-2 未显著优于简单位置标记, 这本身是"
      "\"少样本下特征选择 > 特征数量\"叙事的进一步佐证.")
    L("")

    L("## 5. 复现")
    L("")
    L("```bash")
    L("cd <repo-root>")
    L("python experiments\\iclr_restructuring\\esm2_encoding.py")
    L("```")
    L("")
    L("输出:")
    L("- `experiments/iclr_restructuring/results/esm2_encoding.json`")
    L("- `experiments/iclr_restructuring/results/esm2_encoding_report.md`")
    L("")
    L("---")
    L("")
    L("*脚本: `experiments/iclr_restructuring/esm2_encoding.py`*")

    return "\n".join(lines)


if __name__ == '__main__':
    main()
