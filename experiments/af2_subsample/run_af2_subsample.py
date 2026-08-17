#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AF2 MSA-Subsampling 系综基线推理脚本（CDST Task 6.2-6.4）

本脚本实现预注册协议 experiments/af2_subsample/preregistration.md 中定义的
AF2 MSA-subsampling 系综预测，作为 CDST 方法的直接对照基线。

=== 预注册协议要点 ===
- 基线来源: Monteiro da Silva et al., Nat Commun 15, 2464 (2024)（MdS2024）
- 策略: MSA subsampling + multi-seed 组合
  （不使用 MSA mutagenesis 或 template 遍历；不使用 AF2-Multimer；不使用 AF3）
- AF2 参数（预注册 §3 冻结）:
    model_type       = alphafold2_ptm
    max_seq          = 256
    max_extra_seq    = 512
    use_templates    = false
    num_ensemble     = 1
    recycles         = 3（固定，不启用 early stop）
    Evoformer dropout       = 10%（推理时开启）
    Structure module dropout= 25%（推理时开启）
    MSA subsampling 比例    = 1/4（每次推理随机抽取 max_seq 的 1/4 = 64 条）

=== 配置档 ===
  full       : 5 models × 16 seeds × 3 runs × 2 dropout = 480 predictions/蛋白 (3360 total)
  option_b   : 5 models ×  8 seeds × 3 runs × 1 dropout = 120 predictions/蛋白 (840  total)  [默认]
  downgraded : 5 models ×  4 seeds × 1 run  × 1 dropout =  20 predictions/蛋白 (140  total)

  option_b 为本次实验正式配置（见 protocol_deviation_log.md DEV-001）。
  full 与 downgraded 供资源充足或测试时使用。

=== MSA 策略 ===
- WT: 直接使用现有 data/af2_raw/wt/abl_std.a3m（616K 序列，ColabFold mmseqs2_uniref_env 构建）
- 突变体: 对 WT MSA 的 query 序列执行 in-silico 突变
  （仅替换 query 中突变位点残基，保留 MSA 其余序列不变）。

  ⚠️ 已知近似: 真实突变体 MSA 应独立构建（突变可能影响同源序列对齐）。
     本近似在 protocol_deviation_log.md 中记录。
     该近似的影响: MSA subsampling 仍能引入构象多样性（通过改变 MSA 序列子集），
     但突变对 MSA 的影响未被捕获。对于点突变（单残基替换），此近似的影响较小。

- 每个 run: 从 top-256 MSA 中随机子采样 64 条（不同 run 使用不同随机子集）。
- 同一 run 内的所有 (model, seed, dropout) 组合使用相同的 MSA 子集。
- MSA 子采样 seed 由 run_idx 决定（可复现）。

=== 突变位置映射 ===
  FASTA 位置（1-indexed）= Abl1a 残基号 - 227
  （序列从 V228 开始；S229 = FASTA 第 2 位。经验证:
    M290 → FASTA pos 63 = M ✓
    L301 → FASTA pos 74 = L ✓
    F382 → FASTA pos 155 = F ✓）
  a3m query 序列为 FASTA 的子串（去掉了 N/C 端未对齐残基），
  偏移量通过字符串匹配动态计算（不硬编码）。

=== 输出 ===
  PDB : experiments/af2_subsample/output/<mutant>/run_R/model_M_seed_S[_dropout_X].pdb
  清单: experiments/af2_subsample/output/manifest.json
  （dropout 后缀仅在 full 配置 dropout=2 时出现）

=== 运行方式 ===
  python run_af2_subsample.py --config option_b       # 默认，840 predictions
  python run_af2_subsample.py --config full            # 满配，3360 predictions
  python run_af2_subsample.py --config downgraded      # 降级，140 predictions
  python run_af2_subsample.py --config option_b --dry-run  # 只打印计划

=== 依赖 ===
  优先: ColabFold  (pip install colabfold[alphafold])
  回退: AlphaFold  (原生包 + JAX，需额外配置模型参数)
  若均不可用: 打印安装说明并退出。

=== 注意事项 ===
- 本脚本不修改任何现有文件，仅写入 experiments/af2_subsample/output/
- 支持断点续跑: 已存在的 PDB 文件自动跳过
- GPU 内存管理: 每次预测后清理 JAX/Torch 缓存
- 预估 RTX 4090 (24GB): option_b 约 14-18 小时
- 大 MSA 文件 (156MB, 616K 序列) 仅读取 top-257 条（1 query + 256 MSA），避免内存溢出
"""

import argparse
import gc
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================
# 常量定义（预注册协议冻结参数）
# ============================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 输入数据路径
WT_MSA_PATH = PROJECT_ROOT / "data" / "af2_raw" / "wt" / "abl_std.a3m"
FASTA_DIR = PROJECT_ROOT / "data" / "af3_experiment"
REFERENCE_DIR = PROJECT_ROOT / "data" / "bioemu_abl1"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "af2_subsample" / "output"

# 突变体清单（预注册 §5.4 冻结）
MUTANTS = ["WT", "M290L", "L301I", "M290L_L301I", "F382L", "F382Y", "F382V"]

# Abl1a 编号偏移: FASTA 位置（1-indexed）= Abl1a 残基号 - ABL1A_OFFSET
# 序列从 V228 开始（V228 = FASTA 第 1 位，S229 = FASTA 第 2 位）
# 已通过 M290/L301/F382 三个突变位点交叉验证
ABL1A_OFFSET = 227

# AF2 推理参数（预注册 §3 冻结）
MAX_SEQ = 256                              # MSA 最大序列数
MAX_EXTRA_SEQ = 512                        # 额外 MSA 序列数
SUBSAMPLE_FRACTION = 0.25                  # MSA subsampling 比例 (1/4)
SUBSAMPLE_SIZE = int(MAX_SEQ * SUBSAMPLE_FRACTION)  # = 64 条序列
NUM_RECYCLES = 3                           # 固定 3 次 recycle，不启用 early stop
NUM_MODELS = 5                             # AF2 model 1~5
MODEL_TYPE = "alphafold2_ptm"
NUM_ENSEMBLE = 1
STOP_AT_SCORE = 100                        # 不提前停止（设为满分）

# 读取 MSA 时只取前 N 条（避免 616K 序列全部加载到内存）
MSA_READ_LIMIT = MAX_SEQ + 10              # 256 + 10 余量

# 配置档（预注册 §3 + protocol_deviation_log.md DEV-001）
CONFIGS = {
    "full": {
        "models": 5,
        "seeds": 16,
        "runs": 3,
        "dropouts": [True, False],          # dropout on/off 各一
        "predictions_per_protein": 480,
    },
    "option_b": {
        "models": 5,
        "seeds": 8,
        "runs": 3,
        "dropouts": [True],                # 仅 dropout on
        "predictions_per_protein": 120,
    },
    "downgraded": {
        "models": 5,
        "seeds": 4,
        "runs": 1,
        "dropouts": [True],
        "predictions_per_protein": 20,
    },
}

# ============================================================
# 日志配置
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("af2_subsample")


# ============================================================
# MSA / FASTA 处理函数
# ============================================================

def read_fasta(path: Path) -> str:
    """读取 FASTA 文件，返回纯序列字符串（去掉 header 和换行）。"""
    seq_lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">") or not line:
                continue
            seq_lines.append(line)
    return "".join(seq_lines)


def read_a3m(
    path: Path, max_msa_sequences: Optional[int] = None
) -> Tuple[str, List[str]]:
    """
    读取 a3m 格式 MSA 文件。

    参数:
        path: a3m 文件路径
        max_msa_sequences: 最多读取的 MSA 序列数（不含 query）。
            None 表示读取全部。用于避免 616K 序列全部加载到内存。

    返回:
        query_seq: query 序列（第一条，大写，无 gap）
        msa_seqs: 其余 MSA 序列列表（含 gap，按出现顺序，即按 MSA 质量排序）

    a3m 格式说明:
        - 以 '#' 开头的行是元信息（如 "#263\\t1"），跳过
        - 以 '>' 开头的行是序列 header
        - header 下一行是序列（单行，不换行）
        - 大写 = 对齐列；小写 = 插入残基；'-' = gap
    """
    query_seq = ""
    msa_seqs: List[str] = []
    current_seq: List[str] = []
    is_query = True
    seen_header = False
    msa_count = 0
    reached_limit = False

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line.startswith(">"):
                # 保存上一条序列
                if seen_header:
                    seq = "".join(current_seq)
                    if is_query:
                        query_seq = seq
                        is_query = False
                    else:
                        msa_seqs.append(seq)
                        msa_count += 1
                        if max_msa_sequences and msa_count >= max_msa_sequences:
                            reached_limit = True
                            break
                current_seq = []
                seen_header = True
            else:
                current_seq.append(line.strip())

        # 处理最后一条序列（未因 limit 中断的情况）
        if not reached_limit and seen_header and current_seq:
            seq = "".join(current_seq)
            if is_query:
                query_seq = seq
            else:
                msa_seqs.append(seq)

    return query_seq, msa_seqs


def find_a3m_offset(wt_fasta_seq: str, a3m_query_seq: str) -> int:
    """
    计算 a3m query 序列在 WT FASTA 序列中的偏移量。

    返回: offset（0-indexed），即 a3m query 第 1 位对应 FASTA 第 (offset+1) 位。
         fasta_pos(1-indexed) = a3m_pos(1-indexed) + offset

    策略: 去除 a3m query 中的小写字母（插入残基）和 gap 后，
          在 WT FASTA 中搜索子串匹配。
    """
    # 清理 a3m query: 去掉小写字母（插入）和 gap
    a3m_clean = re.sub(r"[a-z\-]", "", a3m_query_seq)

    # 在 WT FASTA 中查找完整匹配
    idx = wt_fasta_seq.find(a3m_clean)
    if idx >= 0:
        return idx

    # 容错: 用前 30 个字符匹配（可能 C 端有截断）
    prefix = a3m_clean[:30]
    idx = wt_fasta_seq.find(prefix)
    if idx >= 0:
        logger.warning(
            f"a3m query 完整匹配失败，使用前缀匹配，偏移量={idx}。"
            f"可能 C 端有截断。"
        )
        return idx

    raise ValueError(
        f"无法在 WT FASTA 中定位 a3m query 序列。\n"
        f"  WT FASTA 前 60: {wt_fasta_seq[:60]}\n"
        f"  a3m query 前 60: {a3m_clean[:60]}"
    )


def parse_mutations(mutant_name: str) -> List[Tuple[str, int, str]]:
    """
    解析突变体名称，返回突变列表。

    示例:
        "WT"            → []
        "M290L"         → [("M", 290, "L")]
        "M290L_L301I"   → [("M", 290, "L"), ("L", 301, "I")]
        "F382L"         → [("F", 382, "L")]
    """
    if mutant_name == "WT":
        return []

    mutations = []
    for part in mutant_name.split("_"):
        match = re.match(r"^([A-Z])(\d+)([A-Z])$", part)
        if not match:
            raise ValueError(f"无法解析突变: {part}（来自突变体名 {mutant_name}）")
        wt_res, pos_str, mut_res = match.groups()
        mutations.append((wt_res, int(pos_str), mut_res))
    return mutations


def apply_mutations_to_fasta(
    wt_fasta_seq: str, mutations: List[Tuple[str, int, str]]
) -> str:
    """
    对 WT FASTA 序列应用突变，返回突变后序列。

    验证每个突变位点的 WT 残基是否匹配预期，不匹配则抛异常。
    用于位置映射的正确性校验。
    """
    seq = list(wt_fasta_seq)
    for wt_res, abl1a_pos, mut_res in mutations:
        fasta_pos = abl1a_pos - ABL1A_OFFSET  # 1-indexed
        if fasta_pos < 1 or fasta_pos > len(seq):
            raise ValueError(
                f"突变 {wt_res}{abl1a_pos}{mut_res} 的 FASTA 位置 {fasta_pos} 越界"
                f"（序列长度 {len(seq)}）"
            )
        actual = seq[fasta_pos - 1]
        if actual != wt_res:
            raise ValueError(
                f"突变 {wt_res}{abl1a_pos}{mut_res} 校验失败: "
                f"FASTA 位置 {fasta_pos} 实际为 '{actual}'，预期 '{wt_res}'"
            )
        seq[fasta_pos - 1] = mut_res
    return "".join(seq)


def apply_mutations_to_a3m_query(
    a3m_query: str,
    offset: int,
    mutations: List[Tuple[str, int, str]],
) -> str:
    """
    对 a3m query 序列应用突变。

    参数:
        a3m_query: 原始 a3m query 序列
        offset: a3m→FASTA 偏移量（fasta_pos = a3m_pos + offset, 1-indexed）
        mutations: 突变列表 [(wt_res, abl1a_pos, mut_res), ...]

    返回:
        突变后的 a3m query 序列

    说明:
        - 突变仅作用于大写残基（对齐列），不修改小写插入残基
        - 每个突变位点校验 WT 残基匹配
    """
    if not mutations:
        return a3m_query

    seq = list(a3m_query)
    for wt_res, abl1a_pos, mut_res in mutations:
        fasta_pos = abl1a_pos - ABL1A_OFFSET          # 1-indexed FASTA 位置
        a3m_pos = fasta_pos - offset                  # 1-indexed a3m 位置
        if a3m_pos < 1 or a3m_pos > len(seq):
            raise ValueError(
                f"突变 {wt_res}{abl1a_pos}{mut_res} 的 a3m 位置 {a3m_pos} 越界"
                f"（a3m query 长度 {len(seq)}）"
            )
        actual = seq[a3m_pos - 1]
        if actual != wt_res:
            raise ValueError(
                f"突变 {wt_res}{abl1a_pos}{mut_res} 校验失败: "
                f"a3m 位置 {a3m_pos} 实际为 '{actual}'，预期 '{wt_res}'"
            )
        seq[a3m_pos - 1] = mut_res
        logger.info(
            f"  突变 {wt_res}{abl1a_pos}{mut_res}: "
            f"FASTA pos {fasta_pos} → a3m pos {a3m_pos} ✓"
        )
    return "".join(seq)


def subsample_msa(msa_seqs: List[str], n: int, seed: int) -> List[str]:
    """
    从 MSA 序列列表中随机子采样 n 条（不重复）。

    使用固定 seed 保证可复现。若 MSA 序列数 ≤ n，返回全部。
    """
    rng = random.Random(seed)
    if len(msa_seqs) <= n:
        logger.warning(
            f"MSA 序列数 {len(msa_seqs)} ≤ 子采样数 {n}，使用全部序列"
        )
        return msa_seqs[:]
    indices = rng.sample(range(len(msa_seqs)), n)
    return [msa_seqs[i] for i in indices]


def build_a3m_string(
    query_seq: str,
    msa_seqs: List[str],
    query_header: str = "101",
) -> str:
    """
    将 query + MSA 序列列表组装为 a3m 格式字符串。

    输出格式:
        >101
        QUERYSEQ
        >1
        MSASEQ1
        >2
        MSASEQ2
        ...
    """
    lines = [f">{query_header}", query_seq]
    for i, seq in enumerate(msa_seqs):
        lines.append(f">{i + 1}")
        lines.append(seq)
    return "\n".join(lines) + "\n"


# ============================================================
# AF2 推理后端
# ============================================================

def check_colabfold() -> bool:
    """检查 colabfold 是否可导入。"""
    try:
        import colabfold  # noqa: F401
        return True
    except ImportError:
        return False


def check_alphafold() -> bool:
    """检查 alphafold 是否可导入。"""
    try:
        import alphafold  # noqa: F401
        return True
    except ImportError:
        return False


def run_prediction_colabfold(
    a3m_string: str,
    model_idx: int,
    seed: int,
    use_dropout: bool,
    result_dir: Path,
    jobname: str,
    query_sequence: str = "",
) -> Optional[Path]:
    """
    使用 ColabFold batch.run API 运行单次 AF2 预测。

    参数:
        a3m_string: a3m 格式 MSA 字符串（含突变 + 子采样）
        model_idx: 模型编号 (1-5)
        seed: 随机种子
        use_dropout: 是否启用 dropout（True=Evoformer 10%+Structure 25%）
        result_dir: 临时结果目录
        jobname: 任务名（用于输出文件命名）

    返回:
        输出 PDB 文件路径，失败返回 None

    说明:
        - a3m_string 以 '>' 开头时，ColabFold 直接使用该 MSA，不会重新生成
        - 参数对齐预注册 §3 与 data/af2_raw/wt/config.json
        - num_models=1 + model_order=[model_idx] 仅运行指定模型
        - num_seeds=1 + random_seed=seed 使用指定种子
    """
    from colabfold.batch import run as cf_run

    result_dir.mkdir(parents=True, exist_ok=True)

    # ColabFold queries 格式: [(jobname, query_sequence, a3m_lines), ...]
    # a3m_lines 必须是 List[str]（列表包含一个 a3m 字符串），不是单个字符串
    # 当 a3m_lines 非空时，ColabFold 直接使用该 MSA，不重新生成
    if not query_sequence:
        # 从 a3m_string 解析 query 序列（第一个 > 后的序列，大写部分）
        lines = a3m_string.strip().split("\n")
        for i, line in enumerate(lines):
            if line.startswith(">"):
                # 下一个非空行就是 query 序列
                if i + 1 < len(lines):
                    # query 序列：只保留大写字母（对齐列），去掉小写和gap
                    raw_query = lines[i + 1]
                    query_sequence = re.sub(r"[a-z\-]", "", raw_query)
                break

    cf_run(
        queries=[(jobname, query_sequence, [a3m_string])],
        result_dir=str(result_dir),
        num_models=1,
        is_complex=False,                       # 单链预测，非复合物
        model_order=[model_idx],
        num_recycles=NUM_RECYCLES,
        recycle_early_stop_tolerance=None,     # 不启用 early stop
        model_type=MODEL_TYPE,
        num_ensemble=NUM_ENSEMBLE,
        msa_mode="mmseqs2_uniref_env",          # 提供 a3m 时自动跳过 MSA 生成
        num_seeds=1,
        random_seed=seed,
        use_dropout=use_dropout,
        stop_at_score=STOP_AT_SCORE,            # 不提前停止
        num_relax=0,                            # 不做 Amber relax
        keep_existing_results=False,
        rank_by="plddt",
        pair_mode="unpaired_paired",
        max_seq=MAX_SEQ,
        max_extra_seq=MAX_EXTRA_SEQ,
        use_cluster_profile=True,
        data_dir=os.path.expanduser("~/.local/share/colabfold"),
    )

    # 查找输出 PDB 文件（ColabFold 命名: {jobname}_unrelaxed_*.pdb）
    pdb_files = sorted(result_dir.glob(f"{jobname}*.pdb"))
    if not pdb_files:
        logger.error(f"未找到输出 PDB: {result_dir}/{jobname}*.pdb")
        return None

    # 优先选择 unrelaxed PDB（未经过 Amber relax）
    unrelaxed = [f for f in pdb_files if "unrelaxed" in f.name]
    if unrelaxed:
        return unrelaxed[0]
    return pdb_files[0]


def run_prediction_alphafold(
    a3m_string: str,
    model_idx: int,
    seed: int,
    use_dropout: bool,
    result_dir: Path,
    jobname: str,
) -> Optional[Path]:
    """
    使用 alphafold 原生包运行单次 AF2 预测（fallback 路径）。

    需要 alphafold 包 + JAX + 预下载的模型参数。
    此为 ColabFold 不可用时的回退方案。

    注意: AlphaFold 原生 pipeline 需要大量配置（模型参数路径、genetic database 路径等）。
    强烈建议安装 ColabFold（封装了 AlphaFold 全部功能，API 更简洁）:
        pip install colabfold[alphafold]
    """
    logger.error(
        "AlphaFold 原生推理 pipeline 需要大量手动配置:\n"
        "  1. 模型参数: alphafold_params_2022-12-06.tar\n"
        "  2. genetic databases: uniref90, mgnify, bfd, uniclust30, pdb70\n"
        "  3. HHsuite / kalign / hmmer 等工具\n\n"
        "建议安装 ColabFold（封装上述全部依赖）:\n"
        "  pip install colabfold[alphafold]\n"
        "  colabfold_download   # 下载模型参数"
    )
    return None


def print_installation_instructions():
    """打印 ColabFold/AlphaFold 安装说明。"""
    print(
        "\n" + "=" * 70 + "\n"
        "错误: 未找到 ColabFold 或 AlphaFold 包\n"
        "=" * 70 + "\n\n"
        "请安装 ColabFold（推荐，封装了 AlphaFold 全部功能）:\n\n"
        "  pip install colabfold[alphafold]\n\n"
        "安装后下载 AF2 模型参数:\n\n"
        "  colabfold_download\n\n"
        "或手动下载:\n"
        "  https://github.com/deepmind/alphafold/releases/\n"
        "  download/v2.3.2/alphafold_params_2022-12-06.tar\n\n"
        "解压到 ~/.cache/colabfold/params/ 或设置环境变量 ALPHAFOLD_PARAMS_DIR\n"
        "\n" + "=" * 70 + "\n",
        file=sys.stderr,
    )


# ============================================================
# GPU 内存管理
# ============================================================

def clear_gpu_cache():
    """清理 GPU 缓存，防止 OOM。在每次预测后调用。"""
    # JAX 缓存清理
    try:
        import jax
        jax.clear_caches()
    except ImportError:
        pass
    # Torch CUDA 缓存清理（ColabFold 部分后端可能使用）
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    # Python 垃圾回收
    gc.collect()


# ============================================================
# Manifest 管理
# ============================================================

def load_manifest(manifest_path: Path) -> Dict:
    """加载已有 manifest（断点续跑）。"""
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    return {"predictions": []}


def save_manifest(manifest: Dict, manifest_path: Path):
    """保存 manifest 到文件。"""
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def get_completed_set(manifest: Dict) -> set:
    """从 manifest 中提取已完成的预测文件路径集合。"""
    return {
        entry["file_path"]
        for entry in manifest.get("predictions", [])
        if entry.get("status") == "completed"
    }


# ============================================================
# 主推理流程
# ============================================================

def run_experiment(config_name: str, dry_run: bool = False):
    """
    运行完整的 AF2 MSA-subsampling 系综实验。

    参数:
        config_name: 配置档名称 (full / option_b / downgraded)
        dry_run: 若为 True，只打印计划不实际运行预测
    """
    config = CONFIGS[config_name]
    total_per_protein = config["predictions_per_protein"]
    total_all = total_per_protein * len(MUTANTS)

    logger.info("=" * 70)
    logger.info("AF2 MSA-Subsampling 系综基线推理（CDST Task 6.2-6.4）")
    logger.info("=" * 70)
    logger.info(f"配置档: {config_name}")
    logger.info(
        f"  models={config['models']}, seeds={config['seeds']}, "
        f"runs={config['runs']}, dropouts={config['dropouts']}"
    )
    logger.info(f"  predictions/蛋白: {total_per_protein}")
    logger.info(f"  突变体数: {len(MUTANTS)} → {MUTANTS}")
    logger.info(f"  总 predictions: {total_all}")
    logger.info(f"  MSA subsampling: top-{MAX_SEQ} → 随机 {SUBSAMPLE_SIZE} 条/run")
    logger.info(f"  输出目录: {OUTPUT_DIR}")

    # ---- 检查推理后端 ----
    use_colabfold = check_colabfold()
    use_alphafold = (not use_colabfold) and check_alphafold()

    if not (use_colabfold or use_alphafold):
        print_installation_instructions()
        sys.exit(1)

    backend = "colabfold" if use_colabfold else "alphafold"
    logger.info(f"推理后端: {backend}")
    if use_alphafold and not use_colabfold:
        logger.warning(
            "使用 AlphaFold 原生后端（fallback）。建议安装 ColabFold 以获得更稳定的 API。"
        )

    if dry_run:
        logger.info("DRY RUN 模式: 只打印计划，不实际运行预测")

    # ---- 读取 WT MSA（仅 top-N 条，避免内存溢出）----
    logger.info(f"\n读取 WT MSA: {WT_MSA_PATH}")
    logger.info(f"  仅读取前 {MSA_READ_LIMIT} 条 MSA 序列（避免 616K 全加载）")
    wt_a3m_query, wt_msa_seqs = read_a3m(WT_MSA_PATH, max_msa_sequences=MSA_READ_LIMIT)
    logger.info(f"  WT a3m query 长度: {len(wt_a3m_query)}")
    logger.info(f"  读取 MSA 序列数: {len(wt_msa_seqs)}")
    if len(wt_msa_seqs) < MAX_SEQ:
        logger.warning(
            f"  MSA 序列数 {len(wt_msa_seqs)} < max_seq {MAX_SEQ}，"
            f"将使用全部 {len(wt_msa_seqs)} 条作为子采样池"
        )

    # ---- 读取 WT FASTA（用于位置映射校验）----
    wt_fasta_path = FASTA_DIR / "abl1_WT.fasta"
    wt_fasta_seq = read_fasta(wt_fasta_path)
    logger.info(f"  WT FASTA 长度: {len(wt_fasta_seq)}")

    # ---- 计算 a3m→FASTA 偏移量 ----
    offset = find_a3m_offset(wt_fasta_seq, wt_a3m_query)
    logger.info(
        f"  a3m→FASTA 偏移: {offset} "
        f"(a3m pos 1 = FASTA pos {offset + 1})"
    )

    # ---- 校验所有突变位置（在 FASTA 上预检）----
    logger.info("\n校验突变位置映射:")
    for mutant in MUTANTS:
        mutations = parse_mutations(mutant)
        if mutations:
            try:
                apply_mutations_to_fasta(wt_fasta_seq, mutations)
                mut_strs = ", ".join(
                    f"{w}{p}{m}" for w, p, m in mutations
                )
                logger.info(f"  {mutant}: {mut_strs} ✓")
            except ValueError as e:
                logger.error(f"  {mutant}: 校验失败 → {e}")
                sys.exit(1)

    # ---- 准备输出目录 ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest = load_manifest(manifest_path)
    completed_set = get_completed_set(manifest)
    logger.info(
        f"\n已有 manifest: {len(manifest.get('predictions', []))} 条记录, "
        f"其中 {len(completed_set)} 条已完成"
    )

    # 临时目录（ColabFold 输出中转）
    tmp_dir = OUTPUT_DIR / "_tmp"

    # ---- 主循环 ----
    prediction_count = 0
    skipped_count = 0
    failed_count = 0
    new_count = 0

    for mutant in MUTANTS:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"突变体: {mutant}")
        logger.info(f"{'=' * 50}")

        # 解析并应用突变到 a3m query
        mutations = parse_mutations(mutant)
        if mutations:
            logger.info(f"  突变: {mutations}")
        mutant_a3m_query = apply_mutations_to_a3m_query(
            wt_a3m_query, offset, mutations
        )

        # 取 MSA 的前 MAX_SEQ 条作为子采样池
        top_msa = wt_msa_seqs[:MAX_SEQ]
        logger.info(f"  MSA 子采样池: {len(top_msa)} 条 (top-{MAX_SEQ})")

        for run_idx in range(config["runs"]):
            # MSA 子采样 seed（每个 run 不同，可复现）
            msa_seed = run_idx * 1000 + 42
            subsampled = subsample_msa(top_msa, SUBSAMPLE_SIZE, msa_seed)
            logger.info(
                f"  Run {run_idx}: 子采样 {len(subsampled)}/{len(top_msa)} "
                f"(msa_seed={msa_seed})"
            )

            # 构建 a3m 字符串
            a3m_string = build_a3m_string(mutant_a3m_query, subsampled)

            # 创建 run 目录
            run_dir = OUTPUT_DIR / mutant / f"run_{run_idx}"
            run_dir.mkdir(parents=True, exist_ok=True)

            for model_idx in range(1, config["models"] + 1):
                for seed_idx in range(config["seeds"]):
                    for use_dropout in config["dropouts"]:
                        dropout_label = "on" if use_dropout else "off"
                        has_multiple_dropouts = len(config["dropouts"]) > 1

                        # 输出文件名
                        if has_multiple_dropouts:
                            pdb_filename = (
                                f"model_{model_idx}_seed_{seed_idx}"
                                f"_dropout_{dropout_label}.pdb"
                            )
                        else:
                            pdb_filename = f"model_{model_idx}_seed_{seed_idx}.pdb"

                        pdb_path = run_dir / pdb_filename
                        rel_path = str(pdb_path.relative_to(OUTPUT_DIR)).replace(
                            "\\", "/"
                        )
                        prediction_count += 1

                        # ---- 断点续跑: 跳过已完成的 ----
                        if pdb_path.exists() or rel_path in completed_set:
                            skipped_count += 1
                            logger.info(
                                f"  [{prediction_count}/{total_all}] "
                                f"跳过 {mutant}/run_{run_idx}/{pdb_filename} (已存在)"
                            )
                            continue

                        # ---- DRY RUN ----
                        if dry_run:
                            logger.info(
                                f"  [{prediction_count}/{total_all}] "
                                f"DRY RUN {mutant}/run_{run_idx}/{pdb_filename}"
                            )
                            continue

                        # ---- 实际运行预测 ----
                        logger.info(
                            f"  [{prediction_count}/{total_all}] "
                            f"预测 {mutant}/run_{run_idx}/{pdb_filename}"
                        )
                        start_time = time.time()

                        jobname = (
                            f"{mutant}_r{run_idx}_m{model_idx}"
                            f"_s{seed_idx}_d{dropout_label}"
                        )
                        status = "failed"

                        try:
                            # 清理临时目录
                            if tmp_dir.exists():
                                shutil.rmtree(tmp_dir)
                            tmp_dir.mkdir(parents=True, exist_ok=True)

                            if use_colabfold:
                                # query_sequence 传空字符串，让函数内部从 a3m 解析（大写部分）
                                src_pdb = run_prediction_colabfold(
                                    a3m_string, model_idx, seed_idx,
                                    use_dropout, tmp_dir, jobname,
                                )
                            else:
                                src_pdb = run_prediction_alphafold(
                                    a3m_string, model_idx, seed_idx,
                                    use_dropout, tmp_dir, jobname,
                                )

                            if src_pdb and src_pdb.exists():
                                shutil.copy2(str(src_pdb), str(pdb_path))
                                elapsed = time.time() - start_time
                                logger.info(
                                    f"    ✓ 完成 ({elapsed:.1f}s) → {pdb_path.name}"
                                )
                                status = "completed"
                                new_count += 1
                            else:
                                logger.error(
                                    f"    ✗ 预测失败（无输出 PDB）"
                                )
                                failed_count += 1
                        except Exception as e:
                            elapsed = time.time() - start_time
                            logger.error(
                                f"    ✗ 预测异常 ({elapsed:.1f}s): {e}",
                                exc_info=True,
                            )
                            failed_count += 1

                        # ---- 记录到 manifest ----
                        manifest["predictions"].append({
                            "mutant": mutant,
                            "run": run_idx,
                            "model": model_idx,
                            "seed": seed_idx,
                            "dropout": dropout_label,
                            "msa_subsample_seed": msa_seed,
                            "file_path": rel_path,
                            "status": status,
                            "timestamp": datetime.now().isoformat(),
                        })

                        # 每次预测后保存 manifest（防止中断丢失进度）
                        manifest["config"] = config_name
                        manifest["total_expected"] = total_all
                        manifest["backend"] = backend
                        save_manifest(manifest, manifest_path)

                        # ---- 清理 GPU 缓存 ----
                        clear_gpu_cache()

    # ---- 清理临时目录 ----
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    # ---- 保存最终 manifest ----
    manifest["config"] = config_name
    manifest["total_expected"] = total_all
    manifest["backend"] = backend
    manifest["completed_at"] = datetime.now().isoformat()
    save_manifest(manifest, manifest_path)

    # ---- 汇总 ----
    logger.info("\n" + "=" * 70)
    logger.info("实验完成")
    logger.info("=" * 70)
    logger.info(f"总 predictions: {prediction_count}/{total_all}")
    logger.info(f"  新生成: {new_count}")
    logger.info(f"  跳过（已存在）: {skipped_count}")
    logger.info(f"  失败: {failed_count}")
    logger.info(f"Manifest: {manifest_path}")
    if failed_count > 0:
        logger.warning(
            f"有 {failed_count} 个预测失败，请检查日志。"
            f"可重新运行脚本以重试失败的预测。"
        )


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="AF2 MSA-Subsampling 系综基线推理（CDST Task 6.2-6.4）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "配置档说明:\n"
            "  full        5×16×3×2 = 480 predictions/蛋白 (3360 total)\n"
            "  option_b    5×8×3×1  = 120 predictions/蛋白 (840 total) [默认]\n"
            "  downgraded  5×4×1×1  =  20 predictions/蛋白 (140 total)\n"
            "\n"
            "示例:\n"
            "  python run_af2_subsample.py --config option_b\n"
            "  python run_af2_subsample.py --config full\n"
            "  python run_af2_subsample.py --dry-run\n"
        ),
    )
    parser.add_argument(
        "--config",
        choices=["full", "option_b", "downgraded"],
        default="option_b",
        help="配置档（默认 option_b: 120 predictions/protein, 840 total）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划，不实际运行预测",
    )
    args = parser.parse_args()

    run_experiment(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
