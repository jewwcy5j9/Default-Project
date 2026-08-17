#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B1: 独立 MSA 验证 — AF2 MSA-subsampling 系综推理脚本

与 run_af2_subsample.py 的唯一协议差异:
  MSA 来源: 每个突变体使用**真实 MSA 搜索**(突变体序列为 query,
  ColabFold mmseqs2_uniref_env 服务器), 而非 WT MSA + in-silico 突变 query.

其余全部冻结 (Option B):
  5 models x 8 seeds x 3 runs x dropout-on = 120 predictions/蛋白
  top-256 MSA -> 随机 64 条/run (msa_seed = run_idx*1000+42)
  alphafold2_ptm, recycles=3, ensemble=1, 无模板, max_seq=256,
  max_extra_seq=512, use_cluster_profile=True, 无 relax

突变体清单 (B1 预注册冻结): WT, L301I, M290L_L301I, F382V (4 x 120 = 480)

输出:
  experiments/af2_subsample/output_independent_msa/msa/<mutant>.a3m    (搜索产物)
  experiments/af2_subsample/output_independent_msa/output/...          (480 PDB)
  experiments/af2_subsample/output_independent_msa/manifest.json

用法:
  python run_af2_independent_msa.py --preflight      # 环境+网络预检
  python run_af2_independent_msa.py --phase search   # 仅 MSA 搜索 (缓存)
  python run_af2_independent_msa.py --phase infer    # 仅推理 (断点续跑)
  python run_af2_independent_msa.py                  # search + infer
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
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# jax/haiku 兼容补丁: jax>=0.4.25 移除了顶层 jax.linear_util,
# 而 dm-haiku 0.0.10 的 _src/dot.py 仍使用 @jax.linear_util.transformation.
# 在导入 haiku/colabfold.batch 之前注入 (必须位于所有相关 import 之前).
# ---------------------------------------------------------------------------
import jax  # noqa: E402
if not hasattr(jax, "linear_util"):
    import jax._src.linear_util as _jax_lu  # noqa: E402
    jax.linear_util = _jax_lu
import jax.random as _jax_random  # noqa: E402
if not hasattr(_jax_random, "default_prng_impl"):
    import jax._src.random as _jax_src_random  # noqa: E402
    _jax_random.default_prng_impl = _jax_src_random.default_prng_impl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from run_af2_subsample import (  # noqa: E402  (复用冻结参数与工具函数)
    PROJECT_ROOT, WT_MSA_PATH, FASTA_DIR,
    MAX_SEQ, MAX_EXTRA_SEQ, SUBSAMPLE_SIZE, NUM_RECYCLES,
    NUM_MODELS, MODEL_TYPE, NUM_ENSEMBLE, STOP_AT_SCORE,
    MSA_READ_LIMIT, ABL1A_OFFSET,
    read_fasta, read_a3m, find_a3m_offset,
    apply_mutations_to_a3m_query, parse_mutations,
    subsample_msa, build_a3m_string,
    check_colabfold, run_prediction_colabfold, clear_gpu_cache,
    load_manifest, save_manifest, get_completed_set,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("af2_independent_msa")

# ============================================================
# B1 专属常量
# ============================================================

B1_MUTANTS = ["WT", "L301I", "M290L_L301I", "F382V"]  # 预注册冻结

# 独立输出根目录 (原 experiments/af2_subsample/output/ 不触碰)
OUT_ROOT = PROJECT_ROOT / "experiments" / "af2_subsample" / "output_independent_msa"
MSA_DIR = OUT_ROOT / "msa"
PDB_OUT_DIR = OUT_ROOT / "output"
MANIFEST_PATH = OUT_ROOT / "manifest.json"
TMP_DIR = OUT_ROOT / "_tmp"

MMSEQS2_SERVER = "https://api.colabfold.com"


# ============================================================
# Phase 1: 每突变体真实 MSA 搜索
# ============================================================

def check_network(url: str, timeout: int = 15) -> bool:
    """轻量网络连通性检查 (MMseqs2 服务器)."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def rebuild_a3m_with_m_numbers(mutant_name: str, out_a3m: Path) -> bool:
    """从 run_mmseqs2 的缓存目录重建带 M 号的 a3m.

    ColabFold 按 header 数字 M 号区分序列类型:
      M=101   -> query
      102..   -> uniref (main MSA)
      2001..  -> env/bfd (extra MSA)
    run_mmseqs2 (colabfold 1.5.5) 返回的 a3m 丢失 M 号 (header 变 UniRef ID),
    导致 ColabFold 推理时 extra MSA 退化为 1 (pLDDT 暴跌). 这里重写编号.
    """
    d = Path.cwd() / f"{mutant_name}.a3m_env"
    uni = d / "uniref.a3m"
    env = d / "bfd.mgnify30.metaeuk30.smag30.a3m"
    if not uni.exists() or not env.exists():
        return False

    out = []
    m = 101
    for line in uni.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True):
        if line.startswith(">"):
            out.append(f">{m}\n")
            m += 1
        else:
            out.append(line)
    m = 2001
    for line in env.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True):
        if line.startswith(">"):
            out.append(f">{m}\n")
            m += 1
        else:
            out.append(line)
    out_a3m.write_text("".join(out), encoding="utf-8")
    return True


def search_msa_for_mutant(mutant_name: str, query_seq: str, out_a3m: Path) -> bool:
    """对突变体序列执行真实 MSA 搜索 (ColabFold mmseqs2_uniref_env 服务器).

    策略 A: colabfold.colabfold.run_mmseqs2 (ColabFold 内部 MSA 搜索入口),
            再从缓存目录重建带 M 号的 a3m (uniref + env).
    策略 B: colabfold.batch.run 以 a3m_lines=None 触发搜索, 从结果目录取 .a3m.
    两策略均失败 → 明确报错 (网络/服务器/环境问题), 不伪造任何 MSA.
    """
    jobname = f"b1_{mutant_name}"

    # 策略 A
    try:
        from colabfold.colabfold import run_mmseqs2
        logger.info(f"[search] {mutant_name}: run_mmseqs2 "
                    f"(mmseqs2_uniref_env, server={MMSEQS2_SERVER})")
        res = run_mmseqs2(
            query_seq, f"{mutant_name}.a3m", use_env=True,
            use_templates=False, host_url=MMSEQS2_SERVER,
            user_agent="cdst-b1/1.0",
        )
        if rebuild_a3m_with_m_numbers(mutant_name, out_a3m):
            logger.info(f"[search] {mutant_name}: OK -> {out_a3m.name} "
                        f"(rebuilt with M numbers, {out_a3m.stat().st_size} bytes)")
            return True
        a3m_lines = res[0] if isinstance(res, tuple) else res
        if a3m_lines and len(a3m_lines) > 0:
            content = a3m_lines[0] if len(a3m_lines) == 1 else "\n".join(a3m_lines)
            out_a3m.write_text(content + "\n", encoding="utf-8")
            logger.info(f"[search] {mutant_name}: OK (fallback) -> {out_a3m.name}")
            return True
        logger.warning(f"[search] {mutant_name}: run_mmseqs2 returned empty; "
                       f"trying batch.run fallback")
    except Exception as e:
        logger.warning(f"[search] {mutant_name}: run_mmseqs2 failed ({e}); "
                       f"trying batch.run fallback")

    # 策略 B
    try:
        from colabfold.batch import run as cf_run
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR)
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        cf_run(
            queries=[(jobname, query_seq, None)],  # a3m_lines=None -> 触发搜索
            result_dir=str(TMP_DIR),
            num_models=1,
            is_complex=False,
            model_order=[1],
            num_recycles=NUM_RECYCLES,
            recycle_early_stop_tolerance=None,
            model_type=MODEL_TYPE,
            num_ensemble=NUM_ENSEMBLE,
            msa_mode="mmseqs2_uniref_env",
            num_seeds=1,
            random_seed=0,
            use_dropout=True,
            stop_at_score=STOP_AT_SCORE,
            num_relax=0,
            keep_existing_results=False,
            rank_by="plddt",
            pair_mode="unpaired_paired",
            max_seq=MAX_SEQ,
            max_extra_seq=MAX_EXTRA_SEQ,
            use_cluster_profile=True,
            data_dir=os.environ.get(
                "COLABFOLD_DATA_DIR",
                os.path.expanduser("~/.local/share/colabfold")),
        )
        a3m_candidates = sorted(TMP_DIR.glob(f"{jobname}*.a3m"))
        if a3m_candidates:
            shutil.copy2(str(a3m_candidates[0]), str(out_a3m))
            logger.info(f"[search] {mutant_name}: OK (strategy B) -> "
                        f"{out_a3m.name}")
            return True
    except Exception as e:
        logger.error(f"[search] {mutant_name}: strategy B failed ({e})")

    logger.error(
        f"[search] {mutant_name}: 真实 MSA 搜索失败。请检查:\n"
        f"  1. 网络可访问 {MMSEQS2_SERVER} (python run_af2_independent_msa.py --preflight)\n"
        f"  2. colabfold 版本 (pip show colabfold)\n"
        f"  3. 或改用本地 colabfold_search / jackhmmer+UniProt (见预注册 §4)"
    )
    return False


def build_mutant_query_seq(wt_a3m_query: str, offset: int, mutant: str) -> str:
    """构建 MSA 搜索用的突变体 query 序列 (WT 返回原样)."""
    if mutant == "WT":
        return wt_a3m_query
    mutations = parse_mutations(mutant)
    return apply_mutations_to_a3m_query(wt_a3m_query, offset, mutations)


def phase_search() -> None:
    """Phase 1: 为 4 个突变体生成真实 MSA 并缓存到 MSA_DIR."""
    MSA_DIR.mkdir(parents=True, exist_ok=True)

    wt_fasta_seq = read_fasta(FASTA_DIR / "abl1_WT.fasta")
    wt_a3m_query, _ = read_a3m(WT_MSA_PATH, max_msa_sequences=10)
    offset = find_a3m_offset(wt_fasta_seq, wt_a3m_query)
    logger.info(f"a3m->FASTA offset: {offset}")

    for mutant in B1_MUTANTS:
        out_a3m = MSA_DIR / f"{mutant}.a3m"
        if out_a3m.exists():
            logger.info(f"[search] {mutant}: 已缓存, 跳过 ({out_a3m.name})")
            continue
        query_seq = build_mutant_query_seq(wt_a3m_query, offset, mutant)
        logger.info(f"[search] {mutant}: query 长度 {len(query_seq)}")
        if not search_msa_for_mutant(mutant, query_seq, out_a3m):
            logger.error(f"[search] {mutant}: 搜索失败, 终止")
            sys.exit(1)


# ============================================================
# Phase 2: 推理 (与 run_af2_subsample.py option_b 完全一致)
# ============================================================

def phase_infer() -> None:
    """Phase 2: 用缓存的真实突变体 MSA 执行 120 预测/蛋白."""
    if not check_colabfold():
        logger.error("未找到 colabfold 包。安装: pip install colabfold[alphafold]")
        sys.exit(1)

    PDB_OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(MANIFEST_PATH)
    completed_set = get_completed_set(manifest)
    logger.info(f"已有 manifest 记录: {len(manifest.get('predictions', []))}, "
                f"已完成: {len(completed_set)}")

    total_all = len(B1_MUTANTS) * 120
    prediction_count = 0
    new_count = 0
    failed_count = 0

    for mutant in B1_MUTANTS:
        msa_file = MSA_DIR / f"{mutant}.a3m"
        if not msa_file.exists():
            logger.error(f"{mutant}: 缺少 MSA 缓存 {msa_file}, 先运行 --phase search")
            sys.exit(1)

        mutant_query, msa_seqs = read_a3m(msa_file, max_msa_sequences=MSA_READ_LIMIT)
        top_msa = msa_seqs[:MAX_SEQ]
        logger.info(f"{mutant}: query len={len(mutant_query)}, "
                    f"MSA pool={len(top_msa)} (top-{MAX_SEQ})")
        if len(top_msa) < SUBSAMPLE_SIZE:
            logger.warning(f"{mutant}: 搜索到的 MSA 仅 {len(top_msa)} 条 "
                           f"(< 子采样 {SUBSAMPLE_SIZE}), 将使用全部")

        for run_idx in range(3):
            msa_seed = run_idx * 1000 + 42
            subsampled = subsample_msa(top_msa, SUBSAMPLE_SIZE, msa_seed)
            a3m_string = build_a3m_string(mutant_query, subsampled)
            run_dir = PDB_OUT_DIR / mutant / f"run_{run_idx}"
            run_dir.mkdir(parents=True, exist_ok=True)

            for model_idx in range(1, NUM_MODELS + 1):
                for seed_idx in range(8):
                    pdb_path = run_dir / f"model_{model_idx}_seed_{seed_idx}.pdb"
                    rel_path = str(pdb_path.relative_to(PDB_OUT_DIR)).replace("\\", "/")
                    prediction_count += 1

                    if pdb_path.exists() or rel_path in completed_set:
                        logger.info(f"  [{prediction_count}/{total_all}] 跳过 "
                                    f"{mutant}/run_{run_idx}/{pdb_path.name}")
                        continue

                    jobname = (f"b1_{mutant}_r{run_idx}_m{model_idx}_s{seed_idx}")
                    status = "failed"
                    start_time = time.time()
                    try:
                        if TMP_DIR.exists():
                            shutil.rmtree(TMP_DIR)
                        TMP_DIR.mkdir(parents=True, exist_ok=True)
                        src = run_prediction_colabfold(
                            a3m_string, model_idx, seed_idx, True,
                            TMP_DIR, jobname,
                        )
                        if src and src.exists():
                            shutil.copy2(str(src), str(pdb_path))
                            status = "completed"
                            new_count += 1
                            logger.info(f"  [{prediction_count}/{total_all}] ✓ "
                                        f"{pdb_path.name} ({time.time()-start_time:.0f}s)")
                        else:
                            logger.error(f"  [{prediction_count}/{total_all}] ✗ 无输出 PDB")
                    except Exception as e:
                        logger.error(f"  [{prediction_count}/{total_all}] ✗ {e}")
                        failed_count += 1

                    manifest["predictions"].append({
                        "mutant": mutant, "run": run_idx, "model": model_idx,
                        "seed": seed_idx, "dropout": "on",
                        "msa_subsample_seed": msa_seed,
                        "msa_source": str(msa_file.relative_to(PROJECT_ROOT)),
                        "file_path": rel_path, "status": status,
                        "timestamp": datetime.now().isoformat(),
                    })
                    manifest["config"] = "b1_option_b_equivalent"
                    manifest["total_expected"] = total_all
                    save_manifest(manifest, MANIFEST_PATH)
                    clear_gpu_cache()

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    manifest["completed_at"] = datetime.now().isoformat()
    save_manifest(manifest, MANIFEST_PATH)
    logger.info(f"推理完成: 新生成 {new_count}, 失败 {failed_count}")
    if failed_count > 0:
        logger.warning("存在失败预测, 重跑同命令可续跑重试")


# ============================================================
# 预检
# ============================================================

def preflight() -> None:
    print("=" * 70)
    print("B1 预检: 环境 + 网络")
    print("=" * 70)
    ok = True

    try:
        import colabfold
        print(f"[OK] colabfold {getattr(colabfold, '__version__', '?')}")
    except ImportError:
        print("[FAIL] colabfold 未安装 -> pip install colabfold[alphafold]")
        ok = False

    try:
        import jax
        platform = "?"
        try:
            from jax.lib import xla_bridge
            platform = xla_bridge.get_backend().platform
        except Exception:
            pass
        print(f"[OK] jax {jax.__version__} (backend: {platform})")
        if platform not in ("gpu", "cuda"):
            print("     注意: 未检测到 GPU 后端 (推理会极慢或失败)")
    except Exception as e:
        print(f"[FAIL] jax: {e}")
        ok = False

    try:
        import torch
        print(f"[OK] torch {torch.__version__} (cuda: {torch.cuda.is_available()})")
    except Exception as e:
        print(f"[WARN] torch: {e}")

    net = check_network(MMSEQS2_SERVER)
    print(f"[{'OK' if net else 'FAIL'}] MMseqs2 服务器可达: {MMSEQS2_SERVER}")
    ok = ok and net

    wt_msa = WT_MSA_PATH.exists()
    print(f"[{'OK' if wt_msa else 'FAIL'}] WT MSA: {WT_MSA_PATH}")
    ok = ok and wt_msa

    print(f"[OK] 输出目录: {OUT_ROOT}")
    print()
    print("预检" + ("通过" if ok else "失败, 请按提示修复后重试"))
    return 0 if ok else 1


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="B1: 独立 MSA 验证 (AF2 系综)")
    parser.add_argument("--phase", choices=["search", "infer"],
                        help="只执行单个阶段; 缺省执行 search+infer")
    parser.add_argument("--preflight", action="store_true", help="环境+网络预检")
    args = parser.parse_args()

    if args.preflight:
        sys.exit(preflight())

    if args.phase in (None, "search"):
        phase_search()
    if args.phase in (None, "infer"):
        phase_infer()
    if args.phase is None:
        logger.info("两阶段完成。下一步: python analyze_independent_msa.py")


if __name__ == "__main__":
    main()
