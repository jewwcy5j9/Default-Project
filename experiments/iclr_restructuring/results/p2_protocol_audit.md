# P2 协议审计（p2_protocol_audit.md）

- 审计日期：2026-08-07
- 审计对象：`experiments/iclr_restructuring/p2_eval.py` 及其结果 `results/p2_eval_results.json`、`results/p2_stage2_report.md`
- 结论：**INVALID-FOR-K3-CONFIRMATION**；GPU 特征缓存保留复用。

## 缺陷清单

### P0-1：K=3 被压成 K=2（评估目标错误）
- `p2_eval.py:106`：`LowRankCDST(K=2, intervention_dim=d, rank=2, hidden_dim=32)`
- `p2_eval.py:137`：`SimpleCDST(K=2, intervention_dim=d)`
- `p2_eval.py:240-242`：目标折叠为 `non_active = 1.0 - core[m]["pop"][0]`
- `p2_eval.py:163-167`：训练标签构造为 `[p_active, p_non_active]`
- registry 主指标：K=3 simplex 逐状态 MAE（`benchmark_registry.json` §metrics/primary）
- 影响：`Src 0.1767` 只反映 coarse collapse 上的可能信号，不能说明 E1/E2 恢复，不能与 K=3 门限 `0.2560` 直接比较。

### P0-2：内层 CV 特征与标签错配
- `p2_eval.py:292`：测试特征 `Xall[[tr.index(hold2)]]`，标签 `tgt[hold2]`
- 删除 outer holdout 后 `tr` 是压缩后的训练列表，`tr.index(hold2)` 得到的是列表位置，不是原始数据下标；正确写法应为 `Xall[[hold2]]`。
- 影响：内层 selector 分数与候选选择均不可信。

### P0-3：所谓 Src u2 不是 contrast 指标
- `p2_eval.py:64-66`：`SRC_U1`/`SRC_U2` 只是突变体分组名；
- `leave_site_out()` 仍预测 scalar `1-p_active`；
- 计划中的 u2 = `p_E1 - p_E2`，历史基线 MAE = `0.605`；
- 影响：报告 `u2=0.0000-0.0115` 实为 A311I/F405A 的 non-active 饱和分数误差。

### P1-1：LLR 归一化非 fold-local
- `p2_eval.py:237-239`：CV 前用全系统 `max(abs(llr))` 缩放；
- registry `feature_transforms.rule`：fold-local 或外部冻结。

### P1-2：缺少完整 alternative-label sensitivity
- Src L410A 主标签 `[0.73,0.27,0.00]`，替代 global `[0.96,0.03,0.01]`；
- 需在相同候选/fold/seed/fold-local transforms 下完整重跑 nested selection。

### P1-3：结果 schema 不完整
- 缺少：K=3 per-seed predictions、JSD、raw u1/u2、ILR/Helmert、catastrophic folds、support-stratified error、合法 LSO、alternative-label sensitivity、脚本与输入 hash。

### P1-4：LSO 分组由标签行为定义
- 新分组只按结构/序列区域预定义（Abl1 F382_family、290_301；Src N_lobe、C_lobe）。

## 保留资产
- `results/p2_llr_features.json`、`results/p2_site_deltas.npz`、`results/p2_embeddings.npz`、`results/p2_manifest.json`（模型 `esm2_t33_650M_UR50D`，权重 SHA-256 `a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0`，torch 2.12.1+cu130，RTX 4090，reps=20，mask=0.15，seed=0）。

## 处置
- 本文件引用 `SOTA_FOLLOWUP_EXECUTION_PLAN.md`（§2、§6），按 Workstream B 新建 `p2_k3_eval.py` 重跑。
