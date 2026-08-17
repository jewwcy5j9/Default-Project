# ADR-002：Direction 指标的操作化定义

- **日期**：2026-08-08（晚）
- **状态**：FROZEN（在任何新结果产生前落盘）
- **SHA-256**：`c05fc8d15522b0b564dd4dee3ab3b34afd5f9ac162a47588bc4d14f1763fc992`（与 `results/archive/phase0_20260808_pre_audit/MANIFEST.json` 一致）
- **来源**：完成度独立审核（2026-08-08）发现 registry `metrics.direction` 文本与实现存在两源冲突后的裁定。

## 冲突

- `benchmark_registry.json` `metrics.direction`（行 36）写的是：
  `sign(pred - WT) == sign(NMR - WT) in sum-abs-shift sense; tie if |NMR-WT| < 0.05; reported as k/total`
- 历史 canonical 实现 `k3_benchmark.py:114-139` 使用 target 总 L1 shift 判定 tie，再用向量点积定符号。
- 当前 P2 实现 `p2_k3_eval.py:312-325` 与报告使用 ACTIVE 态单坐标符号、`|Δactive|<0.05` 为 tie。
- 两套规则在 Src 上给出不同结果：nested/marker 分别为 6/7、7/7（实现）vs 7/8、7/8（canonical）。

## 决策（FROZEN）

**Direction 统一操作化为 ACTIVE 态符号一致率：**

```text
direction = fraction of mutants with
    sign(pred_active - wt_active) == sign(target_active - wt_active),
over mutants where |target_active - wt_active| >= 0.05;
mutants with |target_active - wt_active| < 0.05 are ties and excluded;
reported as k/total.
```

- 阈值 `0.05` 与 TIE_DELTA 常量一致（`p2_k3_eval.py`、`contrast_definition.json`）。
- `sum-abs-shift` 文本仅保留为补充性描述；一切 hard gate、报告与评分卡使用上述 ACTIVE 定义。
- registry 文本修订：将 `metrics.direction` 改写为本定义，记录旧文本、新 hash 与本 amendment 原因。

## 理由

1. 与当前交接红线一致（K=3 主指标方向基于 ACTIVE 态）。
2. ACTIVE 是 u1 的物理载体，跨两系统定义一致，不依赖 E1/E2 细对比的翻转不稳定性。
3. Src E2 极值缺失/饱和时，sum-abs-shift 的 tie 判定对单一坐标敏感，ACTIVE 定义更稳健。
4. 现有 Phase 0 报告与 P2 实现均已使用 ACTIVE 定义，避免历史报告全部重算。

## 引用

- `experiments/iclr_restructuring/benchmark_registry.json`（metrics.direction）
- `experiments/iclr_restructuring/contrast_definition.json`
- `experiments/iclr_restructuring/p2_k3_eval.py:312-325`
- `experiments/iclr_restructuring/k3_benchmark.py:114-139`（被替代的旧实现）
- `SOTA_FOLLOWUP_EXECUTION_PLAN.md`（总体执行规则 5-6）

## 禁止事项

- 不得根据任何新运行结果修改本 ADR。
- 不得在报告中将两种 direction 定义混用而不加标注。
