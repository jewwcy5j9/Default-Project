# ADR-003：P2 gate 的操作化定义（阶段 2 合规重跑用）

- **日期**：2026-08-08（晚）
- **状态**：FROZEN（在任何新结果产生前落盘）
- **SHA-256**：`f77ca9c31296c335d599532185357c974464751a61c620d7774d47aab0620a37`（与 `results/archive/phase0_20260808_pre_audit/MANIFEST.json` 一致）
- **来源**：完成度独立审核（2026-08-08）发现 `p2_k3_eval.py gates()` 实现与主计划 B8 不一致后的裁定。主计划 `SOTA_FOLLOWUP_EXECUTION_PLAN.md` 不修改；本 ADR 只消除实现歧义。

## 背景缺陷（已审计）

- 原 `gates()`（`p2_k3_eval.py:372-389`）：no-marker 用 `<=`；同一系统同时生成 `< floor` 与 `<= floor` 重复字段；LSO 把全部组合/组的收益打平后做绝对 50% 阈值，允许跨候选拼接；`alt_no_verdict_reversal` 只是阈值复读，未比较 main/alt verdict；缺 catastrophic comparator、single-mutant dominance、schema/index/leakage 与 overall verdict。

## 决策（FROZEN）

### G1. No-marker gate

- 严格使用 `candidate_nested_mae < marker_control_nested_mae`（每系统独立）。
- 理由：主计划 B8 原文"无位点 marker 候选**优于**标记对照"，优于即严格小于。

### G2. Alt-label gate（src）

- 两个独立条件，均须为 true：
  ```text
  alt_l410a_le_0_2560:         alt nested MAE <= 0.2560
  alt_verdict_not_reversed:    main verdict 与 alt verdict 均为 pass 或均为 fail（不反转）
  ```
- verdict = 该标签下 nested MAE 对各自 floor 的通过与否；main floor=0.2560，alt floor=0.2560。

### G3. LSO candidate（同路线原则）

- 对**同一个完整 selection-aware route**（同一候选组合+模型集合、同一 inner selector、同一 outer split），在两个预定义 LSO group（abl1: F382_family / 290_301；src: N_lobe / C_lobe）上分别重跑 inner selector。
- 禁止跨配置拼接：不允许从组合 A 取最好组、从组合 B 取最坏组来"构造"失败或通过。

### G4. LSO comparator

- 冻结为**每个 LSO split 的 training-mean comparator**：对 LSO split 的 outer training mutants 计算逐状态 training-mean MAE，作为该 split 的组 baseline。
- 理由：两系统均有冻结值（abl1 0.2329 / src 0.2911 为全量版本；split 内用 split 自身 training mean），不依赖系统专有历史特征。

### G5. LSO gate

- 同一路线下：
  ```text
  至少一个 group 的组 MAE < 其 comparator（改善）
  且另一 group 的相对恶化 <= 5%（(group_mae - comparator) / comparator <= 0.05）
  ```

### G6. Catastrophic

- catastrophic 定义沿用主计划规则 9（行 188）：candidate per-mutant MAE `> 2x` 同 fold training-mean MAE；baseline error 为 0 时 candidate error `> 0.05` 直接标 catastrophic。
- **Catastrophic comparator**（B8 "catastrophic folds 不多于冻结 comparator"）：冻结为**相同 outer folds 的 marker control**的 catastrophic count（逐 fold 成对比较）。
- training mean 只用于 catastrophic 定义，不兼任 comparator。

### G7. Single-mutant dominance

- 沿用阶段 3 D6 规则：单一 mutant 的 paired improvement 贡献 `<= 50%` 总 paired improvement，否则视为单点驱动。

### G8. Gate 输出结构

- 每系统输出独立布尔值 + overall verdict，禁止重复/冗余字段：

```text
abl1_nested_lt_0_2329            src_nested_le_0_2560
alt_l410a_le_0_2560              alt_verdict_not_reversed
no_marker_strictly_beats_marker  lso_same_route_pass
catastrophic_not_worse_than_control
single_mutant_contribution_le_0_50
schema_valid                     index_alignment_valid
fold_local_valid
overall_go                       (全部 above 为 true 才为 true)
```

## 引用

- `SOTA_FOLLOWUP_EXECUTION_PLAN.md` B8（行 361-373）、规则 9（行 188）、D6（行 534-545）
- `experiments/iclr_restructuring/p2_k3_eval.py`（被替代的 gates/run_nested 实现）
- `experiments/iclr_restructuring/benchmark_registry.json`（baselines_frozen floors）
- `experiments/iclr_restructuring/results/ADR-002-direction-definition.md`

## 禁止事项

- 不得根据任何新运行结果修改本 ADR。
- 不得把 exploratory fixed-combo 结果作为 LSO gate 的依据。
- 不得再次同时生成 `< floor` 与 `<= floor` 重复字段。
