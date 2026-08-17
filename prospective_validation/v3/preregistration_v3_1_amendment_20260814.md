# Prospective v3.1 Amendment (FROZEN): Preregistered Diagnostic Scoring

**Version:** 3.1.0
**Status:** **FROZEN 2026-08-15** (before any external data contact; CAND emails were
still NOT SENT at freeze time — see `contact_log_v3.md`).
**Scope:** adds a diagnostic scoring layer on top of the frozen v3.0 primary
protocol (`preregistration_v3.md`, byte-unchanged). Nothing in v3.0 is modified:
the v3.0 primary gates are scored independently and exactly as frozen.

## 1. Implementation pointer (authoritative)

The scoring layer is implemented in `run_custodian_v3.py` (v3.1; the only change
is the appended `v3_1_diagnostic_scoring` block, the added orthonormal-q
contrast fields, the naive-selection block, and the delete-and-refit gate —
see §4). The frozen package is `prospective_v3_1_20260815.zip` with manifest
`archive_manifest_v3_1.json`; the P6 calibration artifact
`p6_audit_detection_benchmark.json` is included in the package. The custodian
runs the package exactly once; the scoring is computed by the same runner from
its own sealed outputs, so no separate scoring code exists to drift.

## 2. The four predictions (each computable from runner output fields)

**P1 — collision-null calibration.** The runner's collision diagnostic
(`diagnostics.collision`) builds the feature-near graph on the frozen
representation: pairwise Euclidean distance between the 1280-d mutation-level
ESM-2 residue-difference vectors, normalized by the panel median vector norm;
collision tolerance $10^{-8}$; target conflict = L1 population distance
$\ge 0.10$. Prediction: the exact label-permutation null of collision-member
error enrichment either is not estimable (no pair is simultaneously a
collision and a conflict, or all members conflict) or does **not** reject at
$\alpha=0.05$ ($p\ge0.05$).
*Calibration basis:* P6 shows this null is nearly powerless at panel scale
(power $\le0.09$; mean $p$ 0.28--0.51 for planted collisions).
*Meaningful failure (predeclared):* $p<0.05$ — the new panel's collision-error
enrichment is then far stronger than any simulator or kinase calibration,
which would overturn the paper's claim that this null cannot localize risk.

**P2 — state resolution.** The runner computes, from its sealed predictions and
targets, the orthonormal contrasts $q_1=(2,-1,-1)/\sqrt{6}$ and
$q_2=(0,1,-1)/\sqrt{2}$ (`diagnostics.contrasts.orthonormal_q`). Prediction:
$q_2$ MAE $> q_1$ MAE **and** the pooled $K{=}2$ model does not improve the
retained axis (pooled $q_1$-scale MAE $\ge$ full $q_1$ MAE, where the
$q_1$-scale error of the pooled model is $(\sqrt{6}/2)$ times its shared
active-coordinate MAE).
*Calibration basis:* held in all ten CLR rows of both kinase panels; the P6
benchmark gives this detector zero false flags at $\delta=0$ and sensitivity
$0.93$--$1.0$ at panel scale.

**P3 — selection optimism.** The runner's four fast candidates
(`clr_ridge`, `clr_gp`, `training_mean`, `constant_wt`) are evaluated both
ways: *nested* = per-fold inner-LOO selection, and *naive* = the single
candidate with the lowest panel-mean fold MAE (both computed on the identical
fold outputs, so the outer labels are used only once and per-fold selection
never sees them). Prediction: nested-minus-naive panel MAE lies in
$[0.005, 0.15]$.
*Calibration basis:* biological fixed-to-selected gaps are $+0.030$ (Abl1) and
$+0.108$ (Src); the P5 quality-ladder band at $n=8$ is $[0.024, 0.027]$. A
value below the band (or negative) falsifies the reuse-optimism mechanism at
this scale; a value far above exceeds the simulator's mechanism range, which
the manuscript already reports it cannot reproduce.

**P4 — support–error association.** The runner computes the exact
permutation test of the Spearman rank correlation between each held-out
mutation's support distance (minimum Euclidean distance to an outer-training
mutation in that fold's outer-PCA space) and its primary LOO MAE
(`diagnostics.support_error`). Prediction: $\rho > 0$ (positive association).
This tests the support layer on the *rank* structure of the new panel,
independently of P1's pairwise-collision mechanism.

## 3. Scoring rule (frozen)

- Each of P1--P4 scores 1 (holds) or 0 (fails), computed once by the runner at
  reveal. If P1's null is not estimable, P1 is recorded `N/A` and the
  denominator becomes 3.
- **Framework validated** iff (denominator 4 and score $\ge3/4$) or
  (denominator 3 and score $3/3$). This is independent of the v3.0 primary
  gate: predictor success with score $\le2/4$ does **not** validate the audit
  framework; predictor failure with a passing score **does** validate it
  (negative validation is retained, per v3.0).
- Partial passes are reported verbatim with each prediction's failure
  direction; no post-hoc reinterpretation, threshold change, or re-scoring is
  permitted. A score of $2/4$ is predeclared to mean the calibration does not
  transfer.
- The v3.0 five primary gates keep their own frozen interpretation. Their
  joint operating characteristics were **not** simulated before freeze: no
  generative model of cross-system biological effect sizes and measurement
  noise exists in this project, and simulating one would itself be a
  post-hoc assumption. The thresholds come from the two-kinase panels and the
  P5/P6 calibrations; this is a stated limitation of the framework-validation
  claim, not a hidden one.

## 4. Runner changes in v3.1 (additive only)

1. `diagnostics.contrasts.orthonormal_q`: q1/q2 MAEs per §2-P2.
2. `diagnostics.selection.naive_*`: naive-selected candidate and its
   panel-mean MAE; `nested_minus_naive` next to the existing
   `nested_minus_fixed` (both reported; only `nested_minus_naive` is scored).
3. `aggregate.success_gates.delete_one_all_positive`: **true delete-and-refit**
   — for each mutation removed in turn, the complete outer LOO (PCA fits,
   refits, seeds) is re-run on the remaining panel and its mean paired
   improvement must be positive. The previous leave-one-out-of-fold
   contribution statistic is retained as a descriptive field
   (`leave_one_fold_contribution_mean`), never as a gate.
4. `v3_1_diagnostic_scoring`: the P1--P4 values, hits, denominator, and
   `framework_validated`; a definitions-hash of this file's §2/§3 text is
   recorded in the run log so the scored definitions are provably frozen.

## 5. Relationship to the manuscript

The manuscript's Discussion names the network qualitatively
(collision-null power, the $q_2>q_1$ ordering, the optimism band, the
support--error association); this frozen file is the operative scoring
instrument, and the runner output is the only authority for the scores.
No manuscript number depends on the amendment's outcome.
