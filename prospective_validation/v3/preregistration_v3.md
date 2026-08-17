# Prospective v3: independent K=3 NMR population validation

**Version:** 3.0.0  
**Freeze date:** 2026-08-10  
**Status:** frozen before access to any external mutant-population labels  
**Historical record:** v2 and `archive_20260808_adr001.zip` remain unchanged.

## Amendment from v2

V2 mixed a zero-shot physical endpoint with a system-specific LLR few-shot endpoint. V3 replaces both as primary evidence with one system-agnostic predictor: ESM-2 residue embedding differences, outer-fold PCA, and the already fixed LowRankCDST architecture. FoldX and all other models are secondary. The eligibility gate is raised from `n>=5` to an entirely new K=3 panel with `n>=8`.

## Main panel gate

The main prospective claim requires an entirely new system, three quantitative states, at least eight evaluable mutants, a quantitative WT population, and one uniform construct, ligand, temperature, buffer, and state model. State mapping and explicit sequence indices must be frozen before execution. K=3 panels with five to seven mutants and all K=2 panels are supporting evidence only.

Multi-site mutants are allowed. Their intervention vector is the sum of mutant-minus-WT ESM-2 residue-delta vectors at all substituted sites. A single-mutant-only sensitivity is then mandatory.

## Single primary predictor

- Checkpoint: `facebook/esm2_t33_650M_UR50D`.
- No site marker and no position transfer between systems.
- Every outer fold fits PCA only on full per-residue delta rows from outer-training mutants.
- Dimension: `min(20, n_train-1)`.
- LowRankCDST: K=3, rank=2, hidden=32, probability MSE, 800 epochs, Adam (`lr=5e-3`, `weight_decay=1e-4`), five seeds `s*100+outer_test_index`.
- Prediction: normalized mean of the five seed predictions.
- Comparator: arithmetic mean of the outer-training populations on the identical LOO fold.
- Fold error: MAE across the three population components.

## Joint success rule

All conditions must pass: positive mean paired improvement; at least 15% relative MAE improvement; one-sided exact binomial sign-test `p<=0.05`; no primary fold error greater than twice its training-mean baseline error; and positive mean improvement after deleting each mutation in turn.

Failure is retained as independent negative validation. It cannot trigger model, seed, loss, threshold, mutation, or endpoint changes.

## Secondary and diagnostic endpoints

Predeclared secondary controls are FoldX (when frozen state structures exist), LLR-only (when available before reveal), fixed-PCA CLR-Ridge and CLR-GP, constant-WT, and the fully nested fast selector. None can be promoted after reveal.

Diagnostics are support distance versus error; feature/target collisions and conflict-member error; coarse and fine K=3 ILR contrasts; full K=3 versus separately trained pooled K=2 on their shared contrast; fixed versus nested performance; and single-mutant sensitivity. Diagnostics are fully reported. Only the direction of the support--error association is interpreted prospectively; no Src-specific ordering is required.

## Blinding and one reveal

Before reveal, the modeling side may receive sequence, explicit mutation identities, WT populations, conditions, frozen state definitions, and non-directional uncertainty metadata. Mutant populations, ranks, qualitative hints, and population-derived energies remain custodian-private.

The modeling side delivers the code, environment lock, feature package, and hashes. The custodian runs the complete outer LOO in isolation and seals the output. Exactly one reveal is permitted. Every run or reveal event is appended to the reveal log; earlier entries are never edited.
