# Prospective v3 analysis and output plan

1. Validate public and private schemas and exact mutation-ID equality.
2. Verify K=3 simplex constraints, sequence residues, explicit sequence indices, and uniform conditions.
3. Verify hashes of code, public input, private targets, and the target-free ESM-2 feature package.
4. Execute deterministic CPU outer LOO. Each fold records outer-training IDs, PCA fit IDs, dimension, and a hash of PCA components, mean, and explained variance.
5. Store each seed prediction, seed-mean prediction, target, training-mean baseline, constant-WT, CLR-Ridge, CLR-GP, support distance, pooled K=2 result, and nested-selection audit.
6. Compute the five joint primary success gates without rounding intermediate values.
7. Compute all diagnostic endpoints and the single-mutant-only sensitivity.
8. Seal `custodian_result_v3.json`, `fold_seed_predictions_v3.csv`, and `run_log_v3.json`; record their SHA256 values.

The exact support--error permutation enumerates all label permutations when `n<=9`. For larger panels, the result is explicitly labelled a deterministic 100,000-permutation sensitivity and is not used as a primary success gate. The primary paired test is an exact one-sided binomial sign test for all `n`.

No row is excluded after execution. Missing optional secondary inputs are reported as unavailable; they never affect the primary result.
