# T7 fixed ESM-2 PCA provenance amendment

Status: **metadata-only amendment, numerical outputs unchanged** (2026-08-11).

The two canonical GPU artifacts are `t7_fold_local_esm_pca_v2.json` and
`t7_fold_local_esm_pca_v2b.json`. Their original archived SHA256 values are
`1034bb07e90f07c62ff9560aeac43d16e131b2e1d4b749ff4bbbc96b61979f40`
and `2e44f73ac43184d84f51e1fe532119bcba9900eab846996777c9000c58a4b39d`.
The amendment fills two previously null NMR input hashes and records the model
weight hash and Transformers version. Fold predictions, PCA hashes, targets,
and aggregate values are unchanged.

## Inputs and model

- Abl1 NMR JSON SHA256: `8853773bb65b7c01233998f01fa16771c246d1338ba07b9c338cd9b0c64ebeac`.
- Src NMR JSON SHA256: `d4d86555da312451760b9f64509796f8237f127edf3b98607f0c0735ae4944fc`.
- Frozen residue-embedding JSON SHA256: `a07f09cd1af855ba2a2a2026e9f79940956334e67f94c58ed6ecc620fb1e1563`.
- Checkpoint: `facebook/esm2_t33_650M_UR50D`.
- Model-weight SHA256: `a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0`.

## GPU canonical repetitions

Both repetitions used Python 3.12.3, Torch 2.12.1+cu130, NumPy 2.4.4,
scikit-learn 1.9.0, Transformers 5.14.1, CUDA 13.0, and an NVIDIA RTX 4090.
They agree exactly in aggregate: K=3 MAE is 0.1477492904 for Abl1 and
0.3005821130 for Src; pooled K=2 MAE is 0.2135995182 and 0.4090941829.

## CPU cross-device audit

The CPU audit used Python 3.13.14, Torch 2.12.1+cpu, NumPy 2.4.4,
scikit-learn 1.9.0, and Transformers 5.12.1. Its K=3 MAE is 0.1459441629
for Abl1 and 0.3025804989 for Src, absolute differences of 0.00180513 and
0.00199839 from the GPU canonical values. Both are inside the frozen
system-level tolerance of +/-0.003. Device-specific PCA component hashes are
not expected to match; aggregate predictions are compared under the declared
tolerance. SciPy was not imported by the T7 entrypoint and its historical
installed version was not captured, so no version is invented.
