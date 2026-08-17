# P4 controlled support/resolution/selection report

Status: **FROZEN COMPLETE** (2026-08-11 fold-local scaling rerun)

- Independent entrypoint: `experiments/iclr_restructuring/p4_support_resolution_selection.py`.
- Complete factorial: 5 sample sizes x 4 collision separations x 3 hidden contrasts x 200 repeats yields **12,000 generated datasets**. Each dataset is evaluated at 3 candidate counts and 2 target resolutions, yielding **72,000 metric records**.
- Models: fixed CLR-ridge (`alpha=1`) and 1-NN capacity control.
- Candidate features remain raw until evaluation. Every outer scaler is fit on
  outer-training IDs only, and every inner selector scaler is fit on its
  inner-training IDs only.
- Every repeat and setting is retained. The first diagnostic run, whose uncontrolled fine-state sinusoid obscured the intended delta intervention, is preserved under `results/archive/p4_pre_control_fix_20260810/`.

## Frozen observations used in the paper

At `n=20`, `m=1`, and `delta=1.2`, the exact collision (`epsilon=0`) has pair CLR-ridge MAE **0.1749**. The equal-prediction reference is **0.1475**; it is not a lower bound on the two-fit pairwise LOO statistic. At `epsilon=1`, pair MAE is **0.1173**.

Averaged over epsilon at `n=20,m=1`, increasing `delta` from 0 to 1.2 changes full-K=3 fine-contrast MAE from **0.0000** to **0.0514**. At `delta=1.2`, shared-contrast MAE is **0.0264** for full K=3 and **0.0267** for separately trained pooled K=2. Pooling removes the controlled fine-state endpoint rather than improving the shared task.

Averaged across sample size, epsilon, and delta, nested-minus-naive selection MAE for full K=3 is **0.0000**, **0.0011**, and **0.0033** for `m=1,5,20`; pooled K=2 gives **0.0000**, **0.0011**, and **0.0042**. The magnitude is simulator-specific; the increase with candidate multiplicity is the controlled result. These values supersede the pre-fold-local-scaling values.

## Acceptance

- 12,000 generated datasets produce 72,000 detailed JSON records and 360 per-setting CSV rows; candidate-count and resolution comparisons reuse each generated dataset.
- 200 repeats in every setting.
- All probability vectors passed simplex checks.
- Every nested outer fold excluded its held-out index from inner selection.
- Every inner-fold audit excludes its held-out ID from scaler fitting.
- Exact-collision and pooling-matrix tests passed.
- PDF and PNG figure hashes are stored in `p4_support_resolution_selection_manifest.json`.

## Metadata-only amendment (2026-08-13)

The synthetic figure `paper/figures_v2/fig4_synthetic_framework.{pdf,png}` was
regenerated on 2026-08-13 (softened palette, softened bar borders, and
label-overflow fixes); the JSON/CSV records and all numerical values are
unchanged. The manifest's `figure_pdf_sha256` and `figure_png_sha256` track the
current regenerated figure; every other manifest field is unchanged.
