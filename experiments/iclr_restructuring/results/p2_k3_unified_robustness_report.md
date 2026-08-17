# Unified K=3 robustness summary

> The two routes answer different selection questions and are not pooled. Leave-one-observation and double-mutant exclusions summarize frozen outer-fold errors without refitting. Leave-site-out values are separate confirmatory retraining results.

| Route | System | All MAE | Delete-one range | Double-mutant excluded |
|---|---|---:|---:|---:|
| full nested MLP representation selector | Abl1 | 0.2625 | 0.1941--0.3045 | 0.2727 |
| full nested MLP representation selector | Src | 0.3990 | 0.3753--0.4150 | 0.4105 |
| nested candidate-model confirmatory route | Abl1 | 0.4451 | 0.4299--0.4889 | 0.4489 |
| nested candidate-model confirmatory route | Src | 0.3700 | 0.3310--0.4228 | 0.3698 |

## Confirmatory leave-site-out retraining

| System | Held-out family | Route MAE | Training-mean comparator |
|---|---|---:|---:|
| Abl1 | F382_family | 0.2672 | 0.2719 |
| Abl1 | 290_301 | 0.3323 | 0.1430 |
| Src | N_lobe | 0.3537 | 0.3347 |
| Src | C_lobe | 0.4333 | 0.3591 |

The deletion ranges diagnose concentration in individual frozen test errors. They are not cross-validation estimates for a refitted procedure. The leave-site-out rows change the training set and therefore address transfer to an unseen mutation family.
