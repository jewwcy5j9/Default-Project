# Dynamic-range-normalized K=3 errors

Normalized MAE is raw MAE divided by the constant-WT MAE for the same system.
The denominator is the mean absolute mutant-minus-WT population shift over all
mutation-state coordinates.  Thus 1.0 equals constant-WT prediction and lower
is better.  These values are descriptive and are not used for model selection.

| System | Constant-WT MAE | State ranges | Train mean | LLR+pos fold-local | PCA | Full nested |
|---|---:|---:|---:|---:|---:|---:|
| Abl1 | 0.3878 | 0.83/0.10/0.89 | 0.6006 | 0.3611 | 0.3810 | 0.6771 |
| Src | 0.4600 | 0.73/0.84/0.84 | 0.6328 | 0.7484 | 0.6534 | 0.8675 |
