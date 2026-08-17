# Src CLR zero-replacement and label-interval robustness

## Pseudocount sensitivity

| Pseudocount | Position Ridge MAE | Position GP MAE | u2>u1 rows | F405A>L410A rows |
|---:|---:|---:|---:|---:|
| 1e-08 | 0.2580 | 0.2558 | 10/10 | 10/10 |
| 1e-06 | 0.2583 | 0.2560 | 10/10 | 10/10 |
| 0.0001 | 0.2584 | 0.2573 | 10/10 | 10/10 |
| 0.001 | 0.2577 | 0.2596 | 10/10 | 10/10 |
| 0.01 | 0.2545 | 0.2662 | 9/10 | 10/10 |

The full 10-row MAE ranking is not invariant: the position GP is lower than ridge through 1e-4, but the ordering reverses at 1e-3 and 1e-2. Across the complete grid, u2>u1 holds in 49/50 rows and F405A>L410A error holds in 50/50. The artifact retains all rankings and per-mutation errors.

## Bounded label-interval stress test

Across 200 deterministic-seed realizations, GP MAE was no larger than ridge in 10.0%; both rows retained u2>u1 in 60.5%; both rows retained F405A>L410A error in 42.0%.

Limitation: The 0.10 half-width is the conservative upper end of the curator visual-read range. This is not a substitute for independent redigitization from retained pixel coordinates.
