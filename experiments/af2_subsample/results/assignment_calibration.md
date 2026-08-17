# AF2 Assignment Calibration

> These results are protocol-dependent structure assignments, not estimates of thermodynamic populations. Structures generated with shared models, seeds, or MSAs are not treated as independent biological samples.

The preregistered classifications remain unchanged. This report adds cutoff and alignment sensitivity diagnostics. `ambiguous` means the nearest reference passes the cutoff but the RMSD margin (second-nearest minus nearest) is < 0.50 Angstrom. Cutoff failures remain `unclassified`.

## Reference-to-reference calibration

| Alignment region | Active-I1 | Active-I2 | I1-I2 |
|---|---:|---:|---:|
| full_protein | 5.040 | 8.894 | 2.592 |
| n_lobe_act | 5.916 | 7.327 | 3.036 |
| alphaC_only | 1.569 | 2.564 | 2.342 |

## Assignment calibration at 3.0 Angstrom

### original

| Alignment region | active | I1 | I2 | ambiguous | unclassified | margin median | margin q05-q95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_protein | 667 | 0 | 0 | 0 | 173 | 2.702 | 0.696-3.177 |
| n_lobe_act | 674 | 0 | 0 | 0 | 166 | 3.285 | 1.723-3.831 |
| alphaC_only | 0 | 36 | 0 | 801 | 3 | 0.208 | 0.024-0.481 |

Frozen full-protein result check: 840/840 records match the stored assignments.

### fresh_msa

| Alignment region | active | I1 | I2 | ambiguous | unclassified | margin median | margin q05-q95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_protein | 13 | 0 | 0 | 0 | 467 | 0.784 | 0.047-2.317 |
| n_lobe_act | 36 | 0 | 0 | 0 | 444 | 1.455 | 0.073-2.980 |
| alphaC_only | 0 | 3 | 0 | 298 | 179 | 0.257 | 0.041-0.666 |

Frozen full-protein result check: 480/480 records match the stored assignments.

## Threshold curves

Counts below use the original strict cutoff plus argmin convention. The separate ambiguity-aware counts are available in the JSON output for every 0.25 Angstrom increment from 2.0 through 5.0 Angstrom.

### original: full_protein

| Cutoff | active | I1 | I2 | unclassified |
|---:|---:|---:|---:|---:|
| 2.00 | 159 | 0 | 0 | 681 |
| 2.25 | 418 | 0 | 0 | 422 |
| 2.50 | 556 | 0 | 0 | 284 |
| 2.75 | 632 | 0 | 0 | 208 |
| 3.00 | 667 | 0 | 0 | 173 |
| 3.25 | 707 | 0 | 0 | 133 |
| 3.50 | 740 | 0 | 0 | 100 |
| 3.75 | 760 | 0 | 0 | 80 |
| 4.00 | 768 | 0 | 1 | 71 |
| 4.25 | 772 | 0 | 1 | 67 |
| 4.50 | 779 | 0 | 1 | 60 |
| 4.75 | 782 | 0 | 1 | 57 |
| 5.00 | 786 | 0 | 1 | 53 |

### original: n_lobe_act

| Cutoff | active | I1 | I2 | unclassified |
|---:|---:|---:|---:|---:|
| 2.00 | 65 | 0 | 0 | 775 |
| 2.25 | 292 | 0 | 0 | 548 |
| 2.50 | 511 | 0 | 0 | 329 |
| 2.75 | 628 | 0 | 0 | 212 |
| 3.00 | 674 | 0 | 0 | 166 |
| 3.25 | 696 | 0 | 0 | 144 |
| 3.50 | 719 | 0 | 0 | 121 |
| 3.75 | 736 | 0 | 1 | 103 |
| 4.00 | 773 | 0 | 1 | 66 |
| 4.25 | 788 | 0 | 1 | 51 |
| 4.50 | 795 | 0 | 1 | 44 |
| 4.75 | 803 | 1 | 1 | 35 |
| 5.00 | 809 | 1 | 1 | 29 |

### original: alphaC_only

| Cutoff | active | I1 | I2 | unclassified |
|---:|---:|---:|---:|---:|
| 2.00 | 352 | 446 | 0 | 42 |
| 2.25 | 385 | 450 | 0 | 5 |
| 2.50 | 386 | 451 | 0 | 3 |
| 2.75 | 386 | 451 | 0 | 3 |
| 3.00 | 386 | 451 | 0 | 3 |
| 3.25 | 386 | 451 | 0 | 3 |
| 3.50 | 386 | 451 | 0 | 3 |
| 3.75 | 386 | 451 | 0 | 3 |
| 4.00 | 386 | 451 | 0 | 3 |
| 4.25 | 386 | 451 | 0 | 3 |
| 4.50 | 386 | 453 | 0 | 1 |
| 4.75 | 386 | 453 | 0 | 1 |
| 5.00 | 386 | 453 | 0 | 1 |

### fresh_msa: full_protein

| Cutoff | active | I1 | I2 | unclassified |
|---:|---:|---:|---:|---:|
| 2.00 | 0 | 0 | 0 | 480 |
| 2.25 | 1 | 0 | 0 | 479 |
| 2.50 | 4 | 0 | 0 | 476 |
| 2.75 | 9 | 0 | 0 | 471 |
| 3.00 | 13 | 0 | 0 | 467 |
| 3.25 | 15 | 0 | 0 | 465 |
| 3.50 | 29 | 0 | 0 | 451 |
| 3.75 | 42 | 0 | 0 | 438 |
| 4.00 | 52 | 0 | 0 | 428 |
| 4.25 | 74 | 0 | 0 | 406 |
| 4.50 | 91 | 0 | 1 | 388 |
| 4.75 | 107 | 0 | 2 | 371 |
| 5.00 | 128 | 0 | 2 | 350 |

### fresh_msa: n_lobe_act

| Cutoff | active | I1 | I2 | unclassified |
|---:|---:|---:|---:|---:|
| 2.00 | 0 | 0 | 0 | 480 |
| 2.25 | 0 | 0 | 0 | 480 |
| 2.50 | 5 | 0 | 0 | 475 |
| 2.75 | 18 | 0 | 0 | 462 |
| 3.00 | 36 | 0 | 0 | 444 |
| 3.25 | 84 | 0 | 0 | 396 |
| 3.50 | 140 | 0 | 0 | 340 |
| 3.75 | 188 | 0 | 0 | 292 |
| 4.00 | 199 | 0 | 0 | 281 |
| 4.25 | 209 | 1 | 0 | 270 |
| 4.50 | 212 | 1 | 2 | 265 |
| 4.75 | 218 | 1 | 3 | 258 |
| 5.00 | 226 | 1 | 4 | 249 |

### fresh_msa: alphaC_only

| Cutoff | active | I1 | I2 | unclassified |
|---:|---:|---:|---:|---:|
| 2.00 | 235 | 22 | 0 | 223 |
| 2.25 | 261 | 26 | 0 | 193 |
| 2.50 | 262 | 31 | 0 | 187 |
| 2.75 | 262 | 38 | 0 | 180 |
| 3.00 | 262 | 38 | 1 | 179 |
| 3.25 | 263 | 40 | 7 | 170 |
| 3.50 | 263 | 42 | 15 | 160 |
| 3.75 | 267 | 45 | 39 | 129 |
| 4.00 | 267 | 45 | 77 | 91 |
| 4.25 | 267 | 45 | 112 | 56 |
| 4.50 | 269 | 45 | 137 | 29 |
| 4.75 | 271 | 47 | 149 | 13 |
| 5.00 | 272 | 47 | 151 | 10 |

## Provenance

RMSD implementation: `experiments/af2_subsample/classify_states.py` (`16ce03d12ad9fc5a65aa0a22e4926a77ad1e920c4707e25883ceaf6da29275b4`).

- active: `data/bioemu_abl1/ref_6XR6_active.pdb` (`7a50ee06c9fb53b310d1a3089753767875eaf7d7f96eed8db10c6939ef727b73`)
- I1: `data/bioemu_abl1/ref_2HYY_i1.pdb` (`a1ec29a9e9e93e5332849061d6db0bb4a987fba15de11a984081e8ee0f34c244`)
- I2: `data/bioemu_abl1/ref_6XRG_i2.pdb` (`7b82daedf4ff1aee4ce072721a42d2cae2c439f0c530f60ddd9cd5932aa33f66`)
- original structures: `experiments/af2_subsample/output` (collection SHA-256 `4462b31ac81c53ae0c1f7ce671a5c1046ac70881eff5be5e3469c4f23fcb8829`)
- fresh_msa structures: `experiments/af2_subsample/output_independent_msa/output` (collection SHA-256 `f98b5b99e86475e9aca186317f15a07c8f5027fb0b1bba5606cc2730573e313d`)

## Limitations

The 0.5 Angstrom ambiguity cutoff is an explicit diagnostic convention, not a validated physical boundary. Reference perturbation and leave-one-reference stability are not evaluated here. Results depend on the three selected references, the first model/chain-A parsing convention, residue correspondence, and the stated Kabsch alignment regions.
