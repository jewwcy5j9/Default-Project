# Src K=3 Label Provenance Audit

Status: machine-linked descriptive audit, updated 2026-08-11.

## Authority

- Canonical records: `data/nmr_populations/src_k3_canonical.csv`
- Validated loader: `src/data/src_k3_labels.py`
- State order: Active / E1 / E2
- Primary protocol: `src_k3_figs5_met305_primary_v1`
- Sensitivity protocol: `src_k3_figs5_met305_with_table_s2_l410a_substitution_v1`

The primary panel uses the Met305-probe WT and all eight core mutation rows
from Cui et al. (2025), Fig. S5. These are probe-specific visual readings. The
canonical CSV records the curator visual range rather than inventing a
measurement standard deviation.

| Mutation | Primary population | Primary record |
|---|---:|---|
| WT | 0.72 / 0.07 / 0.21 | `figs5_met305__SrcKD-WT` |
| L410A | 0.73 / 0.27 / 0.00 | `figs5_met305__SrcKD-L410A` |
| V332I | 0.48 / 0.52 / 0.00 | `figs5_met305__SrcKD-V332I` |
| L270F+V332I | 0.09 / 0.91 / 0.00 | `figs5_met305__SrcKD-L270F_V332I` |
| L325A | 0.00 / 1.00 / 0.00 | `figs5_met305__SrcKD-L325A` |
| A311I | 0.00 / 1.00 / 0.00 | `figs5_met305__SrcKD-A311I` |
| V380A | 0.00 / 0.62 / 0.38 | `figs5_met305__SrcKD-V380A` |
| V331A | 0.00 / 0.45 / 0.55 | `figs5_met305__SrcKD-V331A` |
| F405A | 0.00 / 0.16 / 0.84 | `figs5_met305__SrcKD-F405A` |

## L410A-Only Substitution

Table S2 supplies `table_s2_global__SrcKD-L410A` = 0.96 / 0.03 / 0.01.
The sensitivity panel substitutes this record for primary L410A while retaining
the primary probe WT and the other seven primary mutation records. The global
WT provenance record is deliberately not used. This is a
`hybrid_single_substitution`, not a second complete global-fit panel.

`p2_k3_src_label_sensitivity.json` records both exact record maps, the sole
substitution, the canonical CSV SHA-256, and locked GP metadata. Feature-defined
collision identities do not change because features are label-independent;
absolute errors and MLP contrast ordering remain conditional on label
provenance.

The legacy ambiguous V332I record remains quarantined as
`legacy_ambiguous__SrcKD-V332I` with `ambiguous_not_used` status.
