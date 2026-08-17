# P1 selector-stability audit of the P0-1 nested selection

Status: **DESCRIPTIVE AUDIT** (2026-08-13). No MLP was retrained; all statistics are computed from the frozen `p2_k3_nested_pca_results.json` inner-LOO fold errors and fixed outer predictions.

## abl1 / nested_mlp

- folds = 6; selection counts = {'pca20': 3, 'llr_only': 2, 'ext': 1}
- near-ties (margin < 0.05): 4/6 (0.67); margin range 0.0181–0.0814 (median 0.0410)
- bootstrap: mean P(selected is top-1) = 0.741; folds with P < 0.5 = 0/6
- regret vs oracle: mean 0.1648, max 0.5710

## abl1 / nested_model_select

- folds = 6; selection counts = {'pca20::LowRankCDST': 1, 'llr_pos::LowRankCDST': 2, 'llr_only::CLR-GP': 1, 'llr_only::LowRankCDST': 1, 'ext::LowRankCDST': 1}
- near-ties (margin < 0.05): 5/6 (0.83); margin range 0.0063–0.0785 (median 0.0198)
- bootstrap: mean P(selected is top-1) = 0.551; folds with P < 0.5 = 4/6

## src / nested_mlp

- folds = 8; selection counts = {'pos': 4, 'pca20': 2, 'ext': 2}
- near-ties (margin < 0.05): 5/8 (0.62); margin range 0.0022–0.0629 (median 0.0374)
- bootstrap: mean P(selected is top-1) = 0.685; folds with P < 0.5 = 2/8
- regret vs oracle: mean 0.1944, max 0.4279

## src / nested_model_select

- folds = 8; selection counts = {'pos::CLR-Ridge': 1, 'pca20::LowRankCDST': 1, 'pos::CLR-GP': 1, 'pca20::CLR-GP': 2, 'llr_pos::CLR-Ridge': 1, 'ext::CLR-GP': 1, 'llr_pos::CLR-GP': 1}
- near-ties (margin < 0.05): 8/8 (1.00); margin range 0.0000–0.0251 (median 0.0000)
- bootstrap: mean P(selected is top-1) = 0.361; folds with P < 0.5 = 6/8

