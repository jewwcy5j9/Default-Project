"""
D: Additivity check on the CANONICAL 2-state Abl1 data (logit space, NMR level).

Claim under test (compositional structure, SS4.3 remark):
    logit(w'_{c1+c2}) - logit(w_WT)  ~=  [logit(w'_{c1}) - logit(w_WT)]
                                        + [logit(w'_{c2}) - logit(w_WT)]

Data (canonical non_ground populations, 2-state):
    WT=0.12, M290L=0.45, L301I=0.75, M290L_L301I=0.92

Also reports the model-level composition using canonical Extended-encoding
LOO predictions (predictions from data/canonical_results.json) as secondary.

Output: results/additivity_2state.json + report lines
"""

import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WT = 0.12
P_SINGLE1 = 0.45   # M290L
P_SINGLE2 = 0.75   # L301I
P_DOUBLE = 0.92    # M290L_L301I

# Canonical Extended-encoding LOO predictions (data/canonical_results.json)
PRED_SINGLE1 = 0.20486879348754883   # M290L
PRED_SINGLE2 = 0.9287528991699219    # L301I
PRED_DOUBLE = 0.671317994594574      # M290L_L301I


def logit(p):
    p = min(max(p, 1e-8), 1 - 1e-8)
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def additive_double(wt, p1, p2):
    """Predicted double-mutant population under logit additivity."""
    shift = (logit(p1) - logit(wt)) + (logit(p2) - logit(wt))
    return sigmoid(logit(wt) + shift)


def main():
    # NMR-level (canonical 2-state)
    pred_nmr = additive_double(WT, P_SINGLE1, P_SINGLE2)
    err_nmr = abs(pred_nmr - P_DOUBLE)

    # Model-level (canonical Extended LOO predictions; same additivity rule)
    pred_model = additive_double(WT, PRED_SINGLE1, PRED_SINGLE2)
    err_model_double = abs(pred_model - P_DOUBLE)
    err_model_vs_model = abs(pred_model - PRED_DOUBLE)

    results = {
        'experiment': 'additivity_2state',
        'claim': 'logit-space additivity of shifts for the Abl1 double mutant',
        'nmr_level': {
            'wt': WT, 'single1_M290L': P_SINGLE1, 'single2_L301I': P_SINGLE2,
            'double_true': P_DOUBLE, 'double_additive_pred': pred_nmr,
            'mae_vs_true': err_nmr,
        },
        'model_level_extended_encoding': {
            'single1_pred': PRED_SINGLE1, 'single2_pred': PRED_SINGLE2,
            'double_pred_model': PRED_DOUBLE,
            'double_additive_pred': pred_model,
            'mae_additive_vs_true': err_model_double,
            'mae_additive_vs_model_double': err_model_vs_model,
        },
        'note': '3-state model-level check (constrained training) reported in '
                'Appendix B: pure additivity MAE 0.068, trained compositional 0.053',
    }
    out = Path(__file__).parent / 'results' / 'additivity_2state.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')

    print("=" * 70)
    print("Additivity (logit space), canonical 2-state Abl1")
    print("=" * 70)
    print(f"NMR level:  additive pred {pred_nmr:.4f} vs true {P_DOUBLE}  "
          f"-> MAE {err_nmr:.4f}")
    print(f"Model level (Extended LOO preds): additive {pred_model:.4f} vs "
          f"true {P_DOUBLE} -> MAE {err_model_double:.4f}")
    print(f"            additive vs model's own double pred "
          f"({PRED_DOUBLE:.4f}) -> |diff| {err_model_vs_model:.4f}")
    print(f"[OK] {out}")


if __name__ == '__main__':
    main()
