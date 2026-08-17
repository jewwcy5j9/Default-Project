"""
S2: Src perturbation extension (n=8 -> n=11, 16 -> 19 perturbations).

Adds pY419 (phospho, active-like), pY530 (phospho, E2 reference),
SrcYEEI (motif construct, E2 reference) to the Src K=3 benchmark, with a
type flag appended to the encodings. Runs LOO over all 11 examples and
reports encoding comparisons + paired tests.

Honest expectation: the perturbations have unique type flags, so LOO
predictions for them are expected to be poor (the training set never saw
their type). The value is (a) a larger benchmark, (b) E2 reference
examples as evaluation targets, (c) paired tests with more power.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from k3_data import (
    SRC_K3, SRC_K3_EXT, SRC_K3_WT_POP,
    enc_src_extended_type, enc_src_pos_type, enc_src_no_dvol_type,
)
from k3_benchmark import run_loo, metrics, paired_tests, block_bootstrap


def main():
    t0 = time.time()
    print("=" * 90)
    print("S2: Src perturbation extension (n=11, K=3)")
    print("=" * 90)
    results = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
               'protocol': 'LOO over 11 examples (8 mutants + 3 perturbations), '
                           'type-flagged encodings'}

    encoders = {
        'Extended_type': (enc_src_extended_type, 11),
        'pos_markers_type': (enc_src_pos_type, 5),
        'no_dVol_type': (enc_src_no_dvol_type, 10),
    }
    for enc_name, (fn, d) in encoders.items():
        res = run_loo(SRC_K3_EXT, SRC_K3_WT_POP, fn, d)
        met = metrics(res['per_mutant'], res['targets'], SRC_K3_WT_POP)
        results[enc_name] = {
            'mae': met['mae'], 'direction': met['direction'],
            'errors': met['mae_per_mutant'],
            'preds': {m: res['per_mutant'][m].tolist() for m in res['per_mutant']},
            'per_seed': {m: [p.tolist() for p in res['per_seed'][m]]
                         for m in res['per_seed']},
        }
        print(f"  {enc_name:<20} MAE={met['mae']:.4f} dir={met['direction']}")
        for m in list(SRC_PERT_NAMES):
            print(f"      {m:<16} true={np.round(np.array(SRC_K3_EXT[m]['pop']),2)} "
                  f"pred={np.round(res['per_mutant'][m],2)} "
                  f"err={met['mae_per_mutant'][m]:.4f}")

    # paired test: Extended vs pos-markers on the 11-example set
    t = paired_tests(results['Extended_type']['errors'],
                     results['pos_markers_type']['errors'])
    bb = block_bootstrap(results['Extended_type']['per_seed'],
                         results['pos_markers_type']['per_seed'],
                         {m: SRC_K3_EXT[m]['pop'] for m in SRC_K3_EXT},
                         list(SRC_K3_EXT.keys()), SRC_K3_WT_POP)
    results['paired_ext_vs_pos'] = {'test': t, 'block_ci': bb}
    print(f"  paired Extended vs pos (n=11): Wilcoxon p={t['wilcoxon_p']:.4f}, "
          f"CI {bb[0]:.3f}-{bb[1]:.3f}")

    # per-mutant comparison with the n=8 core (for reference)
    core = json.loads((Path(__file__).parent / 'results' / 'k3_benchmark_results.json')
                      .read_text(encoding='utf-8'))['src']
    results['core_n8_reference'] = {
        'Extended_10dim': core['Extended_10dim']['mae'],
        'pos_markers_4dim': core['pos_markers_4dim']['mae'],
    }

    out_dir = Path(__file__).parent / 'results'
    out_json = out_dir / 'k3_src_perturbation_results.json'
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False,
                                   default=float), encoding='utf-8')
    print(f"\n[OK] {out_json}  (total {time.time()-t0:.0f}s)")


SRC_PERT_NAMES = ['SrcKD-pY419', 'SrcpY530', 'SrcYEEI']

if __name__ == '__main__':
    main()
