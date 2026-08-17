"""
S3: Experiment-free feature landscape at K=3 (Abl1 core).

Primary claim: a zero-shot pLM thermodynamic proxy (masked-marginal LLR)
is the best experiment-free mutation feature at K=3 Abl1.
  LLR proxy 0.163 (5/5) vs pos-markers 0.276 / Extended 0.300 / ESM-2 emb 0.309
  vs experimental ddG oracle 0.080.

Runs paired tests (per-seed block bootstrap + Wilcoxon) for
  LLR-proxy vs Extended, vs pos-markers, vs ESM-2 embeddings.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from k3_data import ABL1_K3, ABL1_K3_WT_POP, ABL1_SEQ_LEN
from k3_benchmark import run_loo, metrics, paired_tests, block_bootstrap

LLR_FILE = Path(__file__).parent.parent / 'foldx_src' / 'results' / 'esm2_llr_proxy_results.json'
LLR = json.loads(LLR_FILE.read_text(encoding='utf-8'))['abl1']['llr']

ABL1_CORE = {m: ABL1_K3[m] for m in ABL1_K3
             if m not in ('WT', 'H396P', 'M290L_H396P')}
TGT = {m: ABL1_CORE[m]['pop'] for m in ABL1_CORE}


def enc_abl1_llr(name, data):
    enc = np.zeros(5)
    enc[0] = data['pos'] / ABL1_SEQ_LEN
    enc[1] = LLR.get(name, 0.0) / max(abs(v) for v in LLR.values())
    if data['pos'] == 290:
        enc[2] = 1.0
    elif data['pos'] == 301:
        enc[3] = 1.0
    elif data['pos'] == 382:
        enc[4] = 1.0
    return enc


def main():
    t0 = time.time()
    print("=" * 90)
    print("S3: experiment-free feature landscape (Abl1 K=3)")
    print("=" * 90)

    # LLR proxy with per-seed
    res_llr = run_loo(ABL1_CORE, ABL1_K3_WT_POP, enc_abl1_llr, 5)
    met_llr = metrics(res_llr['per_mutant'], res_llr['targets'], ABL1_K3_WT_POP)
    print(f"  LLR proxy:      MAE={met_llr['mae']:.4f} dir={met_llr['direction']}")

    # Reference encodings from the K=3 benchmark (per-seed stored there)
    k3 = json.loads((Path(__file__).parent / 'results' / 'k3_benchmark_results.json')
                    .read_text(encoding='utf-8'))['abl1_core']
    esm2 = json.loads((Path(__file__).parent / 'results' / 'k3_esm2_results.json')
                      .read_text(encoding='utf-8'))['abl1_core']

    refs = {
        'Extended': k3['Extended_10dim'],
        'pos_markers': k3['pos_markers'],
        'ESM2_emb': {'mae_per_mutant': esm2['errors'],
                     'per_seed': None, 'per_mutant_preds': esm2['preds']},
    }
    # ESM-2 per-seed not stored; recompute paired test on mean errors only
    print()
    out = {'llr_proxy': {'mae': met_llr['mae'], 'direction': met_llr['direction'],
                         'errors': met_llr['mae_per_mutant'],
                         'per_seed': {m: [p.tolist() for p in res_llr['per_seed'][m]]
                                      for m in res_llr['per_seed']}},
           'paired': {}}

    for name, ref in refs.items():
        t = paired_tests(ref['mae_per_mutant'], met_llr['mae_per_mutant'])
        label = f'LLR_proxy_vs_{name}'
        if ref.get('per_seed') is not None:
            bb = block_bootstrap(ref['per_seed'], res_llr['per_seed'],
                                 TGT, list(ABL1_CORE.keys()), ABL1_K3_WT_POP)
            out['paired'][label] = {'wilcoxon_p': t['wilcoxon_p'],
                                    'block_ci': bb}
            print(f"  {label:<24} Wilcoxon p={t['wilcoxon_p']:.4f} "
                  f"CI {bb[0]:.3f}-{bb[1]:.3f}")
        else:
            out['paired'][label] = {'wilcoxon_p': t['wilcoxon_p'],
                                    'block_ci': None}
            print(f"  {label:<24} Wilcoxon p={t['wilcoxon_p']:.4f} (mean-errors only)")

    out['landscape'] = {
        'experimental_ddg_oracle': 0.0804,
        'llr_proxy': met_llr['mae'],
        'pos_markers': k3['pos_markers']['mae'],
        'Extended': k3['Extended_10dim']['mae'],
        'ESM2_emb': esm2['mae'],
        'oracle_gain_recovered_pct': (0.3003 - met_llr['mae']) / (0.3003 - 0.0804) * 100,
    }
    print(f"\n  landscape: {out['landscape']}")

    out_dir = Path(__file__).parent / 'results'
    out_json = out_dir / 'k3_llr_proxy_paired.json'
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                                   default=float), encoding='utf-8')
    print(f"[OK] {out_json}  (total {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
