"""ESM-2 masked-marginal LLR proxy at K=3 (Abl1 variant-C substitute)."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from k3_data import ABL1_K3, ABL1_K3_WT_POP, ABL1_SEQ_LEN
from k3_benchmark import run_loo, metrics

LLR_FILE = Path(__file__).parent.parent / 'foldx_src' / 'results' / 'esm2_llr_proxy_results.json'
LLR = json.loads(LLR_FILE.read_text(encoding='utf-8'))['abl1']['llr']


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
    abl1_core = {m: ABL1_K3[m] for m in ABL1_K3
                 if m not in ('WT', 'H396P', 'M290L_H396P')}
    res = run_loo(abl1_core, ABL1_K3_WT_POP, enc_abl1_llr, 5)
    met = metrics(res['per_mutant'], res['targets'], ABL1_K3_WT_POP)
    print(f"Abl1 K=3, variant C with LLR proxy: MAE={met['mae']:.4f} "
          f"dir={met['direction']}")
    for m in abl1_core:
        print(f"  {m:<14} true={np.round(np.array(ABL1_K3[m]['pop']),2)} "
              f"pred={np.round(res['per_mutant'][m],2)} "
              f"err={met['mae_per_mutant'][m]:.4f}")

    out = {'experiment': 'k3_llr_proxy', 'mae': met['mae'],
           'direction': met['direction'], 'errors': met['mae_per_mutant'],
           'preds': {m: res['per_mutant'][m].tolist() for m in res['per_mutant']}}
    out_path = Path(__file__).parent / 'results' / 'k3_llr_proxy_results.json'
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"[OK] {out_path}")


if __name__ == '__main__':
    main()
