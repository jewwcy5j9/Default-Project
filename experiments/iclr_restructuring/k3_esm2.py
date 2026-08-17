"""
ESM-2 features at K=3 (Src + Abl1 core/ext).

Question: do deep (pLM) mutation embeddings capture the E1/E2 direction that
hand-crafted encodings cannot (Src K=3 negative finding)? Also produces the
ESM-2 rows for the K=3 benchmark tables.

Reuses the embedding machinery from esm2_encoding.py (same protocol:
per-residue delta at mutation positions, PCA fit on per-system deltas, d=20).
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from esm2_encoding import (
    ABL1_KD, SRC_FULL, find_position, parse_mutations_from_name,
    load_esm2, compute_all_embeddings, compute_delta_encodings,
    fit_pca_and_reduce,
)
from k3_data import (
    ABL1_K3, ABL1_K3_WT_POP, SRC_K3, SRC_K3_WT_POP,
)
from k3_benchmark import run_loo, metrics

PCA_DIM = 20


def build_pos_map(mutations, wt_seq, system):
    pos_map = {}
    for name, data in mutations.items():
        mut_list = parse_mutations_from_name(name, data)
        entries = []
        for pos, wt_aa, mut_aa in mut_list:
            idx, offset = find_position(wt_seq, pos, wt_aa, system)
            entries.append((idx, offset, wt_aa, mut_aa, pos))
        pos_map[name] = entries
    return pos_map


def main():
    t0 = time.time()
    print("=" * 90)
    print("ESM-2 features at K=3")
    print("=" * 90)
    model, tokenizer, device = load_esm2()
    results = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
               'pca_dim': PCA_DIM, 'model': 'esm2_t33_650M_UR50D'}

    # ---------- Src ----------
    src_mut = {m: SRC_K3[m] for m in SRC_K3 if m != 'SrcKD-WT'}
    print("\n[Src] embeddings...")
    pos_map = build_pos_map(src_mut, SRC_FULL, 'src')
    embeddings, mut_pos = compute_all_embeddings(model, tokenizer, SRC_FULL,
                                                 src_mut, pos_map, 'src', device)
    deltas, rows = compute_delta_encodings(embeddings, mut_pos)
    _, reduced, var_exp = fit_pca_and_reduce(deltas, rows, dims=[PCA_DIM])
    enc_src = reduced[PCA_DIM]
    print(f"  PCA cumvar (d={PCA_DIM}): {var_exp[PCA_DIM]:.3f}")
    res_src = run_loo(src_mut, SRC_K3_WT_POP,
                      lambda m, d: enc_src[m], PCA_DIM)
    met_src = metrics(res_src['per_mutant'], res_src['targets'], SRC_K3_WT_POP)
    results['src'] = {'mae': met_src['mae'], 'direction': met_src['direction'],
                      'errors': met_src['mae_per_mutant'],
                      'preds': {m: res_src['per_mutant'][m].tolist()
                                for m in res_src['per_mutant']},
                      'pca_cumvar': var_exp[PCA_DIM]}
    print(f"  Src ESM2 d={PCA_DIM}: MAE={met_src['mae']:.4f} "
          f"dir={met_src['direction']}")
    for m in ['SrcKD-F405A', 'SrcKD-V380A', 'SrcKD-V331A', 'SrcKD-A311I']:
        print(f"      {m:<14} true={np.round(np.array(SRC_K3[m]['pop']),2)} "
              f"pred={np.round(res_src['per_mutant'][m],2)} "
              f"err={met_src['mae_per_mutant'][m]:.4f}")

    # ---------- Abl1 core ----------
    abl1_core = {m: ABL1_K3[m] for m in ABL1_K3
                 if m not in ('WT', 'H396P', 'M290L_H396P')}
    print("\n[Abl1 core] embeddings...")
    pos_map = build_pos_map(abl1_core, ABL1_KD, 'abl1')
    embeddings, mut_pos = compute_all_embeddings(model, tokenizer, ABL1_KD,
                                                 abl1_core, pos_map, 'abl1', device)
    deltas, rows = compute_delta_encodings(embeddings, mut_pos)
    _, reduced, var_exp = fit_pca_and_reduce(deltas, rows, dims=[PCA_DIM])
    enc_abl1 = reduced[PCA_DIM]
    print(f"  PCA cumvar (d={PCA_DIM}): {var_exp[PCA_DIM]:.3f}")
    res_abl1 = run_loo(abl1_core, ABL1_K3_WT_POP,
                       lambda m, d: enc_abl1[m], PCA_DIM)
    met_abl1 = metrics(res_abl1['per_mutant'], res_abl1['targets'], ABL1_K3_WT_POP)
    results['abl1_core'] = {'mae': met_abl1['mae'],
                            'direction': met_abl1['direction'],
                            'errors': met_abl1['mae_per_mutant'],
                            'preds': {m: res_abl1['per_mutant'][m].tolist()
                                      for m in res_abl1['per_mutant']},
                            'pca_cumvar': var_exp[PCA_DIM]}
    print(f"  Abl1 core ESM2 d={PCA_DIM}: MAE={met_abl1['mae']:.4f} "
          f"dir={met_abl1['direction']}")

    # ---------- Abl1 ext (add H396P, M290L_H396P) ----------
    abl1_ext = {m: ABL1_K3[m] for m in ABL1_K3 if m != 'WT'}
    try:
        print("\n[Abl1 ext] embeddings (H396P, M290L_H396P)...")
        pos_map = build_pos_map(abl1_ext, ABL1_KD, 'abl1')
        embeddings, mut_pos = compute_all_embeddings(model, tokenizer, ABL1_KD,
                                                     abl1_ext, pos_map, 'abl1', device)
        deltas, rows = compute_delta_encodings(embeddings, mut_pos)
        _, reduced, var_exp = fit_pca_and_reduce(deltas, rows, dims=[PCA_DIM])
        enc_abl1x = reduced[PCA_DIM]
        res_abl1x = run_loo(abl1_ext, ABL1_K3_WT_POP,
                            lambda m, d: enc_abl1x[m], PCA_DIM)
        met_abl1x = metrics(res_abl1x['per_mutant'], res_abl1x['targets'],
                            ABL1_K3_WT_POP)
        results['abl1_ext'] = {'mae': met_abl1x['mae'],
                               'direction': met_abl1x['direction'],
                               'errors': met_abl1x['mae_per_mutant'],
                               'pca_cumvar': var_exp[PCA_DIM]}
        print(f"  Abl1 ext ESM2 d={PCA_DIM}: MAE={met_abl1x['mae']:.4f} "
              f"dir={met_abl1x['direction']}")
    except Exception as e:
        print(f"  Abl1 ext FAILED: {e}")
        results['abl1_ext'] = {'error': str(e)}

    out_dir = Path(__file__).parent / 'results'
    out_json = out_dir / 'k3_esm2_results.json'
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False,
                                   default=float), encoding='utf-8')
    print(f"\n[OK] {out_json}  (total {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
