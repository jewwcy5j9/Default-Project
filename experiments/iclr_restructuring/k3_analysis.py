"""K=3 post-hoc analysis: mean-predictor collapse + per-mutant diagnostics."""
import sys
import json
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RES = Path(__file__).parent / 'results' / 'k3_benchmark_results.json'
d = json.loads(RES.read_text(encoding='utf-8'))

WT_A = [0.88, 0.06, 0.06]
WT_S = [0.72, 0.07, 0.21]
TGT_A = {'M290L': [0.55, 0.10, 0.35], 'L301I': [0.25, 0.10, 0.65],
         'M290L_L301I': [0.08, 0.10, 0.82], 'F382L': [0.88, 0.06, 0.06],
         'F382Y': [0.10, 0, 0.90], 'F382V': [0.05, 0, 0.95]}
TGT_S = {'SrcKD-L410A': [0.73, 0.27, 0], 'SrcKD-V332I': [0.48, 0.52, 0],
         'SrcKD-L270F_V332I': [0.09, 0.91, 0], 'SrcKD-L325A': [0, 1, 0],
         'SrcKD-A311I': [0, 1, 0], 'SrcKD-V380A': [0, 0.62, 0.38],
         'SrcKD-V331A': [0, 0.45, 0.55], 'SrcKD-F405A': [0, 0.16, 0.84]}


def collapse_stats(enc, targets, wt_pop):
    names = list(enc['per_mutant_preds'].keys())
    shift_t = [float(np.abs(np.array(targets[m]) - np.array(wt_pop)).sum())
               for m in names]
    shift_p = [float(np.abs(np.array(enc['per_mutant_preds'][m]) - np.array(wt_pop)).sum())
               for m in names]
    var_t = float(np.var(shift_t))
    var_p = float(np.var(shift_p))
    return var_t, var_p, (var_p / var_t * 100 if var_t > 0 else None)


def per_mutant_table(enc, targets):
    names = list(enc['per_mutant_preds'].keys())
    rows = []
    for m in names:
        t = np.array(targets[m])
        p = np.array(enc['per_mutant_preds'][m])
        rows.append((m, t, p, float(np.abs(p - t).mean())))
    return rows


print("=" * 80)
print("K=3 collapse analysis (pred shift variance / target shift variance)")
print("=" * 80)
print("\n--- Abl1 core (n=6) ---")
for ek in ['Extended_10dim', 'C_ddg_5dim', 'Random_10dim', 'Shuffled_10dim',
           'Onehot', 'pos_markers']:
    vt, vp, r = collapse_stats(d['abl1_core'][ek], TGT_A, WT_A)
    print(f"  {ek:<20} ratio = {r:6.1f}%   (target var {vt:.4f})")

print("\n--- Src (n=8) ---")
for ek in ['Extended_10dim', 'pos_markers_4dim', 'Random_10dim',
           'Shuffled_10dim', 'Onehot']:
    vt, vp, r = collapse_stats(d['src'][ek], TGT_S, WT_S)
    print(f"  {ek:<20} ratio = {r:6.1f}%   (target var {vt:.4f})")

print("\n--- Src per-mutant errors: Extended vs pos-markers (K=3) ---")
e_ext = d['src']['Extended_10dim']['mae_per_mutant']
e_pos = d['src']['pos_markers_4dim']['mae_per_mutant']
print(f"  {'mutant':<20} {'Ext':>7} {'pos':>7} {'delta':>7}")
for m in TGT_S:
    print(f"  {m:<20} {e_ext[m]:7.4f} {e_pos[m]:7.4f} {e_ext[m]-e_pos[m]:+7.4f}")

print("\n--- Abl1 core per-mutant errors: Extended vs C (K=3) ---")
e_ext = d['abl1_core']['Extended_10dim']['mae_per_mutant']
e_c = d['abl1_core']['C_ddg_5dim']['mae_per_mutant']
for m in TGT_A:
    print(f"  {m:<16} {e_ext[m]:7.4f} {e_c[m]:7.4f} {e_ext[m]-e_c[m]:+7.4f}")

print("\n--- Src Extended vs pos: which states dominate the error? ---")
for m in ['SrcKD-V332I', 'SrcKD-V380A', 'SrcKD-F405A', 'SrcKD-L270F_V332I']:
    t = np.array(TGT_S[m])
    p_ext = np.array(d['src']['Extended_10dim']['per_mutant_preds'][m])
    p_pos = np.array(d['src']['pos_markers_4dim']['per_mutant_preds'][m])
    print(f"  {m:<20} true={np.round(t,2)} ext={np.round(p_ext,2)} "
          f"pos={np.round(p_pos,2)}")
