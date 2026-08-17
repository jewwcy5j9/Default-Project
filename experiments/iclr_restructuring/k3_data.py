"""
K=3 benchmark data module (Phase-1 revamp).

3-state populations for Abl1 (Xie 2020) and Src (Cui 2025). K=3 makes the
simplex geometry and the rank-2 low-rank structure non-vacuous (K=2 collapses
to scalar regression).

Data provenance:
  - Abl1: data/nmr_populations/xie2020_abl1_FINAL.json (populations field,
    Active/I1/I2). H396P (silver) and M290L_H396P (silver, pH 6.5) extend
    the benchmark beyond the canonical 6 mutants and add the I1 direction.
  - Src : data/nmr_populations/src_k3_canonical.csv, primary Fig S5 Met305
    probe records (A/E1/E2). The explicit L410A-only Table S2 substitution is
    available through src.data.src_k3_labels, not through SRC_K3.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.src_k3_labels import (  # noqa: E402
    SRC_K3_PRIMARY_PROTOCOL_ID,
    build_src_k3_panel,
)

from encoding_ablation_control import (
    AA_PROPERTIES_6_EXT, encode_extended, encode_random_gaussian,
    encode_shuffled_property, encode_onehot_mutation, make_shuffled_table,
    ABL1_SEQ_LEN, SRC_SEQ_LEN,
)
from alternative_encodings import (
    DDG_DATA, DDG_NORM, BLOSUM_NORM,
    encode_ddg_main, encode_blosum62,
    encode_dvol_normalized, encode_no_dvol,
)

# ============================================================
# Abl1 (Xie 2020 Science) - 3-state [Active, I1, I2]
# ============================================================

ABL1_K3 = {
    'WT':            {'wt': None, 'mut': None, 'pos': None, 'pop': [0.88, 0.06, 0.06], 'tier': 'gold'},
    'M290L':         {'wt': 'M',  'mut': 'L',  'pos': 290, 'pop': [0.55, 0.10, 0.35], 'tier': 'gold'},
    'L301I':         {'wt': 'L',  'mut': 'I',  'pos': 301, 'pop': [0.25, 0.10, 0.65], 'tier': 'gold'},
    'M290L_L301I':   {'wt': 'ML', 'mut': 'LI', 'pos': 301, 'pop': [0.08, 0.10, 0.82], 'tier': 'gold'},
    'F382L':         {'wt': 'F',  'mut': 'L',  'pos': 382, 'pop': [0.88, 0.06, 0.06], 'tier': 'gold'},
    'F382Y':         {'wt': 'F',  'mut': 'Y',  'pos': 382, 'pop': [0.10, 0.00, 0.90], 'tier': 'gold'},
    'F382V':         {'wt': 'F',  'mut': 'V',  'pos': 382, 'pop': [0.05, 0.00, 0.95], 'tier': 'gold'},
    'H396P':         {'wt': 'H',  'mut': 'P',  'pos': 396, 'pop': [0.85, 0.15, 0.00], 'tier': 'silver'},
    'M290L_H396P':   {'wt': 'MH', 'mut': 'LP', 'pos': 396, 'pop': [0.50, 0.50, 0.00], 'tier': 'silver_pH65'},
}
ABL1_K3_WT_POP = [0.88, 0.06, 0.06]
ABL1_K3_CORE = [m for m in ABL1_K3 if m != 'H396P' and m != 'M290L_H396P']  # WT + 6 canonical
ABL1_K3_EXT = list(ABL1_K3.keys())  # WT + 6 + 2 (n=8 mutants + WT)

# ============================================================
# Src (Cui 2025 Science, Fig S5 Met305 probe) - [Active, E1, E2]
# ============================================================

_SRC_PRIMARY_PANEL = build_src_k3_panel(SRC_K3_PRIMARY_PROTOCOL_ID)
_SRC_FEATURE_METADATA = {
    'SrcKD-WT': {'wt': None, 'mut': None, 'pos': None},
    'SrcKD-L410A': {'wt': 'L', 'mut': 'A', 'pos': 410},
    'SrcKD-V332I': {'wt': 'V', 'mut': 'I', 'pos': 332},
    'SrcKD-L270F_V332I': {'wt': 'LV', 'mut': 'FI', 'pos': 270},
    'SrcKD-L325A': {'wt': 'L', 'mut': 'A', 'pos': 325},
    'SrcKD-A311I': {'wt': 'A', 'mut': 'I', 'pos': 311},
    'SrcKD-V380A': {'wt': 'V', 'mut': 'A', 'pos': 380},
    'SrcKD-V331A': {'wt': 'V', 'mut': 'A', 'pos': 331},
    'SrcKD-F405A': {'wt': 'F', 'mut': 'A', 'pos': 405},
}
SRC_K3 = {}
for _name, _metadata in _SRC_FEATURE_METADATA.items():
    _population = (_SRC_PRIMARY_PANEL.wt_population if _name == 'SrcKD-WT'
                   else _SRC_PRIMARY_PANEL.targets[_name])
    _record_id = (_SRC_PRIMARY_PANEL.wt_record_id if _name == 'SrcKD-WT'
                  else _SRC_PRIMARY_PANEL.target_record_ids[_name])
    SRC_K3[_name] = {
        **_metadata,
        'pop': list(_population),
        'tier': 'primary_probe',
        'label_record_id': _record_id,
    }
SRC_K3_WT_POP = list(_SRC_PRIMARY_PANEL.wt_population)
SRC_K3_PROTOCOL_ID = _SRC_PRIMARY_PANEL.protocol_id
SRC_K3_CANONICAL_SHA256 = _SRC_PRIMARY_PANEL.canonical_sha256

# ============================================================
# Src perturbation extension (Cui 2025; phosphorylation / motif constructs)
# - SrcKD-pY419: autophosphorylation, active-like [90,5,5] (Fig S5 approx., low)
# - SrcpY530  : C-terminal phosphorylation, E2 reference [0,0,100] (Fig S5+Table S2, medium)
# - SrcYEEI   : YEEI-motif construct, E2 reference [0,0,100] (Table S2, medium)
# type flag: 0 = point mutation, 1 = phosphorylation, 2 = motif construct
# ============================================================

SRC_PERT = {
    'SrcKD-pY419': {'wt': 'Y', 'mut': 'pY', 'pos': 419, 'pop': [0.90, 0.05, 0.05],
                    'tier': 'low', 'type': 1, 'note': 'autophosphorylation, active-like'},
    'SrcpY530':    {'wt': 'Y', 'mut': 'pY', 'pos': 530, 'pop': [0.00, 0.00, 1.00],
                    'tier': 'medium', 'type': 1, 'note': 'C-terminal phosphorylation, E2 reference'},
    'SrcYEEI':     {'wt': None, 'mut': None, 'pos': None, 'pop': [0.00, 0.00, 1.00],
                    'tier': 'medium', 'type': 2, 'note': 'YEEI-motif construct, E2 reference'},
}
SRC_K3_EXT = dict(SRC_K3)
SRC_K3_EXT.update(SRC_PERT)  # 11 examples (8 mutants + 3 perturbations)

# Type-flagged encoders (Extended / pos-markers / no-dVol) for the 11-example set
def _type_flag(data):
    return float(data.get('type', 0))


def enc_src_extended_type(name, data):
    d = dict(data)
    if d.get('pos') is None:
        d['pos'] = 0
    if d.get('wt') is None:
        d['wt'] = d['mut'] = 'Y'  # no single-residue change; delta = 0
    enc = encode_extended(name, d, AA_PROPERTIES_6_EXT, SRC_SEQ_LEN, system='src')
    return np.concatenate([enc, [_type_flag(data)]])


def enc_src_pos_type(name, data):
    enc = np.zeros(4)
    if data['pos']:
        enc[0] = data['pos'] / SRC_SEQ_LEN
    for i, p in enumerate([311, 332, 380]):
        if data['pos'] == p:
            enc[i + 1] = 1.0
    return np.concatenate([enc, [_type_flag(data)]])


def enc_src_no_dvol_type(name, data):
    d = dict(data)
    if d.get('pos') is None:
        d['pos'] = 0
    if d.get('wt') is None:
        d['wt'] = d['mut'] = 'Y'  # no single-residue change; delta = 0
    enc10 = encode_extended(name, d, AA_PROPERTIES_6_EXT, SRC_SEQ_LEN, system='src')
    keep = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    return np.concatenate([enc10[keep], [_type_flag(data)]])

# ============================================================
# Encoders (K-independent feature vectors)
# ============================================================

def enc_abl1_extended(name, data):
    return encode_extended(name, data, AA_PROPERTIES_6_EXT, ABL1_SEQ_LEN, system='abl1')

def enc_abl1_dvol_normalized(name, data):
    return encode_dvol_normalized(name, data, AA_PROPERTIES_6_EXT, ABL1_SEQ_LEN, system='abl1')

def enc_abl1_no_dvol(name, data):
    return encode_no_dvol(name, data, DDG_DATA, DDG_NORM, AA_PROPERTIES_6_EXT,
                          ABL1_SEQ_LEN, system='abl1')

def enc_abl1_ddg_main(name, data):
    return encode_ddg_main(name, data, DDG_DATA, DDG_NORM, ABL1_SEQ_LEN, system='abl1')

def enc_abl1_blosum62(name, data):
    return encode_blosum62(name, data, ABL1_SEQ_LEN, system='abl1')

def enc_abl1_pos_markers(name, data, extra_positions=()):
    """[pos/seq, pos290, pos301, pos382] (+ pos396 for the extended set)."""
    positions = [290, 301, 382] + list(extra_positions)
    enc = np.zeros(len(positions) + 1)
    enc[0] = data['pos'] / ABL1_SEQ_LEN if data['pos'] else 0.0
    for i, p in enumerate(positions):
        if data['pos'] == p:
            enc[i + 1] = 1.0
    return enc

def enc_abl1_onehot(name, data, names):
    return encode_onehot_mutation(name, data, names)

def enc_src_extended(name, data):
    return encode_extended(name, data, AA_PROPERTIES_6_EXT, SRC_SEQ_LEN, system='src')

def enc_src_no_dvol(name, data):
    enc10 = encode_extended(name, data, AA_PROPERTIES_6_EXT, SRC_SEQ_LEN, system='src')
    keep = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    return enc10[keep]

def enc_src_pos_markers(name, data):
    enc = np.zeros(4)
    enc[0] = data['pos'] / SRC_SEQ_LEN if data['pos'] else 0.0
    for i, p in enumerate([311, 332, 380]):
        if data['pos'] == p:
            enc[i + 1] = 1.0
    return enc

def enc_src_onehot(name, data, names):
    return encode_onehot_mutation(name, data, names)


def shuffled_factory(system, shuffle_seed=0, n_draws=5):
    """5-draw averaged Shuffled Property encoder (pre-registered control)."""
    tables = [make_shuffled_table(shuffle_seed + i) for i in range(n_draws)]
    if system == 'abl1':
        def fn(name, data):
            return np.mean([encode_shuffled_property(
                name, data, t, ABL1_SEQ_LEN, system='abl1') for t in tables], axis=0)
        return fn
    def fn(name, data):
        return np.mean([encode_shuffled_property(
            name, data, t, SRC_SEQ_LEN, system='src') for t in tables], axis=0)
    return fn


def random_factory(seed0=0, n_draws=5):
    """5-draw averaged Random Gaussian encoder (pre-registered control)."""
    def fn(name, data, seq_len):
        return np.mean([encode_random_gaussian(name, data, seed0 + i, seq_len)
                        for i in range(n_draws)], axis=0)
    return fn


# Abl1 one-hot needs mutation name list
ABL1_ONEHOT_NAMES = [m for m in ABL1_K3 if m != 'WT']
SRC_ONEHOT_NAMES = [m for m in SRC_K3 if m != 'SrcKD-WT']
