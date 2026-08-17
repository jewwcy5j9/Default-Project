"""
P1-1: Data Hygiene Audit

1. List which NMR values are direct vs inferred (normalized)
2. Direction accuracy denominator (2-state vs 3-state, gold vs silver)
3. F382L tier verification
"""

import json
from pathlib import Path


def audit_data():
    """Audit NMR data quality."""
    
    print("=" * 70)
    print("P1-1: Data Hygiene Audit")
    print("=" * 70)
    
    # Load NMR data
    data_path = Path(__file__).parent.parent.parent / 'data' / 'nmr_populations' / 'xie2020_abl1_FINAL.json'
    with open(data_path) as f:
        nmr = json.load(f)
    
    pops = nmr['populations']
    
    # 1. Direct vs Inferred values
    print("\n" + "-" * 70)
    print("1. VALUE SOURCE AUDIT (Direct vs Inferred)")
    print("-" * 70)
    
    print(f"\n{'Mutant':<15} {'Active':>10} {'I1':>10} {'I2':>10} {'Notes'}")
    print("-" * 70)
    
    for mut, data in pops.items():
        if data['tier'] == 'unavailable':
            continue
        
        tier = data['tier']
        notes = data.get('notes', '')
        
        # Determine which values are direct vs inferred
        a_src = '?' if data['Active'] is None else ('D' if 'Active=' in notes or 'pG=' in notes or 'pA=' in notes else 'N')
        i1_src = '?' if data['I1'] is None else ('D' if 'I1=' in notes else 'N')
        i2_src = '?' if data['I2'] is None else ('D' if 'I2=' in notes or 'pI2=' in notes else 'N')
        
        a_val = f"{data['Active']:.0%}" if data['Active'] is not None else "—"
        i1_val = f"{data['I1']:.0%}" if data['I1'] is not None else "—"
        i2_val = f"{data['I2']:.0%}" if data['I2'] is not None else "—"
        
        # Mark inferred values
        if a_src == 'N':
            a_val += '*'
        if i1_src == 'N':
            i1_val += '*'
        if i2_src == 'N':
            i2_val += '*'
        
        print(f"  {mut:<13} {a_val:>10} {i1_val:>10} {i2_val:>10}  {tier}")
    
    print("\n  Legend: D=Direct from paper, N=Normalized/Inferred, *=inferred")
    
    # Detailed breakdown
    print("\n" + "-" * 70)
    print("DETAILED SOURCE BREAKDOWN")
    print("-" * 70)
    
    source_details = {
        'WT': {
            'Active': ('Direct', 'Fig.1 CEST: pG=88%'),
            'I1': ('Direct', 'Fig.1 CEST: pE1=6%'),
            'I2': ('Direct', 'Fig.1 CEST: pE2=6%'),
        },
        'M290L': {
            'Active': ('Inferred', '100% - I1 - I2 = 100% - 10% - 35% = 55%'),
            'I1': ('Direct', 'Fig.4B: I1=10%'),
            'I2': ('Direct', 'Fig.4B: I2=35%'),
        },
        'L301I': {
            'Active': ('Inferred', '100% - I1 - I2, with I1~6% assumed unchanged'),
            'I1': ('Assumed', '~6% (assumed unchanged from WT)'),
            'I2': ('Direct', 'Fig.4B: I2=65%'),
        },
        'M290L_L301I': {
            'Active': ('Direct', 'Fig.4B: Active=8%'),
            'I1': ('Inferred', '100% - 8% - 82% = 10%'),
            'I2': ('Direct', 'Fig.4B: I2=82%'),
        },
        'T315I': {
            'Active': ('Direct*', 'Fig.4D: pA~93% (in I2M background)'),
            'I1': ('Unknown', 'Not reported'),
            'I2': ('Direct*', 'Fig.4D: pI2~7% (in I2M background)'),
        },
        'F382L': {
            'Active': ('Approx', 'Fig.4E: "similar to WT" → ~88%'),
            'I1': ('Approx', 'Fig.4E: "similar to WT" → ~6%'),
            'I2': ('Approx', 'Fig.4E: "similar to WT" → ~6%'),
        },
        'F382Y': {
            'Active': ('Inferred', 'Residual ~10% split with I1, assumed 5%'),
            'I1': ('Inferred', 'Residual ~10% split with Active, assumed 5%'),
            'I2': ('Direct', 'Fig.4E: pI2~90%'),
        },
    }
    
    for mut, details in source_details.items():
        print(f"\n  {mut}:")
        for state, (src, note) in details.items():
            print(f"    {state}: [{src}] {note}")
    
    # 2. Direction accuracy denominator
    print("\n" + "-" * 70)
    print("2. DIRECTION ACCURACY DENOMINATOR")
    print("-" * 70)
    
    print("""
  Current headline: 64.8% direction accuracy
  
  Breakdown needed:
  - 2-state vs 3-state calculation?
  - Gold-only vs Gold+Silver?
  - Per-mutant direction results?
  
  From headline_comparison.py (2-state, 6 mutants):
  - CDST direction: 66.7% (4/6 correct)
  - AF2 direction: 83.3% (5/6 correct)
  
  Per-mutant (Non-Ground direction):
  - M290L: NMR ↑, CDST ↑, AF2 ↑ → Both correct
  - L301I: NMR ↑, CDST ↑, AF2 ↑ → Both correct  
  - M290L_L301I: NMR ↑, CDST ↑, AF2 ↑ → Both correct
  - T315I: NMR ↓, CDST ↑, AF2 ↓ → AF2 correct, CDST wrong
  - F382L: NMR =, CDST ↑, AF2 ↑ → Tie (no change expected)
  - F382Y: NMR ↑, CDST ↑, AF2 ↑ → Both correct
  
  Note: T315I is silver tier (measured in I2M background)
  Gold-only (4 mutants): CDST 4/4 = 100%, AF2 4/4 = 100%
""")
    
    # 3. F382L tier verification
    print("-" * 70)
    print("3. F382L TIER VERIFICATION")
    print("-" * 70)
    
    print("""
  Current tier: silver
  Reason: "Qualitative: 'mainly occupies active state, similar to WT'"
  
  Question: Does Fig.4E have precise three-state values?
  
  From Xie 2020 Fig.4E description:
  - F401L (F382L): "mainly occupies active state, similar to WT"
  - F401V (F382V): "I2 >95%; I1 not stabilized"
  - F401Y (F382Y): "pI2~90%"
  
  Assessment:
  - F382L is described qualitatively, not quantitatively
  - No explicit percentages given (unlike F382Y with "pI2~90%")
  - Tier should remain SILVER (approximate, not direct measurement)
  
  Recommendation: Keep F382L as silver, note in paper that
  "F382L populations approximated as WT-like based on qualitative
  description in Xie 2020 Fig.4E"
""")
    
    # Summary
    print("=" * 70)
    print("SUMMARY: DATA QUALITY TIERS")
    print("=" * 70)
    
    print("""
  GOLD (direct CEST measurement):
    - WT: All three states direct
    - M290L: I1, I2 direct; Active inferred by normalization
    - M290L_L301I: Active, I2 direct; I1 inferred
    - F382Y: I2 direct; Active, I1 inferred (split residual)
  
  GOLD_NORMALIZED (one state inferred):
    - L301I: I2 direct; I1 assumed unchanged; Active inferred
  
  SILVER (qualitative/indirect):
    - T315I: Measured in I2M background, not isolated kinase
    - F382L: "Similar to WT" approximation
  
  UNAVAILABLE:
    - E255V: Too unstable for CEST
    - H361P: Not studied in Xie 2020
  
  For headline statistics:
    - Use GOLD + GOLD_NORMALIZED (5 mutants)
    - Report SILVER separately as sensitivity analysis
""")


if __name__ == '__main__':
    audit_data()
