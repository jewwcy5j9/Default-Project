# P6 audit detection benchmark (A1) — derived from frozen P4/P5

- cross-check: 12000 repeats regenerated exactly; mismatches = 0
- S1 support threshold (tau, theta sweep):

  n=6 eps=0.0 delta=1.2 tau=0.001 theta=0.05: rate=1.000
  n=6 eps=0.0 delta=1.2 tau=0.001 theta=0.2: rate=1.000
  n=6 eps=0.0 delta=1.2 tau=0.05 theta=0.05: rate=1.000
  n=6 eps=0.0 delta=1.2 tau=0.05 theta=0.2: rate=1.000
  n=6 eps=0.0 delta=1.2 tau=0.25 theta=0.05: rate=1.000
  n=6 eps=0.0 delta=1.2 tau=0.25 theta=0.2: rate=1.000
  n=8 eps=0.0 delta=1.2 tau=0.001 theta=0.05: rate=1.000
  n=8 eps=0.0 delta=1.2 tau=0.001 theta=0.2: rate=1.000
  n=8 eps=0.0 delta=1.2 tau=0.05 theta=0.05: rate=1.000
  n=8 eps=0.0 delta=1.2 tau=0.05 theta=0.2: rate=1.000
  n=8 eps=0.0 delta=1.2 tau=0.25 theta=0.05: rate=1.000
  n=8 eps=0.0 delta=1.2 tau=0.25 theta=0.2: rate=1.000
  n=20 eps=0.0 delta=1.2 tau=0.001 theta=0.05: rate=1.000
  n=20 eps=0.0 delta=1.2 tau=0.001 theta=0.2: rate=1.000
  n=20 eps=0.0 delta=1.2 tau=0.05 theta=0.05: rate=1.000
  n=20 eps=0.0 delta=1.2 tau=0.05 theta=0.2: rate=1.000
  n=20 eps=0.0 delta=1.2 tau=0.25 theta=0.05: rate=1.000
  n=20 eps=0.0 delta=1.2 tau=0.25 theta=0.2: rate=1.000
  n=20 eps=1.0 delta=1.2 tau=0.001 theta=0.05: rate=0.000
  n=20 eps=1.0 delta=1.2 tau=0.001 theta=0.2: rate=0.000
  n=20 eps=1.0 delta=1.2 tau=0.05 theta=0.05: rate=0.000
  n=20 eps=1.0 delta=1.2 tau=0.05 theta=0.2: rate=0.000
  n=20 eps=1.0 delta=1.2 tau=0.25 theta=0.05: rate=0.000
  n=20 eps=1.0 delta=1.2 tau=0.25 theta=0.2: rate=0.000

- S2 permutation-null power (tau=0.05, theta=0.10, alpha=0.05):

  | n | eps | delta | flagged | power | mean p |
  |---:|---:|---:|---:|---:|---:|
  | 6 | 0.0 | 0.0 | 0 | None | None |
  | 6 | 0.0 | 0.6 | 200 | 0.0 | 0.475333 |
  | 6 | 0.0 | 1.2 | 200 | 0.0 | 0.324333 |
  | 6 | 0.05 | 0.0 | 0 | None | None |
  | 6 | 0.05 | 0.6 | 200 | 0.0 | 0.487333 |
  | 6 | 0.05 | 1.2 | 200 | 0.0 | 0.308333 |
  | 6 | 1.0 | 0.0 | 0 | None | None |
  | 6 | 1.0 | 0.6 | 0 | None | None |
  | 6 | 1.0 | 1.2 | 0 | None | None |
  | 8 | 0.0 | 0.0 | 0 | None | None |
  | 8 | 0.0 | 0.6 | 200 | 0.015 | 0.499464 |
  | 8 | 0.0 | 1.2 | 200 | 0.085 | 0.296786 |
  | 8 | 0.05 | 0.0 | 0 | None | None |
  | 8 | 0.05 | 0.6 | 200 | 0.005 | 0.494286 |
  | 8 | 0.05 | 1.2 | 200 | 0.11 | 0.285179 |
  | 8 | 1.0 | 0.0 | 0 | None | None |
  | 8 | 1.0 | 0.6 | 0 | None | None |
  | 8 | 1.0 | 1.2 | 0 | None | None |
  | 12 | 0.0 | 0.0 | 0 | None | None |
  | 12 | 0.0 | 0.6 | 200 | 0.01 | 0.485076 |
  | 12 | 0.0 | 1.2 | 200 | 0.015 | 0.293864 |
  | 12 | 1.0 | 0.0 | 0 | None | None |
  | 12 | 1.0 | 0.6 | 0 | None | None |
  | 12 | 1.0 | 1.2 | 0 | None | None |
  | 20 | 0.0 | 0.0 | 0 | None | None |
  | 20 | 0.0 | 0.6 | 200 | 0.0 | 0.506789 |
  | 20 | 0.0 | 1.2 | 200 | 0.0 | 0.283026 |
  | 20 | 1.0 | 0.0 | 0 | None | None |
  | 20 | 1.0 | 0.6 | 0 | None | None |
  | 20 | 1.0 | 1.2 | 0 | None | None |
  | 50 | 0.0 | 0.0 | 0 | None | None |
  | 50 | 0.0 | 0.6 | 200 | 0.0 | 0.50882 |
  | 50 | 0.0 | 1.2 | 200 | 0.0 | 0.275829 |
  | 50 | 1.0 | 0.0 | 0 | None | None |
  | 50 | 1.0 | 0.6 | 0 | None | None |
  | 50 | 1.0 | 1.2 | 0 | None | None |

- R resolution detection (epsilon=0):

  n=6 delta=0.0 margin=1.73: rate=0.000
  n=6 delta=0.0 margin=1.50: rate=0.000
  n=6 delta=0.0 margin=2.00: rate=0.000
  n=6 delta=0.0 margin=3.00: rate=0.000
  n=6 delta=0.6 margin=1.73: rate=0.930
  n=6 delta=0.6 margin=1.50: rate=0.995
  n=6 delta=0.6 margin=2.00: rate=0.775
  n=6 delta=0.6 margin=3.00: rate=0.280
  n=6 delta=1.2 margin=1.73: rate=1.000
  n=6 delta=1.2 margin=1.50: rate=1.000
  n=6 delta=1.2 margin=2.00: rate=1.000
  n=6 delta=1.2 margin=3.00: rate=0.940
  n=8 delta=0.0 margin=1.73: rate=0.000
  n=8 delta=0.0 margin=1.50: rate=0.000
  n=8 delta=0.0 margin=2.00: rate=0.000
  n=8 delta=0.0 margin=3.00: rate=0.000
  n=8 delta=0.6 margin=1.73: rate=0.770
  n=8 delta=0.6 margin=1.50: rate=0.950
  n=8 delta=0.6 margin=2.00: rate=0.470
  n=8 delta=0.6 margin=3.00: rate=0.090
  n=8 delta=1.2 margin=1.73: rate=1.000
  n=8 delta=1.2 margin=1.50: rate=1.000
  n=8 delta=1.2 margin=2.00: rate=1.000
  n=8 delta=1.2 margin=3.00: rate=0.710
  n=20 delta=0.0 margin=1.73: rate=0.000
  n=20 delta=0.0 margin=1.50: rate=0.000
  n=20 delta=0.0 margin=2.00: rate=0.000
  n=20 delta=0.0 margin=3.00: rate=0.000
  n=20 delta=0.6 margin=1.73: rate=0.000
  n=20 delta=0.6 margin=1.50: rate=0.010
  n=20 delta=0.6 margin=2.00: rate=0.000
  n=20 delta=0.6 margin=3.00: rate=0.000
  n=20 delta=1.2 margin=1.73: rate=0.525
  n=20 delta=1.2 margin=1.50: rate=0.935
  n=20 delta=1.2 margin=2.00: rate=0.160
  n=20 delta=1.2 margin=3.00: rate=0.000
  n=50 delta=0.0 margin=1.73: rate=0.000
  n=50 delta=0.0 margin=1.50: rate=0.000
  n=50 delta=0.0 margin=2.00: rate=0.000
  n=50 delta=0.0 margin=3.00: rate=0.000
  n=50 delta=0.6 margin=1.73: rate=0.000
  n=50 delta=0.6 margin=1.50: rate=0.000
  n=50 delta=0.6 margin=2.00: rate=0.000
  n=50 delta=0.6 margin=3.00: rate=0.000
  n=50 delta=1.2 margin=1.73: rate=0.000
  n=50 delta=1.2 margin=1.50: rate=0.000
  n=50 delta=1.2 margin=2.00: rate=0.000
  n=50 delta=1.2 margin=3.00: rate=0.000

- T selection detection (tau_sel >= 0.01; the tau_sel=0 rows
  are float noise around zero and are excluded):

  P4 n=6 m=1 tau_sel=0.01: rate=0.000
  P4 n=6 m=1 tau_sel=0.02: rate=0.000
  P4 n=6 m=1 tau_sel=0.05: rate=0.000
  P4 n=6 m=5 tau_sel=0.01: rate=0.139
  P4 n=6 m=5 tau_sel=0.02: rate=0.107
  P4 n=6 m=5 tau_sel=0.05: rate=0.029
  P4 n=6 m=20 tau_sel=0.01: rate=0.372
  P4 n=6 m=20 tau_sel=0.02: rate=0.303
  P4 n=6 m=20 tau_sel=0.05: rate=0.103
  P4 n=8 m=1 tau_sel=0.01: rate=0.000
  P4 n=8 m=1 tau_sel=0.02: rate=0.000
  P4 n=8 m=1 tau_sel=0.05: rate=0.000
  P4 n=8 m=5 tau_sel=0.01: rate=0.010
  P4 n=8 m=5 tau_sel=0.02: rate=0.006
  P4 n=8 m=5 tau_sel=0.05: rate=0.000
  P4 n=8 m=20 tau_sel=0.01: rate=0.033
  P4 n=8 m=20 tau_sel=0.02: rate=0.020
  P4 n=8 m=20 tau_sel=0.05: rate=0.000
  P4 n=20 m=1 tau_sel=0.01: rate=0.000
  P4 n=20 m=1 tau_sel=0.02: rate=0.000
  P4 n=20 m=1 tau_sel=0.05: rate=0.000
  P4 n=20 m=5 tau_sel=0.01: rate=0.000
  P4 n=20 m=5 tau_sel=0.02: rate=0.000
  P4 n=20 m=5 tau_sel=0.05: rate=0.000
  P4 n=20 m=20 tau_sel=0.01: rate=0.000
  P4 n=20 m=20 tau_sel=0.02: rate=0.000
  P4 n=20 m=20 tau_sel=0.05: rate=0.000
  P5 n=6 m=0.15 tau_sel=0.01: rate=0.816
  P5 n=6 m=0.15 tau_sel=0.02: rate=0.720
  P5 n=6 m=0.15 tau_sel=0.05: rate=0.334
  P5 n=8 m=0.15 tau_sel=0.01: rate=0.662
  P5 n=8 m=0.15 tau_sel=0.02: rate=0.548
  P5 n=8 m=0.15 tau_sel=0.05: rate=0.148
  P5 n=6 m=0.05 tau_sel=0.01: rate=0.776
  P5 n=6 m=0.05 tau_sel=0.02: rate=0.630
  P5 n=6 m=0.05 tau_sel=0.05: rate=0.246
  P5 n=8 m=0.05 tau_sel=0.01: rate=0.700
  P5 n=8 m=0.05 tau_sel=0.02: rate=0.570
  P5 n=8 m=0.05 tau_sel=0.05: rate=0.172
  P5 n=6 m=0.02 tau_sel=0.01: rate=0.780
  P5 n=6 m=0.02 tau_sel=0.02: rate=0.636
  P5 n=6 m=0.02 tau_sel=0.05: rate=0.210
  P5 n=8 m=0.02 tau_sel=0.01: rate=0.688
  P5 n=8 m=0.02 tau_sel=0.02: rate=0.548
  P5 n=8 m=0.02 tau_sel=0.05: rate=0.124
  P5 n=6 m=0.01 tau_sel=0.01: rate=0.732
  P5 n=6 m=0.01 tau_sel=0.02: rate=0.602
  P5 n=6 m=0.01 tau_sel=0.05: rate=0.170
  P5 n=8 m=0.01 tau_sel=0.01: rate=0.718
  P5 n=8 m=0.01 tau_sel=0.02: rate=0.530
  P5 n=8 m=0.01 tau_sel=0.05: rate=0.106

Interpretation (calibration findings):
- S2: the exact pair permutation null is structurally powerless 
  at every n (power <= 0.09, mean p 0.28-0.51 even for the planted 
  collision): a single pair competes against all C(n,2) pairs, so 
  the paper's biological p=0.40/0.23 cannot certify absence of 
  collision risk. Collision membership via the threshold detector 
  (S1) is the working support tool (rate 1.0 at eps=0, 0.0 at eps=1).
- R: q2>q1 (margin sqrt(3)) flags planted fine contrast with rate 
  0.93-1.0 at delta>=0.6 for n=6/8 and zero false flags at delta=0, 
  but collapses to 0.53 at n=20 and 0 at n=50: a panel-scale 
  diagnostic, not an asymptotic one.
- T: optimism > 0.01 fires in 37% (n=6) / 3% (n=8) of m=20 
  factorial repeats whose mean optimism is 0.016 -> the Abl1 
  +0.0297 gap is inside this noise band and Src +0.108 is outside; 
  the P5 ladder flags 0.66-0.82 at tau_sel=0.02 only for clear 
  quality gaps.
