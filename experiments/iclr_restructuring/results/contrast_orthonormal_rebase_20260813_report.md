# Orthonormal contrast rebase (C4) — derived paper numbers

- constants: q1 error = (sqrt(6)/4) * u1 error = 0.612372 * u1; q2 error = u2/sqrt(2) = 0.707107 * u2
- ordering: q2 > q1  <=>  u2 > (sqrt(3)/2) u1 = 0.866025 u1
- verified rows:
30
- primary_probe: raw 13/15, rebased 13/15, CLR rows rebased 10/10
- l410a_global_fit_substitution: raw 11/15, rebased 13/15, CLR rows rebased 10/10
- pseudocount: raw 49/50, rebased 50/50
- stress both-orderings: raw 0.605, rebased 0.9050
- Table 2: GP-T5 q1=0.1993 q2=0.4276; MLP-current q1=0.3188 q2=0.4841; pooled q1-scale=0.3049
- primary_probe::pos::LowRankCDST: q1=0.3188 q2=0.4841
- primary_probe::pca20::LowRankCDST: q1=0.3591 q2=0.3544
- primary_probe::pos::CLR-Ridge: q1=0.2004 q2=0.4321
- primary_probe::pos::CLR-GP: q1=0.1993 q2=0.4276
- l410a_global_fit_substitution::pos::LowRankCDST: q1=0.3796 q2=0.429
- l410a_global_fit_substitution::pca20::LowRankCDST: q1=0.4485 q2=0.3815
- l410a_global_fit_substitution::pos::CLR-Ridge: q1=0.2392 q2=0.3688
- l410a_global_fit_substitution::pos::CLR-GP: q1=0.2347 q2=0.4496