# Prospective v3 validation report

Validated: 2026-08-11  
Status: **PASS; protocol package ready for a custodian, no external reveal performed**

## Immutability

The following v2 hashes remain unchanged:

- `preregistration.md`: `be28a1a1727ef9411baaf18c87f3812a8daa31582eee94d62fb8599a8f76109b`
- `analysis_plan.md`: `d17b74d09e898ed8a3816f1c87e2491e54348c1d494f00c80eeea479560a7eb1`
- `protocol_lock.json`: `e4794d4e8286f042e0151d94340f1ca90a01d13dfc8765fe97a2198895e743fd`
- `model_manifest.json`: `53b22977273f8feed5c792901b53fc66576f66af2341c411a61b96ab5640d355`
- `archive_20260808_adr001.zip`: `2a5112cbe555830466dcdd43afb35827cdb3ba6bd4d05f0c3ad47b435bf04185`

## Dry run

- Dataset: historical Abl1 six-mutant development fixture.
- Permanent label: `DRY RUN / NOT EVIDENCE`; panel tier: `supporting_only`.
- Primary v3 MAE: `0.14706476879083313`.
- Every PCA fit-ID list equals its outer-training IDs; no held-out mutation appears in fit IDs.
- Every fold/seed vector is nonnegative and sums to one.
- Two complete independent executions give maximum absolute prediction difference `0.0`; all PCA hashes, aggregate metrics, and diagnostics are identical.

## Automated checks

- Prospective unit tests: 6 passed.
- Full K=3 and pooled K=2 paths exercised.
- Exact sign test, support contrast, collision label-permutation endpoint, input schema, outer leakage, and simplex checks exercised.
- Generic synthetic suite: 72,000 records, 360 settings, 200 repeats per setting; hashes match.

The dry-run output contains historical targets solely for development validation and is excluded from the custodian archive and anonymous submission package.
