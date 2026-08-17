# Prospective validation v3

This directory is the dated v3 amendment. Files in the parent directory are the immutable v2 historical record.

## Roles

- Modeling side: freeze public input, prepare target-free ESM-2 features, deliver code/environment/hashes, and never access the private target file before reveal.
- Custodian: retain mutant populations, run once in isolation, seal all outputs, and release them exactly once.
- User: fill real sender identity and send outreach messages personally.

## Execution

```text
python prepare_esm2_features_v3.py --public public_input.json --output esm2_features.npz --manifest feature_manifest.json
python run_custodian_v3.py --public public_input.json --private private_targets.json --features esm2_features.npz --output-dir sealed_output
```

Use `python dry_run_v3.py` only for the Abl1 development rehearsal. Its outputs are permanently marked `DRY RUN / NOT EVIDENCE` and cannot enter the external-validation claim.

The main-panel gate is a new K=3 panel with at least eight mutants. A successful script exit does not imply scientific success; the joint gates are stored under `aggregate.success_gates`.
