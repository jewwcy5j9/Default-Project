#!/usr/bin/env python
"""Prepare label-free ESM-2 residue-difference features for protocol v3.

The public mutation file uses explicit one-based sequence indices.  This avoids
site markers and cross-system residue-number mappings.  For a multi-site mutant,
the 1280-dimensional intervention vector is the sum of mutant-minus-WT residue
deltas at every substituted site.  Full per-residue delta matrices are retained
so the custodian can fit PCA separately inside every outer fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

CHECKPOINT = "facebook/esm2_t33_650M_UR50D"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def embed(model, tokenizer, sequence, device):
    encoded = tokenizer(sequence, return_tensors="pt", truncation=False).to(device)
    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state[0, 1:-1]
    result = hidden.detach().cpu().numpy()
    if result.shape[0] != len(sequence):
        raise RuntimeError("token/residue length mismatch")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path)
    args = parser.parse_args()

    from transformers import EsmModel, EsmTokenizer
    public = json.loads(args.public.read_text(encoding="utf-8"))
    source = str(args.local_checkpoint) if args.local_checkpoint else CHECKPOINT
    tokenizer = EsmTokenizer.from_pretrained(source)
    model = EsmModel.from_pretrained(source)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    wt_sequence = public["wild_type_sequence"]
    wt_embedding = embed(model, tokenizer, wt_sequence, device)
    arrays = {}
    mutation_manifest = {}
    for mutation in public["mutations"]:
        mid = mutation["mutation_id"]
        mutant = list(wt_sequence)
        indices = []
        for substitution in mutation["substitutions"]:
            index = int(substitution["sequence_index_1based"]) - 1
            if mutant[index] != substitution["from"]:
                raise ValueError(f"{mid}: WT residue mismatch at sequence index {index + 1}")
            mutant[index] = substitution["to"]
            indices.append(index)
        mutant_embedding = embed(model, tokenizer, "".join(mutant), device)
        delta_rows = mutant_embedding - wt_embedding
        delta_vector = np.sum(delta_rows[indices], axis=0)
        arrays[f"delta_rows::{mid}"] = delta_rows.astype(np.float32)
        arrays[f"delta_vector::{mid}"] = delta_vector.astype(np.float64)
        mutation_manifest[mid] = {
            "sequence_indices_1based": [index + 1 for index in indices],
            "aggregation": "sum",
            "delta_rows_shape": list(delta_rows.shape),
            "delta_vector_shape": list(delta_vector.shape),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    manifest = {
        "protocol_version": "3.0.0", "checkpoint": CHECKPOINT,
        "checkpoint_source_used": source, "public_input_sha256": sha256_file(args.public),
        "feature_package_sha256": sha256_file(args.output),
        "contains_targets": False, "site_markers": False,
        "cross_system_position_mapping": False,
        # Feature provenance environment: GPU vs CPU regeneration changes
        # embeddings below hash-visibility, so record the runtime so the
        # custodian can detect heterogeneous feature provenance.
        "environment": {"device": torch.empty(1).device.type,
                        "torch": torch.__version__,
                        "numpy": np.__version__},
        "multi_site_rule": "sum of mutant-minus-WT residue embedding deltas at substituted sites",
        "mutations": mutation_manifest,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
