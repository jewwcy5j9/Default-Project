"""Phase-2 (SOTA plan route B, stage 2): frozen ESM-2 features on GPU.

Computes and caches, with a model/version/device manifest:
  1. masked-marginal LLR-only scalars per mutation (protocol identical to
     run_esm2_llr_proxy.py: 20 reps, mask_prob 0.15, seed = SEED + seq_idx,
     BOS offset +1, +/-10 position search, double-mutant = sum of positions).
  2. mutation-site embedding deltas: d_m = E_mut[pos] - E_wt[pos] (1280-dim,
     sum over sites for double mutants), using last hidden state (no masking).
  3. raw full-sequence embeddings cached to npz for later feature variants.

No PCA or projection is fit on the current labels here (external frozen
projection is a downstream step; this script only caches raw outputs).

Usage:
  python p2_gpu_features.py [--seq-max-abl1 287] [--reps 20]

Outputs (results/):
  p2_llr_features.json  LLR per mutation (abl1, src) + diff vs reference
  p2_site_deltas.npz    raw 1280-dim site deltas per (system, mutant)
  p2_embeddings.npz     full last-hidden-state per (system, mutant)
  p2_manifest.json      model sha256, versions, device, timing
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn.functional as F

from esm2_encoding import (
    ABL1_KD, SRC_FULL, ESM2_LOCAL_PATH, ESM2_MODEL_ID, find_position,
)
from encoding_ablation_control import ABL1_DATA, SRC_DATA

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REF_LLR_FILE = HERE.parent / "foldx_src" / "results" / "esm2_llr_proxy_results.json"

MASK_PROB = 0.15
BATCH_SIZE = 10
SEED = 0
AA_LIST = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
           "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

SYSTEMS = {
    "abl1": {"seq": ABL1_KD, "mutations": ABL1_DATA, "system": "abl1"},
    "src": {"seq": SRC_FULL, "mutations": SRC_DATA, "system": "src"},
}


def parse_mutations_from_name(name, data):
    clean = name.replace("SrcKD-", "")
    if "_" not in clean:
        return [(data["pos"], data["wt"], data["mut"])]
    parts = [p for p in clean.split("_") if p]
    out = []
    for p in parts:
        wt_aa, pos_s, mut_aa = p[0], p[1:-1], p[-1]
        out.append((int(pos_s), wt_aa, mut_aa))
    return out


def load_model(args):
    source = ESM2_LOCAL_PATH if Path(ESM2_LOCAL_PATH).exists() else ESM2_MODEL_ID
    print(f"Loading ESM-2 (MaskedLM) from: {source}", flush=True)
    from transformers import EsmForMaskedLM, EsmTokenizer

    t0 = time.time()
    tokenizer = EsmTokenizer.from_pretrained(source)
    model = EsmForMaskedLM.from_pretrained(source)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  loaded in {time.time() - t0:.1f}s | device: {device}", flush=True)
    return model, tokenizer, device, source


def model_sha256(source):
    p = Path(source) / "model.safetensors"
    if not p.exists():
        p = Path(source) / "pytorch_model.bin"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), p.name


def masked_marginal_logprob(model, tokenizer, wt_seq, pos_idx, n_reps, device):
    rng = np.random.default_rng(SEED)
    ids = tokenizer(wt_seq, return_tensors="pt", truncation=True,
                    max_length=1024).input_ids[0].to(device)
    tok_idx = pos_idx + 1
    seq_len = len(ids) - 2
    mask_id = tokenizer.mask_token_id

    masked = []
    for rep in range(n_reps):
        m = ids.clone()
        for j in range(1, seq_len + 1):
            if j != tok_idx and rng.random() < MASK_PROB:
                m[j] = mask_id
        m[tok_idx] = mask_id
        masked.append(m)

    sum_logp = {aa: 0.0 for aa in AA_LIST}
    for start in range(0, n_reps, BATCH_SIZE):
        batch = torch.stack(masked[start:start + BATCH_SIZE])
        with torch.no_grad():
            out = model(batch)
        lp = F.log_softmax(out.logits, dim=-1)[:, tok_idx, :]
        for aa in AA_LIST:
            tid = tokenizer.convert_tokens_to_ids(aa)
            sum_logp[aa] += float(lp[:, tid].sum().item())
    return {aa: v / n_reps for aa, v in sum_logp.items()}


def site_deltas(model, tokenizer, wt_seq, mut_list, system, device):
    """E_mut[pos] - E_wt[pos] per site (sum for double mutants); 1280-dim."""
    def embeds(seq):
        ids = tokenizer(seq, return_tensors="pt", truncation=True,
                        max_length=1024).input_ids.to(device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        return out.hidden_states[-1][0].cpu().numpy()  # (L+2, 1280)

    e_wt = embeds(wt_seq)
    delta = np.zeros_like(e_wt[0])
    positions = []
    for pos, wt_aa, mut_aa in mut_list:
        mut_seq = list(wt_seq)
        idx, offset = find_position(wt_seq, pos, wt_aa, system)
        mut_seq[idx] = mut_aa
        e_mut = embeds("".join(mut_seq))
        delta += e_mut[idx + 1] - e_wt[idx + 1]
        positions.append({"nominal_pos": pos, "seq_idx": idx, "offset": offset,
                          "wt": wt_aa, "mut": mut_aa})
    return delta, positions, e_wt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--skip-llr", action="store_true",
                    help="skip LLR (only embeddings/deltas)")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    model, tokenizer, device, source = load_model(args)
    sha, sha_file = model_sha256(source)

    ref = {}
    if REF_LLR_FILE.exists():
        ref = json.loads(REF_LLR_FILE.read_text(encoding="utf-8"))

    llr_out = {}
    deltas_out = {}
    embeds_out = {}
    tok_dir_out = {}
    manifest = {"model": {"id": ESM2_MODEL_ID, "source": source,
                          "sha256": sha, "weights_file": sha_file},
                "torch": torch.__version__, "cuda": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "reps": args.reps, "mask_prob": MASK_PROB, "seed": SEED,
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    # frozen model-internal token directions: emb[mut_aa] - emb[wt_aa]
    tok_emb = model.esm.embeddings.word_embeddings.weight.detach().cpu().numpy()

    for sysname, cfg in SYSTEMS.items():
        wt_seq = cfg["seq"]
        llr_out[sysname] = {}
        for name, data in cfg["mutations"].items():
            mut_list = parse_mutations_from_name(name, data)
            print(f"[{sysname}] {name}", flush=True)
            pos_info = []
            if not args.skip_llr:
                total = 0.0
                for pos, wt_aa, mut_aa in mut_list:
                    idx, offset = find_position(wt_seq, pos, wt_aa, cfg["system"])
                    lp = masked_marginal_logprob(model, tokenizer, wt_seq, idx,
                                                 args.reps, device)
                    llr_pos = float(lp[mut_aa] - lp[wt_aa])
                    total += llr_pos
                    pos_info.append({"nominal_pos": pos, "seq_idx": idx,
                                     "offset": offset, "wt": wt_aa, "mut": mut_aa,
                                     "llr_single": llr_pos})
                    print(f"    LLR({wt_aa}{pos}->{mut_aa}) = {llr_pos:+.4f}", flush=True)
                llr_out[sysname][name] = {"llr": float(total), "positions": pos_info}
            delta, positions, e_wt = site_deltas(model, tokenizer, wt_seq, mut_list,
                                                 cfg["system"], device)
            deltas_out[f"{sysname}::{name}"] = delta
            embeds_out[f"{sysname}::{name}::wt"] = e_wt
            dir_vec = np.zeros_like(delta)
            for pos, wt_aa, mut_aa in mut_list:
                tid_wt = tokenizer.convert_tokens_to_ids(wt_aa)
                tid_mut = tokenizer.convert_tokens_to_ids(mut_aa)
                dir_vec += tok_emb[tid_mut] - tok_emb[tid_wt]
            tok_dir = float(delta @ dir_vec)
            llr_out[sysname].setdefault(name, {})
            llr_out[sysname][name]["tok_dir"] = tok_dir
            llr_out[sysname][name]["tok_dir_unit"] = float(tok_dir / np.linalg.norm(dir_vec))
            print(f"    site delta: norm={float(np.linalg.norm(delta)):.4f} "
                  f"tok_dir={tok_dir:+.4f}", flush=True)

    # compare LLR vs reference (same weights -> should match closely)
    diff_report = {}
    if not args.skip_llr and ref:
        for sysname in SYSTEMS:
            ref_llr = (ref.get(sysname, {}).get("llr", {})) if isinstance(ref.get(sysname), dict) else {}
            if not ref_llr:
                continue
            for name, v in llr_out[sysname].items():
                if name in ref_llr:
                    diff_report[f"{sysname}/{name}"] = {
                        "ref": float(ref_llr[name]), "new": float(v["llr"]),
                        "abs_diff": float(abs(ref_llr[name] - v["llr"]))}
    max_diff = max([d["abs_diff"] for d in diff_report.values()], default=None)

    np.savez_compressed(RESULTS / "p2_site_deltas.npz", **deltas_out)
    np.savez_compressed(RESULTS / "p2_embeddings.npz", **embeds_out)
    (RESULTS / "p2_llr_features.json").write_text(
        json.dumps({"llr": llr_out, "vs_reference": diff_report,
                    "max_abs_diff_vs_reference": max_diff},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (RESULTS / "p2_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== summary ===")
    if not args.skip_llr:
        print(f"  max |LLR diff| vs reference: {max_diff}")
        for k, d in list(diff_report.items())[:5]:
            print(f"    {k}: ref {d['ref']:+.4f} new {d['new']:+.4f}")
    print(f"  site deltas: {RESULTS / 'p2_site_deltas.npz'}")
    print(f"  embeddings:  {RESULTS / 'p2_embeddings.npz'}")
    print(f"  manifest:    {RESULTS / 'p2_manifest.json'}")
    print(f"[OK] p2 features written")


if __name__ == "__main__":
    main()
