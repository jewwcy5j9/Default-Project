"""P2c: Feature-collision analysis (the core support diagnostic).

If a test mutation's feature vector is close to a training mutation's
feature vector (in the encoding space) while their true shift directions
differ, then any model that is Lipschitz in the features incurs an
irreducible error on at least one of the pair.

Norms (unified with the paper's convention):
  - d_feat = min over training points of ||c_test - c_train||, reported in
    both Euclidean (L2, the metric actually computed for feature distance)
    and L1. For the position-marker pairs quoted in the paper the two
    coincide (only the pos/seq coordinate differs).
  - d_dir  = ||d_test - d_train||_1 (L1 on the shift in probability space),
    matching the statement of Prop. 1; d_dir_l2 is kept for audit trails.
  - per-state MAE floor = (delta - L*eps) / (2*K), K = 3, for the
    canonical collision pairs, evaluated at L in {0, 5, 20}.

Measures, per test point (LOO):
  - d_feat (L1, L2), d_dir (L1) for the nearest training point
  - collision pairs table (d_feat_l2 < tau and d_dir > delta)

Output: results/p2c_feature_collision.json
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from k3_data import ABL1_K3, ABL1_K3_WT_POP, SRC_K3, SRC_K3_WT_POP
from alternative_encodings import DDG_DATA
from k3_llr_proxy import LLR

OUT = Path(__file__).resolve().parent / "results"
ABL1_CORE = {m: ABL1_K3[m] for m in ABL1_K3 if m not in ("WT", "H396P", "M290L_H396P")}
SRC_CORE = {m: SRC_K3[m] for m in SRC_K3 if m != "SrcKD-WT"}
K = 3


def enc_abl1_pos(name, data):
    return np.array([data["pos"] / 534.0, float(data["pos"] == 290),
                     float(data["pos"] == 301), float(data["pos"] == 382)])


def enc_abl1_ddg(name, data):
    return np.array([data["pos"] / 534.0, DDG_DATA.get(name, 0.0) / 3.5,
                     float(data["pos"] == 290), float(data["pos"] == 301),
                     float(data["pos"] == 382)])


def enc_abl1_llr(name, data):
    return np.array([data["pos"] / 534.0,
                     LLR.get(name, 0.0) / max(abs(v) for v in LLR.values()),
                     float(data["pos"] == 290), float(data["pos"] == 301),
                     float(data["pos"] == 382)])


def enc_src_pos(name, data):
    return np.array([data["pos"] / 536.0, float(data["pos"] == 311),
                     float(data["pos"] == 332), float(data["pos"] == 380)])


def analyze(mutations, wt_pop, enc_fn):
    names = list(mutations.keys())
    wt = np.array(wt_pop, dtype=float)
    out = {}
    for held in names:
        tr = [m for m in names if m != held]
        c_h = enc_fn(held, mutations[held])
        d_h = np.array(mutations[held]["pop"]) - wt
        rows = []
        for m in tr:
            c_m = enc_fn(m, mutations[m])
            d_m = np.array(mutations[m]["pop"]) - wt
            rows.append({
                "mut": m,
                "d_feat_l2": float(np.linalg.norm(c_h - c_m)),
                "d_feat_l1": float(np.linalg.norm(c_h - c_m, ord=1)),
                "d_dir_l1": float(np.linalg.norm(d_h - d_m, ord=1)),
                "d_dir_l2": float(np.linalg.norm(d_h - d_m)),
            })
        rows.sort(key=lambda r: r["d_feat_l2"])
        out[held] = {"nearest": rows[0]["mut"],
                     "d_feat_l2_min": rows[0]["d_feat_l2"],
                     "d_feat_l1_min": rows[0]["d_feat_l1"],
                     "d_dir_l1_at_min": rows[0]["d_dir_l1"],
                     "d_dir_l2_at_min": rows[0]["d_dir_l2"],
                     "top3": rows[:3]}
    return out


def collision_floor(delta, eps, K=3, lipschitz=(0.0, 5.0, 20.0)):
    """Per-state MAE floor (delta - L*eps)/(2*K) for a colliding pair."""
    return {"delta": delta, "eps": eps, "K": K,
            "floor_per_state_mae": {
                f"L={L}": float((delta - L * eps) / (2 * K)) for L in lipschitz}}


def main():
    print("=" * 90)
    print("P2c: feature-collision diagnostics (L1 direction, unified norms)")
    print("=" * 90)
    results = {
        "norms": "d_feat = ||c_i - c_j|| (L1/L2 on encoding); "
                 "d_dir = ||d_i - d_j||_1 (L1 on shift); "
                 "per-state MAE floor = (delta - L*eps)/(2K), K=3",
        "abl1_pos": analyze(ABL1_CORE, ABL1_K3_WT_POP, enc_abl1_pos),
        "abl1_ddg": analyze(ABL1_CORE, ABL1_K3_WT_POP, enc_abl1_ddg),
        "abl1_llr": analyze(ABL1_CORE, ABL1_K3_WT_POP, enc_abl1_llr),
        "src_pos": analyze(SRC_CORE, SRC_K3_WT_POP, enc_src_pos),
    }
    for key, d in results.items():
        if key == "norms":
            continue
        print(f"\n  {key}:")
        for m, v in d.items():
            print(f"      {m:<16} nearest={v['nearest']:<12} "
                  f"d_feat_l2={v['d_feat_l2_min']:.3f} "
                  f"d_dir_l1={v['d_dir_l1_at_min']:.3f}")

    # collision pairs: feature-close but direction-distant (both > threshold)
    summary = {}
    for key, d in results.items():
        if key == "norms":
            continue
        n_coll = 0
        for m, v in d.items():
            if v["d_feat_l2_min"] < 0.25 and v["d_dir_l1_at_min"] > 0.6:
                n_coll += 1
        summary[key] = {"collision_count": n_coll, "n": len(d)}
        print(f"  {key}: collision pairs (d_feat_l2<0.25, d_dir_l1>0.6): "
              f"{n_coll}/{len(d)}")

    # per-state MAE floors for the canonical pairs quoted in the paper
    floors = {
        "src_L410A_F405A": collision_floor(
            float(np.linalg.norm(
                np.array(SRC_CORE["SrcKD-L410A"]["pop"]) -
                np.array(SRC_CORE["SrcKD-F405A"]["pop"]), ord=1)),
            0.0093),
        "src_L325A_V331A": collision_floor(
            float(np.linalg.norm(
                np.array(SRC_CORE["SrcKD-L325A"]["pop"]) -
                np.array(SRC_CORE["SrcKD-V331A"]["pop"]), ord=1)),
            0.0112),
        "abl1_F382L_F382Y": collision_floor(
            float(np.linalg.norm(
                np.array(ABL1_CORE["F382L"]["pop"]) -
                np.array(ABL1_CORE["F382Y"]["pop"]), ord=1)),
            0.0),
    }
    results["floors"] = floors
    for name, f in floors.items():
        print(f"\n  floor[{name}]: delta={f['delta']:.3f} eps={f['eps']:.4f}")
        for L, fl in f["floor_per_state_mae"].items():
            print(f"      {L}: {fl:.4f}")

    (OUT / "p2c_feature_collision.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[OK] p2c_feature_collision.json written")


if __name__ == "__main__":
    main()
