"""P6: Audit detection-benchmark (derived analysis; reads frozen P4/P5 artifacts).

A1 of the 2026-08-14 novelty plan: quantify each audit layer's detection power
against the mechanisms planted in the frozen P4/P5 simulator, using ONLY the
frozen JSON records plus deterministically regenerated targets (same seeds,
same generator) for the exact permutation null.

Detectors (operational versions of the paper's tools):
  S1  support threshold: raw pair separation < tau AND pair target
      disagreement > theta  (sweep tau, theta)
  S2  support permutation null (exact, conditional on S1): fraction of all
      C(n,2) pair disagreements >= observed pair disagreement; flag p < 0.05
  R   resolution: fine-contrast MAE > margin * shared-contrast MAE
      (margin sqrt(3) <=> q2 > q1 in the orthonormal basis)
  T   selection: nested - naive > tau_sel (P4 candidate count m; P5 gap ladder)

Outputs (all new, derived):
  results/p6_audit_detection_benchmark.json
  results/p6_audit_detection_benchmark_report.md
  paper/figures_v2/fig4_synthetic_framework.{pdf,png}  (1x4 with new panel d)
  results/p4_support_resolution_selection_manifest.json (metadata amendment:
      figure hashes + note; frozen JSON/CSV unchanged)

Frozen inputs are never modified: p4_support_resolution_selection.json,
p4_support_resolution_selection_by_setting.csv, p5_candidate_heterogeneity.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

import p4_support_resolution_selection as P4

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE.parent.parent / "paper" / "figures_v2"
P4_JSON = RESULTS / "p4_support_resolution_selection.json"
P5_JSON = RESULTS / "p5_candidate_heterogeneity.json"
MANIFEST = RESULTS / "p4_support_resolution_selection_manifest.json"
OUT_JSON = RESULTS / "p6_audit_detection_benchmark.json"
OUT_MD = RESULTS / "p6_audit_detection_benchmark_report.md"

ALPHA = 0.05
TAU_GRID = [0.001, 0.01, 0.05, 0.10, 0.25]
THETA_GRID = [0.05, 0.10, 0.20, 0.40]
MARGIN_GRID = [math.sqrt(3.0), 1.5, 2.0, 3.0]
TAU_SEL_GRID = [0.0, 0.01, 0.02, 0.05]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regenerate_targets(n, epsilon, delta, repeat):
    """Deterministic regeneration with the frozen P4 seed scheme."""
    seed = P4.BASE_SEED + n * 10_000_000 + int(round(epsilon * 100)) * 100_000 \
        + int(round(delta * 10)) * 1_000 + repeat
    rng = np.random.default_rng(seed)
    features, targets = P4.make_dataset(n, epsilon, delta, rng)
    return features, targets, seed


def cross_check(records):
    """Verify regenerated targets/features against frozen per-repeat fields."""
    checked, errors = 0, []
    for r in records:
        if r["resolution"] != "full_k3" or r["m"] != 1:
            continue
        feats, tgt, seed = regenerate_targets(
            r["n"], r["epsilon"], r["delta"], r["repeat"])
        assert seed == r["seed"], (seed, r["seed"])
        observed = 2.0 * r["equal_prediction_pair_floor"]
        regenerated = float(np.abs(tgt[0] - tgt[1]).mean())
        if abs(observed - regenerated) > 1e-9:
            errors.append((r["n"], r["epsilon"], r["delta"], r["repeat"],
                           observed, regenerated))
        coll = bool(abs(feats[0, 0] - feats[1, 0]) < 1e-12)
        if coll != r["exact_collision"]:
            errors.append(("collision_mismatch", r["n"], r["epsilon"],
                           r["delta"], r["repeat"], coll, r["exact_collision"]))
        checked += 1
    return checked, errors


def pair_null_p(tgts):
    """Exact permutation null p for the pair statistic: fraction of all C(n,2)
    pairwise per-state disagreements that reach the observed pair's value."""
    n = tgts.shape[0]
    obs = float(np.abs(tgts[0] - tgts[1]).mean())
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(float(np.abs(tgts[i] - tgts[j]).mean()))
    diffs = np.asarray(diffs)
    return float(np.mean(diffs >= obs - 1e-12)), obs, float(diffs.mean())


def detection_summary(rows, field, mechanism):
    """Detection rate + SE over repeats for one setting bucket."""
    flags = np.asarray([float(r[field]) for r in rows], dtype=float)
    n = len(flags)
    mean = flags.mean()
    se = flags.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
    return {"mechanism": mechanism, "repeats": n, "detection_rate": round(mean, 6),
            "se": round(se, 6)}


def main() -> int:
    p4 = json.loads(P4_JSON.read_text(encoding="utf-8"))
    p5 = json.loads(P5_JSON.read_text(encoding="utf-8"))
    records = p4["records"]

    # ---- deterministic cross-check of regeneration ----
    checked, errors = cross_check(records)
    print(f"cross-check: {checked} repeats verified, {len(errors)} mismatches")
    if errors:
        for e in errors[:5]:
            print("  mismatch:", e)
        return 1

    # dedupe per (n, epsilon, delta, repeat) at full_k3/m=1 for support/resolution
    support_rows = {}
    for r in records:
        if r["resolution"] == "full_k3" and r["m"] == 1:
            support_rows[(r["n"], r["epsilon"], r["delta"], r["repeat"])] = r
    key2tgt = {}
    for key, r in support_rows.items():
        _, tgt, _ = regenerate_targets(key[0], key[1], key[2], key[3])
        key2tgt[key] = tgt

    def buckets(condition):
        out = {}
        for key, r in support_rows.items():
            if condition(r):
                out.setdefault(key[:3], []).append(r)
        return out

    out = {
        "experiment": "P6 audit detection benchmark (A1)",
        "date": "2026-08-14",
        "inputs": [P4_JSON.name, P5_JSON.name],
        "cross_check": {"repeats_verified": checked, "mismatches": len(errors)},
        "detectors": {
            "S1": "raw_pair_separation < tau and pair disagreement > theta",
            "S2": "exact C(n,2) permutation p < 0.05, conditional on S1(tau,theta)",
            "R": "fine_contrast_clr_mae > margin * shared_contrast_clr_mae",
            "T": "selection_optimism > tau_sel",
        },
        "sweeps": {"tau": TAU_GRID, "theta": THETA_GRID,
                   "margin": MARGIN_GRID, "tau_sel": TAU_SEL_GRID},
    }

    # ---- S1: support threshold detection ----
    s1 = []
    for n, eps, delta in sorted({k[:3] for k in support_rows}):
        rows = buckets(lambda r: r["n"] == n and r["epsilon"] == eps
                       and r["delta"] == delta)[(n, eps, delta)]
        for tau in TAU_GRID:
            for theta in THETA_GRID:
                flags = []
                for r in rows:
                    tgt = key2tgt[(n, eps, delta, r["repeat"])]
                    disagree = float(np.abs(tgt[0] - tgt[1]).mean())
                    flags.append(r["raw_pair_separation"] < tau
                                 and disagree > theta)
                rate = float(np.mean(flags))
                if (n, eps, delta) in ((20, 0.0, 1.2), (20, 1.0, 1.2),
                                       (6, 0.0, 1.2), (8, 0.0, 1.2)):
                    s1.append({"n": n, "epsilon": eps, "delta": delta,
                               "tau": tau, "theta": theta,
                               "detection_rate": round(rate, 6)})
    out["S1"] = s1

    # ---- S2: permutation-null power, conditional on S1 flag ----
    # headline: tau=0.05, theta=0.10; power vs (n, eps, delta)
    TAU_REF, THETA_REF = 0.05, 0.10
    s2 = []
    for n in P4.N_GRID:
        for eps in P4.EPS_GRID:
            for delta in P4.DELTA_GRID:
                rows = buckets(lambda r: r["n"] == n and r["epsilon"] == eps
                               and r["delta"] == delta)[(n, eps, delta)]
                flagged, powered, pvals = 0, 0, []
                for r in rows:
                    tgt = key2tgt[(n, eps, delta, r["repeat"])]
                    disagree = float(np.abs(tgt[0] - tgt[1]).mean())
                    if not (r["raw_pair_separation"] < TAU_REF
                            and disagree > THETA_REF):
                        continue
                    flagged += 1
                    pval, obs, _ = pair_null_p(tgt)
                    pvals.append(pval)
                    if pval < ALPHA:
                        powered += 1
                rate = (powered / flagged) if flagged else None
                s2.append({"n": n, "epsilon": eps, "delta": delta,
                           "tau": TAU_REF, "theta": THETA_REF,
                           "n_flagged": flagged,
                           "detection_rate": (round(rate, 6) if rate is not None
                                              else None),
                           "mean_p": round(float(np.mean(pvals)), 6)
                           if pvals else None})
    out["S2"] = s2

    # ---- R: resolution detection ----
    rdet = []
    for n in (6, 8, 20, 50):
        for eps in (0.0,):
            for delta in P4.DELTA_GRID:
                rows = buckets(lambda r: r["n"] == n and r["epsilon"] == eps
                               and r["delta"] == delta)[(n, eps, delta)]
                for margin in MARGIN_GRID:
                    flags = [r["fine_contrast_clr_mae"]
                             > margin * r["shared_contrast_clr_mae"]
                             for r in rows]
                    rdet.append({"n": n, "epsilon": eps, "delta": delta,
                                 "margin": margin,
                                 "detection_rate": round(float(np.mean(flags)), 6)})
    out["R"] = rdet

    # ---- T: selection detection (P4 m ladder; P5 gap ladder) ----
    tdet = []
    full = [r for r in records if r["resolution"] == "full_k3"]
    for n in P4.N_GRID:
        for m in P4.M_GRID:
            rows = [r for r in full if r["n"] == n and r["m"] == m]
            for tau_sel in TAU_SEL_GRID:
                flags = [r["selection_optimism"] > tau_sel for r in rows]
                tdet.append({"source": "P4", "n": n, "mechanism": m,
                             "tau_sel": tau_sel,
                             "detection_rate": round(float(np.mean(flags)), 6)})
    for gap in (0.15, 0.05, 0.02, 0.01):
        for n in (6, 8, 20):
            rows = [r for r in p5["records"] if r["n"] == n
                    and abs(r["gap_step"] - gap) < 1e-12]
            for tau_sel in TAU_SEL_GRID:
                flags = [r["selection_optimism"] > tau_sel for r in rows]
                tdet.append({"source": "P5", "n": n, "mechanism": gap,
                             "tau_sel": tau_sel,
                             "detection_rate": round(float(np.mean(flags)), 6)})
    out["T"] = tdet

    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT_JSON)

    # ---- figure panel d data ----
    panel_d = {
        "R_detection_vs_delta": {
            "fixed": {"epsilon": 0.0, "margin": math.sqrt(3.0),
                      "note": "margin sqrt(3) <=> orthonormal q2 > q1"},
            "series": {str(n): [next(x["detection_rate"] for x in rdet
                                 if x["n"] == n and x["epsilon"] == 0.0
                                 and x["delta"] == d
                                 and abs(x["margin"] - math.sqrt(3.0)) < 1e-9)
                                for d in P4.DELTA_GRID]
                       for n in (6, 8, 20, 50)}},
        "S2_power_vs_delta": {
            "fixed": {"epsilon": 0.0, "tau": TAU_REF, "theta": THETA_REF,
                      "alpha": ALPHA},
            "series": {str(n): [next(x["detection_rate"] for x in s2
                                     if x["n"] == n and x["epsilon"] == 0.0
                                     and x["delta"] == d)
                                for d in P4.DELTA_GRID]
                       for n in P4.N_GRID}}}
    out["figure_panel_d"] = panel_d

    report = render_report(out)
    OUT_MD.write_text(report, encoding="utf-8")
    print("wrote", OUT_MD)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["1x4", "2x2"], default="1x4")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    render_fig4(panel_d, layout=args.layout, outdir=args.outdir)
    if args.outdir is None and args.layout == "1x4":
        amend_manifest()
    return 0


def render_report(out):
    lines = ["# P6 audit detection benchmark (A1) — derived from frozen P4/P5",
             "",
             f"- cross-check: {out['cross_check']['repeats_verified']} repeats "
             f"regenerated exactly; mismatches = {out['cross_check']['mismatches']}",
             "- S1 support threshold (tau, theta sweep):",
             ""]
    s1 = out["S1"]
    keep = [r for r in s1 if (r["n"], r["epsilon"], r["delta"])
            in {(20, 0.0, 1.2), (20, 1.0, 1.2), (6, 0.0, 1.2), (8, 0.0, 1.2)}
            and r["tau"] in (0.001, 0.05, 0.25) and r["theta"] in (0.05, 0.2)]
    for r in keep:
        lines.append(f"  n={r['n']} eps={r['epsilon']} delta={r['delta']} "
                     f"tau={r['tau']} theta={r['theta']}: "
                     f"rate={r['detection_rate']:.3f}")
    lines += ["", "- S2 permutation-null power (tau=0.05, theta=0.10, alpha=0.05):", ""]
    lines.append("  | n | eps | delta | flagged | power | mean p |")
    lines.append("  |---:|---:|---:|---:|---:|---:|")
    for r in out["S2"]:
        if r["epsilon"] in (0.0, 1.0) or (r["epsilon"] == 0.05 and r["n"] in (6, 8)):
            lines.append(f"  | {r['n']} | {r['epsilon']} | {r['delta']} | "
                         f"{r['n_flagged']} | {r['detection_rate']} | {r['mean_p']} |")
    lines += ["", "- R resolution detection (epsilon=0):", ""]
    for r in out["R"]:
        if r["margin"] in (1.7320508075688772, 1.5, 2.0, 3.0):
            lines.append(f"  n={r['n']} delta={r['delta']} margin={r['margin']:.2f}: "
                         f"rate={r['detection_rate']:.3f}")
    lines += ["", "- T selection detection (tau_sel >= 0.01; the tau_sel=0 rows",
              "  are float noise around zero and are excluded):", ""]
    for r in out["T"]:
        if r["tau_sel"] < 0.01:
            continue
        if (r["source"] == "P4" and r["n"] in (6, 8, 20)) or \
           (r["source"] == "P5" and r["n"] in (6, 8)):
            lines.append(f"  {r['source']} n={r['n']} m={r['mechanism']} "
                         f"tau_sel={r['tau_sel']}: rate={r['detection_rate']:.3f}")
    lines += ["",
              "Interpretation (calibration findings):",
              "- S2: the exact pair permutation null is structurally powerless ",
              "  at every n (power <= 0.09, mean p 0.28-0.51 even for the planted ",
              "  collision): a single pair competes against all C(n,2) pairs, so ",
              "  the paper's biological p=0.40/0.23 cannot certify absence of ",
              "  collision risk. Collision membership via the threshold detector ",
              "  (S1) is the working support tool (rate 1.0 at eps=0, 0.0 at eps=1).",
              "- R: q2>q1 (margin sqrt(3)) flags planted fine contrast with rate ",
              "  0.93-1.0 at delta>=0.6 for n=6/8 and zero false flags at delta=0, ",
              "  but collapses to 0.53 at n=20 and 0 at n=50: a panel-scale ",
              "  diagnostic, not an asymptotic one.",
              "- T: optimism > 0.01 fires in 37% (n=6) / 3% (n=8) of m=20 ",
              "  factorial repeats whose mean optimism is 0.016 -> the Abl1 ",
              "  +0.0297 gap is inside this noise band and Src +0.108 is outside; ",
              "  the P5 ladder flags 0.66-0.82 at tau_sel=0.02 only for clear ",
              "  quality gaps."]
    return "\n".join(lines) + "\n"


def render_fig4(panel_d, layout="1x4", outdir=None):
    """Render fig4. layout: '1x4' (frozen four-panel strip) or '2x2' (v3
    enlarged panels). outdir defaults to the paper figures_v2 directory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if outdir is None:
        outdir = FIGURES
    else:
        outdir = Path(outdir)

    summary = json.loads((RESULTS / "p4_support_resolution_selection.json")
                         .read_text(encoding="utf-8"))["summary"]
    if layout == "2x2":
        plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.7,
                             "pdf.fonttype": 42, "ps.fonttype": 42})
        fig, axes = plt.subplots(2, 2, figsize=(7.05, 3.3))
        axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
        title_kw = dict(fontsize=8.5, pad=4.0)
        leg = 6.0
    else:
        plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.7,
                             "pdf.fonttype": 42, "ps.fonttype": 42})
        fig, axes = plt.subplots(1, 4, figsize=(7.05, 2.18))
        axA, axB, axC, axD = axes[0], axes[1], axes[2], axes[3]
        title_kw = dict(fontsize=6.8, pad=3.0)
        leg = 5.2
    colors = {"clr": "#0072B2", "nn": "#D55E00", "pool": "#009E73",
              "fine": "#CC79A7", "floor": "#333333"}
    markers = {0.0: "o", 0.6: "s", 1.2: "^"}

    # A (frozen P4 data)
    ax = axA
    for delta in P4.DELTA_GRID:
        rows = [r for r in summary if r["n"] == 20 and r["m"] == 1
                and r["resolution"] == "full_k3" and r["delta"] == delta]
        rows.sort(key=lambda r: r["epsilon"])
        ax.plot([r["epsilon"] for r in rows],
                [r["pair_clr_ridge_mae_mean"] for r in rows],
                marker=markers[delta], lw=1.2, ms=3.2,
                label=fr"ridge, $\delta={delta:g}$")
    floor_row = [r for r in summary if r["n"] == 20 and r["m"] == 1
                 and r["resolution"] == "full_k3" and r["delta"] == 1.2
                 and r["epsilon"] == 0.0][0]
    ax.scatter([0.0], [floor_row["equal_prediction_pair_floor_mean"]],
               color=colors["floor"], marker="*", s=28, zorder=4,
               label=r"strict floor ($\epsilon=0$)")
    ax.set(xlabel=r"collision separation $\epsilon$", ylabel="collision-pair MAE",
           title="A  Collision error floor")
    ax.set_title("A  Collision error floor", **title_kw)
    ax.legend(frameon=False, fontsize=leg - 1.2, ncol=2, handlelength=1.4)

    # B
    ax = axB
    for resolution, field, label, color, marker in [
        ("full_k3", "shared_contrast_clr_mae_mean", "full K=3: shared", colors["clr"], "o"),
        ("pooled_k2", "shared_contrast_clr_mae_mean", "pooled K=2: shared", colors["pool"], "s"),
        ("full_k3", "fine_contrast_clr_mae_mean", "full K=3: fine", colors["fine"], "^")]:
        ys = []
        for delta in P4.DELTA_GRID:
            rows = [r for r in summary if r["n"] == 20 and r["m"] == 1
                    and r["resolution"] == resolution and r["delta"] == delta]
            ys.append(float(np.mean([r[field] for r in rows])))
        ax.plot(P4.DELTA_GRID, ys, color=color, marker=marker, lw=1.3, ms=3.3,
                label=label)
    ax.set(xlabel=r"hidden fine contrast $\delta$", ylabel="contrast MAE",
           title="B  Pooling hides fine error")
    ax.set_title("B  Pooling hides fine error", **title_kw)
    ax.legend(frameon=False, fontsize=leg - 0.6)

    # C
    ax = axC
    for resolution, label, color, marker in [
        ("full_k3", "full K=3", colors["clr"], "o"),
        ("pooled_k2", "pooled K=2", colors["pool"], "s")]:
        ys, ses = [], []
        for m in P4.M_GRID:
            vals = np.array([r["selection_optimism_mean"] for r in summary
                             if r["m"] == m and r["resolution"] == resolution], float)
            ys.append(vals.mean()); ses.append(vals.std(ddof=1) / np.sqrt(len(vals)))
        ax.errorbar(P4.M_GRID, ys, yerr=ses, color=color, marker=marker, lw=1.3,
                    ms=3.3, capsize=2, label=label)
    ax.axhline(0, color="#777777", lw=0.7)
    ax.set(xlabel="representation candidates m", ylabel="nested − naive MAE",
           title="C  Naive selection optimism", xticks=P4.M_GRID)
    ax.set_title("C  Naive selection optimism", **title_kw)
    ax.legend(frameon=False, fontsize=leg - 0.4)

    # D: contrast-order detector (R) vs planted fine contrast, by n
    ax = axD
    series = panel_d["R_detection_vs_delta"]["series"]
    dmarkers = {6: "o", 8: "s", 20: "D", 50: "v"}
    for n in (6, 8, 20, 50):
        ys = series[str(n)]
        ax.plot(P4.DELTA_GRID, ys, marker=dmarkers[n], lw=1.2, ms=3.0,
                label=f"n={n}")
    ax.set(xlabel=r"hidden fine contrast $\delta$",
           ylabel="detection rate",
           title=r"D  $q_2>q_1$ detector, by $n$",
           ylim=(-0.05, 1.05))
    ax.set_title(r"D  $q_2>q_1$ detector, by $n$", **title_kw)
    ax.legend(frameon=False, fontsize=leg - 0.4, ncol=2, handlelength=1.4)

    for ax in (axA, axB, axC, axD):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", lw=0.45)
    fig.tight_layout(w_pad=0.9, h_pad=1.1)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "fig4_synthetic_framework.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig4_synthetic_framework.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(f"rendered fig4 ({layout}) ->", outdir)


def amend_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["figure_pdf_sha256"] = sha256_file(
        FIGURES / "fig4_synthetic_framework.pdf")
    manifest["figure_png_sha256"] = sha256_file(
        FIGURES / "fig4_synthetic_framework.png")
    manifest["figure_amendment_20260814"] = (
        "panel d added by the P6 derived detection benchmark (A1); panels "
        "a/b/c and all JSON/CSV records unchanged")
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("amended manifest:", manifest["figure_pdf_sha256"][:16], "...")


if __name__ == "__main__":
    raise SystemExit(main())
