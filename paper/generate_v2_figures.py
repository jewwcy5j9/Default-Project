"""Generate publication figures for the support-aware paper rewrite.

The figures follow the local scientific-figure-making skill, load plotted
values from authoritative project artifacts, and export PDF/SVG/PNG versions.

Run from any directory:
    python paper/generate_v2_figures.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
FIG_DIR = PAPER_DIR / "figures_v2"
RESULTS = ROOT / "experiments" / "iclr_restructuring" / "results"
AF2_RESULTS = ROOT / "experiments" / "af2_subsample" / "results"
AF2_DIR = ROOT / "experiments" / "af2_subsample"

PALETTE = {
    # seaborn-deep/NPG-inspired palette.  Every color role is also paired
    # with a marker, line style, outline, or hatch so the figures remain
    # legible in grayscale and for readers with color-vision deficiencies.
    "blue_main": "#5B7CB3",
    "blue_secondary": "#93A4CB",
    "green_1": "#E6F3EF",
    "green_2": "#86C8B7",
    "green_3": "#24A792",
    "red_1": "#F6E3E5",
    "red_2": "#DFA8AD",
    "red_strong": "#C46165",
    "neutral": "#B0B0B0",
    "highlight": "#E8A243",
    "teal": "#24A792",
    "violet": "#887CB6",
}
INK = "#222222"
MID_GRAY = "#6F6F6F"
LIGHT_GRAY = "#E8E8E8"

FIGURE_SIZES = {
    "fig1_workflow": (6.75, 2.20),
    "fig2_support_map": (6.75, 3.05),
    "fig3_resolution": (6.75, 2.60),
    "fig4_synthetic_framework": (6.75, 2.42),
    "fig4_evidence": (6.75, 2.70),
    "fig5_alignment": (6.75, 2.55),
}

SOURCE_PATHS = {
    "collisions": RESULTS / "p2c_feature_collision.json",
    "review": RESULTS / "t5_review_responses.json",
    "review_k2": RESULTS / "t5b_review_responses.json",
    "collision_null": RESULTS / "t6_collision_null.json",
    "baselines": RESULTS / "p1_core_baselines.json",
    "benchmark": RESULTS / "k3_benchmark_results.json",
    "esm": RESULTS / "t7_fold_local_esm_pca_v2.json",
    "constant": RESULTS / "constant_baselines_core.json",
    "llr": RESULTS / "k3_llr_proxy_paired.json",
    "nested": RESULTS / "t1_nested_cv.json",
    "nested_full": RESULTS / "p2_k3_nested_pca_results.json",
    "src_label_sensitivity": RESULTS / "p2_k3_src_label_sensitivity.json",
    "grouped": RESULTS / "p0_grouped_cv.json",
    "energy": RESULTS / "p0_ddg_provenance.json",
    "af2_original": AF2_RESULTS / "t8_af2_region_sensitivity.json",
    "af2_verify": AF2_RESULTS / "t9_reclassify_verify.json",
    "af2_plddt": AF2_RESULTS / "af2_plddt_raw_pdb.json",
    "synthetic_summary": RESULTS / "p4_support_resolution_selection_by_setting.csv",
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.4,
            "axes.labelsize": 7.4,
            "axes.titlesize": 8.0,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.72,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.fontsize": 6.4,
            "legend.frameon": False,
            "legend.handlelength": 1.5,
            "lines.linewidth": 1.15,
            "lines.markersize": 3.6,
            "hatch.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def load_sources() -> dict[str, object]:
    data: dict[str, object] = {}
    for name, path in SOURCE_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing figure source: {path}")
        if path.suffix == ".json":
            data[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            data[name] = path.read_text(encoding="utf-8")
    return data


def assert_close(actual: float, expected: float, label: str, tol: float = 5e-6) -> None:
    if not np.isclose(actual, expected, atol=tol, rtol=0.0):
        raise ValueError(f"{label}: expected {expected}, found {actual}")


def validate_sources(data: dict[str, object]) -> None:
    collisions = data["collisions"]
    review = data["review"]
    null = data["collision_null"]
    esm = data["esm"]
    constant = data["constant"]
    llr = data["llr"]
    nested = data["nested"]
    nested_full = data["nested_full"]
    src_sensitivity = data["src_label_sensitivity"]
    grouped = data["grouped"]
    energy = data["energy"]
    af2_original = data["af2_original"]
    af2_verify = data["af2_verify"]
    af2_plddt = data["af2_plddt"]

    assert_close(
        collisions["src_pos"]["SrcKD-L410A"]["d_dir_l1_at_min"],
        review["collision_separation"]["delta_l1"],
        "Src collision target separation",
    )
    assert len(null["loo_error_null_src_pos"]["collision_members"]) == 4
    assert_close(esm["abl1"]["k3_pca_20dim"]["mae"], 0.1477492904, "Abl1 K=3 ESM")
    assert_close(esm["src"]["k3_pca_20dim"]["mae"], 0.3005821130, "Src K=3 ESM")
    assert_close(constant["abl1_k3_n6"]["training_mean_LOO"]["mean_mae"], 0.2328888889, "Abl1 mean")
    assert_close(llr["llr_proxy"]["mae"], 0.1628576145, "Abl1 LLR")
    assert_close(nested["nested_mae"], 0.2624288582, "Abl1 nested selector")
    assert_close(
        nested_full["systems"]["abl1"]["nested_mlp"]["mae"],
        0.2625464544,
        "Abl1 full nested MLP",
    )
    assert_close(
        nested_full["systems"]["src"]["nested_mlp"]["mae"],
        0.3990474203,
        "Src full nested MLP",
    )
    assert_close(
        src_sensitivity["systems"]["l410a_global_fit_substitution"]["training_mean"]["mae"],
        0.3185714286,
        "Src L410A-substitution training mean",
    )
    current_mlp = src_sensitivity["systems"]["primary_probe"]["fixed_k3"]["pos::LowRankCDST"]
    assert_close(current_mlp["u1_u2_contrast"]["u1"], 0.5206792055, "Src current MLP u1")
    assert_close(current_mlp["u1_u2_contrast"]["u2"], 0.6845696579, "Src current MLP u2")
    assert_close(grouped["pos_markers"]["290_301"]["group_mae"], 0.1426748757, "Grouped CV")
    assert_close(energy["summary"]["roundtrip_nonactive_mae"], 0.0447180932, "Energy round trip")
    assert af2_original["region_sensitivity"]["full_protein_i1i2_hits"] == 0
    assert af2_verify["full_protein_b1_i1i2"] == 0
    assert af2_plddt["schema_version"] == "af2_plddt_raw_pdb_v1"
    assert af2_plddt["protocols"]["original"]["found_structures"] == 840
    assert af2_plddt["protocols"]["fresh_msa"]["found_structures"] == 480


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    verify_text_layout(fig, stem)
    verify_text_within_axes(fig, stem)
    paths = []
    for extension in ("pdf", "svg", "png"):
        path = FIG_DIR / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        paths.append(path)
    plt.close(fig)
    verify_png(paths[-1])
    return paths


def verify_text_layout(fig: plt.Figure, stem: str, tolerance_px: float = 1.0) -> None:
    """Reject overlapping visible text before export."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [
        artist
        for artist in fig.findobj(Text)
        if artist.get_visible() and artist.get_text().strip()
    ]
    overlaps = []
    for index, first in enumerate(texts):
        first_box = Text.get_window_extent(first, renderer)
        for second in texts[index + 1 :]:
            second_box = Text.get_window_extent(second, renderer)
            overlap_width = min(first_box.x1, second_box.x1) - max(first_box.x0, second_box.x0)
            overlap_height = min(first_box.y1, second_box.y1) - max(first_box.y0, second_box.y0)
            if overlap_width > tolerance_px and overlap_height > tolerance_px:
                overlaps.append((first.get_text(), second.get_text()))
    if overlaps:
        detail = "; ".join(f"{left!r} <-> {right!r}" for left, right in overlaps[:8])
        raise ValueError(f"Text overlap in {stem}: {detail}")


def verify_text_within_axes(fig: plt.Figure, stem: str, tolerance_px: float = 6.0) -> None:
    """Reject value labels and annotations that spill outside their axes.

    Complements ``verify_text_layout`` (which only checks text-vs-text) by
    catching text that overflows the axes it belongs to and would collide with
    an adjacent panel, the colorbar, or the canvas margin.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems = []
    for ax in fig.axes:
        ax_box = ax.get_window_extent(renderer)
        for artist in ax.texts:
            if not (artist.get_visible() and artist.get_text().strip()):
                continue
            text_box = artist.get_window_extent(renderer)
            overflow = (
                max(ax_box.x0 - text_box.x0, 0.0)
                + max(text_box.x1 - ax_box.x1, 0.0)
                + max(ax_box.y0 - text_box.y0, 0.0)
                + max(text_box.y1 - ax_box.y1, 0.0)
            )
            if overflow > tolerance_px:
                problems.append((artist.get_text().replace("\n", " ")[:30], round(overflow, 1)))
    if problems:
        detail = "; ".join(f"{text!r} overflow={px}px" for text, px in problems[:8])
        raise ValueError(f"Text spills outside axes in {stem}: {detail}")


def verify_png(path: Path, min_margin_px: int = 5) -> None:
    image = Image.open(path).convert("RGB")
    background = Image.new("RGB", image.size, "white")
    content_box = ImageChops.difference(image, background).getbbox()
    if content_box is None:
        raise ValueError(f"Blank figure output: {path}")
    margins = (
        content_box[0],
        content_box[1],
        image.width - content_box[2],
        image.height - content_box[3],
    )
    if min(margins) < min_margin_px:
        raise ValueError(f"Figure content touches canvas edge in {path}: margins={margins}")


def title(ax: plt.Axes, panel: str, text: str) -> None:
    ax.set_title(rf"$\bf{{({panel})}}$ {text}", loc="left", pad=5.0)


def parse_synthetic_summary(csv_text: str) -> list[dict[str, object]]:
    """Load the frozen P4 setting summary without reading the 48 MB raw JSON."""
    integer_fields = {"n", "m", "repeats"}
    float_fields = {"epsilon", "delta"}
    rows: list[dict[str, object]] = []
    for source in csv.DictReader(io.StringIO(csv_text)):
        row: dict[str, object] = dict(source)
        for field in integer_fields:
            row[field] = int(source[field])
        for field in float_fields:
            row[field] = float(source[field])
        for field, value in source.items():
            if field.endswith("_mean") or field.endswith("_se"):
                row[field] = float(value) if value else np.nan
        rows.append(row)
    if len(rows) != 360 or not all(row["repeats"] == 200 for row in rows):
        raise ValueError("Synthetic summary is not the frozen 360-setting/200-repeat suite")
    return rows


def clean_mutant(name: str) -> str:
    return name.replace("SrcKD-", "").replace("_", "+")


def barycentric(population: tuple[float, float, float]) -> np.ndarray:
    active = np.array([0.50, 0.86])
    e1 = np.array([0.08, 0.08])
    e2 = np.array([0.92, 0.08])
    return population[0] * active + population[1] * e1 + population[2] * e2


def figure1_workflow() -> plt.Figure:
    fig = plt.figure(figsize=FIGURE_SIZES["fig1_workflow"])
    gs = fig.add_gridspec(1, 4, left=0.020, right=0.990, bottom=0.10, top=0.88, wspace=0.44)

    ax = fig.add_subplot(gs[0, 0])
    title(ax, "a", "Population shift")
    triangle = np.array([[0.50, 0.86], [0.08, 0.08], [0.92, 0.08]])
    ax.add_patch(patches.Polygon(triangle, facecolor=LIGHT_GRAY, edgecolor=INK, lw=0.8))
    wt = barycentric((0.72, 0.07, 0.21))
    mut1 = barycentric((0.73, 0.27, 0.00))
    mut2 = barycentric((0.00, 0.16, 0.84))
    for target, color in ((mut1, PALETTE["blue_main"]), (mut2, PALETTE["red_strong"])):
        ax.annotate("", xy=target, xytext=wt, arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.15})
        ax.scatter(*target, s=24, color=color, edgecolor="white", lw=0.4, zorder=3)
    ax.scatter(*wt, s=29, color=INK, edgecolor="white", lw=0.4, zorder=4)
    ax.text(wt[0], wt[1] + 0.070, "WT", ha="center", fontsize=6.8, weight="bold")
    ax.text(0.50, 0.92, "Active", ha="center", va="bottom", fontsize=6.6)
    ax.text(0.03, 0.04, "E1", ha="left", va="top", fontsize=6.6)
    ax.text(0.97, 0.04, "E2", ha="right", va="top", fontsize=6.6)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    title(ax, "b", "Feature support")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    for center, label, conflict in ((0.27, "supported", False), (0.76, "conflict", True)):
        ax.add_patch(patches.Circle((center, 0.52), 0.20, facecolor=LIGHT_GRAY, edgecolor=MID_GRAY, lw=0.65, ls="--"))
        points = [(center - 0.08, 0.48), (center + 0.02, 0.57), (center + 0.09, 0.46)]
        for i, point in enumerate(points):
            color = PALETTE["blue_main"] if not conflict or i == 0 else PALETTE["red_strong"]
            ax.scatter(*point, s=23, color=color, edgecolor="white", lw=0.35, zorder=3)
            direction = (0.00, 0.14) if not conflict or i != 2 else (0.00, -0.14)
            ax.annotate("", xy=(point[0] + direction[0], point[1] + direction[1]), xytext=point,
                        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.8})
        ax.text(center, 0.19, label, ha="center", fontsize=6.5, color=INK)
    ax.text(0.52, 0.89, r"near in $c$  $\nRightarrow$  near in $d$", ha="center", fontsize=6.4)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    title(ax, "c", "State resolution")
    ax.set(xlim=(-1.1, 1.1), ylim=(-1.05, 1.1))
    ax.axhline(0, color=MID_GRAY, lw=0.6)
    ax.axvline(0, color=MID_GRAY, lw=0.6)
    ax.annotate("", xy=(0.92, 0), xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": PALETTE["blue_main"]})
    ax.annotate("", xy=(0, 0.90), xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": PALETTE["red_strong"]})
    ax.text(0.48, -0.18, "shared", ha="center", fontsize=6.4, color=PALETTE["blue_main"])
    ax.text(0.08, 0.48, "fine", va="center", fontsize=6.4, color=PALETTE["red_strong"], rotation=90)
    ax.text(0, -0.58, r"$u_1=A-(E1+E2)$", ha="center", fontsize=6.8)
    ax.text(0, -0.82, r"$u_2=E1-E2$", ha="center", fontsize=6.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax = fig.add_subplot(gs[0, 3])
    title(ax, "d", "Evaluation splits")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    rows = [(0.77, "LOO"), (0.49, "leave-site-out"), (0.21, "nested selection")]
    for y, label in rows:
        ax.text(0.03, y, label, va="center", fontsize=6.3)
        if label == "LOO":
            for x in np.linspace(0.55, 0.93, 5):
                ax.scatter(x, y, s=19, color=PALETTE["blue_secondary"], edgecolor="white", lw=0.3)
            ax.scatter(0.84, y, marker="x", s=27, color=PALETTE["red_strong"], lw=1.1, zorder=4)
        elif label == "leave-site-out":
            for x in (0.58, 0.65, 0.84, 0.91):
                color = PALETTE["blue_secondary"] if x < 0.75 else PALETTE["red_strong"]
                ax.scatter(x, y, s=19, color=color, edgecolor="white", lw=0.3)
            ax.add_patch(patches.Ellipse((0.875, y), 0.16, 0.16, fill=False, edgecolor=PALETTE["red_strong"], lw=0.8, ls="--"))
        else:
            ax.add_patch(patches.Rectangle((0.54, y - 0.10), 0.39, 0.20, facecolor=LIGHT_GRAY, edgecolor=MID_GRAY, lw=0.7))
            ax.add_patch(patches.Rectangle((0.61, y - 0.065), 0.22, 0.13, facecolor="white", edgecolor=PALETTE["blue_main"], lw=0.8))
            ax.text(0.72, y, "select", ha="center", va="center", fontsize=6.0)
    ax.axis("off")
    return fig


def figure2_support(data: dict[str, object]) -> plt.Figure:
    collisions = data["collisions"]
    review = data["review"]
    null = data["collision_null"]
    baselines = data["baselines"]
    benchmark = data["benchmark"]
    esm = data["esm"]

    fig = plt.figure(figsize=FIGURE_SIZES["fig2_support_map"])
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.62, 1.08, 2.34],
        left=0.070, right=0.938, bottom=0.235, top=0.855, wspace=0.70,
    )

    ax = fig.add_subplot(gs[0, 0])
    title(ax, "a", "Nearest-neighbor support")
    ax.fill_betweenx([0.6, 1.9], 0, 0.25, color=PALETTE["red_1"], alpha=0.62, zorder=0)
    ax.axvline(0.25, color=PALETTE["red_strong"], lw=0.7, ls="--")
    ax.axhline(0.6, color=PALETTE["red_strong"], lw=0.7, ls="--")
    ax.text(0.23, 0.955, "conflict\nregion", transform=ax.transAxes,
            color=PALETTE["red_strong"], ha="left", va="top",
            fontsize=6.1, linespacing=0.95)
    systems = [("abl1_pos", "Abl1", "o", PALETTE["blue_main"]), ("src_pos", "Src", "s", PALETTE["red_strong"])]
    for key, label, marker, color in systems:
        for record in collisions[key].values():
            x = record["d_feat_l2_min"]
            y = record["d_dir_l1_at_min"]
            conflict = x < 0.25 and y > 0.6
            ax.scatter(x, y, s=25, marker=marker, facecolor=color if conflict else "white", edgecolor=color, lw=0.8, zorder=3)
        ax.scatter([], [], s=22, marker=marker, facecolor="white", edgecolor=color, lw=0.8, label=label)
    annotation_style = {
        "textcoords": "axes fraction", "ha": "left", "va": "center", "fontsize": 6.2,
        "bbox": {"facecolor": "white", "edgecolor": "none", "pad": 0.35, "alpha": 0.94},
        "arrowprops": {
            "arrowstyle": "-", "lw": 0.55, "color": MID_GRAY,
            "connectionstyle": "angle3,angleA=0,angleB=90",
        },
    }
    # Fixed annotation lanes keep the exact/near-collision labels away from
    # one another and from the dense cluster at x ~= 0.
    ax.annotate("F382 exact\n(3 endpoints)", xy=(0.0, 1.76), xytext=(0.56, 0.93), **annotation_style)
    ax.annotate("L410A / F405A", xy=(0.00933, 1.68), xytext=(0.56, 0.76), **annotation_style)
    ax.annotate("L325A / V331A", xy=(0.01119, 1.10), xytext=(0.56, 0.54), **annotation_style)
    ax.set_xscale("symlog", linthresh=0.02, linscale=0.75, base=10)
    ax.set_xlim(-0.002, 1.65)
    ax.set_ylim(0, 1.92)
    ax.set_xticks([0, 0.01, 0.1, 0.25, 1.0])
    ax.set_xticklabels(["0", "0.01", "0.1", "0.25", "1"])
    ax.set_xlabel(r"Nearest feature distance, $d_{feat}$")
    ax.set_ylabel(r"Shift distance, $d_{shift}^{L1}$")
    ax.legend(loc="lower right", handletextpad=0.3, borderaxespad=0.2, labelspacing=0.25)
    ax.grid(axis="y", color=LIGHT_GRAY, lw=0.45)

    ax = fig.add_subplot(gs[0, 1])
    title(ax, "b", "In-sample separation")
    separation = review["collision_separation"]
    model_data = separation["models"]
    labels = ["Target", "CLR-Ridge", "CLR-GP", "1-NN", "MLP"]
    values = [
        separation["delta_l1"],
        model_data["CLR-Ridge"]["output_separation_l1"],
        model_data["CLR-GP"]["output_separation_l1"],
        model_data["kNN(1)"]["output_separation_l1"],
        model_data["MLP(2 seeds)"]["output_separation_l1"],
    ]
    colors = ["white", PALETTE["neutral"], PALETTE["blue_main"], PALETTE["highlight"], PALETTE["violet"]]
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, values, color=colors, edgecolor=MID_GRAY, lw=0.5, height=0.62)
    bars[0].set_hatch("///")
    seeds = model_data["MLP(2 seeds)"]["seeds"]
    ax.errorbar(values[-1], y[-1], xerr=[[values[-1] - min(seeds)], [max(seeds) - values[-1]]], fmt="none", color=INK, lw=0.8, capsize=2)
    ax.axvline(separation["delta_l1"], color=INK, ls=":", lw=0.7)
    for yi, value in zip(y[:-1], values[:-1]):
        ax.text(max(value + 0.035, 0.055), yi, f"{value:.3f}", va="center", fontsize=6.0)
    # MLP (bottom) label sits clear of its seed-error bar.
    ax.text(max(seeds) + 0.04, y[-1], f"{values[-1]:.3f}", va="center", fontsize=6.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 2.12)
    ax.set_xlabel(r"$L_1$ output separation")
    ax.grid(axis="x", color=LIGHT_GRAY, lw=0.5)

    ax = fig.add_subplot(gs[0, 2])
    stats = null["loo_error_null_src_pos"]
    title(ax, "c", f"LOO MAE (conflict gap {stats['observed_diff']:.3f}; p={stats['p_perm_diff_ge_obs']:.3f})")
    mutants = list(data["constant"]["src_k3_n8"]["mutants"])
    rows = [
        null["loo_error_null_src_pos"]["per_mutant_mae"],
        baselines["src"]["pos"]["CLR-GP"]["errors"],
        benchmark["src"]["Extended_10dim"]["mae_per_mutant"],
        esm["src"]["k3_pca_20dim"]["errors"],
    ]
    row_labels = ["CLR-Ridge\n(position)", "CLR-GP\n(position)", "MLP\n(Extended)", "ESM-2 PCA\n(fold-local)"]
    matrix = np.asarray([[row[m] for m in mutants] for row in rows])
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=0.60, aspect="auto", interpolation="nearest")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.0, color="white" if value > 0.36 else INK)
    members = set(null["loo_error_null_src_pos"]["collision_members"])
    for j, mutant in enumerate(mutants):
        if mutant in members:
            ax.add_patch(patches.Rectangle((j - 0.48, -0.48), 0.96, matrix.shape[0] - 0.04, fill=False,
                                           edgecolor=PALETTE["red_strong"], lw=0.9))
    l325_index = mutants.index("SrcKD-L325A")
    ax.add_patch(patches.Rectangle((l325_index - 0.44, -0.44), 0.88, 0.88, fill=False,
                                   edgecolor=PALETTE["highlight"], lw=1.25, ls=(0, (3, 1))))
    xlabels = [clean_mutant(m) for m in mutants]
    ax.set_xticks(np.arange(len(mutants)))
    ax.set_xticklabels(xlabels, rotation=90, ha="center", va="top", rotation_mode="anchor")
    ax.tick_params(axis="x", pad=13.0)
    for tick, mutant in zip(ax.get_xticklabels(), mutants):
        if mutant in members:
            tick.set_color(PALETTE["red_strong"])
            tick.set_weight("bold")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("MAE", fontsize=6.4, labelpad=3)
    colorbar.set_ticks([0.0, 0.2, 0.4, 0.6])
    colorbar.ax.tick_params(labelsize=6.0, width=0.5, length=2)
    return fig


def figure3_resolution(data: dict[str, object]) -> plt.Figure:
    review = data["review"]
    review_k2 = data["review_k2"]
    contrasts = review["contrast"]
    current_mlp = data["src_label_sensitivity"]["systems"]["primary_probe"]["fixed_k3"]["pos::LowRankCDST"]
    wt_u1, wt_u2 = 0.44, -0.14

    fig = plt.figure(figsize=FIGURE_SIZES["fig3_resolution"])
    gs = fig.add_gridspec(
        1, 3, width_ratios=[2.34, 1.30, 1.18],
        left=0.080, right=0.970, bottom=0.24, top=0.84, wspace=0.60,
    )

    ax = fig.add_subplot(gs[0, 0])
    title(ax, "a", "True and predicted contrast shifts")
    gp = contrasts["clrgp_pos"]["per_mutant"]
    mlp = {
        name: {
            "u1_pred": 2.0 * pred[0] - 1.0,
            "u2_pred": pred[1] - pred[2],
        }
        for name, pred in current_mlp["per_mutant_pred"].items()
    }
    names = list(gp)
    true_points = []
    gp_points = []
    mlp_points = []
    for name in names:
        true = np.array([gp[name]["u1_true"] - wt_u1, gp[name]["u2_true"] - wt_u2])
        gp_pred = np.array([gp[name]["u1_pred"] - wt_u1, gp[name]["u2_pred"] - wt_u2])
        mlp_pred = np.array([mlp[name]["u1_pred"] - wt_u1, mlp[name]["u2_pred"] - wt_u2])
        true_points.append(true)
        gp_points.append(gp_pred)
        mlp_points.append(mlp_pred)
        ax.plot([true[0], gp_pred[0]], [true[1], gp_pred[1]], color=PALETTE["blue_main"], alpha=0.24, lw=0.55)
        ax.plot([true[0], mlp_pred[0]], [true[1], mlp_pred[1]], color=PALETTE["red_strong"], alpha=0.20, lw=0.55)
    true_points = np.asarray(true_points)
    gp_points = np.asarray(gp_points)
    mlp_points = np.asarray(mlp_points)
    ax.scatter(true_points[:, 0], true_points[:, 1], s=22, facecolor="white", edgecolor=INK, lw=0.8, label="NMR target", zorder=4)
    ax.scatter(gp_points[:, 0], gp_points[:, 1], s=20, marker="^", color=PALETTE["blue_main"], edgecolor="white", lw=0.3, label="CLR-GP", zorder=3)
    ax.scatter(mlp_points[:, 0], mlp_points[:, 1], s=18, marker="s", color=PALETTE["red_strong"], edgecolor="white", lw=0.3, label="MLP", zorder=3)
    label_positions = {
        "SrcKD-L410A": (-0.03, 0.28),
        "SrcKD-V332I": (-0.43, 0.76),
        "SrcKD-L270F_V332I": (-1.20, 0.94),
        "SrcKD-L325A": (-1.57, 1.25),
        "SrcKD-V380A": (-1.34, 0.47),
        "SrcKD-V331A": (-1.34, -0.07),
        "SrcKD-F405A": (-1.34, -0.66),
    }
    for name, point in zip(names, true_points):
        if name == "SrcKD-A311I":
            continue
        label = clean_mutant(name)
        if name == "SrcKD-L325A":
            label = "L325A/A311I"
        label_xy = label_positions[name]
        leader = np.linalg.norm(np.asarray(label_xy) - point) > 0.14
        ax.annotate(
            label, point, xytext=label_xy, textcoords="data", fontsize=6.2, color=INK,
            ha="left", va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.35, "alpha": 0.94},
            arrowprops={"arrowstyle": "-", "color": MID_GRAY, "lw": 0.45} if leader else None,
        )
    ax.axhline(0, color=LIGHT_GRAY, lw=0.6)
    ax.axvline(0, color=LIGHT_GRAY, lw=0.6)
    ax.set_xlim(-1.62, 0.22)
    ax.set_ylim(-0.76, 1.30)
    ax.set_xlabel(r"Shared shift, $\Delta u_1$")
    ax.set_ylabel(r"Fine-state shift, $\Delta u_2$")
    ax.legend(loc="lower right", bbox_to_anchor=(0.995, 0.015), ncol=3, columnspacing=0.65,
              handletextpad=0.25, borderaxespad=0.0, fontsize=6.1)

    ax = fig.add_subplot(gs[0, 1])
    title(ax, "b", "Contrast-resolved error")
    true_u1 = np.asarray([gp[name]["u1_true"] for name in names], dtype=float)
    true_u2 = np.asarray([gp[name]["u2_true"] for name in names], dtype=float)
    pred_u1 = np.asarray([mlp[name]["u1_pred"] for name in names], dtype=float)
    pred_u2 = np.asarray([mlp[name]["u2_pred"] for name in names], dtype=float)

    def r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
        return float(1.0 - np.sum((target - prediction) ** 2) / np.sum((target - target.mean()) ** 2))

    mlp_summary = {
        "u1_mae": current_mlp["u1_u2_contrast"]["u1"],
        "u2_mae": current_mlp["u1_u2_contrast"]["u2"],
        "u1_r2": r2_score(true_u1, pred_u1),
        "u2_r2": r2_score(true_u2, pred_u2),
    }
    summaries = [contrasts["clrgp_pos"]["summary"], mlp_summary]
    values = [summaries[0]["u1_mae"], summaries[1]["u1_mae"], summaries[0]["u2_mae"], summaries[1]["u2_mae"]]
    r2_values = [summaries[0]["u1_r2"], summaries[1]["u1_r2"], summaries[0]["u2_r2"], summaries[1]["u2_r2"]]
    y = np.array([3.0, 2.2, 0.9, 0.1])
    colors = [PALETTE["blue_main"], PALETTE["red_2"], PALETTE["blue_main"], PALETTE["red_2"]]
    bars = ax.barh(y, values, height=0.56, color=colors, edgecolor=MID_GRAY, lw=0.5)
    for yi, value in zip(y, values):
        ax.text(value + 0.025, yi, f"{value:.2f}", va="center", fontsize=6.0)
    ax.axhline(1.55, color=LIGHT_GRAY, lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([r"$u_1$ GP", r"$u_1$ MLP", r"$u_2$ GP", r"$u_2$ MLP"])
    ax.set_xlabel("MAE")
    ax.set_xlim(0, 0.92)
    ax.set_ylim(-0.35, 3.45)
    ax.grid(axis="x", color=LIGHT_GRAY, lw=0.45)

    ax = fig.add_subplot(gs[0, 2])
    title(ax, "c", r"Pooling deletes $u_2$")
    pooled = review_k2["k2_clrgp_pos"]["u1_scale_mae"]
    ax.bar(0, pooled, width=0.55, color=PALETTE["teal"], edgecolor=MID_GRAY, lw=0.5)
    ax.text(0, pooled + 0.025, f"{pooled:.3f}", ha="center", fontsize=6.4)
    ax.bar(1, 0.70, bottom=0, width=0.55, facecolor="white", edgecolor=MID_GRAY,
           lw=0.5, hatch="////", zorder=1)
    ax.text(1.0, 0.37, "not\nidentifiable", ha="center", va="center", fontsize=6.0,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.92}, zorder=4)
    ax.axhline(summaries[0]["u1_mae"], color=PALETTE["blue_main"], lw=0.8, ls="--", label="K=3 GP")
    ax.axhline(summaries[1]["u1_mae"], color=PALETTE["red_strong"], lw=0.8, ls=":", label="K=3 MLP")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"pooled $u_1$", r"$u_2$"])
    ax.set_ylabel("MAE")
    ax.set_ylim(0, 0.78)
    ax.legend(loc="upper left", handlelength=1.2, handletextpad=0.3, fontsize=6.0)
    return fig


def figure4_synthetic(data: dict[str, object]) -> plt.Figure:
    """Render the frozen controlled synthetic audit as a paper-native figure."""
    summary = parse_synthetic_summary(data["synthetic_summary"])
    fig = plt.figure(figsize=FIGURE_SIZES["fig4_synthetic_framework"])
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.12, 1.10, 1.0],
        left=0.085, right=0.975, bottom=0.225, top=0.845, wspace=0.60,
    )

    # (a) Exact collisions and separation.  Error bands are Monte Carlo SEM.
    ax = fig.add_subplot(gs[0, 0])
    title(ax, "a", "Collision-pair error")
    delta_styles = [
        (0.0, PALETTE["blue_main"], "o", "-"),
        (0.6, PALETTE["highlight"], "s", "--"),
        (1.2, PALETTE["green_3"], "^", "-."),
    ]
    for delta, color, marker, linestyle in delta_styles:
        rows = [
            row for row in summary
            if row["n"] == 20 and row["m"] == 1
            and row["resolution"] == "full_k3" and row["delta"] == delta
        ]
        rows.sort(key=lambda row: row["epsilon"])
        x = np.asarray([row["epsilon"] for row in rows], float)
        y = np.asarray([row["pair_clr_ridge_mae_mean"] for row in rows], float)
        se = np.asarray([row["pair_clr_ridge_mae_se"] for row in rows], float)
        ax.fill_between(x, y - se, y + se, color=color, alpha=0.12, lw=0)
        ax.plot(x, y, color=color, marker=marker, ls=linestyle, lw=1.25, ms=3.4,
                markeredgecolor="white", markeredgewidth=0.35)
        ax.annotate(
            fr"$\delta={delta:g}$", xy=(x[-1], y[-1]), xytext=(1.4, y[-1]),
            textcoords="data", ha="left", va="center", fontsize=5.7, color=color,
        )
    floor_row = [
        row for row in summary
        if row["n"] == 20 and row["m"] == 1 and row["resolution"] == "full_k3"
        and row["delta"] == 1.2 and row["epsilon"] == 0.0
    ][0]
    floor = floor_row["equal_prediction_pair_floor_mean"]
    ax.scatter(0.0, floor, color=INK, marker="*", s=31, zorder=5)
    ax.annotate(
        "equal-prediction ref.", xy=(0.0, floor), xytext=(0.065, floor - 0.006),
        ha="left", va="center", fontsize=5.7, color=INK,
        arrowprops={"arrowstyle": "-", "lw": 0.5, "color": MID_GRAY},
    )
    ax.set_xlabel(r"Collision separation, $\epsilon$")
    ax.set_ylabel("Collision-pair MAE")
    ax.set_xscale("symlog", linthresh=0.05, linscale=0.75, base=10)
    ax.set_xlim(-0.008, 5.0)
    ax.set_ylim(-0.005, 0.185)
    ax.set_xticks([0.0, 0.05, 0.2, 1.0])
    ax.set_xticklabels(["0", "0.05", "0.2", "1"])
    ax.grid(axis="y", color=LIGHT_GRAY, lw=0.45)

    # (b) Shared and fine contrasts under full versus pooled targets.
    ax = fig.add_subplot(gs[0, 1])
    title(ax, "b", "Resolution controls error")
    series = [
        ("full_k3", "shared_contrast_clr_mae_mean", "full K=3: shared", PALETTE["blue_main"], "o", "-"),
        ("pooled_k2", "shared_contrast_clr_mae_mean", "pooled K=2: shared", PALETTE["green_3"], "s", "--"),
        ("full_k3", "fine_contrast_clr_mae_mean", "full K=3: fine", PALETTE["violet"], "^", "-."),
    ]
    label_y_offsets = {"full K=3: shared": -0.0038, "pooled K=2: shared": 0.0030, "full K=3: fine": 0.0}
    for resolution, field, label, color, marker, linestyle in series:
        ys = []
        for delta in (0.0, 0.6, 1.2):
            rows = [
                row for row in summary
                if row["n"] == 20 and row["m"] == 1
                and row["resolution"] == resolution and row["delta"] == delta
            ]
            ys.append(float(np.mean([row[field] for row in rows])))
        ax.plot((0.0, 0.6, 1.2), ys, color=color, marker=marker, ls=linestyle,
                lw=1.25, ms=3.4, markeredgecolor="white", markeredgewidth=0.35)
        label_y = ys[-1] + label_y_offsets[label]
        ax.annotate(
            label, xy=(1.2, ys[-1]), xytext=(1.255, label_y), textcoords="data",
            ha="left", va="center", fontsize=6.1, color=color,
            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": color},
        )
    ax.set_xlabel(r"Hidden fine contrast, $\delta$")
    ax.set_ylabel("Contrast MAE")
    ax.set_xlim(-0.06, 2.40)
    ax.set_ylim(-0.002, 0.057)
    ax.set_xticks([0.0, 0.6, 1.2])
    ax.grid(axis="y", color=LIGHT_GRAY, lw=0.45)

    # (c) Candidate selection on an evenly spaced categorical x-axis.
    ax = fig.add_subplot(gs[0, 2])
    title(ax, "c", "Selection optimism")
    x = np.arange(3)
    m_values = (1, 5, 20)
    for resolution, label, color, marker, linestyle in [
        ("full_k3", "full K=3", PALETTE["blue_main"], "o", "-"),
        ("pooled_k2", "pooled K=2", PALETTE["green_3"], "s", "--"),
    ]:
        ys, ses = [], []
        for m in m_values:
            values = np.asarray([
                row["selection_optimism_mean"] for row in summary
                if row["m"] == m and row["resolution"] == resolution
            ], float)
            ys.append(float(values.mean()))
            ses.append(float(values.std(ddof=1) / np.sqrt(len(values))))
        ax.errorbar(
            x, ys, yerr=ses, color=color, marker=marker, ls=linestyle, lw=1.25,
            ms=3.5, capsize=2.0, elinewidth=0.85, markeredgecolor="white",
            markeredgewidth=0.35,
        )
        ax.annotate(
            label, xy=(x[-1], ys[-1]), xytext=(2.13, ys[-1]), textcoords="data",
            ha="left", va="center", fontsize=6.1, color=color,
            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": color},
        )
    ax.axhline(0, color=MID_GRAY, lw=0.65)
    ax.set_xlabel(r"Representation candidates, $m$")
    ax.set_ylabel(r"Nested $-$ naive MAE")
    ax.set_xlim(-0.12, 3.02)
    ax.set_ylim(-0.00035, 0.00785)
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in m_values])
    ax.grid(axis="y", color=LIGHT_GRAY, lw=0.45)
    return fig


def figure4_evidence(data: dict[str, object]) -> plt.Figure:
    constant = data["constant"]
    esm = data["esm"]
    llr = data["llr"]
    nested_full = data["nested_full"]
    benchmark = data["benchmark"]
    grouped = data["grouped"]

    fig = plt.figure(figsize=FIGURE_SIZES["fig4_evidence"])
    gs = fig.add_gridspec(1, 3, width_ratios=[1.65, 1.90, 1.65], left=0.175, right=0.970, bottom=0.23, top=0.83, wspace=0.72)

    ax = fig.add_subplot(gs[0, 0])
    title(ax, "a", "Evidence levels (Abl1)")
    values = [
        constant["abl1_k3_n6"]["training_mean_LOO"]["mean_mae"],
        esm["abl1"]["k3_pca_20dim"]["mae"],
        llr["llr_proxy"]["mae"],
        nested_full["systems"]["abl1"]["nested_mlp"]["mae"],
        benchmark["abl1_core"]["C_ddg_5dim"]["mae"],
    ]
    labels = [
        "Training mean\n(baseline)",
        "ESM-2 PCA\nfixed-panel",
        "LLR\nfixed-panel",
        "Full nested MLP\nfold-selected",
        "Energy + position\ntarget-coupled",
    ]
    colors = [PALETTE["neutral"], PALETTE["blue_main"], PALETTE["teal"], PALETTE["violet"], PALETTE["red_1"]]
    y = np.arange(5)[::-1]
    bars = ax.barh(y, values, color=colors, edgecolor=MID_GRAY, lw=0.5, height=0.64)
    bars[-1].set_hatch("///")
    bars[-1].set_edgecolor(PALETTE["red_strong"])
    for yi, value in zip(y, values):
        ax.text(value + 0.007, yi, f"{value:.3f}", va="center", fontsize=6.1)
    ax.axhline(3.5, color=LIGHT_GRAY, lw=0.7)
    ax.axhline(1.5, color=LIGHT_GRAY, lw=0.7)
    ax.axhline(0.5, color=PALETTE["red_strong"], lw=0.7, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("LOO MAE")
    ax.set_xlim(0, 0.335)
    ax.set_ylim(-0.55, 4.55)
    ax.grid(axis="x", color=LIGHT_GRAY, lw=0.5)

    ax = fig.add_subplot(gs[0, 1])
    title(ax, "b", "Fixed-feature endpoint errors")
    mutants = list(constant["abl1_k3_n6"]["mutants"])
    series = [
        (constant["abl1_k3_n6"]["training_mean_LOO"]["per_mutant"], -0.19, "Mean", "x", MID_GRAY),
        (esm["abl1"]["k3_pca_20dim"]["errors"], 0.0, "ESM-2 PCA", "o", PALETTE["blue_main"]),
        (llr["llr_proxy"]["errors"], 0.19, "LLR", "s", PALETTE["teal"]),
    ]
    x = np.arange(len(mutants))
    last_points = {}
    for row, offset, label, marker, color in series:
        y = [row[m] for m in mutants]
        marker_style = {"color": color, "lw": 0.9} if marker == "x" else {
            "color": color, "edgecolor": "white", "lw": 0.35
        }
        ax.scatter(x + offset, y, s=19, marker=marker, label=label, zorder=3, **marker_style)
        last_points[label] = (x[-1] + offset, y[-1], color)
    for index, mutant in enumerate(mutants):
        pair = [esm["abl1"]["k3_pca_20dim"]["errors"][mutant], llr["llr_proxy"]["errors"][mutant]]
        ax.plot([index, index + 0.19], pair, color=LIGHT_GRAY, lw=0.7, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "+") for m in mutants], rotation=90, ha="center", va="top", rotation_mode="anchor")
    ax.set_ylabel("LOO MAE")
    ax.set_xlim(-0.45, 7.95)
    ax.set_ylim(0, 0.56)
    label_heights = {"Mean": last_points["Mean"][1], "ESM-2 PCA": 0.028, "LLR": 0.071}
    for label in ("Mean", "ESM-2 PCA", "LLR"):
        px, py, color = last_points[label]
        ax.annotate(
            label, xy=(px, py), xytext=(5.38, label_heights[label]), textcoords="data",
            ha="left", va="center", fontsize=6.1, color=color,
            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": color},
        )
    ax.grid(axis="y", color=LIGHT_GRAY, lw=0.5)

    ax = fig.add_subplot(gs[0, 2])
    title(ax, "c", "Site-transfer MAE")
    encodings = ["variant_C", "LLR_proxy", "pos_markers"]
    labels = ["Energy\n+ position", "LLR", "Position\nmarkers*"]
    f382 = [grouped[key]["F382_family"]["group_mae"] for key in encodings]
    site_290 = [grouped[key]["290_301"]["group_mae"] for key in encodings]
    x = np.arange(3)
    width = 0.34
    bars1 = ax.bar(x - width / 2, f382, width, color=PALETTE["red_2"], edgecolor=MID_GRAY,
                   lw=0.5, label="F382 family")
    bars2 = ax.bar(x + width / 2, site_290, width, color=PALETTE["blue_main"], edgecolor=MID_GRAY,
                   lw=0.5, hatch="//", label="M290/L301")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.009, f"{bar.get_height():.2f}", ha="center", fontsize=6.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Group MAE")
    ax.set_xlabel("* identity columns constant\nin M290/L301 training", fontsize=6.1, labelpad=3)
    ax.set_ylim(0, 0.41)
    ax.legend(loc="upper center", ncol=2, handlelength=1.0, handletextpad=0.3, columnspacing=0.8)
    ax.grid(axis="y", color=LIGHT_GRAY, lw=0.45)
    return fig


def figure5_alignment(data: dict[str, object]) -> plt.Figure:
    original = data["af2_original"]
    verify = data["af2_verify"]
    plddt = data["af2_plddt"]["protocols"]
    original_plddt = tuple(plddt["original"]["mutant_mean_range"])
    fresh_plddt = tuple(plddt["fresh_msa"]["mutant_mean_range"])

    original_regions = original["region_sensitivity"]["regions"]
    counts = {
        "full": {
            "original": verify["main_840"]["reclassified"],
            "fresh": verify["b1_480_full_protein"]["reclassified"],
        },
        "n_lobe": {
            "original": original_regions["n_lobe_act"]["state_counts"],
            "fresh": verify["b1_region_sensitivity"]["n_lobe_act"]["counts"],
        },
        "alpha_c": {
            "original": original_regions["alphaC_only"]["state_counts"],
            "fresh": verify["b1_region_sensitivity"]["alphaC_only"]["counts"],
        },
    }
    totals = {"original": 840, "fresh": 480}

    fig = plt.figure(figsize=FIGURE_SIZES["fig5_alignment"])
    gs = fig.add_gridspec(
        1, 2, width_ratios=[3.55, 2.15],
        left=0.075, right=0.975, bottom=0.25, top=0.76, wspace=0.48,
    )

    ax = fig.add_subplot(gs[0, 0])
    title(ax, "a", "State assignment by alignment")
    centers = np.arange(3)
    width = 0.30
    state_colors = {
        "active": PALETTE["blue_main"],
        "i1i2": PALETTE["highlight"],
        "unclassified": PALETTE["neutral"],
    }
    for protocol, offset in (("original", -width / 1.7), ("fresh", width / 1.7)):
        for center, region in zip(centers, ("full", "n_lobe", "alpha_c")):
            record = counts[region][protocol]
            active = record.get("active", 0)
            i1i2 = record.get("I1", 0) + record.get("I2", 0)
            unclassified = record.get("unclassified", 0)
            values = [active, i1i2, unclassified]
            if sum(values) != totals[protocol]:
                raise ValueError(f"AF2 counts do not sum for {region}/{protocol}: {values}")
            bottom = 0.0
            for value, state in zip(values, ("active", "i1i2", "unclassified")):
                fraction = value / totals[protocol]
                ax.bar(center + offset, fraction, width, bottom=bottom, color=state_colors[state], edgecolor=MID_GRAY,
                       lw=0.4, zorder=2)
                if value and fraction >= 0.065:
                    text_color = "white" if state == "active" else INK
                    ax.text(center + offset, bottom + fraction / 2, str(value), ha="center", va="center",
                            fontsize=6.1, color=text_color, weight="bold" if fraction > 0.22 else "normal")
                bottom += fraction
    ax.set_xticks(centers)
    ax.set_xticklabels(["Full protein (primary)\nI1/I2: 0/840 | 0/480", "Residues 235-400\norig.: 147-165 C$\\alpha$", r"$\alpha$C 260-300" + "\norig.: 40-41 C$\\alpha$"])
    ax.get_xticklabels()[0].set_color(PALETTE["red_strong"])
    ax.set_ylabel(r"Assigned fraction (3 $\AA$ cutoff)")
    ax.set_ylim(0, 1.10)
    ax.set_yticks([0, 0.5, 1.0])
    state_handles = [patches.Patch(facecolor=state_colors[key], edgecolor=MID_GRAY, lw=0.4, label=label)
                     for key, label in (("active", "Active"), ("i1i2", "I1/I2"), ("unclassified", "Unclassified"))]
    protocol_handles = [
        patches.Patch(facecolor="white", edgecolor=MID_GRAY, label="Original n=840"),
        patches.Patch(facecolor=LIGHT_GRAY, edgecolor=MID_GRAY, label="Fresh MSA n=480"),
    ]
    ax.legend(handles=state_handles + protocol_handles, loc="lower center", bbox_to_anchor=(0.53, 1.16), ncol=5,
              columnspacing=0.72, handlelength=1.05, handletextpad=0.24, labelspacing=0.25)

    ax = fig.add_subplot(gs[0, 1])
    title(ax, "b", "Protocol quality")
    ranges = [original_plddt, fresh_plddt]
    colors = [PALETTE["blue_main"], PALETTE["red_strong"]]
    for y, interval, color in zip((1, 0), ranges, colors):
        ax.hlines(y, interval[0], interval[1], color=color, lw=3.0)
        ax.vlines(interval, y - 0.055, y + 0.055, color=color, lw=0.9)
        ax.text(interval[1] + 0.4, y, f"{interval[0]:.0f}-{interval[1]:.0f}",
                va="center", fontsize=6.3, color=color)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Original", "Fresh MSA"])
    ax.set_xlim(52, 88)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("Mean pLDDT range")
    ax.set_xticks([55, 70, 80])
    ax.grid(axis="x", color=LIGHT_GRAY, lw=0.5)
    return fig


def write_manifest(outputs: list[Path]) -> None:
    manifest = {
        "generator": str((PAPER_DIR / "generate_v2_figures.py").relative_to(ROOT)).replace("\\", "/"),
        "generator_sha256": hashlib.sha256((PAPER_DIR / "generate_v2_figures.py").read_bytes()).hexdigest(),
        "skill": {
            "name": "scientific-figure-making",
            "source": "https://github.com/ChenLiu-1996/figures4papers",
            "commit": "fcd46ec41a11773e8a284c16aa751ae755e920ca",
        },
        "style": {
            "palette": "seaborn-deep/NPG-inspired softened; redundant marker/line/hatch encodings",
            "font": "Arial/Helvetica/DejaVu Sans fallback",
            "exports": "vector PDF/SVG plus 300-DPI PNG",
        },
        "figure_sizes_inches": {name: list(size) for name, size in FIGURE_SIZES.items()},
        "sources": {},
        "outputs": [str(path.relative_to(ROOT)).replace("\\", "/") for path in outputs],
        "output_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in outputs
        },
    }
    for name, path in SOURCE_PATHS.items():
        manifest["sources"][name] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (FIG_DIR / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    apply_publication_style()
    data = load_sources()
    validate_sources(data)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    makers = [
        ("fig1_workflow", lambda: figure1_workflow()),
        ("fig2_support_map", lambda: figure2_support(data)),
        ("fig3_resolution", lambda: figure3_resolution(data)),
        ("fig4_synthetic_framework", lambda: figure4_synthetic(data)),
        ("fig4_evidence", lambda: figure4_evidence(data)),
        ("fig5_alignment", lambda: figure5_alignment(data)),
    ]
    outputs: list[Path] = []
    for stem, maker in makers:
        outputs.extend(save_figure(maker(), stem))
        print(f"[OK] {stem}: PDF, SVG, PNG")
    write_manifest(outputs)
    print(f"[OK] manifest: {FIG_DIR / 'figure_manifest.json'}")


if __name__ == "__main__":
    main()
