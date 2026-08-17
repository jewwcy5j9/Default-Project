#!/usr/bin/env python
"""Audit active submission claims against their machine-readable authorities."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path


PRIMARY_PROTOCOL_ID = "src_k3_figs5_met305_primary_v1"
HYBRID_PROTOCOL_ID = "src_k3_figs5_met305_with_table_s2_l410a_substitution_v1"
PRIMARY_GP = {
    "id": "GP-primary-0.05-v1",
    "kernel": "1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.05)",
    "alpha": 0.0001,
    "normalize_y": True,
    "n_restarts_optimizer": 1,
    "random_state": 0,
}


def _load_json(path: Path, errors: list[str]):
    if not path.is_file():
        errors.append(f"missing required artifact: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read JSON {path}: {exc}")
        return None


def _check_gp(label: str, metadata: dict, errors: list[str]) -> None:
    for key, expected in PRIMARY_GP.items():
        if metadata.get(key) != expected:
            errors.append(
                f"{label} GP {key} drift: expected {expected!r}, "
                f"found {metadata.get(key)!r}"
            )


def _check_manuscript(root: Path, errors: list[str]) -> None:
    import os
    ms_name = os.environ.get("MS_TEX", "main_v3.tex")
    path = root / "paper" / ms_name
    if not path.is_file():
        errors.append(f"missing active manuscript: {path}")
        return
    text = path.read_text(encoding="utf-8")
    normalized_text = " ".join(text.split()).lower()
    forbidden = {
        r"\bprespecified\b": "prespecified evidence wording",
        r"preregistered (evidence|panels?|analyses|results|benchmarks?)\b|"
        r"was preregistered|were preregistered":
            "preregistered evidence wording (retrospective panels)",
        r"essentially unchanged": "unsupported pooling invariance claim",
        r"inferred I2-like frequency": "thermodynamic frequency wording",
        r"Exact paired comparison": "unqualified exact paired inference",
        r"bootstrap CI": "unqualified bootstrap confidence interval",
        r"95\\%\s*CI\s*\(": "unqualified paired 95% CI header",
        r"all 15 CLR rows": "incorrect CLR-row count",
        r"dual-protocol Src": "misleading complete-protocol wording",
        r"two label protocols": "misleading complete-protocol wording",
        r"global protocol": "misleading complete-global-protocol wording",
        r"\b8/8\b": "stale primary Src direction denominator",
        r"\$82\$--\$85\$|82--85": "stale original-pLDDT range",
        r"MLP, \$K=3\$\s*&\s*0\.501": "stale Src Position/MLP contrast row",
        r"MLP, \$K=3\$\s*&\s*0\.521": "stale raw-contrast Src Position/MLP row",
        r"\\label\{app:additional\}": "duplicate Appendix D label",
    }
    for pattern, label in forbidden.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            errors.append(f"{ms_name} contains {label}: /{pattern}/")

    required = {
        "retrospectively fixed": "retrospective PCA evidence role",
        "finite-panel paired sensitivity": "finite-panel paired caveat",
        "overlapping LOO": "overlapping-LOO dependence caveat",
        "site-related mutation families": "site-family dependence caveat",
        "Target ties": "direction target-tie definition",
        "target non-ties": "direction denominator definition",
        "assigned fractions": "structural assignment terminology",
        "not a second complete global-fit": "hybrid-panel limitation",
        "full 10-row MAE rankings are not invariant": "pseudocount-instability disclosure",
        "49/50": "pseudocount contrast count",
        "does not substitute for independent redigitization": "digitization limitation",
        "without refitting": "descriptive deletion limitation",
        "the two routes are not pooled": "robustness-route separation",
        "451/840": "original AF2 alphaC I1-like count",
        "39/480": "fresh-MSA AF2 alphaC I1-like count",
        "GP hyperparameters fitted on training folds only":
            "training-fold GP hyperparameter-fit disclosure",
        "not directly comparable numerically with the fixed-MLP or nested-MLP rows":
            "joint-selector scaling comparability disclosure",
        "the candidate set, model set, and tie-break differ":
            "joint-selector search-space distinction",
        "is not the confirmatory route's $0.3700$ estimate":
            "joint-selector versus confirmatory value distinction",
        "equal-prediction reference for a single shared prediction is $0.1475$":
            "synthetic equal-prediction reference semantics",
        "not a lower bound on this two-fit LOO statistic":
            "synthetic reference is not a two-fit LOO bound",
        "12,000 generated datasets":
            "synthetic generated-dataset count",
        "candidate set, however, was assembled retrospectively":
            "retrospective nested candidate provenance",
        "within-fold resampling audit":
            "selector-stability audit disclosure",
        "in the orthonormal basis, the $q_2$ (e1--e2) mae exceeds":
            "orthonormal contrast-basis reporting",
        "50/50 rows (raw $u_2>u_1$: 49/50)":
            "orthonormal pseudocount count with raw comparison",
        "90.5\\% (raw: 60.5\\%)":
            "orthonormal stress proportion with raw comparison",
        "feature-near graph was fixed":
            "collision null graph definition",
        "count of thresholded conflicts":
            "collision null thresholded conflict count",
        "not sufficient certification":
            "necessary-not-sufficient audit scope",
        "falls back to the first $N$ ordered C$\\alpha$ atoms":
            "AF2 alignment fallback disclosure",
        "alignment-mode audit":
            "AF2 alignment-mode audit disclosure",
        "FoldX mutation free energies":
            "FoldX independent baseline disclosure",
        "failed-QC exploratory evidence":
            "FoldX failed-QC labeling",
        "predeclared quality-control gates":
            "FoldX predeclared QC-gate disclosure",
        "0.0956":
            "FoldX Abl1 repeat-perturbation QC value",
        "0.2396":
            "FoldX Src repeat-perturbation QC value",
        "heterogeneous-candidate stress test":
            "P5 separate-experiment framing",
        "not a one-factor ablation":
            "P5 non-ablation disclosure",
        "0.0157":
            "P4 n=6 m=20 full-K3 optimism convention",
        "0.0198":
            "P4 n=6 m=20 pooled-K2 optimism convention",
        "per-fold oracle":
            "selector regret per-fold oracle naming",
        "CLR-GP (T5 diagnostic)":
            "T5-diagnostic GP row label",
        "CLR-GP (primary)":
            "primary GP row label",
        "MLP (primary probe), $K=3$ & 0.319 & $-1.21$ & 0.484 & $-0.97$":
            "current reproducible Src Position/MLP contrast row",
        "predeclared success":
            "prospective predeclared success gates",
        "deletion robustness":
            "prospective deletion-robustness gate",
        "audit detection benchmark":
            "P6 audit detection-benchmark disclosure",
        "nomological network":
            "audit nomological-network framing",
        "benchmarking-epistemology":
            "benchmarking-epistemology alignment",
        "calibrate the audit":
            "audit calibration sentence",
        "$q_2>q_1$ detection rate":
            "fig4 panel d caption",
    }
    for token, label in required.items():
        if token.lower() not in normalized_text:
            errors.append(f"{ms_name} missing {label}: {token!r}")

    for label in (
        "fig:workflow", "fig:resolution", "fig:alignment", "tab:model-defs",
        "tab:unified-robustness", "tab:per-mut-errors",
    ):
        if f"\\ref{{{label}}}" not in text:
            errors.append(f"{ms_name} does not reference labeled float: {label}")

    evidence_rows = re.findall(
        r"^(Retrospective fixed-panel|Retrospective fixed|Fixed-panel)\s*&\s*(?:Abl1|Src)\s*&",
        text,
        flags=re.MULTILINE,
    )
    if evidence_rows != ["Retrospective fixed-panel"] * 4:
        errors.append(f"{ms_name} fixed-representation evidence labels are inconsistent")


def _check_src_labels(root: Path, errors: list[str]) -> None:
    csv_path = root / "data" / "nmr_populations" / "src_k3_canonical.csv"
    if not csv_path.is_file():
        errors.append(f"missing canonical Src labels: {csv_path}")
        return
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_record = {row["record_id"]: row for row in rows}
    primary_rows = sorted(
        (row for row in rows if row["used_in_primary"].lower() == "true"),
        key=lambda row: int(row["panel_order"]),
    )
    if len(primary_rows) != 9 or primary_rows[0]["mutation_id"] != "SrcKD-WT":
        errors.append("canonical Src primary panel is not WT plus eight mutations")
        return
    replacement_id = "table_s2_global__SrcKD-L410A"
    if replacement_id not in by_record:
        errors.append("canonical Src labels lack the Table-S2 L410A record")
        return

    artifact = _load_json(
        root / "experiments" / "iclr_restructuring" / "results"
        / "p2_k3_src_label_sensitivity.json",
        errors,
    )
    if artifact is None:
        return
    protocols = artifact.get("protocols", {})
    systems = artifact.get("systems", {})
    expected_keys = {"primary_probe", "l410a_global_fit_substitution"}
    if set(protocols) != expected_keys or set(systems) != expected_keys:
        errors.append(
            "Src sensitivity must use only primary_probe and "
            "l410a_global_fit_substitution keys"
        )
        return
    primary = protocols["primary_probe"]
    hybrid = protocols["l410a_global_fit_substitution"]
    if primary.get("protocol_id") != PRIMARY_PROTOCOL_ID:
        errors.append("Src primary protocol ID drift")
    if hybrid.get("protocol_id") != HYBRID_PROTOCOL_ID:
        errors.append("Src L410A-substitution protocol ID drift")
    if primary.get("protocol_kind") != "primary_probe":
        errors.append("Src primary protocol kind drift")
    if hybrid.get("protocol_kind") != "hybrid_single_substitution":
        errors.append("Src sensitivity is not marked as a single-substitution hybrid")

    expected_wt = primary_rows[0]["record_id"]
    if primary.get("wt_record_id") != expected_wt or hybrid.get("wt_record_id") != expected_wt:
        errors.append("Src hybrid panel must retain the primary probe WT record")
    expected_targets = {row["mutation_id"]: row["record_id"] for row in primary_rows[1:]}
    if primary.get("target_record_ids") != expected_targets:
        errors.append("Src primary target records disagree with canonical CSV")
    expected_hybrid = dict(expected_targets)
    expected_hybrid["SrcKD-L410A"] = replacement_id
    if hybrid.get("target_record_ids") != expected_hybrid:
        errors.append("Src hybrid panel changes rows other than canonical L410A")
    expected_substitution = [{
        "mutation_id": "SrcKD-L410A",
        "from_record_id": expected_targets["SrcKD-L410A"],
        "to_record_id": replacement_id,
    }]
    if hybrid.get("substitutions") != expected_substitution:
        errors.append("Src hybrid substitution metadata drift")
    canonical_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if any(protocol.get("canonical_sha256") != canonical_hash for protocol in protocols.values()):
        errors.append("Src sensitivity canonical CSV hash drift")
    _check_gp("Src sensitivity", artifact.get("battery", {}).get("gp_protocol", {}), errors)
    current_mlp = (
        systems.get("primary_probe", {})
        .get("fixed_k3", {})
        .get("pos::LowRankCDST", {})
    )
    contrast = current_mlp.get("u1_u2_contrast", {})
    expected_contrast = {"u1": 0.5206792055330889, "u2": 0.6845696578954912}
    if any(
        not isinstance(contrast.get(key), (int, float))
        or abs(contrast[key] - expected) > 1e-12
        for key, expected in expected_contrast.items()
    ):
        errors.append("Src current Position/MLP contrast values drift")


def _check_src_clr_robustness(root: Path, errors: list[str]) -> None:
    artifact = _load_json(
        root / "experiments" / "iclr_restructuring" / "results"
        / "p2_k3_src_clr_robustness.json",
        errors,
    )
    if artifact is None:
        return
    if artifact.get("schema_version") != "src_clr_label_robustness_v1":
        errors.append("Src CLR robustness schema drift")
    if artifact.get("primary_protocol_id") != PRIMARY_PROTOCOL_ID:
        errors.append("Src CLR robustness primary protocol drift")
    canonical_path = root / "data" / "nmr_populations" / "src_k3_canonical.csv"
    if canonical_path.is_file():
        canonical_hash = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
        if artifact.get("canonical_sha256") != canonical_hash:
            errors.append("Src CLR robustness canonical CSV hash drift")
    _check_gp("Src CLR robustness", artifact.get("gp_protocol", {}), errors)
    sensitivity = artifact.get("pseudocount_sensitivity", {})
    if sensitivity.get("pseudocounts") != [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]:
        errors.append("Src CLR robustness pseudocount grid drift")
    rows = sensitivity.get("rows", [])
    expected_rows = 5 * 5 * 2
    if len(rows) != expected_rows:
        errors.append(f"Src CLR robustness must contain {expected_rows} grid rows")
    expected_combinations = {
        (pseudocount, candidate, model)
        for pseudocount in [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
        for candidate in ["pos", "ext", "llr_pos", "llr_only", "pca20"]
        for model in ["CLR-Ridge", "CLR-GP"]
    }
    combinations = {
        (row.get("pseudocount"), row.get("candidate"), row.get("model"))
        for row in rows
    }
    if combinations != expected_combinations:
        errors.append("Src CLR robustness grid combinations drift")
    summary = sensitivity.get("summary", {})
    expected_summary = {
        "mae_ranking_stable": False,
        "all_rows_u2_gt_u1": False,
        "f405a_vs_l410a_pattern_stable": True,
    }
    if any(summary.get(key) is not value for key, value in expected_summary.items()):
        errors.append("Src CLR pseudocount stability conclusion drift")
    if sum(bool(row.get("u2_gt_u1")) for row in rows) != 49:
        errors.append("Src CLR pseudocount u2>u1 count drift")
    if sum(bool(row.get("f405a_gt_l410a")) for row in rows) != 50:
        errors.append("Src CLR pseudocount F405A/L410A count drift")
    interval = artifact.get("digitization_interval_stress_test") or {}
    if interval.get("status") != "curator_interval_stress_test_not_independent_redigitization":
        errors.append("Src label-interval stress test status/limitation drift")
    if interval.get("realizations", 0) < 200:
        errors.append("Src label-interval stress test must retain at least 200 realizations")
    records = interval.get("records", [])
    if len(records) != interval.get("realizations"):
        errors.append("Src label-interval stress-test record count drift")
    proportions = interval.get("proportions", {})
    expected_proportions = {
        "gp_mae_le_ridge": 0.10,
        "both_u2_gt_u1": 0.605,
        "both_f405a_gt_l410a": 0.42,
    }
    if any(abs(proportions.get(key, -1.0) - value) > 1e-12
           for key, value in expected_proportions.items()):
        errors.append("Src label-interval stress-test proportions drift")
    if "not a substitute for independent redigitization" not in interval.get("limitation", ""):
        errors.append("Src label-interval stress test lacks redigitization limitation")


def _check_primary_gp(root: Path, errors: list[str]) -> None:
    p1 = _load_json(
        root / "experiments" / "iclr_restructuring" / "results"
        / "p1_core_baselines.json",
        errors,
    )
    if p1 is not None:
        _check_gp("P1 primary", p1.get("_protocols", {}).get("CLR-GP", {}), errors)

    nested_path = (
        root / "experiments" / "iclr_restructuring" / "results"
        / "p2_k3_nested_pca_results.json"
    )
    nested = _load_json(nested_path, errors)
    if nested is not None:
        if "gp_protocol" not in nested:
            errors.append("nested PCA artifact missing gp_protocol block")
        else:
            _check_gp("nested PCA", nested["gp_protocol"], errors)
        for system_name, system in nested.get("systems", {}).items():
            protocol = system.get("protocol", {})
            if protocol.get("scaling_primary") != "none (raw features; matches frozen fixed rows)":
                errors.append(f"nested PCA primary scaling drift: {system_name}")
            if protocol.get("scaling_secondary") != "fold-local StandardScaler for all candidates":
                errors.append(f"nested PCA secondary scaling drift: {system_name}")

    followup_source = root / "experiments" / "iclr_restructuring" / "k3_followup.py"
    if not followup_source.is_file():
        errors.append(f"missing K=3 follow-up source: {followup_source}")
    else:
        source = followup_source.read_text(encoding="utf-8")
        if "make_primary_gp()" not in source or "GaussianProcessRegressor(" in source:
            errors.append("k3_followup.py does not use the unified primary GP factory")
        if "GPR-probability-diagnostic" not in source:
            errors.append("k3_followup.py GP role is not explicitly diagnostic")


def _check_unified_robustness(root: Path, errors: list[str]) -> None:
    base = root / "experiments" / "iclr_restructuring" / "results"
    artifact = _load_json(base / "p2_k3_unified_robustness.json", errors)
    if artifact is None:
        return
    if artifact.get("schema_version") != "p2_k3_unified_robustness_v1":
        errors.append("unified K=3 robustness schema drift")
    expected_protocol = {
        "biological_unit": "held-out mutation",
        "descriptive_deletion_refits_model": False,
        "leave_site_out_refits_confirmatory_route": True,
        "routes_must_remain_separate": True,
    }
    if artifact.get("protocol") != expected_protocol:
        errors.append("unified K=3 robustness protocol drift")

    source_paths = {
        "p2_k3_paired_exact_sha256": base / "p2_k3_paired_exact.json",
        "p2_k3_nested_results_sha256": base / "p2_k3_nested_results.json",
    }
    source_hashes = artifact.get("source_hashes", {})
    for key, path in source_paths.items():
        if not path.is_file():
            errors.append(f"missing unified robustness source: {path}")
        elif source_hashes.get(key) != hashlib.sha256(path.read_bytes()).hexdigest():
            errors.append(f"unified K=3 robustness source hash drift: {key}")

    expected = {
        "representation_selection_audit": {
            "abl1": (6, 0.2625464544444444, 0.19406584953333333,
                     0.3045419476666667, 0.2726620573333333),
            "src": (8, 0.39904742025, 0.37533716895238095,
                    0.4150331301904762, 0.41046109823809523),
        },
        "candidate_model_confirmatory": {
            "abl1": (6, 0.44511347594444445, 0.4298732, 0.4888806, 0.4489346),
            "src": (8, 0.37002387679166665, 0.3309654285714286,
                    0.4227814285714286, 0.36982585714285715),
        },
    }
    routes = artifact.get("routes", {})
    if set(routes) != set(expected):
        errors.append("unified K=3 robustness route set drift")
        return
    for route_name, systems in expected.items():
        observed_systems = routes[route_name].get("systems", {})
        for system, values in systems.items():
            row = observed_systems.get(system, {})
            deletion = row.get("leave_one_observation_out", {})
            double = row.get("double_mutant_exclusion", {})
            observed = (
                row.get("n_mutations"),
                row.get("all_mutation_mae"),
                deletion.get("minimum_mae"),
                deletion.get("maximum_mae"),
                double.get("mae"),
            )
            if observed[0] != values[0] or any(
                not isinstance(actual, (int, float)) or abs(actual - target) > 1e-12
                for actual, target in zip(observed[1:], values[1:])
            ):
                errors.append(f"unified K=3 robustness values drift: {route_name}/{system}")
            if "no model or selector is refit" not in deletion.get("definition", ""):
                errors.append(f"unified K=3 deletion limitation drift: {route_name}/{system}")

    lso_expected = {
        ("abl1", "F382_family"): (0.26724303855555553, 0.27185185185185184),
        ("abl1", "290_301"): (0.3323498596666667, 0.14296296296296296),
        ("src", "N_lobe"): (0.35370970780000005, 0.33466666666666667),
        ("src", "C_lobe"): (0.4332842592222222, 0.3591111111111111),
    }
    confirmatory = routes.get("candidate_model_confirmatory", {}).get("systems", {})
    for (system, group), values in lso_expected.items():
        retraining = confirmatory.get(system, {}).get("leave_site_out_retraining", {})
        row = retraining.get("groups", {}).get(group, {})
        if (
            abs(row.get("group_mae", -1.0) - values[0]) > 1e-12
            or abs(row.get("comparator", -1.0) - values[1]) > 1e-12
        ):
            errors.append(f"unified K=3 leave-site-out drift: {system}/{group}")
        if "retrained" not in retraining.get("definition", ""):
            errors.append(f"unified K=3 leave-site-out definition drift: {system}")


def _check_af2(root: Path, errors: list[str]) -> None:
    artifact = _load_json(
        root / "experiments" / "af2_subsample" / "results"
        / "af2_plddt_raw_pdb.json",
        errors,
    )
    if artifact is not None:
        if artifact.get("schema_version") != "af2_plddt_raw_pdb_v1":
            errors.append("AF2 pLDDT raw-PDB schema drift")
        expected = {
            "original": (840, (80.95, 81.74)),
            "fresh_msa": (480, (54.56, 57.36)),
        }
        for name, (count, bounds) in expected.items():
            protocol = artifact.get("protocols", {}).get(name, {})
            if protocol.get("found_structures") != count:
                errors.append(f"AF2 {name} structure count drift")
            values = protocol.get("mutant_mean_range", [])
            if len(values) != 2 or values[0] < bounds[0] or values[1] > bounds[1]:
                errors.append(f"AF2 {name} mutant-mean pLDDT range drift: {values!r}")

    figure_source = root / "paper" / "generate_v2_figures.py"
    if not figure_source.is_file():
        errors.append(f"missing active figure generator: {figure_source}")
    else:
        text = figure_source.read_text(encoding="utf-8")
        if "af2_plddt_raw_pdb.json" not in text or "B1_FINAL_REPORT.md" in text:
            errors.append("Figure 5 must use raw-PDB AF2 pLDDT JSON, not Markdown")

    calibration = _load_json(
        root / "experiments" / "af2_subsample" / "results"
        / "assignment_calibration.json",
        errors,
    )
    if calibration is None:
        return
    if calibration.get("schema_version") != "af2_assignment_calibration_v1":
        errors.append("AF2 assignment calibration schema drift")
    expected_thresholds = [round(2.0 + 0.25 * index, 2) for index in range(13)]
    if calibration.get("method", {}).get("thresholds_angstrom") != expected_thresholds:
        errors.append("AF2 assignment calibration threshold grid drift")
    ambiguity = calibration.get("ambiguity_rule", {})
    if (
        ambiguity.get("margin_cutoff_angstrom") != 0.5
        or "not a thermodynamic population" not in ambiguity.get("interpretation", "")
    ):
        errors.append("AF2 assignment ambiguity rule drift")

    expected_pairs = {
        ("full_protein", "active__I2"): 8.893687880114713,
        ("n_lobe_act", "active__I2"): 7.326877582309726,
        ("alphaC_only", "active__I1"): 1.5686527987774919,
        ("alphaC_only", "active__I2"): 2.564432345596726,
    }
    references = calibration.get("reference_calibration", {})
    for (region, pair), expected_value in expected_pairs.items():
        value = references.get(region, {}).get("pairwise_distances", {}).get(pair, {}).get(
            "rmsd_angstrom", -1.0
        )
        if abs(value - expected_value) > 1e-12:
            errors.append(f"AF2 reference calibration drift: {region}/{pair}")

    expected_protocols = {
        "original": (840, {"active": 674, "I1": 0, "I2": 0,
                           "ambiguous": 0, "unclassified": 166}),
        "fresh_msa": (480, {"active": 36, "I1": 0, "I2": 0,
                            "ambiguous": 0, "unclassified": 444}),
    }
    protocols = calibration.get("protocols", {})
    for name, (count, expected_counts) in expected_protocols.items():
        protocol = protocols.get(name, {})
        consistency = protocol.get("frozen_3A_full_protein_consistency", {})
        if (
            protocol.get("n_structures") != count
            or consistency.get("matching_records") != count
            or consistency.get("mismatch_count") != 0
        ):
            errors.append(f"AF2 frozen assignment reproduction drift: {name}")
        observed_counts = (
            protocol.get("regions", {}).get("n_lobe_act", {})
            .get("ambiguity_at_frozen_threshold", {}).get("counts")
        )
        if observed_counts != expected_counts:
            errors.append(f"AF2 ambiguity counts drift: {name}/n_lobe_act")


def _check_synthetic(root: Path, errors: list[str]) -> None:
    base = root / "experiments" / "iclr_restructuring" / "results"
    artifact = _load_json(base / "p4_support_resolution_selection.json", errors)
    manifest = _load_json(base / "p4_support_resolution_selection_manifest.json", errors)
    if artifact is None or manifest is None:
        return
    if len(artifact.get("records", [])) != 72000:
        errors.append("synthetic artifact must contain exactly 72,000 records")
    if len(artifact.get("summary", [])) != 360:
        errors.append("synthetic artifact must contain exactly 360 setting summaries")
    if not (
        manifest.get("records") == 72000
        and manifest.get("settings") == 360
        and manifest.get("repeats_per_setting") == 200
        and manifest.get("all_settings_retained") is True
    ):
        errors.append("synthetic manifest factorial counts drift")
    if any(
        row.get("repeats") != 200
        or not row.get("all_nested_outer_fold_isolation")
        or not row.get("all_nested_inner_fold_isolation")
        for row in artifact.get("summary", [])
    ):
        errors.append("synthetic summary lacks fold-local isolation or 200 repeats")
    slices = artifact.get("figure_slices", {})
    if slices.get("panel_A", {}).get("fixed") != {"n": 20, "m": 1, "resolution": "full_k3"}:
        errors.append("synthetic panel A slice metadata drift")
    if slices.get("panel_B", {}).get("fixed") != {"n": 20, "m": 1}:
        errors.append("synthetic panel B slice metadata drift")
    if slices.get("panel_B", {}).get("averaged_over") != ["epsilon"]:
        errors.append("synthetic panel B averaging metadata drift")
    if slices.get("panel_C", {}).get("complete_factorial") is not True:
        errors.append("synthetic panel C must use the complete factorial")


def run_audit(root: Path) -> list[str]:
    """Return all active-submission claim errors under ``root``."""
    root = Path(root).resolve()
    errors: list[str] = []
    _check_manuscript(root, errors)
    _check_src_labels(root, errors)
    _check_src_clr_robustness(root, errors)
    _check_primary_gp(root, errors)
    _check_unified_robustness(root, errors)
    _check_af2(root, errors)
    _check_synthetic(root, errors)
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    errors = run_audit(root)
    if errors:
        print(f"SUBMISSION CLAIMS AUDIT: FAIL ({len(errors)} errors)")
        for error in errors:
            print(f"FAIL  {error}")
        return 1
    print("SUBMISSION CLAIMS AUDIT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
