import csv
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from audit_submission_claims import HYBRID_PROTOCOL_ID, PRIMARY_GP, PRIMARY_PROTOCOL_ID, run_audit


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_fixture(root):
    paper = root / "paper"
    paper.mkdir(parents=True)
    paper.joinpath("main_v3.tex").write_text(
        "retrospectively fixed; finite-panel paired sensitivity; overlapping LOO; "
        "site-related mutation families; Target ties; target non-ties; assigned fractions; "
        "not a second complete global-fit panel; full 10-row MAE rankings are not invariant; "
        "49/50; does not substitute for independent redigitization; without refitting; "
        "the two routes are not pooled; 451/840; 39/480; "
        "GP hyperparameters fitted on training folds only; "
        "not directly comparable numerically with the fixed-MLP or nested-MLP rows; "
        "the candidate set, model set, and tie-break differ; "
        "is not the confirmatory route's $0.3700$ estimate; "
         "equal-prediction reference for a single shared prediction is $0.1475$; "
         "not a lower bound on this two-fit LOO statistic; 12,000 generated datasets; "
         "candidate set, however, was assembled retrospectively; "
         "in the orthonormal basis, the $q_2$ (E1--E2) MAE exceeds; "
         "50/50 rows (raw $u_2>u_1$: 49/50); 90.5\\% (raw: 60.5\\%); "
         "feature-near graph was fixed; count of thresholded conflicts; not sufficient certification; "
         "falls back to the first $N$ ordered C$\\alpha$ atoms; alignment-mode audit; "
         "within-fold resampling audit; FoldX mutation free energies; "
         "failed-QC exploratory evidence; predeclared quality-control gates; "
         "0.0956; 0.2396; heterogeneous-candidate stress test; not a one-factor ablation; "
         "0.0157; 0.0198; per-fold oracle; CLR-GP (T5 diagnostic); CLR-GP (primary); "
        "MLP (primary probe), $K=3$ & 0.319 & $-1.21$ & 0.484 & $-0.97$; "
        "predeclared success; deletion robustness; "
        "audit detection benchmark; nomological network; "
        "benchmarking-epistemology; calibrate the audit; "
        "$q_2>q_1$ detection rate; "
        "\\ref{fig:workflow}; \\ref{fig:resolution}; \\ref{fig:alignment}; "
        "\\ref{tab:model-defs}; \\ref{tab:unified-robustness}; "
        "\\ref{tab:per-mut-errors}\n"
        "Retrospective fixed-panel & Abl1 & row 1\n"
        "Retrospective fixed-panel & Abl1 & row 2\n"
        "Retrospective fixed-panel & Src & row 3\n"
        "Retrospective fixed-panel & Src & row 4\n",
        encoding="utf-8",
    )
    paper.joinpath("generate_v2_figures.py").write_text(
        'AF2 = "af2_plddt_raw_pdb.json"', encoding="utf-8"
    )

    csv_path = root / "data" / "nmr_populations" / "src_k3_canonical.csv"
    csv_path.parent.mkdir(parents=True)
    fields = ["record_id", "mutation_id", "used_in_primary", "panel_order"]
    rows = [{
        "record_id": "figs5_met305__SrcKD-WT",
        "mutation_id": "SrcKD-WT",
        "used_in_primary": "true",
        "panel_order": "0",
    }]
    mutations = ["L410A", "V332I", "L270F_V332I", "L325A", "A311I", "V380A", "V331A", "F405A"]
    for index, mutation in enumerate(mutations, 1):
        rows.append({
            "record_id": f"figs5_met305__SrcKD-{mutation}",
            "mutation_id": f"SrcKD-{mutation}",
            "used_in_primary": "true",
            "panel_order": str(index),
        })
    rows.append({
        "record_id": "table_s2_global__SrcKD-L410A",
        "mutation_id": "SrcKD-L410A",
        "used_in_primary": "false",
        "panel_order": "",
    })
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    canonical_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    target_ids = {
        row["mutation_id"]: row["record_id"] for row in rows[1:9]
    }
    hybrid_ids = dict(target_ids)
    hybrid_ids["SrcKD-L410A"] = "table_s2_global__SrcKD-L410A"
    sensitivity = {
        "protocols": {
            "primary_probe": {
                "protocol_id": PRIMARY_PROTOCOL_ID,
                "protocol_kind": "primary_probe",
                "wt_record_id": rows[0]["record_id"],
                "target_record_ids": target_ids,
                "substitutions": [],
                "canonical_sha256": canonical_hash,
            },
            "l410a_global_fit_substitution": {
                "protocol_id": HYBRID_PROTOCOL_ID,
                "protocol_kind": "hybrid_single_substitution",
                "wt_record_id": rows[0]["record_id"],
                "target_record_ids": hybrid_ids,
                "substitutions": [{
                    "mutation_id": "SrcKD-L410A",
                    "from_record_id": target_ids["SrcKD-L410A"],
                    "to_record_id": "table_s2_global__SrcKD-L410A",
                }],
                "canonical_sha256": canonical_hash,
            },
        },
        "systems": {
            "primary_probe": {
                "fixed_k3": {
                    "pos::LowRankCDST": {
                        "u1_u2_contrast": {
                            "u1": 0.5206792055330889,
                            "u2": 0.6845696578954912,
                        },
                    },
                },
            },
            "l410a_global_fit_substitution": {},
        },
        "battery": {"gp_protocol": PRIMARY_GP},
    }
    results = root / "experiments" / "iclr_restructuring" / "results"
    followup_source = root / "experiments" / "iclr_restructuring" / "k3_followup.py"
    followup_source.parent.mkdir(parents=True, exist_ok=True)
    followup_source.write_text(
        "from gp_protocols import make_primary_gp\n"
        "MODEL = 'GPR-probability-diagnostic'\n"
        "def build():\n    return make_primary_gp()\n",
        encoding="utf-8",
    )
    write_json(results / "p2_k3_src_label_sensitivity.json", sensitivity)
    robustness_rows = []
    for pseudocount in [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]:
        for candidate in ["pos", "ext", "llr_pos", "llr_only", "pca20"]:
            for model in ["CLR-Ridge", "CLR-GP"]:
                robustness_rows.append({
                    "pseudocount": pseudocount,
                    "candidate": candidate,
                    "model": model,
                    "u2_gt_u1": not (
                        pseudocount == 1e-2 and candidate == "pca20" and model == "CLR-GP"
                    ),
                    "f405a_gt_l410a": True,
                })
    interval_records = [{
        "gp_mae_le_ridge": index < 20,
        "both_u2_gt_u1": index < 121,
        "both_f405a_gt_l410a": index < 84,
    } for index in range(200)]
    write_json(
        results / "p2_k3_src_clr_robustness.json",
        {
            "schema_version": "src_clr_label_robustness_v1",
            "primary_protocol_id": PRIMARY_PROTOCOL_ID,
            "canonical_sha256": canonical_hash,
            "gp_protocol": PRIMARY_GP,
            "pseudocount_sensitivity": {
                "pseudocounts": [1e-8, 1e-6, 1e-4, 1e-3, 1e-2],
                "rows": robustness_rows,
                "summary": {
                    "mae_ranking_stable": False,
                    "all_rows_u2_gt_u1": False,
                    "f405a_vs_l410a_pattern_stable": True,
                },
            },
            "digitization_interval_stress_test": {
                "status": "curator_interval_stress_test_not_independent_redigitization",
                "realizations": 200,
                "proportions": {
                    "gp_mae_le_ridge": 0.10,
                    "both_u2_gt_u1": 0.605,
                    "both_f405a_gt_l410a": 0.42,
                },
                "records": interval_records,
                "limitation": "not a substitute for independent redigitization",
            },
        },
    )
    write_json(results / "p1_core_baselines.json", {"_protocols": {"CLR-GP": PRIMARY_GP}})
    paired_source = results / "p2_k3_paired_exact.json"
    confirmatory_source = results / "p2_k3_nested_results.json"
    write_json(paired_source, {"fixture": "paired"})
    write_json(confirmatory_source, {"fixture": "confirmatory"})
    # Required since 2026-08-17: the audit no longer silently skips the
    # nested-PCA GP/scaling checks when this artifact is absent.
    write_json(
        results / "p2_k3_nested_pca_results.json",
        {
            "gp_protocol": PRIMARY_GP,
            "systems": {
                "abl1": {"protocol": {
                    "scaling_primary": "none (raw features; matches frozen fixed rows)",
                    "scaling_secondary": "fold-local StandardScaler for all candidates"}},
                "src": {"protocol": {
                    "scaling_primary": "none (raw features; matches frozen fixed rows)",
                    "scaling_secondary": "fold-local StandardScaler for all candidates"}},
            },
        },
    )
    unified_systems = {
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
    routes = {}
    for route_name, systems in unified_systems.items():
        routes[route_name] = {"systems": {}}
        for system, values in systems.items():
            routes[route_name]["systems"][system] = {
                "n_mutations": values[0],
                "all_mutation_mae": values[1],
                "leave_one_observation_out": {
                    "definition": "no model or selector is refit",
                    "minimum_mae": values[2],
                    "maximum_mae": values[3],
                },
                "double_mutant_exclusion": {"mae": values[4]},
            }
    lso = {
        "abl1": {
            "F382_family": {"group_mae": 0.26724303855555553,
                             "comparator": 0.27185185185185184},
            "290_301": {"group_mae": 0.3323498596666667,
                         "comparator": 0.14296296296296296},
        },
        "src": {
            "N_lobe": {"group_mae": 0.35370970780000005,
                        "comparator": 0.33466666666666667},
            "C_lobe": {"group_mae": 0.4332842592222222,
                        "comparator": 0.3591111111111111},
        },
    }
    for system, groups in lso.items():
        routes["candidate_model_confirmatory"]["systems"][system][
            "leave_site_out_retraining"
        ] = {"definition": "the confirmatory route is retrained", "groups": groups}
    write_json(
        results / "p2_k3_unified_robustness.json",
        {
            "schema_version": "p2_k3_unified_robustness_v1",
            "protocol": {
                "biological_unit": "held-out mutation",
                "descriptive_deletion_refits_model": False,
                "leave_site_out_refits_confirmatory_route": True,
                "routes_must_remain_separate": True,
            },
            "source_hashes": {
                "p2_k3_paired_exact_sha256": hashlib.sha256(
                    paired_source.read_bytes()
                ).hexdigest(),
                "p2_k3_nested_results_sha256": hashlib.sha256(
                    confirmatory_source.read_bytes()
                ).hexdigest(),
            },
            "routes": routes,
        },
    )
    write_json(
        root / "experiments" / "af2_subsample" / "results" / "af2_plddt_raw_pdb.json",
        {
            "schema_version": "af2_plddt_raw_pdb_v1",
            "protocols": {
                "original": {"found_structures": 840, "mutant_mean_range": [80.96, 81.73]},
                "fresh_msa": {"found_structures": 480, "mutant_mean_range": [54.57, 57.35]},
            },
        },
    )
    write_json(
        root / "experiments" / "af2_subsample" / "results"
        / "assignment_calibration.json",
        {
            "schema_version": "af2_assignment_calibration_v1",
            "method": {"thresholds_angstrom": [round(2.0 + 0.25 * i, 2)
                                                for i in range(13)]},
            "ambiguity_rule": {
                "margin_cutoff_angstrom": 0.5,
                "interpretation": "diagnostic assignment uncertainty; not a thermodynamic population",
            },
            "reference_calibration": {
                "full_protein": {"pairwise_distances": {
                    "active__I2": {"rmsd_angstrom": 8.893687880114713}}},
                "n_lobe_act": {"pairwise_distances": {
                    "active__I2": {"rmsd_angstrom": 7.326877582309726}}},
                "alphaC_only": {"pairwise_distances": {
                    "active__I1": {"rmsd_angstrom": 1.5686527987774919},
                    "active__I2": {"rmsd_angstrom": 2.564432345596726},
                }},
            },
            "protocols": {
                "original": {
                    "n_structures": 840,
                    "frozen_3A_full_protein_consistency": {
                        "matching_records": 840, "mismatch_count": 0},
                    "regions": {"n_lobe_act": {"ambiguity_at_frozen_threshold": {
                        "counts": {"active": 674, "I1": 0, "I2": 0,
                                   "ambiguous": 0, "unclassified": 166}}}},
                },
                "fresh_msa": {
                    "n_structures": 480,
                    "frozen_3A_full_protein_consistency": {
                        "matching_records": 480, "mismatch_count": 0},
                    "regions": {"n_lobe_act": {"ambiguity_at_frozen_threshold": {
                        "counts": {"active": 36, "I1": 0, "I2": 0,
                                   "ambiguous": 0, "unclassified": 444}}}},
                },
            },
        },
    )
    summary = [{
        "repeats": 200,
        "all_nested_outer_fold_isolation": True,
        "all_nested_inner_fold_isolation": True,
    }] * 360
    write_json(
        results / "p4_support_resolution_selection.json",
        {
            "records": [0] * 72000,
            "summary": summary,
            "figure_slices": {
                "panel_A": {"fixed": {"n": 20, "m": 1, "resolution": "full_k3"}},
                "panel_B": {"fixed": {"n": 20, "m": 1}, "averaged_over": ["epsilon"]},
                "panel_C": {"complete_factorial": True},
            },
        },
    )
    write_json(
        results / "p4_support_resolution_selection_manifest.json",
        {
            "records": 72000,
            "settings": 360,
            "repeats_per_setting": 200,
            "all_settings_retained": True,
        },
    )


def test_clean_fixture_passes(tmp_path):
    make_fixture(tmp_path)
    assert run_audit(tmp_path) == []


def test_stale_claim_and_protocol_drift_fail(tmp_path):
    make_fixture(tmp_path)
    manuscript = tmp_path / "paper" / "main_v3.tex"
    manuscript.write_text(
        manuscript.read_text(encoding="utf-8")
        + " prespecified PCA 8/8 MLP, $K=3$ & 0.501 \\label{app:additional}",
        encoding="utf-8",
    )
    sensitivity_path = (
        tmp_path / "experiments" / "iclr_restructuring" / "results"
        / "p2_k3_src_label_sensitivity.json"
    )
    sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    sensitivity["protocols"]["l410a_global_fit_substitution"]["wt_record_id"] = "table_s2_global__SrcKD-WT"
    sensitivity["battery"]["gp_protocol"]["alpha"] = 0.01
    sensitivity["systems"]["primary_probe"]["fixed_k3"]["pos::LowRankCDST"][
        "u1_u2_contrast"
    ]["u1"] = 0.50
    write_json(sensitivity_path, sensitivity)
    robustness_path = sensitivity_path.with_name("p2_k3_src_clr_robustness.json")
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    robustness["pseudocount_sensitivity"]["summary"]["mae_ranking_stable"] = True
    write_json(robustness_path, robustness)
    unified_path = robustness_path.with_name("p2_k3_unified_robustness.json")
    unified = json.loads(unified_path.read_text(encoding="utf-8"))
    unified["protocol"]["routes_must_remain_separate"] = False
    write_json(unified_path, unified)
    calibration_path = (
        tmp_path / "experiments" / "af2_subsample" / "results"
        / "assignment_calibration.json"
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["ambiguity_rule"]["margin_cutoff_angstrom"] = 0.4
    write_json(calibration_path, calibration)
    errors = run_audit(tmp_path)
    assert any("prespecified" in error for error in errors)
    assert any("8/8" in error for error in errors)
    assert any("stale Src Position/MLP" in error for error in errors)
    assert any("duplicate Appendix D label" in error for error in errors)
    assert any("retain the primary probe WT" in error for error in errors)
    assert any("GP alpha drift" in error for error in errors)
    assert any("current Position/MLP contrast values drift" in error for error in errors)
    assert any("pseudocount stability conclusion drift" in error for error in errors)
    assert any("unified K=3 robustness protocol drift" in error for error in errors)
    assert any("AF2 assignment ambiguity rule drift" in error for error in errors)
