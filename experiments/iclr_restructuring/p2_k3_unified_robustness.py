"""Summarize mutation- and site-level robustness across frozen K=3 routes.

This script does not refit either model. It reports descriptive deletion
summaries from frozen outer-fold errors and keeps the confirmatory route's
existing leave-site-out retraining results in a separate field.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
PAIRED_SOURCE = RESULTS / "p2_k3_paired_exact.json"
CONFIRMATORY_SOURCE = RESULTS / "p2_k3_nested_results.json"
OUT_JSON = RESULTS / "p2_k3_unified_robustness.json"
OUT_MD = RESULTS / "p2_k3_unified_robustness_report.md"

DOUBLE_MUTANTS = {
    "abl1": "M290L_L301I",
    "src": "SrcKD-L270F_V332I",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_frozen_errors(errors: dict[str, float], double_mutant: str) -> dict:
    """Return descriptive deletion summaries without refitting a model."""
    if len(errors) < 2:
        raise ValueError("At least two mutation errors are required")
    if double_mutant not in errors:
        raise ValueError(f"Missing declared double mutant: {double_mutant}")

    normalized = {name: float(value) for name, value in errors.items()}
    all_mae = sum(normalized.values()) / len(normalized)
    leave_one_out = {
        name: sum(value for other, value in normalized.items() if other != name)
        / (len(normalized) - 1)
        for name in normalized
    }
    minimum_mutation = min(leave_one_out, key=leave_one_out.get)
    maximum_mutation = max(leave_one_out, key=leave_one_out.get)
    without_double = [
        value for name, value in normalized.items() if name != double_mutant
    ]

    return {
        "n_mutations": len(normalized),
        "all_mutation_mae": all_mae,
        "leave_one_observation_out": {
            "definition": (
                "mean of the already frozen outer-fold errors after deleting one "
                "held-out mutation; no model or selector is refit"
            ),
            "mae_by_deleted_mutation": leave_one_out,
            "minimum_mae": leave_one_out[minimum_mutation],
            "minimum_when_deleted": minimum_mutation,
            "maximum_mae": leave_one_out[maximum_mutation],
            "maximum_when_deleted": maximum_mutation,
        },
        "double_mutant_exclusion": {
            "excluded_mutation": double_mutant,
            "n_remaining": len(without_double),
            "mae": sum(without_double) / len(without_double),
            "definition": (
                "mean of frozen outer-fold errors after deleting the declared double "
                "mutant; no model or selector is refit"
            ),
        },
    }


def build_result(paired: dict, confirmatory: dict) -> dict:
    routes = {
        "representation_selection_audit": {
            "label": "full nested MLP representation selector",
            "source": "p2_k3_paired_exact.json",
            "interpretation": (
                "representation choice is nested inside each outer fold; candidate model "
                "family is fixed to the MLP"
            ),
            "systems": {},
        },
        "candidate_model_confirmatory": {
            "label": "nested candidate-model confirmatory route",
            "source": "p2_k3_nested_results.json",
            "interpretation": (
                "candidate representation and model family are selected inside each outer "
                "fold under the confirmatory protocol"
            ),
            "systems": {},
        },
    }

    for system in ("abl1", "src"):
        paired_rows = paired["systems"][system]["per_mutation"]
        paired_errors = {
            row["mutation"]: row["mae_nested_mlp"] for row in paired_rows
        }
        routes["representation_selection_audit"]["systems"][system] = (
            summarize_frozen_errors(paired_errors, DOUBLE_MUTANTS[system])
        )

        confirmatory_system = confirmatory["systems"][system]
        confirmatory_errors = confirmatory_system["metrics"]["mae_per_mutant"]
        summary = summarize_frozen_errors(
            confirmatory_errors, DOUBLE_MUTANTS[system]
        )
        summary["all_mutation_mae"] = float(confirmatory_system["metrics"]["mae"])
        summary["mean_of_display_precision_mutation_errors"] = (
            sum(float(value) for value in confirmatory_errors.values())
            / len(confirmatory_errors)
        )
        summary["leave_site_out_retraining"] = {
            "definition": (
                "the entire mutation family is held out and the confirmatory route is "
                "retrained; these values are not row-deletion summaries"
            ),
            "groups": confirmatory_system["metrics"]["leave_site_out"],
        }
        routes["candidate_model_confirmatory"]["systems"][system] = summary

    return {
        "schema_version": "p2_k3_unified_robustness_v1",
        "purpose": (
            "place two frozen K=3 evaluation routes in one robustness artifact without "
            "pooling or comparing their scores as if they were one estimator"
        ),
        "protocol": {
            "biological_unit": "held-out mutation",
            "descriptive_deletion_refits_model": False,
            "leave_site_out_refits_confirmatory_route": True,
            "routes_must_remain_separate": True,
        },
        "source_hashes": {
            "p2_k3_paired_exact_sha256": sha256_file(PAIRED_SOURCE),
            "p2_k3_nested_results_sha256": sha256_file(CONFIRMATORY_SOURCE),
        },
        "routes": routes,
    }


def write_report(result: dict) -> None:
    lines = [
        "# Unified K=3 robustness summary",
        "",
        "> The two routes answer different selection questions and are not pooled. "
        "Leave-one-observation and double-mutant exclusions summarize frozen outer-fold "
        "errors without refitting. Leave-site-out values are separate confirmatory "
        "retraining results.",
        "",
        "| Route | System | All MAE | Delete-one range | Double-mutant excluded |",
        "|---|---|---:|---:|---:|",
    ]
    for route in result["routes"].values():
        for system in ("abl1", "src"):
            summary = route["systems"][system]
            deletion = summary["leave_one_observation_out"]
            double = summary["double_mutant_exclusion"]
            lines.append(
                f"| {route['label']} | {system.capitalize()} | "
                f"{summary['all_mutation_mae']:.4f} | "
                f"{deletion['minimum_mae']:.4f}--{deletion['maximum_mae']:.4f} | "
                f"{double['mae']:.4f} |"
            )

    lines.extend([
        "",
        "## Confirmatory leave-site-out retraining",
        "",
        "| System | Held-out family | Route MAE | Training-mean comparator |",
        "|---|---|---:|---:|",
    ])
    confirmatory = result["routes"]["candidate_model_confirmatory"]["systems"]
    for system in ("abl1", "src"):
        groups = confirmatory[system]["leave_site_out_retraining"]["groups"]
        for name, row in groups.items():
            lines.append(
                f"| {system.capitalize()} | {name} | {row['group_mae']:.4f} | "
                f"{row['comparator']:.4f} |"
            )

    lines.extend([
        "",
        "The deletion ranges diagnose concentration in individual frozen test errors. "
        "They are not cross-validation estimates for a refitted procedure. The "
        "leave-site-out rows change the training set and therefore address transfer to "
        "an unseen mutation family.",
        "",
    ])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paired = json.loads(PAIRED_SOURCE.read_text(encoding="utf-8"))
    confirmatory = json.loads(CONFIRMATORY_SOURCE.read_text(encoding="utf-8"))
    result = build_result(paired, confirmatory)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result)
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
