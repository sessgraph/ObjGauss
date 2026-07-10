from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from objgauss.evaluation.objectstate_controlled_real_readiness_audit import (
    objectstate_controlled_real_readiness_audit,
    validate_objectstate_controlled_real_readiness_audit,
)
from objgauss.datasets.objectstate_real_evidence_bundle import (
    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
    OBJECTSTATE_REAL_GATE_ACCOUNTING_ROW_SCHEMA,
    read_objectstate_real_evidence_bundle,
    validate_objectstate_real_evidence_bundle,
)
from objgauss.datasets.objectstate_teacher_evidence import (
    TEACHER_EVIDENCE_FORBIDDEN_PROVENANCE_KEYS,
)

OBJECTSTATE_CONTROLLED_REAL_IDENTITY_EVAL_SCHEMA = (
    "objgauss-objectstate-controlled-real-identity-eval-v1"
)
OBJECTSTATE_CONTROLLED_REAL_IDENTITY_TEACHER_EVIDENCE_SCHEMA = (
    "objgauss-objectstate-controlled-real-identity-teacher-evidence-v1"
)
OBJECTSTATE_CONTROLLED_REAL_IDENTITY_BASELINES = (
    "random_assignment",
    "xyz_centroid",
    "oracle_target_assignment",
    "assignment_solver_v2",
)

_READY_STATUSES = {"pass", "fail"}

__all__ = (
    "OBJECTSTATE_CONTROLLED_REAL_IDENTITY_EVAL_SCHEMA",
    "OBJECTSTATE_CONTROLLED_REAL_IDENTITY_TEACHER_EVIDENCE_SCHEMA",
    "OBJECTSTATE_CONTROLLED_REAL_IDENTITY_BASELINES",
    "objectstate_controlled_real_identity_eval_from_files",
    "read_objectstate_controlled_real_identity_teacher_evidence",
    "objectstate_controlled_real_identity_eval",
    "objectstate_controlled_real_identity_report",
    "objectstate_controlled_real_identity_accounting_csv",
    "objectstate_controlled_real_identity_pairwise_csv",
    "objectstate_controlled_real_identity_artifact_manifest",
    "validate_objectstate_controlled_real_identity_teacher_evidence",
    "validate_objectstate_controlled_real_identity_eval",
)


def objectstate_controlled_real_identity_eval_from_files(
    bundle_path: str | Path,
    *,
    teacher_evidence_path: str | Path | None = None,
    min_identity_retrieval_at_1: float = 0.75,
    seed: int = 0,
) -> dict[str, Any]:
    bundle = read_objectstate_real_evidence_bundle(bundle_path)
    teacher_evidence = (
        None
        if teacher_evidence_path is None
        else read_objectstate_controlled_real_identity_teacher_evidence(
            teacher_evidence_path
        )
    )
    return objectstate_controlled_real_identity_eval(
        bundle,
        teacher_evidence=teacher_evidence,
        bundle_ref=str(bundle_path),
        teacher_evidence_ref=(
            None if teacher_evidence_path is None else str(teacher_evidence_path)
        ),
        min_identity_retrieval_at_1=min_identity_retrieval_at_1,
        seed=seed,
    )


def read_objectstate_controlled_real_identity_teacher_evidence(
    path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("controlled real identity teacher evidence JSON must be an object")
    return validate_objectstate_controlled_real_identity_teacher_evidence(payload)


def objectstate_controlled_real_identity_eval(
    bundle: Mapping[str, Any],
    *,
    teacher_evidence: Mapping[str, Any] | None = None,
    bundle_ref: str | None = None,
    teacher_evidence_ref: str | None = None,
    min_identity_retrieval_at_1: float = 0.75,
    seed: int = 0,
) -> dict[str, Any]:
    checked = validate_objectstate_real_evidence_bundle(bundle)
    readiness = objectstate_controlled_real_readiness_audit(checked, bundle_ref=bundle_ref)
    teacher = (
        None
        if teacher_evidence is None
        else validate_objectstate_controlled_real_identity_teacher_evidence(
            teacher_evidence
        )
    )
    if teacher is not None and teacher["sample_id"] != checked["sample"]["sample_id"]:
        raise ValueError("controlled real identity teacher evidence sample_id mismatch")
    accounting_rows = [
        row for row in checked["gate_accounting_rows"] if row["evidence_kind"] == "identity"
    ]
    indexes = _indexes(checked)
    teacher_assignments = {} if teacher is None else _teacher_assignment_map(teacher)
    row_results = []
    evaluated_accounting_rows = []
    pairwise_rows: list[dict[str, Any]] = []
    matching_rows: list[dict[str, Any]] = []
    for row in accounting_rows:
        result = _evaluate_identity_accounting_row(
            row,
            checked,
            indexes,
            readiness,
            teacher,
            teacher_assignments,
            min_identity_retrieval_at_1=float(min_identity_retrieval_at_1),
            seed=int(seed),
        )
        row_results.append(result)
        evaluated_accounting_rows.append(result["real_bundle_accounting_row"])
        pairwise_rows.extend(result["pairwise_distances"])
        matching_rows.extend(result["matching"])
    evaluated_bundle = _bundle_with_identity_accounting(checked, evaluated_accounting_rows)
    counts = {
        "identity_rows": len(row_results),
        "evaluated_identity_rows": sum(
            1 for row in row_results if row["eval_status"] in _READY_STATUSES
        ),
        "identity_ready_rows_consumed": sum(
            1 for row in row_results if row["readiness"]["identity_ready"]
        ),
        "identity_rows_evidence_incomplete": sum(
            1 for row in row_results if row["eval_status"] == "evidence_incomplete"
        ),
        "identity_rows_blocked": sum(
            1 for row in row_results if row["eval_status"] == "blocked"
        ),
        "identity_pass_rows": sum(1 for row in row_results if row["eval_status"] == "pass"),
        "identity_fail_rows": sum(1 for row in row_results if row["eval_status"] == "fail"),
    }
    aggregate = _aggregate_metrics(row_results)
    pass_gates = _pass_gates(row_results, aggregate)
    status = _summary_status(row_results, pass_gates)
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_REAL_IDENTITY_EVAL_SCHEMA,
        "kind": "objectstate_controlled_real_identity_eval",
        "status": status,
        "bundle_schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
        "teacher_evidence_schema": (
            None
            if teacher is None
            else OBJECTSTATE_CONTROLLED_REAL_IDENTITY_TEACHER_EVIDENCE_SCHEMA
        ),
        "bundle_ref": bundle_ref,
        "teacher_evidence_ref": teacher_evidence_ref,
        "sample": dict(checked["sample"]),
        "thresholds": {
            "min_identity_retrieval_at_1": float(min_identity_retrieval_at_1),
            "require_oracle_not_below_solver": True,
            "require_solver_not_below_random": True,
        },
        "teacher_evidence": _teacher_summary(teacher),
        "readiness_audit": readiness,
        "row_counts": counts,
        "teacher_evidence_coverage": aggregate["teacher_evidence_coverage"],
        "metrics": aggregate["metrics"],
        "baselines": aggregate["baselines"],
        "baseline_comparison": aggregate["baseline_comparison"],
        "pass_gates": pass_gates,
        "identity_accounting_rows": row_results,
        "evaluated_real_bundle": evaluated_bundle,
        "pairwise_distances": pairwise_rows,
        "matching": matching_rows,
        "issues": _issues(row_results, aggregate),
        "claim_policy": {
            "uses_real_bundle_identity_gt_for_evaluation_only": True,
            "teacher_evidence_is_not_physical_identity": True,
            "missing_teacher_evidence_blocks_not_fails": True,
            "evidence_incomplete_is_not_model_fail": True,
            "writes_identity_accounting": True,
            "prediction_rows_out_of_scope": True,
            "intervention_rows_out_of_scope": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "runs_teacher_model": False,
            "reconstructs_gaussians": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "runs_prediction_eval": False,
            "runs_intervention_eval": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_controlled_real_identity_eval(payload)


def objectstate_controlled_real_identity_report(summary: Mapping[str, Any]) -> str:
    checked = validate_objectstate_controlled_real_identity_eval(summary)
    lines = [
        f"# Controlled Real Identity Eval: {checked['sample']['sample_id']}",
        "",
        f"- status: `{checked['status']}`",
        f"- evaluated_identity_rows: `{checked['row_counts']['evaluated_identity_rows']}`",
        f"- identity_pass_rows: `{checked['row_counts']['identity_pass_rows']}`",
        f"- identity_fail_rows: `{checked['row_counts']['identity_fail_rows']}`",
        f"- identity_rows_blocked: `{checked['row_counts']['identity_rows_blocked']}`",
        f"- identity_rows_evidence_incomplete: `{checked['row_counts']['identity_rows_evidence_incomplete']}`",
        f"- teacher_evidence_coverage: `{checked['teacher_evidence_coverage']:.6f}`",
        "",
        "## Metrics",
        "",
    ]
    for key, value in checked["metrics"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(("", "## Rows", ""))
    for row in checked["identity_accounting_rows"]:
        lines.append(
            "- "
            f"`{row['row_id']}` "
            f"status={row['eval_status']} "
            f"reason={row.get('reason') or 'none'}"
        )
    lines.append("")
    return "\n".join(lines)


def objectstate_controlled_real_identity_accounting_csv(
    summary: Mapping[str, Any],
) -> str:
    checked = validate_objectstate_controlled_real_identity_eval(summary)
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "row_id",
            "eval_status",
            "bundle_accounting_status",
            "object_id",
            "teacher_evidence_coverage",
            "identity_retrieval_at_1",
            "identity_margin",
            "slot_swap_rate",
            "objectstate_drift",
            "assignment_consistency",
            "occlusion_recovery",
            "reason",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for row in checked["identity_accounting_rows"]:
        metrics = row.get("metrics", {})
        writer.writerow(
            {
                "row_id": row["row_id"],
                "eval_status": row["eval_status"],
                "bundle_accounting_status": row["real_bundle_accounting_row"][
                    "accounting_status"
                ],
                "object_id": row.get("object_id", ""),
                "teacher_evidence_coverage": row.get("teacher_evidence_coverage", 0.0),
                "identity_retrieval_at_1": metrics.get("identity_retrieval_at_1", ""),
                "identity_margin": metrics.get("identity_margin", ""),
                "slot_swap_rate": metrics.get("slot_swap_rate", ""),
                "objectstate_drift": metrics.get("objectstate_drift", ""),
                "assignment_consistency": metrics.get("assignment_consistency", ""),
                "occlusion_recovery": metrics.get("occlusion_recovery", ""),
                "reason": row.get("reason", ""),
            }
        )
    return output.getvalue()


def objectstate_controlled_real_identity_pairwise_csv(
    summary: Mapping[str, Any],
) -> str:
    checked = validate_objectstate_controlled_real_identity_eval(summary)
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "row_id_a",
            "row_id_b",
            "same_identity",
            "candidate_distance",
            "oracle_distance",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for row in checked["pairwise_distances"]:
        writer.writerow(row)
    return output.getvalue()


def objectstate_controlled_real_identity_artifact_manifest(
    summary: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    checked = validate_objectstate_controlled_real_identity_eval(summary)
    root = Path(output_dir)
    payload = {
        "schema": "objgauss-objectstate-controlled-real-identity-artifact-manifest-v1",
        "kind": "objectstate_controlled_real_identity_artifact_manifest",
        "sample_id": checked["sample"]["sample_id"],
        "identity_eval_schema": OBJECTSTATE_CONTROLLED_REAL_IDENTITY_EVAL_SCHEMA,
        "artifacts": {
            "summary": str(root / "controlled-real-identity-summary.json"),
            "report": str(root / "controlled-real-identity-report.md"),
            "accounting_csv": str(root / "controlled-real-identity-accounting.csv"),
            "matching": str(root / "controlled-real-identity-matching.json"),
            "pairwise_distances_csv": str(root / "controlled-real-identity-pairwise-distances.csv"),
            "evaluated_real_bundle": str(root / "controlled-real-identity-bundle.json"),
        },
        "claim_policy": {
            "artifacts_are_local_evaluation_outputs": True,
            "does_not_claim_reality_gate_pass": True,
        },
    }
    return payload


def validate_objectstate_controlled_real_identity_teacher_evidence(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled real identity teacher evidence must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_REAL_IDENTITY_TEACHER_EVIDENCE_SCHEMA:
        raise ValueError(
            "unsupported controlled real identity teacher evidence schema: "
            f"{payload.get('schema')}"
        )
    sample_id = _required_str(payload, "sample_id")
    source = _required_str(payload, "teacher_evidence_source")
    if source in {"oracle", "oracle_target_assignment", "physical_identity"}:
        raise ValueError("teacher evidence source cannot be oracle or physical identity")
    allowed_for_evaluation = payload.get("allowed_for_evaluation", True)
    if allowed_for_evaluation is not True:
        raise ValueError("teacher evidence must be allowed for evaluation")
    provenance = _mapping(payload.get("provenance", {}), "provenance")
    forbidden = [
        key for key in TEACHER_EVIDENCE_FORBIDDEN_PROVENANCE_KEYS if key in provenance
    ]
    if forbidden:
        raise ValueError(
            "teacher evidence provenance contains forbidden GT leakage keys: "
            + ", ".join(forbidden)
        )
    assignments = [
        _validate_teacher_assignment(item)
        for item in _sequence(payload.get("assignments"), "assignments")
    ]
    if not assignments:
        raise ValueError("teacher evidence requires assignments")
    return {
        "schema": OBJECTSTATE_CONTROLLED_REAL_IDENTITY_TEACHER_EVIDENCE_SCHEMA,
        "sample_id": sample_id,
        "teacher_evidence_source": source,
        "evidence_policy": str(payload.get("evidence_policy", "semantic")),
        "allowed_for_evaluation": True,
        "provenance": dict(provenance),
        "assignments": assignments,
    }


def validate_objectstate_controlled_real_identity_eval(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled real identity eval must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_REAL_IDENTITY_EVAL_SCHEMA:
        raise ValueError(
            "unsupported controlled real identity eval schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_real_identity_eval":
        raise ValueError("controlled real identity eval kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_real_identity_eval_pass",
        "objectstate_controlled_real_identity_eval_fail",
        "objectstate_controlled_real_identity_eval_blocked",
        "objectstate_controlled_real_identity_eval_incomplete",
    }:
        raise ValueError("controlled real identity eval status is unsupported")
    if payload.get("bundle_schema") != OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA:
        raise ValueError("controlled real identity eval bundle schema mismatch")
    if payload.get("teacher_evidence_schema") not in {
        None,
        OBJECTSTATE_CONTROLLED_REAL_IDENTITY_TEACHER_EVIDENCE_SCHEMA,
    }:
        raise ValueError("controlled real identity eval teacher schema mismatch")
    validate_objectstate_controlled_real_readiness_audit(payload.get("readiness_audit"))
    row_counts = _mapping(payload.get("row_counts"), "row_counts")
    for key in (
        "identity_rows",
        "evaluated_identity_rows",
        "identity_ready_rows_consumed",
        "identity_rows_evidence_incomplete",
        "identity_rows_blocked",
        "identity_pass_rows",
        "identity_fail_rows",
    ):
        _non_negative_int(row_counts.get(key), f"row_counts.{key}")
    _metric(payload.get("teacher_evidence_coverage"), "teacher_evidence_coverage")
    metrics = _mapping(payload.get("metrics"), "metrics")
    for key in (
        "identity_retrieval_at_1",
        "identity_margin",
        "slot_swap_rate",
        "objectstate_drift",
        "assignment_consistency",
        "occlusion_recovery",
    ):
        _metric(metrics.get(key), f"metrics.{key}")
    baselines = _mapping(payload.get("baselines"), "baselines")
    for name in OBJECTSTATE_CONTROLLED_REAL_IDENTITY_BASELINES:
        if name not in baselines:
            raise ValueError(f"controlled real identity eval missing baseline {name}")
    rows = _sequence(payload.get("identity_accounting_rows"), "identity_accounting_rows")
    if len(rows) != row_counts["identity_rows"]:
        raise ValueError("controlled real identity eval row count mismatch")
    for row in rows:
        _validate_eval_row(row)
    validate_objectstate_real_evidence_bundle(payload.get("evaluated_real_bundle"))
    pass_gates = _mapping(payload.get("pass_gates"), "pass_gates")
    if any(not isinstance(value, bool) for value in pass_gates.values()):
        raise ValueError("controlled real identity pass_gates must be bool")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("controlled real identity eval requires issues")
    claim_policy = _mapping(payload.get("claim_policy"), "claim_policy")
    if (
        not claim_policy.get("uses_real_bundle_identity_gt_for_evaluation_only")
        or not claim_policy.get("teacher_evidence_is_not_physical_identity")
        or not claim_policy.get("missing_teacher_evidence_blocks_not_fails")
        or not claim_policy.get("evidence_incomplete_is_not_model_fail")
        or not claim_policy.get("writes_identity_accounting")
        or not claim_policy.get("prediction_rows_out_of_scope")
        or not claim_policy.get("intervention_rows_out_of_scope")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled real identity eval must preserve claim policy")
    non_goals = _mapping(payload.get("non_goals"), "non_goals")
    if any(bool(value) for value in non_goals.values()):
        raise ValueError("controlled real identity eval cannot claim non-goals")
    return dict(payload)


def _evaluate_identity_accounting_row(
    row: Mapping[str, Any],
    bundle: Mapping[str, Any],
    indexes: Mapping[str, Any],
    readiness: Mapping[str, Any],
    teacher: Mapping[str, Any] | None,
    teacher_assignments: Mapping[str, Mapping[str, Any]],
    *,
    min_identity_retrieval_at_1: float,
    seed: int,
) -> dict[str, Any]:
    readiness_row = _readiness_row(readiness, row["row_id"])
    pose_rows = _selected_pose_rows(bundle, row)
    base = {
        "row_id": str(row["row_id"]),
        "evidence_kind": "identity",
        "object_id": row.get("object_id"),
        "source_accounting_status": str(row["accounting_status"]),
        "readiness": {
            "identity_ready": bool(readiness_row and readiness_row["evaluator_ready"]),
            "blocked_reasons": [] if readiness_row is None else list(readiness_row["blocked_reasons"]),
        },
    }
    if not base["readiness"]["identity_ready"]:
        reason = _reason(base["readiness"]["blocked_reasons"], "missing_identity_link")
        return _blocked_eval_row(base, row, "evidence_incomplete", reason)
    if teacher is None:
        return _blocked_eval_row(base, row, "blocked", "missing_teacher_evidence")
    missing = [pose["row_id"] for pose in pose_rows if pose["row_id"] not in teacher_assignments]
    if missing:
        blocked = _blocked_eval_row(base, row, "blocked", "missing_teacher_evidence")
        blocked["missing_teacher_pose_rows"] = missing
        blocked["teacher_evidence_coverage"] = _coverage(pose_rows, teacher_assignments)
        return blocked
    labels = _physical_labels(pose_rows, indexes)
    positions = np.asarray(
        [pose["object_pose_6dof"]["position"] for pose in pose_rows],
        dtype=np.float64,
    )
    visibility = np.asarray([pose["object_visibility"] for pose in pose_rows], dtype=np.float64)
    candidate_slots = [str(teacher_assignments[pose["row_id"]]["slot_id"]) for pose in pose_rows]
    candidate_vectors = _assignment_vectors(pose_rows, teacher_assignments, candidate_slots)
    identity_count = max(1, len(set(labels)))
    baseline_inputs = {
        "random_assignment": _random_slots(len(pose_rows), identity_count, seed=seed),
        "xyz_centroid": _xyz_slots(positions, identity_count),
        "oracle_target_assignment": _oracle_slots(labels),
        "assignment_solver_v2": candidate_slots,
    }
    baseline_metrics = {
        name: _identity_metrics(
            labels,
            slots,
            positions=positions,
            visibility=visibility,
            vectors=(
                candidate_vectors
                if name == "assignment_solver_v2"
                else _slot_vectors(slots)
            ),
        )
        for name, slots in baseline_inputs.items()
    }
    metrics = baseline_metrics["assignment_solver_v2"]
    comparison = _baseline_comparison(baseline_metrics)
    pass_gates = {
        "evaluated_identity_rows_positive": True,
        "identity_retrieval_at_1_min": (
            metrics["identity_retrieval_at_1"] >= float(min_identity_retrieval_at_1)
        ),
        "oracle_not_below_assignment_solver_v2": (
            comparison["oracle_target_assignment_vs_assignment_solver_v2"] >= -1.0e-9
        ),
        "assignment_solver_v2_not_below_random": (
            comparison["assignment_solver_v2_vs_random_assignment"] >= -1.0e-9
        ),
        "teacher_evidence_coverage_complete": (
            _coverage(pose_rows, teacher_assignments) >= 1.0
        ),
    }
    passed = all(pass_gates.values())
    eval_status = "pass" if passed else "fail"
    real_accounting = _real_accounting_row(
        row,
        accounting_status=eval_status,
        metrics=metrics,
        reason=None if passed else "controlled real identity evaluator failed",
    )
    return {
        **base,
        "eval_status": eval_status,
        "teacher_evidence_source": teacher["teacher_evidence_source"],
        "teacher_evidence_coverage": _coverage(pose_rows, teacher_assignments),
        "metrics": metrics,
        "baselines": baseline_metrics,
        "baseline_comparison": comparison,
        "pass_gates": pass_gates,
        "reason": None if passed else "controlled real identity evaluator failed",
        "real_bundle_accounting_row": real_accounting,
        "pairwise_distances": _pairwise_rows(pose_rows, labels, candidate_vectors, baseline_inputs["oracle_target_assignment"]),
        "matching": _matching_rows(pose_rows, labels, candidate_slots),
    }


def _blocked_eval_row(
    base: Mapping[str, Any],
    source_row: Mapping[str, Any],
    eval_status: str,
    reason: str,
) -> dict[str, Any]:
    accounting_status = (
        "evidence_incomplete" if eval_status in {"blocked", "evidence_incomplete"} else eval_status
    )
    return {
        **dict(base),
        "eval_status": eval_status,
        "teacher_evidence_coverage": 0.0,
        "metrics": _empty_metrics(),
        "baselines": _empty_baselines(),
        "baseline_comparison": {
            "assignment_solver_v2_vs_random_assignment": 0.0,
            "assignment_solver_v2_vs_xyz_centroid": 0.0,
            "oracle_target_assignment_vs_assignment_solver_v2": 0.0,
            "identity_model_underperforms_xyz_centroid": False,
        },
        "pass_gates": {
            "evaluated_identity_rows_positive": False,
            "identity_retrieval_at_1_min": False,
            "oracle_not_below_assignment_solver_v2": False,
            "assignment_solver_v2_not_below_random": False,
            "teacher_evidence_coverage_complete": False,
        },
        "reason": reason,
        "real_bundle_accounting_row": _real_accounting_row(
            source_row,
            accounting_status=accounting_status,
            metrics={},
            reason=reason,
        ),
        "pairwise_distances": [],
        "matching": [],
    }


def _real_accounting_row(
    source_row: Mapping[str, Any],
    *,
    accounting_status: str,
    metrics: Mapping[str, Any],
    reason: str | None,
) -> dict[str, Any]:
    row = {
        "schema": OBJECTSTATE_REAL_GATE_ACCOUNTING_ROW_SCHEMA,
        "row_id": str(source_row["row_id"]),
        "evidence_kind": "identity",
        "accounting_status": accounting_status,
        "metrics": dict(metrics),
        "artifact_refs": list(source_row["artifact_refs"]),
        "gt_requirements": dict(source_row["gt_requirements"]),
    }
    if source_row.get("object_id"):
        row["object_id"] = str(source_row["object_id"])
    if reason:
        row["reason"] = reason
    return row


def _bundle_with_identity_accounting(
    bundle: Mapping[str, Any],
    identity_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    replacements = {row["row_id"]: row for row in identity_rows}
    rows = [
        dict(replacements.get(row["row_id"], row))
        for row in bundle["gate_accounting_rows"]
    ]
    updated = {
        "schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
        "kind": "objectstate_real_evidence_bundle",
        "sample": dict(bundle["sample"]),
        "row_schemas": dict(bundle["row_schemas"]),
        "observation_rows": list(bundle["observation_rows"]),
        "object_pose_rows": list(bundle["object_pose_rows"]),
        "identity_link_rows": list(bundle["identity_link_rows"]),
        "action_interval_rows": list(bundle["action_interval_rows"]),
        "state_transition_rows": list(bundle["state_transition_rows"]),
        "gate_accounting_rows": rows,
    }
    return validate_objectstate_real_evidence_bundle(updated)


def _indexes(bundle: Mapping[str, Any]) -> dict[str, Any]:
    physical_by_object_frame = {
        (str(row["object_id"]), str(row["frame_id"])): str(row["physical_identity_id"])
        for row in bundle["identity_link_rows"]
    }
    return {
        "physical_by_object_frame": physical_by_object_frame,
    }


def _selected_pose_rows(
    bundle: Mapping[str, Any],
    accounting: Mapping[str, Any],
) -> list[dict[str, Any]]:
    object_id = accounting.get("object_id")
    rows = [
        dict(row)
        for row in bundle["object_pose_rows"]
        if object_id is None or row["object_id"] == object_id
    ]
    if rows:
        return rows
    return [dict(row) for row in bundle["object_pose_rows"]]


def _physical_labels(
    pose_rows: Sequence[Mapping[str, Any]],
    indexes: Mapping[str, Any],
) -> list[str]:
    labels = []
    for row in pose_rows:
        labels.append(
            indexes["physical_by_object_frame"][
                (str(row["object_id"]), str(row["frame_id"]))
            ]
        )
    return labels


def _teacher_assignment_map(
    teacher: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {str(row["object_pose_row_id"]): row for row in teacher["assignments"]}


def _identity_metrics(
    labels: Sequence[str],
    slots: Sequence[str],
    *,
    positions: np.ndarray,
    visibility: np.ndarray,
    vectors: np.ndarray,
) -> dict[str, float]:
    return {
        "identity_retrieval_at_1": _retrieval_at_1(labels, vectors),
        "identity_margin": _identity_margin(labels, vectors),
        "slot_swap_rate": _slot_swap_rate(labels, slots),
        "objectstate_drift": _objectstate_drift(labels, vectors),
        "assignment_consistency": _assignment_consistency(labels, slots),
        "occlusion_recovery": _occlusion_recovery(labels, slots, visibility),
    }


def _retrieval_at_1(labels: Sequence[str], vectors: np.ndarray) -> float:
    if len(labels) <= 1:
        return 1.0
    correct = 0
    total = 0
    for index, label in enumerate(labels):
        same = []
        wrong = []
        for other, other_label in enumerate(labels):
            if other == index:
                continue
            distance = float(np.linalg.norm(vectors[index] - vectors[other]))
            if other_label == label:
                same.append(distance)
            else:
                wrong.append(distance)
        if not same:
            continue
        total += 1
        if not wrong or min(same) < min(wrong):
            correct += 1
    return 1.0 if total == 0 else float(correct / total)


def _identity_margin(labels: Sequence[str], vectors: np.ndarray) -> float:
    margins = []
    for index, label in enumerate(labels):
        same = []
        wrong = []
        for other, other_label in enumerate(labels):
            if other == index:
                continue
            distance = float(np.linalg.norm(vectors[index] - vectors[other]))
            if other_label == label:
                same.append(distance)
            else:
                wrong.append(distance)
        if same and wrong:
            margins.append(min(wrong) - min(same))
    return 0.0 if not margins else float(np.mean(margins))


def _slot_swap_rate(labels: Sequence[str], slots: Sequence[str]) -> float:
    by_label: dict[str, set[str]] = {}
    for label, slot in zip(labels, slots):
        by_label.setdefault(label, set()).add(slot)
    if not by_label:
        return 0.0
    swapped = sum(1 for values in by_label.values() if len(values) > 1)
    return float(swapped / len(by_label))


def _assignment_consistency(labels: Sequence[str], slots: Sequence[str]) -> float:
    same_pairs = 0
    consistent = 0
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            if labels[left] != labels[right]:
                continue
            same_pairs += 1
            if slots[left] == slots[right]:
                consistent += 1
    return 1.0 if same_pairs == 0 else float(consistent / same_pairs)


def _objectstate_drift(labels: Sequence[str], vectors: np.ndarray) -> float:
    distances = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            if labels[left] == labels[right]:
                distances.append(float(np.linalg.norm(vectors[left] - vectors[right])))
    return 0.0 if not distances else float(np.mean(distances))


def _occlusion_recovery(
    labels: Sequence[str],
    slots: Sequence[str],
    visibility: np.ndarray,
) -> float:
    occluded = [index for index, value in enumerate(visibility) if float(value) < 1.0]
    if not occluded:
        return 1.0
    recovered = 0
    for index in occluded:
        visible_slots = [
            slots[other]
            for other, label in enumerate(labels)
            if label == labels[index] and float(visibility[other]) >= 1.0
        ]
        if visible_slots and slots[index] == max(set(visible_slots), key=visible_slots.count):
            recovered += 1
    return float(recovered / len(occluded))


def _assignment_vectors(
    pose_rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
    slots: Sequence[str],
) -> np.ndarray:
    vectors = []
    for pose, slot in zip(pose_rows, slots):
        assignment = assignments[str(pose["row_id"])]
        embedding = assignment.get("embedding")
        if embedding is None:
            vectors.append(None)
        else:
            vectors.append([float(item) for item in embedding])
    if all(vector is not None for vector in vectors):
        return np.asarray(vectors, dtype=np.float64)
    return _slot_vectors(slots)


def _slot_vectors(slots: Sequence[str]) -> np.ndarray:
    unique = {slot: index for index, slot in enumerate(sorted(set(slots)))}
    result = np.zeros((len(slots), max(1, len(unique))), dtype=np.float64)
    for row, slot in enumerate(slots):
        result[row, unique[slot]] = 1.0
    return result


def _random_slots(count: int, identity_count: int, *, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    return [f"random-{int(item)}" for item in rng.integers(0, max(1, identity_count), size=count)]


def _xyz_slots(positions: np.ndarray, identity_count: int) -> list[str]:
    order = np.argsort(positions[:, 0], kind="mergesort")
    slots = ["xyz-0"] * len(positions)
    for rank, index in enumerate(order):
        slot = min(identity_count - 1, int(rank * max(1, identity_count) / max(1, len(order))))
        slots[int(index)] = f"xyz-{slot}"
    return slots


def _oracle_slots(labels: Sequence[str]) -> list[str]:
    mapping = {label: f"oracle-{index}" for index, label in enumerate(sorted(set(labels)))}
    return [mapping[label] for label in labels]


def _baseline_comparison(baselines: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    solver = baselines["assignment_solver_v2"]["identity_retrieval_at_1"]
    random = baselines["random_assignment"]["identity_retrieval_at_1"]
    xyz = baselines["xyz_centroid"]["identity_retrieval_at_1"]
    oracle = baselines["oracle_target_assignment"]["identity_retrieval_at_1"]
    return {
        "assignment_solver_v2_vs_random_assignment": float(solver - random),
        "assignment_solver_v2_vs_xyz_centroid": float(solver - xyz),
        "oracle_target_assignment_vs_assignment_solver_v2": float(oracle - solver),
        "identity_model_underperforms_xyz_centroid": bool(solver < xyz),
    }


def _aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evaluated = [row for row in rows if row["eval_status"] in _READY_STATUSES]
    if not evaluated:
        return {
            "teacher_evidence_coverage": 0.0,
            "metrics": _empty_metrics(),
            "baselines": _empty_baselines(),
            "baseline_comparison": {
                "assignment_solver_v2_vs_random_assignment": 0.0,
                "assignment_solver_v2_vs_xyz_centroid": 0.0,
                "oracle_target_assignment_vs_assignment_solver_v2": 0.0,
                "identity_model_underperforms_xyz_centroid": False,
            },
        }
    metrics = _mean_metric(row["metrics"] for row in evaluated)
    baselines = {
        name: _mean_metric(row["baselines"][name] for row in evaluated)
        for name in OBJECTSTATE_CONTROLLED_REAL_IDENTITY_BASELINES
    }
    return {
        "teacher_evidence_coverage": float(
            np.mean([float(row["teacher_evidence_coverage"]) for row in evaluated])
        ),
        "metrics": metrics,
        "baselines": baselines,
        "baseline_comparison": _baseline_comparison(baselines),
    }


def _mean_metric(metrics: Sequence[Mapping[str, float]]) -> dict[str, float]:
    metric_rows = list(metrics)
    keys = (
        "identity_retrieval_at_1",
        "identity_margin",
        "slot_swap_rate",
        "objectstate_drift",
        "assignment_consistency",
        "occlusion_recovery",
    )
    return {
        key: float(np.mean([float(metric[key]) for metric in metric_rows]))
        for key in keys
    }


def _pass_gates(
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> dict[str, bool]:
    evaluated = [row for row in rows if row["eval_status"] in _READY_STATUSES]
    comparison = aggregate["baseline_comparison"]
    return {
        "evaluated_identity_rows_positive": bool(evaluated),
        "identity_metrics_present": bool(evaluated)
        and all(_metrics_finite(row["metrics"]) for row in evaluated),
        "oracle_not_below_assignment_solver_v2": (
            comparison["oracle_target_assignment_vs_assignment_solver_v2"] >= -1.0e-9
        ),
        "assignment_solver_v2_not_below_random": (
            comparison["assignment_solver_v2_vs_random_assignment"] >= -1.0e-9
        ),
        "evidence_incomplete_counted_separately": all(
            row["eval_status"] != "fail" or row.get("reason") != "missing_teacher_evidence"
            for row in rows
        ),
    }


def _summary_status(
    rows: Sequence[Mapping[str, Any]],
    pass_gates: Mapping[str, bool],
) -> str:
    if not rows or any(row["eval_status"] == "evidence_incomplete" for row in rows):
        return "objectstate_controlled_real_identity_eval_incomplete"
    if any(row["eval_status"] == "blocked" for row in rows):
        return "objectstate_controlled_real_identity_eval_blocked"
    if all(pass_gates.values()) and all(row["eval_status"] == "pass" for row in rows):
        return "objectstate_controlled_real_identity_eval_pass"
    return "objectstate_controlled_real_identity_eval_fail"


def _issues(rows: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any]) -> list[str]:
    issues = []
    for row in rows:
        if row.get("reason"):
            issues.append(f"{row['row_id']}: {row['reason']}")
    if aggregate["baseline_comparison"]["identity_model_underperforms_xyz_centroid"]:
        issues.append("identity_model_underperforms_xyz_centroid")
    return issues


def _empty_metrics() -> dict[str, float]:
    return {
        "identity_retrieval_at_1": 0.0,
        "identity_margin": 0.0,
        "slot_swap_rate": 0.0,
        "objectstate_drift": 0.0,
        "assignment_consistency": 0.0,
        "occlusion_recovery": 0.0,
    }


def _empty_baselines() -> dict[str, dict[str, float]]:
    return {name: _empty_metrics() for name in OBJECTSTATE_CONTROLLED_REAL_IDENTITY_BASELINES}


def _pairwise_rows(
    pose_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    candidate_vectors: np.ndarray,
    oracle_slots: Sequence[str],
) -> list[dict[str, Any]]:
    oracle_vectors = _slot_vectors(oracle_slots)
    rows = []
    for left in range(len(pose_rows)):
        for right in range(left + 1, len(pose_rows)):
            rows.append(
                {
                    "row_id_a": str(pose_rows[left]["row_id"]),
                    "row_id_b": str(pose_rows[right]["row_id"]),
                    "same_identity": bool(labels[left] == labels[right]),
                    "candidate_distance": float(
                        np.linalg.norm(candidate_vectors[left] - candidate_vectors[right])
                    ),
                    "oracle_distance": float(
                        np.linalg.norm(oracle_vectors[left] - oracle_vectors[right])
                    ),
                }
            )
    return rows


def _matching_rows(
    pose_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    slots: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        {
            "object_pose_row_id": str(pose["row_id"]),
            "frame_id": str(pose["frame_id"]),
            "object_id": str(pose["object_id"]),
            "physical_identity_id": str(label),
            "assignment_slot_id": str(slot),
        }
        for pose, label, slot in zip(pose_rows, labels, slots)
    ]


def _coverage(
    pose_rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
) -> float:
    if not pose_rows:
        return 0.0
    covered = sum(1 for pose in pose_rows if pose["row_id"] in assignments)
    return float(covered / len(pose_rows))


def _readiness_row(readiness: Mapping[str, Any], row_id: str) -> Mapping[str, Any] | None:
    for row in readiness["row_breakdown"]:
        if row["row_id"] == row_id:
            return row
    return None


def _validate_teacher_assignment(value: Any) -> dict[str, Any]:
    row = _mapping(value, "assignment")
    result = {
        "object_pose_row_id": _required_str(row, "object_pose_row_id"),
        "slot_id": _required_str(row, "slot_id"),
    }
    if row.get("embedding") is not None:
        embedding = _float_vector(row["embedding"], "embedding")
        if not embedding:
            raise ValueError("teacher assignment embedding cannot be empty")
        result["embedding"] = embedding
    if row.get("confidence") is not None:
        result["confidence"] = _metric(row["confidence"], "confidence")
    return result


def _validate_eval_row(row: Any) -> None:
    checked = _mapping(row, "identity_accounting_row")
    _required_str(checked, "row_id")
    if checked.get("eval_status") not in {"pass", "fail", "blocked", "evidence_incomplete"}:
        raise ValueError("controlled real identity row eval_status unsupported")
    _mapping(checked.get("metrics"), "metrics")
    _mapping(checked.get("baselines"), "baselines")
    _mapping(checked.get("baseline_comparison"), "baseline_comparison")
    _mapping(checked.get("real_bundle_accounting_row"), "real_bundle_accounting_row")


def _teacher_summary(teacher: Mapping[str, Any] | None) -> dict[str, Any]:
    if teacher is None:
        return {
            "present": False,
            "teacher_evidence_source": None,
            "assignment_count": 0,
        }
    return {
        "present": True,
        "schema": teacher["schema"],
        "teacher_evidence_source": teacher["teacher_evidence_source"],
        "evidence_policy": teacher["evidence_policy"],
        "assignment_count": len(teacher["assignments"]),
        "provenance": dict(teacher["provenance"]),
    }


def _metrics_finite(metrics: Mapping[str, Any]) -> bool:
    return all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in metrics.values()
    )


def _reason(reasons: Sequence[str], fallback: str) -> str:
    return str(reasons[0]) if reasons else fallback


def _required_str(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _float_vector(value: Any, name: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return [_metric(item, name) for item in value]


def _metric(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return int(value)
