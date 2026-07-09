from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_prediction_eval import (
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
    read_objectstate_controlled_prediction_candidates,
    validate_objectstate_controlled_prediction_candidates,
)
from objgauss.core.objectstate_controlled_real_identity_eval import (
    OBJECTSTATE_CONTROLLED_REAL_IDENTITY_EVAL_SCHEMA,
    validate_objectstate_controlled_real_identity_eval,
)
from objgauss.core.objectstate_controlled_real_readiness_audit import (
    objectstate_controlled_real_readiness_audit,
    validate_objectstate_controlled_real_readiness_audit,
)
from objgauss.core.objectstate_real_evidence_bundle import (
    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
    OBJECTSTATE_REAL_GATE_ACCOUNTING_ROW_SCHEMA,
    read_objectstate_real_evidence_bundle,
    validate_objectstate_real_evidence_bundle,
)

OBJECTSTATE_CONTROLLED_REAL_PREDICTION_EVAL_SCHEMA = (
    "objgauss-objectstate-controlled-real-prediction-eval-v1"
)

_EVALUATED_STATUSES = {"pass", "fail"}
_EPSILON = 1.0e-9


def objectstate_controlled_real_prediction_eval_from_files(
    bundle_path: str | Path,
    *,
    identity_eval_path: str | Path | None = None,
    prediction_candidates_path: str | Path | None = None,
    teacher_evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    bundle = read_objectstate_real_evidence_bundle(bundle_path)
    identity_eval = (
        None
        if identity_eval_path is None
        else read_objectstate_controlled_real_identity_eval(identity_eval_path)
    )
    candidates = (
        None
        if prediction_candidates_path is None
        else read_objectstate_controlled_prediction_candidates(prediction_candidates_path)
    )
    return objectstate_controlled_real_prediction_eval(
        bundle,
        identity_eval=identity_eval,
        prediction_candidates=candidates,
        bundle_ref=str(bundle_path),
        identity_eval_ref=None if identity_eval_path is None else str(identity_eval_path),
        prediction_candidates_ref=(
            None if prediction_candidates_path is None else str(prediction_candidates_path)
        ),
        teacher_evidence_ref=(
            None if teacher_evidence_path is None else str(teacher_evidence_path)
        ),
    )


def read_objectstate_controlled_real_identity_eval(
    path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("controlled real identity eval JSON must be an object")
    return validate_objectstate_controlled_real_identity_eval(payload)


def objectstate_controlled_real_prediction_eval(
    bundle: Mapping[str, Any],
    *,
    identity_eval: Mapping[str, Any] | None = None,
    prediction_candidates: Mapping[str, Any] | None = None,
    bundle_ref: str | None = None,
    identity_eval_ref: str | None = None,
    prediction_candidates_ref: str | None = None,
    teacher_evidence_ref: str | None = None,
) -> dict[str, Any]:
    checked = validate_objectstate_real_evidence_bundle(bundle)
    identity = (
        None
        if identity_eval is None
        else validate_objectstate_controlled_real_identity_eval(identity_eval)
    )
    candidates = (
        None
        if prediction_candidates is None
        else validate_objectstate_controlled_prediction_candidates(prediction_candidates)
    )
    sample_id = checked["sample"]["sample_id"]
    if identity is not None and identity["sample"]["sample_id"] != sample_id:
        raise ValueError("controlled real prediction identity eval sample_id mismatch")
    if candidates is not None and candidates["sample_id"] != sample_id:
        raise ValueError("controlled real prediction candidates sample_id mismatch")
    readiness = objectstate_controlled_real_readiness_audit(
        checked,
        bundle_ref=bundle_ref,
    )
    indexes = _indexes(checked)
    candidate_map = _candidate_map(candidates)
    identity_dependency = _identity_dependency(identity)
    prediction_rows = [
        row for row in checked["gate_accounting_rows"] if row["evidence_kind"] == "prediction"
    ]
    row_results = []
    evaluated_accounting_rows = []
    for row in prediction_rows:
        result = _evaluate_prediction_accounting_row(
            row,
            checked,
            indexes,
            readiness,
            identity_dependency,
            candidate_map,
            candidates,
        )
        row_results.append(result)
        evaluated_accounting_rows.append(result["real_bundle_accounting_row"])
    base_bundle = (
        checked
        if identity is None
        else validate_objectstate_real_evidence_bundle(identity["evaluated_real_bundle"])
    )
    evaluated_bundle = _bundle_with_prediction_accounting(
        base_bundle,
        evaluated_accounting_rows,
    )
    aggregate = _aggregate(row_results, readiness)
    counts = aggregate["row_counts"]
    pass_gates = _pass_gates(row_results, aggregate)
    status = _summary_status(row_results, pass_gates)
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_REAL_PREDICTION_EVAL_SCHEMA,
        "kind": "objectstate_controlled_real_prediction_eval",
        "status": status,
        "bundle_schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
        "identity_eval_schema": (
            None if identity is None else OBJECTSTATE_CONTROLLED_REAL_IDENTITY_EVAL_SCHEMA
        ),
        "prediction_candidates_schema": (
            None if candidates is None else OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA
        ),
        "bundle_ref": bundle_ref,
        "identity_eval_ref": identity_eval_ref,
        "prediction_candidates_ref": prediction_candidates_ref,
        "teacher_evidence_ref": teacher_evidence_ref,
        "sample": dict(checked["sample"]),
        "candidate": None if candidates is None else dict(candidates["candidate"]),
        "identity_dependency": identity_dependency,
        "readiness_audit": readiness,
        "row_counts": counts,
        "metrics": aggregate["metrics"],
        "baselines": aggregate["baselines"],
        "pass_gates": pass_gates,
        "duplicate_row_ids": aggregate["duplicate_row_ids"],
        "prediction_accounting_rows": row_results,
        "evaluated_real_bundle": evaluated_bundle,
        "errors": aggregate["errors"],
        "issues": _issues(row_results, aggregate),
        "claim_policy": {
            "uses_real_bundle_pose_gt_for_evaluation_only": True,
            "requires_identity_eval_dependency": True,
            "identity_inconsistent_rows_block_not_fail": True,
            "evidence_incomplete_is_not_model_fail": True,
            "writes_prediction_accounting": True,
            "hold_last_baseline_required": True,
            "constant_velocity_baseline_when_history_allows": True,
            "identity_rows_preserved_from_identity_eval": True,
            "intervention_rows_out_of_scope": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "runs_teacher_model": False,
            "runs_prediction_model": False,
            "reconstructs_gaussians": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "runs_identity_eval": False,
            "runs_intervention_eval": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_controlled_real_prediction_eval(payload)


def objectstate_controlled_real_prediction_report(summary: Mapping[str, Any]) -> str:
    checked = validate_objectstate_controlled_real_prediction_eval(summary)
    lines = [
        f"# Controlled Real Prediction Eval: {checked['sample']['sample_id']}",
        "",
        f"- status: `{checked['status']}`",
        f"- evaluated_prediction_rows: `{checked['row_counts']['evaluated_prediction_rows']}`",
        f"- prediction_pass_rows: `{checked['row_counts']['prediction_pass_rows']}`",
        f"- prediction_fail_rows: `{checked['row_counts']['prediction_fail_rows']}`",
        f"- prediction_rows_blocked: `{checked['row_counts']['prediction_rows_blocked']}`",
        f"- prediction_rows_evidence_incomplete: `{checked['row_counts']['prediction_rows_evidence_incomplete']}`",
        f"- transition_coverage: `{checked['metrics']['transition_coverage']:.6f}`",
        f"- identity_consistency_rate: `{checked['metrics']['identity_consistency_rate']:.6f}`",
        "",
        "## Metrics",
        "",
    ]
    for key, value in checked["metrics"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(("", "## Rows", ""))
    for row in checked["prediction_accounting_rows"]:
        lines.append(
            "- "
            f"`{row['row_id']}` "
            f"status={row['eval_status']} "
            f"reason={row.get('reason') or 'none'}"
        )
    lines.append("")
    return "\n".join(lines)


def objectstate_controlled_real_prediction_accounting_csv(
    summary: Mapping[str, Any],
) -> str:
    checked = validate_objectstate_controlled_real_prediction_eval(summary)
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "row_id",
            "eval_status",
            "bundle_accounting_status",
            "object_id",
            "transition_id",
            "state_ade",
            "state_fde",
            "pose_translation_error",
            "pose_rotation_error",
            "hold_last_ade",
            "kalman_ade",
            "history_ade",
            "state_vs_history_error_ratio",
            "state_vs_kalman_error_ratio",
            "identity_consistent",
            "reason",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for row in checked["prediction_accounting_rows"]:
        metrics = row.get("metrics", {})
        writer.writerow(
            {
                "row_id": row["row_id"],
                "eval_status": row["eval_status"],
                "bundle_accounting_status": row["real_bundle_accounting_row"][
                    "accounting_status"
                ],
                "object_id": row.get("object_id", ""),
                "transition_id": row.get("transition_id", ""),
                "state_ade": metrics.get("state_ade", ""),
                "state_fde": metrics.get("state_fde", ""),
                "pose_translation_error": metrics.get("pose_translation_error", ""),
                "pose_rotation_error": metrics.get("pose_rotation_error", ""),
                "hold_last_ade": metrics.get("hold_last_ade", ""),
                "kalman_ade": metrics.get("kalman_ade", ""),
                "history_ade": metrics.get("history_ade", ""),
                "state_vs_history_error_ratio": metrics.get(
                    "state_vs_history_error_ratio",
                    "",
                ),
                "state_vs_kalman_error_ratio": metrics.get(
                    "state_vs_kalman_error_ratio",
                    "",
                ),
                "identity_consistent": metrics.get("identity_consistent", ""),
                "reason": row.get("reason", ""),
            }
        )
    return output.getvalue()


def objectstate_controlled_real_prediction_errors_csv(
    summary: Mapping[str, Any],
) -> str:
    checked = validate_objectstate_controlled_real_prediction_eval(summary)
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "row_id",
            "transition_id",
            "object_id",
            "source_frame_id",
            "target_frame_id",
            "eval_status",
            "state_error",
            "hold_last_error",
            "kalman_error",
            "history_error",
            "reason",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for row in checked["errors"]:
        writer.writerow(row)
    return output.getvalue()


def objectstate_controlled_real_prediction_artifact_manifest(
    summary: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    checked = validate_objectstate_controlled_real_prediction_eval(summary)
    root = Path(output_dir)
    return {
        "schema": "objgauss-objectstate-controlled-real-prediction-artifact-manifest-v1",
        "kind": "objectstate_controlled_real_prediction_artifact_manifest",
        "sample_id": checked["sample"]["sample_id"],
        "prediction_eval_schema": OBJECTSTATE_CONTROLLED_REAL_PREDICTION_EVAL_SCHEMA,
        "artifacts": {
            "summary": str(root / "controlled-real-prediction-summary.json"),
            "report": str(root / "controlled-real-prediction-report.md"),
            "accounting_csv": str(root / "controlled-real-prediction-accounting.csv"),
            "errors_csv": str(root / "controlled-real-prediction-errors.csv"),
            "baselines": str(root / "controlled-real-prediction-baselines.json"),
            "evaluated_real_bundle": str(root / "evaluated-real-bundle.json"),
        },
        "claim_policy": {
            "artifacts_are_local_evaluation_outputs": True,
            "does_not_claim_reality_gate_pass": True,
        },
    }


def validate_objectstate_controlled_real_prediction_eval(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled real prediction eval must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_REAL_PREDICTION_EVAL_SCHEMA:
        raise ValueError(
            "unsupported controlled real prediction eval schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_real_prediction_eval":
        raise ValueError("controlled real prediction eval kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_real_prediction_eval_pass",
        "objectstate_controlled_real_prediction_eval_fail",
        "objectstate_controlled_real_prediction_eval_blocked",
        "objectstate_controlled_real_prediction_eval_incomplete",
    }:
        raise ValueError("controlled real prediction eval status is unsupported")
    if payload.get("bundle_schema") != OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA:
        raise ValueError("controlled real prediction eval bundle schema mismatch")
    if payload.get("identity_eval_schema") not in {
        None,
        OBJECTSTATE_CONTROLLED_REAL_IDENTITY_EVAL_SCHEMA,
    }:
        raise ValueError("controlled real prediction eval identity schema mismatch")
    if payload.get("prediction_candidates_schema") not in {
        None,
        OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
    }:
        raise ValueError("controlled real prediction candidates schema mismatch")
    validate_objectstate_controlled_real_readiness_audit(payload.get("readiness_audit"))
    row_counts = _mapping(payload.get("row_counts"), "row_counts")
    for key in (
        "prediction_rows",
        "evaluated_prediction_rows",
        "prediction_ready_rows_consumed",
        "prediction_rows_evidence_incomplete",
        "prediction_rows_blocked",
        "prediction_pass_rows",
        "prediction_fail_rows",
    ):
        _non_negative_int(row_counts.get(key), f"row_counts.{key}")
    metrics = _mapping(payload.get("metrics"), "metrics")
    for key in (
        "state_ade",
        "state_fde",
        "pose_translation_error",
        "pose_rotation_error",
        "hold_last_ade",
        "kalman_ade",
        "history_ade",
        "state_vs_history_error_ratio",
        "state_vs_kalman_error_ratio",
        "transition_coverage",
        "identity_consistency_rate",
    ):
        _metric(metrics.get(key), f"metrics.{key}")
    baselines = _mapping(payload.get("baselines"), "baselines")
    for name in ("hold_last", "constant_velocity", "history"):
        if name not in baselines:
            raise ValueError(f"controlled real prediction missing baseline {name}")
    pass_gates = _mapping(payload.get("pass_gates"), "pass_gates")
    if any(not isinstance(value, bool) for value in pass_gates.values()):
        raise ValueError("controlled real prediction pass_gates must be bool")
    duplicate_row_ids = payload.get("duplicate_row_ids")
    if not isinstance(duplicate_row_ids, list):
        raise ValueError("controlled real prediction requires duplicate_row_ids")
    rows = _sequence(
        payload.get("prediction_accounting_rows"),
        "prediction_accounting_rows",
    )
    if len(rows) != row_counts["prediction_rows"]:
        raise ValueError("controlled real prediction row count mismatch")
    for row in rows:
        _validate_eval_row(row)
    validate_objectstate_real_evidence_bundle(payload.get("evaluated_real_bundle"))
    if not isinstance(payload.get("errors"), list):
        raise ValueError("controlled real prediction eval requires errors")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("controlled real prediction eval requires issues")
    claim_policy = _mapping(payload.get("claim_policy"), "claim_policy")
    if (
        not claim_policy.get("uses_real_bundle_pose_gt_for_evaluation_only")
        or not claim_policy.get("requires_identity_eval_dependency")
        or not claim_policy.get("identity_inconsistent_rows_block_not_fail")
        or not claim_policy.get("evidence_incomplete_is_not_model_fail")
        or not claim_policy.get("writes_prediction_accounting")
        or not claim_policy.get("hold_last_baseline_required")
        or not claim_policy.get("constant_velocity_baseline_when_history_allows")
        or not claim_policy.get("identity_rows_preserved_from_identity_eval")
        or not claim_policy.get("intervention_rows_out_of_scope")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled real prediction eval must preserve claim policy")
    non_goals = _mapping(payload.get("non_goals"), "non_goals")
    if any(bool(value) for value in non_goals.values()):
        raise ValueError("controlled real prediction eval cannot claim non-goals")
    return dict(payload)


def _evaluate_prediction_accounting_row(
    row: Mapping[str, Any],
    bundle: Mapping[str, Any],
    indexes: Mapping[str, Any],
    readiness: Mapping[str, Any],
    identity_dependency: Mapping[str, Any],
    candidate_map: Mapping[tuple[str, str, str], Mapping[str, Any]],
    candidates: Mapping[str, Any] | None,
) -> dict[str, Any]:
    readiness_row = _readiness_row(readiness, row["row_id"])
    transition = _row_transition(row, indexes)
    identity_status = _transition_identity_status(transition, indexes)
    base = {
        "row_id": str(row["row_id"]),
        "evidence_kind": "prediction",
        "object_id": row.get("object_id"),
        "transition_id": row.get("transition_id"),
        "source_accounting_status": str(row["accounting_status"]),
        "readiness": {
            "prediction_ready": bool(
                readiness_row and readiness_row["evaluator_ready"]
            ),
            "blocked_reasons": (
                [] if readiness_row is None else list(readiness_row["blocked_reasons"])
            ),
        },
        "identity_consistency": identity_status,
    }
    if not base["readiness"]["prediction_ready"]:
        if identity_status["has_source_target_identity"] and not identity_status["consistent"]:
            return _blocked_eval_row(base, row, "blocked", "identity_not_stable")
        reason = _reason(base["readiness"]["blocked_reasons"], "missing_before_pose")
        return _blocked_eval_row(base, row, "evidence_incomplete", reason)
    if not identity_dependency["stable"]:
        return _blocked_eval_row(base, row, "blocked", "identity_not_stable")
    if candidates is None:
        return _blocked_eval_row(base, row, "blocked", "missing_prediction_candidates")
    if transition is None:
        return _blocked_eval_row(base, row, "evidence_incomplete", "missing_before_pose")
    key = (
        str(transition["source_frame_id"]),
        str(transition["target_frame_id"]),
        str(transition["object_id"]),
    )
    candidate = candidate_map.get(key)
    if candidate is None:
        return _blocked_eval_row(base, row, "blocked", "missing_prediction_candidates")
    source = indexes["pose_by_id"][str(transition["source_pose_row_id"])]
    target = indexes["pose_by_id"][str(transition["target_pose_row_id"])]
    previous = _previous_pose(bundle, source)
    metrics, error_row = _row_metrics(
        transition,
        source,
        target,
        candidate,
        previous,
        identity_status,
    )
    passed = metrics["state_ade"] <= metrics["hold_last_ade"] + _EPSILON
    eval_status = "pass" if passed else "fail"
    reason = None if passed else "prediction_model_underperforms_hold_last"
    artifact_refs = _artifact_refs(row, candidates=candidates)
    real_accounting = _real_accounting_row(
        row,
        accounting_status=eval_status,
        metrics=metrics,
        reason=reason,
        artifact_refs=artifact_refs,
    )
    return {
        **base,
        "eval_status": eval_status,
        "candidate_id": candidates["candidate"]["candidate_id"],
        "metrics": metrics,
        "baselines": _row_baselines(metrics),
        "pass_gates": {
            "state_error_at_or_below_hold_last": passed,
            "identity_consistent": bool(metrics["identity_consistent"]),
            "hold_last_baseline_present": True,
            "history_baseline_present": True,
            "constant_velocity_baseline_present_if_history_allows": (
                not metrics["constant_velocity_history_available"]
                or metrics["constant_velocity_baseline_present"]
            ),
        },
        "reason": reason,
        "real_bundle_accounting_row": real_accounting,
        "error": error_row,
    }


def _blocked_eval_row(
    base: Mapping[str, Any],
    source_row: Mapping[str, Any],
    eval_status: str,
    reason: str,
) -> dict[str, Any]:
    accounting_status = (
        "evidence_incomplete"
        if eval_status in {"blocked", "evidence_incomplete"}
        else eval_status
    )
    metrics = _empty_metrics(base.get("identity_consistency", {}))
    return {
        **dict(base),
        "eval_status": eval_status,
        "metrics": metrics,
        "baselines": _row_baselines(metrics),
        "pass_gates": {
            "state_error_at_or_below_hold_last": False,
            "identity_consistent": bool(metrics["identity_consistent"]),
            "hold_last_baseline_present": False,
            "history_baseline_present": False,
            "constant_velocity_baseline_present_if_history_allows": False,
        },
        "reason": reason,
        "real_bundle_accounting_row": _real_accounting_row(
            source_row,
            accounting_status=accounting_status,
            metrics={},
            reason=reason,
            artifact_refs=list(source_row["artifact_refs"]),
        ),
        "error": _empty_error_row(base, eval_status, reason),
    }


def _real_accounting_row(
    source_row: Mapping[str, Any],
    *,
    accounting_status: str,
    metrics: Mapping[str, Any],
    reason: str | None,
    artifact_refs: Sequence[str],
) -> dict[str, Any]:
    row = {
        "schema": OBJECTSTATE_REAL_GATE_ACCOUNTING_ROW_SCHEMA,
        "row_id": str(source_row["row_id"]),
        "evidence_kind": "prediction",
        "accounting_status": accounting_status,
        "metrics": dict(metrics),
        "artifact_refs": list(_unique_strings(artifact_refs)),
        "gt_requirements": dict(source_row["gt_requirements"]),
    }
    if source_row.get("object_id"):
        row["object_id"] = str(source_row["object_id"])
    if source_row.get("transition_id"):
        row["transition_id"] = str(source_row["transition_id"])
    if reason:
        row["reason"] = reason
    return row


def _bundle_with_prediction_accounting(
    bundle: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    replacements = {row["row_id"]: row for row in prediction_rows}
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
    pose_by_id = {str(row["row_id"]): row for row in bundle["object_pose_rows"]}
    physical_by_object_frame = {
        (str(row["object_id"]), str(row["frame_id"])): str(row["physical_identity_id"])
        for row in bundle["identity_link_rows"]
    }
    transition_by_id = {
        str(row["transition_id"]): row for row in bundle["state_transition_rows"]
    }
    return {
        "pose_by_id": pose_by_id,
        "physical_by_object_frame": physical_by_object_frame,
        "transition_by_id": transition_by_id,
    }


def _candidate_map(
    candidates: Mapping[str, Any] | None,
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    if candidates is None:
        return {}
    result = {}
    for item in candidates["predictions"]:
        key = (
            str(item["source_frame_id"]),
            str(item["target_frame_id"]),
            str(item["object_id"]),
        )
        if key in result:
            raise ValueError(
                "controlled real prediction candidates contain duplicate "
                f"source/target/object tuple: {key[0]} / {key[1]} / {key[2]}"
            )
        result[key] = item
    return result


def _identity_dependency(identity: Mapping[str, Any] | None) -> dict[str, Any]:
    if identity is None:
        return {
            "stable": False,
            "status": "missing_identity_eval",
            "reason": "identity_not_stable",
            "teacher_evidence_coverage": 0.0,
        }
    status = str(identity["status"])
    stable = status == "objectstate_controlled_real_identity_eval_pass"
    return {
        "stable": stable,
        "status": status,
        "reason": None if stable else "identity_not_stable",
        "teacher_evidence_coverage": float(identity.get("teacher_evidence_coverage", 0.0)),
        "identity_pass_rows": int(identity["row_counts"]["identity_pass_rows"]),
        "identity_fail_rows": int(identity["row_counts"]["identity_fail_rows"]),
        "identity_rows_blocked": int(identity["row_counts"]["identity_rows_blocked"]),
    }


def _readiness_row(
    readiness: Mapping[str, Any],
    row_id: str,
) -> Mapping[str, Any] | None:
    for row in readiness["row_breakdown"]:
        if row["row_id"] == row_id:
            return row
    return None


def _row_transition(
    row: Mapping[str, Any],
    indexes: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    transition_id = row.get("transition_id")
    if not isinstance(transition_id, str) or not transition_id:
        return None
    return indexes["transition_by_id"].get(transition_id)


def _transition_identity_status(
    transition: Mapping[str, Any] | None,
    indexes: Mapping[str, Any],
) -> dict[str, Any]:
    if transition is None:
        return {
            "has_transition": False,
            "has_source_target_identity": False,
            "consistent": False,
            "source_physical_identity_id": None,
            "target_physical_identity_id": None,
        }
    object_id = str(transition["object_id"])
    source = indexes["physical_by_object_frame"].get(
        (object_id, str(transition["source_frame_id"]))
    )
    target = indexes["physical_by_object_frame"].get(
        (object_id, str(transition["target_frame_id"]))
    )
    return {
        "has_transition": True,
        "has_source_target_identity": bool(source and target),
        "consistent": bool(source and target and source == target),
        "source_physical_identity_id": source,
        "target_physical_identity_id": target,
    }


def _row_metrics(
    transition: Mapping[str, Any],
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    candidate: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    identity_status: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_position = list(source["object_pose_6dof"]["position"])
    target_position = list(target["object_pose_6dof"]["position"])
    predicted_position = list(candidate["predicted_position"])
    history_position = list(candidate["history_baseline_position"])
    hold_last_position = source_position
    constant_velocity_position, constant_velocity_present = _constant_velocity_position(
        source,
        previous,
        target_timestamp=float(transition["target_timestamp"]),
    )
    state_error = _distance(predicted_position, target_position)
    hold_last_error = _distance(hold_last_position, target_position)
    history_error = _distance(history_position, target_position)
    constant_velocity_error = _distance(constant_velocity_position, target_position)
    rotation_error = _rotation_error(
        source["object_pose_6dof"]["rotation_xyzw"],
        target["object_pose_6dof"]["rotation_xyzw"],
    )
    duration = float(transition["target_timestamp"]) - float(
        transition["source_timestamp"]
    )
    metrics = {
        "state_ade": state_error,
        "state_fde": state_error,
        "pose_translation_error": state_error,
        "pose_rotation_error": rotation_error,
        "hold_last_ade": hold_last_error,
        "kalman_ade": constant_velocity_error,
        "history_ade": history_error,
        "state_vs_history_error_ratio": _ratio(state_error, history_error),
        "state_vs_kalman_error_ratio": _ratio(state_error, constant_velocity_error),
        "prediction_gap_vs_history_model": state_error - history_error,
        "prediction_gap_vs_hold_last": state_error - hold_last_error,
        "prediction_gap_vs_kalman": state_error - constant_velocity_error,
        "transition_duration_s": duration,
        "identity_consistent": bool(identity_status["consistent"]),
        "constant_velocity_history_available": previous is not None,
        "constant_velocity_baseline_present": bool(constant_velocity_present),
    }
    error_row = {
        "row_id": "",
        "transition_id": str(transition["transition_id"]),
        "object_id": str(transition["object_id"]),
        "source_frame_id": str(transition["source_frame_id"]),
        "target_frame_id": str(transition["target_frame_id"]),
        "eval_status": "",
        "state_error": state_error,
        "hold_last_error": hold_last_error,
        "kalman_error": constant_velocity_error,
        "history_error": history_error,
        "reason": "",
    }
    return metrics, error_row


def _previous_pose(
    bundle: Mapping[str, Any],
    source: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    object_id = str(source["object_id"])
    source_ts = float(source["timestamp"])
    previous = [
        row
        for row in bundle["object_pose_rows"]
        if str(row["object_id"]) == object_id and float(row["timestamp"]) < source_ts
    ]
    if not previous:
        return None
    return sorted(previous, key=lambda row: float(row["timestamp"]))[-1]


def _constant_velocity_position(
    source: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    *,
    target_timestamp: float,
) -> tuple[list[float], bool]:
    source_position = list(source["object_pose_6dof"]["position"])
    if previous is None:
        return source_position, False
    dt = float(source["timestamp"]) - float(previous["timestamp"])
    horizon = target_timestamp - float(source["timestamp"])
    if dt <= 0.0 or horizon <= 0.0:
        return source_position, False
    previous_position = list(previous["object_pose_6dof"]["position"])
    return [
        float(source_position[index])
        + (
            (float(source_position[index]) - float(previous_position[index]))
            / dt
            * horizon
        )
        for index in range(3)
    ], True


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    evaluated = [row for row in rows if row["eval_status"] in _EVALUATED_STATUSES]
    metrics = _aggregate_metrics(evaluated, rows, readiness)
    baselines = _aggregate_baselines(evaluated)
    errors = []
    for row in rows:
        error = dict(row["error"])
        error["row_id"] = row["row_id"]
        error["eval_status"] = row["eval_status"]
        error["reason"] = row.get("reason") or ""
        errors.append(error)
    return {
        "row_counts": {
            "prediction_rows": len(rows),
            "evaluated_prediction_rows": len(evaluated),
            "prediction_ready_rows_consumed": sum(
                1 for row in rows if row["readiness"]["prediction_ready"]
            ),
            "prediction_rows_evidence_incomplete": sum(
                1 for row in rows if row["eval_status"] == "evidence_incomplete"
            ),
            "prediction_rows_blocked": sum(
                1 for row in rows if row["eval_status"] == "blocked"
            ),
            "prediction_pass_rows": sum(
                1 for row in rows if row["eval_status"] == "pass"
            ),
            "prediction_fail_rows": sum(
                1 for row in rows if row["eval_status"] == "fail"
            ),
        },
        "metrics": metrics,
        "baselines": baselines,
        "duplicate_row_ids": _duplicate_row_ids(rows),
        "errors": errors,
    }


def _aggregate_metrics(
    evaluated: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    ready_count = int(readiness["prediction_ready_rows"])
    result = {
        "state_ade": _mean(evaluated, "state_ade"),
        "state_fde": _mean(evaluated, "state_fde"),
        "pose_translation_error": _mean(evaluated, "pose_translation_error"),
        "pose_rotation_error": _mean(evaluated, "pose_rotation_error"),
        "hold_last_ade": _mean(evaluated, "hold_last_ade"),
        "kalman_ade": _mean(evaluated, "kalman_ade"),
        "history_ade": _mean(evaluated, "history_ade"),
        "state_vs_history_error_ratio": _mean(
            evaluated,
            "state_vs_history_error_ratio",
        ),
        "state_vs_kalman_error_ratio": _mean(
            evaluated,
            "state_vs_kalman_error_ratio",
        ),
        "transition_coverage": (
            0.0 if ready_count == 0 else float(len(evaluated) / ready_count)
        ),
        "identity_consistency_rate": _identity_consistency_rate(rows),
    }
    return result


def _aggregate_baselines(
    evaluated: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    history_available = sum(
        1 for row in evaluated if row["metrics"]["constant_velocity_history_available"]
    )
    constant_velocity_present = sum(
        1 for row in evaluated if row["metrics"]["constant_velocity_baseline_present"]
    )
    return {
        "hold_last": {
            "present": bool(evaluated),
            "ade": _mean(evaluated, "hold_last_ade"),
        },
        "constant_velocity": {
            "present": constant_velocity_present > 0,
            "ade": _mean(evaluated, "kalman_ade"),
            "history_available_rows": history_available,
            "constant_velocity_rows": constant_velocity_present,
            "fallback_without_previous_pose": "hold_last",
        },
        "history": {
            "present": bool(evaluated),
            "ade": _mean(evaluated, "history_ade"),
        },
    }


def _row_baselines(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "hold_last": {
            "present": bool(metrics.get("hold_last_ade", 0.0) >= 0.0),
            "ade": float(metrics.get("hold_last_ade", 0.0)),
        },
        "constant_velocity": {
            "present": bool(metrics.get("constant_velocity_baseline_present", False)),
            "ade": float(metrics.get("kalman_ade", 0.0)),
        },
        "history": {
            "present": bool(metrics.get("history_ade", 0.0) >= 0.0),
            "ade": float(metrics.get("history_ade", 0.0)),
        },
    }


def _pass_gates(
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> dict[str, bool]:
    counts = aggregate["row_counts"]
    metrics = aggregate["metrics"]
    baselines = aggregate["baselines"]
    history_allows = (
        int(baselines["constant_velocity"]["history_available_rows"]) > 0
    )
    return {
        "evaluated_prediction_rows_positive": (
            counts["evaluated_prediction_rows"] > 0
        ),
        "duplicate_row_ids_empty": not aggregate["duplicate_row_ids"],
        "hold_last_baseline_present": bool(baselines["hold_last"]["present"]),
        "constant_velocity_baseline_present_if_history_allows": (
            not history_allows or bool(baselines["constant_velocity"]["present"])
        ),
        "state_vs_baseline_metrics_present": all(
            _is_number(metrics.get(key))
            for key in (
                "state_vs_history_error_ratio",
                "state_vs_kalman_error_ratio",
                "hold_last_ade",
                "kalman_ade",
                "history_ade",
            )
        ),
        "evidence_incomplete_counted_separately": True,
        "identity_consistency_required": (
            counts["evaluated_prediction_rows"] > 0
            and metrics["identity_consistency_rate"] >= 1.0
        ),
        "state_ade_lte_hold_last_ade": (
            counts["evaluated_prediction_rows"] > 0
            and metrics["state_ade"] <= metrics["hold_last_ade"] + _EPSILON
        ),
        "no_prediction_fail_rows": counts["prediction_fail_rows"] == 0,
    }


def _summary_status(
    rows: Sequence[Mapping[str, Any]],
    pass_gates: Mapping[str, bool],
) -> str:
    if any(row["eval_status"] == "fail" for row in rows):
        return "objectstate_controlled_real_prediction_eval_fail"
    if rows and all(row["eval_status"] == "evidence_incomplete" for row in rows):
        return "objectstate_controlled_real_prediction_eval_incomplete"
    if rows and all(row["eval_status"] == "blocked" for row in rows):
        return "objectstate_controlled_real_prediction_eval_blocked"
    if all(pass_gates.values()):
        return "objectstate_controlled_real_prediction_eval_pass"
    if any(row["eval_status"] == "blocked" for row in rows):
        return "objectstate_controlled_real_prediction_eval_blocked"
    return "objectstate_controlled_real_prediction_eval_incomplete"


def _issues(
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> list[str]:
    issues = []
    if aggregate["duplicate_row_ids"]:
        issues.append("duplicate_row_ids")
    if aggregate["row_counts"]["evaluated_prediction_rows"] == 0:
        issues.append("no_evaluated_prediction_rows")
    if any(row.get("reason") == "identity_not_stable" for row in rows):
        issues.append("identity_not_stable")
    if any(row.get("reason") == "missing_prediction_candidates" for row in rows):
        issues.append("missing_prediction_candidates")
    if aggregate["row_counts"]["prediction_fail_rows"] > 0:
        issues.append("prediction_model_underperforms_hold_last")
    metrics = aggregate["metrics"]
    if (
        aggregate["row_counts"]["evaluated_prediction_rows"] > 0
        and metrics["state_ade"] > metrics["hold_last_ade"] + _EPSILON
    ):
        issues.append("prediction_model_underperforms_hold_last")
    return _unique_strings(issues)


def _identity_consistency_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    scoped = [
        row for row in rows if row["identity_consistency"].get("has_transition")
    ]
    if not scoped:
        return 0.0
    consistent = sum(
        1 for row in scoped if row["identity_consistency"].get("consistent")
    )
    return float(consistent / len(scoped))


def _duplicate_row_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    seen = set()
    duplicates = []
    for row in rows:
        row_id = str(row["row_id"])
        if row_id in seen and row_id not in duplicates:
            duplicates.append(row_id)
        seen.add(row_id)
    return duplicates


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [
        float(row["metrics"][key])
        for row in rows
        if _is_number(row.get("metrics", {}).get(key))
    ]
    return 0.0 if not values else float(sum(values) / len(values))


def _artifact_refs(
    source_row: Mapping[str, Any],
    *,
    candidates: Mapping[str, Any],
) -> list[str]:
    return _unique_strings(
        list(source_row["artifact_refs"])
        + list(candidates["candidate"].get("artifact_refs", ()))
    )


def _empty_metrics(identity_status: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state_ade": 0.0,
        "state_fde": 0.0,
        "pose_translation_error": 0.0,
        "pose_rotation_error": 0.0,
        "hold_last_ade": 0.0,
        "kalman_ade": 0.0,
        "history_ade": 0.0,
        "state_vs_history_error_ratio": 0.0,
        "state_vs_kalman_error_ratio": 0.0,
        "prediction_gap_vs_history_model": 0.0,
        "prediction_gap_vs_hold_last": 0.0,
        "prediction_gap_vs_kalman": 0.0,
        "transition_duration_s": 0.0,
        "identity_consistent": bool(identity_status.get("consistent", False)),
        "constant_velocity_history_available": False,
        "constant_velocity_baseline_present": False,
    }


def _empty_error_row(
    base: Mapping[str, Any],
    eval_status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "row_id": str(base["row_id"]),
        "transition_id": str(base.get("transition_id") or ""),
        "object_id": str(base.get("object_id") or ""),
        "source_frame_id": "",
        "target_frame_id": "",
        "eval_status": eval_status,
        "state_error": "",
        "hold_last_error": "",
        "kalman_error": "",
        "history_error": "",
        "reason": reason,
    }


def _reason(reasons: Sequence[str], fallback: str) -> str:
    for reason in reasons:
        if reason == "missing_evaluator_metrics":
            continue
        return str(reason)
    return fallback


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("distance vectors must have same length")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _rotation_error(left_xyzw: Sequence[float], right_xyzw: Sequence[float]) -> float:
    if len(left_xyzw) != 4 or len(right_xyzw) != 4:
        raise ValueError("rotation_xyzw must have length 4")
    left = _normalized_quaternion(left_xyzw)
    right = _normalized_quaternion(right_xyzw)
    dot = abs(sum(a * b for a, b in zip(left, right)))
    dot = min(1.0, max(-1.0, dot))
    return float(2.0 * math.acos(dot))


def _normalized_quaternion(value: Sequence[float]) -> tuple[float, float, float, float]:
    vector = tuple(float(item) for item in value)
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= 0.0:
        raise ValueError("rotation_xyzw must be non-zero")
    return tuple(item / norm for item in vector)  # type: ignore[return-value]


def _ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= _EPSILON:
        return 0.0 if abs(numerator) <= _EPSILON else 1.0e12
    return float(numerator / denominator)


def _validate_eval_row(row: Any) -> None:
    if not isinstance(row, Mapping):
        raise ValueError("controlled real prediction eval row must be a mapping")
    for key in ("row_id", "evidence_kind", "eval_status"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"controlled real prediction row requires {key}")
    if row["evidence_kind"] != "prediction":
        raise ValueError("controlled real prediction row evidence_kind mismatch")
    if row["eval_status"] not in {
        "pass",
        "fail",
        "blocked",
        "evidence_incomplete",
    }:
        raise ValueError("controlled real prediction row status unsupported")
    if not isinstance(row.get("readiness"), Mapping):
        raise ValueError("controlled real prediction row requires readiness")
    _mapping(row.get("metrics"), "row.metrics")
    _mapping(row.get("baselines"), "row.baselines")
    _mapping(row.get("pass_gates"), "row.pass_gates")
    validate_objectstate_real_evidence_bundle(
        {
            "schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
            "kind": "objectstate_real_evidence_bundle",
            "sample": {
                "sample_id": "validation-placeholder",
                "scene_id": "validation-placeholder",
                "sequence_id": "validation-placeholder",
                "source_dataset": "validation-placeholder",
                "source_kind": "controlled_real",
                "object_category": "validation-placeholder",
                "scenario": "validation-placeholder",
                "gt_provenance": "validation-placeholder",
                "license": "validation-placeholder",
                "observation_modalities": ["rgb"],
                "artifact_refs": ["validation-placeholder"],
            },
            "row_schemas": {
                "observation": "objgauss-objectstate-real-observation-row-v1",
                "object_pose": "objgauss-objectstate-real-object-pose-row-v1",
                "identity_link": "objgauss-objectstate-real-identity-link-row-v1",
                "action_interval": "objgauss-objectstate-real-action-interval-row-v1",
                "state_transition": "objgauss-objectstate-real-state-transition-row-v1",
                "gate_accounting": OBJECTSTATE_REAL_GATE_ACCOUNTING_ROW_SCHEMA,
            },
            "observation_rows": [
                {
                    "schema": "objgauss-objectstate-real-observation-row-v1",
                    "row_id": "validation-observation",
                    "frame_id": "validation-frame",
                    "timestamp": 0.0,
                    "camera_id": "validation-camera",
                    "observation": {"rgb": "validation.png"},
                }
            ],
            "object_pose_rows": [],
            "identity_link_rows": [],
            "action_interval_rows": [],
            "state_transition_rows": [],
            "gate_accounting_rows": [row["real_bundle_accounting_row"]],
        }
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return int(value)


def _metric(value: Any, name: str) -> float:
    if not _is_number(value):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _unique_strings(values: Sequence[str]) -> list[str]:
    result = []
    for value in values:
        text = str(value)
        if text and text not in result:
            result.append(text)
    return result
