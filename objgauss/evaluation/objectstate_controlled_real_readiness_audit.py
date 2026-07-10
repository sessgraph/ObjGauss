from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.datasets.objectstate_real_evidence_bundle import (
    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
    read_objectstate_real_evidence_bundle,
    validate_objectstate_real_evidence_bundle,
)

OBJECTSTATE_CONTROLLED_REAL_READINESS_AUDIT_SCHEMA = (
    "objgauss-objectstate-controlled-real-readiness-audit-v1"
)

CONTROLLED_REAL_READINESS_BLOCK_REASONS = (
    "missing_identity_link",
    "missing_before_pose",
    "missing_after_pose",
    "missing_action_vector",
    "action_interval_no_transition_overlap",
    "missing_teacher_evidence",
    "missing_evaluator_metrics",
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_REAL_READINESS_AUDIT_SCHEMA",
    "CONTROLLED_REAL_READINESS_BLOCK_REASONS",
    "objectstate_controlled_real_readiness_audit_from_file",
    "objectstate_controlled_real_readiness_audit",
    "objectstate_controlled_real_readiness_markdown",
    "objectstate_controlled_real_readiness_breakdown_csv",
    "validate_objectstate_controlled_real_readiness_audit",
)


def objectstate_controlled_real_readiness_audit_from_file(
    bundle_path: str | Path,
) -> dict[str, Any]:
    return objectstate_controlled_real_readiness_audit(
        read_objectstate_real_evidence_bundle(bundle_path),
        bundle_ref=str(bundle_path),
    )


def objectstate_controlled_real_readiness_audit(
    bundle: Mapping[str, Any],
    *,
    bundle_ref: str | None = None,
) -> dict[str, Any]:
    checked = validate_objectstate_real_evidence_bundle(bundle)
    observations = checked["observation_rows"]
    pose_rows = checked["object_pose_rows"]
    identity_links = checked["identity_link_rows"]
    actions = checked["action_interval_rows"]
    transitions = checked["state_transition_rows"]
    accounting_rows = checked["gate_accounting_rows"]
    indexes = _indexes(observations, pose_rows, identity_links, actions, transitions)
    row_breakdown = [
        _audit_accounting_row(row, indexes)
        for row in accounting_rows
    ]
    blocked_reasons = {
        reason: sum(1 for row in row_breakdown if reason in row["blocked_reasons"])
        for reason in CONTROLLED_REAL_READINESS_BLOCK_REASONS
    }
    metrics = {
        "row_count": len(row_breakdown),
        "identity_ready_rows": _ready_count(row_breakdown, "identity"),
        "prediction_ready_rows": _ready_count(row_breakdown, "prediction"),
        "intervention_ready_rows": _ready_count(row_breakdown, "intervention"),
        "evidence_incomplete_rows": sum(
            1 for row in row_breakdown if row["accounting_status"] == "evidence_incomplete"
        ),
        "unsupported_rows": sum(
            1 for row in row_breakdown if row["accounting_status"] == "unsupported"
        ),
        "pass_rows": sum(1 for row in row_breakdown if row["accounting_status"] == "pass"),
        "fail_rows": sum(1 for row in row_breakdown if row["accounting_status"] == "fail"),
        "blocked_rows": sum(1 for row in row_breakdown if row["blocked"]),
    }
    readiness = {
        "identity_ready": metrics["identity_ready_rows"] > 0,
        "prediction_ready": metrics["prediction_ready_rows"] > 0,
        "intervention_ready": metrics["intervention_ready_rows"] > 0,
        "controlled_real_source": checked["sample"]["source_kind"] == "controlled_real",
        "evidence_incomplete_is_not_model_fail": True,
    }
    readiness["all_gate_inputs_ready"] = all(
        (
            readiness["identity_ready"],
            readiness["prediction_ready"],
            readiness["intervention_ready"],
        )
    )
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_REAL_READINESS_AUDIT_SCHEMA,
        "kind": "objectstate_controlled_real_readiness_audit",
        "status": (
            "objectstate_controlled_real_readiness_ready"
            if readiness["all_gate_inputs_ready"]
            else "objectstate_controlled_real_readiness_incomplete"
        ),
        "bundle_schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
        "bundle_ref": bundle_ref,
        "bundle_id": checked["sample"]["sample_id"],
        "sample": dict(checked["sample"]),
        "readiness": readiness,
        "row_count": metrics["row_count"],
        "identity_ready_rows": metrics["identity_ready_rows"],
        "prediction_ready_rows": metrics["prediction_ready_rows"],
        "intervention_ready_rows": metrics["intervention_ready_rows"],
        "evidence_incomplete_rows": metrics["evidence_incomplete_rows"],
        "unsupported_rows": metrics["unsupported_rows"],
        "pass_rows": metrics["pass_rows"],
        "fail_rows": metrics["fail_rows"],
        "blocked_rows": metrics["blocked_rows"],
        "blocked_reasons": blocked_reasons,
        "row_breakdown": row_breakdown,
        "claim_policy": {
            "reads_existing_real_evidence_bundle": True,
            "reports_evaluator_input_readiness": True,
            "evidence_incomplete_is_not_model_fail": True,
            "does_not_run_identity_eval": True,
            "does_not_run_prediction_eval": True,
            "does_not_run_intervention_eval": True,
            "does_not_create_pass_fail_rows": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "reconstructs_gaussians": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "runs_identity_eval": False,
            "runs_prediction_eval": False,
            "runs_intervention_eval": False,
            "creates_reality_rows": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_controlled_real_readiness_audit(payload)


def objectstate_controlled_real_readiness_markdown(
    summary: Mapping[str, Any],
) -> str:
    checked = validate_objectstate_controlled_real_readiness_audit(summary)
    lines = [
        f"# Controlled Real Readiness Audit: {checked['bundle_id']}",
        "",
        f"- status: `{checked['status']}`",
        f"- identity_ready_rows: `{checked['identity_ready_rows']}`",
        f"- prediction_ready_rows: `{checked['prediction_ready_rows']}`",
        f"- intervention_ready_rows: `{checked['intervention_ready_rows']}`",
        f"- evidence_incomplete_rows: `{checked['evidence_incomplete_rows']}`",
        f"- unsupported_rows: `{checked['unsupported_rows']}`",
        f"- blocked_rows: `{checked['blocked_rows']}`",
        "",
        "## Blocked Reasons",
        "",
    ]
    for reason, count in checked["blocked_reasons"].items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(("", "## Rows", ""))
    for row in checked["row_breakdown"]:
        reason_text = ",".join(row["blocked_reasons"]) or "none"
        lines.append(
            "- "
            f"`{row['row_id']}` "
            f"{row['evidence_kind']} "
            f"ready={str(row['evaluator_ready']).lower()} "
            f"status={row['accounting_status']} "
            f"reasons={reason_text}"
        )
    lines.append("")
    return "\n".join(lines)


def objectstate_controlled_real_readiness_breakdown_csv(
    summary: Mapping[str, Any],
) -> str:
    checked = validate_objectstate_controlled_real_readiness_audit(summary)
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "row_id",
            "evidence_kind",
            "accounting_status",
            "evaluator_ready",
            "blocked",
            "blocked_reasons",
            "object_id",
            "action_id",
            "transition_id",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for row in checked["row_breakdown"]:
        writer.writerow(
            {
                "row_id": row["row_id"],
                "evidence_kind": row["evidence_kind"],
                "accounting_status": row["accounting_status"],
                "evaluator_ready": str(row["evaluator_ready"]).lower(),
                "blocked": str(row["blocked"]).lower(),
                "blocked_reasons": ";".join(row["blocked_reasons"]),
                "object_id": row.get("object_id", ""),
                "action_id": row.get("action_id", ""),
                "transition_id": row.get("transition_id", ""),
            }
        )
    return output.getvalue()


def validate_objectstate_controlled_real_readiness_audit(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled real readiness audit must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_REAL_READINESS_AUDIT_SCHEMA:
        raise ValueError(
            "unsupported controlled real readiness audit schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_real_readiness_audit":
        raise ValueError("controlled real readiness audit kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_real_readiness_ready",
        "objectstate_controlled_real_readiness_incomplete",
    }:
        raise ValueError("controlled real readiness audit status is unsupported")
    if payload.get("bundle_schema") != OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA:
        raise ValueError("controlled real readiness audit bundle schema mismatch")
    if not isinstance(payload.get("bundle_id"), str) or not payload["bundle_id"]:
        raise ValueError("controlled real readiness audit requires bundle_id")
    if not isinstance(payload.get("sample"), Mapping):
        raise ValueError("controlled real readiness audit requires sample")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("controlled real readiness audit requires readiness")
    for key in (
        "identity_ready",
        "prediction_ready",
        "intervention_ready",
        "controlled_real_source",
        "evidence_incomplete_is_not_model_fail",
        "all_gate_inputs_ready",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"controlled real readiness missing bool {key}")
    for key in (
        "row_count",
        "identity_ready_rows",
        "prediction_ready_rows",
        "intervention_ready_rows",
        "evidence_incomplete_rows",
        "unsupported_rows",
        "pass_rows",
        "fail_rows",
        "blocked_rows",
    ):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"controlled real readiness requires int {key}")
    reasons = payload.get("blocked_reasons")
    if not isinstance(reasons, Mapping):
        raise ValueError("controlled real readiness requires blocked_reasons")
    for reason in CONTROLLED_REAL_READINESS_BLOCK_REASONS:
        value = reasons.get(reason)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"blocked_reasons.{reason} must be a non-negative int")
    rows = payload.get("row_breakdown")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("controlled real readiness requires row_breakdown")
    if len(rows) != payload["row_count"]:
        raise ValueError("controlled real readiness row_count mismatch")
    for row in rows:
        _validate_breakdown_row(row)
    expected_status = (
        "objectstate_controlled_real_readiness_ready"
        if readiness["all_gate_inputs_ready"]
        else "objectstate_controlled_real_readiness_incomplete"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled real readiness status mismatch")
    claim_policy = payload.get("claim_policy", {})
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("reads_existing_real_evidence_bundle")
        or not claim_policy.get("reports_evaluator_input_readiness")
        or not claim_policy.get("evidence_incomplete_is_not_model_fail")
        or not claim_policy.get("does_not_run_identity_eval")
        or not claim_policy.get("does_not_run_prediction_eval")
        or not claim_policy.get("does_not_run_intervention_eval")
        or not claim_policy.get("does_not_create_pass_fail_rows")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled real readiness audit must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError("controlled real readiness audit cannot claim non-goal behavior")
    return dict(payload)


def _validate_breakdown_row(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("controlled real readiness breakdown row must be a mapping")
    for key in ("row_id", "evidence_kind", "accounting_status"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"controlled real readiness row requires {key}")
    if value["evidence_kind"] not in {"identity", "prediction", "intervention"}:
        raise ValueError("controlled real readiness row evidence_kind is unsupported")
    if value["accounting_status"] not in {
        "pass",
        "fail",
        "evidence_incomplete",
        "unsupported",
    }:
        raise ValueError("controlled real readiness row accounting_status is unsupported")
    for key in ("evaluator_ready", "blocked"):
        if not isinstance(value.get(key), bool):
            raise ValueError(f"controlled real readiness row requires bool {key}")
    reasons = value.get("blocked_reasons")
    if isinstance(reasons, (str, bytes)) or not isinstance(reasons, Sequence):
        raise ValueError("controlled real readiness row requires blocked_reasons")
    for reason in reasons:
        if reason not in CONTROLLED_REAL_READINESS_BLOCK_REASONS:
            raise ValueError(f"controlled real readiness blocker unsupported: {reason}")


def _indexes(
    observations: Sequence[Mapping[str, Any]],
    pose_rows: Sequence[Mapping[str, Any]],
    identity_links: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    identity_by_object: dict[str, list[Mapping[str, Any]]] = {}
    physical_by_object_frame: dict[tuple[str, str], str] = {}
    for row in identity_links:
        object_id = str(row["object_id"])
        identity_by_object.setdefault(object_id, []).append(row)
        physical_by_object_frame[(object_id, str(row["frame_id"]))] = str(
            row["physical_identity_id"]
        )
    return {
        "observation_frames": {str(row["frame_id"]) for row in observations},
        "pose_by_id": {str(row["row_id"]): row for row in pose_rows},
        "identity_by_object": identity_by_object,
        "physical_by_object_frame": physical_by_object_frame,
        "action_by_id": {str(row["action_id"]): row for row in actions},
        "transition_by_id": {str(row["transition_id"]): row for row in transitions},
    }


def _audit_accounting_row(
    row: Mapping[str, Any],
    indexes: Mapping[str, Any],
) -> dict[str, Any]:
    evidence_kind = str(row["evidence_kind"])
    if evidence_kind == "identity":
        structural_reasons = _identity_reasons(row, indexes)
    elif evidence_kind == "prediction":
        structural_reasons = _prediction_reasons(row, indexes)
    else:
        structural_reasons = _intervention_reasons(row, indexes)
    evaluator_ready = not structural_reasons
    reasons = list(structural_reasons)
    if evaluator_ready and row["accounting_status"] == "evidence_incomplete":
        reasons.append("missing_evaluator_metrics")
    blocked = bool(reasons) or row["accounting_status"] == "unsupported"
    return {
        "row_id": str(row["row_id"]),
        "evidence_kind": evidence_kind,
        "accounting_status": str(row["accounting_status"]),
        "evaluator_ready": evaluator_ready,
        "blocked": blocked,
        "blocked_reasons": reasons,
        "object_id": row.get("object_id"),
        "action_id": row.get("action_id"),
        "transition_id": row.get("transition_id"),
    }


def _identity_reasons(
    row: Mapping[str, Any],
    indexes: Mapping[str, Any],
) -> list[str]:
    object_id = row.get("object_id")
    if isinstance(object_id, str) and object_id:
        return [] if _object_has_identity_track(object_id, indexes) else ["missing_identity_link"]
    if any(_object_has_identity_track(object_id, indexes) for object_id in indexes["identity_by_object"]):
        return []
    return ["missing_identity_link"]


def _prediction_reasons(
    row: Mapping[str, Any],
    indexes: Mapping[str, Any],
) -> list[str]:
    transition = _row_transition(row, indexes)
    if transition is None:
        return ["missing_before_pose", "missing_after_pose"]
    return _transition_reasons(transition, indexes)


def _intervention_reasons(
    row: Mapping[str, Any],
    indexes: Mapping[str, Any],
) -> list[str]:
    reasons = []
    transition = _row_transition(row, indexes)
    if transition is None:
        reasons.extend(("missing_before_pose", "missing_after_pose"))
    else:
        reasons.extend(_transition_reasons(transition, indexes))
    action_id = row.get("action_id")
    action = indexes["action_by_id"].get(str(action_id)) if isinstance(action_id, str) else None
    if action is None or not _nonzero_vector(action.get("action_vector")):
        reasons.append("missing_action_vector")
    elif transition is None or not _action_transition_overlap(action, transition):
        reasons.append("action_interval_no_transition_overlap")
    return _unique(reasons)


def _transition_reasons(
    transition: Mapping[str, Any],
    indexes: Mapping[str, Any],
) -> list[str]:
    reasons = []
    source = indexes["pose_by_id"].get(str(transition["source_pose_row_id"]))
    target = indexes["pose_by_id"].get(str(transition["target_pose_row_id"]))
    if source is None:
        reasons.append("missing_before_pose")
    if target is None:
        reasons.append("missing_after_pose")
    if source is None or target is None:
        return reasons
    object_id = str(transition["object_id"])
    source_identity = indexes["physical_by_object_frame"].get(
        (object_id, str(transition["source_frame_id"]))
    )
    target_identity = indexes["physical_by_object_frame"].get(
        (object_id, str(transition["target_frame_id"]))
    )
    if not source_identity or not target_identity or source_identity != target_identity:
        reasons.append("missing_identity_link")
    return reasons


def _row_transition(
    row: Mapping[str, Any],
    indexes: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    transition_id = row.get("transition_id")
    if isinstance(transition_id, str) and transition_id:
        return indexes["transition_by_id"].get(transition_id)
    return None


def _object_has_identity_track(
    object_id: str,
    indexes: Mapping[str, Any],
) -> bool:
    links = indexes["identity_by_object"].get(object_id, ())
    timestamps = {float(row["timestamp"]) for row in links}
    frames = {str(row["frame_id"]) for row in links}
    return len(timestamps) >= 2 and frames.issubset(indexes["observation_frames"])


def _action_transition_overlap(
    action: Mapping[str, Any],
    transition: Mapping[str, Any],
) -> bool:
    referenced_objects = {str(action["object_id"])}
    if action.get("target_object_id"):
        referenced_objects.add(str(action["target_object_id"]))
    if str(transition["object_id"]) not in referenced_objects:
        return False
    return max(float(action["action_start_ts"]), float(transition["source_timestamp"])) < min(
        float(action["action_end_ts"]),
        float(transition["target_timestamp"]),
    )


def _nonzero_vector(value: Any) -> bool:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return False
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return len(vector) == 3 and any(abs(item) > 1.0e-12 for item in vector)


def _ready_count(rows: Sequence[Mapping[str, Any]], evidence_kind: str) -> int:
    return sum(
        1
        for row in rows
        if row["evidence_kind"] == evidence_kind and row["evaluator_ready"]
    )


def _unique(values: Sequence[str]) -> list[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
