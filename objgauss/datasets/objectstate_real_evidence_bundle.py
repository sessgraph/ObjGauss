from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA = (
    "objgauss-objectstate-real-evidence-bundle-v1"
)
OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SUMMARY_SCHEMA = (
    "objgauss-objectstate-real-evidence-bundle-summary-v1"
)
OBJECTSTATE_REAL_OBSERVATION_ROW_SCHEMA = (
    "objgauss-objectstate-real-observation-row-v1"
)
OBJECTSTATE_REAL_OBJECT_POSE_ROW_SCHEMA = (
    "objgauss-objectstate-real-object-pose-row-v1"
)
OBJECTSTATE_REAL_IDENTITY_LINK_ROW_SCHEMA = (
    "objgauss-objectstate-real-identity-link-row-v1"
)
OBJECTSTATE_REAL_ACTION_INTERVAL_ROW_SCHEMA = (
    "objgauss-objectstate-real-action-interval-row-v1"
)
OBJECTSTATE_REAL_STATE_TRANSITION_ROW_SCHEMA = (
    "objgauss-objectstate-real-state-transition-row-v1"
)
OBJECTSTATE_REAL_GATE_ACCOUNTING_ROW_SCHEMA = (
    "objgauss-objectstate-real-gate-accounting-row-v1"
)

OBJECTSTATE_REAL_GATE_EVIDENCE_KINDS = (
    "identity",
    "prediction",
    "intervention",
)
OBJECTSTATE_REAL_GATE_ACCOUNTING_STATUSES = (
    "pass",
    "fail",
    "evidence_incomplete",
    "unsupported",
)
_ACCOUNTING_STATUSES_REQUIRING_EVIDENCE = {"pass", "fail"}

__all__ = (
    "OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA",
    "OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SUMMARY_SCHEMA",
    "OBJECTSTATE_REAL_OBSERVATION_ROW_SCHEMA",
    "OBJECTSTATE_REAL_OBJECT_POSE_ROW_SCHEMA",
    "OBJECTSTATE_REAL_IDENTITY_LINK_ROW_SCHEMA",
    "OBJECTSTATE_REAL_ACTION_INTERVAL_ROW_SCHEMA",
    "OBJECTSTATE_REAL_STATE_TRANSITION_ROW_SCHEMA",
    "OBJECTSTATE_REAL_GATE_ACCOUNTING_ROW_SCHEMA",
    "OBJECTSTATE_REAL_GATE_EVIDENCE_KINDS",
    "OBJECTSTATE_REAL_GATE_ACCOUNTING_STATUSES",
    "read_objectstate_real_evidence_bundle",
    "objectstate_real_evidence_bundle_summary",
    "validate_objectstate_real_evidence_bundle",
    "validate_objectstate_real_evidence_bundle_summary",
)


def read_objectstate_real_evidence_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path)
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("real evidence bundle JSON must be an object")
    return validate_objectstate_real_evidence_bundle(payload)


def objectstate_real_evidence_bundle_summary(
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    checked = validate_objectstate_real_evidence_bundle(bundle)
    observations = checked["observation_rows"]
    pose_rows = checked["object_pose_rows"]
    identity_links = checked["identity_link_rows"]
    actions = checked["action_interval_rows"]
    transitions = checked["state_transition_rows"]
    accounting_rows = checked["gate_accounting_rows"]
    overlaps = _action_transition_overlaps(actions, transitions)
    intervention_accounting_rows = [
        row for row in accounting_rows if row["evidence_kind"] == "intervention"
    ]
    actionable_intervention_rows = [
        row
        for row in intervention_accounting_rows
        if row["accounting_status"] in _ACCOUNTING_STATUSES_REQUIRING_EVIDENCE
    ]
    readiness = {
        "observation_rows_present": bool(observations),
        "object_pose_rows_present": bool(pose_rows),
        "identity_link_rows_present": bool(identity_links),
        "state_transition_rows_present": bool(transitions),
        "gate_accounting_rows_present": bool(accounting_rows),
        "action_interval_rows_present": bool(actions),
        "action_transition_overlap_ready": bool(overlaps),
        "intervention_accounting_refs_ready": all(
            _intervention_accounting_row_has_valid_overlap(row, overlaps)
            for row in actionable_intervention_rows
        ),
    }
    readiness["state_variable_evidence_ready"] = all(
        (
            readiness["observation_rows_present"],
            readiness["object_pose_rows_present"],
            readiness["identity_link_rows_present"],
            readiness["state_transition_rows_present"],
            readiness["gate_accounting_rows_present"],
        )
    )
    readiness["intervention_accounting_ready"] = (
        bool(actionable_intervention_rows)
        and readiness["action_interval_rows_present"]
        and readiness["action_transition_overlap_ready"]
        and readiness["intervention_accounting_refs_ready"]
    )
    metrics = {
        "observation_row_count": len(observations),
        "object_pose_row_count": len(pose_rows),
        "identity_link_row_count": len(identity_links),
        "action_interval_row_count": len(actions),
        "state_transition_row_count": len(transitions),
        "gate_accounting_row_count": len(accounting_rows),
        "action_transition_overlap_count": len(overlaps),
        "action_transition_coverage_rate": (
            0.0 if not actions else float(len({item[0] for item in overlaps}) / len(actions))
        ),
        "gate_accounting_status_counts": _counts(
            row["accounting_status"] for row in accounting_rows
        ),
        "gate_accounting_evidence_kind_counts": _counts(
            row["evidence_kind"] for row in accounting_rows
        ),
    }
    hard_blockers = _summary_blockers(readiness, actionable_intervention_rows)
    payload = {
        "schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SUMMARY_SCHEMA,
        "kind": "objectstate_real_evidence_bundle_summary",
        "status": (
            "objectstate_real_evidence_bundle_ready"
            if readiness["state_variable_evidence_ready"]
            else "objectstate_real_evidence_bundle_incomplete"
        ),
        "bundle_schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
        "sample": dict(checked["sample"]),
        "row_schemas": dict(checked["row_schemas"]),
        "readiness": readiness,
        "metrics": metrics,
        "hard_blockers": hard_blockers,
        "evidence_accounts": {
            "static_scene_evidence": {
                "available": bool(observations),
                "usable_for_state_variable_gate": False,
            },
            "state_variable_evidence": {
                "available": bool(readiness["state_variable_evidence_ready"]),
                "requires_timestamped_identity_pose_action_for_intervention": True,
            },
        },
        "claim_policy": {
            "bundle_is_evidence_authoring_contract": True,
            "evidence_incomplete_is_not_model_fail": True,
            "static_scene_evidence_is_separate_from_state_variable_evidence": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_counterfactual_proof": True,
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
    return validate_objectstate_real_evidence_bundle_summary(payload)


def validate_objectstate_real_evidence_bundle(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("real evidence bundle must be a mapping")
    if payload.get("schema") != OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA:
        raise ValueError(
            "unsupported real evidence bundle schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_real_evidence_bundle":
        raise ValueError("real evidence bundle kind is unsupported")
    sample = _validate_sample(payload.get("sample"))
    row_schemas = _validate_row_schemas(payload.get("row_schemas"))
    observations = tuple(
        _validate_observation_row(row)
        for row in _sequence(payload.get("observation_rows"), "observation_rows")
    )
    pose_rows = tuple(
        _validate_object_pose_row(row)
        for row in _sequence(payload.get("object_pose_rows"), "object_pose_rows")
    )
    identity_links = tuple(
        _validate_identity_link_row(row)
        for row in _sequence(payload.get("identity_link_rows"), "identity_link_rows")
    )
    actions = tuple(
        _validate_action_interval_row(row)
        for row in _sequence(payload.get("action_interval_rows", ()), "action_interval_rows")
    )
    transitions = tuple(
        _validate_state_transition_row(row)
        for row in _sequence(payload.get("state_transition_rows"), "state_transition_rows")
    )
    accounting_rows = tuple(
        _validate_gate_accounting_row(row)
        for row in _sequence(payload.get("gate_accounting_rows"), "gate_accounting_rows")
    )
    _validate_unique_ids("observation row", (row["row_id"] for row in observations))
    _validate_unique_ids("object pose row", (row["row_id"] for row in pose_rows))
    _validate_unique_ids("identity link row", (row["row_id"] for row in identity_links))
    _validate_unique_ids("action interval row", (row["action_id"] for row in actions))
    _validate_unique_ids("state transition row", (row["transition_id"] for row in transitions))
    _validate_unique_ids("gate accounting row", (row["row_id"] for row in accounting_rows))
    observation_frames = {row["frame_id"] for row in observations}
    pose_keys = {(row["object_id"], row["frame_id"]) for row in pose_rows}
    pose_rows_by_id = {row["row_id"]: row for row in pose_rows}
    action_ids = {row["action_id"] for row in actions}
    transition_ids = {row["transition_id"] for row in transitions}
    for row in pose_rows:
        if row["frame_id"] not in observation_frames:
            raise ValueError(
                f"object pose row references unknown frame_id: {row['frame_id']}"
            )
    for row in identity_links:
        if (row["object_id"], row["frame_id"]) not in pose_keys:
            raise ValueError(
                "identity link row must reference an object pose row for the same "
                f"object/frame: {row['object_id']} {row['frame_id']}"
            )
    for row in transitions:
        source_pose = pose_rows_by_id.get(row["source_pose_row_id"])
        target_pose = pose_rows_by_id.get(row["target_pose_row_id"])
        if (
            source_pose is None
            or source_pose["object_id"] != row["object_id"]
            or source_pose["frame_id"] != row["source_frame_id"]
        ):
            raise ValueError(
                "state transition row source must reference object pose row: "
                f"{row['transition_id']}"
            )
        if (
            target_pose is None
            or target_pose["object_id"] != row["object_id"]
            or target_pose["frame_id"] != row["target_frame_id"]
        ):
            raise ValueError(
                "state transition row target must reference object pose row: "
                f"{row['transition_id']}"
            )
    overlaps = _action_transition_overlaps(actions, transitions)
    for row in accounting_rows:
        if row["evidence_kind"] != "intervention":
            continue
        if row["accounting_status"] not in _ACCOUNTING_STATUSES_REQUIRING_EVIDENCE:
            continue
        action_id = row.get("action_id")
        transition_id = row.get("transition_id")
        if not isinstance(action_id, str) or action_id not in action_ids:
            raise ValueError(
                "intervention accounting rows with pass/fail status require a "
                "known action_id"
            )
        if not isinstance(transition_id, str) or transition_id not in transition_ids:
            raise ValueError(
                "intervention accounting rows with pass/fail status require a "
                "known transition_id"
            )
        if (action_id, transition_id) not in overlaps:
            raise ValueError(
                "intervention accounting action interval must overlap the "
                "referenced object state transition"
            )
    checked = {
        "schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
        "kind": "objectstate_real_evidence_bundle",
        "sample": sample,
        "row_schemas": row_schemas,
        "observation_rows": observations,
        "object_pose_rows": pose_rows,
        "identity_link_rows": identity_links,
        "action_interval_rows": actions,
        "state_transition_rows": transitions,
        "gate_accounting_rows": accounting_rows,
    }
    return checked


def validate_objectstate_real_evidence_bundle_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("real evidence bundle summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SUMMARY_SCHEMA:
        raise ValueError(
            "unsupported real evidence bundle summary schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_real_evidence_bundle_summary":
        raise ValueError("real evidence bundle summary kind is unsupported")
    if payload.get("status") not in {
        "objectstate_real_evidence_bundle_ready",
        "objectstate_real_evidence_bundle_incomplete",
    }:
        raise ValueError("real evidence bundle summary status is unsupported")
    _validate_sample(payload.get("sample"))
    _validate_row_schemas(payload.get("row_schemas"))
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("real evidence bundle summary requires readiness")
    for key in (
        "observation_rows_present",
        "object_pose_rows_present",
        "identity_link_rows_present",
        "state_transition_rows_present",
        "gate_accounting_rows_present",
        "action_interval_rows_present",
        "action_transition_overlap_ready",
        "intervention_accounting_refs_ready",
        "state_variable_evidence_ready",
        "intervention_accounting_ready",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"real evidence bundle readiness missing bool {key}")
    expected_status = (
        "objectstate_real_evidence_bundle_ready"
        if readiness["state_variable_evidence_ready"]
        else "objectstate_real_evidence_bundle_incomplete"
    )
    if payload["status"] != expected_status:
        raise ValueError("real evidence bundle summary status must match readiness")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("real evidence bundle summary requires metrics")
    for key in (
        "observation_row_count",
        "object_pose_row_count",
        "identity_link_row_count",
        "action_interval_row_count",
        "state_transition_row_count",
        "gate_accounting_row_count",
        "action_transition_overlap_count",
    ):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"real evidence bundle metric requires int {key}")
    coverage = metrics.get("action_transition_coverage_rate")
    if (
        isinstance(coverage, bool)
        or not isinstance(coverage, (int, float))
        or not math.isfinite(float(coverage))
        or float(coverage) < 0.0
        or float(coverage) > 1.0
    ):
        raise ValueError("action_transition_coverage_rate must be finite in [0, 1]")
    for key in ("gate_accounting_status_counts", "gate_accounting_evidence_kind_counts"):
        counts = metrics.get(key)
        if not isinstance(counts, Mapping) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts.values()
        ):
            raise ValueError(f"real evidence bundle metric requires counts {key}")
    if not isinstance(payload.get("hard_blockers"), list):
        raise ValueError("real evidence bundle summary requires hard_blockers")
    accounts = payload.get("evidence_accounts")
    if not isinstance(accounts, Mapping):
        raise ValueError("real evidence bundle summary requires evidence_accounts")
    for key in ("static_scene_evidence", "state_variable_evidence"):
        if not isinstance(accounts.get(key), Mapping):
            raise ValueError(f"real evidence bundle missing evidence account {key}")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("bundle_is_evidence_authoring_contract")
        or not claim_policy.get("evidence_incomplete_is_not_model_fail")
        or not claim_policy.get("static_scene_evidence_is_separate_from_state_variable_evidence")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_counterfactual_proof")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("real evidence bundle summary must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("reconstructs_gaussians")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("runs_identity_eval")
        or non_goals.get("runs_prediction_eval")
        or non_goals.get("runs_intervention_eval")
        or non_goals.get("creates_reality_rows")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError(
            "real evidence bundle summary cannot claim capture, GT creation, "
            "reconstruction, training, eval, row creation, replay, diffusion, "
            "or viewer mutation"
        )
    return dict(payload)


def _validate_sample(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("real evidence bundle requires sample")
    result = {
        "sample_id": _required_str(value, "sample_id"),
        "scene_id": _required_str(value, "scene_id"),
        "sequence_id": _required_str(value, "sequence_id"),
        "source_dataset": _required_str(value, "source_dataset"),
        "source_kind": _required_str(value, "source_kind"),
        "object_category": _required_str(value, "object_category"),
        "scenario": _required_str(value, "scenario"),
        "gt_provenance": _required_str(value, "gt_provenance"),
        "license": _required_str(value, "license"),
        "observation_modalities": list(
            _non_empty_string_tuple(
                value.get("observation_modalities"),
                "observation_modalities",
            )
        ),
        "artifact_refs": list(
            _non_empty_string_tuple(value.get("artifact_refs"), "artifact_refs")
        ),
    }
    if result["source_kind"] not in {
        "controlled_real",
        "public_replay",
        "open_world_real",
    }:
        raise ValueError("real evidence bundle source_kind is unsupported")
    return result


def _validate_row_schemas(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("real evidence bundle requires row_schemas")
    expected = {
        "observation": OBJECTSTATE_REAL_OBSERVATION_ROW_SCHEMA,
        "object_pose": OBJECTSTATE_REAL_OBJECT_POSE_ROW_SCHEMA,
        "identity_link": OBJECTSTATE_REAL_IDENTITY_LINK_ROW_SCHEMA,
        "action_interval": OBJECTSTATE_REAL_ACTION_INTERVAL_ROW_SCHEMA,
        "state_transition": OBJECTSTATE_REAL_STATE_TRANSITION_ROW_SCHEMA,
        "gate_accounting": OBJECTSTATE_REAL_GATE_ACCOUNTING_ROW_SCHEMA,
    }
    for key, schema in expected.items():
        if value.get(key) != schema:
            raise ValueError(f"real evidence bundle row_schemas.{key} mismatch")
    return dict(expected)


def _validate_observation_row(value: Any) -> dict[str, Any]:
    row = _row_mapping(value, "observation row")
    observation = row.get("observation")
    if not isinstance(observation, Mapping):
        raise ValueError("observation row requires observation")
    result = {
        "schema": _schema(row, OBJECTSTATE_REAL_OBSERVATION_ROW_SCHEMA),
        "row_id": _required_str(row, "row_id"),
        "frame_id": _required_str(row, "frame_id"),
        "timestamp": _float(row, "timestamp"),
        "camera_id": _required_str(row, "camera_id"),
        "observation": {},
    }
    for key in ("rgb", "gaussian"):
        item = observation.get(key)
        if item is not None:
            result["observation"][key] = _non_empty_str(item, f"observation.{key}")
    if not result["observation"]:
        raise ValueError("observation row requires at least one observation ref")
    return result


def _validate_object_pose_row(value: Any) -> dict[str, Any]:
    row = _row_mapping(value, "object pose row")
    pose = row.get("object_pose_6dof")
    if not isinstance(pose, Mapping):
        raise ValueError("object pose row requires object_pose_6dof")
    result = {
        "schema": _schema(row, OBJECTSTATE_REAL_OBJECT_POSE_ROW_SCHEMA),
        "row_id": _required_str(row, "row_id"),
        "frame_id": _required_str(row, "frame_id"),
        "timestamp": _float(row, "timestamp"),
        "camera_id": _required_str(row, "camera_id"),
        "object_id": _required_str(row, "object_id"),
        "object_pose_6dof": {
            "position": _float_vector(pose.get("position"), 3, "position"),
            "rotation_xyzw": _float_vector(
                pose.get("rotation_xyzw"),
                4,
                "rotation_xyzw",
            ),
        },
        "object_visibility": _float(row, "object_visibility"),
    }
    visibility = float(result["object_visibility"])
    if visibility < 0.0 or visibility > 1.0:
        raise ValueError("object_visibility must be in [0, 1]")
    return result


def _validate_identity_link_row(value: Any) -> dict[str, Any]:
    row = _row_mapping(value, "identity link row")
    result = {
        "schema": _schema(row, OBJECTSTATE_REAL_IDENTITY_LINK_ROW_SCHEMA),
        "row_id": _required_str(row, "row_id"),
        "frame_id": _required_str(row, "frame_id"),
        "timestamp": _float(row, "timestamp"),
        "object_id": _required_str(row, "object_id"),
        "physical_identity_id": _required_str(row, "physical_identity_id"),
        "gt_provenance": _required_str(row, "gt_provenance"),
    }
    confidence = row.get("confidence")
    if confidence is not None:
        result["confidence"] = _finite_float(confidence, "confidence")
    return result


def _validate_action_interval_row(value: Any) -> dict[str, Any]:
    row = _row_mapping(value, "action interval row")
    start = _float(row, "action_start_ts")
    end = _float(row, "action_end_ts")
    if end <= start:
        raise ValueError("action interval end must be greater than start")
    result = {
        "schema": _schema(row, OBJECTSTATE_REAL_ACTION_INTERVAL_ROW_SCHEMA),
        "row_id": _required_str(row, "row_id"),
        "action_id": _required_str(row, "action_id"),
        "action_type": _required_str(row, "action_type"),
        "object_id": _required_str(row, "object_id"),
        "action_start_ts": start,
        "action_end_ts": end,
        "action_vector": _float_vector(row.get("action_vector"), 3, "action_vector"),
        "gt_provenance": _required_str(row, "gt_provenance"),
    }
    if not _is_nonzero_vector(result["action_vector"]):
        raise ValueError("action_interval_row requires non-zero action_vector")
    actor = row.get("actor")
    if actor is not None:
        result["actor"] = _non_empty_str(actor, "actor")
    target = row.get("target_object_id")
    if target is not None:
        result["target_object_id"] = _non_empty_str(target, "target_object_id")
    return result


def _validate_state_transition_row(value: Any) -> dict[str, Any]:
    row = _row_mapping(value, "state transition row")
    source_ts = _float(row, "source_timestamp")
    target_ts = _float(row, "target_timestamp")
    if target_ts <= source_ts:
        raise ValueError("state transition target_timestamp must be greater than source_timestamp")
    return {
        "schema": _schema(row, OBJECTSTATE_REAL_STATE_TRANSITION_ROW_SCHEMA),
        "row_id": _required_str(row, "row_id"),
        "transition_id": _required_str(row, "transition_id"),
        "object_id": _required_str(row, "object_id"),
        "source_frame_id": _required_str(row, "source_frame_id"),
        "target_frame_id": _required_str(row, "target_frame_id"),
        "source_timestamp": source_ts,
        "target_timestamp": target_ts,
        "source_pose_row_id": _required_str(row, "source_pose_row_id"),
        "target_pose_row_id": _required_str(row, "target_pose_row_id"),
        "gt_provenance": _required_str(row, "gt_provenance"),
    }


def _validate_gate_accounting_row(value: Any) -> dict[str, Any]:
    row = _row_mapping(value, "gate accounting row")
    evidence_kind = _required_str(row, "evidence_kind")
    if evidence_kind not in OBJECTSTATE_REAL_GATE_EVIDENCE_KINDS:
        raise ValueError("gate accounting evidence_kind is unsupported")
    status = _required_str(row, "accounting_status")
    if status not in OBJECTSTATE_REAL_GATE_ACCOUNTING_STATUSES:
        raise ValueError("gate accounting status is unsupported")
    result = {
        "schema": _schema(row, OBJECTSTATE_REAL_GATE_ACCOUNTING_ROW_SCHEMA),
        "row_id": _required_str(row, "row_id"),
        "evidence_kind": evidence_kind,
        "accounting_status": status,
        "metrics": _metrics(row.get("metrics", {})),
        "artifact_refs": list(
            _non_empty_string_tuple(row.get("artifact_refs"), "artifact_refs")
        ),
        "gt_requirements": _gt_requirements(row.get("gt_requirements")),
    }
    for optional in ("object_id", "action_id", "transition_id", "reason"):
        value = row.get(optional)
        if value is not None:
            result[optional] = _non_empty_str(value, optional)
    if status in {"evidence_incomplete", "unsupported"} and not result.get("reason"):
        raise ValueError("incomplete/unsupported accounting rows require reason")
    return result


def _action_transition_overlaps(
    actions: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
) -> set[tuple[str, str]]:
    overlaps: set[tuple[str, str]] = set()
    for action in actions:
        referenced_objects = {str(action["object_id"])}
        if action.get("target_object_id"):
            referenced_objects.add(str(action["target_object_id"]))
        for transition in transitions:
            if str(transition["object_id"]) not in referenced_objects:
                continue
            if _intervals_overlap(
                float(action["action_start_ts"]),
                float(action["action_end_ts"]),
                float(transition["source_timestamp"]),
                float(transition["target_timestamp"]),
            ):
                overlaps.add((str(action["action_id"]), str(transition["transition_id"])))
    return overlaps


def _intervention_accounting_row_has_valid_overlap(
    row: Mapping[str, Any],
    overlaps: set[tuple[str, str]],
) -> bool:
    action_id = row.get("action_id")
    transition_id = row.get("transition_id")
    return (
        isinstance(action_id, str)
        and isinstance(transition_id, str)
        and (action_id, transition_id) in overlaps
    )


def _summary_blockers(
    readiness: Mapping[str, bool],
    actionable_intervention_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers = []
    for key, message in (
        ("observation_rows_present", "missing observation rows"),
        ("object_pose_rows_present", "missing object pose rows"),
        ("identity_link_rows_present", "missing identity link rows"),
        ("state_transition_rows_present", "missing state transition rows"),
        ("gate_accounting_rows_present", "missing gate accounting rows"),
    ):
        if not readiness[key]:
            blockers.append(message)
    if actionable_intervention_rows:
        if not readiness["action_interval_rows_present"]:
            blockers.append("intervention accounting requires action interval rows")
        if not readiness["action_transition_overlap_ready"]:
            blockers.append("action interval rows do not overlap object transitions")
        if not readiness["intervention_accounting_refs_ready"]:
            blockers.append("intervention accounting rows do not reference overlapping action/transition pairs")
    return blockers


def _intervals_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> bool:
    return start_a <= end_b and end_a >= start_b


def _gt_requirements(value: Any) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise ValueError("gate accounting row requires gt_requirements")
    result = {}
    for key in ("identity", "pose", "action", "timestamp"):
        if not isinstance(value.get(key), bool):
            raise ValueError(f"gt_requirements.{key} must be bool")
        result[key] = bool(value[key])
    return result


def _metrics(value: Any) -> dict[str, float | bool]:
    if not isinstance(value, Mapping):
        raise ValueError("metrics must be a mapping")
    result: dict[str, float | bool] = {}
    for key, item in value.items():
        metric_key = str(key)
        if not metric_key:
            raise ValueError("metric keys must be non-empty")
        if isinstance(item, bool):
            result[metric_key] = bool(item)
            continue
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            result[metric_key] = _finite_float(item, f"metric {metric_key}")
            continue
        raise ValueError(f"metric {metric_key} must be numeric or bool")
    return result


def _counts(values: Sequence[str] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        text = str(value)
        counts[text] = counts.get(text, 0) + 1
    return counts


def _validate_unique_ids(name: str, values: Sequence[str] | Any) -> None:
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            raise ValueError(f"duplicate {name} id: {text}")
        seen.add(text)


def _schema(row: Mapping[str, Any], expected: str) -> str:
    if row.get("schema") != expected:
        raise ValueError(f"unsupported row schema: {row.get('schema')}")
    return expected


def _row_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


def _required_str(row: Mapping[str, Any], key: str) -> str:
    return _non_empty_str(row.get(key), key)


def _non_empty_str(value: Any, name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _float(row: Mapping[str, Any], key: str) -> float:
    return _finite_float(row.get(key), key)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _float_vector(value: Any, length: int, name: str) -> list[float]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a numeric vector")
    result = [_finite_float(item, name) for item in value]
    if len(result) != length:
        raise ValueError(f"{name} must have length {length}")
    return result


def _is_nonzero_vector(values: Sequence[float]) -> bool:
    return any(abs(float(value)) > 0.0 for value in values)


def _non_empty_string_tuple(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(_non_empty_str(value, name) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result
