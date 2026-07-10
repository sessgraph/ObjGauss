from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.datasets.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    validate_objectstate_controlled_capture_manifest,
)
from objgauss.datasets.objectstate_real_evidence_bundle import (
    OBJECTSTATE_REAL_ACTION_INTERVAL_ROW_SCHEMA,
    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
    OBJECTSTATE_REAL_GATE_ACCOUNTING_ROW_SCHEMA,
    OBJECTSTATE_REAL_IDENTITY_LINK_ROW_SCHEMA,
    OBJECTSTATE_REAL_OBJECT_POSE_ROW_SCHEMA,
    OBJECTSTATE_REAL_OBSERVATION_ROW_SCHEMA,
    OBJECTSTATE_REAL_STATE_TRANSITION_ROW_SCHEMA,
    objectstate_real_evidence_bundle_summary,
    validate_objectstate_real_evidence_bundle,
    validate_objectstate_real_evidence_bundle_summary,
)

OBJECTSTATE_CONTROLLED_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA = (
    "objgauss-objectstate-controlled-real-evidence-bundle-adapter-v1"
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA",
    "read_objectstate_controlled_real_evidence_bundle_adapter_summary",
    "objectstate_controlled_real_evidence_bundle_adapter_summary_from_file",
    "objectstate_controlled_real_evidence_bundle_adapter_summary",
    "objectstate_controlled_real_evidence_bundle_from_capture_manifest",
    "validate_objectstate_controlled_real_evidence_bundle_adapter_summary",
)


def read_objectstate_controlled_real_evidence_bundle_adapter_summary(
    path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("controlled real evidence bundle adapter JSON must be an object")
    return validate_objectstate_controlled_real_evidence_bundle_adapter_summary(payload)


def objectstate_controlled_real_evidence_bundle_adapter_summary_from_file(
    capture_manifest: str | Path,
    *,
    scene_id: str | None = None,
    sequence_id: str | None = None,
    source_dataset: str = "local-controlled-capture",
    gt_provenance: str = "controlled capture manifest annotations",
    source_summary_ref: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(Path(capture_manifest).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("controlled capture manifest JSON must be an object")
    return objectstate_controlled_real_evidence_bundle_adapter_summary(
        payload,
        scene_id=scene_id,
        sequence_id=sequence_id,
        source_dataset=source_dataset,
        gt_provenance=gt_provenance,
        source_summary_ref=(
            source_summary_ref if source_summary_ref is not None else str(capture_manifest)
        ),
    )


def objectstate_controlled_real_evidence_bundle_adapter_summary(
    capture_manifest: Mapping[str, Any],
    *,
    scene_id: str | None = None,
    sequence_id: str | None = None,
    source_dataset: str = "local-controlled-capture",
    gt_provenance: str = "controlled capture manifest annotations",
    source_summary_ref: str | None = None,
) -> dict[str, Any]:
    manifest = validate_objectstate_controlled_capture_manifest(capture_manifest)
    bundle = objectstate_controlled_real_evidence_bundle_from_capture_manifest(
        manifest,
        scene_id=scene_id,
        sequence_id=sequence_id,
        source_dataset=source_dataset,
        gt_provenance=gt_provenance,
        source_summary_ref=source_summary_ref,
    )
    bundle_summary = objectstate_real_evidence_bundle_summary(bundle)
    accounting_counts = _count_by(
        row["accounting_status"] for row in bundle["gate_accounting_rows"]
    )
    readiness = {
        "capture_manifest_valid": True,
        "controlled_source_kind": bundle["sample"]["source_kind"] == "controlled_real",
        "real_bundle_ready": (
            bundle_summary["status"] == "objectstate_real_evidence_bundle_ready"
        ),
        "default_accounting_is_evidence_incomplete": bool(
            bundle["gate_accounting_rows"]
        )
        and all(
            row["accounting_status"] == "evidence_incomplete"
            for row in bundle["gate_accounting_rows"]
        ),
        "intervention_pass_not_created_without_metrics": not any(
            row["evidence_kind"] == "intervention"
            and row["accounting_status"] == "pass"
            for row in bundle["gate_accounting_rows"]
        ),
    }
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA,
        "kind": "objectstate_controlled_real_evidence_bundle_adapter",
        "status": (
            "objectstate_controlled_real_evidence_bundle_adapter_ready"
            if readiness["real_bundle_ready"]
            else "objectstate_controlled_real_evidence_bundle_adapter_incomplete"
        ),
        "source_schemas": {
            "controlled_capture": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
            "real_evidence_bundle": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
        },
        "sample_id": bundle["sample"]["sample_id"],
        "source_kind": bundle["sample"]["source_kind"],
        "source_summary_ref": source_summary_ref,
        "row_counts": {
            "observation_rows": len(bundle["observation_rows"]),
            "object_pose_rows": len(bundle["object_pose_rows"]),
            "identity_link_rows": len(bundle["identity_link_rows"]),
            "action_interval_rows": len(bundle["action_interval_rows"]),
            "state_transition_rows": len(bundle["state_transition_rows"]),
            "gate_accounting_rows": len(bundle["gate_accounting_rows"]),
        },
        "accounting_status_counts": {
            status: accounting_counts.get(status, 0)
            for status in ("pass", "fail", "evidence_incomplete", "unsupported")
        },
        "readiness": readiness,
        "bundle": bundle,
        "bundle_summary": bundle_summary,
        "issues": _issues(bundle_summary),
        "claim_policy": {
            "reads_existing_controlled_capture_manifest": True,
            "emits_real_evidence_bundle": True,
            "default_rows_are_evidence_incomplete": True,
            "evidence_incomplete_is_not_model_fail": True,
            "does_not_create_ground_truth": True,
            "does_not_run_identity_eval": True,
            "does_not_run_prediction_eval": True,
            "does_not_run_intervention_eval": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_train_model": True,
            "does_not_claim_metric_pass": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "runs_identity_eval": False,
            "runs_prediction_eval": False,
            "runs_intervention_eval": False,
            "reconstructs_gaussians": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_controlled_real_evidence_bundle_adapter_summary(payload)


def objectstate_controlled_real_evidence_bundle_from_capture_manifest(
    capture_manifest: Mapping[str, Any],
    *,
    scene_id: str | None = None,
    sequence_id: str | None = None,
    source_dataset: str = "local-controlled-capture",
    gt_provenance: str = "controlled capture manifest annotations",
    source_summary_ref: str | None = None,
) -> dict[str, Any]:
    manifest = validate_objectstate_controlled_capture_manifest(capture_manifest)
    sample = manifest["sample"]
    sample_id = str(sample["sample_id"])
    object_pose_rows = _object_pose_rows(
        manifest,
        sample_id=sample_id,
        gt_provenance=gt_provenance,
    )
    state_transition_rows = _state_transition_rows(
        object_pose_rows,
        sample_id=sample_id,
        gt_provenance=gt_provenance,
    )
    action_interval_rows = _action_interval_rows(
        manifest,
        sample_id=sample_id,
        gt_provenance=gt_provenance,
    )
    bundle = {
        "schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
        "kind": "objectstate_real_evidence_bundle",
        "sample": {
            "sample_id": sample_id,
            "scene_id": scene_id or sample_id,
            "sequence_id": sequence_id or sample_id,
            "source_dataset": source_dataset,
            "source_kind": "controlled_real",
            "object_category": str(sample["object_category"]),
            "scenario": str(sample["scenario"]),
            "gt_provenance": gt_provenance,
            "license": str(sample["license"]),
            "observation_modalities": list(sample["observation_modalities"]),
            "artifact_refs": _artifact_refs(sample.get("artifact_refs", ()), source_summary_ref),
        },
        "row_schemas": {
            "observation": OBJECTSTATE_REAL_OBSERVATION_ROW_SCHEMA,
            "object_pose": OBJECTSTATE_REAL_OBJECT_POSE_ROW_SCHEMA,
            "identity_link": OBJECTSTATE_REAL_IDENTITY_LINK_ROW_SCHEMA,
            "action_interval": OBJECTSTATE_REAL_ACTION_INTERVAL_ROW_SCHEMA,
            "state_transition": OBJECTSTATE_REAL_STATE_TRANSITION_ROW_SCHEMA,
            "gate_accounting": OBJECTSTATE_REAL_GATE_ACCOUNTING_ROW_SCHEMA,
        },
        "observation_rows": _observation_rows(manifest, sample_id=sample_id),
        "object_pose_rows": object_pose_rows,
        "identity_link_rows": _identity_link_rows(
            object_pose_rows,
            sample_id=sample_id,
            gt_provenance=gt_provenance,
        ),
        "action_interval_rows": action_interval_rows,
        "state_transition_rows": state_transition_rows,
        "gate_accounting_rows": _default_gate_accounting_rows(
            sample_id=sample_id,
            source_summary_ref=source_summary_ref,
            transitions=state_transition_rows,
            actions=action_interval_rows,
        ),
    }
    return validate_objectstate_real_evidence_bundle(bundle)


def validate_objectstate_controlled_real_evidence_bundle_adapter_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled real evidence bundle adapter summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA:
        raise ValueError(
            "unsupported controlled real evidence bundle adapter schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_real_evidence_bundle_adapter":
        raise ValueError("controlled real evidence bundle adapter kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_real_evidence_bundle_adapter_ready",
        "objectstate_controlled_real_evidence_bundle_adapter_incomplete",
    }:
        raise ValueError("controlled real evidence bundle adapter status is unsupported")
    if payload.get("source_kind") != "controlled_real":
        raise ValueError("controlled real evidence bundle adapter source_kind mismatch")
    if not isinstance(payload.get("sample_id"), str) or not payload["sample_id"]:
        raise ValueError("controlled real evidence bundle adapter requires sample_id")
    schemas = payload.get("source_schemas")
    if not isinstance(schemas, Mapping):
        raise ValueError("controlled real evidence bundle adapter requires source_schemas")
    if schemas.get("controlled_capture") != OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA:
        raise ValueError("controlled real evidence bundle adapter capture schema mismatch")
    if schemas.get("real_evidence_bundle") != OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA:
        raise ValueError("controlled real evidence bundle adapter bundle schema mismatch")
    bundle = payload.get("bundle")
    if not isinstance(bundle, Mapping):
        raise ValueError("controlled real evidence bundle adapter requires bundle")
    validate_objectstate_real_evidence_bundle(bundle)
    bundle_summary = payload.get("bundle_summary")
    if not isinstance(bundle_summary, Mapping):
        raise ValueError("controlled real evidence bundle adapter requires bundle_summary")
    validate_objectstate_real_evidence_bundle_summary(bundle_summary)
    counts = payload.get("row_counts")
    if not isinstance(counts, Mapping):
        raise ValueError("controlled real evidence bundle adapter requires row_counts")
    for key in (
        "observation_rows",
        "object_pose_rows",
        "identity_link_rows",
        "action_interval_rows",
        "state_transition_rows",
        "gate_accounting_rows",
    ):
        value = counts.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"row_counts.{key} must be a non-negative int")
    accounting_counts = payload.get("accounting_status_counts")
    if not isinstance(accounting_counts, Mapping):
        raise ValueError("controlled real evidence bundle adapter requires accounting counts")
    for status in ("pass", "fail", "evidence_incomplete", "unsupported"):
        value = accounting_counts.get(status)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"accounting_status_counts.{status} must be int")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("controlled real evidence bundle adapter requires readiness")
    for key in (
        "capture_manifest_valid",
        "controlled_source_kind",
        "real_bundle_ready",
        "default_accounting_is_evidence_incomplete",
        "intervention_pass_not_created_without_metrics",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"controlled real evidence bundle readiness missing bool {key}")
    expected_status = (
        "objectstate_controlled_real_evidence_bundle_adapter_ready"
        if readiness["real_bundle_ready"]
        else "objectstate_controlled_real_evidence_bundle_adapter_incomplete"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled real evidence bundle adapter status mismatch")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("controlled real evidence bundle adapter requires issues")
    claim_policy = payload.get("claim_policy", {})
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("reads_existing_controlled_capture_manifest")
        or not claim_policy.get("emits_real_evidence_bundle")
        or not claim_policy.get("default_rows_are_evidence_incomplete")
        or not claim_policy.get("evidence_incomplete_is_not_model_fail")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_run_identity_eval")
        or not claim_policy.get("does_not_run_prediction_eval")
        or not claim_policy.get("does_not_run_intervention_eval")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_claim_metric_pass")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled real evidence bundle adapter must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "controlled real evidence bundle adapter cannot claim non-goal behavior"
        )
    return dict(payload)


def _observation_rows(
    manifest: Mapping[str, Any],
    *,
    sample_id: str,
) -> list[dict[str, Any]]:
    camera_id = _camera_id(manifest)
    rows = []
    for frame in manifest["frames"]:
        rows.append(
            {
                "schema": OBJECTSTATE_REAL_OBSERVATION_ROW_SCHEMA,
                "row_id": f"controlled-observation:{sample_id}:{frame['frame_id']}",
                "frame_id": str(frame["frame_id"]),
                "timestamp": float(frame["timestamp"]),
                "camera_id": camera_id,
                "observation": dict(frame["observation"]),
            }
        )
    return rows


def _object_pose_rows(
    manifest: Mapping[str, Any],
    *,
    sample_id: str,
    gt_provenance: str,
) -> list[dict[str, Any]]:
    camera_id = _camera_id(manifest)
    rows = []
    for frame in manifest["frames"]:
        for item in frame["objects"]:
            pose = item.get("pose")
            if not isinstance(pose, Mapping):
                continue
            frame_id = str(frame["frame_id"])
            object_id = str(item["object_id"])
            rows.append(
                {
                    "schema": OBJECTSTATE_REAL_OBJECT_POSE_ROW_SCHEMA,
                    "row_id": f"controlled-pose:{sample_id}:{frame_id}:{object_id}",
                    "frame_id": frame_id,
                    "timestamp": float(frame["timestamp"]),
                    "camera_id": camera_id,
                    "object_id": object_id,
                    "object_pose_6dof": {
                        "position": list(pose["position"]),
                        "rotation_xyzw": list(pose["rotation_xyzw"]),
                    },
                    "object_visibility": _visibility(item),
                    "gt_provenance": gt_provenance,
                }
            )
    return rows


def _identity_link_rows(
    object_pose_rows: Sequence[Mapping[str, Any]],
    *,
    sample_id: str,
    gt_provenance: str,
) -> list[dict[str, Any]]:
    rows = []
    for pose in object_pose_rows:
        frame_id = str(pose["frame_id"])
        object_id = str(pose["object_id"])
        rows.append(
            {
                "schema": OBJECTSTATE_REAL_IDENTITY_LINK_ROW_SCHEMA,
                "row_id": f"controlled-identity:{sample_id}:{frame_id}:{object_id}",
                "frame_id": frame_id,
                "timestamp": float(pose["timestamp"]),
                "object_id": object_id,
                "physical_identity_id": object_id,
                "gt_provenance": gt_provenance,
                "confidence": 1.0,
            }
        )
    return rows


def _action_interval_rows(
    manifest: Mapping[str, Any],
    *,
    sample_id: str,
    gt_provenance: str,
) -> list[dict[str, Any]]:
    rows = []
    for action in manifest.get("actions", ()):
        vector = action.get("vector")
        if not _nonzero_vector(vector):
            continue
        action_id = str(action["action_id"])
        row = {
            "schema": OBJECTSTATE_REAL_ACTION_INTERVAL_ROW_SCHEMA,
            "row_id": f"controlled-action:{sample_id}:{action_id}",
            "action_id": action_id,
            "action_type": str(action["action_type"]),
            "object_id": str(action["object_id"]),
            "action_start_ts": float(action["start_timestamp"]),
            "action_end_ts": float(action["end_timestamp"]),
            "action_vector": [float(item) for item in vector],
            "actor": str(action.get("actor", "unknown")),
            "gt_provenance": gt_provenance,
        }
        if action.get("target_object_id"):
            row["target_object_id"] = str(action["target_object_id"])
        rows.append(row)
    return rows


def _state_transition_rows(
    object_pose_rows: Sequence[Mapping[str, Any]],
    *,
    sample_id: str,
    gt_provenance: str,
) -> list[dict[str, Any]]:
    by_object: dict[str, list[Mapping[str, Any]]] = {}
    for row in object_pose_rows:
        by_object.setdefault(str(row["object_id"]), []).append(row)
    transitions = []
    for object_id, rows in by_object.items():
        ordered = sorted(rows, key=lambda row: float(row["timestamp"]))
        for source, target in zip(ordered, ordered[1:]):
            source_frame_id = str(source["frame_id"])
            target_frame_id = str(target["frame_id"])
            transition_id = (
                "controlled-transition:"
                f"{sample_id}:{object_id}:{source_frame_id}:{target_frame_id}"
            )
            transitions.append(
                {
                    "schema": OBJECTSTATE_REAL_STATE_TRANSITION_ROW_SCHEMA,
                    "row_id": f"{transition_id}:row",
                    "transition_id": transition_id,
                    "object_id": object_id,
                    "source_frame_id": source_frame_id,
                    "target_frame_id": target_frame_id,
                    "source_timestamp": float(source["timestamp"]),
                    "target_timestamp": float(target["timestamp"]),
                    "source_pose_row_id": str(source["row_id"]),
                    "target_pose_row_id": str(target["row_id"]),
                    "gt_provenance": gt_provenance,
                }
            )
    return transitions


def _default_gate_accounting_rows(
    *,
    sample_id: str,
    source_summary_ref: str | None,
    transitions: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    first_transition = transitions[0] if transitions else None
    first_overlap = _first_action_transition_overlap(actions, transitions)
    rows = [
        _accounting_row(
            sample_id=sample_id,
            evidence_kind="identity",
            reason="missing controlled real identity candidate metrics",
            source_summary_ref=source_summary_ref,
            object_id=(
                None if first_transition is None else str(first_transition["object_id"])
            ),
        ),
        _accounting_row(
            sample_id=sample_id,
            evidence_kind="prediction",
            reason="missing controlled real state-vs-history prediction metrics",
            source_summary_ref=source_summary_ref,
            transition=first_transition,
        ),
    ]
    intervention_transition = None
    intervention_action = None
    if first_overlap is not None:
        intervention_action, intervention_transition = first_overlap
    rows.append(
        _accounting_row(
            sample_id=sample_id,
            evidence_kind="intervention",
            reason=(
                "missing controlled real action-conditioned intervention metrics"
                if first_overlap is not None
                else "missing overlapping non-zero action interval and state transition"
            ),
            source_summary_ref=source_summary_ref,
            transition=intervention_transition,
            action=intervention_action,
        )
    )
    return rows


def _accounting_row(
    *,
    sample_id: str,
    evidence_kind: str,
    reason: str,
    source_summary_ref: str | None,
    object_id: str | None = None,
    transition: Mapping[str, Any] | None = None,
    action: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "schema": OBJECTSTATE_REAL_GATE_ACCOUNTING_ROW_SCHEMA,
        "row_id": f"controlled-real-accounting:{sample_id}:{evidence_kind}",
        "evidence_kind": evidence_kind,
        "accounting_status": "evidence_incomplete",
        "metrics": {},
        "artifact_refs": _artifact_refs((), source_summary_ref),
        "gt_requirements": {
            "identity": True,
            "pose": evidence_kind in {"prediction", "intervention"},
            "action": evidence_kind == "intervention",
            "timestamp": True,
        },
        "reason": reason,
    }
    if object_id is not None:
        row["object_id"] = object_id
    if transition is not None:
        row["transition_id"] = str(transition["transition_id"])
        row["object_id"] = str(transition["object_id"])
    if action is not None:
        row["action_id"] = str(action["action_id"])
    return row


def _first_action_transition_overlap(
    actions: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
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
                return action, transition
    return None


def _camera_id(manifest: Mapping[str, Any]) -> str:
    device = manifest["sample"].get("capture_device")
    if isinstance(device, str) and device.strip():
        return device.strip()
    return "controlled-camera"


def _visibility(item: Mapping[str, Any]) -> float:
    if not bool(item.get("visible", True)):
        return 0.0
    occlusion = float(item.get("occlusion_fraction", 0.0))
    visibility = 1.0 - occlusion
    if visibility < 0.0:
        return 0.0
    if visibility > 1.0:
        return 1.0
    return visibility


def _artifact_refs(
    refs: Sequence[Any],
    source_summary_ref: str | None,
) -> list[str]:
    result = [str(item) for item in refs if str(item)]
    if source_summary_ref:
        result.append(str(source_summary_ref))
    if not result:
        result.append("controlled-capture-manifest")
    return result


def _nonzero_vector(value: Any) -> bool:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return False
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return len(vector) == 3 and any(abs(item) > 1.0e-12 for item in vector)


def _intervals_overlap(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> bool:
    return max(left_start, right_start) < min(left_end, right_end)


def _count_by(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _issues(bundle_summary: Mapping[str, Any]) -> list[str]:
    issues = list(bundle_summary.get("hard_blockers", ()))
    if bundle_summary["readiness"]["intervention_accounting_ready"]:
        return issues
    issues.append("controlled real evaluator metrics are required before pass/fail rows")
    return issues
