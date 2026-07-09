from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_bop_capture_adapter import (
    OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
    validate_objectstate_bop_capture_acceptance_summary,
)
from objgauss.core.objectstate_bop_reality_rows import (
    OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA,
    validate_objectstate_bop_reality_rows_summary,
)
from objgauss.core.objectstate_real_evidence_bundle import (
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

OBJECTSTATE_BOP_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA = (
    "objgauss-objectstate-bop-real-evidence-bundle-adapter-v1"
)


def read_objectstate_bop_real_evidence_bundle_adapter_summary(
    path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("BOP real evidence bundle adapter JSON must be an object")
    return validate_objectstate_bop_real_evidence_bundle_adapter_summary(payload)


def objectstate_bop_real_evidence_bundle_adapter_summary_from_files(
    acceptance_summary: str | Path,
    reality_rows_summary: str | Path,
    *,
    source_kind: str | None = None,
    source_summary_ref: str | None = None,
) -> dict[str, Any]:
    acceptance = _read_json_mapping(acceptance_summary, "BOP acceptance summary")
    reality_rows = _read_json_mapping(reality_rows_summary, "BOP reality rows summary")
    return objectstate_bop_real_evidence_bundle_adapter_summary(
        acceptance,
        reality_rows,
        source_kind=source_kind,
        source_summary_ref=(
            source_summary_ref
            if source_summary_ref is not None
            else str(reality_rows_summary)
        ),
    )


def objectstate_bop_real_evidence_bundle_adapter_summary(
    acceptance_summary: Mapping[str, Any],
    reality_rows_summary: Mapping[str, Any],
    *,
    source_kind: str | None = None,
    source_summary_ref: str | None = None,
) -> dict[str, Any]:
    acceptance = validate_objectstate_bop_capture_acceptance_summary(
        acceptance_summary
    )
    reality = validate_objectstate_bop_reality_rows_summary(reality_rows_summary)
    bundle = objectstate_bop_real_evidence_bundle_from_summaries(
        acceptance,
        reality,
        source_kind=source_kind,
        source_summary_ref=source_summary_ref,
    )
    bundle_summary = objectstate_real_evidence_bundle_summary(bundle)
    accounting_counts = _count_by(
        row["accounting_status"] for row in bundle["gate_accounting_rows"]
    )
    readiness = {
        "acceptance_pass": (
            acceptance["status"] == "objectstate_bop_capture_acceptance_pass"
        ),
        "reality_rows_reviewable": (
            reality["status"] == "objectstate_bop_reality_rows_reviewable"
        ),
        "real_bundle_ready": (
            bundle_summary["status"] == "objectstate_real_evidence_bundle_ready"
        ),
        "public_or_controlled_source_kind": bundle["sample"]["source_kind"]
        in {"public_replay", "controlled_real"},
        "intervention_blocked_mapped_to_evidence_incomplete": any(
            row["evidence_kind"] == "intervention"
            and row["accounting_status"] == "evidence_incomplete"
            for row in bundle["gate_accounting_rows"]
        ),
        "intervention_pass_not_created_without_action_gt": not any(
            row["evidence_kind"] == "intervention"
            and row["accounting_status"] == "pass"
            for row in bundle["gate_accounting_rows"]
        ),
    }
    payload = {
        "schema": OBJECTSTATE_BOP_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA,
        "kind": "objectstate_bop_real_evidence_bundle_adapter",
        "status": (
            "objectstate_bop_real_evidence_bundle_adapter_ready"
            if readiness["acceptance_pass"]
            and readiness["reality_rows_reviewable"]
            and readiness["real_bundle_ready"]
            else "objectstate_bop_real_evidence_bundle_adapter_incomplete"
        ),
        "source_schemas": {
            "bop_acceptance": OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
            "bop_reality_rows": OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA,
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
        "issues": _issues(acceptance, reality, bundle_summary),
        "claim_policy": {
            "reads_existing_bop_acceptance": True,
            "reads_existing_bop_reality_rows": True,
            "emits_real_evidence_bundle": True,
            "preserves_public_or_controlled_source_kind": True,
            "blocked_intervention_rows_become_evidence_incomplete": True,
            "does_not_create_ground_truth": True,
            "does_not_run_bop_handoff": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_train_model": True,
            "does_not_claim_intervention_pass": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "captures_video": False,
            "creates_ground_truth": False,
            "runs_bop_handoff": False,
            "reconstructs_gaussians": False,
            "runs_identity_eval": False,
            "runs_prediction_eval": False,
            "runs_intervention_eval": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_bop_real_evidence_bundle_adapter_summary(payload)


def objectstate_bop_real_evidence_bundle_from_summaries(
    acceptance_summary: Mapping[str, Any],
    reality_rows_summary: Mapping[str, Any],
    *,
    source_kind: str | None = None,
    source_summary_ref: str | None = None,
) -> dict[str, Any]:
    acceptance = validate_objectstate_bop_capture_acceptance_summary(
        acceptance_summary
    )
    reality = validate_objectstate_bop_reality_rows_summary(reality_rows_summary)
    manifest = acceptance["adapter"]["manifest"]
    sample = manifest["sample"]
    bundle_source_kind = source_kind or reality["source_kind"]
    if bundle_source_kind not in {"public_replay", "controlled_real"}:
        raise ValueError("BOP real evidence bundle source_kind is unsupported")
    gt_provenance = (
        "BOP scene_camera.json / scene_gt.json / scene_gt_info.json via "
        "objectstate_bop_capture_adapter"
    )
    observation_rows = _observation_rows(manifest)
    object_pose_rows = _object_pose_rows(manifest)
    identity_link_rows = _identity_link_rows(manifest)
    state_transition_rows = _state_transition_rows(object_pose_rows)
    action_interval_rows = _action_interval_rows(manifest)
    gate_accounting_rows = _gate_accounting_rows(
        reality,
        transitions=state_transition_rows,
        actions=action_interval_rows,
        source_summary_ref=source_summary_ref,
    )
    bundle = {
        "schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
        "kind": "objectstate_real_evidence_bundle",
        "sample": {
            "sample_id": str(reality["sample_id"]),
            "scene_id": _scene_id(acceptance),
            "sequence_id": str(reality["sample_id"]),
            "source_dataset": str(
                acceptance["adapter"]["source"].get("dataset_id")
                or "bop-public-dataset"
            ),
            "source_kind": bundle_source_kind,
            "object_category": str(sample["object_category"]),
            "scenario": str(sample["scenario"]),
            "gt_provenance": gt_provenance,
            "license": str(sample["license"]),
            "observation_modalities": list(sample["observation_modalities"]),
            "artifact_refs": _artifact_refs(
                sample.get("artifact_refs", ()),
                acceptance.get("scene_root"),
                source_summary_ref,
            ),
        },
        "row_schemas": {
            "observation": OBJECTSTATE_REAL_OBSERVATION_ROW_SCHEMA,
            "object_pose": OBJECTSTATE_REAL_OBJECT_POSE_ROW_SCHEMA,
            "identity_link": OBJECTSTATE_REAL_IDENTITY_LINK_ROW_SCHEMA,
            "action_interval": OBJECTSTATE_REAL_ACTION_INTERVAL_ROW_SCHEMA,
            "state_transition": OBJECTSTATE_REAL_STATE_TRANSITION_ROW_SCHEMA,
            "gate_accounting": OBJECTSTATE_REAL_GATE_ACCOUNTING_ROW_SCHEMA,
        },
        "observation_rows": observation_rows,
        "object_pose_rows": object_pose_rows,
        "identity_link_rows": identity_link_rows,
        "action_interval_rows": action_interval_rows,
        "state_transition_rows": state_transition_rows,
        "gate_accounting_rows": gate_accounting_rows,
    }
    return validate_objectstate_real_evidence_bundle(bundle)


def validate_objectstate_bop_real_evidence_bundle_adapter_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP real evidence bundle adapter summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA:
        raise ValueError(
            "unsupported BOP real evidence bundle adapter schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_real_evidence_bundle_adapter":
        raise ValueError("BOP real evidence bundle adapter kind is unsupported")
    if payload.get("status") not in {
        "objectstate_bop_real_evidence_bundle_adapter_ready",
        "objectstate_bop_real_evidence_bundle_adapter_incomplete",
    }:
        raise ValueError("BOP real evidence bundle adapter status is unsupported")
    if payload.get("source_kind") not in {"public_replay", "controlled_real"}:
        raise ValueError("BOP real evidence bundle source_kind is unsupported")
    if not isinstance(payload.get("sample_id"), str) or not payload["sample_id"]:
        raise ValueError("BOP real evidence bundle adapter requires sample_id")
    schemas = payload.get("source_schemas")
    if not isinstance(schemas, Mapping):
        raise ValueError("BOP real evidence bundle adapter requires source_schemas")
    if schemas.get("bop_acceptance") != OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA:
        raise ValueError("BOP real evidence bundle adapter acceptance schema mismatch")
    if schemas.get("bop_reality_rows") != OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA:
        raise ValueError("BOP real evidence bundle adapter reality schema mismatch")
    if schemas.get("real_evidence_bundle") != OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA:
        raise ValueError("BOP real evidence bundle adapter bundle schema mismatch")
    bundle = payload.get("bundle")
    if not isinstance(bundle, Mapping):
        raise ValueError("BOP real evidence bundle adapter requires bundle")
    validate_objectstate_real_evidence_bundle(bundle)
    bundle_summary = payload.get("bundle_summary")
    if not isinstance(bundle_summary, Mapping):
        raise ValueError("BOP real evidence bundle adapter requires bundle_summary")
    validate_objectstate_real_evidence_bundle_summary(bundle_summary)
    counts = payload.get("row_counts")
    if not isinstance(counts, Mapping):
        raise ValueError("BOP real evidence bundle adapter requires row_counts")
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
        raise ValueError("BOP real evidence bundle adapter requires accounting counts")
    for status in ("pass", "fail", "evidence_incomplete", "unsupported"):
        value = accounting_counts.get(status)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"accounting_status_counts.{status} must be int")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("BOP real evidence bundle adapter requires readiness")
    for key in (
        "acceptance_pass",
        "reality_rows_reviewable",
        "real_bundle_ready",
        "public_or_controlled_source_kind",
        "intervention_blocked_mapped_to_evidence_incomplete",
        "intervention_pass_not_created_without_action_gt",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"BOP real evidence bundle readiness missing bool {key}")
    expected_status = (
        "objectstate_bop_real_evidence_bundle_adapter_ready"
        if readiness["acceptance_pass"]
        and readiness["reality_rows_reviewable"]
        and readiness["real_bundle_ready"]
        else "objectstate_bop_real_evidence_bundle_adapter_incomplete"
    )
    if payload["status"] != expected_status:
        raise ValueError("BOP real evidence bundle adapter status mismatch")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("BOP real evidence bundle adapter requires issues")
    claim_policy = payload.get("claim_policy", {})
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("reads_existing_bop_acceptance")
        or not claim_policy.get("reads_existing_bop_reality_rows")
        or not claim_policy.get("emits_real_evidence_bundle")
        or not claim_policy.get("preserves_public_or_controlled_source_kind")
        or not claim_policy.get("blocked_intervention_rows_become_evidence_incomplete")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_run_bop_handoff")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_claim_intervention_pass")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP real evidence bundle adapter must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError("BOP real evidence bundle adapter cannot claim non-goal behavior")
    return dict(payload)


def _observation_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for frame in manifest["frames"]:
        observation = {}
        frame_observation = frame.get("observation", {})
        if isinstance(frame_observation, Mapping):
            if frame_observation.get("rgb"):
                observation["rgb"] = str(frame_observation["rgb"])
            if frame_observation.get("gaussian"):
                observation["gaussian"] = str(frame_observation["gaussian"])
        if frame.get("rgb"):
            observation["rgb"] = str(frame["rgb"])
        if frame.get("gaussian"):
            observation["gaussian"] = str(frame["gaussian"])
        rows.append(
            {
                "schema": OBJECTSTATE_REAL_OBSERVATION_ROW_SCHEMA,
                "row_id": f"bop-observation:{frame['frame_id']}",
                "frame_id": str(frame["frame_id"]),
                "timestamp": float(frame["timestamp"]),
                "camera_id": _camera_id(manifest),
                "observation": observation,
            }
        )
    return rows


def _object_pose_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    camera_id = _camera_id(manifest)
    for frame in manifest["frames"]:
        for item in frame["objects"]:
            pose = item["pose"]
            object_id = str(item["object_id"])
            frame_id = str(frame["frame_id"])
            rows.append(
                {
                    "schema": OBJECTSTATE_REAL_OBJECT_POSE_ROW_SCHEMA,
                    "row_id": f"bop-pose:{frame_id}:{object_id}",
                    "frame_id": frame_id,
                    "timestamp": float(frame["timestamp"]),
                    "camera_id": camera_id,
                    "object_id": object_id,
                    "object_pose_6dof": {
                        "position": list(pose["position"]),
                        "rotation_xyzw": list(pose["rotation_xyzw"]),
                    },
                    "object_visibility": _visibility(item),
                }
            )
    return rows


def _identity_link_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for frame in manifest["frames"]:
        for item in frame["objects"]:
            frame_id = str(frame["frame_id"])
            object_id = str(item["object_id"])
            rows.append(
                {
                    "schema": OBJECTSTATE_REAL_IDENTITY_LINK_ROW_SCHEMA,
                    "row_id": f"bop-identity:{frame_id}:{object_id}",
                    "frame_id": frame_id,
                    "timestamp": float(frame["timestamp"]),
                    "object_id": object_id,
                    "physical_identity_id": object_id,
                    "gt_provenance": "BOP object identity imported from scene_gt",
                    "confidence": 1.0,
                }
            )
    return rows


def _action_interval_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for action in manifest.get("actions", ()):
        vector = action.get("action_vector")
        if not _nonzero_vector(vector):
            continue
        action_id = str(action["action_id"])
        rows.append(
            {
                "schema": OBJECTSTATE_REAL_ACTION_INTERVAL_ROW_SCHEMA,
                "row_id": f"bop-action:{action_id}",
                "action_id": action_id,
                "action_type": str(action["action_type"]),
                "object_id": str(action["object_id"]),
                "action_start_ts": float(action["action_start_ts"]),
                "action_end_ts": float(action["action_end_ts"]),
                "action_vector": [float(item) for item in vector],
                "actor": str(action.get("actor", "unknown")),
                "gt_provenance": "BOP adapter action metadata",
            }
        )
    return rows


def _state_transition_rows(
    object_pose_rows: Sequence[Mapping[str, Any]],
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
                f"bop-transition:{object_id}:{source_frame_id}:{target_frame_id}"
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
                    "gt_provenance": "BOP sequential object pose GT",
                }
            )
    return transitions


def _gate_accounting_rows(
    reality: Mapping[str, Any],
    *,
    transitions: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    source_summary_ref: str | None,
) -> list[dict[str, Any]]:
    first_transition = transitions[0] if transitions else None
    first_action = actions[0] if actions else None
    rows = []
    for row in reality["rows"]:
        evidence_kind = str(row["evidence_kind"])
        accounting_status = _accounting_status(row)
        reason = _accounting_reason(row, accounting_status)
        if evidence_kind == "prediction" and accounting_status in {"pass", "fail"}:
            if first_transition is None:
                accounting_status = "evidence_incomplete"
                reason = "BOP prediction row has no state transition rows"
        if evidence_kind == "intervention" and accounting_status in {"pass", "fail"}:
            if first_action is None or first_transition is None:
                accounting_status = "evidence_incomplete"
                reason = (
                    "BOP intervention pass/fail cannot be preserved without "
                    "non-zero action interval and state transition refs"
                )
        accounting = {
            "schema": OBJECTSTATE_REAL_GATE_ACCOUNTING_ROW_SCHEMA,
            "row_id": f"bop-real-accounting:{evidence_kind}",
            "evidence_kind": evidence_kind,
            "accounting_status": accounting_status,
            "metrics": _metrics(row, evidence_kind),
            "artifact_refs": _artifact_refs(
                row.get("artifact_refs", ()),
                source_summary_ref,
            ),
            "gt_requirements": _gt_requirements(row),
        }
        if evidence_kind == "prediction" and first_transition is not None:
            accounting["transition_id"] = str(first_transition["transition_id"])
            accounting["object_id"] = str(first_transition["object_id"])
        if evidence_kind == "intervention" and first_transition is not None:
            accounting["object_id"] = str(first_transition["object_id"])
            if accounting_status in {"pass", "fail"} and first_action is not None:
                accounting["action_id"] = str(first_action["action_id"])
                accounting["transition_id"] = str(first_transition["transition_id"])
        if evidence_kind == "identity" and row.get("object_id"):
            accounting["object_id"] = str(row["object_id"])
        if reason:
            accounting["reason"] = reason
        rows.append(accounting)
    return rows


def _accounting_status(row: Mapping[str, Any]) -> str:
    status = str(row["status"])
    if status == "blocked":
        return "evidence_incomplete"
    if status in {"pass", "fail"}:
        return status
    return "unsupported"


def _accounting_reason(row: Mapping[str, Any], status: str) -> str | None:
    if status == "fail":
        return str(row.get("failure_reason") or "BOP reality row failed")
    if status in {"evidence_incomplete", "unsupported"}:
        return str(
            row.get("block_reason")
            or row.get("failure_reason")
            or "BOP reality row lacks required gate evidence"
        )
    return None


def _metrics(row: Mapping[str, Any], evidence_kind: str) -> dict[str, float | bool]:
    metrics = {
        str(key): value
        for key, value in dict(row.get("metrics", {})).items()
        if isinstance(value, bool)
        or (isinstance(value, (int, float)) and not isinstance(value, bool))
    }
    if (
        evidence_kind == "prediction"
        and "state_vs_history_error_ratio" not in metrics
        and "error_ratio_vs_history_model" in metrics
    ):
        metrics["state_vs_history_error_ratio"] = metrics["error_ratio_vs_history_model"]
    return metrics


def _gt_requirements(row: Mapping[str, Any]) -> dict[str, bool]:
    gt = row.get("ground_truth", {})
    return {
        "identity": bool(gt.get("identity")),
        "pose": bool(gt.get("pose")),
        "action": bool(gt.get("action")),
        "timestamp": bool(gt.get("timestamp")),
    }


def _visibility(item: Mapping[str, Any]) -> float:
    if item.get("visible") is False:
        return 0.0
    if item.get("occlusion_fraction") is not None:
        return max(0.0, min(1.0, 1.0 - float(item["occlusion_fraction"])))
    return 1.0


def _camera_id(manifest: Mapping[str, Any]) -> str:
    return str(manifest["sample"].get("capture_device") or "bop-camera")


def _scene_id(acceptance: Mapping[str, Any]) -> str:
    dataset_id = str(acceptance["adapter"]["source"].get("dataset_id") or "bop")
    scene_root = str(acceptance.get("scene_root") or "")
    scene_name = Path(scene_root).name if scene_root else str(acceptance["sample_id"])
    return f"{dataset_id}:{scene_name}"


def _artifact_refs(*values: Any) -> list[str]:
    refs = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            items = (value,)
        elif isinstance(value, Sequence):
            items = value
        else:
            items = (str(value),)
        for item in items:
            text = str(item).strip()
            if text and text not in refs:
                refs.append(text)
    if not refs:
        refs.append("bop-real-evidence-bundle-adapter")
    return refs


def _issues(
    acceptance: Mapping[str, Any],
    reality: Mapping[str, Any],
    bundle_summary: Mapping[str, Any],
) -> list[str]:
    issues = []
    if acceptance["status"] != "objectstate_bop_capture_acceptance_pass":
        issues.append("BOP acceptance summary is not pass")
    if reality["blocked_row_count"]:
        issues.append("BOP blocked rows were mapped to evidence_incomplete")
    if not bundle_summary["readiness"]["intervention_accounting_ready"]:
        issues.append("real bundle has no action-overlap intervention pass/fail evidence")
    if bundle_summary["status"] != "objectstate_real_evidence_bundle_ready":
        issues.append("real evidence bundle summary is incomplete")
    return issues


def _count_by(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        text = str(value)
        counts[text] = counts.get(text, 0) + 1
    return counts


def _nonzero_vector(value: Any) -> bool:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return False
    try:
        return any(abs(float(item)) > 0.0 for item in value)
    except (TypeError, ValueError):
        return False


def _read_json_mapping(path: str | Path, label: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} JSON must be an object")
    return dict(payload)
