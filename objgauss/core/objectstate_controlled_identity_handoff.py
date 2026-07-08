from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_capture_files import (
    OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA,
    objectstate_controlled_capture_file_audit,
    validate_objectstate_controlled_capture_file_audit_summary,
)
from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    validate_objectstate_controlled_capture_manifest,
)
from objgauss.core.objectstate_controlled_identity_eval import (
    OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA,
    OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA,
    ObjectStateControlledIdentityThresholds,
    evaluate_objectstate_controlled_identity_predictions,
    validate_objectstate_controlled_identity_eval_summary,
    validate_objectstate_controlled_identity_predictions,
)
from objgauss.core.objectstate_controlled_real_rows import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
    OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
    objectstate_controlled_real_rows_summary,
    validate_objectstate_controlled_real_manifest,
    validate_objectstate_controlled_real_rows_summary,
)
from objgauss.core.objectstate_identity_prediction_adapter import (
    objectstate_identity_predictions_from_trainable_artifact,
)
from objgauss.core.objectstate_reality_gate import ObjectStateRealityGateThresholds

OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA = (
    "objgauss-objectstate-controlled-identity-handoff-v1"
)

OBJECTSTATE_CONTROLLED_CANDIDATE_ARTIFACT_FILE_AUDIT_SCHEMA = (
    "objgauss-objectstate-controlled-candidate-artifact-file-audit-v1"
)

OBJECTSTATE_CONTROLLED_IDENTITY_SCENARIO_AUDIT_SCHEMA = (
    "objgauss-objectstate-controlled-identity-scenario-audit-v1"
)


def objectstate_controlled_identity_handoff(
    capture_manifest: Mapping[str, Any],
    model_artifact: Mapping[str, Any],
    *,
    candidate_id: str | None = None,
    source: str = "trainable_kernel_objectstate_nearest_pose_adapter",
    artifact_refs: Sequence[str] | None = None,
    max_centroid_distance: float | None = None,
    identity_thresholds: ObjectStateControlledIdentityThresholds | None = None,
    synthetic_smoke_passed: bool = True,
    min_real_or_public_rows: int = 1,
    capture_root: str | Path = ".",
    check_artifact_refs: bool = False,
    min_rgb_bytes: int = 1,
    min_gaussian_bytes: int = 1,
    hash_files: bool = False,
    candidate_artifact_path: str | Path | None = None,
    min_candidate_artifact_bytes: int = 1,
    hash_candidate_artifact: bool = False,
    min_identity_scenario_frames: int = 3,
    min_occlusion_fraction: float = 0.5,
    min_view_conditions: int = 2,
    min_lighting_conditions: int = 2,
    min_camera_motion_m: float = 0.01,
) -> dict[str, Any]:
    capture_file_audit = objectstate_controlled_capture_file_audit(
        capture_manifest,
        root=capture_root,
        require_gaussian_files=True,
        check_artifact_refs=check_artifact_refs,
        min_rgb_bytes=min_rgb_bytes,
        min_gaussian_bytes=min_gaussian_bytes,
        hash_files=hash_files,
    )
    candidate_artifact_file_audit = _candidate_artifact_file_audit(
        candidate_artifact_path,
        min_bytes=min_candidate_artifact_bytes,
        hash_file=hash_candidate_artifact,
    )
    identity_scenario_audit = _identity_scenario_audit(
        capture_manifest,
        min_frames=min_identity_scenario_frames,
        min_occlusion_fraction=min_occlusion_fraction,
        min_view_conditions=min_view_conditions,
        min_lighting_conditions=min_lighting_conditions,
        min_camera_motion_m=min_camera_motion_m,
    )
    predictions = objectstate_identity_predictions_from_trainable_artifact(
        capture_manifest,
        model_artifact,
        candidate_id=candidate_id,
        source=source,
        artifact_refs=artifact_refs,
        max_centroid_distance=max_centroid_distance,
    )
    candidate_artifact_ref_match = _candidate_artifact_ref_match(
        candidate_artifact_file_audit,
        predictions,
    )
    identity_eval = evaluate_objectstate_controlled_identity_predictions(
        capture_manifest,
        predictions,
        thresholds=identity_thresholds,
    )
    controlled_real_manifest = identity_eval["controlled_real_manifest"]
    controlled_real_summary = objectstate_controlled_real_rows_summary(
        controlled_real_manifest,
        synthetic_smoke_passed=bool(synthetic_smoke_passed),
        thresholds=ObjectStateRealityGateThresholds(
            min_real_or_public_rows=int(min_real_or_public_rows),
            require_identity_pass_row=True,
            require_prediction_pass_row=False,
            require_intervention_pass_row=False,
            fail_on_failed_rows=True,
        ),
    )
    passed = (
        capture_file_audit["status"]
        == "objectstate_controlled_capture_file_audit_pass"
        and candidate_artifact_file_audit["status"]
        == "objectstate_controlled_candidate_artifact_file_audit_pass"
        and candidate_artifact_ref_match["matches"]
        and identity_scenario_audit["status"]
        == "objectstate_controlled_identity_scenario_audit_pass"
        and identity_eval["status"] == "objectstate_controlled_identity_eval_pass"
        and controlled_real_summary["gate"]["status"] == "objectstate_reality_gate_pass"
    )
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA,
        "kind": "objectstate_controlled_identity_handoff",
        "status": (
            "objectstate_controlled_identity_handoff_pass"
            if passed
            else "objectstate_controlled_identity_handoff_fail"
        ),
        "prediction_schema": OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA,
        "capture_file_audit_schema": OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA,
        "candidate_artifact_file_audit_schema": (
            OBJECTSTATE_CONTROLLED_CANDIDATE_ARTIFACT_FILE_AUDIT_SCHEMA
        ),
        "identity_scenario_audit_schema": (
            OBJECTSTATE_CONTROLLED_IDENTITY_SCENARIO_AUDIT_SCHEMA
        ),
        "identity_eval_schema": OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA,
        "controlled_real_manifest_schema": OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
        "controlled_real_rows_schema": OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
        "sample": dict(identity_eval["sample"]),
        "candidate": dict(identity_eval["candidate"]),
        "capture_file_audit": capture_file_audit,
        "candidate_artifact_file_audit": candidate_artifact_file_audit,
        "candidate_artifact_ref_match": candidate_artifact_ref_match,
        "identity_scenario_audit": identity_scenario_audit,
        "identity_predictions": predictions,
        "identity_eval": identity_eval,
        "controlled_real_manifest": controlled_real_manifest,
        "controlled_real_summary": controlled_real_summary,
        "handoff_contract": {
            "writes_predictions": True,
            "writes_identity_eval": True,
            "writes_controlled_real_manifest": True,
            "writes_identity_only_gate_summary": True,
            "requires_capture_file_audit_pass": True,
            "requires_candidate_artifact_file_audit_pass": True,
            "requires_candidate_artifact_ref_match": True,
            "requires_identity_scenario_audit_pass": True,
            "prediction_and_intervention_rows_remain_visible": True,
        },
        "claim_policy": {
            "capture_ground_truth_required": True,
            "capture_bundle_file_audit_required": True,
            "candidate_artifact_required": True,
            "candidate_artifact_file_audit_required": True,
            "candidate_artifact_ref_must_match_file_audit": True,
            "identity_scenario_challenge_required": True,
            "identity_only_stage1_gate": True,
            "does_not_claim_prediction_or_intervention": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "runs_tracking_model": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_controlled_identity_handoff_summary(payload)


def validate_objectstate_controlled_identity_handoff_summary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("controlled identity handoff summary must be a dict")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA:
        raise ValueError(
            f"unsupported controlled identity handoff schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_identity_handoff":
        raise ValueError("controlled identity handoff kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_identity_handoff_pass",
        "objectstate_controlled_identity_handoff_fail",
    }:
        raise ValueError("controlled identity handoff status is unsupported")
    if payload.get("prediction_schema") != OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA:
        raise ValueError("controlled identity handoff has unsupported prediction_schema")
    if payload.get("capture_file_audit_schema") != OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA:
        raise ValueError(
            "controlled identity handoff has unsupported capture_file_audit_schema"
        )
    if (
        payload.get("candidate_artifact_file_audit_schema")
        != OBJECTSTATE_CONTROLLED_CANDIDATE_ARTIFACT_FILE_AUDIT_SCHEMA
    ):
        raise ValueError(
            "controlled identity handoff has unsupported "
            "candidate_artifact_file_audit_schema"
        )
    if (
        payload.get("identity_scenario_audit_schema")
        != OBJECTSTATE_CONTROLLED_IDENTITY_SCENARIO_AUDIT_SCHEMA
    ):
        raise ValueError(
            "controlled identity handoff has unsupported identity_scenario_audit_schema"
        )
    if payload.get("identity_eval_schema") != OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA:
        raise ValueError("controlled identity handoff has unsupported identity_eval_schema")
    if payload.get("controlled_real_manifest_schema") != OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA:
        raise ValueError("controlled identity handoff has unsupported controlled_real_manifest_schema")
    if payload.get("controlled_real_rows_schema") != OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA:
        raise ValueError("controlled identity handoff has unsupported controlled_real_rows_schema")

    capture_file_audit = validate_objectstate_controlled_capture_file_audit_summary(
        payload.get("capture_file_audit")
    )
    candidate_artifact_file_audit = _validate_candidate_artifact_file_audit(
        payload.get("candidate_artifact_file_audit")
    )
    candidate_artifact_ref_match = _validate_candidate_artifact_ref_match(
        payload.get("candidate_artifact_ref_match")
    )
    identity_scenario_audit = _validate_identity_scenario_audit(
        payload.get("identity_scenario_audit")
    )
    predictions = validate_objectstate_controlled_identity_predictions(
        payload.get("identity_predictions")
    )
    identity_eval = validate_objectstate_controlled_identity_eval_summary(
        payload.get("identity_eval")
    )
    controlled_real_manifest = validate_objectstate_controlled_real_manifest(
        payload.get("controlled_real_manifest")
    )
    controlled_real_summary = validate_objectstate_controlled_real_rows_summary(
        payload.get("controlled_real_summary")
    )
    if capture_file_audit["sample"]["sample_id"] != identity_eval["sample"]["sample_id"]:
        raise ValueError("controlled identity handoff file audit sample mismatch")
    if identity_scenario_audit["sample"]["sample_id"] != identity_eval["sample"]["sample_id"]:
        raise ValueError("controlled identity handoff scenario audit sample mismatch")
    if predictions["sample_id"] != identity_eval["sample"]["sample_id"]:
        raise ValueError("controlled identity handoff sample ids must match")
    expected_artifact_refs = list(predictions["candidate"]["artifact_refs"])
    if candidate_artifact_ref_match["artifact_refs"] != expected_artifact_refs:
        raise ValueError("controlled identity handoff candidate artifact refs mismatch")
    if (
        candidate_artifact_ref_match["audited_path"]
        != candidate_artifact_file_audit["file_record"]["path"]
    ):
        raise ValueError("controlled identity handoff candidate audited path mismatch")
    if identity_eval["controlled_real_manifest"] != controlled_real_manifest:
        raise ValueError("controlled identity handoff manifest must come from identity eval")
    if controlled_real_summary["sample"]["sample_id"] != identity_eval["sample"]["sample_id"]:
        raise ValueError("controlled identity handoff controlled-real summary sample mismatch")
    hard_blockers = set(controlled_real_summary["gate"].get("hard_blockers", ()))
    if "prediction_pass_rows_present" in hard_blockers:
        raise ValueError("controlled identity handoff must not require prediction pass rows")
    if "intervention_pass_rows_present" in hard_blockers:
        raise ValueError("controlled identity handoff must not require intervention pass rows")
    expected_status = (
        "objectstate_controlled_identity_handoff_pass"
        if capture_file_audit["status"]
        == "objectstate_controlled_capture_file_audit_pass"
        and candidate_artifact_file_audit["status"]
        == "objectstate_controlled_candidate_artifact_file_audit_pass"
        and candidate_artifact_ref_match["matches"]
        and identity_scenario_audit["status"]
        == "objectstate_controlled_identity_scenario_audit_pass"
        and identity_eval["status"] == "objectstate_controlled_identity_eval_pass"
        and controlled_real_summary["gate"]["status"] == "objectstate_reality_gate_pass"
        else "objectstate_controlled_identity_handoff_fail"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled identity handoff status must match embedded gates")
    handoff_contract = payload.get("handoff_contract", {})
    if (
        not handoff_contract.get("writes_predictions")
        or not handoff_contract.get("writes_identity_eval")
        or not handoff_contract.get("writes_controlled_real_manifest")
        or not handoff_contract.get("writes_identity_only_gate_summary")
        or not handoff_contract.get("requires_capture_file_audit_pass")
        or not handoff_contract.get("requires_candidate_artifact_file_audit_pass")
        or not handoff_contract.get("requires_candidate_artifact_ref_match")
        or not handoff_contract.get("requires_identity_scenario_audit_pass")
        or not handoff_contract.get("prediction_and_intervention_rows_remain_visible")
    ):
        raise ValueError("controlled identity handoff contract is incomplete")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("capture_ground_truth_required")
        or not claim_policy.get("capture_bundle_file_audit_required")
        or not claim_policy.get("candidate_artifact_required")
        or not claim_policy.get("candidate_artifact_file_audit_required")
        or not claim_policy.get("candidate_artifact_ref_must_match_file_audit")
        or not claim_policy.get("identity_scenario_challenge_required")
        or not claim_policy.get("identity_only_stage1_gate")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled identity handoff must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("runs_tracking_model")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("writes_public_samples")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError("controlled identity handoff cannot claim capture, GT, tracking, training, replay, diffusion, or viewer mutation")
    return payload


def _identity_scenario_audit(
    capture_manifest: Mapping[str, Any],
    *,
    min_frames: int,
    min_occlusion_fraction: float,
    min_view_conditions: int,
    min_lighting_conditions: int,
    min_camera_motion_m: float,
) -> dict[str, Any]:
    if min_frames < 3:
        raise ValueError("min_identity_scenario_frames must be at least 3")
    if min_occlusion_fraction < 0.0 or min_occlusion_fraction > 1.0:
        raise ValueError("min_occlusion_fraction must be in [0, 1]")
    if min_view_conditions < 1:
        raise ValueError("min_view_conditions must be at least 1")
    if min_lighting_conditions < 1:
        raise ValueError("min_lighting_conditions must be at least 1")
    if min_camera_motion_m < 0.0:
        raise ValueError("min_camera_motion_m must be non-negative")
    checked_manifest = validate_objectstate_controlled_capture_manifest(capture_manifest)
    frames = checked_manifest["frames"]
    scenario_coverage = _identity_scenario_condition_coverage(frames)
    tracks: dict[str, list[dict[str, Any]]] = {
        item["object_id"]: [] for item in checked_manifest["objects"]
    }
    for index, frame in enumerate(frames):
        for item in frame["objects"]:
            occlusion_fraction = float(item.get("occlusion_fraction", 0.0))
            visible = bool(item.get("visible", True))
            occluded = (not visible) or occlusion_fraction >= min_occlusion_fraction
            tracks[str(item["object_id"])].append(
                {
                    "frame_index": index,
                    "frame_id": frame["frame_id"],
                    "timestamp": frame["timestamp"],
                    "visible": visible,
                    "occlusion_fraction": occlusion_fraction,
                    "occluded": occluded,
                }
            )
    object_tracks = []
    occlusion_reappearance_present = False
    for object_id, observations in tracks.items():
        occluded_indices = [
            item["frame_index"] for item in observations if item["occluded"]
        ]
        clear_visible_indices = [
            item["frame_index"]
            for item in observations
            if item["visible"] and not item["occluded"]
        ]
        reappears = any(
            any(index < occluded for index in clear_visible_indices)
            and any(index > occluded for index in clear_visible_indices)
            for occluded in occluded_indices
        )
        occlusion_reappearance_present = occlusion_reappearance_present or reappears
        object_tracks.append(
            {
                "object_id": object_id,
                "observation_count": len(observations),
                "clear_visible_count": len(clear_visible_indices),
                "occluded_count": len(occluded_indices),
                "occlusion_reappearance": reappears,
            }
        )
    readiness = {
        "min_frame_count_met": len(frames) >= min_frames,
        "occlusion_reappearance_present": occlusion_reappearance_present,
        "min_view_conditions_met": (
            scenario_coverage["view_condition_count"] >= min_view_conditions
        ),
        "min_lighting_conditions_met": (
            scenario_coverage["lighting_condition_count"] >= min_lighting_conditions
        ),
        "camera_motion_present": (
            scenario_coverage["camera_pose_count"] >= 2
            and scenario_coverage["max_camera_translation_m"] >= min_camera_motion_m
        ),
    }
    issues = []
    if not readiness["min_frame_count_met"]:
        issues.append(f"identity scenario requires at least {min_frames} frames")
    if not readiness["occlusion_reappearance_present"]:
        issues.append(
            "identity scenario requires clear-visible-before, occluded, "
            "clear-visible-after observations for at least one object"
        )
    if not readiness["min_view_conditions_met"]:
        issues.append(
            "identity scenario requires at least "
            f"{min_view_conditions} distinct frame.condition.view_id values"
        )
    if not readiness["min_lighting_conditions_met"]:
        issues.append(
            "identity scenario requires at least "
            f"{min_lighting_conditions} distinct frame.condition.lighting_id values"
        )
    if not readiness["camera_motion_present"]:
        issues.append(
            "identity scenario requires at least two frame.condition.camera_pose "
            "values with max translation >= "
            f"{min_camera_motion_m:.6f}m"
        )
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_IDENTITY_SCENARIO_AUDIT_SCHEMA,
        "kind": "objectstate_controlled_identity_scenario_audit",
        "capture_schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "status": (
            "objectstate_controlled_identity_scenario_audit_pass"
            if all(readiness.values())
            else "objectstate_controlled_identity_scenario_audit_fail"
        ),
        "sample": dict(checked_manifest["sample"]),
        "frame_count": len(frames),
        "requirements": {
            "min_frames": int(min_frames),
            "min_occlusion_fraction": float(min_occlusion_fraction),
            "min_view_conditions": int(min_view_conditions),
            "min_lighting_conditions": int(min_lighting_conditions),
            "min_camera_motion_m": float(min_camera_motion_m),
            "requires_occlusion_reappearance": True,
            "requires_view_change": True,
            "requires_lighting_change": True,
            "requires_camera_motion": True,
        },
        "readiness": readiness,
        "scenario_coverage": scenario_coverage,
        "object_tracks": object_tracks,
        "issues": issues,
        "claim_policy": {
            "identity_scenario_required_for_handoff_pass": True,
            "scenario_audit_does_not_read_image_pixels": True,
            "scenario_audit_does_not_prove_model_quality": True,
            "scenario_audit_does_not_verify_lighting_or_camera_motion": True,
            "scenario_audit_uses_manifest_condition_metadata": True,
        },
    }
    return _validate_identity_scenario_audit(payload)


def _validate_identity_scenario_audit(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("identity scenario audit must be a dict")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_IDENTITY_SCENARIO_AUDIT_SCHEMA:
        raise ValueError(
            f"unsupported identity scenario audit schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_identity_scenario_audit":
        raise ValueError("identity scenario audit kind is unsupported")
    if payload.get("capture_schema") != OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA:
        raise ValueError("identity scenario audit has unsupported capture_schema")
    if payload.get("status") not in {
        "objectstate_controlled_identity_scenario_audit_pass",
        "objectstate_controlled_identity_scenario_audit_fail",
    }:
        raise ValueError("identity scenario audit status is unsupported")
    if not isinstance(payload.get("sample"), dict):
        raise ValueError("identity scenario audit requires sample")
    if not isinstance(payload.get("frame_count"), int) or payload["frame_count"] < 1:
        raise ValueError("identity scenario audit requires positive frame_count")
    requirements = payload.get("requirements")
    readiness = payload.get("readiness")
    if not isinstance(requirements, dict):
        raise ValueError("identity scenario audit requires requirements")
    if (
        not isinstance(requirements.get("min_frames"), int)
        or int(requirements["min_frames"]) < 3
        or not isinstance(requirements.get("min_occlusion_fraction"), float)
        or requirements["min_occlusion_fraction"] < 0.0
        or requirements["min_occlusion_fraction"] > 1.0
        or not isinstance(requirements.get("min_view_conditions"), int)
        or int(requirements["min_view_conditions"]) < 1
        or not isinstance(requirements.get("min_lighting_conditions"), int)
        or int(requirements["min_lighting_conditions"]) < 1
        or not isinstance(requirements.get("min_camera_motion_m"), float)
        or requirements["min_camera_motion_m"] < 0.0
        or not requirements.get("requires_occlusion_reappearance")
        or not requirements.get("requires_view_change")
        or not requirements.get("requires_lighting_change")
        or not requirements.get("requires_camera_motion")
    ):
        raise ValueError("identity scenario audit requirements are invalid")
    if not isinstance(readiness, dict):
        raise ValueError("identity scenario audit requires readiness")
    for key in (
        "min_frame_count_met",
        "occlusion_reappearance_present",
        "min_view_conditions_met",
        "min_lighting_conditions_met",
        "camera_motion_present",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"identity scenario audit readiness missing bool {key}")
    expected_status = (
        "objectstate_controlled_identity_scenario_audit_pass"
        if all(readiness.values())
        else "objectstate_controlled_identity_scenario_audit_fail"
    )
    if payload["status"] != expected_status:
        raise ValueError("identity scenario audit status must match readiness")
    scenario_coverage = payload.get("scenario_coverage")
    if not isinstance(scenario_coverage, dict):
        raise ValueError("identity scenario audit requires scenario_coverage")
    if (
        not isinstance(scenario_coverage.get("view_condition_count"), int)
        or scenario_coverage["view_condition_count"] < 0
        or not isinstance(scenario_coverage.get("lighting_condition_count"), int)
        or scenario_coverage["lighting_condition_count"] < 0
        or not isinstance(scenario_coverage.get("camera_pose_count"), int)
        or scenario_coverage["camera_pose_count"] < 0
        or not isinstance(scenario_coverage.get("max_camera_translation_m"), float)
        or scenario_coverage["max_camera_translation_m"] < 0.0
        or not isinstance(scenario_coverage.get("view_ids"), list)
        or not isinstance(scenario_coverage.get("lighting_ids"), list)
    ):
        raise ValueError("identity scenario audit scenario_coverage is invalid")
    if not isinstance(payload.get("object_tracks"), list):
        raise ValueError("identity scenario audit requires object_tracks")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("identity scenario audit requires issues")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("identity_scenario_required_for_handoff_pass")
        or not claim_policy.get("scenario_audit_does_not_read_image_pixels")
        or not claim_policy.get("scenario_audit_does_not_prove_model_quality")
        or not claim_policy.get("scenario_audit_does_not_verify_lighting_or_camera_motion")
        or not claim_policy.get("scenario_audit_uses_manifest_condition_metadata")
    ):
        raise ValueError("identity scenario audit must preserve claim policy")
    return payload


def _identity_scenario_condition_coverage(
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    view_ids: set[str] = set()
    lighting_ids: set[str] = set()
    camera_positions: list[Sequence[float]] = []
    for frame in frames:
        condition = frame.get("condition", {})
        if not isinstance(condition, Mapping):
            continue
        view_id = condition.get("view_id")
        lighting_id = condition.get("lighting_id")
        if isinstance(view_id, str) and view_id:
            view_ids.add(view_id)
        if isinstance(lighting_id, str) and lighting_id:
            lighting_ids.add(lighting_id)
        camera_pose = condition.get("camera_pose")
        if isinstance(camera_pose, Mapping):
            position = camera_pose.get("position")
            if (
                isinstance(position, Sequence)
                and not isinstance(position, (str, bytes))
                and len(position) == 3
            ):
                camera_positions.append([float(component) for component in position])
    max_camera_translation_m = _max_camera_translation(camera_positions)
    return {
        "view_ids": sorted(view_ids),
        "view_condition_count": len(view_ids),
        "lighting_ids": sorted(lighting_ids),
        "lighting_condition_count": len(lighting_ids),
        "camera_pose_count": len(camera_positions),
        "max_camera_translation_m": float(max_camera_translation_m),
    }


def _max_camera_translation(camera_positions: Sequence[Sequence[float]]) -> float:
    max_distance = 0.0
    for left_index, left in enumerate(camera_positions):
        for right in camera_positions[left_index + 1 :]:
            squared = sum(
                (float(left[axis]) - float(right[axis])) ** 2 for axis in range(3)
            )
            max_distance = max(max_distance, squared**0.5)
    return max_distance


def _candidate_artifact_ref_match(
    candidate_artifact_file_audit: Mapping[str, Any],
    predictions: Mapping[str, Any],
) -> dict[str, Any]:
    file_record = candidate_artifact_file_audit["file_record"]
    audited_path = str(file_record.get("path", ""))
    candidate = predictions.get("candidate", {})
    refs = candidate.get("artifact_refs", ()) if isinstance(candidate, Mapping) else ()
    artifact_refs = [str(ref) for ref in refs]
    matches = bool(audited_path and audited_path in artifact_refs)
    payload = {
        "audited_path": audited_path,
        "artifact_refs": artifact_refs,
        "matches": matches,
    }
    if not matches:
        payload["missing_reason"] = "audited candidate artifact path not in artifact_refs"
    return _validate_candidate_artifact_ref_match(payload)


def _validate_candidate_artifact_ref_match(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("candidate artifact ref match must be a dict")
    if not isinstance(payload.get("audited_path"), str):
        raise ValueError("candidate artifact ref match requires audited_path")
    refs = payload.get("artifact_refs")
    if (
        isinstance(refs, (str, bytes))
        or not isinstance(refs, list)
        or any(not isinstance(ref, str) or not ref for ref in refs)
    ):
        raise ValueError("candidate artifact ref match requires artifact_refs")
    if not isinstance(payload.get("matches"), bool):
        raise ValueError("candidate artifact ref match requires matches")
    if not payload["matches"] and not isinstance(payload.get("missing_reason"), str):
        raise ValueError("candidate artifact ref mismatch requires missing_reason")
    return payload


def _candidate_artifact_file_audit(
    artifact_path: str | Path | None,
    *,
    min_bytes: int,
    hash_file: bool,
) -> dict[str, Any]:
    if min_bytes < 0:
        raise ValueError("min_candidate_artifact_bytes must be non-negative")
    if artifact_path is None:
        record = {
            "path": "",
            "exists": False,
            "is_file": False,
            "size_bytes": None,
            "valid": False,
            "missing_reason": "candidate artifact path not provided",
        }
    else:
        path = Path(artifact_path)
        exists = path.exists()
        is_file = bool(exists and path.is_file())
        size_bytes = path.stat().st_size if is_file else None
        valid = bool(exists)
        missing_reason = None
        if not exists:
            valid = False
            missing_reason = "path does not exist"
        elif not is_file:
            valid = False
            missing_reason = "path is not a file"
        elif size_bytes is not None and size_bytes < min_bytes:
            valid = False
            missing_reason = (
                "file smaller than required minimum bytes "
                f"({size_bytes} < {min_bytes})"
            )
        record = {
            "path": str(path),
            "exists": bool(exists),
            "is_file": is_file,
            "size_bytes": size_bytes,
            "valid": valid,
        }
        if missing_reason is not None:
            record["missing_reason"] = missing_reason
        if hash_file and valid and is_file:
            record["sha256"] = _sha256_file(path)
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_CANDIDATE_ARTIFACT_FILE_AUDIT_SCHEMA,
        "kind": "objectstate_controlled_candidate_artifact_file_audit",
        "status": (
            "objectstate_controlled_candidate_artifact_file_audit_pass"
            if record["valid"]
            else "objectstate_controlled_candidate_artifact_file_audit_fail"
        ),
        "requirements": {
            "candidate_artifact_file_required": True,
            "candidate_artifact_must_be_file": True,
            "min_candidate_artifact_bytes": int(min_bytes),
            "file_hash_included": bool(hash_file),
        },
        "file_record": record,
        "claim_policy": {
            "candidate_artifact_file_required_for_handoff_pass": True,
            "candidate_artifact_audit_does_not_train_model": True,
            "candidate_artifact_audit_does_not_prove_model_quality": True,
        },
    }
    return _validate_candidate_artifact_file_audit(payload)


def _validate_candidate_artifact_file_audit(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("candidate artifact file audit must be a dict")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_CANDIDATE_ARTIFACT_FILE_AUDIT_SCHEMA:
        raise ValueError(
            f"unsupported candidate artifact file audit schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_candidate_artifact_file_audit":
        raise ValueError("candidate artifact file audit kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_candidate_artifact_file_audit_pass",
        "objectstate_controlled_candidate_artifact_file_audit_fail",
    }:
        raise ValueError("candidate artifact file audit status is unsupported")
    requirements = payload.get("requirements")
    record = payload.get("file_record")
    claim_policy = payload.get("claim_policy")
    if not isinstance(requirements, dict):
        raise ValueError("candidate artifact file audit requires requirements")
    if not isinstance(record, dict):
        raise ValueError("candidate artifact file audit requires file_record")
    if not isinstance(claim_policy, dict):
        raise ValueError("candidate artifact file audit requires claim_policy")
    if (
        not requirements.get("candidate_artifact_file_required")
        or not requirements.get("candidate_artifact_must_be_file")
        or not isinstance(requirements.get("min_candidate_artifact_bytes"), int)
        or int(requirements["min_candidate_artifact_bytes"]) < 0
        or not isinstance(requirements.get("file_hash_included"), bool)
    ):
        raise ValueError("candidate artifact file audit requirements are invalid")
    for key in ("exists", "is_file", "valid"):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"candidate artifact file audit record missing bool {key}")
    if not isinstance(record.get("path"), str):
        raise ValueError("candidate artifact file audit record requires path")
    if record.get("size_bytes") is not None and (
        not isinstance(record["size_bytes"], int) or int(record["size_bytes"]) < 0
    ):
        raise ValueError("candidate artifact file audit size_bytes is invalid")
    expected_status = (
        "objectstate_controlled_candidate_artifact_file_audit_pass"
        if record["valid"]
        else "objectstate_controlled_candidate_artifact_file_audit_fail"
    )
    if payload["status"] != expected_status:
        raise ValueError("candidate artifact file audit status must match record validity")
    if (
        not claim_policy.get("candidate_artifact_file_required_for_handoff_pass")
        or not claim_policy.get("candidate_artifact_audit_does_not_train_model")
        or not claim_policy.get("candidate_artifact_audit_does_not_prove_model_quality")
    ):
        raise ValueError("candidate artifact file audit must preserve claim policy")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
