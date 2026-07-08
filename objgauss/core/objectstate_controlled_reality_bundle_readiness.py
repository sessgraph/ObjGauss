from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_capture_bundle_readiness import (
    OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_READINESS_SCHEMA,
    objectstate_controlled_capture_bundle_readiness,
    validate_objectstate_controlled_capture_bundle_readiness_summary,
)
from objgauss.core.objectstate_controlled_intervention_eval import (
    read_objectstate_controlled_intervention_candidates,
)
from objgauss.core.objectstate_controlled_prediction_eval import (
    read_objectstate_controlled_prediction_candidates,
)
from objgauss.core.objectstate_identity_prediction_adapter import (
    objectstate_identity_predictions_from_trainable_artifact,
    read_trainable_kernel_identity_source,
)

OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_READINESS_SCHEMA = (
    "objgauss-objectstate-controlled-reality-bundle-readiness-v1"
)


def objectstate_controlled_reality_bundle_readiness(
    root: str | Path,
    trainable_artifact: str | Path,
    prediction_candidates: str | Path,
    intervention_candidates: str | Path,
    *,
    sample_json: str | Path = "sample.json",
    objects_csv: str | Path = "objects.csv",
    frames_csv: str | Path = "frames.csv",
    annotations_csv: str | Path = "annotations.csv",
    actions_csv: str | Path | None = "actions.csv",
    max_centroid_distance: float | None = None,
    min_rgb_bytes: int = 1,
    min_gaussian_bytes: int = 1,
    require_frame_formats: bool = True,
    hash_files: bool = False,
    min_candidate_artifact_bytes: int = 1,
    min_identity_scenario_frames: int = 3,
    min_occlusion_fraction: float = 0.5,
    min_view_conditions: int = 2,
    min_lighting_conditions: int = 2,
    min_camera_motion_m: float = 0.01,
) -> dict[str, Any]:
    bundle_root = Path(root)
    trainable_path = _resolve_input_path(bundle_root, trainable_artifact)
    prediction_path = _resolve_input_path(bundle_root, prediction_candidates)
    intervention_path = _resolve_input_path(bundle_root, intervention_candidates)
    capture_readiness = objectstate_controlled_capture_bundle_readiness(
        bundle_root,
        sample_json=sample_json,
        objects_csv=objects_csv,
        frames_csv=frames_csv,
        annotations_csv=annotations_csv,
        actions_csv=actions_csv,
        require_prediction_ready=True,
        require_intervention_ready=True,
        min_rgb_bytes=min_rgb_bytes,
        min_gaussian_bytes=min_gaussian_bytes,
        require_frame_formats=require_frame_formats,
        hash_files=hash_files,
        candidate_artifact=trainable_path,
        require_candidate_artifact=True,
        min_candidate_artifact_bytes=min_candidate_artifact_bytes,
        min_identity_scenario_frames=min_identity_scenario_frames,
        min_occlusion_fraction=min_occlusion_fraction,
        min_view_conditions=min_view_conditions,
        min_lighting_conditions=min_lighting_conditions,
        min_camera_motion_m=min_camera_motion_m,
    )
    manifest = None
    if capture_readiness.get("import_summary") is not None:
        manifest = capture_readiness["import_summary"]["manifest"]
    trainable = _trainable_artifact_readiness(
        trainable_path,
        manifest=manifest,
        max_centroid_distance=max_centroid_distance,
    )
    prediction = _candidate_readiness(
        prediction_path,
        kind="prediction",
        reader=read_objectstate_controlled_prediction_candidates,
        manifest=manifest,
    )
    intervention = _candidate_readiness(
        intervention_path,
        kind="intervention",
        reader=read_objectstate_controlled_intervention_candidates,
        manifest=manifest,
    )
    readiness = {
        "capture_bundle_ready": bool(
            capture_readiness["readiness"]["capture_bundle_ready"]
        ),
        "identity_bundle_handoff_ready": bool(
            capture_readiness["readiness"]["identity_bundle_handoff_ready"]
        ),
        "intervention_action_gt_ready": bool(
            capture_readiness["readiness"]["intervention_action_gt_ready"]
        ),
        "trainable_artifact_schema_ready": bool(trainable["schema_ready"]),
        "trainable_artifact_binding_ready": bool(trainable["binding_ready"]),
        "prediction_candidates_schema_ready": bool(prediction["schema_ready"]),
        "prediction_candidates_binding_ready": bool(prediction["binding_ready"]),
        "intervention_candidates_schema_ready": bool(intervention["schema_ready"]),
        "intervention_candidates_binding_ready": bool(intervention["binding_ready"]),
    }
    readiness["full_reality_handoff_ready"] = all(readiness.values())
    hard_blockers = _hard_blockers(
        capture_readiness=capture_readiness,
        trainable=trainable,
        prediction=prediction,
        intervention=intervention,
        readiness=readiness,
    )
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_READINESS_SCHEMA,
        "kind": "objectstate_controlled_reality_bundle_readiness",
        "status": (
            "objectstate_controlled_reality_bundle_readiness_ready"
            if readiness["full_reality_handoff_ready"]
            else "objectstate_controlled_reality_bundle_readiness_blocked"
        ),
        "root": str(bundle_root),
        "paths": {
            "trainable_artifact": str(trainable_path),
            "prediction_candidates": str(prediction_path),
            "intervention_candidates": str(intervention_path),
        },
        "capture_readiness_schema": OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_READINESS_SCHEMA,
        "capture_readiness": capture_readiness,
        "trainable_artifact": trainable,
        "prediction_candidates": prediction,
        "intervention_candidates": intervention,
        "readiness": readiness,
        "hard_blockers": hard_blockers,
        "next_actions": _next_actions(hard_blockers),
        "claim_policy": {
            "readiness_audit_is_pre_handoff": True,
            "readiness_audit_may_run_on_incomplete_inputs": True,
            "readiness_audit_checks_candidate_schema_and_bindings": True,
            "readiness_audit_does_not_score_candidate_quality": True,
            "readiness_audit_does_not_create_predictions": True,
            "readiness_audit_does_not_claim_reality_gate_pass": True,
            "readiness_audit_does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "creates_prediction_candidates": False,
            "creates_intervention_candidates": False,
            "reconstructs_gaussians": False,
            "runs_identity_handoff": False,
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
    return validate_objectstate_controlled_reality_bundle_readiness_summary(payload)


def validate_objectstate_controlled_reality_bundle_readiness_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled reality bundle readiness summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_READINESS_SCHEMA:
        raise ValueError(
            "unsupported controlled reality bundle readiness schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_reality_bundle_readiness":
        raise ValueError("controlled reality bundle readiness kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_reality_bundle_readiness_ready",
        "objectstate_controlled_reality_bundle_readiness_blocked",
    }:
        raise ValueError("controlled reality bundle readiness status is unsupported")
    validate_objectstate_controlled_capture_bundle_readiness_summary(
        payload.get("capture_readiness")
    )
    for key in (
        "root",
        "paths",
        "trainable_artifact",
        "prediction_candidates",
        "intervention_candidates",
        "readiness",
        "hard_blockers",
        "next_actions",
    ):
        if key not in payload:
            raise ValueError(f"controlled reality bundle readiness requires {key}")
    readiness = payload["readiness"]
    if not isinstance(readiness, Mapping):
        raise ValueError("controlled reality bundle readiness requires readiness")
    for key in (
        "capture_bundle_ready",
        "identity_bundle_handoff_ready",
        "intervention_action_gt_ready",
        "trainable_artifact_schema_ready",
        "trainable_artifact_binding_ready",
        "prediction_candidates_schema_ready",
        "prediction_candidates_binding_ready",
        "intervention_candidates_schema_ready",
        "intervention_candidates_binding_ready",
        "full_reality_handoff_ready",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"controlled reality readiness missing bool {key}")
    expected_ready = all(
        bool(value)
        for key, value in readiness.items()
        if key != "full_reality_handoff_ready"
    )
    if readiness["full_reality_handoff_ready"] != expected_ready:
        raise ValueError("full reality handoff readiness must match child gates")
    expected_status = (
        "objectstate_controlled_reality_bundle_readiness_ready"
        if readiness["full_reality_handoff_ready"]
        else "objectstate_controlled_reality_bundle_readiness_blocked"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled reality bundle readiness status mismatch")
    if not isinstance(payload.get("hard_blockers"), list):
        raise ValueError("controlled reality bundle readiness blockers must be list")
    if not isinstance(payload.get("next_actions"), list):
        raise ValueError("controlled reality bundle readiness actions must be list")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("readiness_audit_is_pre_handoff")
        or not claim_policy.get("readiness_audit_may_run_on_incomplete_inputs")
        or not claim_policy.get("readiness_audit_checks_candidate_schema_and_bindings")
        or not claim_policy.get("readiness_audit_does_not_score_candidate_quality")
        or not claim_policy.get("readiness_audit_does_not_create_predictions")
        or not claim_policy.get("readiness_audit_does_not_claim_reality_gate_pass")
        or not claim_policy.get("readiness_audit_does_not_claim_world_model")
    ):
        raise ValueError("controlled reality readiness must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("creates_prediction_candidates")
        or non_goals.get("creates_intervention_candidates")
        or non_goals.get("reconstructs_gaussians")
        or non_goals.get("runs_identity_handoff")
        or non_goals.get("runs_prediction_eval")
        or non_goals.get("runs_intervention_eval")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("writes_public_samples")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError(
            "controlled reality readiness cannot claim capture, GT, candidate "
            "creation, reconstruction, handoff, eval, training, replay, diffusion, "
            "public samples, or viewer mutation"
        )
    return dict(payload)


def _trainable_artifact_readiness(
    path: Path,
    *,
    manifest: Mapping[str, Any] | None,
    max_centroid_distance: float | None,
) -> dict[str, Any]:
    base = _file_readiness(path)
    artifact = None
    schema_error = None
    identity_binding_error = None
    identity_prediction_count = 0
    if base["is_file"]:
        try:
            artifact = read_trainable_kernel_identity_source(path)
        except Exception as exc:  # noqa: BLE001 - readiness reports validation errors.
            schema_error = str(exc)
    if artifact is not None and manifest is not None:
        try:
            predictions = objectstate_identity_predictions_from_trainable_artifact(
                manifest,
                artifact,
                artifact_refs=(str(path),),
                max_centroid_distance=max_centroid_distance,
            )
            identity_prediction_count = len(predictions["predictions"])
        except Exception as exc:  # noqa: BLE001 - readiness reports validation errors.
            identity_binding_error = str(exc)
    schema_ready = artifact is not None
    binding_ready = schema_ready and manifest is not None and identity_binding_error is None
    return {
        **base,
        "schema_ready": schema_ready,
        "schema_error": schema_error,
        "binding_ready": binding_ready,
        "binding_error": identity_binding_error,
        "identity_prediction_count": identity_prediction_count,
    }


def _candidate_readiness(
    path: Path,
    *,
    kind: str,
    reader: Any,
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    base = _file_readiness(path)
    candidate = None
    schema_error = None
    if base["is_file"]:
        try:
            candidate = reader(path)
        except Exception as exc:  # noqa: BLE001 - readiness reports validation errors.
            schema_error = str(exc)
    binding = _candidate_binding(kind, candidate, manifest) if candidate is not None else {
        "ready": False,
        "issues": ["candidate schema did not validate"],
        "record_count": 0,
        "sample_id": None,
        "sample_matches_capture": False,
    }
    return {
        **base,
        "schema_ready": candidate is not None,
        "schema_error": schema_error,
        "binding_ready": bool(binding["ready"]),
        "binding_issues": binding["issues"],
        "record_count": int(binding["record_count"]),
        "sample_id": binding["sample_id"],
        "sample_matches_capture": bool(binding["sample_matches_capture"]),
    }


def _candidate_binding(
    kind: str,
    candidate: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if manifest is None:
        return {
            "ready": False,
            "issues": ["capture manifest is not import-ready"],
            "record_count": 0,
            "sample_id": candidate["sample_id"],
            "sample_matches_capture": False,
        }
    issues = []
    sample_matches = candidate["sample_id"] == manifest["sample"]["sample_id"]
    if not sample_matches:
        issues.append("candidate sample_id does not match capture sample_id")
    if kind == "prediction":
        record_count = len(candidate["predictions"])
        issues.extend(_prediction_binding_issues(candidate["predictions"], manifest))
    elif kind == "intervention":
        record_count = len(candidate["interventions"])
        issues.extend(
            _intervention_binding_issues(candidate["interventions"], manifest)
        )
    else:
        raise ValueError(f"unsupported controlled reality candidate kind: {kind}")
    return {
        "ready": sample_matches and not issues and record_count > 0,
        "issues": issues,
        "record_count": record_count,
        "sample_id": candidate["sample_id"],
        "sample_matches_capture": sample_matches,
    }


def _prediction_binding_issues(
    predictions: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues = []
    frames = _frame_map(manifest)
    pose_keys = _pose_keys(manifest)
    seen = set()
    for item in predictions:
        key = (
            item["source_frame_id"],
            item["target_frame_id"],
            item["object_id"],
        )
        if key in seen:
            issues.append(
                "duplicate prediction tuple: "
                f"{key[0]} / {key[1]} / {key[2]}"
            )
            continue
        seen.add(key)
        issues.extend(_frame_object_interval_issues(key, frames, pose_keys))
    return issues


def _intervention_binding_issues(
    interventions: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues = []
    frames = _frame_map(manifest)
    pose_keys = _pose_keys(manifest)
    actions = _action_map(manifest)
    seen = set()
    for item in interventions:
        key = (
            item["source_frame_id"],
            item["target_frame_id"],
            item["object_id"],
            item["action_id"],
        )
        if key in seen:
            issues.append(
                "duplicate intervention tuple: "
                f"{key[0]} / {key[1]} / {key[2]} / {key[3]}"
            )
            continue
        seen.add(key)
        issues.extend(_frame_object_interval_issues(key[:3], frames, pose_keys))
        action = actions.get(item["action_id"])
        if action is None:
            issues.append(f"unknown action_id: {item['action_id']}")
            continue
        if action["object_id"] != item["object_id"]:
            issues.append("intervention action object_id mismatch")
        vector = action.get("vector")
        if not isinstance(vector, Sequence) or len(vector) != 3:
            issues.append("intervention action vector is missing")
        elif sum(float(value) ** 2 for value in vector) <= 1e-16:
            issues.append("intervention action vector must be non-zero")
        source = frames.get(item["source_frame_id"])
        target = frames.get(item["target_frame_id"])
        if source is not None and target is not None:
            if (
                float(action["start_timestamp"]) < float(source["timestamp"])
                or float(action["end_timestamp"]) > float(target["timestamp"])
            ):
                issues.append(
                    "intervention action interval does not fit source/target frames"
                )
    return issues


def _frame_object_interval_issues(
    key: tuple[str, str, str],
    frames: Mapping[str, Mapping[str, Any]],
    pose_keys: set[tuple[str, str]],
) -> list[str]:
    source_frame_id, target_frame_id, object_id = key
    issues = []
    source = frames.get(source_frame_id)
    target = frames.get(target_frame_id)
    if source is None:
        issues.append(f"unknown source_frame_id: {source_frame_id}")
    if target is None:
        issues.append(f"unknown target_frame_id: {target_frame_id}")
    if (source_frame_id, object_id) not in pose_keys:
        issues.append(f"missing source pose: {source_frame_id} / {object_id}")
    if (target_frame_id, object_id) not in pose_keys:
        issues.append(f"missing target pose: {target_frame_id} / {object_id}")
    if source is not None and target is not None:
        if float(target["timestamp"]) <= float(source["timestamp"]):
            issues.append("target frame timestamp must be after source frame")
    return issues


def _file_readiness(path: Path) -> dict[str, Any]:
    exists = path.exists()
    is_file = exists and path.is_file()
    size_bytes = path.stat().st_size if is_file else 0
    return {
        "path": str(path),
        "exists": exists,
        "is_file": is_file,
        "size_bytes": int(size_bytes),
    }


def _resolve_input_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return root / path


def _frame_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {frame["frame_id"]: frame for frame in manifest["frames"]}


def _pose_keys(manifest: Mapping[str, Any]) -> set[tuple[str, str]]:
    result = set()
    for frame in manifest["frames"]:
        frame_id = frame["frame_id"]
        for item in frame["objects"]:
            if item.get("pose") is not None:
                result.add((frame_id, item["object_id"]))
    return result


def _action_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {action["action_id"]: action for action in manifest["actions"]}


def _hard_blockers(
    *,
    capture_readiness: Mapping[str, Any],
    trainable: Mapping[str, Any],
    prediction: Mapping[str, Any],
    intervention: Mapping[str, Any],
    readiness: Mapping[str, bool],
) -> list[str]:
    blockers = []
    if not readiness["capture_bundle_ready"]:
        blockers.append("capture_bundle_ready")
        blockers.extend(
            f"capture:{item}" for item in capture_readiness["hard_blockers"]
        )
    if not readiness["identity_bundle_handoff_ready"]:
        blockers.append("identity_bundle_handoff_ready")
    if not readiness["trainable_artifact_schema_ready"]:
        blockers.append("trainable_artifact_schema_ready")
    if not readiness["trainable_artifact_binding_ready"]:
        blockers.append("trainable_artifact_binding_ready")
    if not readiness["prediction_candidates_schema_ready"]:
        blockers.append("prediction_candidates_schema_ready")
    if not readiness["prediction_candidates_binding_ready"]:
        blockers.append("prediction_candidates_binding_ready")
        blockers.extend(f"prediction:{item}" for item in prediction["binding_issues"])
    if not readiness["intervention_candidates_schema_ready"]:
        blockers.append("intervention_candidates_schema_ready")
    if not readiness["intervention_candidates_binding_ready"]:
        blockers.append("intervention_candidates_binding_ready")
        blockers.extend(
            f"intervention:{item}" for item in intervention["binding_issues"]
        )
    return blockers


def _next_actions(blockers: Sequence[str]) -> list[str]:
    actions = []
    if any(item.startswith("capture:") or item == "capture_bundle_ready" for item in blockers):
        actions.append(
            "complete the controlled capture bundle and rerun capture readiness audit"
        )
    if "trainable_artifact_schema_ready" in blockers:
        actions.append("provide a valid trainable ObjectState artifact JSON")
    if "trainable_artifact_binding_ready" in blockers:
        actions.append("align trainable artifact frame count and ObjectState poses to capture")
    if "prediction_candidates_schema_ready" in blockers:
        actions.append("provide valid prediction candidates JSON")
    if "prediction_candidates_binding_ready" in blockers:
        actions.append("bind prediction candidates to capture frame/object pose GT")
    if "intervention_candidates_schema_ready" in blockers:
        actions.append("provide valid intervention candidates JSON")
    if "intervention_candidates_binding_ready" in blockers:
        actions.append("bind intervention candidates to capture frame/object/action GT")
    if not actions:
        actions.append("run controlled-reality-bundle-handoff")
    return actions
