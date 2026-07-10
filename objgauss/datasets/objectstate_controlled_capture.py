from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.datasets.objectstate_controlled_real_manifest import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
    validate_objectstate_controlled_real_manifest,
)

OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA = (
    "objgauss-objectstate-controlled-capture-manifest-v1"
)
OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA = (
    "objgauss-objectstate-controlled-capture-summary-v1"
)


def read_objectstate_controlled_capture_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("controlled capture manifest JSON must be an object")
    return validate_objectstate_controlled_capture_manifest(payload)


def objectstate_controlled_capture_summary(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    checked_manifest = validate_objectstate_controlled_capture_manifest(manifest)
    frames = checked_manifest["frames"]
    objects = checked_manifest["objects"]
    actions = checked_manifest["actions"]
    observation_coverage = _observation_coverage(frames)
    object_track_counts = _object_track_counts(frames)
    ground_truth = _ground_truth_summary(
        frames,
        actions=actions,
        object_track_counts=object_track_counts,
    )
    readiness = _readiness_summary(
        ground_truth,
        frame_count=len(frames),
        object_track_counts=object_track_counts,
        observation_coverage=observation_coverage,
    )
    issues = _capture_issues(
        ground_truth,
        readiness=readiness,
        observation_coverage=observation_coverage,
    )
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA,
        "kind": "objectstate_controlled_capture_summary",
        "capture_schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": dict(checked_manifest["sample"]),
        "frame_count": len(frames),
        "object_count": len(objects),
        "action_count": len(actions),
        "object_track_counts": dict(object_track_counts),
        "observation_coverage": observation_coverage,
        "ground_truth": ground_truth,
        "readiness": readiness,
        "issues": issues,
        "controlled_real_manifest_seed": (
            objectstate_controlled_real_manifest_from_capture_manifest(checked_manifest)
        ),
        "claim_policy": {
            "capture_manifest_required": True,
            "manifest_does_not_prove_model_quality": True,
            "blocked_seed_rows_are_not_pass_rows": True,
            "candidate_metrics_required_for_pass_rows": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "reconstructs_gaussians": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_controlled_capture_summary(payload)


def objectstate_controlled_real_manifest_from_capture_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    checked_manifest = validate_objectstate_controlled_capture_manifest(manifest)
    summary = _capture_summary_parts(checked_manifest)
    sample = checked_manifest["sample"]
    seed = {
        "schema": OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
        "sample": {
            "sample_id": sample["sample_id"],
            "source_kind": "controlled_real",
            "object_category": sample["object_category"],
            "scenario": sample["scenario"],
            "observation_modalities": sample["observation_modalities"],
            "artifact_refs": sample["artifact_refs"],
            "license": sample["license"],
        },
        "ground_truth": summary["ground_truth"],
        "evidence_rows": [
            {
                "evidence_kind": "identity",
                "status": "blocked",
                "metrics": {},
                "block_reason": _seed_block_reason(
                    summary["readiness"]["identity_stage_ready"],
                    "missing candidate identity metrics: idf1, fragmentation_rate, "
                    "swap_rate, identity_collapse",
                    "capture manifest is not identity-ready: timestamped object_id "
                    "tracks across frames are required",
                ),
            },
            {
                "evidence_kind": "prediction",
                "status": "blocked",
                "metrics": {},
                "block_reason": _seed_block_reason(
                    summary["readiness"]["prediction_stage_ready"],
                    "missing state-vs-history prediction metrics: state_ade, "
                    "history_ade, prediction_gap_vs_history_model",
                    "capture manifest is not prediction-ready: timestamped 6DoF "
                    "pose tracks across frames are required",
                ),
            },
            {
                "evidence_kind": "intervention",
                "status": "blocked",
                "metrics": {},
                "block_reason": _seed_block_reason(
                    summary["readiness"]["intervention_stage_ready"],
                    "missing action-conditioned metrics: action_conditioned_ade, "
                    "counterfactual_outcome_accuracy, wrong_direction_rate",
                    "capture manifest is not intervention-ready: timestamped pose "
                    "tracks and action events are required",
                ),
            },
        ],
    }
    return validate_objectstate_controlled_real_manifest(seed)


def validate_objectstate_controlled_capture_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise TypeError("controlled capture manifest must be a mapping")
    if manifest.get("schema") != OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported controlled capture schema: {manifest.get('schema')}"
        )
    sample = _validate_sample(manifest.get("sample"))
    objects = tuple(_validate_object(item) for item in _sequence(manifest.get("objects"), "objects"))
    if not objects:
        raise ValueError("controlled capture manifest requires at least one object")
    object_ids = {item["object_id"] for item in objects}
    actions = tuple(
        _validate_action(item, object_ids=object_ids)
        for item in _sequence(manifest.get("actions", ()), "actions")
    )
    frames = tuple(
        _validate_frame(item, object_ids=object_ids)
        for item in _sequence(manifest.get("frames"), "frames")
    )
    if not frames:
        raise ValueError("controlled capture manifest requires at least one frame")
    _validate_strictly_increasing_timestamps(frames)
    _validate_frame_action_refs(frames, {action["action_id"] for action in actions})
    _validate_actions_within_frame_range(actions, frames)
    return {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": sample,
        "objects": objects,
        "actions": actions,
        "frames": frames,
    }


def validate_objectstate_controlled_capture_summary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("controlled capture summary must be a dict")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA:
        raise ValueError(f"unsupported controlled capture summary schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_controlled_capture_summary":
        raise ValueError("controlled capture summary kind is unsupported")
    if payload.get("capture_schema") != OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA:
        raise ValueError("controlled capture summary has unsupported capture_schema")
    if int(payload.get("frame_count", 0)) < 1:
        raise ValueError("controlled capture summary requires frames")
    if int(payload.get("object_count", 0)) < 1:
        raise ValueError("controlled capture summary requires objects")
    for key in ("observation_coverage", "ground_truth", "readiness"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"controlled capture summary requires {key}")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("controlled capture summary issues must be a list")
    validate_objectstate_controlled_real_manifest(
        payload.get("controlled_real_manifest_seed")
    )
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("capture_manifest_required")
        or not claim_policy.get("manifest_does_not_prove_model_quality")
        or not claim_policy.get("blocked_seed_rows_are_not_pass_rows")
        or not claim_policy.get("candidate_metrics_required_for_pass_rows")
    ):
        raise ValueError("controlled capture summary must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("reconstructs_gaussians")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError("controlled capture summary cannot claim capture, GT, training, replay, diffusion, or viewer mutation")
    return payload


def _capture_summary_parts(manifest: Mapping[str, Any]) -> dict[str, Any]:
    checked_manifest = validate_objectstate_controlled_capture_manifest(manifest)
    frames = checked_manifest["frames"]
    actions = checked_manifest["actions"]
    object_track_counts = _object_track_counts(frames)
    observation_coverage = _observation_coverage(frames)
    ground_truth = _ground_truth_summary(
        frames,
        actions=actions,
        object_track_counts=object_track_counts,
    )
    readiness = _readiness_summary(
        ground_truth,
        frame_count=len(frames),
        object_track_counts=object_track_counts,
        observation_coverage=observation_coverage,
    )
    return {
        "ground_truth": ground_truth,
        "readiness": readiness,
    }


def _validate_sample(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled capture sample must be a mapping")
    source_kind = str(value.get("source_kind", "controlled_real"))
    if source_kind != "controlled_real":
        raise ValueError("controlled capture sample.source_kind must be controlled_real")
    fps = _number(value.get("fps", 30.0), "sample.fps")
    if fps <= 0:
        raise ValueError("sample.fps must be positive")
    return {
        "sample_id": _required_string(value, "sample_id"),
        "source_kind": source_kind,
        "object_category": _required_string(value, "object_category"),
        "scenario": _required_string(value, "scenario"),
        "fps": fps,
        "capture_device": str(value.get("capture_device", "unknown")),
        "observation_modalities": _string_tuple(
            value.get("observation_modalities", ("rgb",)),
            "observation_modalities",
        ),
        "artifact_refs": _string_tuple(value.get("artifact_refs"), "artifact_refs"),
        "license": _required_string(value, "license"),
    }


def _validate_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled capture objects must be mappings")
    result = {
        "object_id": _required_string(value, "object_id"),
        "category": _required_string(value, "category"),
    }
    if "instance_label" in value:
        result["instance_label"] = _required_string(value, "instance_label")
    if "dimensions_m" in value:
        result["dimensions_m"] = _vector(value["dimensions_m"], "dimensions_m", length=3)
    return result


def _validate_action(value: Any, *, object_ids: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled capture actions must be mappings")
    object_id = _required_string(value, "object_id")
    if object_id not in object_ids:
        raise ValueError(f"controlled capture action references unknown object_id: {object_id}")
    start = _number(value.get("start_timestamp"), "action.start_timestamp")
    end = _number(value.get("end_timestamp"), "action.end_timestamp")
    if end < start:
        raise ValueError("action.end_timestamp must be >= action.start_timestamp")
    result = {
        "action_id": _required_string(value, "action_id"),
        "action_type": _required_string(value, "action_type"),
        "object_id": object_id,
        "start_timestamp": start,
        "end_timestamp": end,
        "actor": str(value.get("actor", "unknown")),
    }
    if "target_object_id" in value:
        target = _required_string(value, "target_object_id")
        if target not in object_ids:
            raise ValueError(
                f"controlled capture action references unknown target_object_id: {target}"
            )
        result["target_object_id"] = target
    if "vector" in value:
        result["vector"] = _vector(value["vector"], "action.vector", length=3)
    return result


def _validate_frame(value: Any, *, object_ids: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled capture frames must be mappings")
    observations = _validate_observation(value.get("observation"))
    frame_objects = tuple(
        _validate_frame_object(item, object_ids=object_ids)
        for item in _sequence(value.get("objects"), "frame.objects")
    )
    seen_ids: set[str] = set()
    for item in frame_objects:
        object_id = item["object_id"]
        if object_id in seen_ids:
            raise ValueError(f"duplicate object_id in frame: {object_id}")
        seen_ids.add(object_id)
    result = {
        "frame_id": _required_string(value, "frame_id"),
        "timestamp": _number(value.get("timestamp"), "frame.timestamp"),
        "observation": observations,
        "objects": frame_objects,
    }
    if "action_id" in value:
        result["action_id"] = _required_string(value, "action_id")
    if "condition" in value:
        result["condition"] = _validate_frame_condition(value["condition"])
    return result


def _validate_observation(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled capture frame.observation must be a mapping")
    rgb = _required_string(value, "rgb")
    result = {"rgb": rgb}
    if "gaussian" in value:
        result["gaussian"] = _required_string(value, "gaussian")
    return result


def _validate_frame_object(value: Any, *, object_ids: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled capture frame objects must be mappings")
    object_id = _required_string(value, "object_id")
    if object_id not in object_ids:
        raise ValueError(f"controlled capture frame references unknown object_id: {object_id}")
    result = {
        "object_id": object_id,
        "visible": bool(value.get("visible", True)),
    }
    if "occlusion_fraction" in value:
        occlusion = _number(value["occlusion_fraction"], "occlusion_fraction")
        if occlusion < 0.0 or occlusion > 1.0:
            raise ValueError("occlusion_fraction must be in [0, 1]")
        result["occlusion_fraction"] = occlusion
    if "pose" in value:
        result["pose"] = _validate_pose(value["pose"])
    return result


def _validate_frame_condition(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled capture frame.condition must be a mapping")
    result: dict[str, Any] = {}
    if "view_id" in value:
        result["view_id"] = _required_string(value, "view_id")
    if "lighting_id" in value:
        result["lighting_id"] = _required_string(value, "lighting_id")
    if "camera_pose" in value:
        result["camera_pose"] = _validate_pose(value["camera_pose"])
    if not result:
        raise ValueError(
            "controlled capture frame.condition must include view_id, "
            "lighting_id, or camera_pose"
        )
    return result


def _validate_pose(value: Any) -> dict[str, list[float]]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled capture pose must be a mapping")
    rotation = _vector(value.get("rotation_xyzw"), "pose.rotation_xyzw", length=4)
    if sum(component * component for component in rotation) <= 0.0:
        raise ValueError("pose.rotation_xyzw must be non-zero")
    return {
        "position": _vector(value.get("position"), "pose.position", length=3),
        "rotation_xyzw": rotation,
    }


def _validate_strictly_increasing_timestamps(frames: Sequence[Mapping[str, Any]]) -> None:
    previous: float | None = None
    for frame in frames:
        timestamp = float(frame["timestamp"])
        if previous is not None and timestamp <= previous:
            raise ValueError("controlled capture frame timestamps must be strictly increasing")
        previous = timestamp


def _validate_frame_action_refs(
    frames: Sequence[Mapping[str, Any]],
    action_ids: set[str],
) -> None:
    for frame in frames:
        action_id = frame.get("action_id")
        if action_id is not None and action_id not in action_ids:
            raise ValueError(f"controlled capture frame references unknown action_id: {action_id}")


def _validate_actions_within_frame_range(
    actions: Sequence[Mapping[str, Any]],
    frames: Sequence[Mapping[str, Any]],
) -> None:
    if not actions:
        return
    start = float(frames[0]["timestamp"])
    end = float(frames[-1]["timestamp"])
    for action in actions:
        if float(action["start_timestamp"]) < start or float(action["end_timestamp"]) > end:
            raise ValueError("controlled capture action timestamps must fall within frame range")


def _object_track_counts(frames: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frames:
        for item in frame["objects"]:
            object_id = str(item["object_id"])
            counts[object_id] = counts.get(object_id, 0) + 1
    return counts


def _observation_coverage(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frame_count = len(frames)
    rgb_frames = sum(1 for frame in frames if frame["observation"].get("rgb"))
    gaussian_frames = sum(1 for frame in frames if frame["observation"].get("gaussian"))
    return {
        "rgb_frames": rgb_frames,
        "gaussian_frames": gaussian_frames,
        "rgb_fraction": rgb_frames / frame_count,
        "gaussian_fraction": gaussian_frames / frame_count,
        "all_frames_have_rgb": rgb_frames == frame_count,
        "all_frames_have_gaussian": gaussian_frames == frame_count,
    }


def _ground_truth_summary(
    frames: Sequence[Mapping[str, Any]],
    *,
    actions: Sequence[Mapping[str, Any]],
    object_track_counts: Mapping[str, int],
) -> dict[str, bool]:
    frame_objects = [item for frame in frames for item in frame["objects"]]
    return {
        "identity": bool(object_track_counts)
        and all(count >= 2 for count in object_track_counts.values()),
        "pose": bool(frame_objects) and all("pose" in item for item in frame_objects),
        "action": bool(actions),
        "timestamp": True,
    }


def _readiness_summary(
    ground_truth: Mapping[str, bool],
    *,
    frame_count: int,
    object_track_counts: Mapping[str, int],
    observation_coverage: Mapping[str, Any],
) -> dict[str, bool]:
    has_multiframe_track = any(count >= 2 for count in object_track_counts.values())
    identity_ready = (
        bool(ground_truth["timestamp"])
        and bool(ground_truth["identity"])
        and frame_count >= 2
        and has_multiframe_track
    )
    prediction_ready = identity_ready and bool(ground_truth["pose"]) and frame_count >= 2
    intervention_ready = prediction_ready and bool(ground_truth["action"])
    return {
        "identity_stage_ready": identity_ready,
        "prediction_stage_ready": prediction_ready,
        "intervention_stage_ready": intervention_ready,
        "real_gaussian_reconstruction_present": bool(
            observation_coverage["all_frames_have_gaussian"]
        ),
    }


def _capture_issues(
    ground_truth: Mapping[str, bool],
    *,
    readiness: Mapping[str, bool],
    observation_coverage: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []
    if not readiness["real_gaussian_reconstruction_present"]:
        issues.append("not all frames reference reconstructed Gaussian evidence")
    if not ground_truth["identity"]:
        issues.append("identity GT requires each declared object to be tracked across frames")
    if not ground_truth["pose"]:
        issues.append("6DoF pose GT is incomplete")
    if not ground_truth["action"]:
        issues.append("action GT is missing")
    if not observation_coverage["all_frames_have_rgb"]:
        issues.append("RGB evidence is incomplete")
    return issues


def _seed_block_reason(ready: bool, ready_reason: str, not_ready_reason: str) -> str:
    return ready_reason if ready else not_ready_reason


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be a non-empty string")
    return result


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    normalized = tuple(str(item) for item in value)
    if not normalized or any(not item for item in normalized):
        raise ValueError(f"{name} must contain non-empty strings")
    return normalized


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _vector(value: Any, name: str, *, length: int) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(value) != length:
        raise ValueError(f"{name} must have length {length}")
    return [_number(item, name) for item in value]


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return float(value)
