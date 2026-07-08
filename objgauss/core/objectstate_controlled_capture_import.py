from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA,
    objectstate_controlled_capture_summary,
    validate_objectstate_controlled_capture_manifest,
    validate_objectstate_controlled_capture_summary,
)

OBJECTSTATE_CONTROLLED_CAPTURE_IMPORT_SCHEMA = (
    "objgauss-objectstate-controlled-capture-import-v1"
)


def objectstate_controlled_capture_manifest_from_bundle(
    root: str | Path,
    *,
    sample_json: str | Path = "sample.json",
    objects_csv: str | Path = "objects.csv",
    frames_csv: str | Path = "frames.csv",
    annotations_csv: str | Path = "annotations.csv",
    actions_csv: str | Path | None = "actions.csv",
) -> dict[str, Any]:
    bundle_root = Path(root)
    sample_path = _resolve_bundle_path(bundle_root, sample_json)
    objects_path = _resolve_bundle_path(bundle_root, objects_csv)
    frames_path = _resolve_bundle_path(bundle_root, frames_csv)
    annotations_path = _resolve_bundle_path(bundle_root, annotations_csv)
    action_path = (
        None if actions_csv is None else _resolve_bundle_path(bundle_root, actions_csv)
    )
    sample = _read_sample_json(sample_path)
    objects = _objects_from_csv(objects_path)
    actions = _actions_from_csv(action_path) if action_path is not None and action_path.exists() else []
    annotations = _annotations_by_frame(annotations_path)
    frames = _frames_from_csv(frames_path, annotations)
    manifest = {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": sample,
        "objects": objects,
        "actions": actions,
        "frames": frames,
    }
    return validate_objectstate_controlled_capture_manifest(manifest)


def objectstate_controlled_capture_import_summary(
    root: str | Path,
    *,
    sample_json: str | Path = "sample.json",
    objects_csv: str | Path = "objects.csv",
    frames_csv: str | Path = "frames.csv",
    annotations_csv: str | Path = "annotations.csv",
    actions_csv: str | Path | None = "actions.csv",
) -> dict[str, Any]:
    bundle_root = Path(root)
    manifest = objectstate_controlled_capture_manifest_from_bundle(
        bundle_root,
        sample_json=sample_json,
        objects_csv=objects_csv,
        frames_csv=frames_csv,
        annotations_csv=annotations_csv,
        actions_csv=actions_csv,
    )
    capture_summary = objectstate_controlled_capture_summary(manifest)
    input_files = {
        "sample_json": str(_resolve_bundle_path(bundle_root, sample_json)),
        "objects_csv": str(_resolve_bundle_path(bundle_root, objects_csv)),
        "frames_csv": str(_resolve_bundle_path(bundle_root, frames_csv)),
        "annotations_csv": str(_resolve_bundle_path(bundle_root, annotations_csv)),
        "actions_csv": (
            None if actions_csv is None else str(_resolve_bundle_path(bundle_root, actions_csv))
        ),
    }
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_IMPORT_SCHEMA,
        "kind": "objectstate_controlled_capture_import",
        "capture_schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "capture_summary_schema": OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA,
        "root": str(bundle_root),
        "input_files": input_files,
        "manifest": manifest,
        "capture_summary": capture_summary,
        "row_counts": {
            "objects": len(manifest["objects"]),
            "frames": len(manifest["frames"]),
            "annotations": sum(len(frame["objects"]) for frame in manifest["frames"]),
            "actions": len(manifest["actions"]),
        },
        "claim_policy": {
            "imports_existing_capture_files": True,
            "validates_manifest_contract": True,
            "does_not_create_ground_truth": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_score_candidate_model": True,
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
    return validate_objectstate_controlled_capture_import_summary(payload)


def validate_objectstate_controlled_capture_import_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled capture import summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_CAPTURE_IMPORT_SCHEMA:
        raise ValueError(
            f"unsupported controlled capture import schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_capture_import":
        raise ValueError("controlled capture import summary kind is unsupported")
    if payload.get("capture_schema") != OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA:
        raise ValueError("controlled capture import summary has unsupported capture_schema")
    if payload.get("capture_summary_schema") != OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA:
        raise ValueError(
            "controlled capture import summary has unsupported capture_summary_schema"
        )
    if not isinstance(payload.get("root"), str) or not payload["root"]:
        raise ValueError("controlled capture import summary requires root")
    input_files = payload.get("input_files")
    if not isinstance(input_files, Mapping):
        raise ValueError("controlled capture import summary requires input_files")
    manifest = validate_objectstate_controlled_capture_manifest(payload.get("manifest"))
    capture_summary = validate_objectstate_controlled_capture_summary(
        payload.get("capture_summary")
    )
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("controlled capture import summary requires row_counts")
    expected_counts = {
        "objects": len(manifest["objects"]),
        "frames": len(manifest["frames"]),
        "annotations": sum(len(frame["objects"]) for frame in manifest["frames"]),
        "actions": len(manifest["actions"]),
    }
    if dict(row_counts) != expected_counts:
        raise ValueError("controlled capture import row_counts must match manifest")
    if capture_summary["sample"]["sample_id"] != manifest["sample"]["sample_id"]:
        raise ValueError("controlled capture import summary sample mismatch")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("imports_existing_capture_files")
        or not claim_policy.get("validates_manifest_contract")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_score_candidate_model")
    ):
        raise ValueError("controlled capture import summary must preserve claim policy")
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
        raise ValueError(
            "controlled capture import cannot claim capture, GT creation, "
            "reconstruction, training, replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def _read_sample_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("controlled capture sample JSON must be an object")
    return dict(payload)


def _objects_from_csv(path: Path) -> list[dict[str, Any]]:
    objects = []
    for row in _read_csv(path):
        item: dict[str, Any] = {
            "object_id": _required(row, "object_id"),
            "category": _required(row, "category"),
        }
        instance_label = _optional(row, "instance_label")
        if instance_label is not None:
            item["instance_label"] = instance_label
        dimensions = _optional_vector(
            row,
            ("dimension_x_m", "dimension_y_m", "dimension_z_m"),
            "dimensions_m",
        )
        if dimensions is not None:
            item["dimensions_m"] = dimensions
        objects.append(item)
    if not objects:
        raise ValueError("controlled capture objects.csv requires at least one object")
    return objects


def _actions_from_csv(path: Path) -> list[dict[str, Any]]:
    actions = []
    for row in _read_csv(path):
        item: dict[str, Any] = {
            "action_id": _required(row, "action_id"),
            "action_type": _required(row, "action_type"),
            "object_id": _required(row, "object_id"),
            "start_timestamp": _float(row, "start_timestamp"),
            "end_timestamp": _float(row, "end_timestamp"),
        }
        actor = _optional(row, "actor")
        if actor is not None:
            item["actor"] = actor
        target = _optional(row, "target_object_id")
        if target is not None:
            item["target_object_id"] = target
        vector = _optional_vector(row, ("vector_x", "vector_y", "vector_z"), "action.vector")
        if vector is not None:
            item["vector"] = vector
        actions.append(item)
    return actions


def _annotations_by_frame(path: Path) -> dict[str, list[dict[str, Any]]]:
    annotations: dict[str, list[dict[str, Any]]] = {}
    for row in _read_csv(path):
        frame_id = _required(row, "frame_id")
        item: dict[str, Any] = {
            "object_id": _required(row, "object_id"),
        }
        visible = _optional_bool(row, "visible")
        if visible is not None:
            item["visible"] = visible
        occlusion = _optional_float(row, "occlusion_fraction")
        if occlusion is not None:
            item["occlusion_fraction"] = occlusion
        pose = _pose_from_columns(
            row,
            position=("x", "y", "z"),
            rotation=("qx", "qy", "qz", "qw"),
            name="annotation pose",
        )
        if pose is not None:
            item["pose"] = pose
        annotations.setdefault(frame_id, []).append(item)
    if not annotations:
        raise ValueError("controlled capture annotations.csv requires at least one row")
    return annotations


def _frames_from_csv(
    path: Path,
    annotations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    frames = []
    seen_frames: set[str] = set()
    for row in _read_csv(path):
        frame_id = _required(row, "frame_id")
        if frame_id in seen_frames:
            raise ValueError(f"controlled capture frames.csv has duplicate frame_id: {frame_id}")
        seen_frames.add(frame_id)
        frame_annotations = annotations_by_frame.get(frame_id)
        if not frame_annotations:
            raise ValueError(
                f"controlled capture frames.csv references frame without annotations: {frame_id}"
            )
        observation = {"rgb": _required(row, "rgb")}
        gaussian = _optional(row, "gaussian")
        if gaussian is not None:
            observation["gaussian"] = gaussian
        frame: dict[str, Any] = {
            "frame_id": frame_id,
            "timestamp": _float(row, "timestamp"),
            "observation": observation,
            "objects": list(frame_annotations),
        }
        action_id = _optional(row, "action_id")
        if action_id is not None:
            frame["action_id"] = action_id
        condition = _frame_condition(row)
        if condition is not None:
            frame["condition"] = condition
        frames.append(frame)
    extra_annotations = sorted(set(annotations_by_frame) - seen_frames)
    if extra_annotations:
        raise ValueError(
            "controlled capture annotations.csv references unknown frame_id: "
            + ", ".join(extra_annotations)
        )
    if not frames:
        raise ValueError("controlled capture frames.csv requires at least one frame")
    return frames


def _frame_condition(row: Mapping[str, str]) -> dict[str, Any] | None:
    condition: dict[str, Any] = {}
    view_id = _optional(row, "view_id")
    if view_id is not None:
        condition["view_id"] = view_id
    lighting_id = _optional(row, "lighting_id")
    if lighting_id is not None:
        condition["lighting_id"] = lighting_id
    camera_pose = _pose_from_columns(
        row,
        position=("camera_x", "camera_y", "camera_z"),
        rotation=("camera_qx", "camera_qy", "camera_qz", "camera_qw"),
        name="frame camera pose",
    )
    if camera_pose is not None:
        condition["camera_pose"] = camera_pose
    return condition or None


def _pose_from_columns(
    row: Mapping[str, str],
    *,
    position: tuple[str, str, str],
    rotation: tuple[str, str, str, str],
    name: str,
) -> dict[str, list[float]] | None:
    keys = position + rotation
    values = [_optional(row, key) for key in keys]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"{name} requires all columns: {', '.join(keys)}")
    return {
        "position": [_float(row, key) for key in position],
        "rotation_xyzw": [_float(row, key) for key in rotation],
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"controlled capture CSV has no header: {path}")
        return [
            {str(key): "" if value is None else str(value).strip() for key, value in row.items()}
            for row in reader
        ]


def _resolve_bundle_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _required(row: Mapping[str, str], key: str) -> str:
    value = _optional(row, key)
    if value is None:
        raise ValueError(f"controlled capture CSV requires {key}")
    return value


def _optional(row: Mapping[str, str], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _float(row: Mapping[str, str], key: str) -> float:
    value = _required(row, key)
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"controlled capture CSV {key} must be numeric") from exc


def _optional_float(row: Mapping[str, str], key: str) -> float | None:
    value = _optional(row, key)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"controlled capture CSV {key} must be numeric") from exc


def _optional_bool(row: Mapping[str, str], key: str) -> bool | None:
    value = _optional(row, key)
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"controlled capture CSV {key} must be boolean")


def _optional_vector(
    row: Mapping[str, str],
    keys: tuple[str, ...],
    name: str,
) -> list[float] | None:
    values = [_optional(row, key) for key in keys]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"{name} requires all columns: {', '.join(keys)}")
    return [_float(row, key) for key in keys]
