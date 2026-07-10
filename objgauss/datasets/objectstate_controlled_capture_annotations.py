from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.datasets.objectstate_controlled_capture_template import (
    ANNOTATIONS_CSV_HEADER,
)

OBJECTSTATE_CONTROLLED_CAPTURE_ANNOTATION_TEMPLATE_SCHEMA = (
    "objgauss-objectstate-controlled-capture-annotation-template-v1"
)
OBJECTSTATE_CONTROLLED_CAPTURE_ANNOTATION_FINALIZE_SCHEMA = (
    "objgauss-objectstate-controlled-capture-annotation-finalize-v1"
)

_TODO_VISIBLE = "TODO_VISIBLE_TRUE_FALSE"
_TODO_OCCLUSION = "TODO_OCCLUSION_0_1"
_TODO_X = "TODO_X_METERS"
_TODO_Y = "TODO_Y_METERS"
_TODO_Z = "TODO_Z_METERS"
_TODO_QX = "TODO_QX"
_TODO_QY = "TODO_QY"
_TODO_QZ = "TODO_QZ"
_TODO_QW = "TODO_QW"
_TODO_PREFIX = "TODO"


def write_objectstate_controlled_capture_annotation_template(
    root: str | Path,
    *,
    frames_csv: str | Path = "frames.csv",
    objects_csv: str | Path = "objects.csv",
    output: str | Path = "annotations.template.csv",
    force: bool = False,
) -> dict[str, Any]:
    bundle_root = Path(root)
    frames_path = _resolve_bundle_path(bundle_root, frames_csv)
    objects_path = _resolve_bundle_path(bundle_root, objects_csv)
    output_path = _resolve_bundle_path(bundle_root, output)
    frames = _read_rows(frames_path, required=("frame_id",))
    objects = _read_rows(objects_path, required=("object_id",))
    frame_ids = [_required(row, "frame_id", "frames.csv") for row in frames]
    object_ids = [_required(row, "object_id", "objects.csv") for row in objects]
    rows = [
        _template_row(frame_id, object_id)
        for frame_id in frame_ids
        for object_id in object_ids
    ]
    issues = []
    if not frame_ids:
        issues.append("frames.csv requires at least one frame row")
    if not object_ids:
        issues.append("objects.csv requires at least one object row")
    if _duplicates(frame_ids):
        issues.append("frames.csv has duplicate frame_id values")
    if _duplicates(object_ids):
        issues.append("objects.csv has duplicate object_id values")
    output_state = _output_state(output_path)
    output_writable = bool(output_state["can_write"] or force)
    if not output_writable:
        issues.append("annotation template already contains rows; pass force to overwrite")
    ready = bool(frame_ids and object_ids and output_writable and not issues)
    wrote = False
    if ready:
        _write_rows(output_path, rows)
        wrote = True
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_ANNOTATION_TEMPLATE_SCHEMA,
        "kind": "objectstate_controlled_capture_annotation_template",
        "status": (
            "objectstate_controlled_capture_annotation_template_ready"
            if wrote
            else "objectstate_controlled_capture_annotation_template_blocked"
        ),
        "root": str(bundle_root),
        "paths": {
            "frames_csv": str(frames_path),
            "objects_csv": str(objects_path),
            "annotation_template_csv": str(output_path),
        },
        "row_counts": {
            "frame_count": len(frame_ids),
            "object_count": len(object_ids),
            "template_annotation_rows": len(rows),
        },
        "output": {**output_state, "wrote_template": wrote},
        "readiness": {
            "frames_present": bool(frame_ids),
            "objects_present": bool(object_ids),
            "output_writable": output_writable,
            "annotation_template_ready": wrote,
        },
        "issues": issues,
        "next_actions": _template_next_actions(wrote),
        "template_policy": {
            "template_status": "draft_not_valid_for_import",
            "todo_values_required": True,
            "finalizer_required": True,
        },
        "claim_policy": {
            "writes_draft_annotation_template": True,
            "draft_not_valid_for_import": True,
            "requires_human_or_external_pose_labels": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_pose": True,
            "does_not_write_annotations_csv": True,
            "does_not_run_handoff": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "infers_pose": False,
            "writes_annotations_csv": False,
            "creates_action_rows": False,
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
    return validate_objectstate_controlled_capture_annotation_template_summary(payload)


def finalize_objectstate_controlled_capture_annotations(
    root: str | Path,
    *,
    source: str | Path = "annotations.template.csv",
    frames_csv: str | Path = "frames.csv",
    objects_csv: str | Path = "objects.csv",
    output: str | Path = "annotations.csv",
    require_all_frame_object_pairs: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    bundle_root = Path(root)
    source_path = _resolve_bundle_path(bundle_root, source)
    frames_path = _resolve_bundle_path(bundle_root, frames_csv)
    objects_path = _resolve_bundle_path(bundle_root, objects_csv)
    output_path = _resolve_bundle_path(bundle_root, output)
    frame_ids = [
        _required(row, "frame_id", "frames.csv")
        for row in _read_rows(frames_path, required=("frame_id",))
    ]
    object_ids = [
        _required(row, "object_id", "objects.csv")
        for row in _read_rows(objects_path, required=("object_id",))
    ]
    source_rows = _read_rows(source_path, required=ANNOTATIONS_CSV_HEADER)
    issues, rows = _finalize_rows(
        source_rows,
        frame_ids=frame_ids,
        object_ids=object_ids,
        require_all_frame_object_pairs=require_all_frame_object_pairs,
    )
    output_state = _output_state(output_path)
    output_writable = bool(output_state["can_write"] or force)
    if not output_writable:
        issues.append("annotations.csv already contains rows; pass force to overwrite")
    ready = bool(rows and output_writable and not issues)
    wrote = False
    if ready:
        _write_rows(output_path, rows)
        wrote = True
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_ANNOTATION_FINALIZE_SCHEMA,
        "kind": "objectstate_controlled_capture_annotation_finalize",
        "status": (
            "objectstate_controlled_capture_annotation_finalize_ready"
            if wrote
            else "objectstate_controlled_capture_annotation_finalize_blocked"
        ),
        "root": str(bundle_root),
        "paths": {
            "source_annotation_csv": str(source_path),
            "frames_csv": str(frames_path),
            "objects_csv": str(objects_path),
            "annotations_csv": str(output_path),
        },
        "requirements": {
            "all_frame_object_pairs_required": bool(require_all_frame_object_pairs),
            "todo_values_rejected": True,
            "pose_required": True,
            "visible_required": True,
            "occlusion_required": True,
        },
        "row_counts": {
            "frame_count": len(frame_ids),
            "object_count": len(object_ids),
            "source_annotation_rows": len(source_rows),
            "finalized_annotation_rows": len(rows),
        },
        "output": {**output_state, "wrote_annotations_csv": wrote},
        "readiness": {
            "source_rows_present": bool(source_rows),
            "known_frames_ready": bool(frame_ids),
            "known_objects_ready": bool(object_ids),
            "output_writable": output_writable,
            "annotations_csv_written": wrote,
            "controlled_capture_annotations_ready": wrote,
        },
        "issues": issues,
        "next_actions": _finalize_next_actions(wrote),
        "claim_policy": {
            "converts_human_or_external_pose_labels": True,
            "source_annotations_required": True,
            "rejects_todo_values": True,
            "validates_frame_object_binding": True,
            "validates_6d_pose_columns": True,
            "does_not_capture_video": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_pose": True,
            "does_not_create_action_rows": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_run_handoff": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "infers_pose": False,
            "creates_action_rows": False,
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
    return validate_objectstate_controlled_capture_annotation_finalize_summary(payload)


def validate_objectstate_controlled_capture_annotation_template_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled capture annotation template summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_CAPTURE_ANNOTATION_TEMPLATE_SCHEMA:
        raise ValueError(
            "unsupported controlled capture annotation template schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_capture_annotation_template":
        raise ValueError("controlled capture annotation template kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_capture_annotation_template_ready",
        "objectstate_controlled_capture_annotation_template_blocked",
    }:
        raise ValueError("controlled capture annotation template status is unsupported")
    _validate_common_summary(payload, ready_key="annotation_template_ready")
    policy = payload.get("template_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("controlled capture annotation template requires policy")
    if policy.get("template_status") != "draft_not_valid_for_import":
        raise ValueError("controlled capture annotation template must remain draft")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("writes_draft_annotation_template")
        or not claim_policy.get("draft_not_valid_for_import")
        or not claim_policy.get("requires_human_or_external_pose_labels")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_infer_pose")
        or not claim_policy.get("does_not_write_annotations_csv")
        or not claim_policy.get("does_not_run_handoff")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled capture annotation template must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "controlled capture annotation template cannot claim capture, GT, "
            "pose inference, annotations.csv output, action rows, reconstruction, "
            "handoff, eval, training, public samples, replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def validate_objectstate_controlled_capture_annotation_finalize_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled capture annotation finalize summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_CAPTURE_ANNOTATION_FINALIZE_SCHEMA:
        raise ValueError(
            "unsupported controlled capture annotation finalize schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_capture_annotation_finalize":
        raise ValueError("controlled capture annotation finalize kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_capture_annotation_finalize_ready",
        "objectstate_controlled_capture_annotation_finalize_blocked",
    }:
        raise ValueError("controlled capture annotation finalize status is unsupported")
    _validate_common_summary(payload, ready_key="controlled_capture_annotations_ready")
    requirements = payload.get("requirements")
    if not isinstance(requirements, Mapping):
        raise ValueError("controlled capture annotation finalize requires requirements")
    for key in (
        "all_frame_object_pairs_required",
        "todo_values_rejected",
        "pose_required",
        "visible_required",
        "occlusion_required",
    ):
        if not isinstance(requirements.get(key), bool):
            raise ValueError(f"controlled capture annotation finalize requires bool {key}")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("converts_human_or_external_pose_labels")
        or not claim_policy.get("source_annotations_required")
        or not claim_policy.get("rejects_todo_values")
        or not claim_policy.get("validates_frame_object_binding")
        or not claim_policy.get("validates_6d_pose_columns")
        or not claim_policy.get("does_not_capture_video")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_infer_pose")
        or not claim_policy.get("does_not_create_action_rows")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_run_handoff")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled capture annotation finalize must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "controlled capture annotation finalize cannot claim capture, GT, "
            "pose inference, action rows, reconstruction, handoff, eval, "
            "training, public samples, replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def _validate_common_summary(payload: Mapping[str, Any], *, ready_key: str) -> None:
    for key in ("root",):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"controlled capture annotation summary requires {key}")
    for section in ("paths", "row_counts", "output", "readiness"):
        if not isinstance(payload.get(section), Mapping):
            raise ValueError(f"controlled capture annotation summary requires {section}")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("controlled capture annotation summary requires issues")
    if not isinstance(payload.get("next_actions"), list):
        raise ValueError("controlled capture annotation summary requires next_actions")
    readiness = payload["readiness"]
    if not isinstance(readiness.get(ready_key), bool):
        raise ValueError(f"controlled capture annotation readiness requires {ready_key}")
    expected_status = (
        payload["status"].rsplit("_", 1)[0] + "_ready"
        if readiness[ready_key]
        else payload["status"].rsplit("_", 1)[0] + "_blocked"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled capture annotation status mismatch")


def _template_row(frame_id: str, object_id: str) -> dict[str, str]:
    return {
        "frame_id": frame_id,
        "object_id": object_id,
        "visible": _TODO_VISIBLE,
        "occlusion_fraction": _TODO_OCCLUSION,
        "x": _TODO_X,
        "y": _TODO_Y,
        "z": _TODO_Z,
        "qx": _TODO_QX,
        "qy": _TODO_QY,
        "qz": _TODO_QZ,
        "qw": _TODO_QW,
    }


def _finalize_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    frame_ids: Sequence[str],
    object_ids: Sequence[str],
    require_all_frame_object_pairs: bool,
) -> tuple[list[str], list[dict[str, str]]]:
    issues: list[str] = []
    known_frames = set(frame_ids)
    known_objects = set(object_ids)
    seen: set[tuple[str, str]] = set()
    finalized: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        frame_id = _required(row, "frame_id", f"annotation row {index}")
        object_id = _required(row, "object_id", f"annotation row {index}")
        key = (frame_id, object_id)
        if key in seen:
            issues.append(f"duplicate annotation row: {frame_id}:{object_id}")
            continue
        seen.add(key)
        if frame_id not in known_frames:
            issues.append(f"annotation references unknown frame_id: {frame_id}")
        if object_id not in known_objects:
            issues.append(f"annotation references unknown object_id: {object_id}")
        finalized_row, row_issues = _finalized_row(row, label=f"{frame_id}:{object_id}")
        issues.extend(row_issues)
        finalized.append(finalized_row)
    if require_all_frame_object_pairs:
        expected = {(frame_id, object_id) for frame_id in frame_ids for object_id in object_ids}
        missing = sorted(expected - seen)
        if missing:
            issues.append(
                "missing frame/object annotation rows: "
                + ", ".join(f"{frame}:{obj}" for frame, obj in missing)
            )
    if not finalized:
        issues.append("annotation source requires at least one row")
    return _dedupe(issues), finalized


def _finalized_row(row: Mapping[str, str], *, label: str) -> tuple[dict[str, str], list[str]]:
    issues: list[str] = []
    result: dict[str, str] = {
        "frame_id": str(row.get("frame_id", "")).strip(),
        "object_id": str(row.get("object_id", "")).strip(),
    }
    visible = _required(row, "visible", label)
    if _is_todo(visible):
        issues.append(f"{label} visible is still TODO")
    elif visible.lower() not in {"true", "false", "1", "0", "yes", "no", "y", "n"}:
        issues.append(f"{label} visible must be boolean")
    result["visible"] = visible
    occlusion = _required(row, "occlusion_fraction", label)
    if _is_todo(occlusion):
        issues.append(f"{label} occlusion_fraction is still TODO")
    else:
        value = _float_value(occlusion, f"{label} occlusion_fraction", issues)
        if value is not None and not 0.0 <= value <= 1.0:
            issues.append(f"{label} occlusion_fraction must be between 0 and 1")
    result["occlusion_fraction"] = occlusion
    for key in ("x", "y", "z", "qx", "qy", "qz", "qw"):
        value = _required(row, key, label)
        if _is_todo(value):
            issues.append(f"{label} {key} is still TODO")
        else:
            _float_value(value, f"{label} {key}", issues)
        result[key] = value
    return result, issues


def _read_rows(path: Path, *, required: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"controlled capture annotation CSV has no header: {path}")
        missing = [key for key in required if key not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"controlled capture annotation CSV {path} missing columns: "
                + ", ".join(missing)
            )
        return [
            {str(key): "" if value is None else str(value).strip() for key, value in row.items()}
            for row in reader
        ]


def _write_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ANNOTATIONS_CSV_HEADER))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ANNOTATIONS_CSV_HEADER})


def _output_state(path: Path) -> dict[str, Any]:
    exists = path.exists()
    is_file = exists and path.is_file()
    existing_row_count = 0
    if is_file:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is not None:
                existing_row_count = sum(1 for _ in reader)
    return {
        "exists_before": bool(exists),
        "is_file_before": bool(is_file),
        "existing_row_count": existing_row_count,
        "can_write": (not exists) or (is_file and existing_row_count == 0),
    }


def _template_next_actions(wrote: bool) -> list[str]:
    if wrote:
        return [
            "fill annotations.template.csv with measured visible, occlusion and 6DoF pose values",
            "run finalize-controlled-capture-annotations to write annotations.csv",
        ]
    return ["populate frames.csv and objects.csv, then rerun annotation template authoring"]


def _finalize_next_actions(wrote: bool) -> list[str]:
    if wrote:
        return [
            "fill actions.csv for intervention segments if needed",
            "rerun audit-controlled-capture-bundle-readiness",
        ]
    return ["replace TODO / blank annotation fields with measured 6DoF GT values"]


def _resolve_bundle_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _required(row: Mapping[str, str], key: str, label: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"{label} requires {key}")
    return value


def _float_value(value: str, label: str, issues: list[str]) -> float | None:
    try:
        return float(value)
    except ValueError:
        issues.append(f"{label} must be numeric")
        return None


def _is_todo(value: str) -> bool:
    return value.strip().upper().startswith(_TODO_PREFIX)


def _duplicates(values: Sequence[str]) -> list[str]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(value for value, count in counts.items() if count > 1)


def _dedupe(values: Sequence[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result
