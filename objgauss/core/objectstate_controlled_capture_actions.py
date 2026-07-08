from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_capture_template import (
    ACTIONS_CSV_HEADER,
)

OBJECTSTATE_CONTROLLED_CAPTURE_ACTION_TEMPLATE_SCHEMA = (
    "objgauss-objectstate-controlled-capture-action-template-v1"
)
OBJECTSTATE_CONTROLLED_CAPTURE_ACTION_FINALIZE_SCHEMA = (
    "objgauss-objectstate-controlled-capture-action-finalize-v1"
)

_TODO_PREFIX = "TODO"
_TODO_ACTION_ID = "TODO_ACTION_ID"
_TODO_ACTION_TYPE = "TODO_ACTION_TYPE"
_TODO_START = "TODO_START_TIMESTAMP"
_TODO_END = "TODO_END_TIMESTAMP"
_TODO_VECTOR_X = "TODO_VECTOR_X_METERS"
_TODO_VECTOR_Y = "TODO_VECTOR_Y_METERS"
_TODO_VECTOR_Z = "TODO_VECTOR_Z_METERS"


def write_objectstate_controlled_capture_action_template(
    root: str | Path,
    *,
    frames_csv: str | Path = "frames.csv",
    objects_csv: str | Path = "objects.csv",
    output: str | Path = "actions.template.csv",
    force: bool = False,
) -> dict[str, Any]:
    bundle_root = Path(root)
    frames_path = _resolve_bundle_path(bundle_root, frames_csv)
    objects_path = _resolve_bundle_path(bundle_root, objects_csv)
    output_path = _resolve_bundle_path(bundle_root, output)
    frames = _read_rows(frames_path, required=("frame_id", "timestamp"))
    objects = _read_rows(objects_path, required=("object_id",))
    frame_ids = [_required(row, "frame_id", "frames.csv") for row in frames]
    object_ids = [_required(row, "object_id", "objects.csv") for row in objects]
    frame_timestamps, timestamp_issues = _frame_timestamps(frames)
    issues = list(timestamp_issues)
    if not frame_ids:
        issues.append("frames.csv requires at least one frame row")
    if not object_ids:
        issues.append("objects.csv requires at least one object row")
    if _duplicates(frame_ids):
        issues.append("frames.csv has duplicate frame_id values")
    if _duplicates(object_ids):
        issues.append("objects.csv has duplicate object_id values")
    rows = [_template_row(object_id) for object_id in object_ids]
    output_state = _output_state(output_path)
    output_writable = bool(output_state["can_write"] or force)
    if not output_writable:
        issues.append("action template already contains rows; pass force to overwrite")
    ready = bool(frame_ids and object_ids and frame_timestamps and output_writable and not issues)
    wrote = False
    if ready:
        _write_rows(output_path, rows)
        wrote = True
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_ACTION_TEMPLATE_SCHEMA,
        "kind": "objectstate_controlled_capture_action_template",
        "status": (
            "objectstate_controlled_capture_action_template_ready"
            if wrote
            else "objectstate_controlled_capture_action_template_blocked"
        ),
        "root": str(bundle_root),
        "paths": {
            "frames_csv": str(frames_path),
            "objects_csv": str(objects_path),
            "action_template_csv": str(output_path),
        },
        "row_counts": {
            "frame_count": len(frame_ids),
            "object_count": len(object_ids),
            "template_action_rows": len(rows),
        },
        "frame_time_range": _time_range(frame_timestamps),
        "output": {**output_state, "wrote_template": wrote},
        "readiness": {
            "frames_present": bool(frame_ids),
            "objects_present": bool(object_ids),
            "frame_timestamps_ready": bool(frame_timestamps) and not timestamp_issues,
            "output_writable": output_writable,
            "action_template_ready": wrote,
        },
        "issues": _dedupe(issues),
        "next_actions": _template_next_actions(wrote),
        "template_policy": {
            "template_status": "draft_not_valid_for_import",
            "todo_values_required": True,
            "finalizer_required": True,
        },
        "claim_policy": {
            "writes_draft_action_template": True,
            "draft_not_valid_for_import": True,
            "requires_human_or_external_action_labels": True,
            "does_not_create_ground_truth": True,
            "does_not_write_actions_csv": True,
            "does_not_run_handoff": True,
            "does_not_claim_intervention_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "creates_annotation_rows": False,
            "writes_actions_csv": False,
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
    return validate_objectstate_controlled_capture_action_template_summary(payload)


def finalize_objectstate_controlled_capture_actions(
    root: str | Path,
    *,
    source: str | Path = "actions.template.csv",
    frames_csv: str | Path = "frames.csv",
    objects_csv: str | Path = "objects.csv",
    output: str | Path = "actions.csv",
    require_nonzero_vector: bool = True,
    require_frame_interval: bool = True,
    require_frame_action_refs: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    bundle_root = Path(root)
    source_path = _resolve_bundle_path(bundle_root, source)
    frames_path = _resolve_bundle_path(bundle_root, frames_csv)
    objects_path = _resolve_bundle_path(bundle_root, objects_csv)
    output_path = _resolve_bundle_path(bundle_root, output)
    frames = _read_rows(frames_path, required=("frame_id", "timestamp"))
    objects = _read_rows(objects_path, required=("object_id",))
    object_ids = [_required(row, "object_id", "objects.csv") for row in objects]
    source_rows = _read_rows(source_path, required=ACTIONS_CSV_HEADER)
    issues, rows, coverage = _finalize_rows(
        source_rows,
        frames=frames,
        object_ids=object_ids,
        require_nonzero_vector=require_nonzero_vector,
        require_frame_interval=require_frame_interval,
        require_frame_action_refs=require_frame_action_refs,
    )
    if _duplicates(object_ids):
        issues.append("objects.csv has duplicate object_id values")
    output_state = _output_state(output_path)
    output_writable = bool(output_state["can_write"] or force)
    if not output_writable:
        issues.append("actions.csv already contains rows; pass force to overwrite")
    ready = bool(rows and output_writable and not issues)
    wrote = False
    if ready:
        _write_rows(output_path, rows)
        wrote = True
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_ACTION_FINALIZE_SCHEMA,
        "kind": "objectstate_controlled_capture_action_finalize",
        "status": (
            "objectstate_controlled_capture_action_finalize_ready"
            if wrote
            else "objectstate_controlled_capture_action_finalize_blocked"
        ),
        "root": str(bundle_root),
        "paths": {
            "source_action_csv": str(source_path),
            "frames_csv": str(frames_path),
            "objects_csv": str(objects_path),
            "actions_csv": str(output_path),
        },
        "requirements": {
            "todo_values_rejected": True,
            "known_object_required": True,
            "valid_time_interval_required": True,
            "frame_interval_required": bool(require_frame_interval),
            "frame_action_refs_required": bool(require_frame_action_refs),
            "nonzero_vector_required": bool(require_nonzero_vector),
        },
        "row_counts": {
            "frame_count": len(frames),
            "object_count": len(object_ids),
            "source_action_rows": len(source_rows),
            "finalized_action_rows": len(rows),
            "covered_frame_count": int(coverage["covered_frame_count"]),
            "frame_action_ref_count": int(coverage["frame_action_ref_count"]),
        },
        "frame_time_range": coverage["frame_time_range"],
        "output": {**output_state, "wrote_actions_csv": wrote},
        "readiness": {
            "source_rows_present": bool(source_rows),
            "known_objects_ready": bool(object_ids) and not _duplicates(object_ids),
            "output_writable": output_writable,
            "actions_csv_written": wrote,
            "controlled_capture_actions_ready": wrote,
        },
        "issues": _dedupe(issues),
        "next_actions": _finalize_next_actions(wrote),
        "claim_policy": {
            "converts_human_or_external_action_labels": True,
            "source_actions_required": True,
            "rejects_todo_values": True,
            "validates_object_binding": True,
            "validates_action_time_interval": True,
            "validates_action_vector": True,
            "writes_actions_csv": bool(wrote),
            "does_not_capture_video": True,
            "does_not_create_ground_truth": True,
            "does_not_create_annotation_rows": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_run_handoff": True,
            "does_not_claim_intervention_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "creates_annotation_rows": False,
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
    return validate_objectstate_controlled_capture_action_finalize_summary(payload)


def validate_objectstate_controlled_capture_action_template_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled capture action template summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_CAPTURE_ACTION_TEMPLATE_SCHEMA:
        raise ValueError(
            "unsupported controlled capture action template schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_capture_action_template":
        raise ValueError("controlled capture action template kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_capture_action_template_ready",
        "objectstate_controlled_capture_action_template_blocked",
    }:
        raise ValueError("controlled capture action template status is unsupported")
    _validate_common_summary(payload, ready_key="action_template_ready")
    policy = payload.get("template_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("controlled capture action template requires policy")
    if policy.get("template_status") != "draft_not_valid_for_import":
        raise ValueError("controlled capture action template must remain draft")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("writes_draft_action_template")
        or not claim_policy.get("draft_not_valid_for_import")
        or not claim_policy.get("requires_human_or_external_action_labels")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_write_actions_csv")
        or not claim_policy.get("does_not_run_handoff")
        or not claim_policy.get("does_not_claim_intervention_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled capture action template must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "controlled capture action template cannot claim capture, GT, "
            "annotations, actions.csv output, reconstruction, handoff, eval, "
            "training, public samples, replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def validate_objectstate_controlled_capture_action_finalize_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled capture action finalize summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_CAPTURE_ACTION_FINALIZE_SCHEMA:
        raise ValueError(
            "unsupported controlled capture action finalize schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_capture_action_finalize":
        raise ValueError("controlled capture action finalize kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_capture_action_finalize_ready",
        "objectstate_controlled_capture_action_finalize_blocked",
    }:
        raise ValueError("controlled capture action finalize status is unsupported")
    _validate_common_summary(payload, ready_key="controlled_capture_actions_ready")
    requirements = payload.get("requirements")
    if not isinstance(requirements, Mapping):
        raise ValueError("controlled capture action finalize requires requirements")
    for key in (
        "todo_values_rejected",
        "known_object_required",
        "valid_time_interval_required",
        "frame_interval_required",
        "frame_action_refs_required",
        "nonzero_vector_required",
    ):
        if not isinstance(requirements.get(key), bool):
            raise ValueError(f"controlled capture action finalize requires bool {key}")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("converts_human_or_external_action_labels")
        or not claim_policy.get("source_actions_required")
        or not claim_policy.get("rejects_todo_values")
        or not claim_policy.get("validates_object_binding")
        or not claim_policy.get("validates_action_time_interval")
        or not claim_policy.get("validates_action_vector")
        or not claim_policy.get("does_not_capture_video")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_create_annotation_rows")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_run_handoff")
        or not claim_policy.get("does_not_claim_intervention_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled capture action finalize must preserve claim policy")
    if bool(claim_policy.get("writes_actions_csv")) != bool(
        payload["readiness"]["controlled_capture_actions_ready"]
    ):
        raise ValueError("controlled capture action finalize write claim mismatch")
    non_goals = payload.get("non_goals", {})
    if any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "controlled capture action finalize cannot claim capture, GT, "
            "annotations, reconstruction, handoff, eval, training, public samples, "
            "replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def _validate_common_summary(payload: Mapping[str, Any], *, ready_key: str) -> None:
    if not isinstance(payload.get("root"), str) or not payload["root"]:
        raise ValueError("controlled capture action summary requires root")
    for section in ("paths", "row_counts", "output", "readiness", "frame_time_range"):
        if not isinstance(payload.get(section), Mapping):
            raise ValueError(f"controlled capture action summary requires {section}")
    row_counts = payload["row_counts"]
    for value in row_counts.values():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("controlled capture action row_counts must be non-negative ints")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("controlled capture action summary requires issues")
    if not isinstance(payload.get("next_actions"), list):
        raise ValueError("controlled capture action summary requires next_actions")
    readiness = payload["readiness"]
    if not isinstance(readiness.get(ready_key), bool):
        raise ValueError(f"controlled capture action readiness requires {ready_key}")
    expected_status = (
        payload["status"].rsplit("_", 1)[0] + "_ready"
        if readiness[ready_key]
        else payload["status"].rsplit("_", 1)[0] + "_blocked"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled capture action status mismatch")


def _template_row(object_id: str) -> dict[str, str]:
    return {
        "action_id": f"{_TODO_ACTION_ID}_{object_id}",
        "action_type": _TODO_ACTION_TYPE,
        "object_id": object_id,
        "start_timestamp": _TODO_START,
        "end_timestamp": _TODO_END,
        "actor": "",
        "target_object_id": "",
        "vector_x": _TODO_VECTOR_X,
        "vector_y": _TODO_VECTOR_Y,
        "vector_z": _TODO_VECTOR_Z,
    }


def _finalize_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    frames: Sequence[Mapping[str, str]],
    object_ids: Sequence[str],
    require_nonzero_vector: bool,
    require_frame_interval: bool,
    require_frame_action_refs: bool,
) -> tuple[list[str], list[dict[str, str]], dict[str, Any]]:
    issues: list[str] = []
    frame_timestamps, timestamp_issues = _frame_timestamps(frames)
    issues.extend(timestamp_issues)
    known_objects = set(object_ids)
    frame_action_refs = {
        str(row.get("action_id", "")).strip()
        for row in frames
        if str(row.get("action_id", "")).strip()
    }
    seen_actions: set[str] = set()
    finalized: list[dict[str, str]] = []
    covered_frames: set[str] = set()
    frame_ref_count = 0
    for index, row in enumerate(rows, start=1):
        label = f"action row {index}"
        finalized_row, row_issues, row_coverage = _finalized_row(
            row,
            label=label,
            known_objects=known_objects,
            frame_timestamps=frame_timestamps,
            frame_action_refs=frame_action_refs,
            require_nonzero_vector=require_nonzero_vector,
            require_frame_interval=require_frame_interval,
            require_frame_action_refs=require_frame_action_refs,
        )
        action_id = finalized_row.get("action_id", "")
        if action_id:
            if action_id in seen_actions:
                row_issues.append(f"duplicate action_id: {action_id}")
            seen_actions.add(action_id)
        issues.extend(row_issues)
        covered_frames.update(row_coverage["covered_frame_ids"])
        frame_ref_count += int(row_coverage["frame_action_ref_count"])
        finalized.append(finalized_row)
    if not finalized:
        issues.append("action source requires at least one row")
    coverage = {
        "covered_frame_count": len(covered_frames),
        "frame_action_ref_count": frame_ref_count,
        "frame_time_range": _time_range([item["timestamp"] for item in frame_timestamps]),
    }
    return _dedupe(issues), finalized, coverage


def _finalized_row(
    row: Mapping[str, str],
    *,
    label: str,
    known_objects: set[str],
    frame_timestamps: Sequence[Mapping[str, Any]],
    frame_action_refs: set[str],
    require_nonzero_vector: bool,
    require_frame_interval: bool,
    require_frame_action_refs: bool,
) -> tuple[dict[str, str], list[str], dict[str, Any]]:
    issues: list[str] = []
    result: dict[str, str] = {}
    action_id = _required_field(row, "action_id", label, issues)
    action_type = _required_field(row, "action_type", label, issues)
    object_id = _required_field(row, "object_id", label, issues)
    result["action_id"] = action_id
    result["action_type"] = action_type
    result["object_id"] = object_id
    if object_id and not _is_todo(object_id) and object_id not in known_objects:
        issues.append(f"{label} references unknown object_id: {object_id}")
    start = _required_numeric_field(row, "start_timestamp", label, issues)
    end = _required_numeric_field(row, "end_timestamp", label, issues)
    result["start_timestamp"] = str(row.get("start_timestamp", "")).strip()
    result["end_timestamp"] = str(row.get("end_timestamp", "")).strip()
    covered_frame_ids: set[str] = set()
    if start is not None and end is not None:
        if end <= start:
            issues.append(f"{label} end_timestamp must be greater than start_timestamp")
        if frame_timestamps:
            min_time = float(frame_timestamps[0]["timestamp"])
            max_time = float(frame_timestamps[-1]["timestamp"])
            if start < min_time or end > max_time:
                issues.append(f"{label} timestamps must fall within frame range")
            covered_frame_ids = {
                str(frame["frame_id"])
                for frame in frame_timestamps
                if start <= float(frame["timestamp"]) <= end
            }
            if require_frame_interval and not covered_frame_ids:
                issues.append(f"{label} interval does not cover any frame timestamp")
        elif require_frame_interval:
            issues.append("frames.csv requires numeric timestamps before actions can finalize")
    for key in ("actor", "target_object_id"):
        value = str(row.get(key, "")).strip()
        if value and _is_todo(value):
            issues.append(f"{label} {key} is still TODO")
        if key == "target_object_id" and value and not _is_todo(value) and value not in known_objects:
            issues.append(f"{label} references unknown target_object_id: {value}")
        result[key] = value
    vector_values = []
    for key in ("vector_x", "vector_y", "vector_z"):
        value = _required_numeric_field(row, key, label, issues)
        result[key] = str(row.get(key, "")).strip()
        if value is not None:
            vector_values.append(value)
    if len(vector_values) == 3 and require_nonzero_vector:
        magnitude_sq = sum(value * value for value in vector_values)
        if magnitude_sq <= 0.0:
            issues.append(f"{label} action vector must be non-zero")
    frame_ref_count = 1 if action_id and action_id in frame_action_refs else 0
    if require_frame_action_refs and not frame_ref_count:
        issues.append(f"{label} action_id is not referenced by frames.csv")
    return result, issues, {
        "covered_frame_ids": covered_frame_ids,
        "frame_action_ref_count": frame_ref_count,
    }


def _read_rows(path: Path, *, required: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"controlled capture action CSV has no header: {path}")
        missing = [key for key in required if key not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"controlled capture action CSV {path} missing columns: "
                + ", ".join(missing)
            )
        return [
            {str(key): "" if value is None else str(value).strip() for key, value in row.items()}
            for row in reader
        ]


def _write_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ACTIONS_CSV_HEADER))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ACTIONS_CSV_HEADER})


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


def _frame_timestamps(
    rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    records: list[dict[str, Any]] = []
    previous: float | None = None
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        frame_id = str(row.get("frame_id", "")).strip()
        if not frame_id:
            issues.append(f"frames.csv row {index} requires frame_id")
            continue
        if frame_id in seen_ids:
            issues.append(f"frames.csv has duplicate frame_id: {frame_id}")
        seen_ids.add(frame_id)
        timestamp_text = str(row.get("timestamp", "")).strip()
        if not timestamp_text:
            issues.append(f"frames.csv row {index} requires timestamp")
            continue
        try:
            timestamp = float(timestamp_text)
        except ValueError:
            issues.append(f"frames.csv row {index} timestamp must be numeric")
            continue
        if previous is not None and timestamp <= previous:
            issues.append("frames.csv timestamps must be strictly increasing")
        previous = timestamp
        records.append(
            {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "action_id": str(row.get("action_id", "")).strip(),
            }
        )
    return records, _dedupe(issues)


def _time_range(timestamps: Sequence[float | Mapping[str, Any]]) -> dict[str, Any]:
    values: list[float] = []
    for item in timestamps:
        if isinstance(item, Mapping):
            values.append(float(item["timestamp"]))
        else:
            values.append(float(item))
    if not values:
        return {"start_timestamp": None, "end_timestamp": None}
    return {
        "start_timestamp": float(min(values)),
        "end_timestamp": float(max(values)),
    }


def _template_next_actions(wrote: bool) -> list[str]:
    if wrote:
        return [
            "fill actions.template.csv with measured action event labels and vectors",
            "run finalize-controlled-capture-actions to write actions.csv",
            "set frames.csv action_id on affected source frames before compiling transitions when needed",
        ]
    return ["populate frames.csv and objects.csv, then rerun action template authoring"]


def _finalize_next_actions(wrote: bool) -> list[str]:
    if wrote:
        return [
            "reference action_id from affected frames.csv rows if source-frame binding is needed",
            "rerun audit-controlled-capture-bundle-readiness with intervention readiness",
            "compile objectstate transitions with --require-action-transition",
        ]
    return ["replace TODO / blank action fields with measured action GT values"]


def _resolve_bundle_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _required(row: Mapping[str, str], key: str, label: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"{label} requires {key}")
    return value


def _required_field(
    row: Mapping[str, str],
    key: str,
    label: str,
    issues: list[str],
) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        issues.append(f"{label} requires {key}")
    elif _is_todo(value):
        issues.append(f"{label} {key} is still TODO")
    return value


def _required_numeric_field(
    row: Mapping[str, str],
    key: str,
    label: str,
    issues: list[str],
) -> float | None:
    value = _required_field(row, key, label, issues)
    if not value or _is_todo(value):
        return None
    try:
        return float(value)
    except ValueError:
        issues.append(f"{label} {key} must be numeric")
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
