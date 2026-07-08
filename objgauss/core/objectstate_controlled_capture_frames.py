from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_capture_template import FRAMES_CSV_HEADER

OBJECTSTATE_CONTROLLED_CAPTURE_FRAMES_SCHEMA = (
    "objgauss-objectstate-controlled-capture-frames-v1"
)

RGB_EXTENSIONS = (".png", ".jpg", ".jpeg")
GAUSSIAN_EXTENSIONS = (".ply", ".splat")


def write_objectstate_controlled_capture_frames(
    root: str | Path,
    *,
    rgb_dir: str | Path = "rgb",
    gaussian_dir: str | Path = "gaussians",
    output: str | Path = "frames.csv",
    fps: float = 30.0,
    start_timestamp: float = 0.0,
    frame_id_prefix: str = "frame-",
    view_id: str = "",
    lighting_id: str = "",
    camera_pose: Sequence[float] | None = None,
    require_gaussian_pairs: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    bundle_root = Path(root)
    rgb_root = _resolve_bundle_path(bundle_root, rgb_dir)
    gaussian_root = _resolve_bundle_path(bundle_root, gaussian_dir)
    output_path = _resolve_bundle_path(bundle_root, output)

    issues: list[str] = []
    camera = _camera_pose_values(camera_pose, issues)
    rgb_records, rgb_issues = _scan_files(
        rgb_root,
        RGB_EXTENSIONS,
        kind="rgb",
        bundle_root=bundle_root,
    )
    gaussian_records, gaussian_issues = _scan_files(
        gaussian_root,
        GAUSSIAN_EXTENSIONS,
        kind="gaussian",
        bundle_root=bundle_root,
    )
    issues.extend(rgb_issues)
    issues.extend(gaussian_issues)

    duplicate_rgb_stems = _duplicate_stems(rgb_records)
    duplicate_gaussian_stems = _duplicate_stems(gaussian_records)
    if duplicate_rgb_stems:
        issues.append(f"duplicate RGB stems: {', '.join(duplicate_rgb_stems)}")
    if duplicate_gaussian_stems:
        issues.append(
            f"duplicate Gaussian stems: {', '.join(duplicate_gaussian_stems)}"
        )

    gaussian_by_stem = {
        record["stem"]: record
        for record in gaussian_records
        if record["stem"] not in duplicate_gaussian_stems
    }
    frame_records: list[dict[str, Any]] = []
    missing_gaussian_files: list[str] = []
    for index, rgb_record in enumerate(rgb_records):
        gaussian_record = gaussian_by_stem.get(rgb_record["stem"])
        if gaussian_record is None:
            missing_gaussian_files.append(rgb_record["relative_path"])
        frame_records.append(
            _frame_record(
                index,
                rgb_record,
                gaussian_record,
                fps=fps,
                start_timestamp=start_timestamp,
                frame_id_prefix=frame_id_prefix,
                view_id=view_id,
                lighting_id=lighting_id,
                camera_pose=camera,
            )
        )

    matched_gaussian_stems = {
        record["rgb_stem"]
        for record in frame_records
        if isinstance(record.get("gaussian"), str) and record["gaussian"]
    }
    extra_gaussian_files = [
        record["relative_path"]
        for record in gaussian_records
        if record["stem"] not in matched_gaussian_stems
    ]

    if not rgb_records:
        issues.append("no RGB frame files found")
    if require_gaussian_pairs and missing_gaussian_files:
        issues.append(
            "missing Gaussian files for RGB stems: "
            + ", ".join(missing_gaussian_files)
        )
    if fps <= 0:
        issues.append("fps must be greater than zero")
    if not frame_id_prefix:
        issues.append("frame_id_prefix must be non-empty")

    output_state = _output_state(output_path)
    output_writable = bool(output_state["can_write"] or force)
    if not output_writable:
        issues.append("frames.csv already contains rows; pass force to overwrite")

    ready_to_write = bool(
        rgb_records
        and fps > 0
        and frame_id_prefix
        and not duplicate_rgb_stems
        and not duplicate_gaussian_stems
        and (not require_gaussian_pairs or not missing_gaussian_files)
        and output_writable
        and not camera["invalid"]
    )
    wrote = False
    if ready_to_write:
        _write_frames_csv(output_path, frame_records)
        wrote = True

    readiness = {
        "rgb_files_present": bool(rgb_records),
        "gaussian_dir_present": gaussian_root.is_dir(),
        "gaussian_pairs_required": bool(require_gaussian_pairs),
        "gaussian_pairs_ready": not missing_gaussian_files,
        "output_writable": output_writable,
        "frames_csv_written": wrote,
        "controlled_capture_frame_rows_ready": wrote,
    }
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_FRAMES_SCHEMA,
        "kind": "objectstate_controlled_capture_frames",
        "status": (
            "objectstate_controlled_capture_frames_ready"
            if readiness["controlled_capture_frame_rows_ready"]
            else "objectstate_controlled_capture_frames_blocked"
        ),
        "root": str(bundle_root),
        "paths": {
            "rgb_dir": str(rgb_root),
            "gaussian_dir": str(gaussian_root),
            "frames_csv": str(output_path),
        },
        "parameters": {
            "fps": float(fps),
            "start_timestamp": float(start_timestamp),
            "frame_id_prefix": frame_id_prefix,
            "view_id": view_id,
            "lighting_id": lighting_id,
            "camera_pose": camera["values"],
            "require_gaussian_pairs": bool(require_gaussian_pairs),
            "force": bool(force),
        },
        "scan": {
            "rgb_file_count": len(rgb_records),
            "gaussian_file_count": len(gaussian_records),
            "frame_row_count": len(frame_records),
            "paired_frame_count": len(matched_gaussian_stems),
            "missing_gaussian_count": len(missing_gaussian_files),
            "extra_gaussian_count": len(extra_gaussian_files),
            "rgb_extensions": list(RGB_EXTENSIONS),
            "gaussian_extensions": list(GAUSSIAN_EXTENSIONS),
            "missing_gaussian_files": missing_gaussian_files,
            "extra_gaussian_files": extra_gaussian_files,
            "duplicate_rgb_stems": duplicate_rgb_stems,
            "duplicate_gaussian_stems": duplicate_gaussian_stems,
        },
        "output": {
            **output_state,
            "wrote_frames_csv": wrote,
        },
        "frames": frame_records,
        "readiness": readiness,
        "issues": issues,
        "next_actions": _next_actions(readiness),
        "claim_policy": {
            "uses_existing_rgb_files": True,
            "uses_existing_gaussian_files": True,
            "writes_timestamped_frame_rows_only": True,
            "requires_pose_annotations_after_this_step": True,
            "does_not_capture_video": True,
            "does_not_create_ground_truth": True,
            "does_not_create_annotation_rows": True,
            "does_not_create_action_rows": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_run_identity_handoff": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "creates_annotation_rows": False,
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
    return validate_objectstate_controlled_capture_frames_summary(payload)


def validate_objectstate_controlled_capture_frames_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled capture frames summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_CAPTURE_FRAMES_SCHEMA:
        raise ValueError(
            "unsupported controlled capture frames schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_capture_frames":
        raise ValueError("controlled capture frames summary kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_capture_frames_ready",
        "objectstate_controlled_capture_frames_blocked",
    }:
        raise ValueError("controlled capture frames summary status is unsupported")
    for key in ("root",):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"controlled capture frames summary requires {key}")
    for section in ("paths", "parameters", "scan", "output", "readiness"):
        if not isinstance(payload.get(section), Mapping):
            raise ValueError(f"controlled capture frames summary requires {section}")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("controlled capture frames summary requires frame list")
    for index, record in enumerate(frames):
        _validate_frame_record(index, record)
    scan = payload["scan"]
    for key in (
        "rgb_file_count",
        "gaussian_file_count",
        "frame_row_count",
        "paired_frame_count",
        "missing_gaussian_count",
        "extra_gaussian_count",
    ):
        if isinstance(scan.get(key), bool) or not isinstance(scan.get(key), int):
            raise ValueError(f"controlled capture frames scan requires int {key}")
    if scan["frame_row_count"] != len(frames):
        raise ValueError("controlled capture frame count must match frames list")
    readiness = payload["readiness"]
    for key in (
        "rgb_files_present",
        "gaussian_dir_present",
        "gaussian_pairs_required",
        "gaussian_pairs_ready",
        "output_writable",
        "frames_csv_written",
        "controlled_capture_frame_rows_ready",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"controlled capture frames readiness requires bool {key}")
    expected_status = (
        "objectstate_controlled_capture_frames_ready"
        if readiness["controlled_capture_frame_rows_ready"]
        else "objectstate_controlled_capture_frames_blocked"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled capture frames status must match readiness")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("controlled capture frames summary requires issues")
    if not isinstance(payload.get("next_actions"), list):
        raise ValueError("controlled capture frames summary requires next_actions")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("uses_existing_rgb_files")
        or not claim_policy.get("uses_existing_gaussian_files")
        or not claim_policy.get("writes_timestamped_frame_rows_only")
        or not claim_policy.get("requires_pose_annotations_after_this_step")
        or not claim_policy.get("does_not_capture_video")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_create_annotation_rows")
        or not claim_policy.get("does_not_create_action_rows")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_run_identity_handoff")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled capture frames must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "controlled capture frames cannot claim capture, GT, annotation, "
            "action, reconstruction, handoff, eval, training, public samples, "
            "replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def _frame_record(
    index: int,
    rgb_record: Mapping[str, Any],
    gaussian_record: Mapping[str, Any] | None,
    *,
    fps: float,
    start_timestamp: float,
    frame_id_prefix: str,
    view_id: str,
    lighting_id: str,
    camera_pose: Mapping[str, Any],
) -> dict[str, Any]:
    timestamp = float(start_timestamp) + (float(index) / float(fps)) if fps > 0 else 0.0
    values = camera_pose["values"]
    return {
        "frame_id": f"{frame_id_prefix}{index:06d}",
        "timestamp": timestamp,
        "rgb": rgb_record["relative_path"],
        "gaussian": (
            gaussian_record["relative_path"]
            if isinstance(gaussian_record, Mapping)
            else ""
        ),
        "action_id": "",
        "view_id": view_id,
        "lighting_id": lighting_id,
        "camera_x": _csv_float(values[0]) if values else "",
        "camera_y": _csv_float(values[1]) if values else "",
        "camera_z": _csv_float(values[2]) if values else "",
        "camera_qx": _csv_float(values[3]) if values else "",
        "camera_qy": _csv_float(values[4]) if values else "",
        "camera_qz": _csv_float(values[5]) if values else "",
        "camera_qw": _csv_float(values[6]) if values else "",
        "rgb_stem": rgb_record["stem"],
        "paired": isinstance(gaussian_record, Mapping),
    }


def _validate_frame_record(index: int, record: Any) -> None:
    if not isinstance(record, Mapping):
        raise TypeError("controlled capture frame records must be mappings")
    if not isinstance(record.get("frame_id"), str) or not record["frame_id"]:
        raise ValueError("controlled capture frame record requires frame_id")
    for key in ("rgb", "gaussian", "action_id", "view_id", "lighting_id"):
        if not isinstance(record.get(key), str):
            raise ValueError(f"controlled capture frame record requires str {key}")
    if isinstance(record.get("timestamp"), bool) or not isinstance(
        record.get("timestamp"),
        (int, float),
    ):
        raise ValueError("controlled capture frame timestamp must be numeric")
    if not isinstance(record.get("paired"), bool):
        raise ValueError("controlled capture frame paired must be bool")


def _scan_files(
    root: Path,
    extensions: Sequence[str],
    *,
    kind: str,
    bundle_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    issues = []
    if not root.exists():
        return [], [f"{kind} directory does not exist: {root}"]
    if not root.is_dir():
        return [], [f"{kind} path is not a directory: {root}"]
    records = [
        {
            "path": str(path),
            "relative_path": _capture_ref(path, bundle_root),
            "name": path.name,
            "stem": path.stem,
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
        }
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]
    records.sort(key=lambda record: _natural_key(record["name"]))
    if not records:
        issues.append(
            f"no {kind} files with extensions {', '.join(extensions)} in {root}"
        )
    return records, issues


def _capture_ref(path: Path, bundle_root: Path) -> str:
    try:
        return path.relative_to(bundle_root).as_posix()
    except ValueError:
        return path.as_posix()


def _duplicate_stems(records: Sequence[Mapping[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for record in records:
        counts[str(record["stem"])] = counts.get(str(record["stem"]), 0) + 1
    return sorted(stem for stem, count in counts.items() if count > 1)


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


def _write_frames_csv(path: Path, frames: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAMES_CSV_HEADER)
        writer.writeheader()
        for frame in frames:
            writer.writerow(
                {
                    "frame_id": frame["frame_id"],
                    "timestamp": _csv_float(frame["timestamp"]),
                    "rgb": frame["rgb"],
                    "gaussian": frame["gaussian"],
                    "action_id": frame["action_id"],
                    "view_id": frame["view_id"],
                    "lighting_id": frame["lighting_id"],
                    "camera_x": frame["camera_x"],
                    "camera_y": frame["camera_y"],
                    "camera_z": frame["camera_z"],
                    "camera_qx": frame["camera_qx"],
                    "camera_qy": frame["camera_qy"],
                    "camera_qz": frame["camera_qz"],
                    "camera_qw": frame["camera_qw"],
                }
            )


def _camera_pose_values(
    value: Sequence[float] | None,
    issues: list[str],
) -> dict[str, Any]:
    if value is None:
        return {"values": None, "invalid": False}
    values = list(value)
    if len(values) != 7:
        issues.append("camera_pose must contain 7 values: x,y,z,qx,qy,qz,qw")
        return {"values": None, "invalid": True}
    try:
        return {"values": [float(item) for item in values], "invalid": False}
    except (TypeError, ValueError):
        issues.append("camera_pose values must be numeric")
        return {"values": None, "invalid": True}


def _next_actions(readiness: Mapping[str, bool]) -> list[str]:
    actions = []
    if not readiness["rgb_files_present"]:
        actions.append("place captured RGB frames in the rgb/ directory")
    if readiness["gaussian_pairs_required"] and not readiness["gaussian_pairs_ready"]:
        actions.append(
            "place same-stem Gaussian files in gaussians/ for every RGB frame"
        )
    if not readiness["output_writable"]:
        actions.append("pass force to overwrite an existing non-empty frames.csv")
    if readiness["controlled_capture_frame_rows_ready"]:
        actions.append("fill annotations.csv with per-frame 6DoF object pose rows")
        actions.append("fill actions.csv for interaction / intervention segments")
        actions.append("rerun audit-controlled-capture-bundle-readiness")
    return actions


def _resolve_bundle_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _natural_key(value: str) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def _csv_float(value: float) -> str:
    return f"{float(value):.6f}"
