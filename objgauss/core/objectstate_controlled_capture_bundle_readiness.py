from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_capture_files import (
    OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA,
    objectstate_controlled_capture_file_audit,
    validate_objectstate_controlled_capture_file_audit_summary,
)
from objgauss.core.objectstate_controlled_capture_import import (
    OBJECTSTATE_CONTROLLED_CAPTURE_IMPORT_SCHEMA,
    objectstate_controlled_capture_import_summary,
    validate_objectstate_controlled_capture_import_summary,
)
from objgauss.core.objectstate_controlled_capture_template import (
    ACTIONS_CSV_HEADER,
    ANNOTATIONS_CSV_HEADER,
    FRAMES_CSV_HEADER,
    OBJECTS_CSV_HEADER,
)

OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_READINESS_SCHEMA = (
    "objgauss-objectstate-controlled-capture-bundle-readiness-v1"
)


def objectstate_controlled_capture_bundle_readiness(
    root: str | Path,
    *,
    sample_json: str | Path = "sample.json",
    objects_csv: str | Path = "objects.csv",
    frames_csv: str | Path = "frames.csv",
    annotations_csv: str | Path = "annotations.csv",
    actions_csv: str | Path | None = "actions.csv",
    require_prediction_ready: bool = False,
    require_intervention_ready: bool = False,
    min_rgb_bytes: int = 1,
    min_gaussian_bytes: int = 1,
    require_frame_formats: bool = True,
    hash_files: bool = False,
    candidate_artifact: str | Path | None = None,
    require_candidate_artifact: bool = False,
    min_candidate_artifact_bytes: int = 1,
    min_identity_scenario_frames: int = 3,
    min_occlusion_fraction: float = 0.5,
    min_view_conditions: int = 2,
    min_lighting_conditions: int = 2,
    min_camera_motion_m: float = 0.01,
) -> dict[str, Any]:
    if min_rgb_bytes < 0:
        raise ValueError("min_rgb_bytes must be non-negative")
    if min_gaussian_bytes < 0:
        raise ValueError("min_gaussian_bytes must be non-negative")
    if min_candidate_artifact_bytes < 0:
        raise ValueError("min_candidate_artifact_bytes must be non-negative")
    bundle_root = Path(root)
    paths = {
        "sample_json": _resolve_bundle_path(bundle_root, sample_json),
        "objects_csv": _resolve_bundle_path(bundle_root, objects_csv),
        "frames_csv": _resolve_bundle_path(bundle_root, frames_csv),
        "annotations_csv": _resolve_bundle_path(bundle_root, annotations_csv),
        "actions_csv": (
            None if actions_csv is None else _resolve_bundle_path(bundle_root, actions_csv)
        ),
    }
    layout = _layout_readiness(bundle_root, paths)
    sample = _sample_readiness(paths["sample_json"])
    csv_info = {
        "objects_csv": _csv_readiness(paths["objects_csv"], OBJECTS_CSV_HEADER),
        "frames_csv": _csv_readiness(paths["frames_csv"], FRAMES_CSV_HEADER),
        "annotations_csv": _csv_readiness(
            paths["annotations_csv"],
            ANNOTATIONS_CSV_HEADER,
        ),
        "actions_csv": _csv_readiness(
            paths["actions_csv"],
            ACTIONS_CSV_HEADER,
            optional=actions_csv is None,
        ),
    }
    row_counts = {
        key: int(value["row_count"]) for key, value in csv_info.items()
    }
    csv_headers_ready = all(value["header_ready"] for value in csv_info.values())
    rows = {
        key: value["rows"] for key, value in csv_info.items() if isinstance(value["rows"], list)
    }
    integrity = _row_integrity(rows)
    import_summary = None
    import_error = None
    if (
        sample["ready"]
        and csv_headers_ready
        and row_counts["objects_csv"] > 0
        and row_counts["frames_csv"] > 0
        and row_counts["annotations_csv"] > 0
        and integrity["ready"]
    ):
        try:
            import_summary = objectstate_controlled_capture_import_summary(
                bundle_root,
                sample_json=sample_json,
                objects_csv=objects_csv,
                frames_csv=frames_csv,
                annotations_csv=annotations_csv,
                actions_csv=actions_csv,
            )
        except Exception as exc:  # noqa: BLE001 - audit reports validation errors.
            import_error = str(exc)
    file_audit = None
    if import_summary is not None:
        file_audit = objectstate_controlled_capture_file_audit(
            import_summary["manifest"],
            root=bundle_root,
            require_gaussian_files=True,
            check_artifact_refs=False,
            min_rgb_bytes=min_rgb_bytes,
            min_gaussian_bytes=min_gaussian_bytes,
            require_frame_formats=require_frame_formats,
            hash_files=hash_files,
        )
    scenario = _identity_scenario_readiness(
        import_summary["manifest"] if import_summary is not None else None,
        min_frames=min_identity_scenario_frames,
        min_occlusion_fraction=min_occlusion_fraction,
        min_view_conditions=min_view_conditions,
        min_lighting_conditions=min_lighting_conditions,
        min_camera_motion_m=min_camera_motion_m,
    )
    intervention_action_gt = _intervention_action_gt_readiness(
        import_summary["manifest"] if import_summary is not None else None
    )
    candidate = _candidate_artifact_readiness(
        candidate_artifact,
        min_bytes=min_candidate_artifact_bytes,
    )
    capture_summary = (
        import_summary["capture_summary"] if import_summary is not None else None
    )
    capture_readiness = (
        capture_summary["readiness"] if capture_summary is not None else {}
    )
    readiness = {
        "layout_ready": bool(layout["ready"]),
        "sample_metadata_ready": bool(sample["ready"]),
        "csv_headers_ready": bool(csv_headers_ready),
        "object_rows_present": row_counts["objects_csv"] > 0,
        "frame_rows_present": row_counts["frames_csv"] > 0,
        "annotation_rows_present": row_counts["annotations_csv"] > 0,
        "frame_annotation_integrity_ready": bool(integrity["ready"]),
        "identity_stage_ready": bool(capture_readiness.get("identity_stage_ready", False)),
        "prediction_stage_ready": bool(
            capture_readiness.get("prediction_stage_ready", False)
        ),
        "intervention_stage_ready": bool(
            capture_readiness.get("intervention_stage_ready", False)
        ),
        "intervention_action_gt_ready": bool(intervention_action_gt["ready"]),
        "capture_import_ready": import_summary is not None,
        "capture_files_ready": (
            file_audit is not None
            and file_audit["status"] == "objectstate_controlled_capture_file_audit_pass"
        ),
        "identity_scenario_ready": bool(scenario["ready"]),
        "candidate_artifact_ready": bool(candidate["ready"]),
    }
    readiness["capture_bundle_ready"] = all(
        (
            readiness["layout_ready"],
            readiness["sample_metadata_ready"],
            readiness["csv_headers_ready"],
            readiness["object_rows_present"],
            readiness["frame_rows_present"],
            readiness["annotation_rows_present"],
            readiness["frame_annotation_integrity_ready"],
            readiness["identity_stage_ready"],
            (not require_prediction_ready or readiness["prediction_stage_ready"]),
            (not require_intervention_ready or readiness["intervention_stage_ready"]),
            (
                not require_intervention_ready
                or readiness["intervention_action_gt_ready"]
            ),
            readiness["capture_import_ready"],
            readiness["capture_files_ready"],
            readiness["identity_scenario_ready"],
        )
    )
    readiness["identity_bundle_handoff_ready"] = bool(
        readiness["capture_bundle_ready"]
        and (not require_candidate_artifact or readiness["candidate_artifact_ready"])
    )
    hard_blockers = _hard_blockers(
        readiness,
        layout=layout,
        sample=sample,
        csv_info=csv_info,
        integrity=integrity,
        import_error=import_error,
        file_audit=file_audit,
        scenario=scenario,
        intervention_action_gt=intervention_action_gt,
        candidate=candidate,
        require_prediction_ready=require_prediction_ready,
        require_intervention_ready=require_intervention_ready,
        require_candidate_artifact=require_candidate_artifact,
    )
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_READINESS_SCHEMA,
        "kind": "objectstate_controlled_capture_bundle_readiness",
        "status": (
            "objectstate_controlled_capture_bundle_readiness_ready"
            if readiness["identity_bundle_handoff_ready"]
            else "objectstate_controlled_capture_bundle_readiness_blocked"
        ),
        "root": str(bundle_root),
        "paths": {
            key: None if value is None else str(value) for key, value in paths.items()
        },
        "requirements": {
            "identity_stage_required": True,
            "prediction_stage_required": bool(require_prediction_ready),
            "intervention_stage_required": bool(require_intervention_ready),
            "intervention_action_gt_required": bool(require_intervention_ready),
            "rgb_files_required": True,
            "gaussian_files_required": True,
            "frame_file_formats_required": bool(require_frame_formats),
            "file_hashes_included": bool(hash_files),
            "candidate_artifact_required": bool(require_candidate_artifact),
            "min_rgb_bytes": int(min_rgb_bytes),
            "min_gaussian_bytes": int(min_gaussian_bytes),
            "min_candidate_artifact_bytes": int(min_candidate_artifact_bytes),
        },
        "layout": layout,
        "sample": sample,
        "csv_files": {
            key: _csv_public_info(value) for key, value in csv_info.items()
        },
        "row_counts": row_counts,
        "row_integrity": integrity,
        "candidate_artifact": candidate,
        "identity_scenario": scenario,
        "intervention_action_gt": intervention_action_gt,
        "import_schema": OBJECTSTATE_CONTROLLED_CAPTURE_IMPORT_SCHEMA,
        "file_audit_schema": OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA,
        "import_summary": import_summary,
        "import_error": import_error,
        "capture_file_audit": file_audit,
        "readiness": readiness,
        "hard_blockers": hard_blockers,
        "next_actions": _next_actions(readiness, hard_blockers),
        "claim_policy": {
            "readiness_audit_is_pre_handoff": True,
            "readiness_audit_may_run_on_incomplete_bundles": True,
            "readiness_audit_does_not_create_ground_truth": True,
            "readiness_audit_does_not_reconstruct_gaussians": True,
            "readiness_audit_does_not_score_candidate_model": True,
            "readiness_audit_does_not_claim_identity_pass": True,
            "readiness_audit_does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "creates_frame_rows": False,
            "creates_annotation_rows": False,
            "creates_action_rows": False,
            "reconstructs_gaussians": False,
            "runs_identity_handoff": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_controlled_capture_bundle_readiness_summary(payload)


def validate_objectstate_controlled_capture_bundle_readiness_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled capture bundle readiness summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_READINESS_SCHEMA:
        raise ValueError(
            "unsupported controlled capture bundle readiness schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_capture_bundle_readiness":
        raise ValueError("controlled capture bundle readiness kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_capture_bundle_readiness_ready",
        "objectstate_controlled_capture_bundle_readiness_blocked",
    }:
        raise ValueError("controlled capture bundle readiness status is unsupported")
    for key in (
        "root",
        "paths",
        "requirements",
        "layout",
        "sample",
        "csv_files",
        "row_counts",
        "row_integrity",
        "candidate_artifact",
        "identity_scenario",
        "intervention_action_gt",
        "readiness",
        "hard_blockers",
        "next_actions",
    ):
        if key not in payload:
            raise ValueError(f"controlled capture bundle readiness requires {key}")
    readiness = payload["readiness"]
    if not isinstance(readiness, Mapping):
        raise ValueError("controlled capture bundle readiness requires readiness")
    for key in (
        "layout_ready",
        "sample_metadata_ready",
        "csv_headers_ready",
        "object_rows_present",
        "frame_rows_present",
        "annotation_rows_present",
        "frame_annotation_integrity_ready",
        "identity_stage_ready",
        "prediction_stage_ready",
        "intervention_stage_ready",
        "intervention_action_gt_ready",
        "capture_import_ready",
        "capture_files_ready",
        "identity_scenario_ready",
        "candidate_artifact_ready",
        "capture_bundle_ready",
        "identity_bundle_handoff_ready",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"controlled capture bundle readiness missing bool {key}")
    expected_status = (
        "objectstate_controlled_capture_bundle_readiness_ready"
        if readiness["identity_bundle_handoff_ready"]
        else "objectstate_controlled_capture_bundle_readiness_blocked"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled capture bundle readiness status must match gates")
    if payload.get("import_summary") is not None:
        validate_objectstate_controlled_capture_import_summary(
            payload["import_summary"]
        )
    if payload.get("capture_file_audit") is not None:
        validate_objectstate_controlled_capture_file_audit_summary(
            payload["capture_file_audit"]
        )
    _validate_intervention_action_gt(payload["intervention_action_gt"])
    if not isinstance(payload.get("hard_blockers"), list):
        raise ValueError("controlled capture bundle readiness hard_blockers must be list")
    if not isinstance(payload.get("next_actions"), list):
        raise ValueError("controlled capture bundle readiness next_actions must be list")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("readiness_audit_is_pre_handoff")
        or not claim_policy.get("readiness_audit_may_run_on_incomplete_bundles")
        or not claim_policy.get("readiness_audit_does_not_create_ground_truth")
        or not claim_policy.get("readiness_audit_does_not_reconstruct_gaussians")
        or not claim_policy.get("readiness_audit_does_not_score_candidate_model")
        or not claim_policy.get("readiness_audit_does_not_claim_identity_pass")
        or not claim_policy.get("readiness_audit_does_not_claim_world_model")
    ):
        raise ValueError("controlled capture bundle readiness must preserve policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("creates_frame_rows")
        or non_goals.get("creates_annotation_rows")
        or non_goals.get("creates_action_rows")
        or non_goals.get("reconstructs_gaussians")
        or non_goals.get("runs_identity_handoff")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("writes_public_samples")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError(
            "controlled capture bundle readiness cannot claim capture, GT, rows, "
            "reconstruction, handoff, training, public samples, replay, diffusion, "
            "or viewer mutation"
        )
    return dict(payload)


def _layout_readiness(
    root: Path,
    paths: Mapping[str, Path | None],
) -> dict[str, Any]:
    file_records = []
    for key in ("sample_json", "objects_csv", "frames_csv", "annotations_csv"):
        path = paths[key]
        file_records.append(_path_record(key, path, expect_file=True))
    if paths.get("actions_csv") is not None:
        file_records.append(
            _path_record("actions_csv", paths["actions_csv"], expect_file=True)
        )
    directory_records = [
        _path_record("rgb_dir", root / "rgb", expect_file=False),
        _path_record("gaussians_dir", root / "gaussians", expect_file=False),
    ]
    ready = all(record["valid"] for record in (*file_records, *directory_records))
    return {
        "ready": ready,
        "files": file_records,
        "directories": directory_records,
        "issues": [
            record["missing_reason"]
            for record in (*file_records, *directory_records)
            if not record["valid"]
        ],
    }


def _validate_intervention_action_gt(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("controlled capture bundle readiness requires intervention_action_gt")
    if not isinstance(payload.get("ready"), bool):
        raise ValueError("intervention_action_gt requires ready bool")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("intervention_action_gt requires readiness")
    for key in (
        "actions_present",
        "nonzero_action_vectors_present",
        "usable_action_transition_present",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"intervention_action_gt readiness requires bool {key}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("intervention_action_gt requires metrics")
    for key in (
        "action_count",
        "nonzero_vector_action_count",
        "usable_action_transition_count",
    ):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"intervention_action_gt metrics requires int {key}")
    usable = payload.get("usable_action_ids")
    if not isinstance(usable, list) or any(not isinstance(item, str) for item in usable):
        raise ValueError("intervention_action_gt requires usable_action_ids list")
    issues = payload.get("issues")
    if not isinstance(issues, list) or any(not isinstance(item, str) for item in issues):
        raise ValueError("intervention_action_gt requires issues list")
    expected_ready = all(bool(readiness[key]) for key in readiness)
    if bool(payload["ready"]) != expected_ready:
        raise ValueError("intervention_action_gt ready must match readiness gates")


def _path_record(kind: str, path: Path | None, *, expect_file: bool) -> dict[str, Any]:
    if path is None:
        return {
            "kind": kind,
            "path": "",
            "exists": False,
            "valid": False,
            "missing_reason": "path not configured",
        }
    exists = path.exists()
    valid = bool(exists and (path.is_file() if expect_file else path.is_dir()))
    reason = None
    if not exists:
        reason = f"{kind} missing"
    elif expect_file and not path.is_file():
        reason = f"{kind} is not a file"
    elif not expect_file and not path.is_dir():
        reason = f"{kind} is not a directory"
    return {
        "kind": kind,
        "path": str(path),
        "exists": bool(exists),
        "valid": valid,
        "missing_reason": reason,
    }


def _sample_readiness(path: Path) -> dict[str, Any]:
    issues = []
    payload = None
    if not path.exists():
        issues.append("sample.json missing")
    else:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # noqa: BLE001 - audit reports parse errors.
            issues.append(f"sample.json parse failed: {exc}")
    if payload is not None:
        if not isinstance(payload, Mapping):
            issues.append("sample.json must be an object")
        else:
            required = (
                "sample_id",
                "source_kind",
                "object_category",
                "scenario",
                "fps",
                "capture_device",
                "observation_modalities",
                "artifact_refs",
                "license",
            )
            for key in required:
                if key not in payload:
                    issues.append(f"sample.json missing {key}")
            if payload.get("source_kind") != "controlled_real":
                issues.append("sample.source_kind must be controlled_real")
            modalities = payload.get("observation_modalities")
            if not (
                isinstance(modalities, Sequence)
                and not isinstance(modalities, (str, bytes))
                and "rgb" in modalities
                and "gaussian" in modalities
            ):
                issues.append("sample.observation_modalities must include rgb and gaussian")
    return {
        "path": str(path),
        "ready": not issues,
        "payload": dict(payload) if isinstance(payload, Mapping) else None,
        "issues": issues,
    }


def _csv_readiness(
    path: Path | None,
    expected_header: Sequence[str],
    *,
    optional: bool = False,
) -> dict[str, Any]:
    if path is None:
        return {
            "path": "",
            "exists": False,
            "header": [],
            "header_ready": bool(optional),
            "row_count": 0,
            "rows": [],
            "issues": [] if optional else ["csv path not configured"],
        }
    issues = []
    header = []
    rows = []
    if not path.exists():
        issues.append(f"{path.name} missing")
    else:
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                header = list(reader.fieldnames or ())
                rows = [
                    {
                        str(key): "" if value is None else str(value).strip()
                        for key, value in row.items()
                    }
                    for row in reader
                ]
        except Exception as exc:  # noqa: BLE001 - audit reports parse errors.
            issues.append(f"{path.name} parse failed: {exc}")
    missing_headers = [key for key in expected_header if key not in header]
    if missing_headers:
        issues.append(
            f"{path.name} missing header columns: " + ", ".join(missing_headers)
        )
    return {
        "path": str(path),
        "exists": path.exists(),
        "header": header,
        "expected_header": list(expected_header),
        "header_ready": not missing_headers and path.exists(),
        "row_count": len(rows),
        "rows": rows,
        "issues": issues,
    }


def _row_integrity(rows: Mapping[str, list[Mapping[str, str]]]) -> dict[str, Any]:
    issues = []
    object_ids = {row.get("object_id", "") for row in rows.get("objects_csv", [])}
    object_ids.discard("")
    frame_ids = []
    timestamps = []
    for row in rows.get("frames_csv", []):
        frame_id = row.get("frame_id", "")
        if frame_id:
            frame_ids.append(frame_id)
        timestamp = row.get("timestamp", "")
        if timestamp:
            try:
                timestamps.append(float(timestamp))
            except ValueError:
                issues.append(f"frame {frame_id or '-'} timestamp must be numeric")
        if not row.get("rgb"):
            issues.append(f"frame {frame_id or '-'} missing rgb ref")
        if not row.get("gaussian"):
            issues.append(f"frame {frame_id or '-'} missing gaussian ref")
    duplicate_frames = sorted(
        frame_id for frame_id in set(frame_ids) if frame_ids.count(frame_id) > 1
    )
    if duplicate_frames:
        issues.append("duplicate frame_id values: " + ", ".join(duplicate_frames))
    if len(timestamps) == len(frame_ids):
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            issues.append("frame timestamps must be strictly increasing")
    frame_set = set(frame_ids)
    annotation_frame_ids = set()
    for row in rows.get("annotations_csv", []):
        frame_id = row.get("frame_id", "")
        object_id = row.get("object_id", "")
        if frame_id:
            annotation_frame_ids.add(frame_id)
        if frame_id and frame_id not in frame_set:
            issues.append(f"annotation references unknown frame_id: {frame_id}")
        if object_id and object_ids and object_id not in object_ids:
            issues.append(f"annotation references unknown object_id: {object_id}")
        missing_pose = [
            key for key in ("x", "y", "z", "qx", "qy", "qz", "qw") if not row.get(key)
        ]
        if missing_pose:
            issues.append(
                f"annotation {frame_id or '-'}:{object_id or '-'} missing pose columns"
            )
    missing_annotation_frames = sorted(frame_set - annotation_frame_ids)
    if missing_annotation_frames:
        issues.append(
            "frames without annotations: " + ", ".join(missing_annotation_frames)
        )
    action_ids = {
        row.get("action_id", "")
        for row in rows.get("actions_csv", [])
        if row.get("action_id", "")
    }
    for row in rows.get("frames_csv", []):
        action_id = row.get("action_id", "")
        if action_id and action_id not in action_ids:
            issues.append(f"frame references unknown action_id: {action_id}")
    return {
        "ready": not issues,
        "issues": issues,
        "frame_id_count": len(frame_set),
        "annotated_frame_count": len(annotation_frame_ids & frame_set),
        "declared_object_count": len(object_ids),
    }


def _identity_scenario_readiness(
    manifest: Mapping[str, Any] | None,
    *,
    min_frames: int,
    min_occlusion_fraction: float,
    min_view_conditions: int,
    min_lighting_conditions: int,
    min_camera_motion_m: float,
) -> dict[str, Any]:
    if manifest is None:
        return {
            "ready": False,
            "requirements": {
                "min_frames": int(min_frames),
                "min_occlusion_fraction": float(min_occlusion_fraction),
                "min_view_conditions": int(min_view_conditions),
                "min_lighting_conditions": int(min_lighting_conditions),
                "min_camera_motion_m": float(min_camera_motion_m),
            },
            "readiness": {
                "min_frame_count_met": False,
                "occlusion_reappearance_present": False,
                "min_view_conditions_met": False,
                "min_lighting_conditions_met": False,
                "camera_motion_present": False,
            },
            "issues": ["capture manifest is not import-ready"],
        }
    frames = list(manifest["frames"])
    view_ids = set()
    lighting_ids = set()
    camera_positions = []
    object_tracks = {item["object_id"]: [] for item in manifest["objects"]}
    for index, frame in enumerate(frames):
        condition = frame.get("condition", {})
        if isinstance(condition, Mapping):
            view_id = condition.get("view_id")
            lighting_id = condition.get("lighting_id")
            if isinstance(view_id, str) and view_id:
                view_ids.add(view_id)
            if isinstance(lighting_id, str) and lighting_id:
                lighting_ids.add(lighting_id)
            pose = condition.get("camera_pose")
            if isinstance(pose, Mapping):
                position = pose.get("position")
                if (
                    isinstance(position, Sequence)
                    and not isinstance(position, (str, bytes))
                    and len(position) == 3
                ):
                    camera_positions.append([float(component) for component in position])
        for item in frame["objects"]:
            occlusion = float(item.get("occlusion_fraction", 0.0))
            visible = bool(item.get("visible", True))
            object_tracks.setdefault(item["object_id"], []).append(
                {
                    "frame_index": index,
                    "visible": visible,
                    "occluded": (not visible) or occlusion >= min_occlusion_fraction,
                }
            )
    occlusion_reappearance = False
    for observations in object_tracks.values():
        occluded_indices = [
            item["frame_index"] for item in observations if item["occluded"]
        ]
        clear_indices = [
            item["frame_index"]
            for item in observations
            if item["visible"] and not item["occluded"]
        ]
        if any(
            any(index < occluded for index in clear_indices)
            and any(index > occluded for index in clear_indices)
            for occluded in occluded_indices
        ):
            occlusion_reappearance = True
            break
    readiness = {
        "min_frame_count_met": len(frames) >= min_frames,
        "occlusion_reappearance_present": occlusion_reappearance,
        "min_view_conditions_met": len(view_ids) >= min_view_conditions,
        "min_lighting_conditions_met": len(lighting_ids) >= min_lighting_conditions,
        "camera_motion_present": (
            len(camera_positions) >= 2
            and _max_camera_translation(camera_positions) >= min_camera_motion_m
        ),
    }
    issues = []
    if not readiness["min_frame_count_met"]:
        issues.append(f"identity scenario requires at least {min_frames} frames")
    if not readiness["occlusion_reappearance_present"]:
        issues.append("identity scenario requires visible-occluded-visible object track")
    if not readiness["min_view_conditions_met"]:
        issues.append(f"identity scenario requires at least {min_view_conditions} views")
    if not readiness["min_lighting_conditions_met"]:
        issues.append(
            f"identity scenario requires at least {min_lighting_conditions} lighting ids"
        )
    if not readiness["camera_motion_present"]:
        issues.append(
            "identity scenario requires camera motion >= "
            f"{min_camera_motion_m:.6f}m"
        )
    return {
        "ready": all(readiness.values()),
        "requirements": {
            "min_frames": int(min_frames),
            "min_occlusion_fraction": float(min_occlusion_fraction),
            "min_view_conditions": int(min_view_conditions),
            "min_lighting_conditions": int(min_lighting_conditions),
            "min_camera_motion_m": float(min_camera_motion_m),
        },
        "readiness": readiness,
        "coverage": {
            "frame_count": len(frames),
            "view_ids": sorted(view_ids),
            "lighting_ids": sorted(lighting_ids),
            "max_camera_translation_m": _max_camera_translation(camera_positions),
        },
        "issues": issues,
    }


def _max_camera_translation(positions: Sequence[Sequence[float]]) -> float:
    max_distance = 0.0
    for left_index, left in enumerate(positions):
        for right in positions[left_index + 1 :]:
            squared = sum(
                (float(left[axis]) - float(right[axis])) ** 2 for axis in range(3)
            )
            max_distance = max(max_distance, squared**0.5)
    return float(max_distance)


def _intervention_action_gt_readiness(
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if manifest is None:
        return {
            "ready": False,
            "readiness": {
                "actions_present": False,
                "nonzero_action_vectors_present": False,
                "usable_action_transition_present": False,
            },
            "metrics": {
                "action_count": 0,
                "nonzero_vector_action_count": 0,
                "usable_action_transition_count": 0,
            },
            "usable_action_ids": [],
            "issues": ["capture manifest is not import-ready"],
        }
    actions = list(manifest.get("actions", ()))
    frames = list(manifest.get("frames", ()))
    issues: list[str] = []
    nonzero_action_ids: set[str] = set()
    usable_action_ids: set[str] = set()
    if not actions:
        issues.append("intervention action GT requires at least one action row")
    object_tracks = _object_pose_tracks(frames)
    for action in actions:
        action_id = str(action.get("action_id", ""))
        vector = action.get("vector")
        if not _is_nonzero_vector(vector):
            issues.append(f"action {action_id or '-'} requires a non-zero vector")
            continue
        nonzero_action_ids.add(action_id)
        refs = [str(action.get("object_id", ""))]
        target = action.get("target_object_id")
        if isinstance(target, str) and target:
            refs.append(target)
        if any(
            _action_fits_object_transition(action, object_tracks.get(object_id, ()))
            for object_id in refs
            if object_id
        ):
            usable_action_ids.add(action_id)
        else:
            issues.append(
                f"action {action_id or '-'} does not fit any referenced object transition"
            )
    readiness = {
        "actions_present": bool(actions),
        "nonzero_action_vectors_present": bool(actions)
        and len(nonzero_action_ids) == len(actions),
        "usable_action_transition_present": bool(actions)
        and len(usable_action_ids) == len(actions),
    }
    if actions and not usable_action_ids:
        issues.append("intervention action GT requires at least one usable action transition")
    return {
        "ready": all(readiness.values()),
        "readiness": readiness,
        "metrics": {
            "action_count": len(actions),
            "nonzero_vector_action_count": len(nonzero_action_ids),
            "usable_action_transition_count": len(usable_action_ids),
        },
        "usable_action_ids": sorted(usable_action_ids),
        "issues": _dedupe(issues),
    }


def _object_pose_tracks(
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    tracks: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        timestamp = float(frame["timestamp"])
        for item in frame.get("objects", ()):
            object_id = str(item.get("object_id", ""))
            if not object_id or "pose" not in item:
                continue
            tracks.setdefault(object_id, []).append(
                {
                    "frame_id": str(frame.get("frame_id", "")),
                    "timestamp": timestamp,
                }
            )
    for observations in tracks.values():
        observations.sort(key=lambda item: item["timestamp"])
    return tracks


def _action_fits_object_transition(
    action: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> bool:
    if len(observations) < 2:
        return False
    start = float(action["start_timestamp"])
    end = float(action["end_timestamp"])
    for source, target in zip(observations[:-1], observations[1:], strict=False):
        if start >= float(source["timestamp"]) and end <= float(target["timestamp"]):
            return True
    return False


def _is_nonzero_vector(value: Any) -> bool:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        return False
    try:
        vector = [float(component) for component in value]
    except (TypeError, ValueError):
        return False
    return sum(component * component for component in vector) > 0.0


def _candidate_artifact_readiness(
    candidate_artifact: str | Path | None,
    *,
    min_bytes: int,
) -> dict[str, Any]:
    if candidate_artifact is None:
        return {
            "path": "",
            "provided": False,
            "exists": False,
            "is_file": False,
            "size_bytes": None,
            "ready": False,
            "issues": ["candidate artifact path not provided"],
        }
    path = Path(candidate_artifact)
    issues = []
    exists = path.exists()
    is_file = bool(exists and path.is_file())
    size = path.stat().st_size if is_file else None
    if not exists:
        issues.append("candidate artifact path does not exist")
    elif not is_file:
        issues.append("candidate artifact path is not a file")
    elif size is not None and size < min_bytes:
        issues.append(
            f"candidate artifact size {size} < min bytes {int(min_bytes)}"
        )
    return {
        "path": str(path),
        "provided": True,
        "exists": bool(exists),
        "is_file": bool(is_file),
        "size_bytes": size,
        "ready": not issues,
        "issues": issues,
    }


def _hard_blockers(
    readiness: Mapping[str, bool],
    *,
    layout: Mapping[str, Any],
    sample: Mapping[str, Any],
    csv_info: Mapping[str, Mapping[str, Any]],
    integrity: Mapping[str, Any],
    import_error: str | None,
    file_audit: Mapping[str, Any] | None,
    scenario: Mapping[str, Any],
    intervention_action_gt: Mapping[str, Any],
    candidate: Mapping[str, Any],
    require_prediction_ready: bool,
    require_intervention_ready: bool,
    require_candidate_artifact: bool,
) -> list[str]:
    blockers = []
    if not readiness["layout_ready"]:
        blockers.extend(str(issue) for issue in layout["issues"])
    if not readiness["sample_metadata_ready"]:
        blockers.extend(str(issue) for issue in sample["issues"])
    if not readiness["csv_headers_ready"]:
        for info in csv_info.values():
            blockers.extend(str(issue) for issue in info["issues"])
    if not readiness["object_rows_present"]:
        blockers.append("objects.csv requires at least one physical object row")
    if not readiness["frame_rows_present"]:
        blockers.append("frames.csv requires timestamped frame rows")
    if not readiness["annotation_rows_present"]:
        blockers.append("annotations.csv requires frame/object pose rows")
    if not readiness["frame_annotation_integrity_ready"]:
        blockers.extend(str(issue) for issue in integrity["issues"])
    if import_error:
        blockers.append(f"capture import failed: {import_error}")
    if not readiness["identity_stage_ready"]:
        blockers.append("capture summary is not identity-stage ready")
    if require_prediction_ready and not readiness["prediction_stage_ready"]:
        blockers.append("capture summary is not prediction-stage ready")
    if require_intervention_ready and not readiness["intervention_stage_ready"]:
        blockers.append("capture summary is not intervention-stage ready")
    if require_intervention_ready and not readiness["intervention_action_gt_ready"]:
        blockers.extend(str(issue) for issue in intervention_action_gt["issues"])
    if file_audit is None:
        blockers.append("capture file audit cannot run until capture import is ready")
    elif file_audit["status"] != "objectstate_controlled_capture_file_audit_pass":
        blockers.extend(str(issue) for issue in file_audit["issues"])
    if not readiness["identity_scenario_ready"]:
        blockers.extend(str(issue) for issue in scenario["issues"])
    if require_candidate_artifact and not readiness["candidate_artifact_ready"]:
        blockers.extend(str(issue) for issue in candidate["issues"])
    return _dedupe(blockers)


def _next_actions(
    readiness: Mapping[str, bool],
    hard_blockers: Sequence[str],
) -> list[str]:
    actions = []
    if not readiness["layout_ready"]:
        actions.append("run init-controlled-capture-bundle or restore missing template files")
    if not readiness["sample_metadata_ready"]:
        actions.append("complete sample.json with controlled_real metadata")
    if not readiness["csv_headers_ready"]:
        actions.append("restore controlled capture CSV headers")
    if not readiness["object_rows_present"]:
        actions.append("declare physical objects in objects.csv")
    if not readiness["frame_rows_present"]:
        actions.append("add timestamped RGB/Gaussian frame rows to frames.csv")
    if not readiness["annotation_rows_present"]:
        actions.append("add per-frame object pose rows to annotations.csv")
    if not readiness["frame_annotation_integrity_ready"]:
        actions.append("fix frame/action/object references and pose columns")
    if not readiness["capture_files_ready"]:
        actions.append("place real RGB and Gaussian files referenced by frames.csv")
    if not readiness["identity_scenario_ready"]:
        actions.append("capture visible-occluded-visible frames with view, lighting, and camera motion metadata")
    if not readiness.get("intervention_action_gt_ready", False):
        actions.append("finalize actions.csv with non-zero vectors that fit object pose transitions")
    if not readiness["candidate_artifact_ready"]:
        actions.append("provide a non-empty candidate ObjectState artifact for handoff")
    if not hard_blockers:
        actions.append("run controlled-identity-bundle-handoff with the candidate artifact")
    return _dedupe(actions)


def _csv_public_info(info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": info["path"],
        "exists": info["exists"],
        "header": list(info["header"]),
        "expected_header": list(info["expected_header"]),
        "header_ready": bool(info["header_ready"]),
        "row_count": int(info["row_count"]),
        "issues": list(info["issues"]),
    }


def _dedupe(values: Sequence[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _resolve_bundle_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
