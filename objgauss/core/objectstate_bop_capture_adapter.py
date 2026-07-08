from __future__ import annotations

import csv
from itertools import permutations
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA,
    objectstate_controlled_capture_summary,
    objectstate_controlled_real_manifest_from_capture_manifest,
    validate_objectstate_controlled_capture_manifest,
    validate_objectstate_controlled_capture_summary,
)
from objgauss.core.objectstate_controlled_capture_files import (
    OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA,
    objectstate_controlled_capture_file_audit,
    validate_objectstate_controlled_capture_file_audit_summary,
)
from objgauss.core.objectstate_controlled_real_rows import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
    validate_objectstate_controlled_real_manifest,
)

OBJECTSTATE_BOP_CAPTURE_ADAPTER_SCHEMA = (
    "objgauss-objectstate-bop-capture-adapter-v1"
)
OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA = (
    "objgauss-objectstate-bop-capture-acceptance-v1"
)
OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA = (
    "objgauss-objectstate-bop-capture-condition-sidecar-v1"
)
OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SUMMARY_SCHEMA = (
    "objgauss-objectstate-bop-capture-condition-sidecar-summary-v1"
)
BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID = "single_instance_per_bop_obj_id"
BOP_IDENTITY_POLICY_POSE_TRACK_PER_OBJ_ID = "pose_track_per_obj_id"
BOP_IDENTITY_POLICIES = (
    BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    BOP_IDENTITY_POLICY_POSE_TRACK_PER_OBJ_ID,
)
DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M = 0.05


def objectstate_bop_capture_manifest_from_scene(
    scene_root: str | Path,
    *,
    sample_id: str,
    dataset_id: str = "bop-ycbv",
    object_category: str = "bop_objects",
    scenario: str = "bop_pose_sequence",
    fps: float = 30.0,
    license_text: str = "BOP dataset terms; verify source dataset license before redistribution",
    rgb_dir: str = "rgb",
    max_frames: int | None = None,
    frame_step: int = 1,
    include_gaussian_refs: bool = False,
    gaussian_dir: str = "gaussians",
    condition_sidecar: str | Path | None = None,
    identity_policy: str = BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    pose_track_max_distance_m: float = DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
) -> dict[str, Any]:
    summary = objectstate_bop_capture_adapter_summary(
        scene_root,
        sample_id=sample_id,
        dataset_id=dataset_id,
        object_category=object_category,
        scenario=scenario,
        fps=fps,
        license_text=license_text,
        rgb_dir=rgb_dir,
        max_frames=max_frames,
        frame_step=frame_step,
        include_gaussian_refs=include_gaussian_refs,
        gaussian_dir=gaussian_dir,
        condition_sidecar=condition_sidecar,
        identity_policy=identity_policy,
        pose_track_max_distance_m=pose_track_max_distance_m,
    )
    return summary["manifest"]


def objectstate_bop_capture_adapter_summary(
    scene_root: str | Path,
    *,
    sample_id: str,
    dataset_id: str = "bop-ycbv",
    object_category: str = "bop_objects",
    scenario: str = "bop_pose_sequence",
    fps: float = 30.0,
    license_text: str = "BOP dataset terms; verify source dataset license before redistribution",
    rgb_dir: str = "rgb",
    max_frames: int | None = None,
    frame_step: int = 1,
    include_gaussian_refs: bool = False,
    gaussian_dir: str = "gaussians",
    condition_sidecar: str | Path | None = None,
    identity_policy: str = BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    pose_track_max_distance_m: float = DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
) -> dict[str, Any]:
    root = Path(scene_root)
    normalized_identity_policy = _identity_policy(identity_policy)
    max_track_distance = _positive_float(
        pose_track_max_distance_m,
        "pose_track_max_distance_m",
    )
    if fps <= 0:
        raise ValueError("fps must be positive")
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")

    scene_camera_path = root / "scene_camera.json"
    scene_gt_path = root / "scene_gt.json"
    scene_gt_info_path = root / "scene_gt_info.json"
    scene_camera = _read_json_mapping(scene_camera_path, "scene_camera.json")
    scene_gt = _read_json_mapping(scene_gt_path, "scene_gt.json")
    scene_gt_info = (
        _read_json_mapping(scene_gt_info_path, "scene_gt_info.json")
        if scene_gt_info_path.exists()
        else {}
    )
    condition_sidecar_path = Path(condition_sidecar) if condition_sidecar else None
    condition_sidecar_payload = _read_condition_sidecar(condition_sidecar_path)

    frame_ids = _selected_frame_ids(
        scene_gt,
        scene_camera,
        max_frames=max_frames,
        frame_step=frame_step,
    )
    safe_dataset_id = _slug(dataset_id)
    identity_plan = _identity_plan_from_scene_gt(
        scene_gt,
        frame_ids,
        dataset_id=safe_dataset_id,
        identity_policy=normalized_identity_policy,
        pose_track_max_distance_m=max_track_distance,
    )
    frames = _frames_from_bop_scene(
        root,
        scene_gt,
        scene_gt_info,
        frame_ids,
        object_id_by_frame_annotation=identity_plan["object_id_by_frame_annotation"],
        fps=float(fps),
        rgb_dir=rgb_dir,
        include_gaussian_refs=include_gaussian_refs,
        gaussian_dir=gaussian_dir,
        condition_sidecar=condition_sidecar_payload,
    )
    modalities = ["rgb", "bop_6d_pose", "bop_visibility", "bop_camera"]
    if include_gaussian_refs:
        modalities.append("gaussian")
    manifest = {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": {
            "sample_id": _required_string(sample_id, "sample_id"),
            "source_kind": "controlled_real",
            "object_category": _required_string(object_category, "object_category"),
            "scenario": _required_string(scenario, "scenario"),
            "fps": float(fps),
            "capture_device": f"{safe_dataset_id}-bop-scene",
            "observation_modalities": modalities,
            "artifact_refs": _artifact_refs(
                include_gt_info=scene_gt_info_path.exists(),
                rgb_dir=rgb_dir,
                include_gaussian_refs=include_gaussian_refs,
                gaussian_dir=gaussian_dir,
            ),
            "license": _required_string(license_text, "license_text"),
        },
        "objects": identity_plan["objects"],
        "actions": [],
        "frames": frames,
    }
    checked_manifest = validate_objectstate_controlled_capture_manifest(manifest)
    capture_summary = objectstate_controlled_capture_summary(checked_manifest)
    controlled_real_seed = objectstate_controlled_real_manifest_from_capture_manifest(
        checked_manifest
    )
    payload = {
        "schema": OBJECTSTATE_BOP_CAPTURE_ADAPTER_SCHEMA,
        "kind": "objectstate_bop_capture_adapter",
        "status": "objectstate_bop_capture_adapter_ready",
        "source": {
            "scene_root": str(root),
            "dataset_id": dataset_id,
            "source_files": {
                "scene_camera": str(scene_camera_path),
                "scene_gt": str(scene_gt_path),
                "scene_gt_info": str(scene_gt_info_path)
                if scene_gt_info_path.exists()
                else None,
                "rgb_dir": str(root / rgb_dir),
                "gaussian_dir": str(root / gaussian_dir)
                if include_gaussian_refs
                else None,
                "condition_sidecar": str(condition_sidecar_path)
                if condition_sidecar_path
                else None,
            },
            "bop_format_notes": {
                "pose_rotation": "cam_R_m2c row-major rotation matrix",
                "pose_translation": "cam_t_m2c converted from millimeters to meters",
                "visibility": "scene_gt_info.visib_fract converted to occlusion_fraction",
            },
        },
        "adapter_policy": {
            "identity_policy": normalized_identity_policy,
            "duplicate_obj_id_policy": _duplicate_obj_id_policy(
                normalized_identity_policy
            ),
            "pose_track_max_distance_m": max_track_distance,
            "uses_bop_pose_gt_for_identity_import": (
                normalized_identity_policy == BOP_IDENTITY_POLICY_POSE_TRACK_PER_OBJ_ID
            ),
            "timestamp_policy": "selected_frame_rank_divided_by_fps",
            "condition_sidecar_policy": "explicit_frame_condition_override",
            "does_not_infer_action": True,
            "does_not_infer_condition_metadata": True,
            "does_not_reconstruct_gaussians": True,
        },
        "row_counts": {
            "objects": len(checked_manifest["objects"]),
            "frames": len(checked_manifest["frames"]),
            "annotations": sum(len(frame["objects"]) for frame in checked_manifest["frames"]),
            "actions": len(checked_manifest["actions"]),
        },
        "selected_frame_ids": frame_ids,
        "manifest_schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "capture_summary_schema": OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA,
        "controlled_real_manifest_schema": OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
        "condition_sidecar_schema": OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA,
        "condition_sidecar": condition_sidecar_payload,
        "manifest": checked_manifest,
        "capture_summary": capture_summary,
        "controlled_real_manifest_seed": controlled_real_seed,
        "readiness": {
            "identity_stage_ready": bool(
                capture_summary["readiness"]["identity_stage_ready"]
            ),
            "prediction_stage_ready": bool(
                capture_summary["readiness"]["prediction_stage_ready"]
            ),
            "intervention_stage_ready": bool(
                capture_summary["readiness"]["intervention_stage_ready"]
            ),
            "real_gaussian_reconstruction_present": bool(
                capture_summary["readiness"]["real_gaussian_reconstruction_present"]
            ),
            "bop_scene_adapter_ready": True,
        },
        "hard_blockers": _hard_blockers(capture_summary),
        "next_actions": _next_actions(capture_summary),
        "claim_policy": {
            "adapter_only": True,
            "imports_existing_bop_scene": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "condition_sidecar_only_controls_frame_conditions": True,
            "does_not_infer_condition_metadata": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_score_candidate_model": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "writes_public_samples": False,
            "reconstructs_gaussians": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "creates_reality_pass_rows": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_bop_capture_adapter_summary(payload)


def objectstate_bop_capture_acceptance_summary(
    scene_root: str | Path,
    *,
    sample_id: str,
    dataset_id: str = "bop-ycbv",
    object_category: str = "bop_objects",
    scenario: str = "bop_pose_sequence",
    fps: float = 30.0,
    license_text: str = "BOP dataset terms; verify source dataset license before redistribution",
    rgb_dir: str = "rgb",
    max_frames: int | None = None,
    frame_step: int = 1,
    include_gaussian_refs: bool = False,
    gaussian_dir: str = "gaussians",
    condition_sidecar: str | Path | None = None,
    require_gaussian_files: bool = False,
    check_artifact_refs: bool = False,
    min_rgb_bytes: int = 1,
    min_gaussian_bytes: int = 1,
    require_frame_formats: bool = True,
    hash_files: bool = False,
    identity_policy: str = BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    pose_track_max_distance_m: float = DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
) -> dict[str, Any]:
    root = Path(scene_root)
    effective_include_gaussian_refs = bool(
        include_gaussian_refs or require_gaussian_files
    )
    adapter = objectstate_bop_capture_adapter_summary(
        root,
        sample_id=sample_id,
        dataset_id=dataset_id,
        object_category=object_category,
        scenario=scenario,
        fps=fps,
        license_text=license_text,
        rgb_dir=rgb_dir,
        max_frames=max_frames,
        frame_step=frame_step,
        include_gaussian_refs=effective_include_gaussian_refs,
        gaussian_dir=gaussian_dir,
        condition_sidecar=condition_sidecar,
        identity_policy=identity_policy,
        pose_track_max_distance_m=pose_track_max_distance_m,
    )
    file_audit = objectstate_controlled_capture_file_audit(
        adapter["manifest"],
        root=root,
        require_gaussian_files=require_gaussian_files,
        check_artifact_refs=check_artifact_refs,
        min_rgb_bytes=min_rgb_bytes,
        min_gaussian_bytes=min_gaussian_bytes,
        require_frame_formats=require_frame_formats,
        hash_files=hash_files,
    )
    adapter_readiness = adapter["readiness"]
    file_readiness = file_audit["readiness"]
    readiness = {
        "bop_scene_adapter_ready": bool(adapter_readiness["bop_scene_adapter_ready"]),
        "capture_file_audit_pass": (
            file_audit["status"] == "objectstate_controlled_capture_file_audit_pass"
        ),
        "rgb_files_present": bool(file_readiness["rgb_files_present"]),
        "gaussian_files_present": bool(file_readiness["gaussian_files_present"]),
        "identity_stage_ready": bool(adapter_readiness["identity_stage_ready"]),
        "prediction_stage_ready": bool(adapter_readiness["prediction_stage_ready"]),
        "intervention_stage_ready": bool(adapter_readiness["intervention_stage_ready"]),
        "phase1_gaussian_evidence_required": bool(require_gaussian_files),
        "phase1_gaussian_evidence_ready": bool(
            require_gaussian_files and file_readiness["gaussian_files_present"]
        ),
    }
    readiness["bop_capture_acceptance_ready"] = bool(
        readiness["bop_scene_adapter_ready"]
        and readiness["capture_file_audit_pass"]
    )
    hard_blockers = _acceptance_hard_blockers(
        adapter,
        file_audit,
        require_gaussian_files=require_gaussian_files,
    )
    payload = {
        "schema": OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
        "kind": "objectstate_bop_capture_acceptance",
        "status": (
            "objectstate_bop_capture_acceptance_pass"
            if readiness["bop_capture_acceptance_ready"]
            else "objectstate_bop_capture_acceptance_fail"
        ),
        "adapter_schema": OBJECTSTATE_BOP_CAPTURE_ADAPTER_SCHEMA,
        "file_audit_schema": OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA,
        "scene_root": str(root),
        "sample_id": adapter["manifest"]["sample"]["sample_id"],
        "requirements": {
            "gaussian_files_required": bool(require_gaussian_files),
            "gaussian_refs_included": bool(effective_include_gaussian_refs),
            "artifact_refs_checked": bool(check_artifact_refs),
            "frame_file_formats_required": bool(require_frame_formats),
            "file_hashes_included": bool(hash_files),
            "condition_sidecar_loaded": bool(condition_sidecar),
        },
        "readiness": readiness,
        "hard_blockers": hard_blockers,
        "next_actions": _acceptance_next_actions(
            readiness,
            require_gaussian_files=require_gaussian_files,
        ),
        "adapter": adapter,
        "file_audit": file_audit,
        "manifest": adapter["manifest"],
        "controlled_real_manifest_seed": adapter["controlled_real_manifest_seed"],
        "claim_policy": {
            "acceptance_only": True,
            "imports_existing_bop_scene": True,
            "runs_file_audit": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "condition_sidecar_only_controls_frame_conditions": True,
            "does_not_infer_condition_metadata": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_score_candidate_model": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "writes_public_samples": False,
            "reconstructs_gaussians": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "creates_reality_pass_rows": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_bop_capture_acceptance_summary(payload)


def objectstate_bop_capture_condition_sidecar_summary(
    scene_root: str | Path,
    *,
    condition_csv: str | Path | None = None,
    max_frames: int | None = None,
    frame_step: int = 1,
    default_lighting_id: str = "bop-default",
    min_view_conditions: int = 2,
    min_lighting_conditions: int = 2,
    min_camera_motion_m: float = 0.01,
) -> dict[str, Any]:
    root = Path(scene_root)
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if min_view_conditions < 1:
        raise ValueError("min_view_conditions must be >= 1")
    if min_lighting_conditions < 1:
        raise ValueError("min_lighting_conditions must be >= 1")
    if min_camera_motion_m < 0:
        raise ValueError("min_camera_motion_m must be >= 0")

    scene_camera = _read_json_mapping(root / "scene_camera.json", "scene_camera.json")
    scene_gt = _read_json_mapping(root / "scene_gt.json", "scene_gt.json")
    frame_ids = _selected_frame_ids(
        scene_gt,
        scene_camera,
        max_frames=max_frames,
        frame_step=frame_step,
    )
    condition_csv_path = Path(condition_csv) if condition_csv else None
    csv_conditions = _read_condition_csv(condition_csv_path)
    default_lighting = _required_string(default_lighting_id, "default_lighting_id")
    frames: dict[str, Any] = {}
    for frame_id in frame_ids:
        condition = {
            "view_id": f"bop-camera-frame-{frame_id:06d}",
            "lighting_id": default_lighting,
        }
        if frame_id in csv_conditions:
            condition.update(csv_conditions[frame_id])
        frames[str(frame_id)] = condition
    sidecar = validate_objectstate_bop_capture_condition_sidecar(
        {
            "schema": OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA,
            "kind": "objectstate_bop_capture_condition_sidecar",
            "frames": frames,
            "condition_policy": {
                "sidecar_only": True,
                "does_not_create_ground_truth": True,
                "does_not_infer_from_pixels": True,
            },
        }
    )
    coverage = _condition_sidecar_coverage(sidecar, frame_ids)
    readiness = {
        "selected_frames_covered": coverage["selected_frame_count"]
        == coverage["sidecar_frame_count"],
        "condition_csv_loaded": condition_csv_path is not None,
        "min_view_conditions_met": coverage["view_condition_count"]
        >= min_view_conditions,
        "min_lighting_conditions_met": coverage["lighting_condition_count"]
        >= min_lighting_conditions,
        "camera_motion_present": (
            coverage["camera_pose_count"] >= 2
            and coverage["max_camera_translation_m"] >= min_camera_motion_m
        ),
    }
    readiness["identity_scenario_metadata_ready"] = bool(
        readiness["selected_frames_covered"]
        and readiness["min_view_conditions_met"]
        and readiness["min_lighting_conditions_met"]
        and readiness["camera_motion_present"]
    )
    issues = _condition_sidecar_issues(
        readiness,
        condition_csv_loaded=condition_csv_path is not None,
        min_lighting_conditions=min_lighting_conditions,
        min_camera_motion_m=min_camera_motion_m,
    )
    payload = {
        "schema": OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SUMMARY_SCHEMA,
        "kind": "objectstate_bop_capture_condition_sidecar_summary",
        "status": (
            "objectstate_bop_capture_condition_sidecar_identity_ready"
            if readiness["identity_scenario_metadata_ready"]
            else "objectstate_bop_capture_condition_sidecar_needs_metadata"
        ),
        "scene_root": str(root),
        "source": {
            "scene_camera": str(root / "scene_camera.json"),
            "scene_gt": str(root / "scene_gt.json"),
            "condition_csv": str(condition_csv_path) if condition_csv_path else None,
        },
        "selected_frame_ids": frame_ids,
        "requirements": {
            "min_view_conditions": int(min_view_conditions),
            "min_lighting_conditions": int(min_lighting_conditions),
            "min_camera_motion_m": float(min_camera_motion_m),
        },
        "row_counts": {
            "selected_frames": len(frame_ids),
            "csv_condition_rows": len(csv_conditions),
            "sidecar_frames": len(sidecar["frames"]),
        },
        "sidecar_schema": OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA,
        "sidecar": sidecar,
        "condition_csv_template": _condition_csv_template_rows(
            sidecar,
            frame_ids,
        ),
        "coverage": coverage,
        "readiness": readiness,
        "issues": issues,
        "next_actions": _condition_sidecar_next_actions(readiness),
        "claim_policy": {
            "sidecar_authoring_only": True,
            "imports_existing_bop_scene": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_from_pixels": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_train_model": True,
            "does_not_run_handoff": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "creates_ground_truth": False,
            "infers_conditions_from_pixels": False,
            "reconstructs_gaussians": False,
            "runs_identity_handoff": False,
            "runs_prediction_handoff": False,
            "trains_model": False,
            "writes_public_samples": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_bop_capture_condition_sidecar_summary(payload)


def validate_objectstate_bop_capture_condition_sidecar(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP capture condition sidecar must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA:
        raise ValueError(
            "unsupported BOP capture condition sidecar schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_capture_condition_sidecar":
        raise ValueError("BOP capture condition sidecar kind is unsupported")
    frames = payload.get("frames")
    if not isinstance(frames, Mapping) or not frames:
        raise ValueError("BOP capture condition sidecar requires frames")
    checked_frames: dict[str, Any] = {}
    for key, value in frames.items():
        frame_id = _int_key(key, "condition sidecar frame id")
        normalized_key = str(frame_id)
        if normalized_key in checked_frames:
            raise ValueError(
                "BOP capture condition sidecar has duplicate frame id after "
                f"normalization: {key}"
            )
        checked_frames[normalized_key] = _validate_sidecar_condition(
            value,
            frame_id=frame_id,
        )
    policy = payload.get("condition_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("BOP capture condition sidecar requires condition_policy")
    if (
        not policy.get("sidecar_only")
        or not policy.get("does_not_create_ground_truth")
        or not policy.get("does_not_infer_from_pixels")
    ):
        raise ValueError("BOP capture condition sidecar must preserve condition policy")
    return {
        "schema": OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA,
        "kind": "objectstate_bop_capture_condition_sidecar",
        "frames": checked_frames,
        "condition_policy": {
            "sidecar_only": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_from_pixels": True,
        },
    }


def validate_objectstate_bop_capture_condition_sidecar_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP capture condition sidecar summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SUMMARY_SCHEMA:
        raise ValueError(
            "unsupported BOP capture condition sidecar summary schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_capture_condition_sidecar_summary":
        raise ValueError("BOP capture condition sidecar summary kind is unsupported")
    if payload.get("status") not in {
        "objectstate_bop_capture_condition_sidecar_identity_ready",
        "objectstate_bop_capture_condition_sidecar_needs_metadata",
    }:
        raise ValueError("BOP capture condition sidecar summary status is unsupported")
    sidecar = validate_objectstate_bop_capture_condition_sidecar(payload.get("sidecar"))
    if payload.get("sidecar_schema") != OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA:
        raise ValueError("BOP capture condition sidecar summary sidecar_schema mismatch")
    selected = payload.get("selected_frame_ids")
    if not isinstance(selected, list) or not selected:
        raise ValueError("BOP capture condition sidecar summary requires selected frames")
    if len(sidecar["frames"]) != len(selected):
        raise ValueError("BOP capture condition sidecar frame count mismatch")
    template = payload.get("condition_csv_template")
    if not isinstance(template, list) or len(template) != len(selected):
        raise ValueError("BOP capture condition sidecar summary requires CSV template")
    for row in template:
        if not isinstance(row, Mapping):
            raise ValueError("BOP capture condition sidecar CSV template rows must map")
        for key in _condition_csv_template_fieldnames():
            if key not in row:
                raise ValueError(
                    "BOP capture condition sidecar CSV template row missing "
                    f"{key}"
                )
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping) or not readiness:
        raise ValueError("BOP capture condition sidecar summary requires readiness")
    if any(not isinstance(value, bool) for value in readiness.values()):
        raise ValueError("BOP capture condition sidecar readiness values must be bool")
    expected_status = (
        "objectstate_bop_capture_condition_sidecar_identity_ready"
        if readiness["identity_scenario_metadata_ready"]
        else "objectstate_bop_capture_condition_sidecar_needs_metadata"
    )
    if payload["status"] != expected_status:
        raise ValueError("BOP capture condition sidecar status must match readiness")
    coverage = payload.get("coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("BOP capture condition sidecar summary requires coverage")
    if int(coverage.get("selected_frame_count", 0)) != len(selected):
        raise ValueError("BOP capture condition sidecar coverage frame count mismatch")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("BOP capture condition sidecar issues must be a list")
    if not isinstance(payload.get("next_actions"), list):
        raise ValueError("BOP capture condition sidecar next_actions must be a list")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("sidecar_authoring_only")
        or not claim_policy.get("imports_existing_bop_scene")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_infer_from_pixels")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_run_handoff")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP capture condition sidecar summary must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP capture condition sidecar summary cannot claim downloads, GT, "
            "condition inference, reconstruction, handoff, training, public samples, "
            "or viewer defaults"
        )
    return dict(payload)


def validate_objectstate_bop_capture_adapter_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP capture adapter summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_CAPTURE_ADAPTER_SCHEMA:
        raise ValueError(f"unsupported BOP capture adapter schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_bop_capture_adapter":
        raise ValueError("BOP capture adapter kind is unsupported")
    if payload.get("status") != "objectstate_bop_capture_adapter_ready":
        raise ValueError("BOP capture adapter status is unsupported")
    for key in ("source", "adapter_policy", "row_counts", "selected_frame_ids"):
        if key not in payload:
            raise ValueError(f"BOP capture adapter summary requires {key}")
    adapter_policy = payload.get("adapter_policy")
    if not isinstance(adapter_policy, Mapping):
        raise ValueError("BOP capture adapter summary requires adapter_policy")
    identity_policy = _identity_policy(str(adapter_policy.get("identity_policy", "")))
    if adapter_policy.get("duplicate_obj_id_policy") != _duplicate_obj_id_policy(
        identity_policy
    ):
        raise ValueError("BOP capture adapter duplicate_obj_id_policy mismatch")
    _positive_float(
        adapter_policy.get("pose_track_max_distance_m"),
        "adapter_policy.pose_track_max_distance_m",
    )
    if not isinstance(
        adapter_policy.get("uses_bop_pose_gt_for_identity_import"),
        bool,
    ):
        raise ValueError(
            "BOP capture adapter policy requires "
            "uses_bop_pose_gt_for_identity_import"
        )
    manifest = validate_objectstate_controlled_capture_manifest(
        payload.get("manifest")
    )
    capture_summary = validate_objectstate_controlled_capture_summary(
        payload.get("capture_summary")
    )
    validate_objectstate_controlled_real_manifest(
        payload.get("controlled_real_manifest_seed")
    )
    if (
        payload.get("condition_sidecar_schema")
        != OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA
    ):
        raise ValueError("BOP capture adapter condition_sidecar_schema mismatch")
    condition_sidecar = payload.get("condition_sidecar")
    if condition_sidecar is not None:
        validate_objectstate_bop_capture_condition_sidecar(condition_sidecar)
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("BOP capture adapter row_counts must be a mapping")
    expected = {
        "objects": len(manifest["objects"]),
        "frames": len(manifest["frames"]),
        "annotations": sum(len(frame["objects"]) for frame in manifest["frames"]),
        "actions": len(manifest["actions"]),
    }
    if dict(row_counts) != expected:
        raise ValueError("BOP capture adapter row_counts must match manifest")
    selected = payload.get("selected_frame_ids")
    if not isinstance(selected, list) or len(selected) != len(manifest["frames"]):
        raise ValueError("BOP capture adapter selected_frame_ids must match frames")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("BOP capture adapter requires readiness")
    for key in (
        "identity_stage_ready",
        "prediction_stage_ready",
        "intervention_stage_ready",
        "real_gaussian_reconstruction_present",
        "bop_scene_adapter_ready",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"BOP capture adapter readiness requires bool {key}")
    if readiness["identity_stage_ready"] != capture_summary["readiness"]["identity_stage_ready"]:
        raise ValueError("BOP capture adapter identity readiness mismatch")
    if not isinstance(payload.get("hard_blockers"), list):
        raise ValueError("BOP capture adapter hard_blockers must be a list")
    if not isinstance(payload.get("next_actions"), list):
        raise ValueError("BOP capture adapter next_actions must be a list")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("adapter_only")
        or not claim_policy.get("imports_existing_bop_scene")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("condition_sidecar_only_controls_frame_conditions")
        or not claim_policy.get("does_not_infer_condition_metadata")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_score_candidate_model")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP capture adapter must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("downloads_dataset")
        or non_goals.get("writes_public_samples")
        or non_goals.get("reconstructs_gaussians")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("creates_reality_pass_rows")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError(
            "BOP capture adapter cannot claim downloads, public samples, "
            "reconstruction, training, pass rows, replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def validate_objectstate_bop_capture_acceptance_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP capture acceptance summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA:
        raise ValueError(
            f"unsupported BOP capture acceptance schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_capture_acceptance":
        raise ValueError("BOP capture acceptance kind is unsupported")
    if payload.get("status") not in {
        "objectstate_bop_capture_acceptance_pass",
        "objectstate_bop_capture_acceptance_fail",
    }:
        raise ValueError("BOP capture acceptance status is unsupported")
    adapter = validate_objectstate_bop_capture_adapter_summary(
        payload.get("adapter")
    )
    file_audit = validate_objectstate_controlled_capture_file_audit_summary(
        payload.get("file_audit")
    )
    validate_objectstate_controlled_capture_manifest(payload.get("manifest"))
    validate_objectstate_controlled_real_manifest(
        payload.get("controlled_real_manifest_seed")
    )
    if payload.get("adapter_schema") != OBJECTSTATE_BOP_CAPTURE_ADAPTER_SCHEMA:
        raise ValueError("BOP capture acceptance adapter_schema mismatch")
    if payload.get("file_audit_schema") != OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA:
        raise ValueError("BOP capture acceptance file_audit_schema mismatch")
    if payload.get("sample_id") != adapter["manifest"]["sample"]["sample_id"]:
        raise ValueError("BOP capture acceptance sample_id mismatch")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("BOP capture acceptance requires readiness")
    for key in (
        "bop_scene_adapter_ready",
        "capture_file_audit_pass",
        "rgb_files_present",
        "gaussian_files_present",
        "identity_stage_ready",
        "prediction_stage_ready",
        "intervention_stage_ready",
        "phase1_gaussian_evidence_required",
        "phase1_gaussian_evidence_ready",
        "bop_capture_acceptance_ready",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"BOP capture acceptance readiness requires bool {key}")
    if readiness["capture_file_audit_pass"] != (
        file_audit["status"] == "objectstate_controlled_capture_file_audit_pass"
    ):
        raise ValueError("BOP capture acceptance file audit readiness mismatch")
    expected_status = (
        "objectstate_bop_capture_acceptance_pass"
        if readiness["bop_capture_acceptance_ready"]
        else "objectstate_bop_capture_acceptance_fail"
    )
    if payload["status"] != expected_status:
        raise ValueError("BOP capture acceptance status must match readiness")
    requirements = payload.get("requirements")
    if not isinstance(requirements, Mapping):
        raise ValueError("BOP capture acceptance requires requirements")
    for key in (
        "gaussian_files_required",
        "gaussian_refs_included",
        "artifact_refs_checked",
        "frame_file_formats_required",
        "file_hashes_included",
        "condition_sidecar_loaded",
    ):
        if not isinstance(requirements.get(key), bool):
            raise ValueError(f"BOP capture acceptance requirements requires bool {key}")
    if not isinstance(payload.get("hard_blockers"), list):
        raise ValueError("BOP capture acceptance hard_blockers must be a list")
    if not isinstance(payload.get("next_actions"), list):
        raise ValueError("BOP capture acceptance next_actions must be a list")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("acceptance_only")
        or not claim_policy.get("imports_existing_bop_scene")
        or not claim_policy.get("runs_file_audit")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("condition_sidecar_only_controls_frame_conditions")
        or not claim_policy.get("does_not_infer_condition_metadata")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_score_candidate_model")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP capture acceptance must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("downloads_dataset")
        or non_goals.get("writes_public_samples")
        or non_goals.get("reconstructs_gaussians")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("creates_reality_pass_rows")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError(
            "BOP capture acceptance cannot claim downloads, public samples, "
            "reconstruction, training, pass rows, replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def _identity_policy(value: str) -> str:
    policy = _required_string(value, "identity_policy")
    if policy not in BOP_IDENTITY_POLICIES:
        raise ValueError(
            "identity_policy must be one of "
            f"{', '.join(BOP_IDENTITY_POLICIES)}"
        )
    return policy


def _duplicate_obj_id_policy(identity_policy: str) -> str:
    if identity_policy == BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID:
        return "fail_fast"
    if identity_policy == BOP_IDENTITY_POLICY_POSE_TRACK_PER_OBJ_ID:
        return "pose_track_per_obj_id"
    raise ValueError(f"unsupported identity_policy: {identity_policy}")


def _selected_frame_ids(
    scene_gt: Mapping[str, Any],
    scene_camera: Mapping[str, Any],
    *,
    max_frames: int | None,
    frame_step: int,
) -> list[int]:
    gt_ids = {_int_key(key, "scene_gt frame id") for key in scene_gt}
    camera_ids = {_int_key(key, "scene_camera frame id") for key in scene_camera}
    frame_ids = sorted(gt_ids & camera_ids)
    if not frame_ids:
        raise ValueError("BOP scene requires overlapping scene_gt and scene_camera frame ids")
    selected = frame_ids[::frame_step]
    if max_frames is not None:
        selected = selected[:max_frames]
    if not selected:
        raise ValueError("BOP frame selection produced no frames")
    return selected


def _objects_from_scene_gt(
    scene_gt: Mapping[str, Any],
    frame_ids: Sequence[int],
    *,
    dataset_id: str,
) -> list[dict[str, Any]]:
    return _identity_plan_from_scene_gt(
        scene_gt,
        frame_ids,
        dataset_id=dataset_id,
        identity_policy=BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
        pose_track_max_distance_m=DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
    )["objects"]


def _identity_plan_from_scene_gt(
    scene_gt: Mapping[str, Any],
    frame_ids: Sequence[int],
    *,
    dataset_id: str,
    identity_policy: str,
    pose_track_max_distance_m: float,
) -> dict[str, Any]:
    if identity_policy == BOP_IDENTITY_POLICY_POSE_TRACK_PER_OBJ_ID:
        return _pose_track_identity_plan_from_scene_gt(
            scene_gt,
            frame_ids,
            dataset_id=dataset_id,
            pose_track_max_distance_m=pose_track_max_distance_m,
        )
    obj_ids: set[int] = set()
    object_id_by_frame_annotation: dict[tuple[int, int], str] = {}
    for frame_id in frame_ids:
        annotations = _annotation_list(scene_gt, frame_id)
        _reject_duplicate_obj_ids(annotations, frame_id=frame_id)
        for annotation_index, annotation in enumerate(annotations):
            obj_id = _obj_id(annotation)
            obj_ids.add(obj_id)
            object_id_by_frame_annotation[(frame_id, annotation_index)] = (
                _controlled_object_id(dataset_id, obj_id)
            )
    objects = [
        _bop_object_record(
            _controlled_object_id(dataset_id, obj_id),
            obj_id=obj_id,
            instance_index=None,
        )
        for obj_id in sorted(obj_ids)
    ]
    return {
        "objects": objects,
        "object_id_by_frame_annotation": object_id_by_frame_annotation,
    }


def _pose_track_identity_plan_from_scene_gt(
    scene_gt: Mapping[str, Any],
    frame_ids: Sequence[int],
    *,
    dataset_id: str,
    pose_track_max_distance_m: float,
) -> dict[str, Any]:
    if not frame_ids:
        raise ValueError("pose_track_per_obj_id requires selected frames")
    first_frame_id = int(frame_ids[0])
    first_groups = _annotations_by_obj_id(
        _annotation_list(scene_gt, first_frame_id)
    )
    tracks_by_obj: dict[int, list[dict[str, Any]]] = {}
    object_id_by_frame_annotation: dict[tuple[int, int], str] = {}
    for obj_id in sorted(first_groups):
        entries = sorted(
            first_groups[obj_id],
            key=lambda item: (
                _translation_m(item[1])[0],
                _translation_m(item[1])[1],
                _translation_m(item[1])[2],
                item[0],
            ),
        )
        tracks: list[dict[str, Any]] = []
        for instance_index, (annotation_index, annotation) in enumerate(
            entries,
            start=1,
        ):
            object_id = _controlled_object_instance_id(
                dataset_id,
                obj_id,
                instance_index,
            )
            tracks.append(
                {
                    "obj_id": obj_id,
                    "instance_index": instance_index,
                    "object_id": object_id,
                    "previous_translation": _translation_m(annotation),
                }
            )
            object_id_by_frame_annotation[(first_frame_id, annotation_index)] = object_id
        tracks_by_obj[obj_id] = tracks

    expected_obj_ids = set(tracks_by_obj)
    for frame_id in frame_ids[1:]:
        groups = _annotations_by_obj_id(_annotation_list(scene_gt, int(frame_id)))
        current_obj_ids = set(groups)
        if current_obj_ids != expected_obj_ids:
            missing = sorted(expected_obj_ids - current_obj_ids)
            added = sorted(current_obj_ids - expected_obj_ids)
            raise ValueError(
                "pose_track_per_obj_id requires stable obj_id sets across selected "
                f"frames; frame {frame_id} missing={missing} added={added}"
            )
        for obj_id in sorted(expected_obj_ids):
            tracks = tracks_by_obj[obj_id]
            entries = groups[obj_id]
            if len(entries) != len(tracks):
                raise ValueError(
                    "pose_track_per_obj_id requires stable instance counts for "
                    f"obj_id={obj_id}; frame {frame_id} has {len(entries)}, "
                    f"expected {len(tracks)}"
                )
            assignment = _unique_pose_track_assignment(
                tracks,
                entries,
                frame_id=int(frame_id),
                obj_id=obj_id,
                pose_track_max_distance_m=pose_track_max_distance_m,
            )
            for track_index, annotation_index, translation in assignment:
                track = tracks[track_index]
                object_id_by_frame_annotation[(int(frame_id), annotation_index)] = str(
                    track["object_id"]
                )
                track["previous_translation"] = translation

    objects = [
        _bop_object_record(
            str(track["object_id"]),
            obj_id=obj_id,
            instance_index=int(track["instance_index"]),
        )
        for obj_id in sorted(tracks_by_obj)
        for track in sorted(
            tracks_by_obj[obj_id],
            key=lambda item: int(item["instance_index"]),
        )
    ]
    return {
        "objects": objects,
        "object_id_by_frame_annotation": object_id_by_frame_annotation,
    }


def _annotations_by_obj_id(
    annotations: Sequence[Mapping[str, Any]],
) -> dict[int, list[tuple[int, Mapping[str, Any]]]]:
    groups: dict[int, list[tuple[int, Mapping[str, Any]]]] = {}
    for annotation_index, annotation in enumerate(annotations):
        groups.setdefault(_obj_id(annotation), []).append((annotation_index, annotation))
    return groups


def _unique_pose_track_assignment(
    tracks: Sequence[Mapping[str, Any]],
    entries: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    frame_id: int,
    obj_id: int,
    pose_track_max_distance_m: float,
) -> list[tuple[int, int, list[float]]]:
    best_cost: float | None = None
    second_best_cost: float | None = None
    best_permutation: tuple[int, ...] | None = None
    for entry_permutation in permutations(range(len(entries))):
        distances = []
        for track_index, entry_index in enumerate(entry_permutation):
            previous = tracks[track_index].get("previous_translation")
            if (
                not isinstance(previous, Sequence)
                or isinstance(previous, (str, bytes))
                or len(previous) != 3
            ):
                raise ValueError("pose track is missing previous_translation")
            current_translation = _translation_m(entries[entry_index][1])
            distances.append(math.dist([float(v) for v in previous], current_translation))
        if any(distance > pose_track_max_distance_m for distance in distances):
            continue
        cost = float(sum(distances))
        if best_cost is None or cost < best_cost - 1e-12:
            second_best_cost = best_cost
            best_cost = cost
            best_permutation = tuple(entry_permutation)
        elif second_best_cost is None or cost < second_best_cost:
            second_best_cost = cost
    if best_permutation is None or best_cost is None:
        raise ValueError(
            "pose_track_per_obj_id could not match obj_id="
            f"{obj_id} instances in frame {frame_id} within "
            f"{pose_track_max_distance_m:.6f}m"
        )
    if second_best_cost is not None and abs(second_best_cost - best_cost) <= 1e-12:
        raise ValueError(
            "pose_track_per_obj_id ambiguous pose continuity for "
            f"obj_id={obj_id} in frame {frame_id}"
        )
    return [
        (
            track_index,
            entries[entry_index][0],
            _translation_m(entries[entry_index][1]),
        )
        for track_index, entry_index in enumerate(best_permutation)
    ]


def _bop_object_record(
    object_id: str,
    *,
    obj_id: int,
    instance_index: int | None,
) -> dict[str, Any]:
    if instance_index is None:
        return {
            "object_id": object_id,
            "category": f"bop_obj_{obj_id:06d}",
            "instance_label": f"BOP object {obj_id}",
        }
    return {
        "object_id": object_id,
        "category": f"bop_obj_{obj_id:06d}",
        "instance_label": f"BOP object {obj_id} instance {instance_index}",
    }


def _frames_from_bop_scene(
    root: Path,
    scene_gt: Mapping[str, Any],
    scene_gt_info: Mapping[str, Any],
    frame_ids: Sequence[int],
    *,
    object_id_by_frame_annotation: Mapping[tuple[int, int], str],
    fps: float,
    rgb_dir: str,
    include_gaussian_refs: bool,
    gaussian_dir: str,
    condition_sidecar: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    frames = []
    for index, frame_id in enumerate(frame_ids):
        annotations = _annotation_list(scene_gt, frame_id)
        info_items = _gt_info_list(scene_gt_info, frame_id)
        objects = []
        for annotation_index, annotation in enumerate(annotations):
            info = info_items[annotation_index] if annotation_index < len(info_items) else {}
            object_id = object_id_by_frame_annotation.get((int(frame_id), annotation_index))
            if object_id is None:
                raise ValueError(
                    "BOP identity policy did not assign object_id for "
                    f"frame {frame_id} annotation {annotation_index}"
                )
            frame_object = {
                "object_id": object_id,
                "visible": _visible_from_info(info),
                "pose": {
                    "position": _translation_m(annotation),
                    "rotation_xyzw": _rotation_xyzw(annotation),
                },
            }
            if "visib_fract" in info:
                frame_object["occlusion_fraction"] = 1.0 - _clamp01(
                    _number(info["visib_fract"], "visib_fract")
                )
            objects.append(frame_object)
        observation = {"rgb": _rgb_ref(root, frame_id, rgb_dir=rgb_dir)}
        if include_gaussian_refs:
            observation["gaussian"] = f"{gaussian_dir}/{frame_id:06d}.ply"
        frames.append(
            {
                "frame_id": f"bop-frame-{frame_id:06d}",
                "timestamp": index / fps,
                "observation": observation,
                "condition": _condition_for_frame(
                    frame_id,
                    condition_sidecar=condition_sidecar,
                ),
                "objects": objects,
            }
        )
    return frames


def _artifact_refs(
    *,
    include_gt_info: bool,
    rgb_dir: str,
    include_gaussian_refs: bool,
    gaussian_dir: str,
) -> list[str]:
    refs = ["scene_camera.json", "scene_gt.json", f"{rgb_dir}/"]
    if include_gt_info:
        refs.append("scene_gt_info.json")
    if include_gaussian_refs:
        refs.append(f"{gaussian_dir}/")
    return refs


def _hard_blockers(capture_summary: Mapping[str, Any]) -> list[str]:
    blockers = [
        "BOP adapter output still needs local per-frame Gaussian reconstruction before full Phase 1",
        "BOP scene has no action events, so intervention rows remain blocked",
        "dataset license must be reviewed before redistribution or public demo claims",
    ]
    if not capture_summary["readiness"]["identity_stage_ready"]:
        blockers.append("selected BOP scene is not identity-stage ready")
    if not capture_summary["readiness"]["prediction_stage_ready"]:
        blockers.append("selected BOP scene is not prediction-stage ready")
    return blockers


def _acceptance_hard_blockers(
    adapter: Mapping[str, Any],
    file_audit: Mapping[str, Any],
    *,
    require_gaussian_files: bool,
) -> list[str]:
    gaussian_ready = bool(file_audit["readiness"]["gaussian_files_present"])
    blockers = [
        blocker
        for blocker in adapter["hard_blockers"]
        if not (
            require_gaussian_files
            and gaussian_ready
            and str(blocker).startswith("BOP adapter output still needs")
        )
    ]
    if file_audit["status"] != "objectstate_controlled_capture_file_audit_pass":
        blockers.extend(file_audit["issues"])
    if not require_gaussian_files:
        blockers.append(
            "per-frame Gaussian files were not required in this acceptance; "
            "rerun with require_gaussian_files before Phase 1 identity rows"
        )
    return blockers


def _acceptance_next_actions(
    readiness: Mapping[str, bool],
    *,
    require_gaussian_files: bool,
) -> list[str]:
    actions = []
    if not readiness["capture_file_audit_pass"]:
        actions.append("fix missing or invalid BOP RGB / Gaussian frame files")
    if not require_gaussian_files:
        actions.append(
            "reconstruct per-frame Gaussian evidence and rerun with --require-gaussian-files"
        )
    if require_gaussian_files and not readiness["phase1_gaussian_evidence_ready"]:
        actions.append("create valid gaussians/<frame>.ply or .splat files for each selected BOP frame")
    actions.extend(
        [
            "create ObjectState candidate artifact for the accepted BOP frames",
            "run controlled identity and prediction handoff on the accepted manifest",
        ]
    )
    return actions


def _next_actions(capture_summary: Mapping[str, Any]) -> list[str]:
    actions = [
        "run file audit on the generated controlled capture manifest with the BOP scene root",
        "reconstruct per-frame Gaussian evidence under ignored outputs/ and attach gaussian refs",
        "create ObjectState candidate artifact for the adapted BOP frames",
        "run controlled identity and prediction evaluators before considering HOT3D action-like rows",
    ]
    if not capture_summary["readiness"]["identity_stage_ready"]:
        actions.insert(0, "choose a BOP subset where each selected object appears across at least two frames")
    return actions


def _read_condition_sidecar(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return validate_objectstate_bop_capture_condition_sidecar(
        _read_json_mapping(path, "condition_sidecar")
    )


def _read_condition_csv(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"BOP condition CSV does not exist: {path}")
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "frame_id" not in reader.fieldnames:
            raise ValueError("BOP condition CSV requires a frame_id column")
        for index, row in enumerate(reader, start=2):
            frame_id = _int_key(row.get("frame_id"), f"condition CSV row {index} frame_id")
            if frame_id in rows:
                raise ValueError(f"duplicate frame_id in BOP condition CSV: {frame_id}")
            condition: dict[str, Any] = {}
            view_id = _optional_csv_string(row.get("view_id"))
            lighting_id = _optional_csv_string(row.get("lighting_id"))
            if view_id is not None:
                condition["view_id"] = view_id
            if lighting_id is not None:
                condition["lighting_id"] = lighting_id
            pose_values = [
                row.get("camera_x"),
                row.get("camera_y"),
                row.get("camera_z"),
                row.get("camera_qx"),
                row.get("camera_qy"),
                row.get("camera_qz"),
                row.get("camera_qw"),
            ]
            if any(_csv_cell_present(value) for value in pose_values):
                if not all(_csv_cell_present(value) for value in pose_values):
                    raise ValueError(
                        "BOP condition CSV camera pose columns must be complete "
                        f"for frame {frame_id}"
                    )
                condition["camera_pose"] = {
                    "position": [
                        _csv_float(row.get("camera_x"), "camera_x"),
                        _csv_float(row.get("camera_y"), "camera_y"),
                        _csv_float(row.get("camera_z"), "camera_z"),
                    ],
                    "rotation_xyzw": [
                        _csv_float(row.get("camera_qx"), "camera_qx"),
                        _csv_float(row.get("camera_qy"), "camera_qy"),
                        _csv_float(row.get("camera_qz"), "camera_qz"),
                        _csv_float(row.get("camera_qw"), "camera_qw"),
                    ],
                }
            if not condition:
                raise ValueError(
                    "BOP condition CSV rows must include view_id, lighting_id, "
                    f"or complete camera pose columns; empty row for frame {frame_id}"
                )
            rows[frame_id] = condition
    return rows


def _condition_sidecar_coverage(
    sidecar: Mapping[str, Any],
    frame_ids: Sequence[int],
) -> dict[str, Any]:
    frames = sidecar.get("frames", {})
    if not isinstance(frames, Mapping):
        frames = {}
    view_ids: set[str] = set()
    lighting_ids: set[str] = set()
    camera_positions: list[list[float]] = []
    missing_camera_pose_frame_ids: list[int] = []
    for frame_id in frame_ids:
        condition = frames.get(str(frame_id), {})
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
            else:
                missing_camera_pose_frame_ids.append(frame_id)
        else:
            missing_camera_pose_frame_ids.append(frame_id)
    return {
        "selected_frame_count": len(frame_ids),
        "sidecar_frame_count": len(frames),
        "view_ids": sorted(view_ids),
        "view_condition_count": len(view_ids),
        "lighting_ids": sorted(lighting_ids),
        "lighting_condition_count": len(lighting_ids),
        "camera_pose_count": len(camera_positions),
        "missing_camera_pose_frame_ids": missing_camera_pose_frame_ids,
        "max_camera_translation_m": _max_translation(camera_positions),
    }


def _condition_csv_template_rows(
    sidecar: Mapping[str, Any],
    frame_ids: Sequence[int],
) -> list[dict[str, Any]]:
    frames = sidecar.get("frames", {})
    if not isinstance(frames, Mapping):
        frames = {}
    rows = []
    for frame_id in frame_ids:
        condition = frames.get(str(frame_id), {})
        if not isinstance(condition, Mapping):
            condition = {}
        camera_pose = condition.get("camera_pose")
        position: Sequence[float] | None = None
        rotation: Sequence[float] | None = None
        if isinstance(camera_pose, Mapping):
            pose_position = camera_pose.get("position")
            pose_rotation = camera_pose.get("rotation_xyzw")
            if isinstance(pose_position, Sequence) and not isinstance(
                pose_position,
                (str, bytes),
            ):
                position = pose_position
            if isinstance(pose_rotation, Sequence) and not isinstance(
                pose_rotation,
                (str, bytes),
            ):
                rotation = pose_rotation
        rows.append(
            {
                "frame_id": int(frame_id),
                "view_id": str(condition.get("view_id", "")),
                "lighting_id": str(condition.get("lighting_id", "")),
                "camera_x": _template_float(position, 0),
                "camera_y": _template_float(position, 1),
                "camera_z": _template_float(position, 2),
                "camera_qx": _template_float(rotation, 0),
                "camera_qy": _template_float(rotation, 1),
                "camera_qz": _template_float(rotation, 2),
                "camera_qw": _template_float(rotation, 3),
            }
        )
    return rows


def _condition_csv_template_fieldnames() -> list[str]:
    return [
        "frame_id",
        "view_id",
        "lighting_id",
        "camera_x",
        "camera_y",
        "camera_z",
        "camera_qx",
        "camera_qy",
        "camera_qz",
        "camera_qw",
    ]


def _template_float(values: Sequence[float] | None, index: int) -> str:
    if values is None or len(values) <= index:
        return ""
    return f"{float(values[index]):.9g}"


def _condition_sidecar_issues(
    readiness: Mapping[str, bool],
    *,
    condition_csv_loaded: bool,
    min_lighting_conditions: int,
    min_camera_motion_m: float,
) -> list[str]:
    issues = []
    if not condition_csv_loaded:
        issues.append(
            "condition CSV was not provided; generated a default sidecar template "
            "that still needs explicit lighting and camera_pose metadata"
        )
    if not readiness["min_lighting_conditions_met"]:
        issues.append(
            f"identity route requires at least {min_lighting_conditions} lighting conditions"
        )
    if not readiness["camera_motion_present"]:
        issues.append(
            "identity route requires camera_pose metadata with max translation "
            f">= {min_camera_motion_m:.6f}m"
        )
    return issues


def _condition_sidecar_next_actions(readiness: Mapping[str, bool]) -> list[str]:
    if readiness["identity_scenario_metadata_ready"]:
        return [
            "pass the sidecar with --condition-sidecar to BOP acceptance and route audits",
            "continue preparing per-frame Gaussian evidence and a bound ObjectState candidate artifact",
        ]
    actions = [
        "fill a condition CSV with frame_id, view_id, lighting_id, and camera pose columns",
        "regenerate the sidecar and rerun audit-bop-phase1-local-row with --condition-sidecar",
    ]
    if not readiness["camera_motion_present"]:
        actions.insert(
            0,
            "record explicit camera_pose positions for at least two selected BOP frames",
        )
    return actions


def _condition_for_frame(
    frame_id: int,
    *,
    condition_sidecar: Mapping[str, Any] | None,
) -> dict[str, Any]:
    condition: dict[str, Any] = {
        "view_id": f"bop-camera-frame-{frame_id:06d}",
        "lighting_id": "bop-default",
    }
    if condition_sidecar is None:
        return condition
    frames = condition_sidecar.get("frames")
    if not isinstance(frames, Mapping):
        return condition
    override = frames.get(str(frame_id))
    if override is None:
        return condition
    if not isinstance(override, Mapping):
        raise ValueError(f"condition sidecar frame {frame_id} must be a mapping")
    condition.update(dict(override))
    return condition


def _validate_sidecar_condition(
    value: Any,
    *,
    frame_id: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"BOP capture condition sidecar frame {frame_id} must be a mapping"
        )
    result: dict[str, Any] = {}
    if "view_id" in value:
        result["view_id"] = _required_string(value["view_id"], "view_id")
    if "lighting_id" in value:
        result["lighting_id"] = _required_string(value["lighting_id"], "lighting_id")
    if "camera_pose" in value:
        result["camera_pose"] = _validate_sidecar_camera_pose(
            value["camera_pose"],
            frame_id=frame_id,
        )
    if not result:
        raise ValueError(
            f"BOP capture condition sidecar frame {frame_id} must include "
            "view_id, lighting_id, or camera_pose"
        )
    return result


def _validate_sidecar_camera_pose(
    value: Any,
    *,
    frame_id: int,
) -> dict[str, list[float]]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"BOP capture condition sidecar camera_pose frame {frame_id} must be a mapping"
        )
    rotation = _numeric_vector(
        value.get("rotation_xyzw"),
        "camera_pose.rotation_xyzw",
        length=4,
    )
    if sum(component * component for component in rotation) <= 0.0:
        raise ValueError("camera_pose.rotation_xyzw must be non-zero")
    return {
        "position": _numeric_vector(
            value.get("position"),
            "camera_pose.position",
            length=3,
        ),
        "rotation_xyzw": rotation,
    }


def _optional_csv_string(value: Any) -> str | None:
    if not _csv_cell_present(value):
        return None
    return _required_string(str(value), "condition CSV cell")


def _csv_cell_present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _csv_float(value: Any, name: str) -> float:
    if not _csv_cell_present(value):
        raise ValueError(f"{name} must be present")
    try:
        return float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"BOP scene requires {label}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _annotation_list(scene_gt: Mapping[str, Any], frame_id: int) -> list[Mapping[str, Any]]:
    value = scene_gt.get(str(frame_id), scene_gt.get(f"{frame_id:06d}"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"BOP scene_gt frame {frame_id} must contain annotations")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"BOP scene_gt frame {frame_id} annotations must be objects")
    return value


def _gt_info_list(scene_gt_info: Mapping[str, Any], frame_id: int) -> list[Mapping[str, Any]]:
    if not scene_gt_info:
        return []
    value = scene_gt_info.get(str(frame_id), scene_gt_info.get(f"{frame_id:06d}", []))
    if not isinstance(value, list):
        raise ValueError(f"BOP scene_gt_info frame {frame_id} must be a list")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"BOP scene_gt_info frame {frame_id} entries must be objects")
    return value


def _reject_duplicate_obj_ids(
    annotations: Sequence[Mapping[str, Any]],
    *,
    frame_id: int,
) -> None:
    seen: set[int] = set()
    for annotation in annotations:
        obj_id = _obj_id(annotation)
        if obj_id in seen:
            raise ValueError(
                "BOP adapter requires one instance per obj_id in each selected "
                f"frame; duplicate obj_id={obj_id} in frame {frame_id}"
            )
        seen.add(obj_id)


def _obj_id(annotation: Mapping[str, Any]) -> int:
    value = annotation.get("obj_id")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("BOP annotation obj_id must be an integer")
    if value < 1:
        raise ValueError("BOP annotation obj_id must be positive")
    return int(value)


def _translation_m(annotation: Mapping[str, Any]) -> list[float]:
    values = _numeric_vector(annotation.get("cam_t_m2c"), "cam_t_m2c", length=3)
    return [value / 1000.0 for value in values]


def _rotation_xyzw(annotation: Mapping[str, Any]) -> list[float]:
    matrix = _numeric_vector(annotation.get("cam_R_m2c"), "cam_R_m2c", length=9)
    return _matrix3_to_quaternion_xyzw(matrix)


def _matrix3_to_quaternion_xyzw(matrix: Sequence[float]) -> list[float]:
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = [float(v) for v in matrix]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m21 - m12) / scale
        y = (m02 - m20) / scale
        z = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / scale
        x = 0.25 * scale
        y = (m01 + m10) / scale
        z = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / scale
        x = (m01 + m10) / scale
        y = 0.25 * scale
        z = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / scale
        x = (m02 + m20) / scale
        y = (m12 + m21) / scale
        z = 0.25 * scale
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("cam_R_m2c produced a zero quaternion")
    return [x / norm, y / norm, z / norm, w / norm]


def _visible_from_info(info: Mapping[str, Any]) -> bool:
    if "visib_fract" not in info:
        return True
    return _number(info["visib_fract"], "visib_fract") > 0.0


def _rgb_ref(root: Path, frame_id: int, *, rgb_dir: str) -> str:
    for extension in (".png", ".jpg", ".jpeg"):
        ref = f"{rgb_dir}/{frame_id:06d}{extension}"
        if (root / ref).exists():
            return ref
    raise FileNotFoundError(
        f"BOP adapter could not find RGB file for frame {frame_id} under {root / rgb_dir}"
    )


def _controlled_object_id(dataset_id: str, obj_id: int) -> str:
    return f"{dataset_id}-obj-{obj_id:06d}"


def _controlled_object_instance_id(
    dataset_id: str,
    obj_id: int,
    instance_index: int,
) -> str:
    return f"{dataset_id}-obj-{obj_id:06d}-inst-{instance_index:06d}"


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", _required_string(value, "dataset_id"))
    result = result.strip("-_.")
    if not result:
        raise ValueError("dataset_id must contain at least one alphanumeric character")
    return result.lower()


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _int_key(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer-like key") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _numeric_vector(value: Any, name: str, *, length: int) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(value) != length:
        raise ValueError(f"{name} must have length {length}")
    return [_number(item, name) for item in value]


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _positive_float(value: Any, name: str) -> float:
    result = _number(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _max_translation(positions: Sequence[Sequence[float]]) -> float:
    max_distance = 0.0
    for index, first in enumerate(positions):
        for second in positions[index + 1 :]:
            distance = math.dist(first, second)
            if distance > max_distance:
                max_distance = distance
    return max_distance
