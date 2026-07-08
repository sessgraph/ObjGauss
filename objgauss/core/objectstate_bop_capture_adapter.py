from __future__ import annotations

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
) -> dict[str, Any]:
    root = Path(scene_root)
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

    frame_ids = _selected_frame_ids(
        scene_gt,
        scene_camera,
        max_frames=max_frames,
        frame_step=frame_step,
    )
    safe_dataset_id = _slug(dataset_id)
    objects = _objects_from_scene_gt(scene_gt, frame_ids, dataset_id=safe_dataset_id)
    frames = _frames_from_bop_scene(
        root,
        scene_gt,
        scene_gt_info,
        frame_ids,
        dataset_id=safe_dataset_id,
        fps=float(fps),
        rgb_dir=rgb_dir,
        include_gaussian_refs=include_gaussian_refs,
        gaussian_dir=gaussian_dir,
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
        "objects": objects,
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
            },
            "bop_format_notes": {
                "pose_rotation": "cam_R_m2c row-major rotation matrix",
                "pose_translation": "cam_t_m2c converted from millimeters to meters",
                "visibility": "scene_gt_info.visib_fract converted to occlusion_fraction",
            },
        },
        "adapter_policy": {
            "identity_policy": "single_instance_per_bop_obj_id",
            "duplicate_obj_id_policy": "fail_fast",
            "timestamp_policy": "selected_frame_rank_divided_by_fps",
            "does_not_infer_action": True,
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
    require_gaussian_files: bool = False,
    check_artifact_refs: bool = False,
    min_rgb_bytes: int = 1,
    min_gaussian_bytes: int = 1,
    require_frame_formats: bool = True,
    hash_files: bool = False,
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
    manifest = validate_objectstate_controlled_capture_manifest(
        payload.get("manifest")
    )
    capture_summary = validate_objectstate_controlled_capture_summary(
        payload.get("capture_summary")
    )
    validate_objectstate_controlled_real_manifest(
        payload.get("controlled_real_manifest_seed")
    )
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
    obj_ids: set[int] = set()
    for frame_id in frame_ids:
        annotations = _annotation_list(scene_gt, frame_id)
        _reject_duplicate_obj_ids(annotations, frame_id=frame_id)
        for annotation in annotations:
            obj_ids.add(_obj_id(annotation))
    return [
        {
            "object_id": _controlled_object_id(dataset_id, obj_id),
            "category": f"bop_obj_{obj_id:06d}",
            "instance_label": f"BOP object {obj_id}",
        }
        for obj_id in sorted(obj_ids)
    ]


def _frames_from_bop_scene(
    root: Path,
    scene_gt: Mapping[str, Any],
    scene_gt_info: Mapping[str, Any],
    frame_ids: Sequence[int],
    *,
    dataset_id: str,
    fps: float,
    rgb_dir: str,
    include_gaussian_refs: bool,
    gaussian_dir: str,
) -> list[dict[str, Any]]:
    frames = []
    for index, frame_id in enumerate(frame_ids):
        annotations = _annotation_list(scene_gt, frame_id)
        _reject_duplicate_obj_ids(annotations, frame_id=frame_id)
        info_items = _gt_info_list(scene_gt_info, frame_id)
        objects = []
        for annotation_index, annotation in enumerate(annotations):
            info = info_items[annotation_index] if annotation_index < len(info_items) else {}
            obj_id = _obj_id(annotation)
            frame_object = {
                "object_id": _controlled_object_id(dataset_id, obj_id),
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
                "condition": {
                    "view_id": f"bop-camera-frame-{frame_id:06d}",
                    "lighting_id": "bop-default",
                },
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


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
