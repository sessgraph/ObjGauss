from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from objgauss.core.objectstate_bop_capture_adapter import (
    BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
)
from objgauss.core.objectstate_bop_baseline_local_row_handoff import (
    OBJECTSTATE_BOP_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA,
    objectstate_bop_baseline_local_row_handoff,
    validate_objectstate_bop_baseline_local_row_handoff_summary,
)
from objgauss.core.objectstate_bop_rgbd_gaussian_export import (
    OBJECTSTATE_BOP_RGBD_GAUSSIAN_EXPORT_SCHEMA,
    objectstate_bop_rgbd_gaussian_export,
    validate_objectstate_bop_rgbd_gaussian_export_summary,
)
from objgauss.core.objectstate_controlled_identity_eval import (
    ObjectStateControlledIdentityThresholds,
)
from objgauss.core.objectstate_controlled_prediction_eval import (
    ObjectStateControlledPredictionThresholds,
)

OBJECTSTATE_BOP_RGBD_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA = (
    "objgauss-objectstate-bop-rgbd-baseline-local-row-handoff-v1"
)


def objectstate_bop_rgbd_baseline_local_row_handoff(
    scene_root: str | Path,
    *,
    output_root: str | Path,
    sample_id: str,
    candidate_artifact: str | Path | None = None,
    dataset_id: str = "bop-ycbv",
    object_category: str = "bop_objects",
    scenario: str = "bop_pose_sequence",
    fps: float = 30.0,
    license_text: str = "BOP dataset terms; verify source dataset license before redistribution",
    rgb_dir: str = "rgb",
    depth_dir: str = "depth",
    gaussian_dir: str = "gaussians",
    condition_sidecar: str | Path | None = None,
    max_frames: int | None = None,
    frame_step: int = 1,
    identity_policy: str = BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    pose_track_max_distance_m: float = DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
    pixel_stride: int = 1,
    max_points_per_frame: int | None = 50_000,
    min_depth_m: float = 0.0,
    max_depth_m: float | None = None,
    overwrite_gaussian_evidence: bool = False,
    ply_format: str = "binary_little_endian",
    baseline_candidate_id: str = "bop-gaussian-centroid-baseline",
    identity_candidate_source: str = "bop_gaussian_centroid_single_state_baseline",
    max_centroid_distance: float | None = None,
    prediction_policy: str = "constant_velocity",
    prediction_candidate_id: str = "bop-constant-velocity-baseline",
    prediction_candidate_source: str = "controlled-prediction-baseline",
    prediction_confidence: float = 0.5,
    check_artifact_refs: bool = False,
    min_rgb_bytes: int = 1,
    min_gaussian_bytes: int = 1,
    require_frame_formats: bool = True,
    hash_files: bool = False,
    min_candidate_artifact_bytes: int = 1,
    hash_candidate_artifact: bool = False,
    min_identity_scenario_frames: int = 3,
    min_occlusion_fraction: float = 0.5,
    min_view_conditions: int = 2,
    min_lighting_conditions: int = 2,
    min_camera_motion_m: float = 0.01,
    identity_thresholds: ObjectStateControlledIdentityThresholds | None = None,
    prediction_thresholds: ObjectStateControlledPredictionThresholds | None = None,
    synthetic_smoke_passed: bool = True,
    min_real_or_public_rows: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    scene = Path(scene_root)
    out = Path(output_root)
    artifact_path = (
        Path(candidate_artifact)
        if candidate_artifact is not None
        else out / "objectstates.json"
    )
    rgbd_export = objectstate_bop_rgbd_gaussian_export(
        scene,
        sample_id=sample_id,
        dataset_id=dataset_id,
        object_category=object_category,
        scenario=scenario,
        fps=fps,
        license_text=license_text,
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        gaussian_dir=gaussian_dir,
        max_frames=max_frames,
        frame_step=frame_step,
        identity_policy=identity_policy,
        pose_track_max_distance_m=pose_track_max_distance_m,
        pixel_stride=pixel_stride,
        max_points_per_frame=max_points_per_frame,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        overwrite=overwrite_gaussian_evidence,
        ply_format=ply_format,
    )
    baseline_handoff: dict[str, Any] | None = None
    if rgbd_export["readiness"]["phase1_gaussian_evidence_written"]:
        baseline_handoff = objectstate_bop_baseline_local_row_handoff(
            scene,
            output_root=out,
            sample_id=sample_id,
            candidate_artifact=artifact_path,
            dataset_id=dataset_id,
            object_category=object_category,
            scenario=scenario,
            fps=fps,
            license_text=license_text,
            rgb_dir=rgb_dir,
            gaussian_dir=gaussian_dir,
            condition_sidecar=condition_sidecar,
            max_frames=max_frames,
            frame_step=frame_step,
            identity_policy=identity_policy,
            pose_track_max_distance_m=pose_track_max_distance_m,
            baseline_candidate_id=baseline_candidate_id,
            identity_candidate_source=identity_candidate_source,
            max_centroid_distance=max_centroid_distance,
            prediction_policy=prediction_policy,
            prediction_candidate_id=prediction_candidate_id,
            prediction_candidate_source=prediction_candidate_source,
            prediction_confidence=prediction_confidence,
            check_artifact_refs=check_artifact_refs,
            min_rgb_bytes=min_rgb_bytes,
            min_gaussian_bytes=min_gaussian_bytes,
            require_frame_formats=require_frame_formats,
            hash_files=hash_files,
            min_candidate_artifact_bytes=min_candidate_artifact_bytes,
            hash_candidate_artifact=hash_candidate_artifact,
            min_identity_scenario_frames=min_identity_scenario_frames,
            min_occlusion_fraction=min_occlusion_fraction,
            min_view_conditions=min_view_conditions,
            min_lighting_conditions=min_lighting_conditions,
            min_camera_motion_m=min_camera_motion_m,
            identity_thresholds=identity_thresholds,
            prediction_thresholds=prediction_thresholds,
            synthetic_smoke_passed=synthetic_smoke_passed,
            min_real_or_public_rows=min_real_or_public_rows,
            force=force,
        )

    reviewability = _reviewability(rgbd_export, baseline_handoff)
    pass_gates = _pass_gates(baseline_handoff)
    status = (
        "objectstate_bop_rgbd_baseline_local_row_handoff_reviewable"
        if all(reviewability.values())
        else "objectstate_bop_rgbd_baseline_local_row_handoff_incomplete"
    )
    payload = {
        "schema": OBJECTSTATE_BOP_RGBD_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA,
        "kind": "objectstate_bop_rgbd_baseline_local_row_handoff",
        "status": status,
        "rgbd_gaussian_export_schema": OBJECTSTATE_BOP_RGBD_GAUSSIAN_EXPORT_SCHEMA,
        "baseline_local_row_handoff_schema": (
            OBJECTSTATE_BOP_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA
        ),
        "scene_root": str(scene),
        "output_root": str(out),
        "sample_id": sample_id,
        "candidate_artifact": str(artifact_path),
        "gaussian_dir": gaussian_dir,
        "reviewability_gates": reviewability,
        "pass_gates": pass_gates,
        "row_counts": {
            "selected_frames": rgbd_export["row_counts"]["selected_frames"],
            "exported_frames": rgbd_export["row_counts"]["exported_frames"],
            "missing_depth_files": rgbd_export["row_counts"]["missing_depth_files"],
            "rgbd_total_vertices": rgbd_export["row_counts"]["total_vertices"],
            "baseline_frames": _nested_count(
                baseline_handoff, "baseline_frames"
            ),
            "baseline_states": _nested_count(
                baseline_handoff, "baseline_states"
            ),
            "baseline_total_gaussians": _nested_count(
                baseline_handoff, "baseline_total_gaussians"
            ),
            "identity_predictions": _nested_count(
                baseline_handoff, "identity_predictions"
            ),
            "prediction_candidates": _nested_count(
                baseline_handoff, "prediction_candidates"
            ),
        },
        "files": _files(scene, out, artifact_path, gaussian_dir, baseline_handoff),
        "rgbd_export": rgbd_export,
        "baseline_local_row_handoff": baseline_handoff,
        "issues": _issues(rgbd_export, baseline_handoff, reviewability),
        "claim_policy": {
            "writes_rgbd_gaussian_evidence_seed": True,
            "writes_baseline_candidate_artifact_when_rgbd_ready": True,
            "runs_local_row_handoff_when_rgbd_ready": True,
            "requires_existing_bop_scene": True,
            "requires_existing_rgbd_depth_frames": True,
            "rgbd_backprojection_only": True,
            "uses_depth_pixels_for_geometry": True,
            "uses_bop_camera_intrinsics": True,
            "uses_gaussian_centroid_only_for_objectstate_baseline": True,
            "baseline_expected_to_be_negative_evidence": True,
            "reviewable_allows_metric_pass_or_fail": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_use_object_pose_gt_for_rgbd_geometry": True,
            "does_not_use_bop_pose_gt_for_objectstate_prediction": True,
            "does_not_run_splatfacto_or_3dgs_optimization": True,
            "does_not_train_model": True,
            "does_not_claim_intervention_gate": True,
            "does_not_claim_metric_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "captures_video": False,
            "creates_ground_truth": False,
            "uses_pose_gt_for_rgbd_geometry": False,
            "uses_pose_gt_for_objectstate_prediction": False,
            "runs_splatfacto_or_3dgs_optimization": False,
            "runs_tracking_model": False,
            "runs_learned_prediction_model": False,
            "runs_intervention_model": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_bop_rgbd_baseline_local_row_handoff_summary(payload)


def validate_objectstate_bop_rgbd_baseline_local_row_handoff_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP RGB-D baseline local-row handoff summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_RGBD_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA:
        raise ValueError(
            "unsupported BOP RGB-D baseline local-row handoff schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_rgbd_baseline_local_row_handoff":
        raise ValueError("BOP RGB-D baseline local-row handoff kind is unsupported")
    if payload.get("rgbd_gaussian_export_schema") != (
        OBJECTSTATE_BOP_RGBD_GAUSSIAN_EXPORT_SCHEMA
    ):
        raise ValueError("BOP RGB-D baseline local-row export schema mismatch")
    if payload.get("baseline_local_row_handoff_schema") != (
        OBJECTSTATE_BOP_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA
    ):
        raise ValueError("BOP RGB-D baseline local-row handoff schema mismatch")
    for key in (
        "scene_root",
        "output_root",
        "sample_id",
        "candidate_artifact",
        "gaussian_dir",
    ):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"BOP RGB-D baseline local-row handoff requires {key}")
    rgbd_export = validate_objectstate_bop_rgbd_gaussian_export_summary(
        payload.get("rgbd_export")
    )
    baseline_handoff = payload.get("baseline_local_row_handoff")
    if baseline_handoff is not None:
        baseline_handoff = (
            validate_objectstate_bop_baseline_local_row_handoff_summary(
                baseline_handoff
            )
        )
    elif rgbd_export["readiness"]["phase1_gaussian_evidence_written"]:
        raise ValueError(
            "BOP RGB-D baseline local-row handoff missing baseline handoff "
            "despite ready RGB-D export"
        )
    gates = payload.get("reviewability_gates")
    if not isinstance(gates, Mapping) or not gates:
        raise ValueError("BOP RGB-D baseline local-row handoff requires gates")
    if dict(gates) != _reviewability(rgbd_export, baseline_handoff):
        raise ValueError("BOP RGB-D baseline local-row reviewability mismatch")
    if any(not isinstance(value, bool) for value in gates.values()):
        raise ValueError("BOP RGB-D baseline local-row gates must be bool")
    expected_status = (
        "objectstate_bop_rgbd_baseline_local_row_handoff_reviewable"
        if all(gates.values())
        else "objectstate_bop_rgbd_baseline_local_row_handoff_incomplete"
    )
    if payload.get("status") != expected_status:
        raise ValueError("BOP RGB-D baseline local-row status mismatch")
    if dict(payload.get("pass_gates", {})) != _pass_gates(baseline_handoff):
        raise ValueError("BOP RGB-D baseline local-row pass gates mismatch")
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("BOP RGB-D baseline local-row handoff requires row counts")
    for key in (
        "selected_frames",
        "exported_frames",
        "missing_depth_files",
        "rgbd_total_vertices",
        "baseline_frames",
        "baseline_states",
        "baseline_total_gaussians",
        "identity_predictions",
        "prediction_candidates",
    ):
        value = row_counts.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"BOP RGB-D baseline local-row handoff invalid count: {key}"
            )
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("BOP RGB-D baseline local-row handoff requires files")
    for key in (
        "candidate_artifact",
        "gaussian_dir",
        "phase1_evidence_ledger",
        "identity_evidence_package_summary",
        "prediction_evidence_package_summary",
    ):
        if not isinstance(files.get(key), str) or not files[key]:
            raise ValueError(f"BOP RGB-D baseline local-row handoff missing {key}")
    for key in ("issues",):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"BOP RGB-D baseline local-row handoff requires {key}")
    claim_policy = payload.get("claim_policy")
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("writes_rgbd_gaussian_evidence_seed")
        or not claim_policy.get("writes_baseline_candidate_artifact_when_rgbd_ready")
        or not claim_policy.get("runs_local_row_handoff_when_rgbd_ready")
        or not claim_policy.get("requires_existing_bop_scene")
        or not claim_policy.get("requires_existing_rgbd_depth_frames")
        or not claim_policy.get("rgbd_backprojection_only")
        or not claim_policy.get("uses_depth_pixels_for_geometry")
        or not claim_policy.get("uses_bop_camera_intrinsics")
        or not claim_policy.get("uses_gaussian_centroid_only_for_objectstate_baseline")
        or not claim_policy.get("baseline_expected_to_be_negative_evidence")
        or not claim_policy.get("reviewable_allows_metric_pass_or_fail")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_use_object_pose_gt_for_rgbd_geometry")
        or not claim_policy.get("does_not_use_bop_pose_gt_for_objectstate_prediction")
        or not claim_policy.get("does_not_run_splatfacto_or_3dgs_optimization")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_claim_intervention_gate")
        or not claim_policy.get("does_not_claim_metric_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP RGB-D baseline local-row claim policy is too broad")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP RGB-D baseline local-row cannot claim downloads, capture, GT, "
            "pose-GT geometry or prediction, 3DGS optimization, tracking, "
            "learned prediction, intervention, training, public samples, replay, "
            "diffusion, or viewer mutation"
        )
    return dict(payload)


def _reviewability(
    rgbd_export: Mapping[str, Any],
    baseline_handoff: Mapping[str, Any] | None,
) -> dict[str, bool]:
    if baseline_handoff is None:
        return {
            "rgbd_export_ready": bool(
                rgbd_export["readiness"]["phase1_gaussian_evidence_written"]
            ),
            "baseline_candidate_written": False,
            "baseline_candidate_ready_for_identity_handoff": False,
            "local_row_identity_handoff_reviewable": False,
            "local_row_prediction_handoff_reviewable": False,
            "phase1_evidence_ledger_identity_reviewable": False,
            "phase1_evidence_ledger_prediction_reviewable": False,
        }
    gates = baseline_handoff["reviewability_gates"]
    return {
        "rgbd_export_ready": bool(
            rgbd_export["readiness"]["phase1_gaussian_evidence_written"]
        ),
        "baseline_candidate_written": bool(gates["baseline_candidate_written"]),
        "baseline_candidate_ready_for_identity_handoff": bool(
            gates["baseline_candidate_ready_for_identity_handoff"]
        ),
        "local_row_identity_handoff_reviewable": bool(
            gates["local_row_identity_handoff_reviewable"]
        ),
        "local_row_prediction_handoff_reviewable": bool(
            gates["local_row_prediction_handoff_reviewable"]
        ),
        "phase1_evidence_ledger_identity_reviewable": bool(
            gates["phase1_evidence_ledger_identity_reviewable"]
        ),
        "phase1_evidence_ledger_prediction_reviewable": bool(
            gates["phase1_evidence_ledger_prediction_reviewable"]
        ),
    }


def _pass_gates(baseline_handoff: Mapping[str, Any] | None) -> dict[str, bool]:
    if baseline_handoff is None:
        return {
            "identity_handoff_pass": False,
            "prediction_eval_pass": False,
        }
    return dict(baseline_handoff["pass_gates"])


def _nested_count(
    baseline_handoff: Mapping[str, Any] | None,
    key: str,
) -> int:
    if baseline_handoff is None:
        return 0
    return int(baseline_handoff["row_counts"][key])


def _files(
    scene_root: Path,
    output_root: Path,
    artifact_path: Path,
    gaussian_dir: str,
    baseline_handoff: Mapping[str, Any] | None,
) -> dict[str, str]:
    if baseline_handoff is not None:
        return {
            "candidate_artifact": baseline_handoff["files"]["candidate_artifact"],
            "gaussian_dir": str(Path(baseline_handoff["scene_root"]) / gaussian_dir),
            "phase1_evidence_ledger": baseline_handoff["files"][
                "phase1_evidence_ledger"
            ],
            "identity_evidence_package_summary": baseline_handoff["files"][
                "identity_evidence_package_summary"
            ],
            "prediction_evidence_package_summary": baseline_handoff["files"][
                "prediction_evidence_package_summary"
            ],
        }
    return {
        "candidate_artifact": str(artifact_path),
        "gaussian_dir": str(scene_root / gaussian_dir),
        "phase1_evidence_ledger": str(output_root / "phase1-evidence-ledger.json"),
        "identity_evidence_package_summary": str(
            output_root
            / "identity-handoff"
            / "identity-evidence-package-summary.json"
        ),
        "prediction_evidence_package_summary": str(
            output_root
            / "reality-candidates"
            / "prediction-evidence-package-summary.json"
        ),
    }


def _issues(
    rgbd_export: Mapping[str, Any],
    baseline_handoff: Mapping[str, Any] | None,
    gates: Mapping[str, bool],
) -> list[str]:
    issues: list[str] = []
    if not gates["rgbd_export_ready"]:
        issues.extend(str(item) for item in rgbd_export["hard_blockers"])
    if baseline_handoff is None and gates["rgbd_export_ready"]:
        issues.append("RGB-D export is ready but baseline local-row handoff did not run")
    if baseline_handoff is not None:
        issues.extend(str(item) for item in baseline_handoff["issues"])
    return issues
