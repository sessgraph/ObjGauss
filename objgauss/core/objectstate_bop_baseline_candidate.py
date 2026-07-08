from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from objgauss.core.io_ply import read_ply
from objgauss.core.objectstate_bop_capture_adapter import (
    OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
    objectstate_bop_capture_acceptance_summary,
    validate_objectstate_bop_capture_acceptance_summary,
)
from objgauss.core.trainable_artifact import (
    TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
    validate_trainable_kernel_model_artifact,
)

OBJECTSTATE_BOP_BASELINE_CANDIDATE_SCHEMA = (
    "objgauss-objectstate-bop-baseline-candidate-v1"
)


def write_objectstate_bop_gaussian_centroid_baseline_candidate(
    scene_root: str | Path,
    *,
    output: str | Path,
    sample_id: str,
    dataset_id: str = "bop-ycbv",
    object_category: str = "bop_objects",
    scenario: str = "bop_pose_sequence",
    fps: float = 30.0,
    license_text: str = "BOP dataset terms; verify source dataset license before redistribution",
    rgb_dir: str = "rgb",
    gaussian_dir: str = "gaussians",
    condition_sidecar: str | Path | None = None,
    max_frames: int | None = None,
    frame_step: int = 1,
    candidate_id: str = "bop-gaussian-centroid-baseline",
    force: bool = False,
) -> dict[str, Any]:
    root = Path(scene_root)
    output_path = Path(output)
    if output_path.exists() and not force:
        raise FileExistsError(f"BOP baseline candidate already exists: {output_path}")
    acceptance = objectstate_bop_capture_acceptance_summary(
        root,
        sample_id=sample_id,
        dataset_id=dataset_id,
        object_category=object_category,
        scenario=scenario,
        fps=fps,
        license_text=license_text,
        rgb_dir=rgb_dir,
        include_gaussian_refs=True,
        gaussian_dir=gaussian_dir,
        condition_sidecar=condition_sidecar,
        max_frames=max_frames,
        frame_step=frame_step,
        require_gaussian_files=True,
    )
    artifact, frame_records = _artifact_from_acceptance(
        acceptance,
        scene_root=root,
        output_path=output_path,
        candidate_id=candidate_id,
    )
    validate_trainable_kernel_model_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    row_counts = {
        "frames": len(frame_records),
        "states": sum(record["state_count"] for record in frame_records),
        "total_gaussians": sum(record["gaussian_count"] for record in frame_records),
    }
    payload = {
        "schema": OBJECTSTATE_BOP_BASELINE_CANDIDATE_SCHEMA,
        "kind": "objectstate_bop_gaussian_centroid_baseline_candidate",
        "status": "objectstate_bop_baseline_candidate_written",
        "scene_root": str(root),
        "sample_id": sample_id,
        "output": str(output_path),
        "acceptance_schema": OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
        "target_artifact_schema": TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
        "baseline_policy": {
            "baseline_id": "gaussian_centroid_single_state",
            "uses_per_frame_gaussian_evidence": True,
            "uses_gaussian_xyz_centroid_and_bbox": True,
            "single_state_per_frame": True,
            "expected_to_be_negative_evidence_for_multi_object_identity": True,
            "does_not_use_bop_pose_gt_for_prediction": True,
            "does_not_use_bop_object_ids_for_prediction": True,
            "does_not_train_model": True,
            "does_not_claim_metric_pass": True,
        },
        "row_counts": row_counts,
        "readiness": {
            "bop_acceptance_available": True,
            "phase1_gaussian_evidence_ready": bool(
                acceptance["readiness"]["phase1_gaussian_evidence_ready"]
            ),
            "target_artifact_written": output_path.is_file(),
            "target_artifact_valid": True,
            "ready_for_identity_handoff": True,
        },
        "frame_records": frame_records,
        "acceptance": acceptance,
        "next_commands": [
            (
                "uv run objgauss object-state audit-bop-phase1-local-row "
                f"{root} --output-root {output_path.parent} --sample-id {sample_id} "
                f"--dataset-id {dataset_id} --candidate-artifact {output_path}"
            ),
            (
                "uv run objgauss object-state bop-identity-handoff "
                f"{root} --output-root {output_path.parent} --sample-id {sample_id} "
                f"--dataset-id {dataset_id} --candidate-artifact {output_path}"
            ),
        ],
        "claim_policy": {
            "writes_baseline_candidate_artifact": True,
            "requires_existing_bop_scene": True,
            "requires_existing_gaussian_evidence": True,
            "uses_gaussian_centroid_only": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_use_pose_gt_for_prediction": True,
            "does_not_infer_condition_metadata": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_train_model": True,
            "does_not_run_identity_handoff": True,
            "does_not_run_identity_eval": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "creates_ground_truth": False,
            "uses_pose_gt_for_prediction": False,
            "infers_condition_metadata": False,
            "reconstructs_gaussians": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "runs_identity_handoff": False,
            "runs_identity_eval": False,
            "creates_reality_pass_rows": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_bop_baseline_candidate_summary(payload)


def validate_objectstate_bop_baseline_candidate_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP baseline candidate summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_BASELINE_CANDIDATE_SCHEMA:
        raise ValueError(
            "unsupported BOP baseline candidate schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_gaussian_centroid_baseline_candidate":
        raise ValueError("BOP baseline candidate kind is unsupported")
    if payload.get("status") != "objectstate_bop_baseline_candidate_written":
        raise ValueError("BOP baseline candidate status is unsupported")
    for key in ("scene_root", "sample_id", "output"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"BOP baseline candidate requires {key}")
    if payload.get("acceptance_schema") != OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA:
        raise ValueError("BOP baseline candidate acceptance schema mismatch")
    if payload.get("target_artifact_schema") != TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA:
        raise ValueError("BOP baseline candidate target artifact schema mismatch")
    policy = payload.get("baseline_policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("baseline_id") != "gaussian_centroid_single_state"
        or not policy.get("uses_per_frame_gaussian_evidence")
        or not policy.get("uses_gaussian_xyz_centroid_and_bbox")
        or not policy.get("single_state_per_frame")
        or not policy.get("expected_to_be_negative_evidence_for_multi_object_identity")
        or not policy.get("does_not_use_bop_pose_gt_for_prediction")
        or not policy.get("does_not_use_bop_object_ids_for_prediction")
        or not policy.get("does_not_train_model")
        or not policy.get("does_not_claim_metric_pass")
    ):
        raise ValueError("BOP baseline candidate must preserve baseline policy")
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("BOP baseline candidate requires row_counts")
    for key in ("frames", "states", "total_gaussians"):
        if not isinstance(row_counts.get(key), int) or row_counts[key] < 0:
            raise ValueError(f"BOP baseline candidate count {key} invalid")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("BOP baseline candidate requires readiness")
    for key in (
        "bop_acceptance_available",
        "phase1_gaussian_evidence_ready",
        "target_artifact_written",
        "target_artifact_valid",
        "ready_for_identity_handoff",
    ):
        if readiness.get(key) is not True:
            raise ValueError(f"BOP baseline candidate readiness {key} must be true")
    acceptance = validate_objectstate_bop_capture_acceptance_summary(
        payload.get("acceptance")
    )
    if not acceptance["readiness"]["phase1_gaussian_evidence_ready"]:
        raise ValueError("BOP baseline candidate requires Gaussian evidence ready")
    frame_records = payload.get("frame_records")
    if not isinstance(frame_records, list) or len(frame_records) != row_counts["frames"]:
        raise ValueError("BOP baseline candidate frame record count mismatch")
    total_states = 0
    total_gaussians = 0
    for record in frame_records:
        _validate_frame_record(record)
        total_states += record["state_count"]
        total_gaussians += record["gaussian_count"]
    if row_counts["states"] != total_states:
        raise ValueError("BOP baseline candidate state count mismatch")
    if row_counts["total_gaussians"] != total_gaussians:
        raise ValueError("BOP baseline candidate Gaussian count mismatch")
    for key in ("next_commands",):
        values = payload.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"BOP baseline candidate {key} must be strings")
    claim_policy = payload.get("claim_policy")
    if not isinstance(claim_policy, Mapping):
        raise ValueError("BOP baseline candidate requires claim policy")
    for key in (
        "writes_baseline_candidate_artifact",
        "requires_existing_bop_scene",
        "requires_existing_gaussian_evidence",
        "uses_gaussian_centroid_only",
        "does_not_download_dataset",
        "does_not_create_ground_truth",
        "does_not_use_pose_gt_for_prediction",
        "does_not_infer_condition_metadata",
        "does_not_reconstruct_gaussians",
        "does_not_train_model",
        "does_not_run_identity_handoff",
        "does_not_run_identity_eval",
        "does_not_claim_reality_gate_pass",
        "does_not_claim_world_model",
    ):
        if not claim_policy.get(key):
            raise ValueError(f"BOP baseline candidate claim policy missing {key}")
    non_goals = payload.get("non_goals")
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP baseline candidate cannot claim downloads, GT, pose-GT prediction, "
            "condition inference, reconstruction, training, identity handoff/eval, "
            "pass rows, public samples, replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def _artifact_from_acceptance(
    acceptance: Mapping[str, Any],
    *,
    scene_root: Path,
    output_path: Path,
    candidate_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = acceptance["manifest"]
    frame_records = []
    object_state_frames = []
    assignments = []
    for frame_index, frame in enumerate(manifest["frames"]):
        gaussian_ref = _gaussian_ref(frame)
        gaussian_path = scene_root / gaussian_ref
        state, record = _state_from_gaussian_file(
            gaussian_path,
            frame_id=frame["frame_id"],
            frame_index=frame_index,
            gaussian_ref=gaussian_ref,
        )
        frame_records.append(record)
        object_state_frames.append(
            {
                "frame_index": frame_index,
                "states": [state],
                "derived_object_ids": [0],
            }
        )
        assignments.append(
            {
                "frame_index": frame_index,
                "shape": [1, 1],
                "matrix": [[1.0]],
            }
        )
    artifact = {
        "schema": TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
        "kind": "trainable_kernel_mvp_model",
        "label": candidate_id,
        "source": {
            "input": str(output_path),
            "scene_root": str(scene_root),
            "sample_id": manifest["sample"]["sample_id"],
            "candidate_source": "bop_gaussian_centroid_single_state_baseline",
            "baseline_policy": "gaussian_xyz_centroid_bbox_only",
        },
        "training": {
            "schema": "objgauss-v1-trainable-kernel-mvp-v1",
            "frame_count": len(object_state_frames),
            "source": "bop_gaussian_centroid_baseline_candidate",
            "optimization_steps": 0,
        },
        "renderer_api": {},
        "learned_parameters": {"decoder_colors": []},
        "assignments": assignments,
        "object_states": object_state_frames,
        "artifact_policy": {
            "baseline_candidate_only": True,
            "not_a_training_run": True,
            "uses_gaussian_evidence_centroid_only": True,
            "does_not_use_bop_pose_gt_for_prediction": True,
            "git_policy": "do_not_commit_training_outputs_by_default",
            "viewer_policy": "not a browser artifact until explicitly converted",
        },
    }
    return artifact, frame_records


def _state_from_gaussian_file(
    gaussian_path: Path,
    *,
    frame_id: str,
    frame_index: int,
    gaussian_ref: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cloud = read_ply(gaussian_path)
    cloud.require_fields(("x", "y", "z"))
    if cloud.count < 1:
        raise ValueError(f"BOP baseline candidate requires non-empty Gaussian PLY: {gaussian_path}")
    xyz = np.column_stack(
        [
            np.asarray(cloud.vertices["x"], dtype=np.float64),
            np.asarray(cloud.vertices["y"], dtype=np.float64),
            np.asarray(cloud.vertices["z"], dtype=np.float64),
        ]
    )
    if not np.isfinite(xyz).all():
        raise ValueError(f"BOP baseline candidate Gaussian PLY has non-finite xyz: {gaussian_path}")
    centroid = xyz.mean(axis=0)
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    state = {
        "id": 0,
        "slot_mass": float(cloud.count),
        "confidence": 1.0,
        "mass_fraction": 1.0,
        "assignment_entropy": 0.0,
        "normalized_assignment_entropy": 0.0,
        "centroid": _round_vector(centroid),
        "bbox": [_round_vector(bbox_min), _round_vector(bbox_max)],
        "feature": _round_vector(centroid),
        "status": "active",
        "diagnostics": ["bop_gaussian_centroid_single_state_baseline"],
    }
    record = {
        "frame_index": int(frame_index),
        "frame_id": str(frame_id),
        "gaussian_ref": str(gaussian_ref),
        "gaussian_path": str(gaussian_path),
        "gaussian_count": int(cloud.count),
        "state_count": 1,
        "centroid": state["centroid"],
        "bbox": state["bbox"],
    }
    return state, record


def _validate_frame_record(record: Any) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("BOP baseline candidate frame record must map")
    if not isinstance(record.get("frame_index"), int):
        raise ValueError("BOP baseline candidate frame record requires frame_index")
    for key in ("frame_id", "gaussian_ref", "gaussian_path"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"BOP baseline candidate frame record requires {key}")
    for key in ("gaussian_count", "state_count"):
        if not isinstance(record.get(key), int) or record[key] < 1:
            raise ValueError(f"BOP baseline candidate frame record invalid {key}")
    _validate_vector(record.get("centroid"), "centroid")
    bbox = record.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 2:
        raise ValueError("BOP baseline candidate bbox must be [[min], [max]]")
    _validate_vector(bbox[0], "bbox min")
    _validate_vector(bbox[1], "bbox max")


def _gaussian_ref(frame: Mapping[str, Any]) -> Path:
    observation = frame.get("observation")
    if not isinstance(observation, Mapping):
        raise ValueError("BOP baseline candidate frame missing observation")
    gaussian = observation.get("gaussian")
    if not isinstance(gaussian, str) or not gaussian:
        raise ValueError("BOP baseline candidate requires Gaussian refs")
    return Path(gaussian)


def _round_vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.round(values.astype(float), 6).tolist()]


def _validate_vector(value: Any, name: str) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"BOP baseline candidate {name} must be length-3 list")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"BOP baseline candidate {name} must contain numbers")
