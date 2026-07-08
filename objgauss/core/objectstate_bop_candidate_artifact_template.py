from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from objgauss.core.objectstate_bop_capture_adapter import (
    BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
    OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
    objectstate_bop_capture_acceptance_summary,
    validate_objectstate_bop_capture_acceptance_summary,
)
from objgauss.core.trainable_artifact import (
    TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
    validate_trainable_kernel_model_artifact,
)

OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SCHEMA = (
    "objgauss-objectstate-bop-candidate-artifact-template-v1"
)
OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SUMMARY_SCHEMA = (
    "objgauss-objectstate-bop-candidate-artifact-template-summary-v1"
)
OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_FINALIZE_SCHEMA = (
    "objgauss-objectstate-bop-candidate-artifact-finalize-v1"
)


def write_objectstate_bop_candidate_artifact_template(
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
    identity_policy: str = BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    pose_track_max_distance_m: float = DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
    candidate_id: str = "bop-objectstate-candidate",
    candidate_source: str = "local-objectstate-model-output",
    target_artifact_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(scene_root)
    output_path = Path(output)
    if output_path.exists() and not force:
        raise FileExistsError(f"BOP candidate artifact template exists: {output_path}")
    acceptance = objectstate_bop_capture_acceptance_summary(
        root,
        sample_id=sample_id,
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
        require_gaussian_files=True,
    )
    target_path = (
        Path(target_artifact_path)
        if target_artifact_path is not None
        else output_path.with_name("objectstates.json")
    )
    template = _template_payload(
        acceptance,
        scene_root=root,
        output_path=output_path,
        target_path=target_path,
        candidate_id=candidate_id,
        candidate_source=candidate_source,
    )
    checked_template = validate_objectstate_bop_candidate_artifact_template(template)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(checked_template, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SUMMARY_SCHEMA,
        "kind": "objectstate_bop_candidate_artifact_template_summary",
        "status": "objectstate_bop_candidate_artifact_template_ready",
        "scene_root": str(root),
        "sample_id": sample_id,
        "output": str(output_path),
        "target_artifact": str(target_path),
        "acceptance_schema": OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
        "template_schema": OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SCHEMA,
        "target_artifact_schema": TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
        "row_counts": {
            "frames": len(checked_template["object_state_frames"]),
            "state_placeholders": sum(
                len(frame["state_placeholders"])
                for frame in checked_template["object_state_frames"]
            ),
        },
        "readiness": {
            "bop_acceptance_available": True,
            "phase1_gaussian_evidence_ready": bool(
                acceptance["readiness"]["phase1_gaussian_evidence_ready"]
            ),
            "template_written": output_path.is_file(),
            "draft_not_valid_for_identity_route": True,
        },
        "acceptance": acceptance,
        "next_commands": [
            (
                "uv run objgauss object-state audit-bop-phase1-local-row "
                f"{root} --output-root {target_path.parent} --sample-id {sample_id} "
                f"--dataset-id {dataset_id} --candidate-artifact {target_path}"
            ),
            (
                "uv run objgauss object-state audit-bop-identity-route "
                f"{root} --output-root {target_path.parent} --sample-id {sample_id} "
                f"--dataset-id {dataset_id} --candidate-artifact {target_path}"
            ),
        ],
        "claim_policy": {
            "authoring_aid_only": True,
            "writes_draft_template": True,
            "template_schema_differs_from_target_artifact_schema": True,
            "template_is_not_valid_for_identity_route": True,
            "omits_pose_gt_values_from_placeholders": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_condition_metadata": True,
            "does_not_generate_model_output": True,
            "does_not_train_model": True,
            "does_not_run_identity_handoff": True,
            "does_not_run_identity_eval": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "creates_ground_truth": False,
            "infers_condition_metadata": False,
            "generates_model_output": False,
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
    return validate_objectstate_bop_candidate_artifact_template_summary(summary)


def finalize_objectstate_bop_candidate_artifact_template(
    template: str | Path | Mapping[str, Any],
    *,
    output: str | Path | None = None,
    scene_root: str | Path | None = None,
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
    identity_policy: str = BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    pose_track_max_distance_m: float = DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
    gt_leakage_tolerance: float = 1e-9,
    reconstruction_noise_robustness: float | None = None,
    reconstruction_noise_variant_count: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    template_path, raw_template = _read_template(template)
    checked_template = validate_objectstate_bop_candidate_artifact_template(raw_template)
    root = Path(scene_root) if scene_root is not None else Path(checked_template["scene_root"])
    output_path = (
        Path(output)
        if output is not None
        else Path(checked_template["target_artifact"])
    )
    if output_path.exists() and not force:
        raise FileExistsError(f"BOP candidate artifact exists: {output_path}")
    tolerance = _non_negative_float(gt_leakage_tolerance, "gt_leakage_tolerance")
    acceptance = objectstate_bop_capture_acceptance_summary(
        root,
        sample_id=checked_template["sample_id"],
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
        require_gaussian_files=True,
    )
    _check_template_acceptance_binding(checked_template, acceptance)
    artifact, leakage_record = _artifact_from_filled_template(
        checked_template,
        acceptance,
        output_path=output_path,
        template_path=template_path,
        gt_leakage_tolerance=tolerance,
        reconstruction_noise_robustness=reconstruction_noise_robustness,
        reconstruction_noise_variant_count=reconstruction_noise_variant_count,
    )
    validate_trainable_kernel_model_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    row_counts = {
        "frames": len(artifact["object_states"]),
        "states": sum(len(frame["states"]) for frame in artifact["object_states"]),
    }
    summary = {
        "schema": OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_FINALIZE_SCHEMA,
        "kind": "objectstate_bop_candidate_artifact_finalize",
        "status": "objectstate_bop_candidate_artifact_finalized",
        "scene_root": str(root),
        "sample_id": checked_template["sample_id"],
        "template": str(template_path) if template_path is not None else "",
        "output": str(output_path),
        "template_schema": OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SCHEMA,
        "target_artifact_schema": TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
        "acceptance_schema": OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
        "row_counts": row_counts,
        "readiness": {
            "phase1_gaussian_evidence_ready": bool(
                acceptance["readiness"]["phase1_gaussian_evidence_ready"]
            ),
            "template_filled": True,
            "target_artifact_written": output_path.is_file(),
            "target_artifact_valid": True,
            "pose_gt_leakage_rejected": bool(leakage_record["checked"]),
            "identity_evidence_present": "identity_evidence" in artifact,
        },
        "pose_gt_leakage_guard": leakage_record,
        "acceptance": acceptance,
        "next_commands": [
            (
                "uv run objgauss object-state audit-bop-phase1-local-row "
                f"{root} --output-root {output_path.parent} "
                f"--sample-id {checked_template['sample_id']} --dataset-id {dataset_id} "
                f"--candidate-artifact {output_path}"
            ),
            (
                "uv run objgauss object-state audit-bop-identity-route "
                f"{root} --output-root {output_path.parent} "
                f"--sample-id {checked_template['sample_id']} --dataset-id {dataset_id} "
                f"--candidate-artifact {output_path}"
            ),
        ],
        "claim_policy": {
            "finalizes_filled_template_only": True,
            "target_schema_matches_identity_route": True,
            "rejects_todo_values": True,
            "rejects_exact_pose_gt_centroid_leakage": True,
            "does_not_generate_model_output": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_train_model": True,
            "does_not_run_identity_handoff": True,
            "does_not_run_identity_eval": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "creates_ground_truth": False,
            "generates_model_output": False,
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
    return validate_objectstate_bop_candidate_artifact_finalize_summary(summary)


def validate_objectstate_bop_candidate_artifact_template(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP candidate artifact template must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SCHEMA:
        raise ValueError(
            "unsupported BOP candidate artifact template schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_candidate_artifact_template":
        raise ValueError("BOP candidate artifact template kind is unsupported")
    if payload.get("template_status") != "draft_not_valid_for_identity_route":
        raise ValueError("BOP candidate artifact template must remain draft-only")
    if payload.get("target_artifact_schema") != TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA:
        raise ValueError("BOP candidate artifact template target schema mismatch")
    if payload.get("target_artifact_schema") == payload.get("schema"):
        raise ValueError("BOP candidate artifact template schema must differ from target")
    for key in ("sample_id", "scene_root", "target_artifact"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"BOP candidate artifact template requires {key}")
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("BOP candidate artifact template requires candidate")
    for key in ("candidate_id", "source"):
        if not isinstance(candidate.get(key), str) or not candidate[key]:
            raise ValueError(f"BOP candidate artifact template candidate requires {key}")
    frames = payload.get("object_state_frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("BOP candidate artifact template requires frames")
    for index, frame in enumerate(frames):
        _validate_template_frame(frame, expected_index=index)
    policy = payload.get("artifact_policy")
    if (
        not isinstance(policy, Mapping)
        or not policy.get("draft_only")
        or not policy.get("pose_gt_values_omitted")
        or not policy.get("fill_from_model_outputs_not_ground_truth")
        or not policy.get("do_not_submit_template_to_identity_route")
    ):
        raise ValueError("BOP candidate artifact template must preserve artifact policy")
    return dict(payload)


def validate_objectstate_bop_candidate_artifact_finalize_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP candidate artifact finalize summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_FINALIZE_SCHEMA:
        raise ValueError(
            "unsupported BOP candidate artifact finalize schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_candidate_artifact_finalize":
        raise ValueError("BOP candidate artifact finalize kind is unsupported")
    if payload.get("status") != "objectstate_bop_candidate_artifact_finalized":
        raise ValueError("BOP candidate artifact finalize status is unsupported")
    for key in ("scene_root", "sample_id", "output"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"BOP candidate artifact finalize requires {key}")
    if payload.get("template_schema") != OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SCHEMA:
        raise ValueError("BOP candidate artifact finalize template schema mismatch")
    if payload.get("target_artifact_schema") != TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA:
        raise ValueError("BOP candidate artifact finalize target schema mismatch")
    if payload.get("acceptance_schema") != OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA:
        raise ValueError("BOP candidate artifact finalize acceptance schema mismatch")
    row_counts = payload.get("row_counts")
    readiness = payload.get("readiness")
    if not isinstance(row_counts, Mapping) or not isinstance(readiness, Mapping):
        raise ValueError("BOP candidate artifact finalize requires counts")
    for key in ("frames", "states"):
        if not isinstance(row_counts.get(key), int) or row_counts[key] <= 0:
            raise ValueError(f"BOP candidate artifact finalize count {key} invalid")
    for key in (
        "phase1_gaussian_evidence_ready",
        "template_filled",
        "target_artifact_written",
        "target_artifact_valid",
        "pose_gt_leakage_rejected",
        "identity_evidence_present",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"BOP candidate artifact finalize readiness {key} invalid")
    validate_objectstate_bop_capture_acceptance_summary(payload.get("acceptance"))
    leakage = payload.get("pose_gt_leakage_guard")
    if not isinstance(leakage, Mapping) or not leakage.get("checked"):
        raise ValueError("BOP candidate artifact finalize requires leakage guard")
    if not isinstance(leakage.get("min_distance_to_pose_gt"), (int, float)):
        raise ValueError("BOP candidate artifact finalize requires leakage distance")
    commands = payload.get("next_commands")
    if not isinstance(commands, list) or not commands or any(
        not isinstance(command, str) or not command for command in commands
    ):
        raise ValueError("BOP candidate artifact finalize requires next commands")
    claim_policy = payload.get("claim_policy")
    non_goals = payload.get("non_goals")
    if not isinstance(claim_policy, Mapping) or not isinstance(non_goals, Mapping):
        raise ValueError("BOP candidate artifact finalize requires policy")
    if (
        not claim_policy.get("finalizes_filled_template_only")
        or not claim_policy.get("target_schema_matches_identity_route")
        or not claim_policy.get("rejects_todo_values")
        or not claim_policy.get("rejects_exact_pose_gt_centroid_leakage")
        or not claim_policy.get("does_not_generate_model_output")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP candidate artifact finalize claim policy is too broad")
    if any(bool(value) for value in non_goals.values()):
        raise ValueError("BOP candidate artifact finalize non_goals cannot claim work")
    return dict(payload)


def validate_objectstate_bop_candidate_artifact_template_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP candidate artifact template summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SUMMARY_SCHEMA:
        raise ValueError(
            "unsupported BOP candidate artifact template summary schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_candidate_artifact_template_summary":
        raise ValueError("BOP candidate artifact template summary kind is unsupported")
    if payload.get("status") != "objectstate_bop_candidate_artifact_template_ready":
        raise ValueError("BOP candidate artifact template summary status is unsupported")
    for key in ("scene_root", "sample_id", "output", "target_artifact"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"BOP candidate artifact template summary requires {key}")
    if payload.get("acceptance_schema") != OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA:
        raise ValueError("BOP candidate artifact template acceptance schema mismatch")
    if payload.get("template_schema") != OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SCHEMA:
        raise ValueError("BOP candidate artifact template summary template schema mismatch")
    if payload.get("target_artifact_schema") != TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA:
        raise ValueError("BOP candidate artifact template summary target schema mismatch")
    row_counts = payload.get("row_counts")
    readiness = payload.get("readiness")
    if not isinstance(row_counts, Mapping) or not isinstance(readiness, Mapping):
        raise ValueError("BOP candidate artifact template summary requires counts")
    for key in ("frames", "state_placeholders"):
        if not isinstance(row_counts.get(key), int) or row_counts[key] < 0:
            raise ValueError(f"BOP candidate artifact template count {key} invalid")
    for key in (
        "bop_acceptance_available",
        "phase1_gaussian_evidence_ready",
        "template_written",
        "draft_not_valid_for_identity_route",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"BOP candidate artifact template readiness {key} invalid")
    validate_objectstate_bop_capture_acceptance_summary(payload.get("acceptance"))
    commands = payload.get("next_commands")
    if not isinstance(commands, list) or not commands or any(
        not isinstance(command, str) or not command for command in commands
    ):
        raise ValueError("BOP candidate artifact template requires next commands")
    claim_policy = payload.get("claim_policy")
    non_goals = payload.get("non_goals")
    if not isinstance(claim_policy, Mapping) or not isinstance(non_goals, Mapping):
        raise ValueError("BOP candidate artifact template summary requires policy")
    if (
        not claim_policy.get("authoring_aid_only")
        or not claim_policy.get("template_schema_differs_from_target_artifact_schema")
        or not claim_policy.get("template_is_not_valid_for_identity_route")
        or not claim_policy.get("omits_pose_gt_values_from_placeholders")
        or not claim_policy.get("does_not_generate_model_output")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP candidate artifact template claim policy is too broad")
    if any(bool(value) for value in non_goals.values()):
        raise ValueError("BOP candidate artifact template non_goals cannot claim work")
    return dict(payload)


def _template_payload(
    acceptance: Mapping[str, Any],
    *,
    scene_root: Path,
    output_path: Path,
    target_path: Path,
    candidate_id: str,
    candidate_source: str,
) -> dict[str, Any]:
    manifest = acceptance["manifest"]
    frames = [
        _template_frame(index, frame)
        for index, frame in enumerate(manifest["frames"])
    ]
    return {
        "schema": OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SCHEMA,
        "kind": "objectstate_bop_candidate_artifact_template",
        "template_status": "draft_not_valid_for_identity_route",
        "target_artifact_schema": TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
        "sample_id": manifest["sample"]["sample_id"],
        "scene_root": str(scene_root),
        "template_path": str(output_path),
        "target_artifact": str(target_path),
        "candidate": {
            "candidate_id": _required_string(candidate_id, "candidate_id"),
            "source": _required_string(candidate_source, "candidate_source"),
        },
        "target_artifact_shape": {
            "required_schema": TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
            "required_kind": "trainable_kernel_mvp_model",
            "required_training_schema": "objgauss-v1-trainable-kernel-mvp-v1",
            "frame_count": len(frames),
            "required_top_level_keys": [
                "schema",
                "kind",
                "label",
                "source",
                "training",
                "learned_parameters",
                "assignments",
                "object_states",
                "artifact_policy",
            ],
        },
        "object_state_frames": frames,
        "artifact_policy": {
            "draft_only": True,
            "pose_gt_values_omitted": True,
            "fill_from_model_outputs_not_ground_truth": True,
            "do_not_submit_template_to_identity_route": True,
            "target_artifact_should_live_under_ignored_outputs": True,
        },
        "non_goals": {
            "creates_valid_trainable_artifact": False,
            "copies_pose_gt_into_centroids": False,
            "runs_model": False,
            "runs_identity_eval": False,
            "creates_pass_row": False,
        },
    }


def _template_frame(index: int, frame: Mapping[str, Any]) -> dict[str, Any]:
    observation = frame["observation"]
    objects = frame["objects"]
    return {
        "frame_index": index,
        "frame_id": frame["frame_id"],
        "gaussian_ref": observation.get("gaussian"),
        "object_ids": [item["object_id"] for item in objects],
        "state_placeholders": [
            {
                "object_id": item["object_id"],
                "state_id": "TODO integer candidate slot id",
                "centroid": "TODO model-predicted ObjectState centroid [x,y,z]",
                "bbox": "TODO model-predicted ObjectState bbox [[min],[max]]",
                "confidence": "TODO model confidence in [0,1]",
                "note": "Do not copy BOP pose GT values into this template.",
            }
            for item in objects
        ],
    }


def _validate_template_frame(frame: Any, *, expected_index: int) -> None:
    if not isinstance(frame, Mapping):
        raise ValueError("BOP candidate artifact template frame must be a mapping")
    if frame.get("frame_index") != expected_index:
        raise ValueError("BOP candidate artifact template frame_index mismatch")
    for key in ("frame_id", "gaussian_ref"):
        if not isinstance(frame.get(key), str) or not frame[key]:
            raise ValueError(f"BOP candidate artifact template frame requires {key}")
    object_ids = frame.get("object_ids")
    placeholders = frame.get("state_placeholders")
    if not isinstance(object_ids, list) or not object_ids:
        raise ValueError("BOP candidate artifact template frame requires object ids")
    if not isinstance(placeholders, list) or len(placeholders) != len(object_ids):
        raise ValueError("BOP candidate artifact template placeholder count mismatch")
    for item in placeholders:
        if not isinstance(item, Mapping):
            raise ValueError("BOP candidate artifact template placeholder must be mapping")
        if item.get("object_id") not in object_ids:
            raise ValueError("BOP candidate artifact template placeholder object mismatch")
        for key in ("state_id", "centroid", "bbox", "confidence"):
            if key not in item:
                raise ValueError(
                    f"BOP candidate artifact template placeholder requires {key}"
                )
        note = item.get("note")
        if not isinstance(note, str) or not note:
            raise ValueError("BOP candidate artifact template placeholder requires note")
        serialized = json.dumps(item)
        if "pose.position" in serialized or "target_position" in serialized:
            raise ValueError("BOP candidate artifact template must omit GT pose values")


def _read_template(
    template: str | Path | Mapping[str, Any],
) -> tuple[Path | None, Mapping[str, Any]]:
    if isinstance(template, Mapping):
        return None, template
    path = Path(template)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("BOP candidate artifact template JSON must be an object")
    return path, payload


def _check_template_acceptance_binding(
    template: Mapping[str, Any],
    acceptance: Mapping[str, Any],
) -> None:
    if acceptance["status"] != "objectstate_bop_capture_acceptance_pass":
        raise ValueError("BOP candidate artifact finalize requires BOP acceptance pass")
    frames = template["object_state_frames"]
    capture_frames = acceptance["manifest"]["frames"]
    if len(frames) != len(capture_frames):
        raise ValueError("BOP candidate artifact template frame count mismatch")
    for index, (template_frame, capture_frame) in enumerate(zip(frames, capture_frames)):
        if template_frame["frame_index"] != index:
            raise ValueError("BOP candidate artifact template frame_index mismatch")
        if template_frame["frame_id"] != capture_frame["frame_id"]:
            raise ValueError("BOP candidate artifact template frame_id mismatch")
        expected_gaussian = capture_frame["observation"].get("gaussian")
        if template_frame["gaussian_ref"] != expected_gaussian:
            raise ValueError("BOP candidate artifact template gaussian_ref mismatch")
        expected_objects = [item["object_id"] for item in capture_frame["objects"]]
        if list(template_frame["object_ids"]) != expected_objects:
            raise ValueError("BOP candidate artifact template object id mismatch")


def _artifact_from_filled_template(
    template: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    *,
    output_path: Path,
    template_path: Path | None,
    gt_leakage_tolerance: float,
    reconstruction_noise_robustness: float | None,
    reconstruction_noise_variant_count: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    object_state_frames = []
    assignments = []
    min_gt_distance: float | None = None
    leak_hits: list[dict[str, Any]] = []
    for frame_index, (template_frame, capture_frame) in enumerate(
        zip(template["object_state_frames"], acceptance["manifest"]["frames"])
    ):
        gt_positions = {
            item["object_id"]: _numeric_vector(
                item["pose"]["position"],
                f"BOP capture frame {frame_index} object pose.position",
            )
            for item in capture_frame["objects"]
        }
        states = []
        state_ids = set()
        for placeholder in template_frame["state_placeholders"]:
            state = _state_from_placeholder(
                placeholder,
                frame_index=frame_index,
                gt_positions=gt_positions,
            )
            if state["id"] in state_ids:
                raise ValueError(
                    "BOP candidate artifact finalizer requires unique state_id per frame"
                )
            state_ids.add(state["id"])
            distance = _distance(state["centroid"], gt_positions[placeholder["object_id"]])
            min_gt_distance = (
                distance
                if min_gt_distance is None
                else min(min_gt_distance, distance)
            )
            if distance <= gt_leakage_tolerance:
                leak_hits.append(
                    {
                        "frame_index": frame_index,
                        "frame_id": template_frame["frame_id"],
                        "object_id": placeholder["object_id"],
                        "state_id": state["id"],
                        "distance": distance,
                    }
                )
            states.append(state)
        if leak_hits:
            first = leak_hits[0]
            raise ValueError(
                "BOP candidate artifact finalizer detected pose GT centroid leakage "
                f"at frame {first['frame_id']} object {first['object_id']}"
            )
        object_state_frames.append(
            {
                "frame_index": frame_index,
                "states": states,
                "derived_object_ids": [int(state["id"]) for state in states],
            }
        )
        assignments.append(_identity_assignment(frame_index, len(states)))
    artifact = {
        "schema": TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
        "kind": "trainable_kernel_mvp_model",
        "label": template["candidate"]["candidate_id"],
        "source": {
            "input": str(output_path),
            "template": str(template_path) if template_path is not None else None,
            "sample_id": template["sample_id"],
            "candidate_source": template["candidate"]["source"],
        },
        "training": {
            "schema": "objgauss-v1-trainable-kernel-mvp-v1",
            "frame_count": len(object_state_frames),
            "source": "bop_candidate_artifact_finalizer",
            "optimization_steps": 0,
        },
        "renderer_api": {},
        "learned_parameters": {"decoder_colors": []},
        "assignments": assignments,
        "object_states": object_state_frames,
        "artifact_policy": {
            "candidate_packaging_only": True,
            "finalized_from_filled_bop_template": True,
            "not_a_training_run": True,
            "pose_gt_leakage_guard_applied": True,
            "git_policy": "do_not_commit_training_outputs_by_default",
            "viewer_policy": "not a browser artifact until explicitly converted",
        },
    }
    identity_evidence = _identity_evidence(
        reconstruction_noise_robustness,
        reconstruction_noise_variant_count,
    )
    if identity_evidence is not None:
        artifact["identity_evidence"] = identity_evidence
    leakage_record = {
        "checked": True,
        "tolerance": gt_leakage_tolerance,
        "min_distance_to_pose_gt": float(min_gt_distance or 0.0),
        "leak_count": 0,
    }
    return artifact, leakage_record


def _state_from_placeholder(
    placeholder: Mapping[str, Any],
    *,
    frame_index: int,
    gt_positions: Mapping[str, list[float]],
) -> dict[str, Any]:
    _reject_todo(placeholder, f"frame {frame_index} placeholder")
    object_id = placeholder["object_id"]
    if object_id not in gt_positions:
        raise ValueError("BOP candidate artifact placeholder object missing in GT map")
    state_id = _integer(placeholder["state_id"], "state_id")
    centroid = _numeric_vector(placeholder["centroid"], "centroid")
    bbox = _bbox(placeholder["bbox"])
    confidence = _unit_float(placeholder["confidence"], "confidence")
    return {
        "id": state_id,
        "slot_mass": confidence,
        "confidence": confidence,
        "mass_fraction": 1.0 / max(1, len(gt_positions)),
        "assignment_entropy": 0.0,
        "normalized_assignment_entropy": 0.0,
        "centroid": centroid,
        "bbox": bbox,
        "feature": centroid,
        "status": "active",
        "diagnostics": ["bop_candidate_finalized_from_template"],
    }


def _identity_assignment(frame_index: int, state_count: int) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "shape": [state_count, state_count],
        "matrix": [
            [
                1.0 if row == column else 0.0
                for column in range(state_count)
            ]
            for row in range(state_count)
        ],
    }


def _identity_evidence(
    robustness: float | None,
    variant_count: int | None,
) -> dict[str, Any] | None:
    if robustness is None and variant_count is None:
        return None
    if robustness is None or variant_count is None:
        raise ValueError(
            "BOP candidate artifact identity evidence requires robustness and variant count"
        )
    count = _integer(variant_count, "reconstruction_noise_variant_count")
    if count < 1:
        raise ValueError("reconstruction_noise_variant_count must be >= 1")
    return {
        "reconstruction_noise_robustness": _unit_float(
            robustness,
            "reconstruction_noise_robustness",
        ),
        "reconstruction_noise_variant_count": count,
    }


def _reject_todo(value: Any, name: str) -> None:
    serialized = json.dumps(value, sort_keys=True)
    if "TODO" in serialized:
        raise ValueError(f"{name} still contains TODO values")


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _unit_float(value: Any, name: str) -> float:
    number = _float(value, name)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _non_negative_float(value: Any, name: str) -> float:
    number = _float(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return number


def _float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return number


def _numeric_vector(value: Any, name: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise TypeError(f"{name} must be a length-3 numeric list")
    if len(value) != 3:
        raise ValueError(f"{name} must have length 3")
    return [_float(item, name) for item in value]


def _bbox(value: Any) -> list[list[float]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise TypeError("bbox must be [[min],[max]]")
    if len(value) != 2:
        raise ValueError("bbox must have two corners")
    corners = [_numeric_vector(item, "bbox corner") for item in value]
    for axis, (low, high) in enumerate(zip(corners[0], corners[1])):
        if low > high:
            raise ValueError(f"bbox min corner exceeds max corner at axis {axis}")
    return corners


def _distance(left: list[float], right: list[float]) -> float:
    return float(
        sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)) ** 0.5
    )


def _required_string(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value
