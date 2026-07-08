from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from objgauss.core.objectstate_bop_capture_adapter import (
    OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
    objectstate_bop_capture_acceptance_summary,
    validate_objectstate_bop_capture_acceptance_summary,
)
from objgauss.core.trainable_artifact import TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA

OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SCHEMA = (
    "objgauss-objectstate-bop-candidate-artifact-template-v1"
)
OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SUMMARY_SCHEMA = (
    "objgauss-objectstate-bop-candidate-artifact-template-summary-v1"
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
        for key in ("state_id", "centroid", "bbox", "confidence", "note"):
            value = item.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"BOP candidate artifact template placeholder requires {key}"
                )
        serialized = json.dumps(item)
        if "pose.position" in serialized or "target_position" in serialized:
            raise ValueError("BOP candidate artifact template must omit GT pose values")


def _required_string(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value
