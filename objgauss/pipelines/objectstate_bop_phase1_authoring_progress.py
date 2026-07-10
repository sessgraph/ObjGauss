from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from objgauss.pipelines.objectstate_bop_candidate_artifact_template import (
    validate_objectstate_bop_candidate_artifact_template,
)
from objgauss.datasets.objectstate_bop_capture_adapter import (
    validate_objectstate_bop_capture_condition_sidecar,
)
from objgauss.datasets.objectstate_bop_local_row_batch_spec import (
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
    read_objectstate_bop_local_row_batch_spec,
    validate_objectstate_bop_local_row_batch_spec,
)
from objgauss.pipelines.trainable_artifact import (
    TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
    validate_trainable_kernel_model_artifact,
)

OBJECTSTATE_BOP_PHASE1_AUTHORING_PROGRESS_SCHEMA = (
    "objgauss-objectstate-bop-phase1-authoring-progress-v1"
)

Validator = Callable[[Any], Any]


def objectstate_bop_phase1_authoring_progress(
    batch_spec: str | Path | Mapping[str, Any],
    *,
    min_gaussian_bytes: int = 1,
) -> dict[str, Any]:
    spec_path: Path | None = None
    if isinstance(batch_spec, (str, Path)):
        spec_path = Path(batch_spec)
        spec = read_objectstate_bop_local_row_batch_spec(spec_path)
    else:
        spec = validate_objectstate_bop_local_row_batch_spec(batch_spec)
    spec_root = spec_path.parent if spec_path is not None else Path.cwd()
    defaults = spec.get("defaults", {})
    min_bytes = _non_negative_int(min_gaussian_bytes, "min_gaussian_bytes")

    sample_records = [
        _sample_record(
            index,
            sample,
            defaults=defaults,
            spec_root=spec_root,
            spec_path=spec_path,
            min_gaussian_bytes=min_bytes,
        )
        for index, sample in enumerate(spec["samples"])
    ]
    row_counts = _row_counts(sample_records)
    readiness = _readiness(sample_records)
    payload = {
        "schema": OBJECTSTATE_BOP_PHASE1_AUTHORING_PROGRESS_SCHEMA,
        "kind": "objectstate_bop_phase1_authoring_progress",
        "status": (
            "objectstate_bop_phase1_authoring_ready_for_batch_readiness"
            if readiness["all_samples_ready_for_batch_readiness_input"]
            else "objectstate_bop_phase1_authoring_in_progress"
        ),
        "batch_spec_schema": OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
        "target_artifact_schema": TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
        "batch_spec": str(spec_path) if spec_path is not None else "",
        "spec_root": str(spec_root),
        "requirements": {
            "min_gaussian_bytes": min_bytes,
            "candidate_artifact_schema": TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
            "target_sidecar_schema": (
                "objgauss-objectstate-bop-capture-condition-sidecar-v1"
            ),
        },
        "row_counts": row_counts,
        "readiness": readiness,
        "sample_records": sample_records,
        "sample_table_markdown": _sample_table_markdown(sample_records),
        "hard_blockers": _hard_blockers(sample_records, readiness),
        "issues": _issues(sample_records),
        "next_actions": _next_actions(sample_records, readiness),
        "next_commands": _batch_next_commands(spec_path, readiness),
        "claim_policy": {
            "read_only_authoring_progress_audit": True,
            "uses_explicit_batch_spec": True,
            "checks_sample_workspace_helpers": True,
            "checks_target_condition_sidecar": True,
            "checks_per_frame_gaussian_evidence_paths": True,
            "checks_target_candidate_artifact_schema": True,
            "does_not_download_dataset": True,
            "does_not_copy_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_condition_metadata": True,
            "does_not_create_condition_sidecar": True,
            "does_not_create_candidate_artifact": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_run_readiness": True,
            "does_not_run_handoff": True,
            "does_not_run_identity_eval": True,
            "does_not_run_prediction_eval": True,
            "does_not_train_model": True,
            "does_not_claim_metric_pass": True,
            "does_not_claim_intervention_gate": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "copies_dataset": False,
            "creates_ground_truth": False,
            "infers_condition_metadata": False,
            "creates_condition_sidecar": False,
            "creates_candidate_artifact": False,
            "reconstructs_gaussians": False,
            "runs_readiness": False,
            "runs_handoff": False,
            "runs_identity_eval": False,
            "runs_prediction_eval": False,
            "runs_intervention_model": False,
            "trains_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_bop_phase1_authoring_progress_summary(payload)


def validate_objectstate_bop_phase1_authoring_progress_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP Phase 1 authoring progress summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_PHASE1_AUTHORING_PROGRESS_SCHEMA:
        raise ValueError(
            "unsupported BOP Phase 1 authoring progress schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_phase1_authoring_progress":
        raise ValueError("BOP Phase 1 authoring progress kind is unsupported")
    if payload.get("status") not in {
        "objectstate_bop_phase1_authoring_ready_for_batch_readiness",
        "objectstate_bop_phase1_authoring_in_progress",
    }:
        raise ValueError("BOP Phase 1 authoring progress status is unsupported")
    if payload.get("batch_spec_schema") != OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA:
        raise ValueError("BOP Phase 1 authoring progress batch spec schema mismatch")
    if payload.get("target_artifact_schema") != TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA:
        raise ValueError("BOP Phase 1 authoring progress target artifact schema mismatch")
    if not isinstance(payload.get("spec_root"), str) or not payload["spec_root"]:
        raise ValueError("BOP Phase 1 authoring progress requires spec_root")
    requirements = payload.get("requirements")
    if not isinstance(requirements, Mapping):
        raise ValueError("BOP Phase 1 authoring progress requires requirements")
    _non_negative_int(requirements.get("min_gaussian_bytes"), "min_gaussian_bytes")
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("BOP Phase 1 authoring progress requires row counts")
    for key in (
        "samples",
        "scene_roots_present",
        "sample_workspace_helpers_present",
        "target_condition_sidecars_present",
        "target_condition_sidecars_valid",
        "target_candidate_artifacts_present",
        "target_candidate_artifacts_valid",
        "selected_frames",
        "expected_gaussian_files",
        "present_gaussian_files",
        "missing_gaussian_files",
        "samples_ready_for_batch_readiness_input",
    ):
        if not isinstance(row_counts.get(key), int):
            raise ValueError(f"BOP Phase 1 authoring progress count {key} invalid")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("BOP Phase 1 authoring progress requires readiness")
    for key in (
        "sample_count_nonzero",
        "all_scene_roots_present",
        "all_sample_workspace_helpers_present",
        "all_target_condition_sidecars_valid",
        "all_gaussian_evidence_present",
        "all_target_candidate_artifacts_valid",
        "all_samples_ready_for_batch_readiness_input",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"BOP Phase 1 authoring progress readiness {key} invalid")
    records = payload.get("sample_records")
    if not isinstance(records, list) or len(records) != row_counts["samples"]:
        raise ValueError("BOP Phase 1 authoring progress record count mismatch")
    for record in records:
        _validate_sample_record(record)
    if row_counts != _row_counts(records):
        raise ValueError("BOP Phase 1 authoring progress row counts mismatch")
    expected_readiness = _readiness(records)
    if dict(readiness) != expected_readiness:
        raise ValueError("BOP Phase 1 authoring progress readiness mismatch")
    expected_status = (
        "objectstate_bop_phase1_authoring_ready_for_batch_readiness"
        if readiness["all_samples_ready_for_batch_readiness_input"]
        else "objectstate_bop_phase1_authoring_in_progress"
    )
    if payload["status"] != expected_status:
        raise ValueError("BOP Phase 1 authoring progress status mismatch")
    if not isinstance(payload.get("sample_table_markdown"), str):
        raise ValueError("BOP Phase 1 authoring progress requires sample table")
    for key in ("hard_blockers", "issues", "next_actions", "next_commands"):
        values = payload.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"BOP Phase 1 authoring progress {key} must be strings")
    claim_policy = payload.get("claim_policy")
    if not isinstance(claim_policy, Mapping):
        raise ValueError("BOP Phase 1 authoring progress requires claim policy")
    for key in (
        "read_only_authoring_progress_audit",
        "uses_explicit_batch_spec",
        "checks_sample_workspace_helpers",
        "checks_target_condition_sidecar",
        "checks_per_frame_gaussian_evidence_paths",
        "checks_target_candidate_artifact_schema",
        "does_not_download_dataset",
        "does_not_copy_dataset",
        "does_not_create_ground_truth",
        "does_not_infer_condition_metadata",
        "does_not_create_condition_sidecar",
        "does_not_create_candidate_artifact",
        "does_not_reconstruct_gaussians",
        "does_not_run_readiness",
        "does_not_run_handoff",
        "does_not_run_identity_eval",
        "does_not_run_prediction_eval",
        "does_not_train_model",
        "does_not_claim_metric_pass",
        "does_not_claim_intervention_gate",
        "does_not_claim_world_model",
    ):
        if not claim_policy.get(key):
            raise ValueError(f"BOP Phase 1 authoring progress policy missing {key}")
    non_goals = payload.get("non_goals")
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP Phase 1 authoring progress cannot claim downloads, copies, GT, "
            "condition inference, target file creation, reconstruction, readiness, "
            "handoff, eval, training, public samples, replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def _sample_record(
    index: int,
    sample: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    spec_root: Path,
    spec_path: Path | None,
    min_gaussian_bytes: int,
) -> dict[str, Any]:
    merged = {**defaults, **sample}
    sample_id = _required_string(sample.get("sample_id"), "sample_id")
    scene_root = _resolve_path(sample["scene_root"], spec_root)
    candidate_artifact = _resolve_path(sample["candidate_artifact"], spec_root)
    condition_sidecar = (
        _resolve_path(merged["condition_sidecar"], spec_root)
        if merged.get("condition_sidecar")
        else candidate_artifact.with_name("bop-condition-sidecar.json")
    )
    authoring_root = candidate_artifact.parent
    condition_csv_template = authoring_root / "bop-conditions.template.csv"
    condition_sidecar_draft = authoring_root / "bop-condition-sidecar.draft.json"
    objectstate_template = authoring_root / "objectstates.template.json"
    readme = authoring_root / "README.md"
    gaussian_dir = str(merged.get("gaussian_dir", "gaussians"))
    depth_dir = str(merged.get("depth_dir", "depth"))
    max_frames = _optional_int(merged.get("max_frames"))
    frame_step = _positive_int(merged.get("frame_step", 1), "frame_step")
    frame_ids, frame_issues = _selected_frame_ids(
        scene_root,
        max_frames=max_frames,
        frame_step=frame_step,
    )
    gaussian_records = _gaussian_records(
        scene_root,
        frame_ids,
        gaussian_dir=gaussian_dir,
        min_gaussian_bytes=min_gaussian_bytes,
    )
    depth_records = _frame_file_records(scene_root, frame_ids, directory=depth_dir, suffix=".png")
    sidecar_valid, sidecar_issue = _validate_json_file(
        condition_sidecar,
        validate_objectstate_bop_capture_condition_sidecar,
        "target condition sidecar",
    )
    template_valid, template_issue = _validate_json_file(
        objectstate_template,
        validate_objectstate_bop_candidate_artifact_template,
        "ObjectState template draft",
    )
    artifact_valid, artifact_issue = _validate_json_file(
        candidate_artifact,
        validate_trainable_kernel_model_artifact,
        "target ObjectState candidate artifact",
    )
    helper_readiness = {
        "condition_csv_template_present": condition_csv_template.is_file(),
        "condition_sidecar_draft_present": condition_sidecar_draft.is_file(),
        "readme_present": readme.is_file(),
    }
    selected_count = len(frame_ids)
    present_gaussians = sum(1 for record in gaussian_records if record["valid"])
    missing_gaussians = len(gaussian_records) - present_gaussians
    complete_depth = bool(depth_records) and all(record["valid"] for record in depth_records)
    readiness = {
        "scene_root_present": scene_root.is_dir(),
        "scene_frame_index_available": bool(frame_ids),
        **helper_readiness,
        "sample_workspace_helpers_present": all(helper_readiness.values()),
        "target_condition_sidecar_present": condition_sidecar.is_file(),
        "target_condition_sidecar_valid": sidecar_valid,
        "gaussian_evidence_expected": bool(gaussian_records),
        "gaussian_evidence_present": bool(gaussian_records) and missing_gaussians == 0,
        "depth_export_candidate": bool(gaussian_records)
        and missing_gaussians > 0
        and complete_depth,
        "objectstate_template_present": objectstate_template.is_file(),
        "objectstate_template_valid": template_valid,
        "target_candidate_artifact_present": candidate_artifact.is_file(),
        "target_candidate_artifact_valid": artifact_valid,
    }
    readiness["ready_for_batch_readiness_input"] = bool(
        readiness["scene_root_present"]
        and readiness["scene_frame_index_available"]
        and readiness["target_condition_sidecar_valid"]
        and readiness["gaussian_evidence_present"]
        and readiness["target_candidate_artifact_valid"]
    )
    issues = _sample_issues(
        readiness,
        frame_issues=frame_issues,
        validation_issues=[sidecar_issue, template_issue, artifact_issue],
        missing_gaussians=missing_gaussians,
    )
    paths = {
        "scene_root": str(scene_root),
        "authoring_root": str(authoring_root),
        "condition_csv_template": str(condition_csv_template),
        "condition_sidecar_draft": str(condition_sidecar_draft),
        "target_condition_sidecar": str(condition_sidecar),
        "objectstate_template": str(objectstate_template),
        "target_candidate_artifact": str(candidate_artifact),
        "readme": str(readme),
        "gaussian_dir": str(scene_root / gaussian_dir),
        "depth_dir": str(scene_root / depth_dir),
    }
    return {
        "sample_id": sample_id,
        "index": int(index),
        "paths": paths,
        "options": {
            "dataset_id": str(merged.get("dataset_id", "bop-ycbv")),
            "object_category": str(merged.get("object_category", "bop_objects")),
            "scenario": str(merged.get("scenario", "bop_pose_sequence")),
            "gaussian_dir": gaussian_dir,
            "depth_dir": depth_dir,
            "max_frames": max_frames,
            "frame_step": frame_step,
            "min_gaussian_bytes": min_gaussian_bytes,
        },
        "selected_frame_ids": [int(frame_id) for frame_id in frame_ids],
        "row_counts": {
            "selected_frames": selected_count,
            "expected_gaussian_files": len(gaussian_records),
            "present_gaussian_files": present_gaussians,
            "missing_gaussian_files": missing_gaussians,
            "depth_files_present": sum(1 for record in depth_records if record["valid"]),
        },
        "expected_gaussian_files": gaussian_records,
        "missing_gaussian_files": [
            record for record in gaussian_records if not record["valid"]
        ],
        "readiness": readiness,
        "issues": issues,
        "next_commands": _sample_next_commands(
            spec_path=spec_path,
            sample_id=sample_id,
            paths=paths,
            options={
                "dataset_id": str(merged.get("dataset_id", "bop-ycbv")),
                "object_category": str(merged.get("object_category", "bop_objects")),
                "scenario": str(merged.get("scenario", "bop_pose_sequence")),
                "max_frames": max_frames,
                "frame_step": frame_step,
            },
            readiness=readiness,
        ),
    }


def _validate_sample_record(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("BOP Phase 1 authoring progress sample record must map")
    if not isinstance(record.get("sample_id"), str) or not record["sample_id"]:
        raise ValueError("BOP Phase 1 authoring progress sample requires sample_id")
    if not isinstance(record.get("index"), int):
        raise ValueError("BOP Phase 1 authoring progress sample requires index")
    paths = record.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("BOP Phase 1 authoring progress sample requires paths")
    for key in (
        "scene_root",
        "authoring_root",
        "condition_csv_template",
        "condition_sidecar_draft",
        "target_condition_sidecar",
        "objectstate_template",
        "target_candidate_artifact",
        "readme",
        "gaussian_dir",
        "depth_dir",
    ):
        if not isinstance(paths.get(key), str) or not paths[key]:
            raise ValueError(f"BOP Phase 1 authoring progress path {key} invalid")
    selected = record.get("selected_frame_ids")
    if not isinstance(selected, list) or any(not isinstance(value, int) for value in selected):
        raise ValueError("BOP Phase 1 authoring progress selected frames invalid")
    counts = record.get("row_counts")
    if not isinstance(counts, Mapping):
        raise ValueError("BOP Phase 1 authoring progress sample requires counts")
    for key in (
        "selected_frames",
        "expected_gaussian_files",
        "present_gaussian_files",
        "missing_gaussian_files",
        "depth_files_present",
    ):
        if not isinstance(counts.get(key), int):
            raise ValueError(f"BOP Phase 1 authoring progress sample count {key} invalid")
    readiness = record.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("BOP Phase 1 authoring progress sample requires readiness")
    for key in (
        "scene_root_present",
        "scene_frame_index_available",
        "condition_csv_template_present",
        "condition_sidecar_draft_present",
        "readme_present",
        "sample_workspace_helpers_present",
        "target_condition_sidecar_present",
        "target_condition_sidecar_valid",
        "gaussian_evidence_expected",
        "gaussian_evidence_present",
        "depth_export_candidate",
        "objectstate_template_present",
        "objectstate_template_valid",
        "target_candidate_artifact_present",
        "target_candidate_artifact_valid",
        "ready_for_batch_readiness_input",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(
                f"BOP Phase 1 authoring progress sample readiness {key} invalid"
            )
    for key in ("expected_gaussian_files", "missing_gaussian_files"):
        values = record.get(key)
        if not isinstance(values, list):
            raise ValueError(f"BOP Phase 1 authoring progress sample {key} invalid")
        for value in values:
            _validate_file_record(value)
    for key in ("issues", "next_commands"):
        values = record.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"BOP Phase 1 authoring progress sample {key} invalid")
    expected_missing = [
        value for value in record["expected_gaussian_files"] if not value["valid"]
    ]
    if record["missing_gaussian_files"] != expected_missing:
        raise ValueError("BOP Phase 1 authoring progress missing files mismatch")
    if counts["selected_frames"] != len(selected):
        raise ValueError("BOP Phase 1 authoring progress selected frame count mismatch")
    if counts["expected_gaussian_files"] != len(record["expected_gaussian_files"]):
        raise ValueError("BOP Phase 1 authoring progress Gaussian count mismatch")
    if counts["missing_gaussian_files"] != len(record["missing_gaussian_files"]):
        raise ValueError("BOP Phase 1 authoring progress missing count mismatch")


def _validate_file_record(record: Any) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("BOP Phase 1 authoring progress file record must map")
    for key in ("frame_id", "ref", "path"):
        if not isinstance(record.get(key), str):
            raise ValueError(f"BOP Phase 1 authoring progress file record missing {key}")
    for key in ("exists", "is_file", "valid"):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"BOP Phase 1 authoring progress file record {key} invalid")
    size = record.get("size_bytes")
    if size is not None and not isinstance(size, int):
        raise ValueError("BOP Phase 1 authoring progress file record size invalid")
    reason = record.get("missing_reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("BOP Phase 1 authoring progress missing reason invalid")


def _row_counts(records: list[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "samples": len(records),
        "scene_roots_present": sum(
            1 for record in records if record["readiness"]["scene_root_present"]
        ),
        "sample_workspace_helpers_present": sum(
            1
            for record in records
            if record["readiness"]["sample_workspace_helpers_present"]
        ),
        "target_condition_sidecars_present": sum(
            1
            for record in records
            if record["readiness"]["target_condition_sidecar_present"]
        ),
        "target_condition_sidecars_valid": sum(
            1
            for record in records
            if record["readiness"]["target_condition_sidecar_valid"]
        ),
        "target_candidate_artifacts_present": sum(
            1
            for record in records
            if record["readiness"]["target_candidate_artifact_present"]
        ),
        "target_candidate_artifacts_valid": sum(
            1
            for record in records
            if record["readiness"]["target_candidate_artifact_valid"]
        ),
        "selected_frames": sum(record["row_counts"]["selected_frames"] for record in records),
        "expected_gaussian_files": sum(
            record["row_counts"]["expected_gaussian_files"] for record in records
        ),
        "present_gaussian_files": sum(
            record["row_counts"]["present_gaussian_files"] for record in records
        ),
        "missing_gaussian_files": sum(
            record["row_counts"]["missing_gaussian_files"] for record in records
        ),
        "samples_ready_for_batch_readiness_input": sum(
            1
            for record in records
            if record["readiness"]["ready_for_batch_readiness_input"]
        ),
    }


def _readiness(records: list[Mapping[str, Any]]) -> dict[str, bool]:
    sample_count = len(records)
    return {
        "sample_count_nonzero": bool(records),
        "all_scene_roots_present": bool(records)
        and all(record["readiness"]["scene_root_present"] for record in records),
        "all_sample_workspace_helpers_present": bool(records)
        and all(
            record["readiness"]["sample_workspace_helpers_present"]
            for record in records
        ),
        "all_target_condition_sidecars_valid": bool(records)
        and all(
            record["readiness"]["target_condition_sidecar_valid"]
            for record in records
        ),
        "all_gaussian_evidence_present": bool(records)
        and all(record["readiness"]["gaussian_evidence_present"] for record in records),
        "all_target_candidate_artifacts_valid": bool(records)
        and all(
            record["readiness"]["target_candidate_artifact_valid"]
            for record in records
        ),
        "all_samples_ready_for_batch_readiness_input": bool(records)
        and sum(
            1
            for record in records
            if record["readiness"]["ready_for_batch_readiness_input"]
        )
        == sample_count,
    }


def _sample_issues(
    readiness: Mapping[str, bool],
    *,
    frame_issues: list[str],
    validation_issues: list[str | None],
    missing_gaussians: int,
) -> list[str]:
    issues = list(frame_issues)
    issues.extend(issue for issue in validation_issues if issue)
    if not readiness["sample_workspace_helpers_present"]:
        issues.append("sample authoring helper files are not all present")
    if not readiness["target_condition_sidecar_present"]:
        issues.append("target condition sidecar is not present yet")
    if missing_gaussians:
        issues.append(f"{missing_gaussians} selected frame Gaussian files are missing")
    if not readiness["target_candidate_artifact_present"]:
        issues.append("target ObjectState candidate artifact is not present yet")
    return issues


def _hard_blockers(
    records: list[Mapping[str, Any]],
    readiness: Mapping[str, bool],
) -> list[str]:
    blockers = []
    if not readiness["sample_count_nonzero"]:
        blockers.append("batch spec contains no samples")
    if not readiness["all_scene_roots_present"]:
        blockers.append("one or more BOP sample scene roots are missing")
    if not readiness["all_target_condition_sidecars_valid"]:
        blockers.append("one or more target BOP condition sidecars are missing or invalid")
    if not readiness["all_gaussian_evidence_present"]:
        blockers.append("one or more samples are missing per-frame Gaussian evidence")
    if not readiness["all_target_candidate_artifacts_valid"]:
        blockers.append("one or more target ObjectState candidate artifacts are missing or invalid")
    for record in records:
        for issue in record["issues"]:
            blockers.append(f"{record['sample_id']}: {issue}")
    return blockers


def _issues(records: list[Mapping[str, Any]]) -> list[str]:
    issues = []
    for record in records:
        for issue in record["issues"]:
            issues.append(f"{record['sample_id']}: {issue}")
    return issues


def _next_actions(
    records: list[Mapping[str, Any]],
    readiness: Mapping[str, bool],
) -> list[str]:
    if readiness["all_samples_ready_for_batch_readiness_input"]:
        return ["run audit-bop-local-row-batch-readiness on the batch spec"]
    actions = []
    if not readiness["all_sample_workspace_helpers_present"]:
        actions.append("run init-bop-phase1-sample-workspaces for the batch spec")
    if not readiness["all_target_condition_sidecars_valid"]:
        actions.append("fill and write real target bop-condition-sidecar.json files")
    if any(record["readiness"]["depth_export_candidate"] for record in records):
        actions.append(
            "run export-bop-rgbd-gaussian-evidence for samples with complete depth frames"
        )
    elif not readiness["all_gaussian_evidence_present"]:
        actions.append("reconstruct or place per-frame Gaussian PLY evidence files")
    if not readiness["all_target_candidate_artifacts_valid"]:
        actions.append(
            "fill and finalize BOP ObjectState candidate artifacts as objectstates.json"
        )
    actions.append("rerun audit-bop-phase1-authoring-progress after filling target files")
    return actions


def _sample_next_commands(
    *,
    spec_path: Path | None,
    sample_id: str,
    paths: Mapping[str, str],
    options: Mapping[str, Any],
    readiness: Mapping[str, bool],
) -> list[str]:
    if readiness["ready_for_batch_readiness_input"]:
        if spec_path is None:
            return []
        return [
            (
                "uv run objgauss object-state audit-bop-local-row-batch-readiness "
                f"{spec_path} --summary-output {spec_path.parent / 'bop-local-row-batch-readiness.json'}"
            )
        ]
    frame_opts = _frame_opts(
        max_frames=options.get("max_frames"),
        frame_step=int(options.get("frame_step", 1)),
    )
    common_opts = (
        f"--sample-id {sample_id} --dataset-id {options['dataset_id']} "
        f"--object-category {options['object_category']} --scenario {options['scenario']}"
    )
    commands = []
    if not readiness["target_condition_sidecar_valid"]:
        commands.append(
            (
                "uv run objgauss object-state init-bop-condition-sidecar "
                f"{paths['scene_root']} --condition-csv {paths['condition_csv_template']} "
                f"--output {paths['target_condition_sidecar']} "
                f"--summary-output {Path(paths['target_condition_sidecar']).with_name('bop-condition-sidecar-summary.json')} "
                f"{frame_opts} --require-identity-ready"
            ).strip()
        )
    if not readiness["gaussian_evidence_present"]:
        if readiness["depth_export_candidate"]:
            commands.append(
                (
                    "uv run objgauss object-state export-bop-rgbd-gaussian-evidence "
                    f"{paths['scene_root']} {common_opts} {frame_opts} --require-ready"
                ).strip()
            )
        else:
            commands.append(
                (
                    "uv run objgauss object-state audit-bop-gaussian-evidence "
                    f"{paths['scene_root']} {common_opts} {frame_opts} "
                    f"--condition-sidecar {paths['target_condition_sidecar']}"
                ).strip()
            )
    if not readiness["target_candidate_artifact_valid"]:
        if readiness["gaussian_evidence_present"]:
            commands.append(
                (
                    "uv run objgauss object-state generate-bop-objectstate-baseline-candidate "
                    f"{paths['scene_root']} --output {paths['target_candidate_artifact']} "
                    f"{common_opts} --condition-sidecar {paths['target_condition_sidecar']} "
                    f"{frame_opts} --require-ready"
                ).strip()
            )
        if not readiness["objectstate_template_valid"]:
            commands.append(
                (
                    "uv run objgauss object-state init-bop-objectstate-artifact-template "
                    f"{paths['scene_root']} --output {paths['objectstate_template']} "
                    f"{common_opts} --condition-sidecar {paths['target_condition_sidecar']} "
                    f"--target-artifact-path {paths['target_candidate_artifact']} {frame_opts}"
                ).strip()
            )
        else:
            commands.append(
                (
                    "uv run objgauss object-state finalize-bop-objectstate-artifact-template "
                    f"{paths['objectstate_template']} --output {paths['target_candidate_artifact']} "
                    f"--scene-root {paths['scene_root']} --dataset-id {options['dataset_id']} "
                    f"--object-category {options['object_category']} --scenario {options['scenario']} "
                    f"--condition-sidecar {paths['target_condition_sidecar']} {frame_opts}"
                ).strip()
            )
    return commands


def _batch_next_commands(
    spec_path: Path | None,
    readiness: Mapping[str, bool],
) -> list[str]:
    if spec_path is None:
        return []
    if readiness["all_samples_ready_for_batch_readiness_input"]:
        return [
            (
                "uv run objgauss object-state audit-bop-local-row-batch-readiness "
                f"{spec_path} --summary-output {spec_path.parent / 'bop-local-row-batch-readiness.json'} "
                "--require-ready"
            )
        ]
    return [
        (
            "uv run objgauss object-state init-bop-phase1-sample-workspaces "
            f"{spec_path}"
        ),
        (
            "uv run objgauss object-state audit-bop-phase1-authoring-progress "
            f"{spec_path} --summary-output {spec_path.parent / 'bop-phase1-authoring-progress.json'}"
        ),
    ]


def _sample_table_markdown(records: list[Mapping[str, Any]]) -> str:
    lines = [
        "| sample_id | helpers | sidecar | gaussians | candidate | ready |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        readiness = record["readiness"]
        counts = record["row_counts"]
        lines.append(
            "| {sample} | {helpers} | {sidecar} | {present}/{expected} | "
            "{candidate} | {ready} |".format(
                sample=record["sample_id"],
                helpers=_yes_no(readiness["sample_workspace_helpers_present"]),
                sidecar=_yes_no(readiness["target_condition_sidecar_valid"]),
                present=counts["present_gaussian_files"],
                expected=counts["expected_gaussian_files"],
                candidate=_yes_no(readiness["target_candidate_artifact_valid"]),
                ready=_yes_no(readiness["ready_for_batch_readiness_input"]),
            )
        )
    return "\n".join(lines) + "\n"


def _selected_frame_ids(
    scene_root: Path,
    *,
    max_frames: int | None,
    frame_step: int,
) -> tuple[list[int], list[str]]:
    if not scene_root.exists():
        return [], [f"scene root missing: {scene_root}"]
    issues = []
    scene_camera = _read_json_mapping_or_issue(scene_root / "scene_camera.json", issues)
    scene_gt = _read_json_mapping_or_issue(scene_root / "scene_gt.json", issues)
    if scene_camera is None or scene_gt is None:
        return [], issues
    camera_ids = {_int_key(key) for key in scene_camera}
    gt_ids = {_int_key(key) for key in scene_gt}
    frame_ids = sorted(frame_id for frame_id in camera_ids & gt_ids if frame_id is not None)
    if not frame_ids:
        issues.append("scene_camera.json and scene_gt.json have no shared frame ids")
        return [], issues
    selected = frame_ids[::frame_step]
    if max_frames is not None:
        selected = selected[:max_frames]
    if not selected:
        issues.append("selected frame list is empty")
    return selected, issues


def _gaussian_records(
    scene_root: Path,
    frame_ids: list[int],
    *,
    gaussian_dir: str,
    min_gaussian_bytes: int,
) -> list[dict[str, Any]]:
    return [
        _file_record(
            scene_root / gaussian_dir / f"{frame_id:06d}.ply",
            frame_id=frame_id,
            ref=f"{gaussian_dir}/{frame_id:06d}.ply",
            min_bytes=min_gaussian_bytes,
        )
        for frame_id in frame_ids
    ]


def _frame_file_records(
    scene_root: Path,
    frame_ids: list[int],
    *,
    directory: str,
    suffix: str,
) -> list[dict[str, Any]]:
    return [
        _file_record(
            scene_root / directory / f"{frame_id:06d}{suffix}",
            frame_id=frame_id,
            ref=f"{directory}/{frame_id:06d}{suffix}",
            min_bytes=1,
        )
        for frame_id in frame_ids
    ]


def _file_record(
    path: Path,
    *,
    frame_id: int,
    ref: str,
    min_bytes: int,
) -> dict[str, Any]:
    exists = path.exists()
    is_file = path.is_file()
    size = path.stat().st_size if is_file else None
    valid = bool(is_file and size is not None and size >= min_bytes)
    reason = None
    if not exists:
        reason = "missing"
    elif not is_file:
        reason = "not a file"
    elif size is not None and size < min_bytes:
        reason = f"size {size} < min {min_bytes}"
    return {
        "frame_id": str(frame_id),
        "ref": ref,
        "path": str(path),
        "exists": exists,
        "is_file": is_file,
        "size_bytes": size,
        "valid": valid,
        "missing_reason": reason,
    }


def _validate_json_file(path: Path, validator: Validator, label: str) -> tuple[bool, str | None]:
    if not path.exists():
        return False, None
    if not path.is_file():
        return False, f"{label} path is not a file: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - audit reports invalid local files.
        return False, f"{label} JSON is invalid: {exc}"
    if not isinstance(payload, Mapping):
        return False, f"{label} JSON must be an object"
    try:
        validator(dict(payload))
    except Exception as exc:  # noqa: BLE001 - audit reports invalid local files.
        return False, f"{label} schema is invalid: {exc}"
    return True, None


def _read_json_mapping_or_issue(path: Path, issues: list[str]) -> Mapping[str, Any] | None:
    if not path.is_file():
        issues.append(f"required BOP JSON missing: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - audit reports invalid local files.
        issues.append(f"required BOP JSON invalid: {path}: {exc}")
        return None
    if not isinstance(payload, Mapping):
        issues.append(f"required BOP JSON must be an object: {path}")
        return None
    return payload


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"BOP Phase 1 authoring progress requires {name}")
    return value


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError("BOP Phase 1 authoring progress max_frames must be >= 1")
    return parsed


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"BOP Phase 1 authoring progress {name} must be >= 1")
    return parsed


def _non_negative_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"BOP Phase 1 authoring progress {name} must be >= 0")
    return parsed


def _int_key(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _frame_opts(*, max_frames: Any, frame_step: int) -> str:
    opts = []
    if max_frames is not None:
        opts.extend(["--max-frames", str(max_frames)])
    if frame_step != 1:
        opts.extend(["--frame-step", str(frame_step)])
    return " ".join(opts)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


__all__ = (
    "OBJECTSTATE_BOP_PHASE1_AUTHORING_PROGRESS_SCHEMA",
    "objectstate_bop_phase1_authoring_progress",
    "validate_objectstate_bop_phase1_authoring_progress_summary",
)
