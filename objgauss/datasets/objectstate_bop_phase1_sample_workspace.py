from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from objgauss.datasets.objectstate_bop_capture_adapter import (
    objectstate_bop_capture_condition_sidecar_summary,
    validate_objectstate_bop_capture_condition_sidecar_summary,
)
from objgauss.datasets.objectstate_bop_local_row_batch_spec import (
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
    read_objectstate_bop_local_row_batch_spec,
    validate_objectstate_bop_local_row_batch_spec,
)

OBJECTSTATE_BOP_PHASE1_SAMPLE_WORKSPACES_SCHEMA = (
    "objgauss-objectstate-bop-phase1-sample-workspaces-v1"
)


def objectstate_bop_phase1_sample_workspaces(
    batch_spec: str | Path | Mapping[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    spec_path: Path | None = None
    if isinstance(batch_spec, (str, Path)):
        spec_path = Path(batch_spec)
        spec = read_objectstate_bop_local_row_batch_spec(spec_path)
    else:
        spec = validate_objectstate_bop_local_row_batch_spec(batch_spec)
    spec_root = spec_path.parent if spec_path is not None else Path.cwd()
    defaults = spec.get("defaults", {})

    sample_records = [
        _init_sample_workspace(
            sample,
            defaults=defaults,
            spec_root=spec_root,
            force=force,
        )
        for sample in spec["samples"]
    ]
    row_counts = {
        "samples": len(sample_records),
        "scene_roots_present": sum(
            1 for record in sample_records if record["readiness"]["scene_root_present"]
        ),
        "condition_csv_templates_written": sum(
            1
            for record in sample_records
            if record["readiness"]["condition_csv_template_written"]
        ),
        "condition_sidecar_drafts_written": sum(
            1
            for record in sample_records
            if record["readiness"]["condition_sidecar_draft_written"]
        ),
        "target_condition_sidecars_present": sum(
            1
            for record in sample_records
            if record["readiness"]["target_condition_sidecar_present"]
        ),
        "target_candidate_artifacts_present": sum(
            1
            for record in sample_records
            if record["readiness"]["target_candidate_artifact_present"]
        ),
    }
    readiness = {
        "sample_count_nonzero": bool(sample_records),
        "all_scene_roots_present": row_counts["scene_roots_present"]
        == row_counts["samples"],
        "all_condition_templates_written": row_counts[
            "condition_csv_templates_written"
        ]
        == row_counts["samples"],
        "authoring_workspace_initialized": bool(sample_records)
        and all(record["readiness"]["readme_written"] for record in sample_records),
    }
    readiness["sample_workspaces_ready_to_author"] = bool(
        readiness["sample_count_nonzero"]
        and readiness["all_scene_roots_present"]
        and readiness["all_condition_templates_written"]
        and readiness["authoring_workspace_initialized"]
    )
    payload = {
        "schema": OBJECTSTATE_BOP_PHASE1_SAMPLE_WORKSPACES_SCHEMA,
        "kind": "objectstate_bop_phase1_sample_workspaces",
        "status": (
            "objectstate_bop_phase1_sample_workspaces_initialized"
            if readiness["sample_workspaces_ready_to_author"]
            else "objectstate_bop_phase1_sample_workspaces_blocked"
        ),
        "batch_spec_schema": OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
        "batch_spec": str(spec_path) if spec_path is not None else "",
        "spec_root": str(spec_root),
        "row_counts": row_counts,
        "readiness": readiness,
        "sample_records": sample_records,
        "hard_blockers": _hard_blockers(sample_records, readiness),
        "issues": _issues(sample_records),
        "next_commands": _batch_next_commands(spec_path),
        "claim_policy": {
            "initializes_sample_authoring_workspaces": True,
            "uses_explicit_batch_spec": True,
            "writes_condition_csv_templates": True,
            "writes_condition_sidecar_drafts": True,
            "writes_per_sample_readmes": True,
            "does_not_download_dataset": True,
            "does_not_copy_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_condition_metadata": True,
            "does_not_create_target_condition_sidecar": True,
            "does_not_create_candidate_artifact": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_run_readiness": True,
            "does_not_run_handoff": True,
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
            "creates_target_condition_sidecar": False,
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
    return validate_objectstate_bop_phase1_sample_workspaces_summary(payload)


def validate_objectstate_bop_phase1_sample_workspaces_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP Phase 1 sample workspaces summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_PHASE1_SAMPLE_WORKSPACES_SCHEMA:
        raise ValueError(
            "unsupported BOP Phase 1 sample workspaces schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_phase1_sample_workspaces":
        raise ValueError("BOP Phase 1 sample workspaces kind is unsupported")
    if payload.get("status") not in {
        "objectstate_bop_phase1_sample_workspaces_initialized",
        "objectstate_bop_phase1_sample_workspaces_blocked",
    }:
        raise ValueError("BOP Phase 1 sample workspaces status is unsupported")
    if payload.get("batch_spec_schema") != OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA:
        raise ValueError("BOP Phase 1 sample workspaces batch spec schema mismatch")
    if not isinstance(payload.get("spec_root"), str) or not payload["spec_root"]:
        raise ValueError("BOP Phase 1 sample workspaces requires spec_root")
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("BOP Phase 1 sample workspaces requires row counts")
    for key in (
        "samples",
        "scene_roots_present",
        "condition_csv_templates_written",
        "condition_sidecar_drafts_written",
        "target_condition_sidecars_present",
        "target_candidate_artifacts_present",
    ):
        if not isinstance(row_counts.get(key), int):
            raise ValueError(f"BOP Phase 1 sample workspaces row count {key} invalid")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("BOP Phase 1 sample workspaces requires readiness")
    for key in (
        "sample_count_nonzero",
        "all_scene_roots_present",
        "all_condition_templates_written",
        "authoring_workspace_initialized",
        "sample_workspaces_ready_to_author",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"BOP Phase 1 sample workspaces readiness {key} invalid")
    records = payload.get("sample_records")
    if not isinstance(records, list) or len(records) != row_counts["samples"]:
        raise ValueError("BOP Phase 1 sample workspaces sample record count mismatch")
    for record in records:
        _validate_sample_record(record)
    expected_status = (
        "objectstate_bop_phase1_sample_workspaces_initialized"
        if readiness["sample_workspaces_ready_to_author"]
        else "objectstate_bop_phase1_sample_workspaces_blocked"
    )
    if payload["status"] != expected_status:
        raise ValueError("BOP Phase 1 sample workspaces status mismatch")
    for key in ("hard_blockers", "issues", "next_commands"):
        values = payload.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"BOP Phase 1 sample workspaces {key} must be string list")
    claim_policy = payload.get("claim_policy")
    if not isinstance(claim_policy, Mapping):
        raise ValueError("BOP Phase 1 sample workspaces requires claim policy")
    for key in (
        "initializes_sample_authoring_workspaces",
        "uses_explicit_batch_spec",
        "writes_condition_csv_templates",
        "writes_condition_sidecar_drafts",
        "writes_per_sample_readmes",
        "does_not_download_dataset",
        "does_not_copy_dataset",
        "does_not_create_ground_truth",
        "does_not_infer_condition_metadata",
        "does_not_create_target_condition_sidecar",
        "does_not_create_candidate_artifact",
        "does_not_reconstruct_gaussians",
        "does_not_run_readiness",
        "does_not_run_handoff",
        "does_not_train_model",
        "does_not_claim_metric_pass",
        "does_not_claim_intervention_gate",
        "does_not_claim_world_model",
    ):
        if not claim_policy.get(key):
            raise ValueError(
                f"BOP Phase 1 sample workspaces claim policy missing {key}"
            )
    non_goals = payload.get("non_goals")
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP Phase 1 sample workspaces cannot claim downloads, copies, GT, "
            "condition inference, target sidecars, candidate artifacts, "
            "Gaussian reconstruction, readiness, handoff, eval, intervention, "
            "training, public samples, replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def _init_sample_workspace(
    sample: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    spec_root: Path,
    force: bool,
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
    authoring_root.mkdir(parents=True, exist_ok=True)

    condition_summary: dict[str, Any] | None = None
    write_errors: list[str] = []
    if scene_root.exists():
        try:
            condition_summary = objectstate_bop_capture_condition_sidecar_summary(
                scene_root,
                max_frames=_optional_int(merged.get("max_frames")),
                frame_step=int(merged.get("frame_step", 1)),
                min_view_conditions=int(merged.get("min_view_conditions", 2)),
                min_lighting_conditions=int(
                    merged.get("min_lighting_conditions", 2)
                ),
                min_camera_motion_m=float(merged.get("min_camera_motion_m", 0.01)),
            )
            _write_condition_csv_template(
                condition_csv_template,
                condition_summary["condition_csv_template"],
                force=force,
            )
            _write_json(
                condition_sidecar_draft,
                condition_summary["sidecar"],
                force=force,
            )
        except Exception as exc:  # pragma: no cover - surfaced in summary.
            write_errors.append(f"condition authoring failed: {exc}")
    else:
        write_errors.append(f"scene root missing: {scene_root}")

    _write_text(
        readme,
        _sample_readme(
            sample_id=sample_id,
            scene_root=scene_root,
            candidate_artifact=candidate_artifact,
            condition_sidecar=condition_sidecar,
            condition_csv_template=condition_csv_template,
            objectstate_template=objectstate_template,
            merged=merged,
        ),
        force=force,
    )

    readiness = {
        "scene_root_present": scene_root.exists(),
        "authoring_root_created": authoring_root.is_dir(),
        "condition_csv_template_written": condition_csv_template.is_file(),
        "condition_sidecar_draft_written": condition_sidecar_draft.is_file(),
        "readme_written": readme.is_file(),
        "target_condition_sidecar_present": condition_sidecar.is_file(),
        "target_candidate_artifact_present": candidate_artifact.is_file(),
    }
    record = {
        "sample_id": sample_id,
        "paths": {
            "scene_root": str(scene_root),
            "authoring_root": str(authoring_root),
            "condition_csv_template": str(condition_csv_template),
            "condition_sidecar_draft": str(condition_sidecar_draft),
            "target_condition_sidecar": str(condition_sidecar),
            "objectstate_template": str(objectstate_template),
            "target_candidate_artifact": str(candidate_artifact),
            "readme": str(readme),
        },
        "readiness": readiness,
        "condition_sidecar_summary_schema": (
            condition_summary["schema"] if condition_summary else ""
        ),
        "condition_sidecar_status": (
            condition_summary["status"] if condition_summary else "not_generated"
        ),
        "condition_sidecar_summary": condition_summary,
        "issues": _sample_issues(readiness, write_errors),
        "next_commands": _sample_next_commands(
            sample_id=sample_id,
            scene_root=scene_root,
            candidate_artifact=candidate_artifact,
            condition_sidecar=condition_sidecar,
            condition_csv_template=condition_csv_template,
            objectstate_template=objectstate_template,
            merged=merged,
        ),
    }
    return record


def _validate_sample_record(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("BOP Phase 1 sample workspace record must map")
    if not isinstance(record.get("sample_id"), str) or not record["sample_id"]:
        raise ValueError("BOP Phase 1 sample workspace record requires sample_id")
    paths = record.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("BOP Phase 1 sample workspace record requires paths")
    for key in (
        "scene_root",
        "authoring_root",
        "condition_csv_template",
        "condition_sidecar_draft",
        "target_condition_sidecar",
        "objectstate_template",
        "target_candidate_artifact",
        "readme",
    ):
        if not isinstance(paths.get(key), str) or not paths[key]:
            raise ValueError(f"BOP Phase 1 sample workspace path {key} invalid")
    readiness = record.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("BOP Phase 1 sample workspace record requires readiness")
    for key in (
        "scene_root_present",
        "authoring_root_created",
        "condition_csv_template_written",
        "condition_sidecar_draft_written",
        "readme_written",
        "target_condition_sidecar_present",
        "target_candidate_artifact_present",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"BOP Phase 1 sample workspace readiness {key} invalid")
    if record.get("condition_sidecar_summary") is not None:
        validate_objectstate_bop_capture_condition_sidecar_summary(
            record["condition_sidecar_summary"]
        )
    if record.get("condition_sidecar_summary_schema") not in {
        "",
        "objgauss-objectstate-bop-capture-condition-sidecar-summary-v1",
    }:
        raise ValueError("BOP Phase 1 sample workspace sidecar schema invalid")
    for key in ("issues", "next_commands"):
        values = record.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"BOP Phase 1 sample workspace {key} must be string list")


def _sample_readme(
    *,
    sample_id: str,
    scene_root: Path,
    candidate_artifact: Path,
    condition_sidecar: Path,
    condition_csv_template: Path,
    objectstate_template: Path,
    merged: Mapping[str, Any],
) -> str:
    lines = [
        f"# BOP Phase 1 Sample Workspace: {sample_id}",
        "",
        f"- scene_root: `{scene_root}`",
        f"- target_condition_sidecar: `{condition_sidecar}`",
        f"- target_candidate_artifact: `{candidate_artifact}`",
        "",
        "This directory is an authoring scaffold only. It does not create",
        "ground truth, real condition metadata, ObjectState candidates,",
        "Gaussian evidence, metric pass rows, intervention evidence, or",
        "world-model claims.",
        "",
        "## Next Commands",
        "",
    ]
    for command in _sample_next_commands(
        sample_id=sample_id,
        scene_root=scene_root,
        candidate_artifact=candidate_artifact,
        condition_sidecar=condition_sidecar,
        condition_csv_template=condition_csv_template,
        objectstate_template=objectstate_template,
        merged=merged,
    ):
        lines.append(f"- `{command}`")
    lines.append("")
    return "\n".join(lines)


def _sample_next_commands(
    *,
    sample_id: str,
    scene_root: Path,
    candidate_artifact: Path,
    condition_sidecar: Path,
    condition_csv_template: Path,
    objectstate_template: Path,
    merged: Mapping[str, Any],
) -> list[str]:
    dataset_id = merged.get("dataset_id", "bop-ycbv")
    object_category = merged.get("object_category", "bop_objects")
    scenario = merged.get("scenario", "bop_pose_sequence")
    max_frames = _optional_int(merged.get("max_frames"))
    frame_step = int(merged.get("frame_step", 1))
    frame_opts = _frame_opts(max_frames=max_frames, frame_step=frame_step)
    common_opts = (
        f"--sample-id {sample_id} --dataset-id {dataset_id} "
        f"--object-category {object_category} --scenario {scenario}"
    )
    return [
        (
            "uv run objgauss object-state init-bop-condition-sidecar "
            f"{scene_root} --condition-csv {condition_csv_template} "
            f"--output {condition_sidecar} "
            f"--summary-output {condition_sidecar.with_name('bop-condition-sidecar-summary.json')} "
            f"{frame_opts} --require-identity-ready"
        ).strip(),
        (
            "uv run objgauss object-state export-bop-rgbd-gaussian-evidence "
            f"{scene_root} {common_opts} {frame_opts} --require-ready"
        ).strip(),
        (
            "uv run objgauss object-state generate-bop-objectstate-baseline-candidate "
            f"{scene_root} --output {candidate_artifact} {common_opts} "
            f"--condition-sidecar {condition_sidecar} {frame_opts} --require-ready"
        ).strip(),
        (
            "uv run objgauss object-state init-bop-objectstate-artifact-template "
            f"{scene_root} --output {objectstate_template} {common_opts} "
            f"--condition-sidecar {condition_sidecar} --target-artifact-path {candidate_artifact} "
            f"{frame_opts}"
        ).strip(),
        (
            "uv run objgauss object-state finalize-bop-objectstate-artifact-template "
            f"{objectstate_template} --output {candidate_artifact} "
            f"--scene-root {scene_root} --dataset-id {dataset_id} "
            f"--object-category {object_category} --scenario {scenario} "
            f"--condition-sidecar {condition_sidecar} {frame_opts}"
        ).strip(),
    ]


def _sample_issues(readiness: Mapping[str, bool], write_errors: list[str]) -> list[str]:
    issues = list(write_errors)
    if not readiness["target_condition_sidecar_present"]:
        issues.append("target condition sidecar is not present yet")
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
    if not readiness["all_condition_templates_written"]:
        blockers.append("one or more condition CSV templates were not written")
    for record in records:
        for issue in record["issues"]:
            if issue.startswith("scene root missing") or issue.startswith(
                "condition authoring failed"
            ):
                blockers.append(f"{record['sample_id']}: {issue}")
    return blockers


def _issues(records: list[Mapping[str, Any]]) -> list[str]:
    issues = []
    for record in records:
        for issue in record["issues"]:
            issues.append(f"{record['sample_id']}: {issue}")
    return issues


def _batch_next_commands(spec_path: Path | None) -> list[str]:
    if spec_path is None:
        return []
    return [
        (
            "uv run objgauss object-state audit-bop-phase1-authoring-progress "
            f"{spec_path} --summary-output {spec_path.parent / 'bop-phase1-authoring-progress.json'}"
        ),
        (
            "uv run objgauss object-state audit-bop-local-row-batch-readiness "
            f"{spec_path} --summary-output {spec_path.parent / 'bop-local-row-batch-readiness.json'}"
        )
    ]


def _write_condition_csv_template(
    path: Path,
    rows: list[Mapping[str, Any]],
    *,
    force: bool,
) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, body: str, *, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"BOP Phase 1 sample workspaces require {name}")
    return value


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _frame_opts(*, max_frames: int | None, frame_step: int) -> str:
    opts = []
    if max_frames is not None:
        opts.extend(["--max-frames", str(max_frames)])
    if frame_step != 1:
        opts.extend(["--frame-step", str(frame_step)])
    return " ".join(opts)


__all__ = (
    "OBJECTSTATE_BOP_PHASE1_SAMPLE_WORKSPACES_SCHEMA",
    "objectstate_bop_phase1_sample_workspaces",
    "validate_objectstate_bop_phase1_sample_workspaces_summary",
)
