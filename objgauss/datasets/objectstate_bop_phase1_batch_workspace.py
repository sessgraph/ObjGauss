from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

from objgauss.datasets.objectstate_bop_local_row_batch_authoring import (
    objectstate_bop_local_row_batch_spec_authoring,
    validate_objectstate_bop_local_row_batch_spec_authoring_summary,
)
from objgauss.datasets.objectstate_bop_phase1_subset_selector import (
    OBJECTSTATE_BOP_PHASE1_SUBSET_SELECTOR_SCHEMA,
    objectstate_bop_phase1_subset_selector,
    validate_objectstate_bop_phase1_subset_selector_summary,
)

OBJECTSTATE_BOP_PHASE1_BATCH_WORKSPACE_SCHEMA = (
    "objgauss-objectstate-bop-phase1-batch-workspace-v1"
)


def objectstate_bop_phase1_batch_workspace(
    dataset_root: str | Path,
    *,
    workspace_root: str | Path,
    dataset_id: str = "bop-ycbv",
    object_category: str = "bop_objects",
    scenario: str = "bop_pose_sequence",
    fps: float = 30.0,
    license_text: str = (
        "BOP dataset terms; verify source dataset license before redistribution"
    ),
    rgb_dir: str = "rgb",
    depth_dir: str = "depth",
    gaussian_dir: str = "gaussians",
    max_frames: int | None = None,
    frame_step: int = 1,
    max_depth: int = 3,
    max_scene_candidates: int = 20,
    min_frames: int = 3,
    min_objects: int = 1,
    min_persistent_objects: int = 1,
    batch_id: str = "bop-phase1-local-row-batch",
    batch_output_root: str | Path = "handoff",
) -> dict[str, Any]:
    root = Path(dataset_root)
    workspace = Path(workspace_root)
    samples_csv = workspace / "samples.csv"
    selector_summary_path = workspace / "selector-summary.json"
    batch_spec_path = workspace / "bop-local-row-batch.json"
    batch_spec_summary_path = workspace / "batch-spec-authoring-summary.json"
    readme_path = workspace / "README.md"
    artifact_root = workspace / "artifacts"

    selector = objectstate_bop_phase1_subset_selector(
        root,
        dataset_id=dataset_id,
        output_root=workspace,
        object_category=object_category,
        scenario=scenario,
        fps=fps,
        license_text=license_text,
        rgb_dir=rgb_dir,
        max_frames=max_frames,
        frame_step=frame_step,
        max_depth=max_depth,
        max_scene_candidates=max_scene_candidates,
        min_frames=min_frames,
        min_objects=min_objects,
        min_persistent_objects=min_persistent_objects,
    )

    workspace.mkdir(parents=True, exist_ok=True)
    _write_json(selector_summary_path, selector)
    csv_rows = _write_samples_csv(
        samples_csv,
        selector,
        artifact_root=artifact_root,
        dataset_id=dataset_id,
        object_category=object_category,
        scenario=scenario,
        max_frames=max_frames,
        frame_step=frame_step,
    )

    batch_spec_summary: dict[str, Any] | None = None
    if csv_rows:
        batch_spec_summary = objectstate_bop_local_row_batch_spec_authoring(
            samples_csv,
            output=batch_spec_path,
            batch_id=batch_id,
            batch_output_root=batch_output_root,
            dataset_id=dataset_id,
            object_category=object_category,
            scenario=scenario,
            fps=fps,
            license_text=license_text,
            rgb_dir=rgb_dir,
            depth_dir=depth_dir,
            gaussian_dir=gaussian_dir,
        )
        _write_json(batch_spec_summary_path, batch_spec_summary)

    readme_path.write_text(
        _workspace_readme(
            dataset_root=root,
            workspace=workspace,
            samples_csv=samples_csv,
            batch_spec=batch_spec_path,
            csv_rows=csv_rows,
        ),
        encoding="utf-8",
    )

    readiness = {
        "selector_ready": selector["recommended"] is not None,
        "samples_csv_has_rows": bool(csv_rows),
        "batch_spec_written": batch_spec_path.is_file(),
        "batch_spec_inputs_ready": bool(
            batch_spec_summary
            and batch_spec_summary["status"]
            == "objectstate_bop_local_row_batch_spec_authoring_ready"
        ),
        "workspace_reviewable": bool(csv_rows and batch_spec_summary),
    }
    payload = {
        "schema": OBJECTSTATE_BOP_PHASE1_BATCH_WORKSPACE_SCHEMA,
        "kind": "objectstate_bop_phase1_batch_workspace",
        "status": (
            "objectstate_bop_phase1_batch_workspace_authored"
            if readiness["workspace_reviewable"]
            else "objectstate_bop_phase1_batch_workspace_blocked"
        ),
        "dataset_root": str(root),
        "workspace_root": str(workspace),
        "selector_schema": OBJECTSTATE_BOP_PHASE1_SUBSET_SELECTOR_SCHEMA,
        "files": {
            "selector_summary": str(selector_summary_path),
            "samples_csv": str(samples_csv),
            "batch_spec": str(batch_spec_path),
            "batch_spec_authoring_summary": str(batch_spec_summary_path),
            "readme": str(readme_path),
        },
        "row_counts": {
            "selector_candidates": selector["row_counts"]["scene_candidates"],
            "ready_selector_candidates": selector["row_counts"]["ready_candidates"],
            "samples_csv_rows": len(csv_rows),
        },
        "readiness": readiness,
        "selector_summary": selector,
        "batch_spec_authoring_summary": batch_spec_summary,
        "hard_blockers": _hard_blockers(selector, csv_rows),
        "issues": _issues(batch_spec_summary),
        "next_commands": _next_commands(batch_spec_path, bool(csv_rows)),
        "claim_policy": {
            "writes_local_workspace": True,
            "uses_bop_subset_selector": True,
            "writes_samples_csv": True,
            "writes_batch_spec_when_samples_exist": True,
            "does_not_download_dataset": True,
            "does_not_copy_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_condition_metadata": True,
            "does_not_create_candidate_artifact": True,
            "does_not_create_condition_sidecar": True,
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
            "creates_candidate_artifact": False,
            "creates_condition_sidecar": False,
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
    return validate_objectstate_bop_phase1_batch_workspace_summary(payload)


def validate_objectstate_bop_phase1_batch_workspace_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP Phase 1 batch workspace summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_PHASE1_BATCH_WORKSPACE_SCHEMA:
        raise ValueError(
            "unsupported BOP Phase 1 batch workspace schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_phase1_batch_workspace":
        raise ValueError("BOP Phase 1 batch workspace kind is unsupported")
    if payload.get("status") not in {
        "objectstate_bop_phase1_batch_workspace_authored",
        "objectstate_bop_phase1_batch_workspace_blocked",
    }:
        raise ValueError("BOP Phase 1 batch workspace status is unsupported")
    for key in ("dataset_root", "workspace_root", "selector_schema"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"BOP Phase 1 batch workspace requires {key}")
    if payload["selector_schema"] != OBJECTSTATE_BOP_PHASE1_SUBSET_SELECTOR_SCHEMA:
        raise ValueError("BOP Phase 1 batch workspace selector schema mismatch")
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("BOP Phase 1 batch workspace requires files")
    for key in (
        "selector_summary",
        "samples_csv",
        "batch_spec",
        "batch_spec_authoring_summary",
        "readme",
    ):
        if not isinstance(files.get(key), str) or not files[key]:
            raise ValueError(f"BOP Phase 1 batch workspace missing file {key}")
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("BOP Phase 1 batch workspace requires row counts")
    for key in ("selector_candidates", "ready_selector_candidates", "samples_csv_rows"):
        if not isinstance(row_counts.get(key), int):
            raise ValueError(f"BOP Phase 1 batch workspace row count {key} invalid")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("BOP Phase 1 batch workspace requires readiness")
    for key in (
        "selector_ready",
        "samples_csv_has_rows",
        "batch_spec_written",
        "batch_spec_inputs_ready",
        "workspace_reviewable",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"BOP Phase 1 batch workspace readiness {key} invalid")
    selector = validate_objectstate_bop_phase1_subset_selector_summary(
        payload.get("selector_summary")
    )
    if row_counts["selector_candidates"] != selector["row_counts"]["scene_candidates"]:
        raise ValueError("BOP Phase 1 batch workspace selector count mismatch")
    if (
        row_counts["ready_selector_candidates"]
        != selector["row_counts"]["ready_candidates"]
    ):
        raise ValueError("BOP Phase 1 batch workspace ready count mismatch")
    batch_spec_summary = payload.get("batch_spec_authoring_summary")
    if batch_spec_summary is not None:
        batch_spec_summary = validate_objectstate_bop_local_row_batch_spec_authoring_summary(
            batch_spec_summary
        )
    if readiness["workspace_reviewable"] != bool(
        row_counts["samples_csv_rows"] and batch_spec_summary
    ):
        raise ValueError("BOP Phase 1 batch workspace reviewable mismatch")
    expected_status = (
        "objectstate_bop_phase1_batch_workspace_authored"
        if readiness["workspace_reviewable"]
        else "objectstate_bop_phase1_batch_workspace_blocked"
    )
    if payload["status"] != expected_status:
        raise ValueError("BOP Phase 1 batch workspace status mismatch")
    for key in ("hard_blockers", "issues", "next_commands"):
        values = payload.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"BOP Phase 1 batch workspace {key} must be string list")
    claim_policy = payload.get("claim_policy")
    if not isinstance(claim_policy, Mapping):
        raise ValueError("BOP Phase 1 batch workspace requires claim policy")
    for key in (
        "writes_local_workspace",
        "uses_bop_subset_selector",
        "writes_samples_csv",
        "writes_batch_spec_when_samples_exist",
        "does_not_download_dataset",
        "does_not_copy_dataset",
        "does_not_create_ground_truth",
        "does_not_infer_condition_metadata",
        "does_not_create_candidate_artifact",
        "does_not_create_condition_sidecar",
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
                f"BOP Phase 1 batch workspace claim policy missing {key}"
            )
    non_goals = payload.get("non_goals")
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP Phase 1 batch workspace cannot claim downloads, copies, GT, "
            "condition inference, candidate artifacts, sidecars, Gaussian "
            "reconstruction, readiness, handoff, eval, intervention, training, "
            "public samples, replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def _write_samples_csv(
    output: Path,
    selector: Mapping[str, Any],
    *,
    artifact_root: Path,
    dataset_id: str,
    object_category: str,
    scenario: str,
    max_frames: int | None,
    frame_step: int,
) -> list[dict[str, str]]:
    rows = []
    for candidate in selector["candidates"]:
        if not candidate["readiness"]["phase1_seed_ready"]:
            continue
        sample_id = candidate["sample_id"]
        sample_artifact_root = artifact_root / sample_id
        rows.append(
            {
                "sample_id": sample_id,
                "scene_root": _path_for_csv(candidate["scene_root"], output.parent),
                "candidate_artifact": _path_for_csv(
                    sample_artifact_root / "objectstates.json",
                    output.parent,
                ),
                "condition_sidecar": _path_for_csv(
                    sample_artifact_root / "bop-condition-sidecar.json",
                    output.parent,
                ),
                "output_root": f"samples/{sample_id}",
                "dataset_id": dataset_id,
                "object_category": object_category,
                "scenario": scenario,
                "max_frames": "" if max_frames is None else str(max_frames),
                "frame_step": str(frame_step),
            }
        )
    fieldnames = [
        "sample_id",
        "scene_root",
        "candidate_artifact",
        "condition_sidecar",
        "output_root",
        "dataset_id",
        "object_category",
        "scenario",
        "max_frames",
        "frame_step",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _workspace_readme(
    *,
    dataset_root: Path,
    workspace: Path,
    samples_csv: Path,
    batch_spec: Path,
    csv_rows: list[dict[str, str]],
) -> str:
    lines = [
        "# ObjGauss BOP Phase 1 Batch Workspace",
        "",
        f"- dataset_root: `{dataset_root}`",
        f"- workspace_root: `{workspace}`",
        f"- samples_csv: `{samples_csv}`",
        f"- batch_spec: `{batch_spec}`",
        f"- sample_rows: `{len(csv_rows)}`",
        "",
        "This workspace is an authoring scaffold only. It does not create",
        "ground truth, ObjectState candidates, condition sidecars, Gaussian",
        "evidence, metric pass rows, intervention evidence, or world-model",
        "claims.",
        "",
        "## Next Commands",
        "",
    ]
    for command in _next_commands(batch_spec, bool(csv_rows)):
        lines.append(f"- `{command}`")
    lines.append("")
    return "\n".join(lines)


def _next_commands(batch_spec: Path, has_rows: bool) -> list[str]:
    if not has_rows:
        return [
            "uv run objgauss object-state select-bop-phase1-subset <dataset-root> --require-ready"
        ]
    return [
        (
            "uv run objgauss object-state audit-bop-local-row-batch-readiness "
            f"{batch_spec} --summary-output {batch_spec.parent / 'bop-local-row-batch-readiness.json'}"
        ),
        (
            "uv run objgauss object-state bop-local-row-batch-handoff "
            f"{batch_spec} --summary-output {batch_spec.parent / 'bop-local-row-batch-handoff-summary.json'}"
        ),
    ]


def _hard_blockers(selector: Mapping[str, Any], csv_rows: list[dict[str, str]]) -> list[str]:
    blockers = list(selector["hard_blockers"])
    if not csv_rows:
        blockers.append("no selector-ready BOP scenes were written to samples.csv")
    return blockers


def _issues(batch_spec_summary: Mapping[str, Any] | None) -> list[str]:
    if batch_spec_summary is None:
        return []
    return list(batch_spec_summary["issues"])


def _path_for_csv(value: str | Path, csv_root: Path) -> str:
    path = Path(value)
    absolute = path if path.is_absolute() else Path.cwd() / path
    return os.path.relpath(absolute, csv_root)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


__all__ = (
    "OBJECTSTATE_BOP_PHASE1_BATCH_WORKSPACE_SCHEMA",
    "objectstate_bop_phase1_batch_workspace",
    "validate_objectstate_bop_phase1_batch_workspace_summary",
)
