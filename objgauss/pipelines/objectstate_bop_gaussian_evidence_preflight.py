from __future__ import annotations

import shutil
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Mapping

from objgauss.datasets.objectstate_bop_capture_adapter import (
    BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
    OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
    objectstate_bop_capture_acceptance_summary,
    validate_objectstate_bop_capture_acceptance_summary,
)
from objgauss.datasets.objectstate_controlled_capture_environment import (
    OBJECTSTATE_CONTROLLED_CAPTURE_ENVIRONMENT_SCHEMA,
    objectstate_controlled_capture_environment,
    validate_objectstate_controlled_capture_environment_summary,
)
from objgauss.datasets.objectstate_controlled_capture_files import (
    objectstate_controlled_capture_missing_files_markdown,
)

OBJECTSTATE_BOP_GAUSSIAN_EVIDENCE_PREFLIGHT_SCHEMA = (
    "objgauss-objectstate-bop-gaussian-evidence-preflight-v1"
)

__all__ = (
    "OBJECTSTATE_BOP_GAUSSIAN_EVIDENCE_PREFLIGHT_SCHEMA",
    "objectstate_bop_gaussian_evidence_preflight",
    "validate_objectstate_bop_gaussian_evidence_preflight_summary",
)

CommandResolver = Callable[[str], str | None]
Importer = Callable[[str], Any]


def objectstate_bop_gaussian_evidence_preflight(
    scene_root: str | Path,
    *,
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
    min_rgb_bytes: int = 1,
    min_gaussian_bytes: int = 1,
    require_frame_formats: bool = True,
    hash_files: bool = False,
    dev_root: str | Path = "/dev",
    command_resolver: CommandResolver = shutil.which,
    importer: Importer = import_module,
) -> dict[str, Any]:
    root = Path(scene_root)
    environment = objectstate_controlled_capture_environment(
        dev_root=dev_root,
        command_resolver=command_resolver,
        importer=importer,
    )
    acceptance, acceptance_issue = _acceptance_or_issue(
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
        min_rgb_bytes=min_rgb_bytes,
        min_gaussian_bytes=min_gaussian_bytes,
        require_frame_formats=require_frame_formats,
        hash_files=hash_files,
    )
    gaussian_records = _gaussian_records(acceptance)
    missing_gaussian_records = [
        record for record in gaussian_records if not record["valid"]
    ]
    readiness = {
        "bop_acceptance_available": acceptance is not None,
        "rgb_files_present": bool(
            acceptance and acceptance["readiness"]["rgb_files_present"]
        ),
        "gaussian_refs_expected": bool(gaussian_records),
        "gaussian_files_present": bool(
            acceptance and acceptance["readiness"]["gaussian_files_present"]
        ),
        "phase1_gaussian_evidence_ready": bool(
            acceptance
            and acceptance["readiness"]["phase1_gaussian_evidence_ready"]
        ),
        "gaussian_reconstruction_tools_ready": bool(
            environment["readiness"]["gaussian_reconstruction_ready"]
        ),
    }
    readiness["ready_for_phase1_acceptance"] = bool(
        readiness["bop_acceptance_available"]
        and readiness["phase1_gaussian_evidence_ready"]
    )
    payload = {
        "schema": OBJECTSTATE_BOP_GAUSSIAN_EVIDENCE_PREFLIGHT_SCHEMA,
        "kind": "objectstate_bop_gaussian_evidence_preflight",
        "status": (
            "objectstate_bop_gaussian_evidence_ready"
            if readiness["ready_for_phase1_acceptance"]
            else "objectstate_bop_gaussian_evidence_blocked"
        ),
        "scene_root": str(root),
        "sample_id": sample_id,
        "dataset_id": dataset_id,
        "acceptance_schema": OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
        "environment_schema": OBJECTSTATE_CONTROLLED_CAPTURE_ENVIRONMENT_SCHEMA,
        "requirements": {
            "gaussian_dir": gaussian_dir,
            "min_rgb_bytes": int(min_rgb_bytes),
            "min_gaussian_bytes": int(min_gaussian_bytes),
            "frame_file_formats_required": bool(require_frame_formats),
            "file_hashes_included": bool(hash_files),
            "condition_sidecar_loaded": bool(condition_sidecar),
        },
        "readiness": readiness,
        "row_counts": {
            "selected_frames": (
                acceptance["adapter"]["row_counts"]["frames"] if acceptance else 0
            ),
            "expected_gaussian_files": len(gaussian_records),
            "missing_gaussian_files": len(missing_gaussian_records),
        },
        "expected_gaussian_files": gaussian_records,
        "missing_gaussian_files": missing_gaussian_records,
        "missing_gaussian_files_markdown": (
            objectstate_controlled_capture_missing_files_markdown(
                missing_gaussian_records
            )
        ),
        "acceptance": acceptance,
        "acceptance_issue": acceptance_issue,
        "environment": environment,
        "hard_blockers": _hard_blockers(
            readiness,
            acceptance_issue=acceptance_issue,
            missing_count=len(missing_gaussian_records),
        ),
        "next_actions": _next_actions(
            readiness,
            acceptance_issue=acceptance_issue,
            missing_count=len(missing_gaussian_records),
        ),
        "next_commands": _next_commands(
            root,
            sample_id=sample_id,
            dataset_id=dataset_id,
            gaussian_dir=gaussian_dir,
            condition_sidecar=condition_sidecar,
        ),
        "claim_policy": {
            "read_only_preflight": True,
            "runs_bop_acceptance_audit": True,
            "runs_environment_preflight": True,
            "does_not_download_dataset": True,
            "does_not_copy_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_condition_metadata": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_run_handoff": True,
            "does_not_train_model": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "copies_dataset": False,
            "creates_ground_truth": False,
            "infers_condition_metadata": False,
            "reconstructs_gaussians": False,
            "runs_handoff": False,
            "trains_model": False,
            "writes_public_samples": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_bop_gaussian_evidence_preflight_summary(payload)


def validate_objectstate_bop_gaussian_evidence_preflight_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP Gaussian evidence preflight summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_GAUSSIAN_EVIDENCE_PREFLIGHT_SCHEMA:
        raise ValueError(
            "unsupported BOP Gaussian evidence preflight schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_gaussian_evidence_preflight":
        raise ValueError("BOP Gaussian evidence preflight kind is unsupported")
    if payload.get("status") not in {
        "objectstate_bop_gaussian_evidence_ready",
        "objectstate_bop_gaussian_evidence_blocked",
    }:
        raise ValueError("BOP Gaussian evidence preflight status is unsupported")
    for key in ("scene_root", "sample_id", "dataset_id"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"BOP Gaussian evidence preflight requires {key}")
    if payload.get("acceptance_schema") != OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA:
        raise ValueError("BOP Gaussian evidence preflight acceptance_schema mismatch")
    if (
        payload.get("environment_schema")
        != OBJECTSTATE_CONTROLLED_CAPTURE_ENVIRONMENT_SCHEMA
    ):
        raise ValueError("BOP Gaussian evidence preflight environment_schema mismatch")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping) or not readiness:
        raise ValueError("BOP Gaussian evidence preflight requires readiness")
    if any(not isinstance(value, bool) for value in readiness.values()):
        raise ValueError("BOP Gaussian evidence preflight readiness values must be bool")
    expected_status = (
        "objectstate_bop_gaussian_evidence_ready"
        if readiness["ready_for_phase1_acceptance"]
        else "objectstate_bop_gaussian_evidence_blocked"
    )
    if payload["status"] != expected_status:
        raise ValueError("BOP Gaussian evidence preflight status mismatch")
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("BOP Gaussian evidence preflight requires row_counts")
    for key in (
        "selected_frames",
        "expected_gaussian_files",
        "missing_gaussian_files",
    ):
        if not isinstance(row_counts.get(key), int):
            raise ValueError(f"BOP Gaussian evidence preflight row count {key} invalid")
    expected_files = payload.get("expected_gaussian_files")
    missing_files = payload.get("missing_gaussian_files")
    if not isinstance(expected_files, list) or not isinstance(missing_files, list):
        raise ValueError("BOP Gaussian evidence preflight file lists must be lists")
    for record in expected_files:
        _validate_gaussian_record(record)
    for record in missing_files:
        _validate_gaussian_record(record)
        if record["valid"]:
            raise ValueError("BOP Gaussian evidence missing list cannot include valid files")
    if row_counts["expected_gaussian_files"] != len(expected_files):
        raise ValueError("BOP Gaussian evidence expected file count mismatch")
    if row_counts["missing_gaussian_files"] != len(missing_files):
        raise ValueError("BOP Gaussian evidence missing file count mismatch")
    acceptance = payload.get("acceptance")
    if acceptance is not None:
        validate_objectstate_bop_capture_acceptance_summary(acceptance)
    environment = validate_objectstate_controlled_capture_environment_summary(
        payload.get("environment")
    )
    if (
        readiness["gaussian_reconstruction_tools_ready"]
        != environment["readiness"]["gaussian_reconstruction_ready"]
    ):
        raise ValueError("BOP Gaussian evidence tool readiness mismatch")
    for key in ("hard_blockers", "next_actions", "next_commands"):
        values = payload.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"BOP Gaussian evidence preflight {key} must be strings")
    if not isinstance(payload.get("missing_gaussian_files_markdown"), str):
        raise ValueError("BOP Gaussian evidence preflight requires missing markdown")
    claim_policy = payload.get("claim_policy")
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("read_only_preflight")
        or not claim_policy.get("runs_bop_acceptance_audit")
        or not claim_policy.get("runs_environment_preflight")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_copy_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_infer_condition_metadata")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_run_handoff")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP Gaussian evidence preflight must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP Gaussian evidence preflight cannot claim downloads, copies, "
            "GT, condition inference, reconstruction, handoff, training, public "
            "samples, or viewer mutation"
        )
    return dict(payload)


def _acceptance_or_issue(
    scene_root: Path,
    **kwargs: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        acceptance = objectstate_bop_capture_acceptance_summary(
            scene_root,
            require_gaussian_files=True,
            include_gaussian_refs=True,
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - preflight reports blocked inputs.
        return None, f"BOP acceptance audit failed: {exc}"
    return acceptance, None


def _gaussian_records(acceptance: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if acceptance is None:
        return []
    records = acceptance["file_audit"]["file_records"]["gaussian"]
    return [
        {
            "kind": str(record.get("kind", "gaussian")),
            "frame_id": str(record.get("frame_id", "")),
            "ref": str(record.get("ref", "")),
            "path": str(record.get("path", "")),
            "exists": bool(record.get("exists")),
            "is_file": bool(record.get("is_file")),
            "size_bytes": record.get("size_bytes"),
            "valid": bool(record.get("valid")),
            "missing_reason": (
                str(record["missing_reason"])
                if record.get("missing_reason") is not None
                else None
            ),
        }
        for record in records
    ]


def _validate_gaussian_record(record: Any) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("BOP Gaussian evidence file record must map")
    for key in ("kind", "frame_id", "ref", "path"):
        if not isinstance(record.get(key), str):
            raise ValueError(f"BOP Gaussian evidence file record requires {key}")
    for key in ("exists", "is_file", "valid"):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"BOP Gaussian evidence file record requires bool {key}")
    size = record.get("size_bytes")
    if size is not None and not isinstance(size, int):
        raise ValueError("BOP Gaussian evidence file size must be int or null")
    reason = record.get("missing_reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("BOP Gaussian evidence missing_reason must be string or null")


def _hard_blockers(
    readiness: Mapping[str, bool],
    *,
    acceptance_issue: str | None,
    missing_count: int,
) -> list[str]:
    blockers = []
    if acceptance_issue is not None:
        blockers.append(acceptance_issue)
    if not readiness["rgb_files_present"] and acceptance_issue is None:
        blockers.append("BOP scene RGB files are missing or invalid")
    if missing_count:
        blockers.append(
            f"{missing_count} selected frame Gaussian files are missing or invalid"
        )
    if (
        not readiness["phase1_gaussian_evidence_ready"]
        and not readiness["gaussian_reconstruction_tools_ready"]
    ):
        blockers.append(
            "Gaussian reconstruction tools are not ready on this host "
            "(need ns-train/ns-export and COLMAP or ns-process-data)"
        )
    return blockers


def _next_actions(
    readiness: Mapping[str, bool],
    *,
    acceptance_issue: str | None,
    missing_count: int,
) -> list[str]:
    if acceptance_issue is not None:
        return [
            "select or place a local BOP scene root that passes the BOP adapter",
            "rerun select-bop-phase1-subset if starting from a dataset root",
        ]
    if readiness["phase1_gaussian_evidence_ready"]:
        return [
            "run accept-bop-capture-scene with --require-gaussian-files to write the accepted manifest",
            "create or bind the ObjectState candidate artifact before identity/prediction handoff",
        ]
    actions = []
    if missing_count:
        actions.append(
            "reconstruct or place one Gaussian PLY/.splat under the scene gaussian dir for each selected BOP frame"
        )
    if not readiness["gaussian_reconstruction_tools_ready"]:
        actions.append(
            "rerun this preflight on a reconstruction host with COLMAP/Nerfstudio tools visible"
        )
    actions.append("rerun audit-bop-gaussian-evidence after Gaussian files are present")
    return actions


def _next_commands(
    scene_root: Path,
    *,
    sample_id: str,
    dataset_id: str,
    gaussian_dir: str,
    condition_sidecar: str | Path | None,
) -> list[str]:
    condition_arg = (
        f" --condition-sidecar {condition_sidecar}" if condition_sidecar else ""
    )
    return [
        (
            "uv run objgauss object-state accept-bop-capture-scene "
            f"{scene_root} --sample-id {sample_id} --dataset-id {dataset_id} "
            f"--gaussian-dir {gaussian_dir}{condition_arg} "
            "--require-gaussian-files"
        ),
        (
            "uv run objgauss object-state audit-bop-phase1-local-row "
            f"{scene_root} --output-root outputs/captures/{sample_id} "
            f"--sample-id {sample_id} --dataset-id {dataset_id}{condition_arg}"
        ),
    ]
