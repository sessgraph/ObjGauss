from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.datasets.objectstate_bop_local_row_batch_spec import (
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
    validate_objectstate_bop_local_row_batch_spec,
)

OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_AUTHORING_SCHEMA = (
    "objgauss-objectstate-bop-local-row-batch-spec-authoring-v1"
)

_REQUIRED_COLUMNS = ("sample_id", "scene_root", "candidate_artifact")
_INPUT_PATH_COLUMNS = ("scene_root", "candidate_artifact", "condition_sidecar")
_OPTION_COLUMNS = {
    "condition_sidecar",
    "output_root",
    "dataset_id",
    "object_category",
    "scenario",
    "fps",
    "license_text",
    "license",
    "rgb_dir",
    "depth_dir",
    "gaussian_dir",
    "max_frames",
    "frame_step",
    "check_artifact_refs",
    "min_rgb_bytes",
    "min_gaussian_bytes",
    "require_frame_formats",
    "hash_files",
    "min_candidate_artifact_bytes",
    "hash_candidate_artifact",
    "min_identity_scenario_frames",
    "min_occlusion_fraction",
    "min_view_conditions",
    "min_lighting_conditions",
    "min_camera_motion_m",
    "identity_candidate_id",
    "identity_candidate_source",
    "max_centroid_distance",
    "prediction_policy",
    "prediction_candidate_id",
    "prediction_candidate_source",
    "prediction_confidence",
    "synthetic_smoke_passed",
    "min_real_or_public_rows",
}
_INT_COLUMNS = {
    "max_frames",
    "frame_step",
    "min_rgb_bytes",
    "min_gaussian_bytes",
    "min_candidate_artifact_bytes",
    "min_identity_scenario_frames",
    "min_view_conditions",
    "min_lighting_conditions",
    "min_real_or_public_rows",
}
_FLOAT_COLUMNS = {
    "fps",
    "min_occlusion_fraction",
    "min_camera_motion_m",
    "max_centroid_distance",
    "prediction_confidence",
}
_BOOL_COLUMNS = {
    "check_artifact_refs",
    "require_frame_formats",
    "hash_files",
    "hash_candidate_artifact",
    "synthetic_smoke_passed",
}


def objectstate_bop_local_row_batch_spec_authoring(
    samples_csv: str | Path,
    *,
    output: str | Path,
    batch_id: str = "bop-local-row-batch",
    batch_output_root: str | Path = "bop-local-row-batch",
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
    relative_paths: bool = True,
) -> dict[str, Any]:
    csv_path = Path(samples_csv)
    output_path = Path(output)
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("BOP local row batch spec authoring requires batch_id")
    if fps <= 0:
        raise ValueError("fps must be positive")
    rows = _read_sample_rows(csv_path)
    spec_root = output_path.parent
    samples = [
        _sample_from_row(
            row,
            csv_root=csv_path.parent,
            spec_root=spec_root,
            relative_paths=relative_paths,
        )
        for row in rows
    ]
    spec = {
        "schema": OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
        "kind": "objectstate_bop_local_row_batch_spec",
        "batch": {
            "batch_id": batch_id,
            "output_root": str(batch_output_root),
        },
        "defaults": {
            "dataset_id": dataset_id,
            "object_category": object_category,
            "scenario": scenario,
            "fps": float(fps),
            "license_text": license_text,
            "rgb_dir": rgb_dir,
            "depth_dir": depth_dir,
            "gaussian_dir": gaussian_dir,
        },
        "samples": samples,
        "claim_policy": {
            "local_only": True,
            "does_not_create_ground_truth": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_train_model": True,
            "does_not_claim_world_model": True,
        },
    }
    checked_spec = validate_objectstate_bop_local_row_batch_spec(spec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, checked_spec)

    records = [
        _sample_record(sample, csv_root=csv_path.parent, spec_root=spec_root)
        for sample in checked_spec["samples"]
    ]
    readiness = _readiness(records)
    payload = {
        "schema": OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_AUTHORING_SCHEMA,
        "kind": "objectstate_bop_local_row_batch_spec_authoring",
        "status": (
            "objectstate_bop_local_row_batch_spec_authoring_ready"
            if all(readiness.values())
            else "objectstate_bop_local_row_batch_spec_authoring_blocked"
        ),
        "samples_csv": str(csv_path),
        "output": str(output_path),
        "batch_spec_schema": OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
        "path_policy": {
            "csv_relative_paths_resolved_from": str(csv_path.parent),
            "spec_relative_paths_resolved_from": str(spec_root),
            "writes_relative_input_paths": bool(relative_paths),
            "sample_output_root_relative_to_batch_output_root": True,
        },
        "row_counts": {
            "samples": len(records),
            "scene_roots_present": sum(
                1 for record in records if record["paths"]["scene_root"]["exists"]
            ),
            "candidate_artifacts_present": sum(
                1
                for record in records
                if record["paths"]["candidate_artifact"]["exists"]
            ),
            "condition_sidecars_present": sum(
                1
                for record in records
                if record["paths"]["condition_sidecar"].get("exists") is True
            ),
            "condition_sidecars_declared": sum(
                1
                for record in records
                if record["paths"]["condition_sidecar"].get("declared") is True
            ),
        },
        "readiness": readiness,
        "batch_spec": checked_spec,
        "sample_records": records,
        "next_commands": _next_commands(output_path),
        "hard_blockers": _hard_blockers(records, readiness),
        "issues": _issues(records, readiness),
        "claim_policy": {
            "writes_batch_spec": True,
            "uses_csv_rows": True,
            "validates_native_batch_spec": True,
            "checks_local_input_paths": True,
            "does_not_download_dataset": True,
            "does_not_copy_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_condition_metadata": True,
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
    return validate_objectstate_bop_local_row_batch_spec_authoring_summary(payload)


def validate_objectstate_bop_local_row_batch_spec_authoring_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP local row batch spec authoring summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_AUTHORING_SCHEMA:
        raise ValueError(
            "unsupported BOP local row batch spec authoring schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_local_row_batch_spec_authoring":
        raise ValueError("BOP local row batch spec authoring kind is unsupported")
    if payload.get("status") not in {
        "objectstate_bop_local_row_batch_spec_authoring_ready",
        "objectstate_bop_local_row_batch_spec_authoring_blocked",
    }:
        raise ValueError("BOP local row batch spec authoring status is unsupported")
    for key in ("samples_csv", "output", "batch_spec_schema"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"BOP local row batch spec authoring requires {key}")
    if payload.get("batch_spec_schema") != OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA:
        raise ValueError("BOP local row batch spec authoring schema mismatch")
    batch_spec = validate_objectstate_bop_local_row_batch_spec(
        payload.get("batch_spec")
    )
    sample_records = payload.get("sample_records")
    if not isinstance(sample_records, list):
        raise ValueError("BOP local row batch spec authoring requires sample records")
    if len(sample_records) != len(batch_spec["samples"]):
        raise ValueError("BOP local row batch spec authoring sample count mismatch")
    for record, sample in zip(sample_records, batch_spec["samples"]):
        _validate_sample_record(record, sample)
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping) or not readiness:
        raise ValueError("BOP local row batch spec authoring requires readiness")
    if any(not isinstance(value, bool) for value in readiness.values()):
        raise ValueError("BOP local row batch spec authoring readiness must be bool")
    expected_readiness = _readiness(sample_records)
    if dict(readiness) != expected_readiness:
        raise ValueError("BOP local row batch spec authoring readiness mismatch")
    expected_status = (
        "objectstate_bop_local_row_batch_spec_authoring_ready"
        if all(readiness.values())
        else "objectstate_bop_local_row_batch_spec_authoring_blocked"
    )
    if payload["status"] != expected_status:
        raise ValueError("BOP local row batch spec authoring status mismatch")
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("BOP local row batch spec authoring requires row counts")
    for key in (
        "samples",
        "scene_roots_present",
        "candidate_artifacts_present",
        "condition_sidecars_present",
        "condition_sidecars_declared",
    ):
        if isinstance(row_counts.get(key), bool) or not isinstance(row_counts.get(key), int):
            raise ValueError(f"BOP local row batch spec authoring row count {key} invalid")
    if row_counts["samples"] != len(sample_records):
        raise ValueError("BOP local row batch spec authoring row count mismatch")
    path_policy = payload.get("path_policy")
    if not isinstance(path_policy, Mapping):
        raise ValueError("BOP local row batch spec authoring requires path policy")
    for key in (
        "csv_relative_paths_resolved_from",
        "spec_relative_paths_resolved_from",
    ):
        if not isinstance(path_policy.get(key), str) or not path_policy[key]:
            raise ValueError(f"BOP local row batch spec authoring path policy {key} invalid")
    for key in ("writes_relative_input_paths", "sample_output_root_relative_to_batch_output_root"):
        if not isinstance(path_policy.get(key), bool):
            raise ValueError(f"BOP local row batch spec authoring path policy {key} invalid")
    for key in ("next_commands", "hard_blockers", "issues"):
        values = payload.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"BOP local row batch spec authoring {key} must be strings")
    claim_policy = payload.get("claim_policy")
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("writes_batch_spec")
        or not claim_policy.get("uses_csv_rows")
        or not claim_policy.get("validates_native_batch_spec")
        or not claim_policy.get("checks_local_input_paths")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_copy_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_infer_condition_metadata")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_run_readiness")
        or not claim_policy.get("does_not_run_handoff")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_claim_metric_pass")
        or not claim_policy.get("does_not_claim_intervention_gate")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP local row batch spec authoring must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP local row batch spec authoring cannot claim downloads, copies, "
            "GT, condition inference, reconstruction, readiness, handoff, eval, "
            "training, public samples, replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def _read_sample_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("BOP local row batch samples CSV requires a header")
        missing = [column for column in _REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(
                "BOP local row batch samples CSV missing required columns: "
                + ", ".join(missing)
            )
        rows: list[dict[str, str]] = []
        for index, row in enumerate(reader, start=2):
            normalized = {
                key: str(value).strip()
                for key, value in row.items()
                if key is not None and value is not None
            }
            if not any(normalized.values()):
                continue
            for column in _REQUIRED_COLUMNS:
                if not normalized.get(column):
                    raise ValueError(
                        f"BOP local row batch samples CSV row {index} missing {column}"
                    )
            rows.append(normalized)
    if not rows:
        raise ValueError("BOP local row batch samples CSV contains no sample rows")
    return rows


def _sample_from_row(
    row: Mapping[str, str],
    *,
    csv_root: Path,
    spec_root: Path,
    relative_paths: bool,
) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "sample_id": row["sample_id"],
        "scene_root": _spec_path_value(
            row["scene_root"],
            csv_root=csv_root,
            spec_root=spec_root,
            relative_paths=relative_paths,
        ),
        "candidate_artifact": _spec_path_value(
            row["candidate_artifact"],
            csv_root=csv_root,
            spec_root=spec_root,
            relative_paths=relative_paths,
        ),
    }
    for key in sorted(_OPTION_COLUMNS):
        value = row.get(key)
        if value is None or value == "":
            continue
        if key in _INPUT_PATH_COLUMNS:
            sample[key] = _spec_path_value(
                value,
                csv_root=csv_root,
                spec_root=spec_root,
                relative_paths=relative_paths,
            )
        elif key in _INT_COLUMNS:
            sample[key] = int(value)
        elif key in _FLOAT_COLUMNS:
            sample[key] = float(value)
        elif key in _BOOL_COLUMNS:
            sample[key] = _parse_bool(value, key)
        else:
            sample[key] = value
    return sample


def _spec_path_value(
    value: str,
    *,
    csv_root: Path,
    spec_root: Path,
    relative_paths: bool,
) -> str:
    path = Path(value)
    resolved = path if path.is_absolute() else csv_root / path
    if not relative_paths:
        return str(resolved)
    return os.path.relpath(resolved, spec_root)


def _sample_record(
    sample: Mapping[str, Any],
    *,
    csv_root: Path,
    spec_root: Path,
) -> dict[str, Any]:
    paths = {
        "scene_root": _path_record(
            sample["scene_root"],
            csv_root=csv_root,
            spec_root=spec_root,
            expected="dir",
            declared=True,
        ),
        "candidate_artifact": _path_record(
            sample["candidate_artifact"],
            csv_root=csv_root,
            spec_root=spec_root,
            expected="file",
            declared=True,
        ),
        "condition_sidecar": _path_record(
            sample.get("condition_sidecar"),
            csv_root=csv_root,
            spec_root=spec_root,
            expected="file",
            declared=sample.get("condition_sidecar") is not None,
        ),
    }
    return {
        "sample_id": sample["sample_id"],
        "paths": paths,
        "ready": bool(
            paths["scene_root"]["exists"]
            and paths["candidate_artifact"]["exists"]
            and (
                not paths["condition_sidecar"]["declared"]
                or paths["condition_sidecar"]["exists"]
            )
        ),
        "spec": dict(sample),
    }


def _path_record(
    value: Any,
    *,
    csv_root: Path,
    spec_root: Path,
    expected: str,
    declared: bool,
) -> dict[str, Any]:
    if value is None:
        return {
            "declared": False,
            "spec_ref": None,
            "path": None,
            "expected": expected,
            "exists": None,
        }
    path = Path(str(value))
    resolved = path if path.is_absolute() else spec_root / path
    exists = resolved.exists()
    type_ok = resolved.is_dir() if expected == "dir" else resolved.is_file()
    return {
        "declared": bool(declared),
        "spec_ref": str(value),
        "path": str(resolved),
        "expected": expected,
        "exists": bool(exists and type_ok),
        "raw_exists": bool(exists),
        "csv_root": str(csv_root),
    }


def _validate_sample_record(record: Any, sample: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("BOP local row batch spec authoring sample record must map")
    if record.get("sample_id") != sample["sample_id"]:
        raise ValueError("BOP local row batch spec authoring sample id mismatch")
    if not isinstance(record.get("ready"), bool):
        raise ValueError("BOP local row batch spec authoring sample ready invalid")
    if record.get("spec") != dict(sample):
        raise ValueError("BOP local row batch spec authoring sample spec mismatch")
    paths = record.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("BOP local row batch spec authoring sample paths invalid")
    for key in ("scene_root", "candidate_artifact", "condition_sidecar"):
        _validate_path_record(paths.get(key), key)


def _validate_path_record(record: Any, key: str) -> None:
    if not isinstance(record, Mapping):
        raise ValueError(f"BOP local row batch spec authoring path {key} invalid")
    if not isinstance(record.get("declared"), bool):
        raise ValueError(f"BOP local row batch spec authoring path {key} declared invalid")
    if record["declared"]:
        for field in ("spec_ref", "path", "expected", "csv_root"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise ValueError(
                    f"BOP local row batch spec authoring path {key} missing {field}"
                )
        if not isinstance(record.get("exists"), bool):
            raise ValueError(f"BOP local row batch spec authoring path {key} exists invalid")
        if not isinstance(record.get("raw_exists"), bool):
            raise ValueError(
                f"BOP local row batch spec authoring path {key} raw_exists invalid"
            )
    elif record.get("exists") is not None:
        raise ValueError(f"BOP local row batch spec authoring path {key} undeclared invalid")


def _readiness(records: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    return {
        "sample_count_nonzero": bool(records),
        "all_scene_roots_present": bool(records)
        and all(record["paths"]["scene_root"]["exists"] for record in records),
        "all_candidate_artifacts_present": bool(records)
        and all(record["paths"]["candidate_artifact"]["exists"] for record in records),
        "all_declared_condition_sidecars_present": all(
            (not record["paths"]["condition_sidecar"]["declared"])
            or record["paths"]["condition_sidecar"]["exists"]
            for record in records
        ),
        "native_batch_spec_valid": True,
    }


def _hard_blockers(
    records: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, bool],
) -> list[str]:
    blockers: list[str] = []
    if not readiness["sample_count_nonzero"]:
        blockers.append("samples CSV produced no batch rows")
    if not readiness["all_scene_roots_present"]:
        blockers.append("one or more sample scene_root directories are missing")
    if not readiness["all_candidate_artifacts_present"]:
        blockers.append("one or more sample candidate_artifact files are missing")
    if not readiness["all_declared_condition_sidecars_present"]:
        blockers.append("one or more declared condition_sidecar files are missing")
    for record in records:
        if record["ready"]:
            continue
        for key, path_record in record["paths"].items():
            if path_record.get("declared") and not path_record.get("exists"):
                blockers.append(
                    f"{record['sample_id']}: {key} not found or wrong type: "
                    f"{path_record.get('path')}"
                )
    return _dedupe(blockers)


def _issues(
    records: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, bool],
) -> list[str]:
    issues = [
        f"readiness gate failed: {gate}"
        for gate, passed in readiness.items()
        if not passed
    ]
    for record in records:
        if record["ready"]:
            continue
        issues.append(f"sample not ready: {record['sample_id']}")
    return _dedupe(issues)


def _next_commands(output: Path) -> list[str]:
    return [
        (
            "uv run objgauss object-state audit-bop-local-row-batch-readiness "
            f"{output} --summary-output {output.parent / 'bop-local-row-batch-readiness.json'}"
        ),
        (
            "uv run objgauss object-state bop-local-row-batch-handoff "
            f"{output} --summary-output {output.parent / 'bop-local-row-batch-handoff-summary.json'}"
        ),
    ]


def _parse_bool(value: str, key: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"BOP local row batch CSV boolean column {key} is invalid")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _as_strings(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return [str(value) for value in values]


def _dedupe(values: Any) -> list[str]:
    result = []
    seen: set[str] = set()
    for value in _as_strings(values):
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


__all__ = (
    "OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_AUTHORING_SCHEMA",
    "objectstate_bop_local_row_batch_spec_authoring",
    "validate_objectstate_bop_local_row_batch_spec_authoring_summary",
)
