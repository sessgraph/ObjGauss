from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    objectstate_controlled_capture_summary,
    validate_objectstate_controlled_capture_manifest,
)

OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA = (
    "objgauss-objectstate-controlled-capture-file-audit-v1"
)


def objectstate_controlled_capture_file_audit(
    manifest: Mapping[str, Any],
    *,
    root: str | Path = ".",
    require_gaussian_files: bool = True,
    check_artifact_refs: bool = False,
) -> dict[str, Any]:
    checked_manifest = validate_objectstate_controlled_capture_manifest(manifest)
    root_path = Path(root)
    capture_summary = objectstate_controlled_capture_summary(checked_manifest)
    rgb_records = []
    gaussian_records = []
    artifact_records = []
    for frame in checked_manifest["frames"]:
        observation = frame["observation"]
        rgb_records.append(
            _file_record(
                "rgb",
                observation["rgb"],
                root_path=root_path,
                frame_id=frame["frame_id"],
            )
        )
        if "gaussian" in observation:
            gaussian_records.append(
                _file_record(
                    "gaussian",
                    observation["gaussian"],
                    root_path=root_path,
                    frame_id=frame["frame_id"],
                )
            )
        elif require_gaussian_files:
            gaussian_records.append(
                {
                    "kind": "gaussian",
                    "frame_id": frame["frame_id"],
                    "ref": "",
                    "path": "",
                    "exists": False,
                    "missing_reason": "frame missing gaussian observation reference",
                }
            )
    if check_artifact_refs:
        artifact_records = [
            _file_record("artifact_ref", ref, root_path=root_path, frame_id=None)
            for ref in checked_manifest["sample"]["artifact_refs"]
        ]
    rgb_counts = _counts(rgb_records)
    gaussian_counts = _counts(gaussian_records)
    artifact_counts = _counts(artifact_records)
    missing = [
        record
        for record in (*rgb_records, *gaussian_records, *artifact_records)
        if not record["exists"]
    ]
    readiness = {
        "rgb_files_present": rgb_counts["missing"] == 0,
        "gaussian_files_present": (
            not require_gaussian_files
            or (gaussian_counts["referenced"] > 0 and gaussian_counts["missing"] == 0)
        ),
        "artifact_refs_present": (
            not check_artifact_refs or artifact_counts["missing"] == 0
        ),
    }
    readiness["capture_bundle_files_ready"] = all(readiness.values())
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA,
        "kind": "objectstate_controlled_capture_file_audit",
        "capture_schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "status": (
            "objectstate_controlled_capture_file_audit_pass"
            if readiness["capture_bundle_files_ready"]
            else "objectstate_controlled_capture_file_audit_fail"
        ),
        "root": str(root_path),
        "sample": dict(checked_manifest["sample"]),
        "frame_count": len(checked_manifest["frames"]),
        "requirements": {
            "rgb_files_required": True,
            "gaussian_files_required": bool(require_gaussian_files),
            "artifact_refs_checked": bool(check_artifact_refs),
        },
        "file_counts": {
            "rgb": rgb_counts,
            "gaussian": gaussian_counts,
            "artifact_refs": artifact_counts,
        },
        "readiness": readiness,
        "issues": _issues(readiness, rgb_counts, gaussian_counts, artifact_counts),
        "missing_files": missing,
        "capture_summary": capture_summary,
        "missing_files_markdown": objectstate_controlled_capture_missing_files_markdown(
            missing
        ),
        "claim_policy": {
            "capture_manifest_required": True,
            "file_existence_required_for_ready_bundle": True,
            "file_audit_does_not_create_ground_truth": True,
            "file_audit_does_not_prove_model_quality": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "reconstructs_gaussians": False,
            "reads_image_pixels": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_controlled_capture_file_audit_summary(payload)


def objectstate_controlled_capture_missing_files_markdown(
    missing_files: list[dict[str, Any]],
) -> str:
    lines = [
        "| kind | frame_id | ref | path | reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in missing_files:
        lines.append(
            "| {kind} | {frame_id} | {ref} | {path} | {reason} |".format(
                kind=record.get("kind", ""),
                frame_id=record.get("frame_id") or "-",
                ref=record.get("ref") or "-",
                path=record.get("path") or "-",
                reason=record.get("missing_reason", "missing"),
            )
        )
    if not missing_files:
        lines.append("| - | - | - | - | no missing files |")
    return "\n".join(lines)


def validate_objectstate_controlled_capture_file_audit_summary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("controlled capture file audit summary must be a dict")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA:
        raise ValueError(
            f"unsupported controlled capture file audit schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_capture_file_audit":
        raise ValueError("controlled capture file audit kind is unsupported")
    if payload.get("capture_schema") != OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA:
        raise ValueError("controlled capture file audit has unsupported capture_schema")
    if payload.get("status") not in {
        "objectstate_controlled_capture_file_audit_pass",
        "objectstate_controlled_capture_file_audit_fail",
    }:
        raise ValueError("controlled capture file audit status is unsupported")
    requirements = payload.get("requirements")
    counts = payload.get("file_counts")
    readiness = payload.get("readiness")
    if not isinstance(requirements, dict):
        raise ValueError("controlled capture file audit requires requirements")
    if not isinstance(counts, dict):
        raise ValueError("controlled capture file audit requires file_counts")
    if not isinstance(readiness, dict):
        raise ValueError("controlled capture file audit requires readiness")
    for key in ("rgb", "gaussian", "artifact_refs"):
        _validate_counts(counts.get(key), key)
    if not isinstance(payload.get("missing_files"), list):
        raise ValueError("controlled capture file audit missing_files must be a list")
    for key in (
        "rgb_files_present",
        "gaussian_files_present",
        "artifact_refs_present",
        "capture_bundle_files_ready",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"controlled capture file audit readiness missing bool {key}")
    expected_status = (
        "objectstate_controlled_capture_file_audit_pass"
        if readiness["capture_bundle_files_ready"]
        else "objectstate_controlled_capture_file_audit_fail"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled capture file audit status must match readiness")
    if not isinstance(payload.get("missing_files_markdown"), str):
        raise ValueError("controlled capture file audit requires missing_files_markdown")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("capture_manifest_required")
        or not claim_policy.get("file_existence_required_for_ready_bundle")
        or not claim_policy.get("file_audit_does_not_create_ground_truth")
        or not claim_policy.get("file_audit_does_not_prove_model_quality")
    ):
        raise ValueError("controlled capture file audit must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("reconstructs_gaussians")
        or non_goals.get("reads_image_pixels")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("writes_public_samples")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError("controlled capture file audit cannot claim capture, GT, reconstruction, training, replay, diffusion, or viewer mutation")
    return payload


def _file_record(
    kind: str,
    ref: str,
    *,
    root_path: Path,
    frame_id: str | None,
) -> dict[str, Any]:
    path = Path(ref)
    resolved = path if path.is_absolute() else root_path / path
    exists = resolved.exists()
    result = {
        "kind": kind,
        "frame_id": frame_id,
        "ref": ref,
        "path": str(resolved),
        "exists": bool(exists),
    }
    if not exists:
        result["missing_reason"] = "path does not exist"
    return result


def _counts(records: list[dict[str, Any]]) -> dict[str, int]:
    referenced = len(records)
    existing = sum(1 for record in records if record["exists"])
    missing = referenced - existing
    return {
        "referenced": referenced,
        "existing": existing,
        "missing": missing,
    }


def _issues(
    readiness: Mapping[str, bool],
    rgb_counts: Mapping[str, int],
    gaussian_counts: Mapping[str, int],
    artifact_counts: Mapping[str, int],
) -> list[str]:
    issues = []
    if not readiness["rgb_files_present"]:
        issues.append(f"missing RGB files: {rgb_counts['missing']}")
    if not readiness["gaussian_files_present"]:
        issues.append(f"missing Gaussian files: {gaussian_counts['missing']}")
    if not readiness["artifact_refs_present"]:
        issues.append(f"missing sample artifact refs: {artifact_counts['missing']}")
    return issues


def _validate_counts(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"controlled capture file audit counts missing {name}")
    for key in ("referenced", "existing", "missing"):
        if not isinstance(value.get(key), int) or int(value[key]) < 0:
            raise ValueError(f"controlled capture file audit {name} count {key} invalid")
    if int(value["referenced"]) != int(value["existing"]) + int(value["missing"]):
        raise ValueError(f"controlled capture file audit {name} counts are inconsistent")
