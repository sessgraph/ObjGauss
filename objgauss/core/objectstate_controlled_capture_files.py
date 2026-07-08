from __future__ import annotations

import hashlib
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
    min_rgb_bytes: int = 1,
    min_gaussian_bytes: int = 1,
    require_frame_formats: bool = True,
    hash_files: bool = False,
) -> dict[str, Any]:
    if min_rgb_bytes < 0:
        raise ValueError("min_rgb_bytes must be non-negative")
    if min_gaussian_bytes < 0:
        raise ValueError("min_gaussian_bytes must be non-negative")
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
                min_bytes=min_rgb_bytes,
                require_file=True,
                require_format=require_frame_formats,
                hash_file=hash_files,
            )
        )
        if "gaussian" in observation:
            gaussian_records.append(
                _file_record(
                    "gaussian",
                    observation["gaussian"],
                    root_path=root_path,
                    frame_id=frame["frame_id"],
                    min_bytes=min_gaussian_bytes,
                    require_file=True,
                    require_format=require_frame_formats,
                    hash_file=hash_files,
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
                    "is_file": False,
                    "size_bytes": None,
                    "valid": False,
                    "missing_reason": "frame missing gaussian observation reference",
                }
            )
    if check_artifact_refs:
        artifact_records = [
            _file_record(
                "artifact_ref",
                ref,
                root_path=root_path,
                frame_id=None,
                min_bytes=0,
                require_file=False,
                require_format=False,
                hash_file=False,
            )
            for ref in checked_manifest["sample"]["artifact_refs"]
        ]
    rgb_counts = _counts(rgb_records)
    gaussian_counts = _counts(gaussian_records)
    artifact_counts = _counts(artifact_records)
    missing = [
        record
        for record in (*rgb_records, *gaussian_records, *artifact_records)
        if not record["valid"]
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
        "frame_formats_valid": (
            not require_frame_formats
            or (
                rgb_counts["missing"] == 0
                and (
                    not require_gaussian_files
                    or (
                        gaussian_counts["referenced"] > 0
                        and gaussian_counts["missing"] == 0
                    )
                )
            )
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
            "frame_refs_must_be_files": True,
            "min_rgb_bytes": int(min_rgb_bytes),
            "min_gaussian_bytes": int(min_gaussian_bytes),
            "frame_file_formats_required": bool(require_frame_formats),
            "file_hashes_included": bool(hash_files),
        },
        "file_counts": {
            "rgb": rgb_counts,
            "gaussian": gaussian_counts,
            "artifact_refs": artifact_counts,
        },
        "file_records": {
            "rgb": rgb_records,
            "gaussian": gaussian_records,
            "artifact_refs": artifact_records,
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
            "nonempty_frame_files_required_for_ready_bundle": True,
            "recognized_frame_formats_required_for_ready_bundle": bool(
                require_frame_formats
            ),
            "file_audit_does_not_create_ground_truth": True,
            "file_audit_does_not_prove_model_quality": True,
            "format_audit_does_not_decode_image_pixels": True,
            "format_audit_does_not_fully_parse_gaussian_payload": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "reconstructs_gaussians": False,
            "reads_image_pixels": False,
            "fully_parses_gaussian_files": False,
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
    file_records = payload.get("file_records")
    if not isinstance(file_records, dict):
        raise ValueError("controlled capture file audit requires file_records")
    for key in ("rgb", "gaussian", "artifact_refs"):
        if not isinstance(file_records.get(key), list):
            raise ValueError(f"controlled capture file audit file_records missing {key}")
    for key in (
        "rgb_files_present",
        "gaussian_files_present",
        "artifact_refs_present",
        "frame_formats_valid",
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
        or not claim_policy.get("nonempty_frame_files_required_for_ready_bundle")
        or not isinstance(
            claim_policy.get("recognized_frame_formats_required_for_ready_bundle"),
            bool,
        )
        or not claim_policy.get("file_audit_does_not_create_ground_truth")
        or not claim_policy.get("file_audit_does_not_prove_model_quality")
        or not claim_policy.get("format_audit_does_not_decode_image_pixels")
        or not claim_policy.get("format_audit_does_not_fully_parse_gaussian_payload")
    ):
        raise ValueError("controlled capture file audit must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("reconstructs_gaussians")
        or non_goals.get("reads_image_pixels")
        or non_goals.get("fully_parses_gaussian_files")
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
    min_bytes: int,
    require_file: bool,
    require_format: bool,
    hash_file: bool,
) -> dict[str, Any]:
    path = Path(ref)
    resolved = path if path.is_absolute() else root_path / path
    exists = resolved.exists()
    is_file = bool(exists and resolved.is_file())
    size_bytes = resolved.stat().st_size if is_file else None
    valid = bool(exists)
    missing_reason = None
    if not exists:
        valid = False
        missing_reason = "path does not exist"
    elif require_file and not is_file:
        valid = False
        missing_reason = "path is not a file"
    elif is_file and size_bytes is not None and size_bytes < min_bytes:
        valid = False
        missing_reason = (
            "file smaller than required minimum bytes "
            f"({size_bytes} < {min_bytes})"
        )
    format_record = None
    if require_format and is_file and kind in {"rgb", "gaussian"}:
        format_record = _frame_format_record(kind, resolved, size_bytes=size_bytes)
        if not format_record["valid"] and valid:
            valid = False
            missing_reason = format_record["reason"]
    result = {
        "kind": kind,
        "frame_id": frame_id,
        "ref": ref,
        "path": str(resolved),
        "exists": bool(exists),
        "is_file": is_file,
        "size_bytes": size_bytes,
        "valid": valid,
    }
    if format_record is not None:
        result["format"] = format_record
    if missing_reason is not None:
        result["missing_reason"] = missing_reason
    if hash_file and valid and is_file:
        result["sha256"] = _sha256_file(resolved)
    return result


def _frame_format_record(
    kind: str,
    path: Path,
    *,
    size_bytes: int | None,
) -> dict[str, Any]:
    prefix = _read_prefix(path)
    if kind == "rgb":
        return _rgb_format_record(prefix)
    if kind == "gaussian":
        return _gaussian_format_record(path, prefix, size_bytes=size_bytes)
    return {
        "valid": False,
        "format": "unsupported",
        "reason": f"unsupported frame format audit kind: {kind}",
    }


def _rgb_format_record(prefix: bytes) -> dict[str, Any]:
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return {"valid": True, "format": "png", "reason": "recognized PNG signature"}
    if prefix.startswith(b"\xff\xd8\xff"):
        return {"valid": True, "format": "jpeg", "reason": "recognized JPEG signature"}
    if len(prefix) >= 12 and prefix[:4] == b"RIFF" and prefix[8:12] == b"WEBP":
        return {"valid": True, "format": "webp", "reason": "recognized WebP signature"}
    if prefix.startswith((b"P6\n", b"P6\r\n", b"P3\n", b"P3\r\n")):
        return {"valid": True, "format": "ppm", "reason": "recognized PPM signature"}
    return {
        "valid": False,
        "format": "unknown",
        "reason": "unrecognized rgb frame format; expected PNG, JPEG, WebP, or PPM",
    }


def _gaussian_format_record(
    path: Path,
    prefix: bytes,
    *,
    size_bytes: int | None,
) -> dict[str, Any]:
    if prefix.startswith((b"ply\n", b"ply\r\n")):
        return _ply_format_record(prefix)
    if path.suffix.lower() == ".splat":
        valid_size = bool(size_bytes is not None and size_bytes >= 32 and size_bytes % 32 == 0)
        return {
            "valid": valid_size,
            "format": "splat" if valid_size else "invalid_splat",
            "reason": (
                "recognized raw .splat size multiple"
                if valid_size
                else "raw .splat must be non-empty and a multiple of 32 bytes"
            ),
        }
    return {
        "valid": False,
        "format": "unknown",
        "reason": "unrecognized gaussian frame format; expected PLY or raw .splat",
    }


def _ply_format_record(prefix: bytes) -> dict[str, Any]:
    header_end = prefix.find(b"end_header")
    if header_end < 0:
        return {
            "valid": False,
            "format": "invalid_ply",
            "reason": "PLY header missing end_header",
        }
    header_text = prefix[:header_end].decode("ascii", errors="ignore")
    has_format = (
        "format ascii 1.0" in header_text
        or "format binary_little_endian 1.0" in header_text
        or "format binary_big_endian 1.0" in header_text
    )
    has_vertex = "element vertex " in header_text
    valid = has_format and has_vertex
    return {
        "valid": valid,
        "format": "ply" if valid else "invalid_ply",
        "reason": (
            "recognized PLY header with vertex element"
            if valid
            else "PLY header requires format and element vertex declarations"
        ),
    }


def _read_prefix(path: Path, *, max_bytes: int = 4096) -> bytes:
    with path.open("rb") as handle:
        return handle.read(max_bytes)


def _counts(records: list[dict[str, Any]]) -> dict[str, int]:
    referenced = len(records)
    existing = sum(1 for record in records if record["exists"])
    valid = sum(1 for record in records if record["valid"])
    missing = referenced - valid
    return {
        "referenced": referenced,
        "existing": existing,
        "valid": valid,
        "missing": missing,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _issues(
    readiness: Mapping[str, bool],
    rgb_counts: Mapping[str, int],
    gaussian_counts: Mapping[str, int],
    artifact_counts: Mapping[str, int],
) -> list[str]:
    issues = []
    if not readiness["rgb_files_present"]:
        issues.append(f"invalid or missing RGB files: {rgb_counts['missing']}")
    if not readiness["gaussian_files_present"]:
        issues.append(f"invalid or missing Gaussian files: {gaussian_counts['missing']}")
    if not readiness["artifact_refs_present"]:
        issues.append(
            f"invalid or missing sample artifact refs: {artifact_counts['missing']}"
        )
    return issues


def _validate_counts(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"controlled capture file audit counts missing {name}")
    for key in ("referenced", "existing", "valid", "missing"):
        if not isinstance(value.get(key), int) or int(value[key]) < 0:
            raise ValueError(f"controlled capture file audit {name} count {key} invalid")
    if int(value["referenced"]) != int(value["valid"]) + int(value["missing"]):
        raise ValueError(f"controlled capture file audit {name} counts are inconsistent")
    if int(value["valid"]) > int(value["existing"]):
        raise ValueError(f"controlled capture file audit {name} valid exceeds existing")
