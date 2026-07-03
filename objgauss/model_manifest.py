from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from objgauss.core.chunk_index import CHUNK_INDEX_SCHEMA, read_chunk_index, validate_chunk_index
from objgauss.core.trainable_artifact import (
    TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
    validate_trainable_kernel_model_artifact,
)

MODEL_ARTIFACT_MANIFEST_SCHEMA = "objgauss-model-artifact-manifest-v1"

ARTIFACT_ROLES = frozenset(
    {
        "quick_splat",
        "object_edit",
        "diagnostic_full",
        "source_gaussian",
        "object_field",
        "training_summary",
        "trainable_kernel",
        "quality_report",
        "compressed_chunked",
    }
)

DELIVERY_TIERS = frozenset(
    {
        "browser_quick",
        "browser_edit",
        "diagnostic",
        "training_internal",
        "quality_evidence",
    }
)

BROWSER_READY_TIERS = frozenset({"browser_quick", "browser_edit"})
BROWSER_ARTIFACT_ROLES = frozenset({"quick_splat", "object_edit", "trainable_kernel", "compressed_chunked"})


@dataclass(frozen=True)
class ModelManifestValidationResult:
    manifest_id: str | None
    asset_id: str | None
    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    artifact_count: int
    browser_ready_artifacts: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "objgauss-model-artifact-validation-v1",
            "manifest_id": self.manifest_id,
            "asset_id": self.asset_id,
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "artifact_count": int(self.artifact_count),
            "browser_ready_artifacts": int(self.browser_ready_artifacts),
        }


def build_model_artifact(
    *,
    role: str,
    path: str | Path,
    format: str,
    delivery_tier: str,
    browser_ready: bool | None = None,
    gaussian_count: int | None = None,
    object_count: int | None = None,
    byte_size: int | None = None,
    sha256: str | None = None,
    label: str | None = None,
    note: str | None = None,
    chunk_index: dict[str, Any] | None = None,
    compression: dict[str, Any] | None = None,
    lod: dict[str, Any] | None = None,
    object_id_coverage: dict[str, Any] | None = None,
    compute_hash: bool = False,
) -> dict[str, Any]:
    _validate_role(role)
    _validate_delivery_tier(delivery_tier)
    path = Path(path)
    resolved_browser_ready = delivery_tier in BROWSER_READY_TIERS if browser_ready is None else bool(browser_ready)
    if path.exists():
        if byte_size is None:
            byte_size = path.stat().st_size
        if compute_hash and sha256 is None:
            sha256 = _sha256(path)
    artifact = {
        "role": role,
        "path": str(path),
        "format": format,
        "delivery_tier": delivery_tier,
        "browser_ready": resolved_browser_ready,
        "gaussian_count": _optional_positive_int(gaussian_count, "gaussian_count"),
        "object_count": _optional_positive_int(object_count, "object_count"),
        "byte_size": _optional_positive_int(byte_size, "byte_size"),
        "sha256": sha256,
        "label": label,
        "note": note,
        "chunk_index": chunk_index,
        "compression": compression,
        "lod": lod,
        "object_id_coverage": object_id_coverage,
    }
    return {key: value for key, value in artifact.items() if value is not None}


def build_compressed_chunked_artifact(
    *,
    payload_path: str | Path,
    chunk_index: dict[str, Any] | str | Path,
    chunk_index_path: str | Path | None = None,
    delivery_tier: str = "browser_edit",
    browser_ready: bool | None = None,
    label: str | None = None,
    note: str | None = None,
    compute_hash: bool = True,
) -> dict[str, Any]:
    """Build a browser-ready compressed chunked artifact from an OGC payload index."""

    payload_path = Path(payload_path)
    index, resolved_index_path = _resolve_chunk_index(
        chunk_index,
        chunk_index_path=chunk_index_path,
    )
    index_validation = validate_chunk_index(index)
    if not index_validation.passed:
        raise ValueError("; ".join(index_validation.errors))
    if resolved_index_path is None:
        raise ValueError("chunk_index_path is required when chunk_index is provided as an object")

    payload_metadata = index.get("payload") if isinstance(index.get("payload"), dict) else {}
    byte_size = _positive_int_or_none(payload_metadata.get("byte_size"))
    if byte_size is None and payload_path.exists():
        byte_size = payload_path.stat().st_size
    sha256 = _optional_string(payload_metadata.get("sha256"))
    if sha256 is None and compute_hash and payload_path.exists():
        sha256 = _sha256(payload_path)

    artifact = build_model_artifact(
        role="compressed_chunked",
        path=payload_path,
        format=_optional_string(payload_metadata.get("format")) or _format_from_path(payload_path, fallback=".ogc"),
        delivery_tier=delivery_tier,
        browser_ready=browser_ready,
        gaussian_count=_positive_int_or_none(index.get("gaussian_count")),
        object_count=_positive_int_or_none(index.get("object_count")),
        byte_size=byte_size,
        sha256=sha256,
        label=label,
        note=note,
        chunk_index={
            "schema": index.get("schema"),
            "path": str(resolved_index_path),
            "chunk_count": len(index.get("chunks", [])) if isinstance(index.get("chunks"), list) else None,
            "sort_key": index.get("sort_key"),
            "chunk_size_target": index.get("chunk_size_target"),
        },
        compression=_compression_metadata(index),
        lod=_lod_metadata(index),
        object_id_coverage=index.get("object_id_coverage"),
    )
    errors: list[str] = []
    warnings: list[str] = []
    _validate_artifact(artifact, index=0, errors=errors, warnings=warnings)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def build_model_artifact_manifest(
    *,
    manifest_id: str,
    asset_id: str,
    name: str,
    source: dict[str, Any],
    license: str,
    artifacts: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    stage: str = "development",
    gaussian_count: int | None = None,
    object_count: int | None = None,
    quality_evidence: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    limitations: list[str] | tuple[str, ...] = (),
    created_from: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not manifest_id:
        raise ValueError("manifest_id is required")
    if not asset_id:
        raise ValueError("asset_id is required")
    if not name:
        raise ValueError("name is required")
    if not license:
        raise ValueError("license is required")
    if not isinstance(source, dict) or not source:
        raise ValueError("source must be a non-empty object")
    if not artifacts:
        raise ValueError("at least one artifact is required")

    payload = {
        "schema": MODEL_ARTIFACT_MANIFEST_SCHEMA,
        "manifest_id": manifest_id,
        "asset_id": asset_id,
        "name": name,
        "stage": stage,
        "source": source,
        "license": license,
        "counts": {
            "gaussians": _optional_positive_int(gaussian_count, "gaussian_count"),
            "objects": _optional_positive_int(object_count, "object_count"),
        },
        "artifacts": list(artifacts),
        "quality_evidence": list(quality_evidence),
        "limitations": list(limitations),
        "created_from": created_from or {},
    }
    validation = validate_model_artifact_manifest(payload, require_browser_ready=True)
    if not validation.passed:
        raise ValueError("; ".join(validation.errors))
    return payload


def manifest_from_training_output(
    training_manifest: str | Path,
    *,
    manifest_id: str | None = None,
    name: str | None = None,
    source: dict[str, Any] | None = None,
    license: str,
) -> dict[str, Any]:
    path = Path(training_manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    asset_id = _required_string(payload.get("asset_id"), "asset_id")
    gaussian_count = _optional_positive_int(payload.get("gaussian_count"), "gaussian_count")
    object_count = _optional_positive_int(payload.get("slots"), "slots")
    artifacts: list[dict[str, Any]] = []
    if payload.get("public_splat") or payload.get("splat_path"):
        artifacts.append(
            build_model_artifact(
                role="quick_splat",
                path=payload.get("public_splat") or payload["splat_path"],
                format=".splat",
                delivery_tier="browser_quick",
                gaussian_count=gaussian_count,
            )
        )
    if payload.get("public_object_ply") or payload.get("object_ply"):
        artifacts.append(
            build_model_artifact(
                role="object_edit",
                path=payload.get("public_object_ply") or payload["object_ply"],
                format=".ply",
                delivery_tier="browser_edit",
                gaussian_count=gaussian_count,
                object_count=object_count,
            )
        )
    if payload.get("gaussian_ply"):
        artifacts.append(
            build_model_artifact(
                role="diagnostic_full",
                path=payload["gaussian_ply"],
                format=".ply",
                delivery_tier="diagnostic",
                browser_ready=False,
                gaussian_count=gaussian_count,
                note="Full diagnostic Gaussian PLY; the frontend must not request this by default.",
            )
        )
    if payload.get("trained_field"):
        artifacts.append(
            build_model_artifact(
                role="object_field",
                path=payload["trained_field"],
                format=".npz",
                delivery_tier="training_internal",
                browser_ready=False,
                gaussian_count=gaussian_count,
                object_count=object_count,
            )
        )
    quality_evidence = []
    if payload.get("training"):
        quality_evidence.append(
            {
                "kind": "mask_training_summary",
                "source": str(path),
                "summary": payload["training"],
            }
        )
    return build_model_artifact_manifest(
        manifest_id=manifest_id or f"{asset_id}-model-artifacts",
        asset_id=asset_id,
        name=name or asset_id,
        source=source
        or {
            "type": payload.get("gaussian_source") or "external_training_output",
            "dataset": payload.get("dataset"),
            "input": payload.get("input"),
        },
        license=license,
        artifacts=artifacts,
        gaussian_count=gaussian_count,
        object_count=object_count,
        quality_evidence=quality_evidence,
        limitations=[
            "Development-stage model artifact contract; not a stable release manifest.",
        ],
        created_from={
            "training_manifest": str(path),
            "mask_manifest": payload.get("mask_manifest"),
            "input_format": payload.get("input_format"),
        },
    )


def manifest_from_trainable_kernel_model_artifact(
    trainable_artifact: str | Path | dict[str, Any],
    *,
    artifact_path: str | Path | None = None,
    quality_report_path: str | Path | None = None,
    quality_report_file: str | Path | None = None,
    manifest_id: str | None = None,
    asset_id: str | None = None,
    name: str | None = None,
    source: dict[str, Any] | None = None,
    license: str,
    compute_hash: bool = True,
) -> dict[str, Any]:
    """Wrap a trainable kernel model artifact in a viewer-ready model manifest."""

    if isinstance(trainable_artifact, dict):
        payload = dict(trainable_artifact)
        artifact_file: Path | None = None
    else:
        artifact_file = Path(trainable_artifact)
        payload = json.loads(artifact_file.read_text(encoding="utf-8"))
    validate_trainable_kernel_model_artifact(payload)

    if artifact_path is None:
        if artifact_file is None:
            raise ValueError("artifact_path is required when trainable_artifact is provided as a dict")
        artifact_path = artifact_file

    training = payload["training"]
    sample = payload.get("source", {}).get("sample")
    gaussian_count = _trainable_gaussian_count(payload)
    object_count = _optional_positive_int(training.get("slots"), "slots")
    resolved_asset_id = asset_id or _trainable_asset_id(payload, artifact_file)
    artifact_route = str(artifact_path)
    byte_size = artifact_file.stat().st_size if artifact_file is not None and artifact_file.exists() else None
    sha256 = _sha256(artifact_file) if compute_hash and artifact_file is not None and artifact_file.exists() else None
    artifacts = [
        build_model_artifact(
            role="trainable_kernel",
            path=artifact_route,
            format=".json",
            delivery_tier="browser_edit",
            browser_ready=True,
            gaussian_count=gaussian_count,
            object_count=object_count,
            byte_size=byte_size,
            sha256=sha256,
            label=str(payload.get("label") or "trainable-kernel-model-artifact"),
            note="Trainable kernel Debug OS artifact; browser-ready for ObjectState inspection.",
        )
    ]
    if quality_report_path is not None:
        artifacts.append(
            _quality_report_artifact(
                quality_report_path=quality_report_path,
                quality_report_file=quality_report_file,
                compute_hash=compute_hash,
            )
        )

    return build_model_artifact_manifest(
        manifest_id=manifest_id or f"{resolved_asset_id}-trainable-model-artifacts",
        asset_id=resolved_asset_id,
        name=name or str(payload.get("label") or resolved_asset_id),
        source=source
        or {
            "type": "trainable_kernel_model_artifact",
            "input": payload.get("source", {}).get("input"),
            "target_source": sample.get("target_source") if isinstance(sample, dict) else None,
        },
        license=license,
        artifacts=artifacts,
        gaussian_count=gaussian_count,
        object_count=object_count,
        quality_evidence=[
            {
                "kind": "trainable_kernel_training_summary",
                "source": artifact_route,
                "summary": training,
            },
            *(
                [
                    {
                        "kind": "trainable_kernel_renderer_api",
                        "source": artifact_route,
                        "summary": payload["renderer_api"],
                    }
                ]
                if isinstance(payload.get("renderer_api"), dict)
                else []
            ),
        ],
        limitations=[
            "Development-stage trainable kernel handoff; source assets remain governed by their original license.",
            "This manifest exposes a small Debug OS JSON artifact, not a production model release.",
        ],
        created_from={
            "trainable_model_artifact": artifact_route,
            "schema": TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
            "input": payload.get("source", {}).get("input"),
            "sample": sample if isinstance(sample, dict) else None,
        },
    )


def manifest_from_sample_bundle(
    sample_bundle: str | Path,
    *,
    manifest_id: str | None = None,
    name: str | None = None,
    source: dict[str, Any] | None = None,
    license: str,
) -> dict[str, Any]:
    path = Path(sample_bundle)
    payload = json.loads(path.read_text(encoding="utf-8"))
    asset_id = _required_string(payload.get("asset_id"), "asset_id")
    gaussian_count = _optional_positive_int(payload.get("gaussian_count"), "gaussian_count")
    object_count = _optional_positive_int(payload.get("slot_count"), "slot_count")
    artifacts: list[dict[str, Any]] = []
    if payload.get("splat_path"):
        artifacts.append(
            build_model_artifact(
                role="quick_splat",
                path=payload["splat_path"],
                format=".splat",
                delivery_tier="browser_quick",
                gaussian_count=gaussian_count,
            )
        )
    if payload.get("object_ply"):
        artifacts.append(
            build_model_artifact(
                role="object_edit",
                path=payload["object_ply"],
                format=".ply",
                delivery_tier="browser_edit",
                gaussian_count=_optional_positive_int(
                    payload.get("object_ply_gaussian_count"),
                    "object_ply_gaussian_count",
                )
                or gaussian_count,
                object_count=object_count,
            )
        )
    if payload.get("gaussian_ply"):
        artifacts.append(
            build_model_artifact(
                role="source_gaussian",
                path=payload["gaussian_ply"],
                format=".ply",
                delivery_tier="training_internal",
                browser_ready=False,
                gaussian_count=gaussian_count,
            )
        )
    if payload.get("object_field_path"):
        artifacts.append(
            build_model_artifact(
                role="object_field",
                path=payload["object_field_path"],
                format=".npz",
                delivery_tier="training_internal",
                browser_ready=False,
                gaussian_count=payload.get("object_field_gaussian_count") or gaussian_count,
                object_count=payload.get("object_field_slot_count") or object_count,
            )
        )
    quality_evidence = []
    if payload.get("training"):
        quality_evidence.append(
            {
                "kind": "sample_bundle_training_summary",
                "source": str(path),
                "summary": payload["training"],
            }
        )
    return build_model_artifact_manifest(
        manifest_id=manifest_id or f"{asset_id}-sample-model-artifacts",
        asset_id=asset_id,
        name=name or str(payload.get("sample_id") or asset_id),
        source=source
        or {
            "type": "sample_bundle",
            "dataset": payload.get("dataset"),
            "mask_manifest": payload.get("mask_manifest"),
        },
        license=license,
        artifacts=artifacts,
        gaussian_count=gaussian_count,
        object_count=object_count,
        quality_evidence=quality_evidence,
        limitations=[
            "Derived from a sample bundle; source assets remain governed by their original license.",
        ],
        created_from={
            "sample_bundle": str(path),
            "training_manifest": payload.get("training_manifest"),
            "mask_manifest": payload.get("mask_manifest"),
        },
    )


def _trainable_gaussian_count(payload: dict[str, Any]) -> int | None:
    sample = payload.get("source", {}).get("sample")
    if isinstance(sample, dict):
        sample_count = _optional_positive_int(sample.get("sampled_count"), "sampled_count")
        if sample_count is not None:
            return sample_count
    assignments = payload.get("assignments")
    if isinstance(assignments, list) and assignments:
        shape = assignments[0].get("shape") if isinstance(assignments[0], dict) else None
        if isinstance(shape, list) and shape:
            return _optional_positive_int(shape[0], "assignments[0].shape[0]")
    return None


def _trainable_asset_id(payload: dict[str, Any], artifact_file: Path | None) -> str:
    source_input = payload.get("source", {}).get("input")
    if source_input:
        return f"{Path(str(source_input)).stem}-trainable-kernel"
    if artifact_file is not None:
        return f"{artifact_file.stem}-trainable-kernel"
    return "trainable-kernel-model"


def _quality_report_artifact(
    *,
    quality_report_path: str | Path,
    quality_report_file: str | Path | None,
    compute_hash: bool,
) -> dict[str, Any]:
    report_file = Path(quality_report_file) if quality_report_file is not None else Path(quality_report_path)
    byte_size = report_file.stat().st_size if report_file.exists() else None
    sha256 = _sha256(report_file) if compute_hash and report_file.exists() else None
    return build_model_artifact(
        role="quality_report",
        path=quality_report_path,
        format=".json",
        delivery_tier="browser_edit",
        browser_ready=True,
        byte_size=byte_size,
        sha256=sha256,
        label="ObjectState quality report",
        note="Debug OS metrics evidence derived from the trainable kernel artifact.",
    )


def manifest_from_asset_library_entry(
    entry: dict[str, Any],
    *,
    manifest_id: str | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    asset_id = _required_string(entry.get("id"), "id")
    name = _required_string(entry.get("name"), "name")
    license = _required_string(entry.get("license"), "license")
    gaussian_count = _optional_positive_int(
        entry.get("objectPlyGaussianCount", entry.get("gaussianCount")),
        "gaussianCount",
    )
    artifacts: list[dict[str, Any]] = []
    if entry.get("splatPath"):
        artifacts.append(
            build_model_artifact(
                role="quick_splat",
                path=entry["splatPath"],
                format=".splat",
                delivery_tier="browser_quick",
                gaussian_count=entry.get("gaussianCount"),
                byte_size=_asset_size_bytes(entry.get("splatSizeLabel")),
                label=entry.get("splatFileName"),
            )
        )
    local_path = entry.get("localPath")
    if local_path:
        role, delivery_tier, browser_ready = _asset_object_artifact_route(entry)
        artifacts.append(
            build_model_artifact(
                role=role,
                path=local_path,
                format=_format_from_path(local_path, fallback=".ply"),
                delivery_tier=delivery_tier,
                browser_ready=browser_ready,
                gaussian_count=gaussian_count,
                byte_size=_asset_size_bytes(entry.get("objectPlySizeLabel")),
                label=entry.get("fileName"),
                note=(
                    "Deferred or large object PLY from asset library; not a default browser artifact."
                    if role == "diagnostic_full"
                    else None
                ),
            )
        )
    if not artifacts:
        raise ValueError(f"asset library entry {asset_id!r} does not describe a Gaussian viewer artifact")
    limitations = []
    if entry.get("deferObjectPly"):
        limitations.append("Object PLY is deferred and must not be requested by quick-view routes.")
    if entry.get("objectPlySizeLabel"):
        limitations.append(f"Object PLY size label: {entry['objectPlySizeLabel']}")
    return build_model_artifact_manifest(
        manifest_id=manifest_id or f"{asset_id}-asset-model-artifacts",
        asset_id=asset_id,
        name=name,
        source=source
        or {
            "type": "asset_library_entry",
            "source_type": entry.get("sourceType"),
            "source_name": entry.get("sourceName"),
            "source_url": entry.get("sourceUrl"),
            "pipeline_stage": entry.get("pipelineStage"),
        },
        license=license,
        artifacts=artifacts,
        gaussian_count=gaussian_count,
        object_count=_optional_positive_int(entry.get("objectCount"), "objectCount"),
        quality_evidence=[],
        limitations=limitations,
        created_from={
            "asset_library_entry": asset_id,
            "category": entry.get("category"),
            "status": entry.get("status"),
            "use_cases": entry.get("useCases", []),
        },
    )


def write_model_artifact_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    validation = validate_model_artifact_manifest(payload, require_browser_ready=True)
    if not validation.passed:
        raise ValueError("; ".join(validation.errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_model_artifact_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validation = validate_model_artifact_manifest(payload)
    if not validation.passed:
        raise ValueError("; ".join(validation.errors))
    return payload


def validate_model_artifact_manifest(
    payload: dict[str, Any],
    *,
    require_browser_ready: bool = False,
    require_object_edit: bool = False,
) -> ModelManifestValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    schema = payload.get("schema")
    if schema != MODEL_ARTIFACT_MANIFEST_SCHEMA:
        errors.append(f"schema must be {MODEL_ARTIFACT_MANIFEST_SCHEMA!r}")
    manifest_id = _optional_string(payload.get("manifest_id"))
    asset_id = _optional_string(payload.get("asset_id"))
    if manifest_id is None:
        errors.append("manifest_id is required")
    if asset_id is None:
        errors.append("asset_id is required")
    if _optional_string(payload.get("name")) is None:
        errors.append("name is required")
    if _optional_string(payload.get("license")) is None:
        errors.append("license is required")
    if not isinstance(payload.get("source"), dict) or not payload.get("source"):
        errors.append("source must be a non-empty object")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("artifacts must be a non-empty list")
        artifacts = []

    roles: set[str] = set()
    browser_ready_count = 0
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            errors.append(f"artifacts[{index}] must be an object")
            continue
        _validate_artifact(artifact, index=index, errors=errors, warnings=warnings)
        role = artifact.get("role")
        if isinstance(role, str):
            roles.add(role)
        if artifact.get("browser_ready") is True:
            browser_ready_count += 1

    if require_browser_ready and browser_ready_count == 0:
        errors.append("at least one browser_ready artifact is required")
    if require_object_edit and "object_edit" not in roles:
        errors.append("object_edit artifact is required")
    for role in ("diagnostic_full", "source_gaussian"):
        for artifact in artifacts:
            if isinstance(artifact, dict) and artifact.get("role") == role and artifact.get("browser_ready") is True:
                errors.append(f"{role} artifact must not be browser_ready")
    return ModelManifestValidationResult(
        manifest_id=manifest_id,
        asset_id=asset_id,
        passed=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        artifact_count=len(artifacts),
        browser_ready_artifacts=browser_ready_count,
    )


def _validate_artifact(
    artifact: dict[str, Any],
    *,
    index: int,
    errors: list[str],
    warnings: list[str],
) -> None:
    role = artifact.get("role")
    if not isinstance(role, str) or role not in ARTIFACT_ROLES:
        errors.append(f"artifacts[{index}].role is invalid")
    path = artifact.get("path")
    if not isinstance(path, str) or not path:
        errors.append(f"artifacts[{index}].path is required")
    fmt = artifact.get("format")
    if not isinstance(fmt, str) or not fmt:
        errors.append(f"artifacts[{index}].format is required")
    delivery_tier = artifact.get("delivery_tier")
    if not isinstance(delivery_tier, str) or delivery_tier not in DELIVERY_TIERS:
        errors.append(f"artifacts[{index}].delivery_tier is invalid")
    browser_ready = artifact.get("browser_ready")
    if not isinstance(browser_ready, bool):
        errors.append(f"artifacts[{index}].browser_ready must be boolean")
    if delivery_tier in BROWSER_READY_TIERS and browser_ready is not True:
        errors.append(f"artifacts[{index}] delivery_tier {delivery_tier!r} requires browser_ready=true")
    if delivery_tier not in BROWSER_READY_TIERS and browser_ready is True:
        warnings.append(f"artifacts[{index}] marks non-browser delivery tier as browser_ready")
    if role in BROWSER_ARTIFACT_ROLES and delivery_tier not in BROWSER_READY_TIERS:
        warnings.append(f"artifacts[{index}] role {role!r} is not marked as a browser delivery tier")
    for key in ("gaussian_count", "object_count", "byte_size"):
        value = artifact.get(key)
        if value is not None and (not isinstance(value, int) or value < 0):
            errors.append(f"artifacts[{index}].{key} must be a non-negative integer")
    sha256 = artifact.get("sha256")
    if sha256 is not None and (not isinstance(sha256, str) or len(sha256) != 64):
        errors.append(f"artifacts[{index}].sha256 must be a 64-character hex string")
    if sha256 is not None and isinstance(sha256, str) and len(sha256) == 64:
        try:
            int(sha256, 16)
        except ValueError:
            errors.append(f"artifacts[{index}].sha256 must be lowercase or uppercase hex")
    if role == "compressed_chunked":
        _validate_compressed_chunked_artifact(artifact, index=index, errors=errors, warnings=warnings)


def _validate_compressed_chunked_artifact(
    artifact: dict[str, Any],
    *,
    index: int,
    errors: list[str],
    warnings: list[str],
) -> None:
    browser_ready = artifact.get("browser_ready") is True
    if not browser_ready:
        warnings.append(f"artifacts[{index}] compressed_chunked artifact is not browser_ready")
        return

    if artifact.get("delivery_tier") not in BROWSER_READY_TIERS:
        errors.append(f"artifacts[{index}] browser_ready compressed_chunked artifact must use a browser tier")
    for key in ("gaussian_count", "object_count", "byte_size"):
        if not _is_positive_int(artifact.get(key)):
            errors.append(f"artifacts[{index}].{key} is required for browser_ready compressed_chunked artifacts")
    if _optional_string(artifact.get("sha256")) is None:
        errors.append(f"artifacts[{index}].sha256 is required for browser_ready compressed_chunked artifacts")

    chunk_index = artifact.get("chunk_index")
    if not isinstance(chunk_index, dict):
        errors.append(f"artifacts[{index}].chunk_index is required for browser_ready compressed_chunked artifacts")
    else:
        _validate_chunk_index(chunk_index, index=index, errors=errors)

    compression = artifact.get("compression")
    if not isinstance(compression, dict):
        errors.append(f"artifacts[{index}].compression is required for browser_ready compressed_chunked artifacts")
    else:
        if _optional_string(compression.get("codec")) is None:
            errors.append(f"artifacts[{index}].compression.codec is required")
        if _optional_string(compression.get("version")) is None:
            errors.append(f"artifacts[{index}].compression.version is required")
        if _optional_string(compression.get("layout")) is None:
            errors.append(f"artifacts[{index}].compression.layout is required")

    lod = artifact.get("lod")
    if not isinstance(lod, dict):
        errors.append(f"artifacts[{index}].lod is required for browser_ready compressed_chunked artifacts")
    else:
        _validate_lod(lod, artifact=artifact, index=index, errors=errors)

    coverage = artifact.get("object_id_coverage")
    if not isinstance(coverage, dict):
        errors.append(
            f"artifacts[{index}].object_id_coverage is required for browser_ready compressed_chunked artifacts"
        )
    else:
        _validate_object_id_coverage(coverage, artifact=artifact, index=index, errors=errors)


def _validate_chunk_index(chunk_index: dict[str, Any], *, index: int, errors: list[str]) -> None:
    if chunk_index.get("schema") != CHUNK_INDEX_SCHEMA:
        errors.append(f"artifacts[{index}].chunk_index.schema must be {CHUNK_INDEX_SCHEMA!r}")
    if _optional_string(chunk_index.get("path")) is None:
        errors.append(f"artifacts[{index}].chunk_index.path is required")
    if not _is_positive_int(chunk_index.get("chunk_count")):
        errors.append(f"artifacts[{index}].chunk_index.chunk_count must be a positive integer")
    if _optional_string(chunk_index.get("sort_key")) is None:
        errors.append(f"artifacts[{index}].chunk_index.sort_key is required")


def _validate_lod(
    lod: dict[str, Any],
    *,
    artifact: dict[str, Any],
    index: int,
    errors: list[str],
) -> None:
    levels = lod.get("levels")
    if not isinstance(levels, list) or not levels:
        errors.append(f"artifacts[{index}].lod.levels must be a non-empty list")
        return
    max_gaussians = artifact.get("gaussian_count")
    for level_index, level in enumerate(levels):
        if not isinstance(level, dict):
            errors.append(f"artifacts[{index}].lod.levels[{level_index}] must be an object")
            continue
        level_id = level.get("level")
        if not isinstance(level_id, int) or level_id < 0:
            errors.append(f"artifacts[{index}].lod.levels[{level_index}].level must be a non-negative integer")
        gaussian_count = level.get("gaussian_count")
        if not _is_positive_int(gaussian_count):
            errors.append(f"artifacts[{index}].lod.levels[{level_index}].gaussian_count must be positive")
        elif isinstance(max_gaussians, int) and gaussian_count > max_gaussians:
            errors.append(f"artifacts[{index}].lod.levels[{level_index}].gaussian_count exceeds artifact count")


def _validate_object_id_coverage(
    coverage: dict[str, Any],
    *,
    artifact: dict[str, Any],
    index: int,
    errors: list[str],
) -> None:
    if coverage.get("has_object_ids") is not True:
        errors.append(f"artifacts[{index}].object_id_coverage.has_object_ids must be true")
    if _optional_string(coverage.get("field")) is None:
        errors.append(f"artifacts[{index}].object_id_coverage.field is required")
    if _optional_string(coverage.get("mode")) is None:
        errors.append(f"artifacts[{index}].object_id_coverage.mode is required")
    object_count = coverage.get("object_count")
    if not _is_positive_int(object_count):
        errors.append(f"artifacts[{index}].object_id_coverage.object_count must be a positive integer")
    artifact_object_count = artifact.get("object_count")
    if isinstance(artifact_object_count, int) and isinstance(object_count, int) and artifact_object_count != object_count:
        errors.append(f"artifacts[{index}].object_id_coverage.object_count must match artifact object_count")


def _resolve_chunk_index(
    chunk_index: dict[str, Any] | str | Path,
    *,
    chunk_index_path: str | Path | None,
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(chunk_index, (str, Path)):
        path = Path(chunk_index)
        return read_chunk_index(path), path
    if isinstance(chunk_index, dict):
        raw_path = chunk_index_path or chunk_index.get("path")
        resolved_path = Path(raw_path) if isinstance(raw_path, (str, Path)) else None
        return dict(chunk_index), resolved_path
    raise TypeError("chunk_index must be a dict or path")


def _compression_metadata(index: dict[str, Any]) -> dict[str, Any]:
    compression = index.get("compression")
    if isinstance(compression, dict):
        return dict(compression)
    return {
        "codec": "objgauss-ogc-prototype",
        "version": "0.1",
        "layout": "object-aware-chunked-uncompressed",
    }


def _lod_metadata(index: dict[str, Any]) -> dict[str, Any]:
    lod = index.get("lod")
    if isinstance(lod, dict):
        return dict(lod)
    return {
        "selection": "object-aware-index-placeholder",
        "levels": [
            {
                "level": 0,
                "ratio": 1.0,
                "gaussian_count": _positive_int_or_none(index.get("gaussian_count")) or 1,
            }
        ],
    }


def _validate_role(role: str) -> None:
    if role not in ARTIFACT_ROLES:
        raise ValueError(f"unknown artifact role: {role}")


def _validate_delivery_tier(delivery_tier: str) -> None:
    if delivery_tier not in DELIVERY_TIERS:
        raise ValueError(f"unknown delivery tier: {delivery_tier}")


def _optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name} must be >= 0")
    return resolved


def _positive_int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is required")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_object_artifact_route(entry: dict[str, Any]) -> tuple[str, str, bool]:
    size_label = str(entry.get("objectPlySizeLabel") or "")
    deferred = bool(entry.get("deferObjectPly"))
    large = "GB" in size_label.upper() or "1.15GB" in size_label
    if deferred or large:
        return "diagnostic_full", "diagnostic", False
    return "object_edit", "browser_edit", True


def _asset_size_bytes(label: object) -> int | None:
    if not isinstance(label, str) or not label:
        return None
    compact = label.strip().upper().replace(" ", "")
    multipliers = (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024))
    for suffix, multiplier in multipliers:
        if suffix in compact:
            raw = compact.split(suffix, 1)[0]
            try:
                return int(float(raw) * multiplier)
            except ValueError:
                return None
    return None


def _format_from_path(path: object, *, fallback: str) -> str:
    if not isinstance(path, str):
        return fallback
    suffix = Path(path).suffix
    return suffix or fallback
