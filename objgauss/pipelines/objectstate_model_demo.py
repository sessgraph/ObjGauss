from __future__ import annotations

import hashlib
import json
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.io import append_or_replace_property, write_ply
from objgauss.model_manifest import (
    MODEL_ARTIFACT_MANIFEST_SCHEMA,
    build_model_artifact,
    build_model_artifact_manifest,
    validate_model_artifact_manifest,
    write_model_artifact_manifest,
)
from objgauss.pipelines.objectstate_model_v0 import (
    OBJECTSTATE_MODEL_V0_STATE_SCHEMA,
    ObjectStateModelV0TrainingResult,
    objectstate_model_v0_state_from_dict,
    train_objectstate_model_v0,
    validate_objectstate_model_v0_training_summary,
)
from objgauss.pipelines.trainable_quality import (
    TRAINABLE_QUALITY_REPORT_SCHEMA,
    validate_trainable_quality_report,
)

__all__ = (
    "ObjectStateModelDemoResult",
    "write_objectstate_model_demo",
)


@dataclass(frozen=True)
class ObjectStateModelDemoResult:
    output_dir: Path
    checkpoint_path: Path
    before_ply_path: Path
    after_ply_path: Path
    metrics_path: Path
    quality_report_path: Path
    manifest_path: Path
    summary: dict[str, Any]
    quality_report: dict[str, Any]
    manifest: dict[str, Any]
    viewer_manifest_path: Path | None = None
    viewer_url: str | None = None


def write_objectstate_model_demo(
    cloud: GaussianCloud,
    *,
    input_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    license: str,
    target_source: str,
    object_id_field: str = "object_id",
    source_kind: str = "public_replay",
    split: str = "train",
    source_url: str | None = None,
    source_splat: str | Path | None = None,
    viewer_dir: str | Path | None = None,
    frame_field: str = "source_frame",
    hidden_dim: int = 24,
    heldout_stride: int = 4,
    iterations: int = 240,
    learning_rate: float = 8e-2,
    assignment_weight: float = 1.0,
    compactness_weight: float = 0.02,
    semantic_weight: float = 0.02,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> ObjectStateModelDemoResult:
    """Write one held-out-frame ObjectState Model v0 demo bundle."""

    if not run_id:
        raise ValueError("run_id is required")
    if not license:
        raise ValueError("license is required")
    if not target_source:
        raise ValueError("target_source is required")

    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"model input does not exist: {input_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_assignment, source_object_ids = _target_assignment_from_cloud(
        cloud,
        object_id_field=object_id_field,
    )
    training = train_objectstate_model_v0(
        cloud,
        object_id_field=object_id_field,
        frame_field=frame_field,
        hidden_dim=hidden_dim,
        heldout_stride=heldout_stride,
        iterations=iterations,
        learning_rate=learning_rate,
        assignment_weight=assignment_weight,
        compactness_weight=compactness_weight,
        semantic_weight=semantic_weight,
        weight_decay=weight_decay,
        seed=seed,
    )
    checkpoint_path = output_dir / "checkpoint.json"
    _write_json(checkpoint_path, training.final_state.as_dict(include_arrays=True))
    before_ply_path = output_dir / "before.ply"
    after_ply_path = output_dir / "after-object-color.ply"
    _write_assignment_visualization(
        before_ply_path,
        cloud,
        np.argmax(training.initial_assignment, axis=1),
        comments=("ObjGauss ObjectState Model v0 before training",),
    )
    _write_assignment_visualization(
        after_ply_path,
        cloud,
        np.argmax(training.final_assignment, axis=1),
        comments=("ObjGauss ObjectState Model v0 after training",),
    )
    metrics_path = output_dir / "metrics.json"
    summary = training.as_dict()
    summary["dataset"] = {
        "sample_id": run_id,
        "source_kind": source_kind,
        "split": split,
        "license": license,
        "gaussian_ref": str(input_path),
        "gaussian_count": cloud.count,
        "target_source": target_source,
    }
    summary["checkpoint"] = {
        "path": str(checkpoint_path),
        "schema": OBJECTSTATE_MODEL_V0_STATE_SCHEMA,
        "roundtrip_ok": _checkpoint_roundtrip_ok(training),
    }
    summary["visualization_refs"] = {
        "before_ply": str(before_ply_path),
        "after_ply": str(after_ply_path),
        "coloring": "argmax(A[N,K])",
    }
    summary["summary_path"] = str(metrics_path)
    summary["demo"] = {
        "phase": "Phase M / Phase 2.5",
        "run_id": run_id,
        "input": str(input_path),
        "source_splat": None if source_splat is None else str(source_splat),
        "source_url": source_url,
        "object_id_field": object_id_field,
        "source_object_ids": source_object_ids,
        "target_source": target_source,
        "claim": "same_scene_heldout_frame_object_assignment",
    }
    checked_summary = validate_objectstate_model_v0_training_summary(summary)
    _write_json(metrics_path, checked_summary)

    quality_report = _model_v0_quality_report(checked_summary, run_id=run_id)
    quality_report_path = output_dir / "quality-report.json"
    _write_json(quality_report_path, quality_report)

    artifacts = [
        _manifest_artifact(
            actual_path=after_ply_path,
            manifest_path="after-object-color.ply",
            role="object_edit",
            format=".ply",
            delivery_tier="browser_edit",
            gaussian_count=cloud.count,
            object_count=target_assignment.shape[1],
            label="ObjectState assignment after training",
        ),
        _manifest_artifact(
            actual_path=metrics_path,
            manifest_path="metrics.json",
            role="training_summary",
            format=".json",
            delivery_tier="browser_edit",
            label="ObjectState assignment run metrics",
        ),
        _manifest_artifact(
            actual_path=quality_report_path,
            manifest_path="quality-report.json",
            role="quality_report",
            format=".json",
            delivery_tier="browser_edit",
            label="ObjectState assignment quality report",
        ),
        _manifest_artifact(
            actual_path=input_path,
            manifest_path=_relative_manifest_path(input_path, output_dir),
            role="model_input",
            format=input_path.suffix or ".ply",
            delivery_tier="browser_edit",
            gaussian_count=cloud.count,
            label="ObjectState Model v0 browser inference input",
        ),
        _manifest_artifact(
            actual_path=checkpoint_path,
            manifest_path="checkpoint.json",
            role="objectstate_model",
            format=".json",
            delivery_tier="browser_edit",
            gaussian_count=cloud.count,
            object_count=target_assignment.shape[1],
            label="ObjectState Model v0 checkpoint",
        ),
        _manifest_artifact(
            actual_path=before_ply_path,
            manifest_path="before.ply",
            role="diagnostic_full",
            format=".ply",
            delivery_tier="diagnostic",
            browser_ready=False,
            gaussian_count=cloud.count,
            object_count=target_assignment.shape[1],
            label="Assignment before training",
        ),
    ]
    if source_splat is not None:
        source_splat_path = Path(source_splat)
        if not source_splat_path.is_file():
            raise FileNotFoundError(f"source_splat does not exist: {source_splat_path}")
        artifacts.insert(
            0,
            _manifest_artifact(
                actual_path=source_splat_path,
                manifest_path=_relative_manifest_path(source_splat_path, output_dir),
                role="quick_splat",
                format=".splat",
                delivery_tier="browser_quick",
                gaussian_count=cloud.count,
                label="Raw Gaussian source",
            ),
        )

    manifest = build_model_artifact_manifest(
        manifest_id=f"{run_id}-model-artifacts",
        asset_id=run_id,
        name=f"ObjectState model demo: {run_id}",
        stage="objectstate-model-v0-demo",
        source={
            "type": "objectstate_model_v0_training",
            "run_id": run_id,
            "input": str(input_path),
            "source_url": source_url,
            "target_source": target_source,
            "split": split,
        },
        license=license,
        gaussian_count=cloud.count,
        object_count=target_assignment.shape[1],
        artifacts=artifacts,
        quality_evidence=[
            {
                "kind": "objectstate_model_v0_training",
                "status": checked_summary["status"],
                "run_id": run_id,
                "metrics_path": "metrics.json",
                "checkpoint_path": "checkpoint.json",
                "checkpoint_sha256": _sha256(checkpoint_path),
                "target_source": target_source,
            }
        ],
        limitations=[
            "Same-scene held-out-frame assignment demo; target object_id is teacher evidence.",
            "Held-out metrics do not establish cross-scene generalization.",
            "Does not validate temporal identity, prediction, intervention, or the Reality Gate.",
            "Local research output; verify source license before any public release.",
        ],
        created_from={
            "run_schema": checked_summary["schema"],
            "model_state_schema": OBJECTSTATE_MODEL_V0_STATE_SCHEMA,
            "checkpoint": "checkpoint.json",
        },
    )
    validation = validate_model_artifact_manifest(
        manifest,
        require_browser_ready=True,
        require_object_edit=True,
    )
    if not validation.passed:
        raise ValueError("; ".join(validation.errors))
    manifest_path = output_dir / "model-manifest.json"
    write_model_artifact_manifest(manifest_path, manifest)
    viewer_manifest_path = None
    viewer_url = None
    if viewer_dir is not None:
        viewer_manifest_path = _stage_viewer_package(
            viewer_dir=Path(viewer_dir),
            output_dir=output_dir,
            manifest=manifest,
            input_path=input_path,
            source_splat=None if source_splat is None else Path(source_splat),
        )
        viewer_url = _viewer_url(viewer_manifest_path)
    return ObjectStateModelDemoResult(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        before_ply_path=before_ply_path,
        after_ply_path=after_ply_path,
        metrics_path=metrics_path,
        quality_report_path=quality_report_path,
        manifest_path=manifest_path,
        summary=checked_summary,
        quality_report=quality_report,
        manifest=manifest,
        viewer_manifest_path=viewer_manifest_path,
        viewer_url=viewer_url,
    )


def _target_assignment_from_cloud(
    cloud: GaussianCloud,
    *,
    object_id_field: str,
) -> tuple[np.ndarray, list[int]]:
    if object_id_field not in cloud.fields:
        raise ValueError(f"Gaussian cloud is missing supervision field {object_id_field!r}")
    values = np.asarray(cloud.vertices[object_id_field])
    if values.ndim != 1 or values.shape[0] != cloud.count:
        raise ValueError("object supervision must be one scalar per Gaussian")
    numeric = values.astype(np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("object supervision must be finite")
    rounded = np.rint(numeric)
    if not np.allclose(numeric, rounded):
        raise ValueError("object supervision must contain integer ids")
    labels = rounded.astype(np.int64)
    if np.any(labels < 0):
        raise ValueError("object supervision must contain non-negative ids")
    source_ids = sorted(int(value) for value in np.unique(labels))
    if len(source_ids) < 2:
        raise ValueError("ObjectState model demo requires at least two supervised objects")
    remap = {source_id: index for index, source_id in enumerate(source_ids)}
    contiguous = np.asarray([remap[int(value)] for value in labels], dtype=np.int64)
    target = np.zeros((cloud.count, len(source_ids)), dtype=np.float32)
    target[np.arange(cloud.count), contiguous] = 1.0
    return target, source_ids


def _model_v0_quality_report(summary: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    before = summary["heldout_before_metrics"]
    after = summary["heldout_after_metrics"]
    assignment = summary["assignment_diagnostics"]["heldout_after"]
    coverage_matches = (
        summary["split"]["train_object_ids"]
        == summary["split"]["heldout_object_ids"]
        and len(summary["split"]["heldout_object_ids"]) == summary["config"]["slots"]
    )
    gates = [
        {
            "name": "loss_decreased",
            "status": "pass" if summary["loss"]["total_decreased"] else "warn",
            "value": 1.0 if summary["loss"]["total_decreased"] else 0.0,
            "threshold": 1.0,
        },
        {
            "name": "heldout_ari",
            "status": "pass" if after["ari"] >= 0.5 else "warn",
            "value": float(after["ari"]),
            "threshold": 0.5,
        },
        {
            "name": "checkpoint_roundtrip",
            "status": "pass" if summary["checkpoint"]["roundtrip_ok"] else "warn",
            "value": 1.0 if summary["checkpoint"]["roundtrip_ok"] else 0.0,
            "threshold": 1.0,
        },
        {
            "name": "train_heldout_object_coverage",
            "status": "pass" if coverage_matches else "warn",
            "value": 1.0 if coverage_matches else 0.0,
            "threshold": 1.0,
        },
    ]
    report = {
        "schema": TRAINABLE_QUALITY_REPORT_SCHEMA,
        "report_id": f"{run_id}-model-v0-quality",
        "status": "pass" if all(gate["status"] == "pass" for gate in gates) else "warn",
        "source": {
            "type": "objectstate_model_v0_training",
            "run_id": run_id,
            "metrics_path": "metrics.json",
        },
        "metrics": {
            "ari": float(after["ari"]),
            "mean_best_iou": float(after["mean_best_iou"]),
            "object_purity": float(after["purity"]),
            "before_ari": float(before["ari"]),
            "before_mean_best_iou": float(before["mean_best_iou"]),
            "before_object_purity": float(before["purity"]),
            "train_ari": float(summary["train_after_metrics"]["ari"]),
            "train_mean_best_iou": float(
                summary["train_after_metrics"]["mean_best_iou"]
            ),
            "train_object_purity": float(summary["train_after_metrics"]["purity"]),
            "assignment_confidence": float(assignment["mean_confidence"]),
            "assignment_entropy": float(assignment["mean_normalized_entropy"]),
            "effective_slots": float(assignment["effective_slots"]),
            "initial_loss": float(summary["loss"]["initial"]["total_loss"]),
            "final_loss": float(summary["loss"]["final"]["total_loss"]),
        },
        "gates": gates,
        "limitations": [
            "Metrics use held-out frames from the same scene, not cross-scene data.",
            "This report is model-demo evidence, not real-world identity or causal validation.",
        ],
    }
    validate_trainable_quality_report(report)
    return report


def _manifest_artifact(
    *,
    actual_path: Path,
    manifest_path: str,
    role: str,
    format: str,
    delivery_tier: str,
    browser_ready: bool | None = None,
    gaussian_count: int | None = None,
    object_count: int | None = None,
    label: str,
) -> dict[str, Any]:
    artifact = build_model_artifact(
        role=role,
        path=actual_path,
        format=format,
        delivery_tier=delivery_tier,
        browser_ready=browser_ready,
        gaussian_count=gaussian_count,
        object_count=object_count,
        label=label,
        compute_hash=True,
    )
    artifact["path"] = manifest_path
    return artifact


def _relative_manifest_path(path: Path, output_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), output_dir.resolve())).as_posix()


def _stage_viewer_package(
    *,
    viewer_dir: Path,
    output_dir: Path,
    manifest: dict[str, Any],
    input_path: Path,
    source_splat: Path | None,
) -> Path:
    if viewer_dir.resolve() == output_dir.resolve():
        raise ValueError("viewer_dir must differ from output_dir")
    viewer_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_staged_package(viewer_dir)
    role_files = {
        "object_edit": (output_dir / "after-object-color.ply", "after-object-color.ply"),
        "training_summary": (output_dir / "metrics.json", "metrics.json"),
        "quality_report": (output_dir / "quality-report.json", "quality-report.json"),
        "model_input": (input_path, "model-input.ply"),
        "objectstate_model": (output_dir / "checkpoint.json", "checkpoint.json"),
        "diagnostic_full": (output_dir / "before.ply", "before.ply"),
    }
    if source_splat is not None:
        role_files["quick_splat"] = (source_splat, "raw-source.splat")

    staged = deepcopy(manifest)
    for artifact in staged["artifacts"]:
        role = artifact["role"]
        if role not in role_files:
            raise ValueError(f"viewer package has no staging policy for artifact role {role!r}")
        source, file_name = role_files[role]
        if not source.is_file():
            raise FileNotFoundError(f"viewer package source is missing: {source}")
        target = viewer_dir / file_name
        shutil.copy2(source, target)
        artifact["path"] = file_name
        if artifact.get("sha256") and artifact["sha256"] != _sha256(target):
            raise ValueError(f"viewer package hash mismatch for {role}")

    checkpoint_target = viewer_dir / "checkpoint.json"
    expected_checkpoint_hash = staged["quality_evidence"][0]["checkpoint_sha256"]
    if _sha256(checkpoint_target) != expected_checkpoint_hash:
        raise ValueError("viewer package checkpoint hash mismatch")
    staged["quality_evidence"][0]["checkpoint_path"] = "checkpoint.json"
    staged["created_from"]["checkpoint"] = "checkpoint.json"
    _validate_staged_run_binding(staged, viewer_dir)
    validation = validate_model_artifact_manifest(
        staged,
        require_browser_ready=True,
        require_object_edit=True,
    )
    if not validation.passed:
        raise ValueError("; ".join(validation.errors))
    manifest_path = viewer_dir / "model-manifest.json"
    write_model_artifact_manifest(manifest_path, staged)
    expected_files = {
        "model-manifest.json",
        *(file_name for _source, file_name in role_files.values()),
    }
    actual_files = {path.name for path in viewer_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError(
            "viewer package contains unexpected files: "
            f"{sorted(actual_files - expected_files)}"
        )
    return manifest_path


def _clear_previous_staged_package(viewer_dir: Path) -> None:
    manifest_path = viewer_dir / "model-manifest.json"
    if not manifest_path.is_file():
        return
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("viewer_dir contains an unreadable prior model manifest") from error
    if previous.get("schema") != MODEL_ARTIFACT_MANIFEST_SCHEMA:
        raise ValueError("viewer_dir prior manifest is not managed by ObjGauss")
    managed_names = {"model-manifest.json", "supervision-source.ply"}
    for artifact in previous.get("artifacts", []):
        if isinstance(artifact, dict):
            managed_names.add(Path(str(artifact.get("path", ""))).name)
    for evidence in previous.get("quality_evidence", []):
        if isinstance(evidence, dict):
            managed_names.add(Path(str(evidence.get("checkpoint_path", ""))).name)
    created_checkpoint = previous.get("created_from", {}).get("checkpoint")
    if created_checkpoint:
        managed_names.add(Path(str(created_checkpoint)).name)
    for name in managed_names - {""}:
        candidate = viewer_dir / name
        if candidate.is_file():
            candidate.unlink()


def _validate_staged_run_binding(manifest: dict[str, Any], viewer_dir: Path) -> None:
    metrics = json.loads((viewer_dir / "metrics.json").read_text(encoding="utf-8"))
    quality = json.loads((viewer_dir / "quality-report.json").read_text(encoding="utf-8"))
    run_id = metrics.get("demo", {}).get("run_id")
    if not run_id or run_id != manifest.get("source", {}).get("run_id"):
        raise ValueError("viewer package manifest and metrics run_id mismatch")
    if run_id != quality.get("source", {}).get("run_id"):
        raise ValueError("viewer package quality report and metrics run_id mismatch")


def _viewer_url(manifest_path: Path) -> str | None:
    public_root = (Path.cwd() / "public").resolve()
    try:
        route = manifest_path.resolve().relative_to(public_root)
    except ValueError:
        return None
    return f"/?modelArtifactManifest=/{route.as_posix()}"


def _checkpoint_roundtrip_ok(training: ObjectStateModelV0TrainingResult) -> bool:
    restored = training.final_state.as_dict(include_arrays=True)
    restored_state = objectstate_model_v0_state_from_dict(restored)
    expected = training.final_state.predict(training.cloud)
    actual = restored_state.predict(training.cloud)
    return bool(
        np.allclose(expected, actual, atol=1e-7, rtol=1e-7)
        and np.array_equal(np.argmax(expected, axis=1), np.argmax(actual, axis=1))
    )


def _write_assignment_visualization(
    path: Path,
    cloud: GaussianCloud,
    labels: np.ndarray,
    *,
    comments: tuple[str, ...],
) -> None:
    palette = np.asarray(
        [
            [230, 57, 70],
            [42, 157, 143],
            [69, 123, 157],
            [244, 162, 97],
            [131, 56, 236],
            [38, 70, 83],
            [255, 183, 3],
            [0, 109, 119],
        ],
        dtype=np.uint8,
    )
    normalized = np.asarray(labels, dtype=np.int32)
    colors = palette[normalized % palette.shape[0]]
    vertices = append_or_replace_property(cloud.vertices, "red", colors[:, 0], "u1")
    vertices = append_or_replace_property(vertices, "green", colors[:, 1], "u1")
    vertices = append_or_replace_property(vertices, "blue", colors[:, 2], "u1")
    vertices = append_or_replace_property(vertices, "object_id", normalized, "i4")
    vertices = append_or_replace_property(
        vertices,
        "predicted_object_id",
        normalized,
        "i4",
    )
    write_ply(
        path,
        GaussianCloud(vertices=vertices, comments=comments, source_format="ascii"),
        fmt="ascii",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
