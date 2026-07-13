from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.objects import apply_object_colors, assign_object_ids
from objgauss.datasets.objectstate_multi_object_synthetic import (
    MultiObjectSyntheticDataset,
    build_multi_object_synthetic_dataset,
)
from objgauss.evaluation.objectstate_instance_segmentation import (
    evaluate_multi_object_instance_benchmark,
)
from objgauss.model_manifest import (
    build_model_artifact,
    build_model_artifact_manifest,
)
from objgauss.pipelines.objectstate_model_v0 import train_objectstate_model_v0
from objgauss.ply import write_ply
from objgauss.splat import write_splat

OBJECTSTATE_MULTI_OBJECT_RUN_SCHEMA = "objgauss-objectstate-multi-object-run-v1"

__all__ = (
    "OBJECTSTATE_MULTI_OBJECT_RUN_SCHEMA",
    "MultiObjectBenchmarkRun",
    "run_multi_object_instance_benchmark",
)


@dataclass(frozen=True)
class MultiObjectBenchmarkRun:
    output_dir: Path
    dataset_manifest_path: Path
    checkpoint_path: Path
    training_summary_path: Path
    benchmark_path: Path
    manifest_path: Path
    viewer_manifest_path: Path | None
    scene_artifacts: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def run_multi_object_instance_benchmark(
    output_dir: str | Path,
    *,
    run_id: str = "objectstate-multi-object-m2",
    scene_count: int = 12,
    points_per_instance: int = 128,
    heldout_stride: int = 3,
    iterations: int = 240,
    learning_rate: float = 0.08,
    hidden_dim: int = 24,
    connected_component_radius: float = 0.18,
    seed: int = 0,
    dataset_seed: int = 20260713,
    viewer_dir: str | Path | None = None,
) -> MultiObjectBenchmarkRun:
    if not run_id:
        raise ValueError("run_id is required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_multi_object_synthetic_dataset(
        scene_count=scene_count,
        points_per_instance=points_per_instance,
        heldout_stride=heldout_stride,
        split_seed=seed,
        seed=dataset_seed,
    )
    training = train_objectstate_model_v0(
        dataset.cloud,
        object_id_field="gt_instance_id",
        frame_field="scene_id",
        heldout_stride=heldout_stride,
        hidden_dim=hidden_dim,
        iterations=iterations,
        learning_rate=learning_rate,
        seed=seed,
    )
    benchmark = evaluate_multi_object_instance_benchmark(
        dataset,
        training.final_assignment,
        connected_component_radius=connected_component_radius,
        seed=seed,
    )

    dataset_manifest_path = output_dir / "dataset-manifest.json"
    checkpoint_path = output_dir / "checkpoint.json"
    training_summary_path = output_dir / "training-summary.json"
    benchmark_path = output_dir / "benchmark.json"
    manifest_path = output_dir / "model-manifest.json"
    _write_json(dataset_manifest_path, dataset.as_dict())
    _write_json(checkpoint_path, training.final_state.as_dict(include_arrays=True))
    _write_json(training_summary_path, training.as_dict())

    scene_artifacts = _write_heldout_scenes(
        dataset,
        np.argmax(training.final_assignment, axis=1),
        output_dir / "scenes",
    )
    summary = {
        "schema": OBJECTSTATE_MULTI_OBJECT_RUN_SCHEMA,
        "kind": "objectstate_multi_object_instance_benchmark_run",
        "run_id": run_id,
        "status": benchmark["status"],
        "dataset": {
            "schema": dataset.schema,
            "manifest": dataset_manifest_path.name,
            "scene_count": len(dataset.scenes),
            "gaussian_count": dataset.cloud.count,
            "train_scene_ids": list(dataset.train_scene_ids),
            "heldout_scene_ids": list(dataset.heldout_scene_ids),
        },
        "model": {
            "schema": training.final_state.schema,
            "checkpoint": checkpoint_path.name,
            "training_summary": training_summary_path.name,
            "training_status": training.as_dict()["status"],
            "feature_order": list(training.final_state.as_dict()["feature_order"]),
        },
        "benchmark": benchmark,
        "scene_artifacts": list(scene_artifacts),
        "claim_policy": {
            "lego_color_rule_metrics_not_objectness_evidence": True,
            "synthetic_multi_object_instance_evidence_only": True,
            "failed_result_remains_visible": True,
            "does_not_claim_reality_gate_pass": True,
        },
    }
    _write_json(benchmark_path, summary)
    _write_viewer_manifest(
        manifest_path,
        run_id=run_id,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        benchmark=benchmark,
        scene_artifacts=scene_artifacts,
    )
    viewer_manifest_path = None
    if viewer_dir is not None:
        viewer_manifest_path = _stage_viewer_package(
            viewer_dir=Path(viewer_dir),
            output_dir=output_dir,
            manifest_path=manifest_path,
            checkpoint_path=checkpoint_path,
            selected_scene=scene_artifacts[0],
        )
    return MultiObjectBenchmarkRun(
        output_dir=output_dir,
        dataset_manifest_path=dataset_manifest_path,
        checkpoint_path=checkpoint_path,
        training_summary_path=training_summary_path,
        benchmark_path=benchmark_path,
        manifest_path=manifest_path,
        viewer_manifest_path=viewer_manifest_path,
        scene_artifacts=scene_artifacts,
        summary=summary,
    )


def _write_heldout_scenes(
    dataset: MultiObjectSyntheticDataset,
    predicted_labels: np.ndarray,
    output_dir: Path,
) -> tuple[dict[str, Any], ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_values = dataset.cloud.vertices["scene_id"].astype(np.int64, copy=False)
    target_values = dataset.cloud.vertices["gt_instance_id"].astype(np.int64, copy=False)
    scene_artifacts: list[dict[str, Any]] = []
    for scene_id in dataset.heldout_scene_ids:
        scene_dir = output_dir / f"scene-{scene_id:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        mask = scene_values == scene_id
        raw = _model_input_cloud(dataset.cloud.with_vertices(dataset.cloud.vertices[mask].copy()))
        prediction = apply_object_colors(
            assign_object_ids(raw, np.asarray(predicted_labels[mask], dtype=np.int32))
        )
        ground_truth = apply_object_colors(
            assign_object_ids(raw, np.asarray(target_values[mask], dtype=np.int32))
        )
        raw_path = scene_dir / "raw.ply"
        raw_splat_path = scene_dir / "raw.splat"
        prediction_path = scene_dir / "prediction.ply"
        ground_truth_path = scene_dir / "ground-truth.ply"
        write_ply(raw_path, raw, fmt="binary_little_endian")
        write_splat(raw_splat_path, raw)
        write_ply(prediction_path, prediction, fmt="binary_little_endian")
        write_ply(ground_truth_path, ground_truth, fmt="binary_little_endian")
        scene_artifacts.append(
            {
                "scene_id": int(scene_id),
                "gaussian_count": raw.count,
                "object_count": int(np.unique(target_values[mask]).shape[0]),
                "raw": _artifact_record(raw_path, output_dir.parent),
                "raw_splat": _artifact_record(raw_splat_path, output_dir.parent),
                "prediction": _artifact_record(prediction_path, output_dir.parent),
                "ground_truth": _artifact_record(ground_truth_path, output_dir.parent),
            }
        )
    return tuple(scene_artifacts)


def _write_viewer_manifest(
    path: Path,
    *,
    run_id: str,
    output_dir: Path,
    checkpoint_path: Path,
    benchmark: dict[str, Any],
    scene_artifacts: tuple[dict[str, Any], ...],
) -> None:
    selected = scene_artifacts[0]
    scene_id = int(selected["scene_id"])

    def artifact(
        role: str,
        record: dict[str, Any],
        *,
        format: str,
        delivery_tier: str = "browser_edit",
        object_count: int | None = None,
        label: str,
    ) -> dict[str, Any]:
        actual = output_dir / record["path"]
        payload = build_model_artifact(
            role=role,
            path=actual,
            format=format,
            delivery_tier=delivery_tier,
            browser_ready=True,
            gaussian_count=int(selected["gaussian_count"]),
            object_count=object_count,
            compute_hash=True,
            label=label,
        )
        payload["path"] = record["path"]
        return payload

    checkpoint_record = _artifact_record(checkpoint_path, output_dir)
    artifacts = [
        artifact(
            "quick_splat",
            selected["raw_splat"],
            format=".splat",
            delivery_tier="browser_quick",
            label="Held-out raw Gaussian appearance",
        ),
        artifact(
            "model_input",
            selected["raw"],
            format=".ply",
            label="Held-out Model v0 input without instance GT",
        ),
        artifact(
            "object_edit",
            selected["prediction"],
            format=".ply",
            object_count=int(selected["object_count"]),
            label="Held-out Model v0 prediction",
        ),
        artifact(
            "ground_truth",
            selected["ground_truth"],
            format=".ply",
            object_count=int(selected["object_count"]),
            label="Independent held-out instance ground truth",
        ),
        artifact(
            "objectstate_model",
            checkpoint_record,
            format=".json",
            object_count=int(selected["object_count"]),
            label="Model v0 checkpoint trained on disjoint scenes",
        ),
    ]
    scene_row = next(
        scene for scene in benchmark["scenes"] if int(scene["scene_id"]) == scene_id
    )
    manifest = build_model_artifact_manifest(
        manifest_id=f"{run_id}-model-artifacts",
        asset_id=run_id,
        name=f"ObjectState M2 multi-object scene {scene_id}",
        stage="objectstate-multi-object-m2",
        source={
            "type": "procedural_multi_object_instance_benchmark",
            "run_id": run_id,
            "scene_id": scene_id,
            "target_source": "procedural_instance_authorship",
            "target_derived_from_rgb": False,
            "split": "heldout_scene",
        },
        license="ObjGauss procedural synthetic research fixture",
        gaussian_count=int(selected["gaussian_count"]),
        object_count=int(selected["object_count"]),
        artifacts=artifacts,
        quality_evidence=[
            {
                "kind": "multi_object_instance_segmentation_benchmark",
                "schema": benchmark["schema"],
                "source": "benchmark.json",
                "status": benchmark["status"],
                "leakage_gate": benchmark["leakage_gate"],
                "aggregate": benchmark["aggregate"],
                "comparison": benchmark["comparison"],
                "scene": scene_row,
            }
        ],
        limitations=[
            "Synthetic multi-object instance evidence; not a controlled-real or Reality Gate pass.",
            "Model v0 is baseline-inferior when the recorded comparison verdict says so.",
            "NeRF Lego color-rule metrics are wiring evidence only and are not objectness evidence.",
        ],
        created_from={
            "dataset_manifest": "dataset-manifest.json",
            "benchmark": "benchmark.json",
            "checkpoint": "checkpoint.json",
        },
    )
    _write_json(path, manifest)


def _model_input_cloud(cloud: GaussianCloud) -> GaussianCloud:
    forbidden = {"gt_instance_id", "shape_id", "object_id", "predicted_object_id"}
    names = [name for name in cloud.fields if name not in forbidden]
    dtype = np.dtype([(name, cloud.vertices.dtype.fields[name][0]) for name in names])
    vertices = np.empty(cloud.count, dtype=dtype)
    for name in names:
        vertices[name] = cloud.vertices[name]
    return GaussianCloud(
        vertices=vertices,
        comments=cloud.comments
        + ("model input excludes gt_instance_id and target-derived object_id",),
        source_format=cloud.source_format,
    )


def _stage_viewer_package(
    *,
    viewer_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    selected_scene: dict[str, Any],
) -> Path:
    if viewer_dir.resolve() == output_dir.resolve():
        raise ValueError("viewer_dir must differ from output_dir")
    viewer_dir.mkdir(parents=True, exist_ok=True)
    dependencies = [
        manifest_path,
        checkpoint_path,
        *(
            output_dir / selected_scene[name]["path"]
            for name in ("raw", "raw_splat", "prediction", "ground_truth")
        ),
    ]
    for source in dependencies:
        target = viewer_dir / source.relative_to(output_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return viewer_dir / manifest_path.name


def _artifact_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "byte_size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
