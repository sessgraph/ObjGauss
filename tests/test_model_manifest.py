from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.lod import LOD_SCHEMA
from objgauss.core.ogc_payload import write_ogc_payload
from objgauss.model_manifest import (
    CHUNK_INDEX_SCHEMA,
    MODEL_ARTIFACT_MANIFEST_SCHEMA,
    build_compressed_chunked_artifact,
    build_model_artifact,
    build_model_artifact_manifest,
    manifest_from_asset_library_entry,
    manifest_from_sample_bundle,
    manifest_from_trainable_kernel_model_artifact,
    manifest_from_training_output,
    read_model_artifact_manifest,
    validate_model_artifact_manifest,
    write_model_artifact_manifest,
)
from objgauss.core.trainable_artifact import write_trainable_kernel_model_artifact
from objgauss.core.trainable_kernel import make_trainable_kernel_mvp_fixture, train_kernel_mvp


def test_model_artifact_manifest_roundtrip(tmp_path):
    splat = tmp_path / "scene.splat"
    object_ply = tmp_path / "scene_objects.ply"
    full_ply = tmp_path / "scene_full.ply"
    splat.write_bytes(b"splat")
    object_ply.write_bytes(b"object-ply")
    full_ply.write_bytes(b"full-ply")

    manifest = build_model_artifact_manifest(
        manifest_id="lego-model-artifacts",
        asset_id="nerf-lego-trained",
        name="NeRF Lego Trained",
        source={"type": "external_3dgs_training_output", "dataset": "nerf-synthetic-lego"},
        license="NeRF official sample data; research use",
        gaussian_count=100,
        object_count=4,
        artifacts=[
            build_model_artifact(
                role="quick_splat",
                path=splat,
                format=".splat",
                delivery_tier="browser_quick",
                gaussian_count=100,
                compute_hash=True,
            ),
            build_model_artifact(
                role="object_edit",
                path=object_ply,
                format=".ply",
                delivery_tier="browser_edit",
                gaussian_count=100,
                object_count=4,
                compute_hash=True,
            ),
            build_model_artifact(
                role="diagnostic_full",
                path=full_ply,
                format=".ply",
                delivery_tier="diagnostic",
                browser_ready=False,
                gaussian_count=100,
                compute_hash=True,
            ),
        ],
        quality_evidence=[{"kind": "route_audit", "status": "passed"}],
        limitations=["development-stage fixture"],
    )

    assert manifest["schema"] == MODEL_ARTIFACT_MANIFEST_SCHEMA
    assert manifest["counts"] == {"gaussians": 100, "objects": 4}
    assert manifest["artifacts"][0]["browser_ready"] is True
    assert manifest["artifacts"][2]["browser_ready"] is False
    assert manifest["artifacts"][0]["sha256"] == hashlib.sha256(b"splat").hexdigest()

    validation = validate_model_artifact_manifest(manifest, require_browser_ready=True, require_object_edit=True)
    assert validation.passed
    assert validation.browser_ready_artifacts == 2

    path = tmp_path / "model-artifacts.json"
    write_model_artifact_manifest(path, manifest)
    loaded = read_model_artifact_manifest(path)
    assert loaded == manifest


def test_model_artifact_manifest_rejects_browser_ready_diagnostic_full(tmp_path):
    full_ply = tmp_path / "full.ply"
    full_ply.write_bytes(b"full")
    payload = {
        "schema": MODEL_ARTIFACT_MANIFEST_SCHEMA,
        "manifest_id": "bad",
        "asset_id": "bad-asset",
        "name": "Bad Asset",
        "stage": "development",
        "source": {"type": "fixture"},
        "license": "fixture",
        "counts": {"gaussians": 10, "objects": None},
        "artifacts": [
            build_model_artifact(
                role="diagnostic_full",
                path=full_ply,
                format=".ply",
                delivery_tier="browser_quick",
                browser_ready=True,
            )
        ],
        "quality_evidence": [],
        "limitations": [],
        "created_from": {},
    }

    validation = validate_model_artifact_manifest(payload, require_browser_ready=True)

    assert not validation.passed
    assert any("diagnostic_full artifact must not be browser_ready" in error for error in validation.errors)


def test_model_artifact_manifest_allows_browser_ready_trainable_kernel(tmp_path):
    artifact_path = tmp_path / "trainable-model-artifact.json"
    artifact_path.write_text('{"schema":"objgauss-trainable-kernel-model-artifact-v1"}', encoding="utf-8")
    manifest = build_model_artifact_manifest(
        manifest_id="trainable-debug-model-artifacts",
        asset_id="trainable-debug",
        name="Trainable Debug",
        source={"type": "trainable_kernel_debug_fixture"},
        license="fixture",
        gaussian_count=4,
        object_count=2,
        artifacts=[
            build_model_artifact(
                role="trainable_kernel",
                path=artifact_path,
                format=".json",
                delivery_tier="browser_edit",
                gaussian_count=4,
                object_count=2,
                compute_hash=True,
            )
        ],
        limitations=["Small trainable-kernel artifact for ObjectState Debug OS handoff."],
    )

    roles = {artifact["role"]: artifact for artifact in manifest["artifacts"]}
    assert roles["trainable_kernel"]["browser_ready"] is True
    assert roles["trainable_kernel"]["delivery_tier"] == "browser_edit"
    validation = validate_model_artifact_manifest(manifest, require_browser_ready=True)
    assert validation.passed
    assert validation.browser_ready_artifacts == 1


def test_model_artifact_manifest_allows_browser_ready_quality_report(tmp_path):
    report_path = tmp_path / "quality-report.json"
    report_path.write_text('{"schema":"objgauss-object-state-quality-report-v1"}', encoding="utf-8")
    manifest = build_model_artifact_manifest(
        manifest_id="quality-debug-model-artifacts",
        asset_id="quality-debug",
        name="Quality Debug",
        source={"type": "quality_report_debug_fixture"},
        license="fixture",
        artifacts=[
            build_model_artifact(
                role="quality_report",
                path=report_path,
                format=".json",
                delivery_tier="browser_edit",
                compute_hash=True,
            )
        ],
        limitations=["Small ObjectState quality report for Debug OS handoff."],
    )

    roles = {artifact["role"]: artifact for artifact in manifest["artifacts"]}
    assert roles["quality_report"]["browser_ready"] is True
    assert roles["quality_report"]["delivery_tier"] == "browser_edit"
    validation = validate_model_artifact_manifest(manifest, require_browser_ready=True)
    assert validation.passed
    assert validation.browser_ready_artifacts == 1


def test_manifest_from_trainable_kernel_model_artifact(tmp_path):
    result = train_kernel_mvp(
        make_trainable_kernel_mvp_fixture(),
        slots=2,
        iterations=4,
        learning_rate=0.35,
        seed=7,
    )
    artifact_path = tmp_path / "trainable-kernel-model.json"
    write_trainable_kernel_model_artifact(
        artifact_path,
        result,
        input_path="fixture://trainable-kernel-mvp",
    )

    manifest = manifest_from_trainable_kernel_model_artifact(
        artifact_path,
        artifact_path="trainable-kernel-model.json",
        manifest_id="trainable-kernel-model-artifacts",
        asset_id="trainable-kernel-fixture",
        name="Trainable kernel fixture",
        license="fixture",
    )

    roles = {artifact["role"]: artifact for artifact in manifest["artifacts"]}
    assert manifest["schema"] == MODEL_ARTIFACT_MANIFEST_SCHEMA
    assert manifest["asset_id"] == "trainable-kernel-fixture"
    assert manifest["counts"] == {"gaussians": 6, "objects": 2}
    assert roles["trainable_kernel"]["path"] == "trainable-kernel-model.json"
    assert roles["trainable_kernel"]["browser_ready"] is True
    assert roles["trainable_kernel"]["delivery_tier"] == "browser_edit"
    assert roles["trainable_kernel"]["byte_size"] == artifact_path.stat().st_size
    assert roles["trainable_kernel"]["sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert manifest["quality_evidence"][0]["kind"] == "trainable_kernel_training_summary"
    assert manifest["created_from"]["schema"] == "objgauss-trainable-kernel-model-artifact-v1"
    validation = validate_model_artifact_manifest(manifest, require_browser_ready=True)
    assert validation.passed
    assert validation.browser_ready_artifacts == 1


def test_manifest_from_training_output(tmp_path):
    training_manifest = tmp_path / "training-output-manifest.json"
    training_manifest.write_text(
        json.dumps(
            {
                "asset_id": "lego-trained",
                "dataset": "outputs/assets/training/nerf-synthetic-lego",
                "input": "outputs/assets/gaussians/lego/gaussians.ply",
                "input_format": "ply",
                "gaussian_source": "external_3dgs_training_output",
                "gaussian_count": 42,
                "slots": 3,
                "gaussian_ply": "outputs/assets/gaussians/lego/gaussians.ply",
                "splat_path": "outputs/assets/gaussians/lego/gaussians.splat",
                "object_ply": "outputs/assets/gaussians/lego/object_aware_gaussians.ply",
                "trained_field": "outputs/assets/gaussians/lego/object_field_trained.npz",
                "mask_manifest": "outputs/masks/lego/mask-manifest.json",
                "training": {"initial_loss": 2.0, "final_loss": 0.5, "supervised_gaussians": 20},
            }
        ),
        encoding="utf-8",
    )

    manifest = manifest_from_training_output(
        training_manifest,
        name="Lego trained fixture",
        license="NeRF official sample data; research use",
    )

    roles = {artifact["role"]: artifact for artifact in manifest["artifacts"]}
    assert manifest["schema"] == MODEL_ARTIFACT_MANIFEST_SCHEMA
    assert manifest["asset_id"] == "lego-trained"
    assert manifest["counts"] == {"gaussians": 42, "objects": 3}
    assert roles["quick_splat"]["delivery_tier"] == "browser_quick"
    assert roles["object_edit"]["delivery_tier"] == "browser_edit"
    assert roles["diagnostic_full"]["browser_ready"] is False
    assert roles["object_field"]["delivery_tier"] == "training_internal"
    assert manifest["quality_evidence"][0]["kind"] == "mask_training_summary"


def test_manifest_from_sample_bundle():
    sample = {
        "sample_id": "objgauss-lego-alpha-fgbg-test",
        "asset_id": "nerf-synthetic-lego",
        "dataset": "outputs/assets/training/nerf-synthetic-lego",
        "mask_manifest": "outputs/masks/lego/mask-manifest.json",
        "training_manifest": "outputs/assets/gaussians/lego/training-output-manifest.json",
        "gaussian_ply": "outputs/assets/gaussians/lego/gaussians.ply",
        "splat_path": "public/samples/lego_alpha_proxy.splat",
        "object_ply": "public/samples/lego_alpha_v1_objects.ply",
        "object_field_path": "outputs/assets/gaussians/lego/object_field_trained.npz",
        "gaussian_count": 40,
        "object_ply_gaussian_count": 40,
        "object_field_gaussian_count": 40,
        "object_field_slot_count": 2,
        "slot_count": 2,
        "training": {"initial_loss": 1.0, "final_loss": 0.2},
    }
    path = _write_json(sample)

    manifest = manifest_from_sample_bundle(path, license="NeRF sample data; research use")

    roles = {artifact["role"]: artifact for artifact in manifest["artifacts"]}
    assert roles["quick_splat"]["browser_ready"] is True
    assert roles["object_edit"]["browser_ready"] is True
    assert roles["source_gaussian"]["browser_ready"] is False
    assert roles["object_field"]["delivery_tier"] == "training_internal"
    assert manifest["quality_evidence"][0]["kind"] == "sample_bundle_training_summary"


def test_manifest_from_asset_library_entry_maps_regular_object_ply_to_browser_edit():
    entry = {
        "id": "plush-3dgs-local",
        "name": "Plush 3DGS 示例",
        "sourceType": "gaussian",
        "category": "本地样例",
        "status": "已接入",
        "pipelineStage": "Demo 可用",
        "useCases": ["Demo预览"],
        "localPath": "/samples/plush_objects.ply",
        "splatPath": "/samples/plush.splat",
        "fileName": "plush_objects.ply",
        "sourceName": "cakewalk/splat-data",
        "sourceUrl": "https://huggingface.co/cakewalk/splat-data/blob/main/plush.splat",
        "license": "来源许可混合，仅用于本地测试",
    }

    manifest = manifest_from_asset_library_entry(entry)

    roles = {artifact["role"]: artifact for artifact in manifest["artifacts"]}
    assert roles["quick_splat"]["delivery_tier"] == "browser_quick"
    assert roles["object_edit"]["delivery_tier"] == "browser_edit"
    assert roles["object_edit"]["browser_ready"] is True


def test_manifest_from_asset_library_entry_keeps_near1m_full_ply_diagnostic():
    entry = {
        "id": "nerf-lego-trained-near1m-random1300k-local",
        "name": "NeRF Lego near-1M 训练输出样例",
        "sourceType": "gaussian",
        "category": "本地诊断",
        "status": "训练后可用",
        "pipelineStage": "near-1M terminal proof candidate",
        "useCases": ["near-1M训练输出", "ObjectField", "WebGPU SLA", "对象编辑"],
        "localPath": "/samples/nerf_lego_trained_near1m_random1300k_objects.ply",
        "splatPath": "/samples/nerf_lego_trained_near1m_random1300k.splat",
        "fileName": "nerf_lego_trained_near1m_random1300k_objects.ply",
        "splatFileName": "nerf_lego_trained_near1m_random1300k.splat",
        "gaussianCount": 4503634,
        "deferObjectPly": True,
        "objectPlySizeLabel": "1.15GB",
        "splatSizeLabel": "144MB",
        "sourceName": "Splatfacto tuned near-1M + NeRF Synthetic Lego",
        "sourceUrl": "https://github.com/bmild/nerf",
        "license": "NeRF 官方示例数据，仅训练/研究使用",
    }

    manifest = manifest_from_asset_library_entry(entry)

    roles = {artifact["role"]: artifact for artifact in manifest["artifacts"]}
    assert roles["quick_splat"]["browser_ready"] is True
    assert roles["diagnostic_full"]["delivery_tier"] == "diagnostic"
    assert roles["diagnostic_full"]["browser_ready"] is False
    assert "Deferred or large object PLY" in roles["diagnostic_full"]["note"]
    assert "object_edit" not in roles
    validation = validate_model_artifact_manifest(manifest, require_browser_ready=True)
    assert validation.passed


def test_browser_ready_chunked_gaussian_artifact_contract():
    chunked = build_model_artifact(
        role="compressed_chunked",
        path="public/samples/lego.ogc",
        format=".ogc",
        delivery_tier="browser_edit",
        gaussian_count=1_000_000,
        object_count=4,
        byte_size=64_000_000,
        sha256="a" * 64,
        chunk_index={
            "schema": CHUNK_INDEX_SCHEMA,
            "path": "public/samples/lego.index.json",
            "chunk_count": 128,
            "sort_key": "object_id+morton_xyz",
            "chunk_size_target": 8192,
        },
        compression={
            "codec": "objgauss-ogc",
            "version": "0.1",
            "layout": "object-aware-chunked",
            "quantization": {
                "xyz": "chunk-aabb-uint16",
                "opacity": "uint8",
                "object_id": "uint16-rle",
            },
        },
        lod={
            "selection": "object-aware-importance",
            "levels": [
                {"level": 0, "gaussian_count": 1_000_000, "ratio": 1.0},
                {"level": 1, "gaussian_count": 250_000, "ratio": 0.25},
                {"level": 2, "gaussian_count": 50_000, "ratio": 0.05},
            ],
        },
        object_id_coverage={
            "field": "object_id",
            "mode": "complete",
            "has_object_ids": True,
            "object_count": 4,
        },
    )
    manifest = build_model_artifact_manifest(
        manifest_id="lego-chunked-model-artifacts",
        asset_id="lego-chunked",
        name="Lego Chunked",
        source={"type": "object_aware_gaussian_codec", "input": "outputs/lego/object_aware_gaussians.ply"},
        license="NeRF sample data; research use",
        gaussian_count=1_000_000,
        object_count=4,
        artifacts=[
            build_model_artifact(
                role="quick_splat",
                path="public/samples/lego.splat",
                format=".splat",
                delivery_tier="browser_quick",
                gaussian_count=1_000_000,
                byte_size=144_000_000,
            ),
            chunked,
            build_model_artifact(
                role="diagnostic_full",
                path="outputs/lego/object_aware_gaussians.ply",
                format=".ply",
                delivery_tier="diagnostic",
                browser_ready=False,
                gaussian_count=4_503_634,
                object_count=4,
                byte_size=1_234_000_000,
            ),
        ],
        limitations=["Chunked OGC fixture only; codec implementation is not included in this manifest test."],
    )

    roles = {artifact["role"]: artifact for artifact in manifest["artifacts"]}
    assert roles["compressed_chunked"]["browser_ready"] is True
    assert roles["compressed_chunked"]["delivery_tier"] == "browser_edit"
    assert roles["compressed_chunked"]["chunk_index"]["schema"] == CHUNK_INDEX_SCHEMA
    assert roles["compressed_chunked"]["object_id_coverage"]["has_object_ids"] is True
    assert roles["diagnostic_full"]["browser_ready"] is False
    validation = validate_model_artifact_manifest(manifest, require_browser_ready=True)
    assert validation.passed


def test_build_compressed_chunked_artifact_from_ogc_payload_index(tmp_path):
    cloud = _object_cloud()
    payload_path = tmp_path / "scene.ogc"
    index_path = tmp_path / "scene.index.json"
    result = write_ogc_payload(payload_path, cloud, index_path=index_path, chunk_size_target=2)

    artifact = build_compressed_chunked_artifact(
        payload_path=result.payload_path,
        chunk_index=index_path,
        label="Object-aware chunked OGC",
    )

    assert artifact["role"] == "compressed_chunked"
    assert artifact["path"] == str(payload_path)
    assert artifact["format"] == ".ogc"
    assert artifact["delivery_tier"] == "browser_edit"
    assert artifact["browser_ready"] is True
    assert artifact["gaussian_count"] == 6
    assert artifact["object_count"] == 2
    assert artifact["byte_size"] == result.byte_size
    assert artifact["sha256"] == result.sha256
    assert artifact["chunk_index"] == {
        "schema": CHUNK_INDEX_SCHEMA,
        "path": str(index_path),
        "chunk_count": len(result.index["chunks"]),
        "sort_key": "object_id+morton_xyz",
        "chunk_size_target": 2,
    }
    assert artifact["compression"]["codec"] == "objgauss-ogc-prototype"
    assert artifact["compression"]["quantization"]["schema"] == "objgauss-local-quantization-v1"
    assert artifact["compression"]["quantization"]["estimated_payload_byte_size"] == 6 * 10
    assert artifact["lod"]["schema"] == LOD_SCHEMA
    assert [level["gaussian_count"] for level in artifact["lod"]["levels"]] == [6, 4, 2, 2]
    assert artifact["object_id_coverage"] == {
        "field": "object_id",
        "mode": "complete",
        "has_object_ids": True,
        "object_count": 2,
    }

    manifest = build_model_artifact_manifest(
        manifest_id="fixture-ogc-model-artifacts",
        asset_id="fixture-ogc",
        name="Fixture OGC",
        source={"type": "object_aware_gaussian_codec", "input": "outputs/fixture/object_aware_gaussians.ply"},
        license="fixture",
        gaussian_count=6,
        object_count=2,
        artifacts=[
            build_model_artifact(
                role="quick_splat",
                path="public/samples/fixture.splat",
                format=".splat",
                delivery_tier="browser_quick",
                gaussian_count=6,
            ),
            artifact,
            build_model_artifact(
                role="diagnostic_full",
                path="outputs/fixture/object_aware_gaussians.ply",
                format=".ply",
                delivery_tier="diagnostic",
                browser_ready=False,
                gaussian_count=6,
                object_count=2,
            ),
        ],
    )

    roles = {entry["role"]: entry for entry in manifest["artifacts"]}
    assert roles["compressed_chunked"]["browser_ready"] is True
    assert roles["diagnostic_full"]["browser_ready"] is False
    validation = validate_model_artifact_manifest(manifest, require_browser_ready=True)
    assert validation.passed


def test_build_compressed_chunked_artifact_requires_index_path_for_inline_index(tmp_path):
    cloud = _object_cloud()
    result = write_ogc_payload(tmp_path / "scene.ogc", cloud, chunk_size_target=2)

    with pytest.raises(ValueError, match="chunk_index_path"):
        build_compressed_chunked_artifact(
            payload_path=result.payload_path,
            chunk_index=result.index,
        )


def test_browser_ready_chunked_artifact_requires_index_lod_and_object_coverage():
    payload = _base_manifest_payload(
        artifacts=[
            build_model_artifact(
                role="compressed_chunked",
                path="public/samples/bad.ogc",
                format=".ogc",
                delivery_tier="browser_edit",
                gaussian_count=100,
                object_count=2,
                byte_size=4096,
                sha256="b" * 64,
                compression={"codec": "objgauss-ogc", "version": "0.1", "layout": "object-aware-chunked"},
            )
        ]
    )

    validation = validate_model_artifact_manifest(payload, require_browser_ready=True)

    assert not validation.passed
    assert "artifacts[0].chunk_index is required for browser_ready compressed_chunked artifacts" in validation.errors
    assert "artifacts[0].lod is required for browser_ready compressed_chunked artifacts" in validation.errors
    assert (
        "artifacts[0].object_id_coverage is required for browser_ready compressed_chunked artifacts"
        in validation.errors
    )


def test_browser_ready_chunked_artifact_requires_hash_counts_and_bytes():
    payload = _base_manifest_payload(
        artifacts=[
            build_model_artifact(
                role="compressed_chunked",
                path="public/samples/bad.ogc",
                format=".ogc",
                delivery_tier="browser_edit",
                chunk_index={
                    "schema": CHUNK_INDEX_SCHEMA,
                    "path": "public/samples/bad.index.json",
                    "chunk_count": 3,
                    "sort_key": "object_id+morton_xyz",
                },
                compression={"codec": "objgauss-ogc", "version": "0.1", "layout": "object-aware-chunked"},
                lod={"levels": [{"level": 0, "gaussian_count": 10}]},
                object_id_coverage={
                    "field": "object_id",
                    "mode": "complete",
                    "has_object_ids": True,
                    "object_count": 2,
                },
            )
        ]
    )

    validation = validate_model_artifact_manifest(payload, require_browser_ready=True)

    assert not validation.passed
    assert "artifacts[0].gaussian_count is required for browser_ready compressed_chunked artifacts" in validation.errors
    assert "artifacts[0].object_count is required for browser_ready compressed_chunked artifacts" in validation.errors
    assert "artifacts[0].byte_size is required for browser_ready compressed_chunked artifacts" in validation.errors
    assert "artifacts[0].sha256 is required for browser_ready compressed_chunked artifacts" in validation.errors


def test_manifest_requires_browser_ready_artifact():
    payload = {
        "schema": MODEL_ARTIFACT_MANIFEST_SCHEMA,
        "manifest_id": "source-only",
        "asset_id": "source-only",
        "name": "Source only",
        "stage": "development",
        "source": {"type": "fixture"},
        "license": "fixture",
        "counts": {"gaussians": 1, "objects": None},
        "artifacts": [
            {
                "role": "source_gaussian",
                "path": "outputs/source.ply",
                "format": ".ply",
                "delivery_tier": "training_internal",
                "browser_ready": False,
            }
        ],
        "quality_evidence": [],
        "limitations": [],
        "created_from": {},
    }

    validation = validate_model_artifact_manifest(payload, require_browser_ready=True)

    assert not validation.passed
    assert "at least one browser_ready artifact is required" in validation.errors


def _base_manifest_payload(*, artifacts):
    return {
        "schema": MODEL_ARTIFACT_MANIFEST_SCHEMA,
        "manifest_id": "fixture",
        "asset_id": "fixture-asset",
        "name": "Fixture Asset",
        "stage": "development",
        "source": {"type": "fixture"},
        "license": "fixture",
        "counts": {"gaussians": None, "objects": None},
        "artifacts": artifacts,
        "quality_evidence": [],
        "limitations": [],
        "created_from": {},
    }


def _object_cloud() -> GaussianCloud:
    vertices = np.zeros(
        6,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("opacity", "f4"),
            ("object_id", "i4"),
        ],
    )
    vertices["object_id"] = np.array([1, 0, 1, 0, 1, 0], dtype=np.int32)
    vertices["x"] = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    vertices["y"] = np.array([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0], dtype=np.float32)
    vertices["z"] = np.zeros(6, dtype=np.float32)
    vertices["red"] = np.array([10, 20, 30, 40, 50, 60], dtype=np.uint8)
    vertices["green"] = np.array([11, 21, 31, 41, 51, 61], dtype=np.uint8)
    vertices["blue"] = np.array([12, 22, 32, 42, 52, 62], dtype=np.uint8)
    vertices["opacity"] = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    return GaussianCloud(vertices=vertices, source_format="fixture")


def _write_json(payload):
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "payload.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
