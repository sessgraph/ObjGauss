from __future__ import annotations

import hashlib
import json

import numpy as np

from objgauss.cli import main
from objgauss.core.assignment_metrics import assignment_clustering_metrics
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.io import read_ply, write_ply
from objgauss.model_manifest import validate_model_artifact_manifest
from objgauss.pipelines.objectstate_model_demo import write_objectstate_model_demo
from objgauss.pipelines.objectstate_model_v0 import objectstate_model_v0_state_from_dict


def test_objectstate_model_demo_writes_same_run_bundle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cloud = _object_cloud()
    input_path = tmp_path / "public-sample.ply"
    source_splat = tmp_path / "public-sample.splat"
    output_dir = tmp_path / "demo"
    viewer_dir = tmp_path / "public" / "models" / "objectstate-model-demo-local"
    viewer_dir.mkdir(parents=True)
    (viewer_dir / "supervision-source.ply").write_bytes(b"legacy-generated-source")
    (viewer_dir / "model-manifest.json").write_text(
        json.dumps(
            {
                "schema": "objgauss-model-artifact-manifest-v1",
                "artifacts": [
                    {"role": "source_gaussian", "path": "supervision-source.ply"}
                ],
                "quality_evidence": [],
                "created_from": {},
            }
        ),
        encoding="utf-8",
    )
    write_ply(input_path, cloud, fmt="ascii")
    source_splat.write_bytes(b"small-splat-source")

    result = write_objectstate_model_demo(
        cloud,
        input_path=input_path,
        output_dir=output_dir,
        run_id="public-sample-model-demo",
        license="research fixture",
        target_source="object_id test supervision",
        source_url="https://example.invalid/public-sample",
        source_splat=source_splat,
        viewer_dir=viewer_dir,
        hidden_dim=12,
        heldout_stride=2,
        iterations=120,
        learning_rate=0.12,
        weight_decay=0.0,
        seed=7,
    )

    assert result.summary["status"] == "objectstate_model_v0_training_pass"
    assert result.summary["dataset"]["source_kind"] == "public_replay"
    assert result.summary["dataset"]["target_source"] == "object_id test supervision"
    assert result.checkpoint_path.name == "checkpoint.json"
    assert result.before_ply_path.name == "before.ply"
    assert result.after_ply_path.name == "after-object-color.ply"
    assert result.metrics_path.name == "metrics.json"
    assert result.manifest_path.name == "model-manifest.json"
    assert result.viewer_manifest_path == viewer_dir / "model-manifest.json"
    assert result.viewer_url == (
        "/?modelArtifactManifest=/models/objectstate-model-demo-local/model-manifest.json"
    )
    assert not (output_dir / "summary.json").exists()
    assert not (output_dir / "assignment-before.ply").exists()
    assert not (output_dir / "assignment-after.ply").exists()

    after_cloud = read_ply(result.after_ply_path)
    assert after_cloud.vertices["object_id"].tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    assert after_cloud.vertices["predicted_object_id"].tolist() == [0, 0, 1, 1, 0, 0, 1, 1]

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    quality = json.loads(result.quality_report_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(result.checkpoint_path.read_text(encoding="utf-8"))
    assert metrics["demo"]["run_id"] == "public-sample-model-demo"
    assert quality["metrics"]["ari"] == metrics["heldout_after_metrics"]["ari"]
    assert quality["metrics"]["mean_best_iou"] == metrics["heldout_after_metrics"]["mean_best_iou"]
    assert quality["metrics"]["object_purity"] == metrics["heldout_after_metrics"]["purity"]

    restored = objectstate_model_v0_state_from_dict(checkpoint)
    checkpoint_ids = np.argmax(restored.predict(cloud), axis=1)
    artifact_ids = after_cloud.vertices["object_id"].astype(np.int64)
    np.testing.assert_array_equal(checkpoint_ids, artifact_ids)
    heldout_mask = np.isin(
        cloud.vertices["source_frame"],
        np.asarray(metrics["split"]["heldout_frame_ids"], dtype=np.int64),
    )
    independently_recomputed = assignment_clustering_metrics(
        artifact_ids[heldout_mask],
        cloud.vertices["object_id"][heldout_mask],
    )
    assert independently_recomputed["ari"] == metrics["heldout_after_metrics"]["ari"]
    assert independently_recomputed["mean_best_iou"] == metrics["heldout_after_metrics"][
        "mean_best_iou"
    ]
    assert independently_recomputed["purity"] == metrics["heldout_after_metrics"]["purity"]

    assert validate_model_artifact_manifest(
        manifest,
        require_browser_ready=True,
        require_object_edit=True,
    ).passed
    roles = {artifact["role"]: artifact for artifact in manifest["artifacts"]}
    assert roles["object_edit"]["path"] == "after-object-color.ply"
    assert roles["training_summary"]["path"] == "metrics.json"
    assert roles["quality_report"]["path"] == "quality-report.json"
    assert roles["model_input"]["role"] == "model_input"
    assert roles["objectstate_model"]["path"] == "checkpoint.json"
    assert roles["quick_splat"]["sha256"] == hashlib.sha256(b"small-splat-source").hexdigest()
    assert manifest["quality_evidence"][0]["run_id"] == metrics["demo"]["run_id"]
    assert manifest["quality_evidence"][0]["checkpoint_sha256"] == hashlib.sha256(
        result.checkpoint_path.read_bytes()
    ).hexdigest()
    staged_manifest = json.loads(result.viewer_manifest_path.read_text(encoding="utf-8"))
    staged_roles = {artifact["role"]: artifact for artifact in staged_manifest["artifacts"]}
    assert staged_roles["quick_splat"]["path"] == "raw-source.splat"
    assert staged_roles["object_edit"]["path"] == "after-object-color.ply"
    assert staged_roles["model_input"]["path"] == "model-input.ply"
    assert staged_roles["objectstate_model"]["path"] == "checkpoint.json"
    assert (viewer_dir / "checkpoint.json").is_file()
    assert hashlib.sha256((viewer_dir / "checkpoint.json").read_bytes()).hexdigest() == (
        staged_manifest["quality_evidence"][0]["checkpoint_sha256"]
    )
    assert {path.name for path in viewer_dir.iterdir()} == {
        "after-object-color.ply",
        "before.ply",
        "checkpoint.json",
        "metrics.json",
        "model-input.ply",
        "model-manifest.json",
        "quality-report.json",
        "raw-source.splat",
    }


def test_objectstate_model_demo_cli_requires_explicit_provenance(tmp_path, capsys):
    input_path = tmp_path / "public-sample.ply"
    output_dir = tmp_path / "cli-demo"
    write_ply(input_path, _object_cloud(), fmt="ascii")

    assert (
        main(
            [
                "training",
                "objectstate-model-demo",
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--run-id",
                "cli-public-demo",
                "--license",
                "research fixture",
                "--target-source",
                "object_id test supervision",
                "--iterations",
                "8",
                "--viewer-dir",
                str(tmp_path / "viewer-package"),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "run_id=cli-public-demo" in output
    assert f"checkpoint={output_dir / 'checkpoint.json'}" in output
    assert f"model_manifest={output_dir / 'model-manifest.json'}" in output
    assert f"viewer_manifest={tmp_path / 'viewer-package' / 'model-manifest.json'}" in output


def _object_cloud() -> GaussianCloud:
    dtype = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "f4"),
            ("green", "f4"),
            ("blue", "f4"),
            ("opacity", "f4"),
            ("object_id", "i4"),
            ("source_frame", "i4"),
        ]
    )
    vertices = np.zeros(8, dtype=dtype)
    vertices["x"] = [-1.0, -0.8, 0.8, 1.0, -0.98, -0.78, 0.82, 1.02]
    vertices["y"] = [0.0, 0.05, 0.0, -0.05, 0.01, 0.06, 0.01, -0.04]
    vertices["red"] = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    vertices["blue"] = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    vertices["opacity"] = 1.0
    vertices["object_id"] = [4, 4, 9, 9, 4, 4, 9, 9]
    vertices["source_frame"] = [0, 0, 0, 0, 1, 1, 1, 1]
    return GaussianCloud(vertices)
