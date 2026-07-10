from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.pipelines.objectstate_bop_prediction_baseline_handoff import (
    OBJECTSTATE_BOP_PREDICTION_BASELINE_HANDOFF_SCHEMA,
    objectstate_bop_prediction_baseline_handoff,
    validate_objectstate_bop_prediction_baseline_handoff_summary,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n"
PLY_BYTES = (
    b"ply\n"
    b"format ascii 1.0\n"
    b"element vertex 1\n"
    b"property float x\n"
    b"property float y\n"
    b"property float z\n"
    b"end_header\n"
    b"0 0 0\n"
)


def test_bop_prediction_baseline_handoff_writes_reviewable_package(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "prediction-package"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)

    summary = objectstate_bop_prediction_baseline_handoff(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
        candidate_id="fixture-bop-baseline",
        candidate_source="unit-test BOP baseline",
        artifact_ref="outputs/captures/bop-ycbv-scene-000001/objectstates.json",
        confidence=0.9,
    )

    assert summary["schema"] == OBJECTSTATE_BOP_PREDICTION_BASELINE_HANDOFF_SCHEMA
    assert summary["status"] == "objectstate_bop_prediction_baseline_handoff_reviewable"
    assert validate_objectstate_bop_prediction_baseline_handoff_summary(summary) == summary
    assert summary["readiness"] == {
        "bop_acceptance_pass": True,
        "phase1_gaussian_evidence_ready": True,
        "template_ready": True,
        "baseline_candidates_ready": True,
        "prediction_eval_ready": True,
        "prediction_evidence_package_reviewable": True,
        "phase1_evidence_ledger_prediction_reviewable": True,
    }
    assert summary["row_counts"]["prediction_drafts"] == 4
    assert summary["row_counts"]["prediction_candidates"] == 4
    assert summary["prediction_eval_summary"]["status"] == (
        "objectstate_controlled_prediction_eval_pass"
    )
    assert summary["prediction_evidence_package"]["status"] == (
        "objectstate_controlled_prediction_evidence_package_reviewable"
    )
    assert (output_root / "capture-manifest.json").is_file()
    assert (output_root / "bop-acceptance-summary.json").is_file()
    assert (output_root / "bop-file-audit.json").is_file()
    assert (output_root / "bop-missing-files.md").is_file()
    assert (output_root / "reality-candidates" / "template-summary.json").is_file()
    assert (
        output_root
        / "reality-candidates"
        / "prediction-baseline-summary.json"
    ).is_file()
    assert (
        output_root
        / "reality-candidates"
        / "prediction-evidence-package-summary.json"
    ).is_file()
    phase1_ledger = _read_json(output_root / "phase1-evidence-ledger.json")
    assert phase1_ledger["maturity"] == "prediction_reviewable"
    assert (
        phase1_ledger["phase1_evidence_gates"]["prediction_evidence_reviewable"]
        is True
    )
    assert summary["phase1_evidence_ledger_summary"] == phase1_ledger


def test_bop_prediction_baseline_handoff_blocks_without_gaussian_files(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "prediction-package"
    _write_bop_scene(scene_root)

    summary = objectstate_bop_prediction_baseline_handoff(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
        candidate_id="fixture-bop-baseline",
        candidate_source="unit-test BOP baseline",
        artifact_ref="outputs/captures/bop-ycbv-scene-000001/objectstates.json",
    )

    assert summary["status"] == "objectstate_bop_prediction_baseline_handoff_incomplete"
    assert summary["readiness"]["bop_acceptance_pass"] is False
    assert summary["readiness"]["phase1_gaussian_evidence_ready"] is False
    assert summary["readiness"]["prediction_evidence_package_reviewable"] is False
    assert (
        summary["readiness"]["phase1_evidence_ledger_prediction_reviewable"]
        is False
    )
    assert "phase1 Gaussian evidence is not ready" in summary["issues"]
    assert (
        "phase1 evidence ledger does not expose reviewable prediction evidence"
        in summary["issues"]
    )
    assert (
        summary["prediction_evidence_package"]["reviewability_gates"][
            "phase1_gaussian_evidence_ready"
        ]
        is False
    )


def test_bop_prediction_baseline_handoff_relative_output_root_package_paths(
    tmp_path,
    monkeypatch,
):
    scene_root = tmp_path / "bop-scene"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    monkeypatch.chdir(tmp_path)

    summary = objectstate_bop_prediction_baseline_handoff(
        "bop-scene",
        output_root="prediction-package",
        sample_id="bop-ycbv-scene-000001",
        candidate_id="fixture-bop-baseline",
        candidate_source="unit-test BOP baseline",
        artifact_ref="prediction-package/objectstates.json",
    )

    package = summary["prediction_evidence_package"]
    assert summary["status"] == "objectstate_bop_prediction_baseline_handoff_reviewable"
    assert package["status"] == (
        "objectstate_controlled_prediction_evidence_package_reviewable"
    )
    assert package["candidate_dir"] == "prediction-package/reality-candidates"
    assert all(
        "prediction-package/prediction-package" not in record["path"]
        for record in package["files"]
    )


def test_object_state_bop_prediction_baseline_handoff_cli(
    tmp_path,
    capsys,
):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "prediction-package"
    summary_path = tmp_path / "bop-prediction-baseline-handoff-summary.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)

    assert (
        main(
            [
                "object-state",
                "bop-prediction-baseline-handoff",
                str(scene_root),
                "--output-root",
                str(output_root),
                "--sample-id",
                "bop-ycbv-scene-000001",
                "--candidate-id",
                "cli-bop-baseline",
                "--candidate-source",
                "cli BOP baseline",
                "--artifact-ref",
                "outputs/captures/bop-ycbv-scene-000001/objectstates.json",
                "--confidence",
                "0.8",
                "--summary-output",
                str(summary_path),
                "--require-reviewable",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = _read_json(summary_path)

    assert f"schema={OBJECTSTATE_BOP_PREDICTION_BASELINE_HANDOFF_SCHEMA}" in stdout
    assert "bop_prediction_baseline_handoff_status=objectstate_bop_prediction_baseline_handoff_reviewable" in stdout
    assert "phase1_gaussian_evidence_ready=true" in stdout
    assert "baseline_candidates_ready=true" in stdout
    assert "prediction_eval_status=objectstate_controlled_prediction_eval_pass" in stdout
    assert "prediction_evidence_package_reviewable=true" in stdout
    assert "phase1_evidence_ledger_prediction_reviewable=true" in stdout
    assert "prediction_candidate_count=4" in stdout
    assert "prediction_evidence_package_summary=" in stdout
    assert "phase1_evidence_ledger=" in stdout
    assert summary["candidate"]["candidate_id"] == "cli-bop-baseline"
    assert summary["status"] == "objectstate_bop_prediction_baseline_handoff_reviewable"


def _write_bop_scene(root) -> None:
    (root / "rgb").mkdir(parents=True)
    for frame_id in range(3):
        (root / "rgb" / f"{frame_id:06d}.png").write_bytes(PNG_BYTES)
    scene_camera = {
        str(frame_id): {
            "cam_K": [572.4, 0.0, 325.2, 0.0, 573.5, 242.0, 0.0, 0.0, 1.0],
            "depth_scale": 1.0,
        }
        for frame_id in range(3)
    }
    identity_rotation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    scene_gt = {}
    scene_gt_info = {}
    for frame_id in range(3):
        scene_gt[str(frame_id)] = [
            {
                "obj_id": 1,
                "cam_R_m2c": identity_rotation,
                "cam_t_m2c": [10.0 + frame_id, 20.0, 30.0],
            },
            {
                "obj_id": 2,
                "cam_R_m2c": identity_rotation,
                "cam_t_m2c": [40.0 + frame_id, 50.0, 60.0],
            },
        ]
        scene_gt_info[str(frame_id)] = [
            {"visib_fract": 1.0 - frame_id * 0.1},
            {"visib_fract": 1.0},
        ]
    _write_json(root / "scene_camera.json", scene_camera)
    _write_json(root / "scene_gt.json", scene_gt)
    _write_json(root / "scene_gt_info.json", scene_gt_info)


def _write_gaussian_frames(root) -> None:
    (root / "gaussians").mkdir()
    for frame_id in range(3):
        (root / "gaussians" / f"{frame_id:06d}.ply").write_bytes(PLY_BYTES)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
