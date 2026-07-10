from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.datasets.objectstate_bop_capture_adapter import (
    objectstate_bop_capture_acceptance_summary,
)
from objgauss.evaluation.objectstate_controlled_prediction_eval import (
    evaluate_objectstate_controlled_prediction_candidates,
)
from objgauss.pipelines.objectstate_controlled_prediction_evidence_package import (
    OBJECTSTATE_CONTROLLED_PREDICTION_EVIDENCE_PACKAGE_SCHEMA,
    objectstate_controlled_prediction_evidence_package,
    validate_objectstate_controlled_prediction_evidence_package_summary,
)
from objgauss.pipelines.objectstate_controlled_prediction_baseline import (
    write_objectstate_controlled_prediction_baseline_candidates,
)
from objgauss.pipelines.objectstate_controlled_reality_candidate_template import (
    write_objectstate_controlled_reality_candidate_templates_from_manifest,
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


def test_controlled_prediction_evidence_package_is_reviewable(tmp_path):
    _write_reviewable_prediction_package(tmp_path)

    summary = objectstate_controlled_prediction_evidence_package(tmp_path)

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_PREDICTION_EVIDENCE_PACKAGE_SCHEMA
    assert (
        summary["status"]
        == "objectstate_controlled_prediction_evidence_package_reviewable"
    )
    assert summary["sample_id"] == "bop-ycbv-scene-000001"
    assert summary["acceptance"]["phase1_gaussian_evidence_ready"] is True
    assert summary["prediction"]["prediction_candidate_count"] == 4
    assert summary["prediction"]["prediction_row_status"] == "pass"
    assert all(summary["reviewability_gates"].values())
    assert summary["issues"] == []
    assert (
        validate_objectstate_controlled_prediction_evidence_package_summary(summary)
        == summary
    )


def test_controlled_prediction_evidence_package_reports_missing_eval(tmp_path):
    paths = _write_reviewable_prediction_package(tmp_path)
    (paths["candidate_dir"] / "prediction-eval-summary.json").unlink()

    summary = objectstate_controlled_prediction_evidence_package(tmp_path)

    assert (
        summary["status"]
        == "objectstate_controlled_prediction_evidence_package_incomplete"
    )
    assert summary["reviewability_gates"]["required_files_present"] is False
    assert summary["reviewability_gates"]["prediction_eval_present"] is False
    assert any("prediction_eval_summary" in issue for issue in summary["issues"])


def test_object_state_audit_controlled_prediction_evidence_package_cli(
    tmp_path,
    capsys,
):
    _write_reviewable_prediction_package(tmp_path)
    summary_path = tmp_path / "prediction-evidence-package-summary.json"

    assert (
        main(
            [
                "object-state",
                "audit-controlled-prediction-evidence-package",
                str(tmp_path),
                "--summary-output",
                str(summary_path),
                "--require-reviewable",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_CONTROLLED_PREDICTION_EVIDENCE_PACKAGE_SCHEMA}" in stdout
    assert "reviewable=true" in stdout
    assert "phase1_gaussian_evidence_ready=true" in stdout
    assert "prediction_eval_status=objectstate_controlled_prediction_eval_pass" in stdout
    assert "prediction_candidate_count=4" in stdout
    assert "prediction_row_status=pass" in stdout
    assert "issue_count=0" in stdout
    assert (
        summary["status"]
        == "objectstate_controlled_prediction_evidence_package_reviewable"
    )


def _write_reviewable_prediction_package(root):
    _write_bop_scene(root)
    _write_gaussian_frames(root)
    capture_manifest_path = root / "capture-manifest.json"
    candidate_dir = root / "reality-candidates"

    acceptance = objectstate_bop_capture_acceptance_summary(
        root,
        sample_id="bop-ycbv-scene-000001",
        require_gaussian_files=True,
        hash_files=True,
    )
    _write_json(capture_manifest_path, acceptance["manifest"])
    _write_json(root / "bop-acceptance-summary.json", acceptance)
    _write_json(root / "bop-file-audit.json", acceptance["file_audit"])
    (root / "bop-missing-files.md").write_text(
        acceptance["file_audit"]["missing_files_markdown"],
        encoding="utf-8",
    )

    template_summary = write_objectstate_controlled_reality_candidate_templates_from_manifest(
        capture_manifest_path,
        output_dir=candidate_dir,
        candidate_id="bop-ycbv-predictor-v0",
        candidate_source="unit-test BOP predictor",
        artifact_ref="outputs/captures/bop-ycbv-scene-000001/objectstates.json",
    )
    _write_json(candidate_dir / "template-summary.json", template_summary)
    prediction_template_path = candidate_dir / "prediction-candidates.template.json"
    baseline_summary = write_objectstate_controlled_prediction_baseline_candidates(
        capture_manifest_path,
        prediction_template_path,
        output_dir=candidate_dir,
        policy="constant_velocity",
        candidate_id="bop-ycbv-predictor-v0",
        candidate_source="unit-test BOP constant-velocity baseline",
        artifact_ref="outputs/captures/bop-ycbv-scene-000001/objectstates.json",
        confidence=0.95,
    )
    _write_json(candidate_dir / "prediction-baseline-summary.json", baseline_summary)
    finalize_summary = baseline_summary["prediction_finalize_summary"]
    _write_json(candidate_dir / "prediction-finalize-summary.json", finalize_summary)
    prediction_candidates = _read_json(candidate_dir / "prediction-candidates.json")
    prediction_eval = evaluate_objectstate_controlled_prediction_candidates(
        acceptance["manifest"],
        prediction_candidates,
    )
    _write_json(candidate_dir / "prediction-eval-summary.json", prediction_eval)
    _write_json(
        candidate_dir / "controlled-real-prediction.json",
        prediction_eval["controlled_real_manifest"],
    )
    return {"candidate_dir": candidate_dir}


def _write_bop_scene(root) -> None:
    (root / "rgb").mkdir()
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
