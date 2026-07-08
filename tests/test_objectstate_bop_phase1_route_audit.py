from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.core.objectstate_bop_phase1_route_audit import (
    OBJECTSTATE_BOP_PHASE1_ROUTE_AUDIT_SCHEMA,
    objectstate_bop_phase1_route_audit,
    validate_objectstate_bop_phase1_route_audit_summary,
)
from objgauss.core.objectstate_bop_prediction_baseline_handoff import (
    objectstate_bop_prediction_baseline_handoff,
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


def test_bop_phase1_route_audit_reports_missing_scene(tmp_path):
    summary = objectstate_bop_phase1_route_audit(
        tmp_path / "missing-scene",
        output_root=tmp_path / "prediction-package",
        sample_id="bop-ycbv-scene-000001",
    )

    assert summary["schema"] == OBJECTSTATE_BOP_PHASE1_ROUTE_AUDIT_SCHEMA
    assert summary["status"] == "objectstate_bop_phase1_route_audit_blocked"
    assert summary["readiness"]["bop_acceptance_available"] is False
    assert summary["readiness"]["route_ready_for_prediction_handoff"] is False
    assert any("local BOP scene cannot be accepted" in item for item in summary["hard_blockers"])
    assert any("place a local BOP scene" in item for item in summary["next_actions"])
    assert validate_objectstate_bop_phase1_route_audit_summary(summary) == summary


def test_bop_phase1_route_audit_reports_handoff_ready(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "prediction-package"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)

    summary = objectstate_bop_phase1_route_audit(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
    )

    assert summary["status"] == "objectstate_bop_phase1_route_audit_handoff_ready"
    assert summary["readiness"]["bop_acceptance_pass"] is True
    assert summary["readiness"]["phase1_gaussian_evidence_ready"] is True
    assert summary["readiness"]["route_ready_for_prediction_handoff"] is True
    assert summary["readiness"]["prediction_evidence_package_present"] is False
    assert any("bop-prediction-baseline-handoff" in item for item in summary["next_actions"])


def test_bop_phase1_route_audit_reports_prediction_reviewable(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "prediction-package"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    objectstate_bop_prediction_baseline_handoff(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
        candidate_id="fixture-bop-baseline",
        candidate_source="unit-test BOP baseline",
        artifact_ref="outputs/captures/bop-ycbv-scene-000001/objectstates.json",
    )

    summary = objectstate_bop_phase1_route_audit(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
    )

    assert (
        summary["status"]
        == "objectstate_bop_phase1_route_audit_prediction_reviewable"
    )
    assert summary["readiness"]["prediction_evidence_package_reviewable"] is True
    assert summary["readiness"]["phase1_evidence_ledger_prediction_reviewable"] is True
    assert summary["readiness"]["route_has_reviewable_prediction_evidence"] is True
    assert (
        summary["records"]["phase1_evidence_ledger"]["payload"]["maturity"]
        == "prediction_reviewable"
    )
    assert any("identity and intervention gates remain unproven" in item for item in summary["hard_blockers"])


def test_object_state_audit_bop_phase1_route_cli(tmp_path, capsys):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "prediction-package"
    summary_path = tmp_path / "bop-phase1-route-summary.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)

    assert (
        main(
            [
                "object-state",
                "audit-bop-phase1-route",
                str(scene_root),
                "--output-root",
                str(output_root),
                "--sample-id",
                "bop-ycbv-scene-000001",
                "--summary-output",
                str(summary_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_BOP_PHASE1_ROUTE_AUDIT_SCHEMA}" in stdout
    assert "bop_phase1_route_status=objectstate_bop_phase1_route_audit_handoff_ready" in stdout
    assert "readiness.route_ready_for_prediction_handoff=true" in stdout
    assert "readiness.route_has_reviewable_prediction_evidence=false" in stdout
    assert "next_action=" in stdout
    assert summary["status"] == "objectstate_bop_phase1_route_audit_handoff_ready"


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


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
