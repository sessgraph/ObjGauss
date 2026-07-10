from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.pipelines.objectstate_bop_gaussian_evidence_preflight import (
    OBJECTSTATE_BOP_GAUSSIAN_EVIDENCE_PREFLIGHT_SCHEMA,
    objectstate_bop_gaussian_evidence_preflight,
    validate_objectstate_bop_gaussian_evidence_preflight_summary,
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


def test_bop_gaussian_evidence_preflight_reports_missing_gaussians(tmp_path):
    _write_bop_scene(tmp_path)

    summary = objectstate_bop_gaussian_evidence_preflight(
        tmp_path,
        sample_id="bop-ycbv-scene-000001",
        command_resolver=_reconstruction_tool_resolver,
        importer=_missing_importer,
    )

    assert summary["schema"] == OBJECTSTATE_BOP_GAUSSIAN_EVIDENCE_PREFLIGHT_SCHEMA
    assert summary["status"] == "objectstate_bop_gaussian_evidence_blocked"
    assert summary["readiness"]["bop_acceptance_available"] is True
    assert summary["readiness"]["rgb_files_present"] is True
    assert summary["readiness"]["gaussian_refs_expected"] is True
    assert summary["readiness"]["gaussian_files_present"] is False
    assert summary["readiness"]["phase1_gaussian_evidence_ready"] is False
    assert summary["readiness"]["gaussian_reconstruction_tools_ready"] is True
    assert summary["row_counts"] == {
        "selected_frames": 3,
        "expected_gaussian_files": 3,
        "missing_gaussian_files": 3,
    }
    assert summary["missing_gaussian_files"][0]["ref"] == "gaussians/000000.ply"
    assert "selected frame Gaussian files are missing" in " ".join(
        summary["hard_blockers"]
    )
    assert (
        validate_objectstate_bop_gaussian_evidence_preflight_summary(summary)
        == summary
    )


def test_bop_gaussian_evidence_preflight_passes_with_gaussian_files(tmp_path):
    _write_bop_scene(tmp_path)
    _write_gaussian_frames(tmp_path)

    summary = objectstate_bop_gaussian_evidence_preflight(
        tmp_path,
        sample_id="bop-ycbv-scene-000001",
        command_resolver=_missing_command_resolver,
        importer=_missing_importer,
    )

    assert summary["status"] == "objectstate_bop_gaussian_evidence_ready"
    assert summary["readiness"]["phase1_gaussian_evidence_ready"] is True
    assert summary["readiness"]["gaussian_reconstruction_tools_ready"] is False
    assert summary["row_counts"]["missing_gaussian_files"] == 0
    assert "no missing files" in summary["missing_gaussian_files_markdown"]
    assert summary["hard_blockers"] == []


def test_bop_gaussian_evidence_preflight_cli_writes_outputs(tmp_path, capsys):
    _write_bop_scene(tmp_path)
    summary_path = tmp_path / "bop-gaussian-evidence-summary.json"
    missing_path = tmp_path / "bop-missing-gaussians.md"

    assert (
        main(
            [
                "object-state",
                "audit-bop-gaussian-evidence",
                str(tmp_path),
                "--sample-id",
                "bop-ycbv-scene-000001",
                "--summary-output",
                str(summary_path),
                "--missing-gaussians-output",
                str(missing_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    missing_markdown = missing_path.read_text(encoding="utf-8")

    assert f"schema={OBJECTSTATE_BOP_GAUSSIAN_EVIDENCE_PREFLIGHT_SCHEMA}" in stdout
    assert (
        "bop_gaussian_evidence_status="
        "objectstate_bop_gaussian_evidence_blocked"
    ) in stdout
    assert "missing_gaussian_files=3" in stdout
    assert f"missing_gaussians_markdown={missing_path}" in stdout
    assert summary["row_counts"]["missing_gaussian_files"] == 3
    assert "gaussians/000000.ply" in missing_markdown


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


def _reconstruction_tool_resolver(command: str) -> str | None:
    if command in {"colmap", "ns-train", "ns-export"}:
        return f"/usr/bin/{command}"
    return None


def _missing_command_resolver(command: str) -> str | None:
    return None


def _missing_importer(name: str):
    raise ImportError(name)
