from __future__ import annotations

import csv
import json

from objgauss.cli import main
from objgauss.core.objectstate_controlled_capture_bundle_readiness import (
    objectstate_controlled_capture_bundle_readiness,
)
from objgauss.core.objectstate_controlled_capture_frames import (
    OBJECTSTATE_CONTROLLED_CAPTURE_FRAMES_SCHEMA,
    validate_objectstate_controlled_capture_frames_summary,
    write_objectstate_controlled_capture_frames,
)
from objgauss.core.objectstate_controlled_capture_template import (
    write_objectstate_controlled_capture_bundle_template,
)


def test_populate_controlled_capture_frames_writes_timestamped_rows(tmp_path):
    _write_template_with_files(tmp_path, count=3)

    summary = write_objectstate_controlled_capture_frames(
        tmp_path,
        fps=30.0,
        view_id="front",
        lighting_id="bright",
        camera_pose=[0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0],
    )

    rows = _csv_rows(tmp_path / "frames.csv")
    assert summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_FRAMES_SCHEMA
    assert summary["status"] == "objectstate_controlled_capture_frames_ready"
    assert summary["scan"]["frame_row_count"] == 3
    assert summary["scan"]["paired_frame_count"] == 3
    assert summary["output"]["wrote_frames_csv"] is True
    assert summary["issues"] == []
    assert rows[0]["frame_id"] == "frame-000000"
    assert rows[0]["timestamp"] == "0.000000"
    assert rows[0]["rgb"] == "rgb/000000.png"
    assert rows[0]["gaussian"] == "gaussians/000000.ply"
    assert rows[0]["view_id"] == "front"
    assert rows[0]["lighting_id"] == "bright"
    assert rows[0]["camera_qw"] == "1.000000"
    assert rows[1]["timestamp"] == "0.033333"
    assert validate_objectstate_controlled_capture_frames_summary(summary) == summary

    readiness = objectstate_controlled_capture_bundle_readiness(tmp_path)
    assert readiness["readiness"]["frame_rows_present"] is True
    assert readiness["readiness"]["annotation_rows_present"] is False
    assert readiness["readiness"]["capture_bundle_ready"] is False


def test_populate_controlled_capture_frames_blocks_missing_gaussian_pair(tmp_path):
    write_objectstate_controlled_capture_bundle_template(
        tmp_path,
        sample_id="controlled-tabletop-cup-box-001",
        objects=[{"object_id": "cup-001", "category": "cup"}],
    )
    _write_rgb(tmp_path / "rgb" / "000000.png")

    summary = write_objectstate_controlled_capture_frames(tmp_path)

    assert summary["status"] == "objectstate_controlled_capture_frames_blocked"
    assert summary["scan"]["rgb_file_count"] == 1
    assert summary["scan"]["missing_gaussian_count"] == 1
    assert summary["output"]["wrote_frames_csv"] is False
    assert any("missing Gaussian files" in issue for issue in summary["issues"])
    assert _csv_rows(tmp_path / "frames.csv") == []


def test_populate_controlled_capture_frames_cli(tmp_path, capsys):
    _write_template_with_files(tmp_path, count=2)
    summary_path = tmp_path / "frames-summary.json"

    assert (
        main(
            [
                "object-state",
                "populate-controlled-capture-frames",
                str(tmp_path),
                "--fps",
                "2",
                "--view-id",
                "side",
                "--lighting-id",
                "dim",
                "--summary-output",
                str(summary_path),
                "--require-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _csv_rows(tmp_path / "frames.csv")
    assert f"schema={OBJECTSTATE_CONTROLLED_CAPTURE_FRAMES_SCHEMA}" in stdout
    assert "frame_rows_ready=true" in stdout
    assert "frame_row_count=2" in stdout
    assert "paired_frame_count=2" in stdout
    assert summary["status"] == "objectstate_controlled_capture_frames_ready"
    assert rows[1]["timestamp"] == "0.500000"
    assert rows[1]["view_id"] == "side"


def test_populate_controlled_capture_frames_refuses_non_empty_frames_without_force(
    tmp_path,
):
    _write_template_with_files(tmp_path, count=1)
    write_objectstate_controlled_capture_frames(tmp_path)

    summary = write_objectstate_controlled_capture_frames(tmp_path)

    assert summary["status"] == "objectstate_controlled_capture_frames_blocked"
    assert summary["output"]["existing_row_count"] == 1
    assert summary["readiness"]["output_writable"] is False
    assert summary["output"]["wrote_frames_csv"] is False

    forced = write_objectstate_controlled_capture_frames(
        tmp_path,
        start_timestamp=1.0,
        force=True,
    )
    rows = _csv_rows(tmp_path / "frames.csv")
    assert forced["status"] == "objectstate_controlled_capture_frames_ready"
    assert rows[0]["timestamp"] == "1.000000"


def _write_template_with_files(root, *, count: int) -> None:
    write_objectstate_controlled_capture_bundle_template(
        root,
        sample_id="controlled-tabletop-cup-box-001",
        objects=[{"object_id": "cup-001", "category": "cup"}],
    )
    for index in range(count):
        stem = f"{index:06d}"
        _write_rgb(root / "rgb" / f"{stem}.png")
        _write_ply(root / "gaussians" / f"{stem}.ply")


def _write_rgb(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")


def _write_ply(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")


def _csv_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
