from __future__ import annotations

import csv
import json

from objgauss.cli import main
from objgauss.core.objectstate_controlled_capture_annotations import (
    OBJECTSTATE_CONTROLLED_CAPTURE_ANNOTATION_FINALIZE_SCHEMA,
    OBJECTSTATE_CONTROLLED_CAPTURE_ANNOTATION_TEMPLATE_SCHEMA,
    finalize_objectstate_controlled_capture_annotations,
    validate_objectstate_controlled_capture_annotation_finalize_summary,
    validate_objectstate_controlled_capture_annotation_template_summary,
    write_objectstate_controlled_capture_annotation_template,
)
from objgauss.core.objectstate_controlled_capture_bundle_readiness import (
    objectstate_controlled_capture_bundle_readiness,
)
from objgauss.core.objectstate_controlled_capture_frames import (
    write_objectstate_controlled_capture_frames,
)
from objgauss.core.objectstate_controlled_capture_import import (
    objectstate_controlled_capture_manifest_from_bundle,
)
from objgauss.core.objectstate_controlled_capture_template import (
    write_objectstate_controlled_capture_bundle_template,
)


def test_annotation_template_writes_draft_rows_only(tmp_path):
    _write_bundle_with_frames(tmp_path, frame_count=2, object_count=2)

    summary = write_objectstate_controlled_capture_annotation_template(tmp_path)

    rows = _csv_rows(tmp_path / "annotations.template.csv")
    assert summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_ANNOTATION_TEMPLATE_SCHEMA
    assert summary["status"] == "objectstate_controlled_capture_annotation_template_ready"
    assert summary["row_counts"]["template_annotation_rows"] == 4
    assert summary["template_policy"]["template_status"] == "draft_not_valid_for_import"
    assert rows[0]["frame_id"] == "frame-000000"
    assert rows[0]["object_id"] == "object-000"
    assert rows[0]["visible"].startswith("TODO")
    assert _csv_rows(tmp_path / "annotations.csv") == []
    readiness = objectstate_controlled_capture_bundle_readiness(tmp_path)
    assert readiness["readiness"]["annotation_rows_present"] is False
    assert validate_objectstate_controlled_capture_annotation_template_summary(summary) == summary


def test_finalize_annotations_rejects_todo_values_without_writing(tmp_path):
    _write_bundle_with_frames(tmp_path, frame_count=1, object_count=1)
    write_objectstate_controlled_capture_annotation_template(tmp_path)

    summary = finalize_objectstate_controlled_capture_annotations(tmp_path)

    assert summary["status"] == "objectstate_controlled_capture_annotation_finalize_blocked"
    assert summary["output"]["wrote_annotations_csv"] is False
    assert any("TODO" in issue for issue in summary["issues"])
    assert _csv_rows(tmp_path / "annotations.csv") == []


def test_finalize_annotations_writes_importable_pose_rows(tmp_path):
    _write_bundle_with_frames(tmp_path, frame_count=2, object_count=2)
    write_objectstate_controlled_capture_annotation_template(tmp_path)
    _fill_annotation_template(tmp_path / "annotations.template.csv")

    summary = finalize_objectstate_controlled_capture_annotations(tmp_path)

    rows = _csv_rows(tmp_path / "annotations.csv")
    manifest = objectstate_controlled_capture_manifest_from_bundle(tmp_path)
    readiness = objectstate_controlled_capture_bundle_readiness(tmp_path)
    assert summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_ANNOTATION_FINALIZE_SCHEMA
    assert summary["status"] == "objectstate_controlled_capture_annotation_finalize_ready"
    assert summary["row_counts"]["finalized_annotation_rows"] == 4
    assert summary["issues"] == []
    assert len(rows) == 4
    assert rows[0]["visible"] == "true"
    assert rows[0]["qw"] == "1.0"
    assert len(manifest["frames"]) == 2
    assert len(manifest["frames"][0]["objects"]) == 2
    assert "pose" in manifest["frames"][0]["objects"][0]
    assert readiness["readiness"]["annotation_rows_present"] is True
    assert readiness["readiness"]["frame_annotation_integrity_ready"] is True
    assert validate_objectstate_controlled_capture_annotation_finalize_summary(summary) == summary


def test_controlled_capture_annotation_cli_template_and_finalize(tmp_path, capsys):
    _write_bundle_with_frames(tmp_path, frame_count=2, object_count=1)
    template_summary_path = tmp_path / "annotation-template-summary.json"
    finalize_summary_path = tmp_path / "annotation-finalize-summary.json"

    assert (
        main(
            [
                "object-state",
                "init-controlled-capture-annotations",
                str(tmp_path),
                "--summary-output",
                str(template_summary_path),
                "--require-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    template_summary = json.loads(template_summary_path.read_text(encoding="utf-8"))
    assert f"schema={OBJECTSTATE_CONTROLLED_CAPTURE_ANNOTATION_TEMPLATE_SCHEMA}" in stdout
    assert "annotation_template_ready=true" in stdout
    assert template_summary["status"] == (
        "objectstate_controlled_capture_annotation_template_ready"
    )

    _fill_annotation_template(tmp_path / "annotations.template.csv")
    assert (
        main(
            [
                "object-state",
                "finalize-controlled-capture-annotations",
                str(tmp_path),
                "--summary-output",
                str(finalize_summary_path),
                "--require-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    finalize_summary = json.loads(finalize_summary_path.read_text(encoding="utf-8"))
    assert f"schema={OBJECTSTATE_CONTROLLED_CAPTURE_ANNOTATION_FINALIZE_SCHEMA}" in stdout
    assert "annotations_ready=true" in stdout
    assert "finalized_annotation_rows=2" in stdout
    assert finalize_summary["status"] == (
        "objectstate_controlled_capture_annotation_finalize_ready"
    )


def test_finalize_annotations_requires_full_frame_object_coverage(tmp_path):
    _write_bundle_with_frames(tmp_path, frame_count=2, object_count=2)
    write_objectstate_controlled_capture_annotation_template(tmp_path)
    rows = _filled_rows(_csv_rows(tmp_path / "annotations.template.csv"))[:-1]
    _write_csv(tmp_path / "annotations.template.csv", rows)

    summary = finalize_objectstate_controlled_capture_annotations(tmp_path)

    assert summary["status"] == "objectstate_controlled_capture_annotation_finalize_blocked"
    assert any("missing frame/object annotation rows" in issue for issue in summary["issues"])
    assert _csv_rows(tmp_path / "annotations.csv") == []


def _write_bundle_with_frames(root, *, frame_count: int, object_count: int) -> None:
    write_objectstate_controlled_capture_bundle_template(
        root,
        sample_id="controlled-tabletop-cup-box-001",
        objects=[
            {
                "object_id": f"object-{index:03d}",
                "category": "test_object",
            }
            for index in range(object_count)
        ],
    )
    for index in range(frame_count):
        stem = f"{index:06d}"
        _write_rgb(root / "rgb" / f"{stem}.png")
        _write_ply(root / "gaussians" / f"{stem}.ply")
    write_objectstate_controlled_capture_frames(root)


def _fill_annotation_template(path) -> None:
    _write_csv(path, _filled_rows(_csv_rows(path)))


def _filled_rows(rows):
    filled = []
    for index, row in enumerate(rows):
        item = dict(row)
        item.update(
            {
                "visible": "true",
                "occlusion_fraction": "0.0",
                "x": f"{0.1 + index * 0.01:.3f}",
                "y": "0.2",
                "z": "0.3",
                "qx": "0.0",
                "qy": "0.0",
                "qz": "0.0",
                "qw": "1.0",
            }
        )
        filled.append(item)
    return filled


def _write_rgb(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")


def _write_ply(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="utf-8")


def _csv_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "frame_id",
                "object_id",
                "visible",
                "occlusion_fraction",
                "x",
                "y",
                "z",
                "qx",
                "qy",
                "qz",
                "qw",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
