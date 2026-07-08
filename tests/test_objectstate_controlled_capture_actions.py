from __future__ import annotations

import csv
import json

from objgauss.cli import main
from objgauss.core.objectstate_controlled_capture_actions import (
    OBJECTSTATE_CONTROLLED_CAPTURE_ACTION_FINALIZE_SCHEMA,
    OBJECTSTATE_CONTROLLED_CAPTURE_ACTION_TEMPLATE_SCHEMA,
    finalize_objectstate_controlled_capture_actions,
    validate_objectstate_controlled_capture_action_finalize_summary,
    validate_objectstate_controlled_capture_action_template_summary,
    write_objectstate_controlled_capture_action_template,
)
from objgauss.core.objectstate_controlled_capture_annotations import (
    finalize_objectstate_controlled_capture_annotations,
    write_objectstate_controlled_capture_annotation_template,
)
from objgauss.core.objectstate_controlled_capture_import import (
    objectstate_controlled_capture_manifest_from_bundle,
)
from objgauss.core.objectstate_controlled_capture_template import (
    ACTIONS_CSV_HEADER,
    write_objectstate_controlled_capture_bundle_template,
)
from objgauss.core.objectstate_controlled_capture_frames import (
    write_objectstate_controlled_capture_frames,
)


def test_action_template_writes_draft_rows_only(tmp_path):
    _write_bundle_with_frames(tmp_path, frame_count=2, object_count=2)

    summary = write_objectstate_controlled_capture_action_template(tmp_path)

    rows = _csv_rows(tmp_path / "actions.template.csv")
    assert summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_ACTION_TEMPLATE_SCHEMA
    assert summary["status"] == "objectstate_controlled_capture_action_template_ready"
    assert summary["row_counts"]["template_action_rows"] == 2
    assert summary["template_policy"]["template_status"] == "draft_not_valid_for_import"
    assert rows[0]["action_id"].startswith("TODO")
    assert rows[0]["object_id"] == "object-000"
    assert rows[0]["vector_x"].startswith("TODO")
    assert _csv_rows(tmp_path / "actions.csv") == []
    assert validate_objectstate_controlled_capture_action_template_summary(summary) == summary


def test_finalize_actions_rejects_todo_values_without_writing(tmp_path):
    _write_bundle_with_frames(tmp_path, frame_count=2, object_count=1)
    write_objectstate_controlled_capture_action_template(tmp_path)

    summary = finalize_objectstate_controlled_capture_actions(tmp_path)

    assert summary["status"] == "objectstate_controlled_capture_action_finalize_blocked"
    assert summary["output"]["wrote_actions_csv"] is False
    assert any("TODO" in issue for issue in summary["issues"])
    assert _csv_rows(tmp_path / "actions.csv") == []


def test_finalize_actions_writes_importable_action_rows(tmp_path):
    _write_bundle_with_frames(tmp_path, frame_count=3, object_count=1)
    _write_importable_annotations(tmp_path)
    write_objectstate_controlled_capture_action_template(tmp_path)
    _fill_action_template(tmp_path / "actions.template.csv")
    _set_frame_action_id(tmp_path / "frames.csv", "frame-000000", "push-left-001")

    summary = finalize_objectstate_controlled_capture_actions(
        tmp_path,
        require_frame_action_refs=True,
    )

    rows = _csv_rows(tmp_path / "actions.csv")
    manifest = objectstate_controlled_capture_manifest_from_bundle(tmp_path)
    assert summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_ACTION_FINALIZE_SCHEMA
    assert summary["status"] == "objectstate_controlled_capture_action_finalize_ready"
    assert summary["row_counts"]["finalized_action_rows"] == 1
    assert summary["row_counts"]["covered_frame_count"] == 2
    assert summary["row_counts"]["frame_action_ref_count"] == 1
    assert summary["issues"] == []
    assert rows[0]["action_id"] == "push-left-001"
    assert rows[0]["vector_x"] == "-0.02"
    assert manifest["actions"][0]["vector"] == [-0.02, 0.0, 0.0]
    assert manifest["actions"][0]["object_id"] == "object-000"
    assert validate_objectstate_controlled_capture_action_finalize_summary(summary) == summary


def test_finalize_actions_rejects_zero_vector(tmp_path):
    _write_bundle_with_frames(tmp_path, frame_count=2, object_count=1)
    write_objectstate_controlled_capture_action_template(tmp_path)
    _fill_action_template(tmp_path / "actions.template.csv", vector=("0", "0", "0"))

    summary = finalize_objectstate_controlled_capture_actions(tmp_path)

    assert summary["status"] == "objectstate_controlled_capture_action_finalize_blocked"
    assert any("action vector must be non-zero" in issue for issue in summary["issues"])
    assert _csv_rows(tmp_path / "actions.csv") == []


def test_finalize_actions_can_require_frame_action_refs(tmp_path):
    _write_bundle_with_frames(tmp_path, frame_count=2, object_count=1)
    write_objectstate_controlled_capture_action_template(tmp_path)
    _fill_action_template(tmp_path / "actions.template.csv")

    summary = finalize_objectstate_controlled_capture_actions(
        tmp_path,
        require_frame_action_refs=True,
    )

    assert summary["status"] == "objectstate_controlled_capture_action_finalize_blocked"
    assert any("not referenced by frames.csv" in issue for issue in summary["issues"])
    assert _csv_rows(tmp_path / "actions.csv") == []


def test_controlled_capture_action_cli_template_and_finalize(tmp_path, capsys):
    _write_bundle_with_frames(tmp_path, frame_count=2, object_count=1)
    template_summary_path = tmp_path / "action-template-summary.json"
    finalize_summary_path = tmp_path / "action-finalize-summary.json"

    assert (
        main(
            [
                "object-state",
                "init-controlled-capture-actions",
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
    assert f"schema={OBJECTSTATE_CONTROLLED_CAPTURE_ACTION_TEMPLATE_SCHEMA}" in stdout
    assert "action_template_ready=true" in stdout
    assert template_summary["status"] == "objectstate_controlled_capture_action_template_ready"

    _fill_action_template(tmp_path / "actions.template.csv")
    assert (
        main(
            [
                "object-state",
                "finalize-controlled-capture-actions",
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
    assert f"schema={OBJECTSTATE_CONTROLLED_CAPTURE_ACTION_FINALIZE_SCHEMA}" in stdout
    assert "actions_ready=true" in stdout
    assert "finalized_action_rows=1" in stdout
    assert finalize_summary["status"] == "objectstate_controlled_capture_action_finalize_ready"


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
    write_objectstate_controlled_capture_frames(root, fps=2.0)


def _write_importable_annotations(root) -> None:
    write_objectstate_controlled_capture_annotation_template(root)
    _write_csv(
        root / "annotations.template.csv",
        _filled_annotation_rows(_csv_rows(root / "annotations.template.csv")),
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
    finalize_objectstate_controlled_capture_annotations(root)


def _fill_action_template(path, *, vector=("-0.02", "0.0", "0.0")) -> None:
    rows = []
    for row in _csv_rows(path):
        item = dict(row)
        item.update(
            {
                "action_id": "push-left-001",
                "action_type": "push_left",
                "object_id": "object-000",
                "start_timestamp": "0.0",
                "end_timestamp": "0.5",
                "actor": "human",
                "target_object_id": "",
                "vector_x": vector[0],
                "vector_y": vector[1],
                "vector_z": vector[2],
            }
        )
        rows.append(item)
    _write_csv(path, rows, fieldnames=ACTIONS_CSV_HEADER)


def _filled_annotation_rows(rows):
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


def _set_frame_action_id(path, frame_id: str, action_id: str) -> None:
    rows = _csv_rows(path)
    for row in rows:
        if row["frame_id"] == frame_id:
            row["action_id"] = action_id
    _write_csv(
        path,
        rows,
        fieldnames=(
            "frame_id",
            "timestamp",
            "rgb",
            "gaussian",
            "action_id",
            "view_id",
            "lighting_id",
            "camera_x",
            "camera_y",
            "camera_z",
            "camera_qx",
            "camera_qy",
            "camera_qz",
            "camera_qw",
        ),
    )


def _write_rgb(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")


def _write_ply(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="utf-8")


def _csv_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, *, fieldnames) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
