from __future__ import annotations

import csv
import json

import pytest

from objgauss.cli import main
from objgauss.core.objectstate_controlled_capture_import import (
    objectstate_controlled_capture_manifest_from_bundle,
)
from objgauss.core.objectstate_controlled_capture_template import (
    OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_TEMPLATE_SCHEMA,
    write_objectstate_controlled_capture_bundle_template,
    validate_objectstate_controlled_capture_bundle_template_summary,
)


def test_controlled_capture_template_writes_headers_without_fake_rows(tmp_path):
    summary = write_objectstate_controlled_capture_bundle_template(
        tmp_path,
        sample_id="controlled-tabletop-cup-box-001",
    )

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_TEMPLATE_SCHEMA
    assert summary["status"] == "objectstate_controlled_capture_bundle_template_ready"
    assert summary["object_row_count"] == 0
    assert summary["sample"]["source_kind"] == "controlled_real"
    assert summary["sample"]["observation_modalities"] == ["rgb", "gaussian"]
    assert (tmp_path / "sample.json").is_file()
    assert (tmp_path / "objects.csv").is_file()
    assert (tmp_path / "frames.csv").is_file()
    assert (tmp_path / "annotations.csv").is_file()
    assert (tmp_path / "actions.csv").is_file()
    assert (tmp_path / "README.md").is_file()
    assert (tmp_path / "rgb").is_dir()
    assert (tmp_path / "gaussians").is_dir()

    assert _csv_rows(tmp_path / "objects.csv") == []
    assert _csv_rows(tmp_path / "frames.csv") == []
    assert _csv_rows(tmp_path / "annotations.csv") == []
    assert _csv_rows(tmp_path / "actions.csv") == []
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "controlled-identity-bundle-handoff" in readme
    assert "docs/training/controlled-real-capture-runbook.md" in readme
    assert "audit-controlled-capture-bundle-readiness" in readme
    assert "does not contain captured RGB frames" in readme
    assert "Do not copy large captures" in readme
    assert validate_objectstate_controlled_capture_bundle_template_summary(summary) == summary

    with pytest.raises(ValueError, match="objects.csv requires at least one object"):
        objectstate_controlled_capture_manifest_from_bundle(tmp_path)


def test_controlled_capture_template_writes_declared_objects_and_refuses_overwrite(
    tmp_path,
):
    summary = write_objectstate_controlled_capture_bundle_template(
        tmp_path,
        sample_id="controlled-tabletop-cup-box-001",
        object_category="cup_box",
        scenario="cross_view_occlusion_reappearance",
        fps=30.0,
        capture_device="fixture-camera",
        objects=[
            {
                "object_id": "cup-001",
                "category": "cup",
                "instance_label": "blue cup",
                "dimensions_m": [0.08, 0.08, 0.1],
            },
            {
                "object_id": "box-001",
                "category": "box",
                "instance_label": "red box",
            },
        ],
    )

    rows = _csv_rows(tmp_path / "objects.csv")
    assert summary["object_row_count"] == 2
    assert rows[0]["object_id"] == "cup-001"
    assert rows[0]["dimension_x_m"] == "0.08"
    assert rows[1]["object_id"] == "box-001"
    sample = json.loads((tmp_path / "sample.json").read_text(encoding="utf-8"))
    assert sample["object_category"] == "cup_box"
    assert sample["capture_device"] == "fixture-camera"

    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        write_objectstate_controlled_capture_bundle_template(
            tmp_path,
            sample_id="controlled-tabletop-cup-box-001",
        )

    forced = write_objectstate_controlled_capture_bundle_template(
        tmp_path,
        sample_id="controlled-tabletop-cup-box-001",
        objects=[
            {
                "object_id": "cup-001",
                "category": "cup",
            }
        ],
        force=True,
    )
    assert forced["object_row_count"] == 1
    assert len(_csv_rows(tmp_path / "objects.csv")) == 1


def test_object_state_init_controlled_capture_bundle_cli_writes_template(
    tmp_path,
    capsys,
):
    summary_path = tmp_path / "template-summary.json"

    assert (
        main(
            [
                "object-state",
                "init-controlled-capture-bundle",
                str(tmp_path / "capture"),
                "--sample-id",
                "controlled-tabletop-cup-box-001",
                "--object-category",
                "cup_box",
                "--capture-device",
                "fixture-camera",
                "--object",
                "cup-001:cup:blue cup",
                "--object",
                "box-001:box:red box",
                "--summary-output",
                str(summary_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sample = json.loads(
        (tmp_path / "capture" / "sample.json").read_text(encoding="utf-8")
    )
    rows = _csv_rows(tmp_path / "capture" / "objects.csv")

    assert f"schema={OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_TEMPLATE_SCHEMA}" in stdout
    assert "sample_id=controlled-tabletop-cup-box-001" in stdout
    assert "object_row_count=2" in stdout
    assert "identity_bundle_handoff_command=" in stdout
    assert summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_TEMPLATE_SCHEMA
    assert sample["object_category"] == "cup_box"
    assert sample["capture_device"] == "fixture-camera"
    assert rows[0]["object_id"] == "cup-001"
    assert rows[1]["instance_label"] == "red box"
    assert _csv_rows(tmp_path / "capture" / "frames.csv") == []


def _csv_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
