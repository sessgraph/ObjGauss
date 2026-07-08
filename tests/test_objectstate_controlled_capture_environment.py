from __future__ import annotations

import json
from types import SimpleNamespace

from objgauss.cli import main
from objgauss.core.objectstate_controlled_capture_environment import (
    OBJECTSTATE_CONTROLLED_CAPTURE_ENVIRONMENT_SCHEMA,
    objectstate_controlled_capture_environment,
    validate_objectstate_controlled_capture_environment_summary,
)


def test_controlled_capture_environment_reports_blockers(tmp_path):
    dev_root = tmp_path / "dev"
    dev_root.mkdir()

    summary = objectstate_controlled_capture_environment(
        dev_root=dev_root,
        command_resolver=lambda _command: None,
        importer=_missing_importer,
    )

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_ENVIRONMENT_SCHEMA
    assert summary["status"] == "objectstate_controlled_capture_environment_blocked"
    assert summary["readiness"]["video_device_visible"] is False
    assert summary["readiness"]["rgb_capture_ready"] is False
    assert summary["readiness"]["gaussian_reconstruction_ready"] is False
    assert "no video device visible under dev_root" in summary["hard_blockers"]
    assert any("physical capture host" in action for action in summary["next_actions"])
    assert validate_objectstate_controlled_capture_environment_summary(summary) == summary


def test_controlled_capture_environment_reports_ready_host(tmp_path):
    dev_root = tmp_path / "dev"
    dev_root.mkdir()
    (dev_root / "video0").write_text("", encoding="utf-8")
    (dev_root / "media0").write_text("", encoding="utf-8")

    summary = objectstate_controlled_capture_environment(
        dev_root=dev_root,
        command_resolver=_fake_command_resolver,
        importer=_fake_importer,
    )

    assert summary["status"] == "objectstate_controlled_capture_environment_ready"
    assert summary["devices"]["video_devices"] == [str(dev_root / "video0")]
    assert summary["commands"]["ffmpeg"]["available"] is True
    assert summary["commands"]["ns_train"]["available"] is True
    assert summary["python_modules"]["cv2"]["available"] is True
    assert summary["readiness"]["rgb_capture_ready"] is True
    assert summary["readiness"]["gaussian_reconstruction_ready"] is True
    assert summary["readiness"]["controlled_capture_environment_ready"] is True
    assert summary["hard_blockers"] == []


def test_object_state_audit_controlled_capture_environment_cli(tmp_path, capsys):
    dev_root = tmp_path / "dev"
    dev_root.mkdir()
    summary_path = tmp_path / "environment-summary.json"

    assert (
        main(
            [
                "object-state",
                "audit-controlled-capture-environment",
                "--dev-root",
                str(dev_root),
                "--summary-output",
                str(summary_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_CONTROLLED_CAPTURE_ENVIRONMENT_SCHEMA}" in stdout
    assert "environment_status=objectstate_controlled_capture_environment_blocked" in stdout
    assert "video_devices=0" in stdout
    assert "controlled_capture_environment_ready=false" in stdout
    assert summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_ENVIRONMENT_SCHEMA
    assert summary["readiness"]["video_device_visible"] is False


def _fake_command_resolver(command: str) -> str | None:
    known = {
        "ffmpeg",
        "v4l2-ctl",
        "colmap",
        "ns-process-data",
        "ns-train",
        "ns-export",
    }
    if command not in known:
        return None
    return f"/usr/bin/{command}"


def _fake_importer(module: str):
    if module == "cv2":
        return SimpleNamespace(__version__="4.0-fixture")
    raise ImportError(module)


def _missing_importer(module: str):
    raise ImportError(module)
