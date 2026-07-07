from __future__ import annotations

import hashlib
import json

from objgauss.cli import main
from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
)
from objgauss.core.objectstate_controlled_capture_files import (
    OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA,
    objectstate_controlled_capture_file_audit,
    objectstate_controlled_capture_missing_files_markdown,
    validate_objectstate_controlled_capture_file_audit_summary,
)


def test_controlled_capture_file_audit_passes_when_frame_files_exist(tmp_path):
    manifest = _capture_manifest()
    _write_bundle_files(tmp_path, frame_count=3)

    summary = objectstate_controlled_capture_file_audit(
        manifest,
        root=tmp_path,
        check_artifact_refs=True,
    )

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA
    assert summary["status"] == "objectstate_controlled_capture_file_audit_pass"
    assert summary["file_counts"]["rgb"] == {
        "referenced": 3,
        "existing": 3,
        "valid": 3,
        "missing": 0,
    }
    assert summary["file_counts"]["gaussian"] == {
        "referenced": 3,
        "existing": 3,
        "valid": 3,
        "missing": 0,
    }
    assert summary["file_counts"]["artifact_refs"]["missing"] == 0
    assert summary["readiness"]["capture_bundle_files_ready"] is True
    assert summary["missing_files"] == []
    assert "no missing files" in summary["missing_files_markdown"]
    assert validate_objectstate_controlled_capture_file_audit_summary(summary) is summary


def test_controlled_capture_file_audit_fails_missing_gaussian_file(tmp_path):
    manifest = _capture_manifest()
    _write_bundle_files(tmp_path, frame_count=3)
    (tmp_path / "gaussians" / "000001.ply").unlink()

    summary = objectstate_controlled_capture_file_audit(manifest, root=tmp_path)

    assert summary["status"] == "objectstate_controlled_capture_file_audit_fail"
    assert summary["file_counts"]["gaussian"]["missing"] == 1
    assert summary["readiness"]["gaussian_files_present"] is False
    assert summary["readiness"]["capture_bundle_files_ready"] is False
    assert summary["missing_files"][0]["kind"] == "gaussian"
    assert summary["missing_files"][0]["frame_id"] == "frame-000001"
    assert "gaussians/000001.ply" in summary["missing_files_markdown"]


def test_controlled_capture_file_audit_fails_empty_rgb_file(tmp_path):
    manifest = _capture_manifest()
    _write_bundle_files(tmp_path, frame_count=3)
    (tmp_path / "rgb" / "000001.png").write_bytes(b"")

    summary = objectstate_controlled_capture_file_audit(manifest, root=tmp_path)

    assert summary["status"] == "objectstate_controlled_capture_file_audit_fail"
    assert summary["file_counts"]["rgb"] == {
        "referenced": 3,
        "existing": 3,
        "valid": 2,
        "missing": 1,
    }
    assert summary["readiness"]["rgb_files_present"] is False
    assert summary["missing_files"][0]["kind"] == "rgb"
    assert summary["missing_files"][0]["exists"] is True
    assert summary["missing_files"][0]["valid"] is False
    assert "file smaller than required minimum bytes" in summary["missing_files"][0][
        "missing_reason"
    ]


def test_controlled_capture_file_audit_allows_rgb_only_when_gaussian_not_required(tmp_path):
    manifest = _capture_manifest(include_gaussian=False)
    _write_bundle_files(tmp_path, frame_count=3, include_gaussian=False)

    summary = objectstate_controlled_capture_file_audit(
        manifest,
        root=tmp_path,
        require_gaussian_files=False,
    )

    assert summary["status"] == "objectstate_controlled_capture_file_audit_pass"
    assert summary["file_counts"]["gaussian"] == {
        "referenced": 0,
        "existing": 0,
        "valid": 0,
        "missing": 0,
    }
    assert summary["readiness"]["gaussian_files_present"] is True
    assert summary["capture_summary"]["readiness"]["real_gaussian_reconstruction_present"] is False


def test_controlled_capture_file_audit_can_check_artifact_refs(tmp_path):
    manifest = _capture_manifest()
    _write_bundle_files(tmp_path, frame_count=3)
    for path in (tmp_path / "gaussians").glob("*.ply"):
        path.unlink()
    (tmp_path / "gaussians").rmdir()

    summary = objectstate_controlled_capture_file_audit(
        manifest,
        root=tmp_path,
        require_gaussian_files=False,
        check_artifact_refs=True,
    )

    assert summary["status"] == "objectstate_controlled_capture_file_audit_fail"
    assert summary["file_counts"]["artifact_refs"]["missing"] == 1
    assert summary["readiness"]["artifact_refs_present"] is False


def test_controlled_capture_file_audit_can_hash_frame_files(tmp_path):
    manifest = _capture_manifest()
    _write_bundle_files(tmp_path, frame_count=3)

    summary = objectstate_controlled_capture_file_audit(
        manifest,
        root=tmp_path,
        check_artifact_refs=True,
        hash_files=True,
    )

    expected_rgb_hash = hashlib.sha256(b"rgb").hexdigest()

    assert summary["status"] == "objectstate_controlled_capture_file_audit_pass"
    assert summary["requirements"]["file_hashes_included"] is True
    assert summary["file_records"]["rgb"][0]["sha256"] == expected_rgb_hash
    assert len(summary["file_records"]["gaussian"][0]["sha256"]) == 64
    assert "sha256" not in summary["file_records"]["artifact_refs"][0]


def test_controlled_capture_missing_files_markdown_handles_empty_list():
    markdown = objectstate_controlled_capture_missing_files_markdown([])

    assert "no missing files" in markdown


def test_object_state_audit_controlled_capture_files_cli_writes_summary_and_markdown(
    tmp_path,
    capsys,
):
    manifest = _capture_manifest()
    _write_bundle_files(tmp_path, frame_count=3)
    manifest_path = tmp_path / "capture-manifest.json"
    summary_path = tmp_path / "file-audit.json"
    missing_path = tmp_path / "missing-files.md"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert (
        main(
            [
                "object-state",
                "audit-controlled-capture-files",
                str(manifest_path),
                "--summary-output",
                str(summary_path),
                "--missing-files-output",
                str(missing_path),
                "--check-artifact-refs",
                "--require-pass",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    missing = missing_path.read_text(encoding="utf-8")

    assert f"schema={OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA}" in stdout
    assert "file_audit_status=objectstate_controlled_capture_file_audit_pass" in stdout
    assert "rgb_valid=3/3" in stdout
    assert "gaussian_valid=3/3" in stdout
    assert summary["status"] == "objectstate_controlled_capture_file_audit_pass"
    assert "no missing files" in missing


def _capture_manifest(*, include_gaussian: bool = True):
    frames = []
    for index, timestamp in enumerate((0.0, 0.033333, 0.066667)):
        observation = {"rgb": f"rgb/{index:06d}.png"}
        if include_gaussian:
            observation["gaussian"] = f"gaussians/{index:06d}.ply"
        frames.append(
            {
                "frame_id": f"frame-{index:06d}",
                "timestamp": timestamp,
                "observation": observation,
                "objects": [
                    {
                        "object_id": "cup-001",
                        "visible": True,
                        "occlusion_fraction": 0.0,
                        "pose": {
                            "position": [0.1 + index * 0.01, 0.2, 0.3],
                            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    }
                ],
            }
        )
    artifact_refs = ["capture-manifest.json", "rgb/"]
    if include_gaussian:
        artifact_refs.append("gaussians/")
    return {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": {
            "sample_id": "controlled-tabletop-cup-file-audit-001",
            "source_kind": "controlled_real",
            "object_category": "cup",
            "scenario": "cross_view_occlusion_reappearance",
            "fps": 30.0,
            "capture_device": "fixture-camera",
            "observation_modalities": ["rgb", "gaussian"] if include_gaussian else ["rgb"],
            "artifact_refs": artifact_refs,
            "license": "local controlled capture; not public release",
        },
        "objects": [
            {"object_id": "cup-001", "category": "cup", "instance_label": "blue cup"}
        ],
        "actions": [],
        "frames": frames,
    }


def _write_bundle_files(
    root,
    *,
    frame_count: int,
    include_gaussian: bool = True,
) -> None:
    (root / "capture-manifest.json").write_text("{}", encoding="utf-8")
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    for index in range(frame_count):
        (root / "rgb" / f"{index:06d}.png").write_text("rgb", encoding="utf-8")
    if include_gaussian:
        (root / "gaussians").mkdir(parents=True, exist_ok=True)
        for index in range(frame_count):
            (root / "gaussians" / f"{index:06d}.ply").write_text(
                "ply",
                encoding="utf-8",
            )
