from __future__ import annotations

import binascii
import json
import struct
import zlib

import numpy as np

from objgauss.cli import main
from objgauss.core.io_ply import read_ply
from objgauss.core.objectstate_bop_rgbd_gaussian_export import (
    OBJECTSTATE_BOP_RGBD_GAUSSIAN_EXPORT_SCHEMA,
    objectstate_bop_rgbd_gaussian_export,
    validate_objectstate_bop_rgbd_gaussian_export_summary,
)


def test_bop_rgbd_gaussian_export_writes_depth_backprojection_ply(tmp_path):
    _write_bop_rgbd_scene(tmp_path)

    summary = objectstate_bop_rgbd_gaussian_export(
        tmp_path,
        sample_id="bop-ycbv-scene-000001",
        ply_format="ascii",
        max_points_per_frame=None,
    )

    assert summary["schema"] == OBJECTSTATE_BOP_RGBD_GAUSSIAN_EXPORT_SCHEMA
    assert summary["status"] == "objectstate_bop_rgbd_gaussian_export_ready"
    assert summary["readiness"]["phase1_gaussian_evidence_written"] is True
    assert summary["row_counts"] == {
        "selected_frames": 2,
        "exported_frames": 2,
        "missing_depth_files": 0,
        "zero_point_frames": 0,
        "total_vertices": 6,
    }
    assert summary["frame_exports"][0]["output_ref"] == "gaussians/000000.ply"
    assert summary["frame_exports"][0]["color_source"] == "rgb_png"
    assert (
        validate_objectstate_bop_rgbd_gaussian_export_summary(summary)
        == summary
    )

    cloud = read_ply(tmp_path / "gaussians" / "000000.ply")
    assert cloud.count == 3
    assert cloud.fields == ("x", "y", "z", "red", "green", "blue")
    assert np.isclose(float(cloud.vertices["z"][0]), 1.0)
    assert int(cloud.vertices["red"][0]) == 255


def test_bop_rgbd_gaussian_export_blocks_missing_depth(tmp_path):
    _write_bop_rgbd_scene(tmp_path, include_depth=False)

    summary = objectstate_bop_rgbd_gaussian_export(
        tmp_path,
        sample_id="bop-ycbv-scene-000001",
    )

    assert summary["status"] == "objectstate_bop_rgbd_gaussian_export_blocked"
    assert summary["readiness"]["depth_files_present"] is False
    assert summary["row_counts"]["missing_depth_files"] == 2
    assert "selected BOP frames are missing depth PNG files" in summary["hard_blockers"]
    assert not (tmp_path / "gaussians" / "000000.ply").exists()


def test_bop_rgbd_gaussian_export_cli_writes_summary(tmp_path, capsys):
    _write_bop_rgbd_scene(tmp_path)
    summary_path = tmp_path / "bop-rgbd-gaussian-export-summary.json"

    assert (
        main(
            [
                "object-state",
                "export-bop-rgbd-gaussian-evidence",
                str(tmp_path),
                "--sample-id",
                "bop-ycbv-scene-000001",
                "--summary-output",
                str(summary_path),
                "--ply-format",
                "ascii",
                "--require-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_BOP_RGBD_GAUSSIAN_EXPORT_SCHEMA}" in stdout
    assert (
        "bop_rgbd_gaussian_export_status="
        "objectstate_bop_rgbd_gaussian_export_ready"
    ) in stdout
    assert "exported_frames=2" in stdout
    assert "total_vertices=6" in stdout
    assert summary["row_counts"]["exported_frames"] == 2
    assert (tmp_path / "gaussians" / "000001.ply").is_file()


def _write_bop_rgbd_scene(root, *, include_depth: bool = True) -> None:
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    depth = np.array([[1000, 2000], [0, 3000]], dtype=np.uint16)
    for frame_id in range(2):
        (root / "rgb" / f"{frame_id:06d}.png").write_bytes(_png_bytes(rgb))
        if include_depth:
            (root / "depth" / f"{frame_id:06d}.png").write_bytes(_png_bytes(depth))
    scene_camera = {
        str(frame_id): {
            "cam_K": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "depth_scale": 1.0,
        }
        for frame_id in range(2)
    }
    identity_rotation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    scene_gt = {
        str(frame_id): [
            {
                "obj_id": 1,
                "cam_R_m2c": identity_rotation,
                "cam_t_m2c": [10.0 + frame_id, 20.0, 30.0],
            }
        ]
        for frame_id in range(2)
    }
    scene_gt_info = {str(frame_id): [{"visib_fract": 1.0}] for frame_id in range(2)}
    _write_json(root / "scene_camera.json", scene_camera)
    _write_json(root / "scene_gt.json", scene_gt)
    _write_json(root / "scene_gt_info.json", scene_gt_info)


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _png_bytes(array: np.ndarray) -> bytes:
    if array.dtype == np.uint8:
        bit_depth = 8
        row_bytes = array
    elif array.dtype == np.uint16:
        bit_depth = 16
        row_bytes = array.astype(">u2", copy=False)
    else:
        raise TypeError("test PNG helper supports uint8 and uint16 only")
    if array.ndim == 2:
        height, width = array.shape
        color_type = 0
    elif array.ndim == 3 and array.shape[2] == 3:
        height, width, _channels = array.shape
        color_type = 2
    else:
        raise ValueError("test PNG helper supports grayscale or RGB arrays")
    raw = b"".join(
        b"\x00" + row_bytes[row_index].tobytes()
        for row_index in range(height)
    )
    ihdr = struct.pack(
        ">IIBBBBB",
        width,
        height,
        bit_depth,
        color_type,
        0,
        0,
        0,
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )


def _chunk(kind: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)
