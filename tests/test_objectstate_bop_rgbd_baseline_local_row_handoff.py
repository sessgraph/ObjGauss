from __future__ import annotations

import binascii
import json
import struct
import zlib

import numpy as np

from objgauss.cli import main
from objgauss.core.objectstate_bop_capture_adapter import (
    OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA,
)
from objgauss.core.objectstate_bop_rgbd_baseline_local_row_handoff import (
    OBJECTSTATE_BOP_RGBD_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA,
    objectstate_bop_rgbd_baseline_local_row_handoff,
    validate_objectstate_bop_rgbd_baseline_local_row_handoff_summary,
)


def test_bop_rgbd_baseline_local_row_handoff_writes_reviewable_negative_evidence(
    tmp_path,
):
    scene_root = tmp_path / "bop-rgbd-scene"
    output_root = tmp_path / "rgbd-baseline-local-row"
    sidecar_path = scene_root / "bop-condition-sidecar.json"
    _write_bop_rgbd_scene(scene_root)
    _write_json(sidecar_path, _condition_sidecar_payload())

    summary = objectstate_bop_rgbd_baseline_local_row_handoff(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-rgbd-scene-000001",
        condition_sidecar=sidecar_path,
        ply_format="ascii",
        max_points_per_frame=None,
    )

    assert summary["schema"] == OBJECTSTATE_BOP_RGBD_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA
    assert (
        validate_objectstate_bop_rgbd_baseline_local_row_handoff_summary(summary)
        == summary
    )
    assert summary["status"] == (
        "objectstate_bop_rgbd_baseline_local_row_handoff_reviewable"
    )
    assert summary["rgbd_export"]["status"] == (
        "objectstate_bop_rgbd_gaussian_export_ready"
    )
    assert summary["baseline_local_row_handoff"]["status"] == (
        "objectstate_bop_baseline_local_row_handoff_reviewable"
    )
    assert summary["row_counts"]["selected_frames"] == 3
    assert summary["row_counts"]["exported_frames"] == 3
    assert summary["row_counts"]["missing_depth_files"] == 0
    assert summary["row_counts"]["rgbd_total_vertices"] == 9
    assert summary["row_counts"]["baseline_frames"] == 3
    assert summary["row_counts"]["baseline_states"] == 3
    assert summary["row_counts"]["baseline_total_gaussians"] == 9
    assert summary["reviewability_gates"] == {
        "rgbd_export_ready": True,
        "baseline_candidate_written": True,
        "baseline_candidate_ready_for_identity_handoff": True,
        "local_row_identity_handoff_reviewable": True,
        "local_row_prediction_handoff_reviewable": True,
        "phase1_evidence_ledger_identity_reviewable": True,
        "phase1_evidence_ledger_prediction_reviewable": True,
    }
    assert summary["pass_gates"]["identity_handoff_pass"] is False
    assert summary["pass_gates"]["prediction_eval_pass"] is True
    assert summary["claim_policy"]["rgbd_backprojection_only"] is True
    assert (
        summary["claim_policy"]["does_not_use_object_pose_gt_for_rgbd_geometry"]
        is True
    )
    assert (
        summary["claim_policy"][
            "does_not_use_bop_pose_gt_for_objectstate_prediction"
        ]
        is True
    )
    assert (scene_root / "gaussians" / "000000.ply").is_file()
    assert (output_root / "objectstates.json").is_file()
    assert (output_root / "phase1-evidence-ledger.json").is_file()


def test_bop_rgbd_baseline_local_row_handoff_blocks_missing_depth(tmp_path):
    scene_root = tmp_path / "bop-rgbd-scene"
    output_root = tmp_path / "rgbd-baseline-local-row"
    _write_bop_rgbd_scene(scene_root, include_depth=False)

    summary = objectstate_bop_rgbd_baseline_local_row_handoff(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-rgbd-scene-000001",
    )

    assert summary["status"] == (
        "objectstate_bop_rgbd_baseline_local_row_handoff_incomplete"
    )
    assert summary["rgbd_export"]["status"] == (
        "objectstate_bop_rgbd_gaussian_export_blocked"
    )
    assert summary["baseline_local_row_handoff"] is None
    assert summary["reviewability_gates"]["rgbd_export_ready"] is False
    assert summary["reviewability_gates"]["baseline_candidate_written"] is False
    assert summary["pass_gates"] == {
        "identity_handoff_pass": False,
        "prediction_eval_pass": False,
    }
    assert summary["row_counts"]["missing_depth_files"] == 3
    assert any("missing depth PNG" in issue for issue in summary["issues"])
    assert not (output_root / "objectstates.json").exists()


def test_bop_rgbd_baseline_local_row_handoff_cli(tmp_path, capsys):
    scene_root = tmp_path / "bop-rgbd-scene"
    output_root = tmp_path / "rgbd-baseline-local-row"
    sidecar_path = scene_root / "bop-condition-sidecar.json"
    summary_path = tmp_path / "rgbd-baseline-local-row-summary.json"
    _write_bop_rgbd_scene(scene_root)
    _write_json(sidecar_path, _condition_sidecar_payload())

    assert (
        main(
            [
                "object-state",
                "bop-rgbd-baseline-local-row-handoff",
                str(scene_root),
                "--output-root",
                str(output_root),
                "--sample-id",
                "bop-ycbv-rgbd-scene-000001",
                "--condition-sidecar",
                str(sidecar_path),
                "--summary-output",
                str(summary_path),
                "--ply-format",
                "ascii",
                "--require-reviewable",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = _read_json(summary_path)

    assert f"schema={OBJECTSTATE_BOP_RGBD_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA}" in stdout
    assert "bop_rgbd_baseline_local_row_handoff_status=" in stdout
    assert "reviewability.rgbd_export_ready=true" in stdout
    assert "reviewability.local_row_identity_handoff_reviewable=true" in stdout
    assert "pass.identity_handoff_pass=false" in stdout
    assert "rgbd_total_vertices=9" in stdout
    assert summary["status"] == (
        "objectstate_bop_rgbd_baseline_local_row_handoff_reviewable"
    )


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
    for frame_id in range(3):
        (root / "rgb" / f"{frame_id:06d}.png").write_bytes(_png_bytes(rgb))
        if include_depth:
            (root / "depth" / f"{frame_id:06d}.png").write_bytes(_png_bytes(depth))
    scene_camera = {
        str(frame_id): {
            "cam_K": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "depth_scale": 1.0,
        }
        for frame_id in range(3)
    }
    identity_rotation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    scene_gt = {}
    scene_gt_info = {}
    visibility_by_frame = (1.0, 0.2, 1.0)
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
            {
                "bbox_obj": [10, 20, 30, 40],
                "bbox_visib": [10, 20, 30, 40],
                "px_count_all": 1000,
                "px_count_valid": 1000,
                "px_count_visib": int(1000 * visibility_by_frame[frame_id]),
                "visib_fract": visibility_by_frame[frame_id],
            },
            {
                "bbox_obj": [50, 60, 30, 40],
                "bbox_visib": [50, 60, 30, 40],
                "px_count_all": 900,
                "px_count_valid": 900,
                "px_count_visib": 900,
                "visib_fract": 1.0,
            },
        ]
    _write_json(root / "scene_camera.json", scene_camera)
    _write_json(root / "scene_gt.json", scene_gt)
    _write_json(root / "scene_gt_info.json", scene_gt_info)


def _condition_sidecar_payload():
    return {
        "schema": OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA,
        "kind": "objectstate_bop_capture_condition_sidecar",
        "frames": {
            "0": {
                "view_id": "front",
                "lighting_id": "bright",
                "camera_pose": {
                    "position": [0.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "1": {
                "view_id": "front",
                "lighting_id": "dim",
                "camera_pose": {
                    "position": [0.02, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "000002": {
                "view_id": "right",
                "lighting_id": "dim",
                "camera_pose": {
                    "position": [0.04, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
        "condition_policy": {
            "sidecar_only": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_from_pixels": True,
        },
    }


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


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
