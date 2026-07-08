from __future__ import annotations

import json
import zlib
from pathlib import Path
from struct import unpack
from typing import Any, Mapping

import numpy as np

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.io_ply import write_ply
from objgauss.core.objectstate_bop_capture_adapter import (
    BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
    OBJECTSTATE_BOP_CAPTURE_ADAPTER_SCHEMA,
    objectstate_bop_capture_adapter_summary,
    validate_objectstate_bop_capture_adapter_summary,
)

OBJECTSTATE_BOP_RGBD_GAUSSIAN_EXPORT_SCHEMA = (
    "objgauss-objectstate-bop-rgbd-gaussian-export-v1"
)


def objectstate_bop_rgbd_gaussian_export(
    scene_root: str | Path,
    *,
    sample_id: str,
    dataset_id: str = "bop-ycbv",
    object_category: str = "bop_objects",
    scenario: str = "bop_pose_sequence",
    fps: float = 30.0,
    license_text: str = "BOP dataset terms; verify source dataset license before redistribution",
    rgb_dir: str = "rgb",
    depth_dir: str = "depth",
    gaussian_dir: str = "gaussians",
    max_frames: int | None = None,
    frame_step: int = 1,
    identity_policy: str = BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    pose_track_max_distance_m: float = DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
    pixel_stride: int = 1,
    max_points_per_frame: int | None = 50_000,
    min_depth_m: float = 0.0,
    max_depth_m: float | None = None,
    overwrite: bool = False,
    ply_format: str = "binary_little_endian",
) -> dict[str, Any]:
    root = Path(scene_root)
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be >= 1")
    if max_points_per_frame is not None and max_points_per_frame < 1:
        raise ValueError("max_points_per_frame must be >= 1")
    if min_depth_m < 0:
        raise ValueError("min_depth_m must be >= 0")
    if max_depth_m is not None and max_depth_m <= min_depth_m:
        raise ValueError("max_depth_m must be greater than min_depth_m")
    if ply_format not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError("ply_format is unsupported")

    adapter = objectstate_bop_capture_adapter_summary(
        root,
        sample_id=sample_id,
        dataset_id=dataset_id,
        object_category=object_category,
        scenario=scenario,
        fps=fps,
        license_text=license_text,
        rgb_dir=rgb_dir,
        max_frames=max_frames,
        frame_step=frame_step,
        identity_policy=identity_policy,
        pose_track_max_distance_m=pose_track_max_distance_m,
        include_gaussian_refs=True,
        gaussian_dir=gaussian_dir,
    )
    scene_camera = _read_json_mapping(root / "scene_camera.json", "scene_camera.json")
    frame_exports: list[dict[str, Any]] = []
    for frame_id in adapter["selected_frame_ids"]:
        frame_exports.append(
            _export_frame(
                root,
                frame_id,
                scene_camera,
                adapter,
                depth_dir=depth_dir,
                gaussian_dir=gaussian_dir,
                pixel_stride=pixel_stride,
                max_points_per_frame=max_points_per_frame,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
                overwrite=overwrite,
                ply_format=ply_format,
            )
        )

    exported = [record for record in frame_exports if record["valid"]]
    missing_depth = [record for record in frame_exports if not record["depth_file_present"]]
    zero_point = [
        record
        for record in frame_exports
        if record["depth_file_present"] and record["vertex_count"] == 0
    ]
    readiness = {
        "bop_adapter_ready": adapter["status"]
        == "objectstate_bop_capture_adapter_ready",
        "depth_files_present": not missing_depth,
        "selected_frames_exported": len(exported) == len(frame_exports),
        "gaussian_files_written_or_present": len(exported) == len(frame_exports),
        "phase1_gaussian_evidence_written": bool(frame_exports)
        and len(exported) == len(frame_exports),
    }
    payload = {
        "schema": OBJECTSTATE_BOP_RGBD_GAUSSIAN_EXPORT_SCHEMA,
        "kind": "objectstate_bop_rgbd_gaussian_export",
        "status": (
            "objectstate_bop_rgbd_gaussian_export_ready"
            if readiness["phase1_gaussian_evidence_written"]
            else "objectstate_bop_rgbd_gaussian_export_blocked"
        ),
        "scene_root": str(root),
        "sample_id": sample_id,
        "dataset_id": dataset_id,
        "adapter_schema": OBJECTSTATE_BOP_CAPTURE_ADAPTER_SCHEMA,
        "requirements": {
            "rgb_dir": rgb_dir,
            "depth_dir": depth_dir,
            "gaussian_dir": gaussian_dir,
            "pixel_stride": int(pixel_stride),
            "identity_policy": identity_policy,
            "pose_track_max_distance_m": float(pose_track_max_distance_m),
            "max_points_per_frame": max_points_per_frame,
            "min_depth_m": float(min_depth_m),
            "max_depth_m": float(max_depth_m) if max_depth_m is not None else None,
            "overwrite": bool(overwrite),
            "ply_format": ply_format,
        },
        "readiness": readiness,
        "row_counts": {
            "selected_frames": len(frame_exports),
            "exported_frames": len(exported),
            "missing_depth_files": len(missing_depth),
            "zero_point_frames": len(zero_point),
            "total_vertices": int(
                sum(record["vertex_count"] or 0 for record in exported)
            ),
        },
        "frame_exports": frame_exports,
        "adapter": adapter,
        "hard_blockers": _hard_blockers(missing_depth, zero_point),
        "next_actions": _next_actions(readiness, missing_depth, zero_point),
        "claim_policy": {
            "writes_local_gaussian_evidence": True,
            "rgbd_backprojection_only": True,
            "uses_bop_camera_intrinsics": True,
            "uses_depth_pixels_for_geometry": True,
            "does_not_use_object_pose_gt_for_geometry": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_condition_metadata": True,
            "does_not_train_splatfacto": True,
            "does_not_create_checkpoint": True,
            "does_not_run_identity_or_prediction_handoff": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "creates_ground_truth": False,
            "infers_condition_metadata": False,
            "trains_gaussian_model": False,
            "writes_public_samples": False,
            "creates_reality_pass_rows": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_bop_rgbd_gaussian_export_summary(payload)


def validate_objectstate_bop_rgbd_gaussian_export_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP RGB-D Gaussian export summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_RGBD_GAUSSIAN_EXPORT_SCHEMA:
        raise ValueError(
            "unsupported BOP RGB-D Gaussian export schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_rgbd_gaussian_export":
        raise ValueError("BOP RGB-D Gaussian export kind is unsupported")
    if payload.get("status") not in {
        "objectstate_bop_rgbd_gaussian_export_ready",
        "objectstate_bop_rgbd_gaussian_export_blocked",
    }:
        raise ValueError("BOP RGB-D Gaussian export status is unsupported")
    for key in ("scene_root", "sample_id", "dataset_id"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"BOP RGB-D Gaussian export requires {key}")
    if payload.get("adapter_schema") != OBJECTSTATE_BOP_CAPTURE_ADAPTER_SCHEMA:
        raise ValueError("BOP RGB-D Gaussian export adapter_schema mismatch")
    requirements = payload.get("requirements")
    readiness = payload.get("readiness")
    row_counts = payload.get("row_counts")
    if not isinstance(requirements, Mapping):
        raise ValueError("BOP RGB-D Gaussian export requires requirements")
    if not isinstance(readiness, Mapping):
        raise ValueError("BOP RGB-D Gaussian export requires readiness")
    if not isinstance(row_counts, Mapping):
        raise ValueError("BOP RGB-D Gaussian export requires row_counts")
    for key in (
        "bop_adapter_ready",
        "depth_files_present",
        "selected_frames_exported",
        "gaussian_files_written_or_present",
        "phase1_gaussian_evidence_written",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"BOP RGB-D Gaussian export readiness {key} invalid")
    expected_status = (
        "objectstate_bop_rgbd_gaussian_export_ready"
        if readiness["phase1_gaussian_evidence_written"]
        else "objectstate_bop_rgbd_gaussian_export_blocked"
    )
    if payload["status"] != expected_status:
        raise ValueError("BOP RGB-D Gaussian export status mismatch")
    for key in (
        "selected_frames",
        "exported_frames",
        "missing_depth_files",
        "zero_point_frames",
        "total_vertices",
    ):
        if not isinstance(row_counts.get(key), int):
            raise ValueError(f"BOP RGB-D Gaussian export row count {key} invalid")
    frame_exports = payload.get("frame_exports")
    if not isinstance(frame_exports, list) or not frame_exports:
        raise ValueError("BOP RGB-D Gaussian export requires frame_exports")
    for record in frame_exports:
        _validate_frame_export(record)
    if row_counts["selected_frames"] != len(frame_exports):
        raise ValueError("BOP RGB-D Gaussian export selected frame count mismatch")
    if row_counts["exported_frames"] != sum(1 for item in frame_exports if item["valid"]):
        raise ValueError("BOP RGB-D Gaussian export exported frame count mismatch")
    if row_counts["missing_depth_files"] != sum(
        1 for item in frame_exports if not item["depth_file_present"]
    ):
        raise ValueError("BOP RGB-D Gaussian export missing depth count mismatch")
    validate_objectstate_bop_capture_adapter_summary(payload.get("adapter"))
    for key in ("hard_blockers", "next_actions"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"BOP RGB-D Gaussian export requires {key}")
    claim_policy = payload.get("claim_policy")
    non_goals = payload.get("non_goals")
    if not isinstance(claim_policy, Mapping) or not isinstance(non_goals, Mapping):
        raise ValueError("BOP RGB-D Gaussian export requires claim policy")
    if (
        not claim_policy.get("rgbd_backprojection_only")
        or not claim_policy.get("does_not_use_object_pose_gt_for_geometry")
        or not claim_policy.get("does_not_train_splatfacto")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP RGB-D Gaussian export claim policy is too broad")
    if (
        non_goals.get("downloads_dataset")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("writes_public_samples")
        or non_goals.get("creates_reality_pass_rows")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError("BOP RGB-D Gaussian export non_goals cannot claim broad work")
    return dict(payload)


def _export_frame(
    root: Path,
    frame_id: int,
    scene_camera: Mapping[str, Any],
    adapter: Mapping[str, Any],
    *,
    depth_dir: str,
    gaussian_dir: str,
    pixel_stride: int,
    max_points_per_frame: int | None,
    min_depth_m: float,
    max_depth_m: float | None,
    overwrite: bool,
    ply_format: str,
) -> dict[str, Any]:
    frame_key = str(frame_id)
    camera = scene_camera.get(frame_key)
    if not isinstance(camera, Mapping):
        raise ValueError(f"scene_camera missing frame {frame_id}")
    depth_ref = f"{depth_dir}/{frame_id:06d}.png"
    output_ref = f"{gaussian_dir}/{frame_id:06d}.ply"
    depth_path = root / depth_ref
    output_path = root / output_ref
    rgb_ref = _rgb_ref_for_frame(adapter, frame_id)
    rgb_path = root / rgb_ref

    if not depth_path.is_file():
        return _frame_record(
            frame_id,
            rgb_ref=rgb_ref,
            depth_ref=depth_ref,
            output_ref=output_ref,
            output_path=output_path,
            depth_file_present=False,
            valid=False,
            vertex_count=0,
            color_source="not_read",
            action="blocked_missing_depth",
            issue=f"missing depth file: {depth_ref}",
        )
    if output_path.exists() and not overwrite:
        return _frame_record(
            frame_id,
            rgb_ref=rgb_ref,
            depth_ref=depth_ref,
            output_ref=output_ref,
            output_path=output_path,
            depth_file_present=True,
            valid=output_path.is_file() and output_path.stat().st_size > 0,
            vertex_count=None,
            color_source="not_read_existing_output",
            action="kept_existing",
            issue=None if output_path.is_file() else f"output is not a file: {output_ref}",
        )

    depth = _read_depth_png(depth_path)
    rgb, color_source = _read_rgb_for_depth(rgb_path, depth.shape)
    vertices = _rgbd_vertices(
        depth,
        rgb,
        camera,
        pixel_stride=pixel_stride,
        max_points_per_frame=max_points_per_frame,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    if vertices.shape[0] == 0:
        return _frame_record(
            frame_id,
            rgb_ref=rgb_ref,
            depth_ref=depth_ref,
            output_ref=output_ref,
            output_path=output_path,
            depth_file_present=True,
            valid=False,
            vertex_count=0,
            color_source=color_source,
            action="blocked_zero_points",
            issue="depth image produced no valid points",
        )
    write_ply(
        output_path,
        GaussianCloud(
            vertices=vertices,
            comments=(
                "ObjGauss BOP RGB-D depth-backprojection evidence",
                "not a trained Splatfacto or optimized 3DGS checkpoint",
            ),
            source_format=ply_format,
        ),
        fmt=ply_format,
    )
    return _frame_record(
        frame_id,
        rgb_ref=rgb_ref,
        depth_ref=depth_ref,
        output_ref=output_ref,
        output_path=output_path,
        depth_file_present=True,
        valid=True,
        vertex_count=int(vertices.shape[0]),
        color_source=color_source,
        action="wrote_ply",
        issue=None,
    )


def _rgbd_vertices(
    depth: np.ndarray,
    rgb: np.ndarray | None,
    camera: Mapping[str, Any],
    *,
    pixel_stride: int,
    max_points_per_frame: int | None,
    min_depth_m: float,
    max_depth_m: float | None,
) -> np.ndarray:
    cam_k = camera.get("cam_K")
    if not isinstance(cam_k, list) or len(cam_k) != 9:
        raise ValueError("scene_camera cam_K must contain 9 values")
    fx, fy = float(cam_k[0]), float(cam_k[4])
    cx, cy = float(cam_k[2]), float(cam_k[5])
    if fx == 0.0 or fy == 0.0:
        raise ValueError("scene_camera cam_K has zero focal length")
    depth_scale = float(camera.get("depth_scale", 1.0))
    depth_m = depth.astype(np.float32) * np.float32(depth_scale / 1000.0)
    sampled = depth_m[::pixel_stride, ::pixel_stride]
    valid = sampled > float(min_depth_m)
    if max_depth_m is not None:
        valid &= sampled <= float(max_depth_m)
    rows, cols = np.nonzero(valid)
    if rows.shape[0] == 0:
        return _empty_vertices()
    z = sampled[rows, cols].astype(np.float32, copy=False)
    u = (cols * pixel_stride).astype(np.float32)
    v = (rows * pixel_stride).astype(np.float32)
    if max_points_per_frame is not None and z.shape[0] > max_points_per_frame:
        keep = np.linspace(0, z.shape[0] - 1, max_points_per_frame, dtype=np.int64)
        z = z[keep]
        u = u[keep]
        v = v[keep]
        rows = rows[keep]
        cols = cols[keep]
    vertices = np.empty(
        z.shape[0],
        dtype=np.dtype(
            [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        ),
    )
    vertices["x"] = (u - np.float32(cx)) * z / np.float32(fx)
    vertices["y"] = (v - np.float32(cy)) * z / np.float32(fy)
    vertices["z"] = z
    if rgb is None:
        vertices["red"] = 127
        vertices["green"] = 127
        vertices["blue"] = 127
    else:
        if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
            rgb_rows = np.clip(
                np.round(v * (rgb.shape[0] - 1) / max(depth.shape[0] - 1, 1)).astype(int),
                0,
                rgb.shape[0] - 1,
            )
            rgb_cols = np.clip(
                np.round(u * (rgb.shape[1] - 1) / max(depth.shape[1] - 1, 1)).astype(int),
                0,
                rgb.shape[1] - 1,
            )
        else:
            rgb_rows = (rows * pixel_stride).astype(int)
            rgb_cols = (cols * pixel_stride).astype(int)
        colors = rgb[rgb_rows, rgb_cols, :3]
        vertices["red"] = colors[:, 0]
        vertices["green"] = colors[:, 1]
        vertices["blue"] = colors[:, 2]
    return vertices


def _empty_vertices() -> np.ndarray:
    return np.empty(
        0,
        dtype=np.dtype(
            [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        ),
    )


def _read_depth_png(path: Path) -> np.ndarray:
    image = _read_png(path)
    if image.ndim == 3:
        image = image[:, :, 0]
    if image.dtype.kind not in {"u", "i"}:
        raise ValueError(f"depth PNG must be integer encoded: {path}")
    return image.astype(np.uint16, copy=False)


def _read_rgb_for_depth(
    path: Path,
    depth_shape: tuple[int, int],
) -> tuple[np.ndarray | None, str]:
    if not path.is_file():
        return None, "constant_gray_missing_rgb"
    if path.suffix.lower() != ".png":
        return None, "constant_gray_unsupported_rgb_format"
    image = _read_png(path)
    if image.ndim == 2:
        rgb = np.repeat(image[:, :, None].astype(np.uint8, copy=False), 3, axis=2)
    else:
        rgb = image[:, :, :3].astype(np.uint8, copy=False)
    if rgb.shape[:2] != depth_shape:
        return rgb, "rgb_png_resampled"
    return rgb, "rgb_png"


def _read_png(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"not a PNG file: {path}")
    offset = 8
    width = height = bit_depth = color_type = interlace = None
    idat = bytearray()
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError(f"malformed PNG chunk header: {path}")
        length = unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        offset += 8
        chunk_data = data[offset : offset + length]
        offset += length + 4
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _compression, _filter, interlace = unpack(
                ">IIBBBBB",
                chunk_data,
            )
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
    if None in {width, height, bit_depth, color_type, interlace}:
        raise ValueError(f"PNG missing IHDR: {path}")
    if interlace != 0:
        raise ValueError(f"interlaced PNG is unsupported: {path}")
    if bit_depth not in {8, 16}:
        raise ValueError(f"PNG bit depth unsupported for BOP RGB-D export: {bit_depth}")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise ValueError(f"PNG color type unsupported for BOP RGB-D export: {color_type}")
    bytes_per_sample = bit_depth // 8
    scanline_length = int(width) * channels * bytes_per_sample
    raw = zlib.decompress(bytes(idat))
    rows = []
    previous = bytearray(scanline_length)
    source_offset = 0
    bpp = max(1, channels * bytes_per_sample)
    for _row_index in range(int(height)):
        filter_type = raw[source_offset]
        source_offset += 1
        scanline = bytearray(raw[source_offset : source_offset + scanline_length])
        source_offset += scanline_length
        reconstructed = _unfilter_png_scanline(scanline, previous, filter_type, bpp)
        rows.append(bytes(reconstructed))
        previous = reconstructed
    if source_offset != len(raw):
        raise ValueError(f"PNG has trailing decompressed bytes: {path}")
    pixel_bytes = b"".join(rows)
    if bit_depth == 8:
        array = np.frombuffer(pixel_bytes, dtype=np.uint8)
    else:
        array = np.frombuffer(pixel_bytes, dtype=">u2").astype(np.uint16)
    array = array.reshape((int(height), int(width), channels))
    if channels == 1:
        return array[:, :, 0]
    return array


def _unfilter_png_scanline(
    scanline: bytearray,
    previous: bytearray,
    filter_type: int,
    bpp: int,
) -> bytearray:
    output = bytearray(len(scanline))
    for index, value in enumerate(scanline):
        left = output[index - bpp] if index >= bpp else 0
        up = previous[index] if previous else 0
        up_left = previous[index - bpp] if previous and index >= bpp else 0
        if filter_type == 0:
            predicted = 0
        elif filter_type == 1:
            predicted = left
        elif filter_type == 2:
            predicted = up
        elif filter_type == 3:
            predicted = (left + up) // 2
        elif filter_type == 4:
            predicted = _paeth(left, up, up_left)
        else:
            raise ValueError(f"unsupported PNG filter type: {filter_type}")
        output[index] = (value + predicted) & 0xFF
    return output


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"BOP RGB-D Gaussian export missing {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _rgb_ref_for_frame(adapter: Mapping[str, Any], frame_id: int) -> str:
    for frame in adapter["manifest"]["frames"]:
        if frame["frame_id"] == f"bop-frame-{frame_id:06d}":
            return frame["observation"]["rgb"]
    raise ValueError(f"adapter manifest missing frame {frame_id}")


def _frame_record(
    frame_id: int,
    *,
    rgb_ref: str,
    depth_ref: str,
    output_ref: str,
    output_path: Path,
    depth_file_present: bool,
    valid: bool,
    vertex_count: int | None,
    color_source: str,
    action: str,
    issue: str | None,
) -> dict[str, Any]:
    return {
        "frame_id": f"{frame_id:06d}",
        "rgb_ref": rgb_ref,
        "depth_ref": depth_ref,
        "output_ref": output_ref,
        "output_path": str(output_path),
        "depth_file_present": bool(depth_file_present),
        "valid": bool(valid),
        "vertex_count": vertex_count,
        "color_source": color_source,
        "action": action,
        "issue": issue,
    }


def _validate_frame_export(record: Any) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("BOP RGB-D Gaussian export frame record must be a mapping")
    for key in ("frame_id", "rgb_ref", "depth_ref", "output_ref", "output_path", "color_source", "action"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"BOP RGB-D Gaussian export frame record requires {key}")
    for key in ("depth_file_present", "valid"):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"BOP RGB-D Gaussian export frame record {key} invalid")
    vertex_count = record.get("vertex_count")
    if vertex_count is not None and (
        not isinstance(vertex_count, int) or vertex_count < 0
    ):
        raise ValueError("BOP RGB-D Gaussian export vertex_count invalid")
    issue = record.get("issue")
    if issue is not None and not isinstance(issue, str):
        raise ValueError("BOP RGB-D Gaussian export issue must be a string")


def _hard_blockers(
    missing_depth: list[Mapping[str, Any]],
    zero_point: list[Mapping[str, Any]],
) -> list[str]:
    blockers = []
    if missing_depth:
        blockers.append("selected BOP frames are missing depth PNG files")
    if zero_point:
        blockers.append("selected BOP depth frames produced no valid RGB-D points")
    return blockers


def _next_actions(
    readiness: Mapping[str, bool],
    missing_depth: list[Mapping[str, Any]],
    zero_point: list[Mapping[str, Any]],
) -> list[str]:
    actions = []
    if missing_depth:
        actions.append("place BOP depth/<frame>.png files next to the selected RGB frames")
    if zero_point:
        actions.append("check depth_scale, min_depth_m, max_depth_m, and depth PNG encoding")
    if readiness["phase1_gaussian_evidence_written"]:
        actions.append("rerun audit-bop-gaussian-evidence with --require-ready")
        actions.append("continue to BOP identity / prediction route audits")
    return actions
