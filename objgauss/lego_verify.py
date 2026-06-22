from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from objgauss.object_field import load_object_field, object_field_metrics
from objgauss.ply import read_ply


@dataclass(frozen=True)
class LegoClosureVerification:
    manifest_path: Path
    passed: bool
    checks: tuple[dict[str, object], ...]
    summary: dict[str, object]


def verify_lego_alpha_closure_demo(
    manifest_path: str | Path = "outputs/demos/lego-alpha-closure/lego-alpha-closure-manifest.json",
    *,
    asset_library_path: str | Path = "src/assetLibrary.js",
    require_public_copy: bool = True,
    min_frames: int = 2,
) -> LegoClosureVerification:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise ValueError(f"Lego closure manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    dataset = _resolve_manifest_path(manifest.get("dataset"), manifest_path)
    splat_path = _resolve_manifest_path(manifest.get("splat_path"), manifest_path)
    mask_manifest_path = _resolve_manifest_path(manifest.get("mask_manifest"), manifest_path)
    trained_field_path = _resolve_manifest_path(manifest.get("trained_field"), manifest_path)
    output_ply_path = _resolve_manifest_path(manifest.get("output_ply"), manifest_path)
    public_ply_path = _optional_manifest_path(manifest.get("public_ply"), manifest_path)
    public_splat_path = _optional_manifest_path(manifest.get("public_splat"), manifest_path)
    gaussian_count = int(manifest.get("gaussian_count", 0) or 0)
    object_count = int(manifest.get("object_count", 0) or 0)

    add("dataset_exists", dataset.exists(), str(dataset))
    add(
        "gaussian_source_is_proxy",
        manifest.get("gaussian_source") == "nerf_rgba_pose_proxy",
        str(manifest.get("gaussian_source")),
    )
    add(
        "semantic_source_is_real_2d_masks",
        manifest.get("semantic_source") == "nerf_rgba_color_masks",
        str(manifest.get("semantic_source")),
    )
    add(
        "proxy_splat_available",
        splat_path.exists() and splat_path.stat().st_size > 0,
        str(splat_path),
    )

    frame_count, mask_count, missing_images, missing_masks = _check_mask_manifest(
        mask_manifest_path,
        dataset,
    )
    add("mask_manifest_exists", mask_manifest_path.exists(), str(mask_manifest_path))
    add(
        "mask_manifest_uses_multiview",
        frame_count >= min_frames and mask_count > 0,
        f"{frame_count} frames / {mask_count} masks",
    )
    add("mask_manifest_images_exist", missing_images == 0, f"missing_images={missing_images}")
    add("mask_files_exist", missing_masks == 0, f"missing_masks={missing_masks}")

    field_shape = None
    active_slots = 0
    if trained_field_path.exists():
        field = load_object_field(trained_field_path)
        metrics = object_field_metrics(field)
        field_shape = tuple(int(value) for value in field.logits.shape)
        active_slots = metrics.active_slots
        add("object_field_saved", True, str(trained_field_path))
        add(
            "object_field_shape_matches_scene",
            field.gaussian_count == gaussian_count and field.slots == object_count,
            f"logits={field_shape} scene=({gaussian_count}, {object_count})",
        )
        add("object_field_has_active_slots", active_slots >= min(object_count, 2), f"active_slots={active_slots}")
    else:
        add("object_field_saved", False, str(trained_field_path))
        add("object_field_shape_matches_scene", False, "missing field")
        add("object_field_has_active_slots", False, "missing field")

    training = manifest.get("training") if isinstance(manifest.get("training"), dict) else {}
    initial_loss = _optional_float(training.get("initial_loss"))
    final_loss = _optional_float(training.get("final_loss"))
    supervised = int(training.get("supervised_gaussians", 0) or 0)
    add("mask_votes_supervise_gaussians", supervised > 0, f"supervised_gaussians={supervised}")
    add(
        "projection_loss_decreased",
        initial_loss is not None and final_loss is not None and final_loss < initial_loss,
        f"{initial_loss} -> {final_loss}",
    )

    exported_gaussians, exported_objects = _check_output_ply(
        output_ply_path,
        gaussian_count,
        object_count,
        add,
    )
    if require_public_copy:
        public_ok = bool(
            public_ply_path
            and public_splat_path
            and public_ply_path.exists()
            and public_splat_path.exists()
        )
        add("public_assets_available", public_ok, f"{public_ply_path} / {public_splat_path}")
    else:
        public_ok = True

    registered = _frontend_asset_registered(asset_library_path)
    add("frontend_asset_registered", registered, str(asset_library_path))
    passed = all(bool(check["passed"]) for check in checks)
    return LegoClosureVerification(
        manifest_path=manifest_path,
        passed=passed,
        checks=tuple(checks),
        summary={
            "gaussians": gaussian_count,
            "objects": object_count,
            "mask_frames": frame_count,
            "masks": mask_count,
            "field_shape": field_shape,
            "active_slots": active_slots,
            "supervised_gaussians": supervised,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "exported_gaussians": exported_gaussians,
            "exported_objects": exported_objects,
            "public_copy": public_ok,
            "missing_images": missing_images,
            "missing_masks": missing_masks,
        },
    )


def _check_mask_manifest(path: Path, dataset: Path) -> tuple[int, int, int, int]:
    if not path.exists():
        return 0, 0, 0, 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return 0, 0, 0, 0
    missing_images = 0
    missing_masks = 0
    mask_count = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        image_path = frame.get("image_path")
        if not isinstance(image_path, str) or not (dataset / image_path).exists():
            missing_images += 1
        masks = frame.get("masks")
        if not isinstance(masks, list):
            continue
        for mask in masks:
            if not isinstance(mask, dict):
                continue
            mask_count += 1
            mask_path = mask.get("mask_path")
            if not isinstance(mask_path, str) or not (path.parent / mask_path).exists():
                missing_masks += 1
    return len(frames), mask_count, missing_images, missing_masks


def _check_output_ply(
    path: Path,
    gaussian_count: int,
    object_count: int,
    add,
) -> tuple[int, int]:
    if not path.exists():
        add("viewer_ply_available", False, str(path))
        add("viewer_ply_exports_object_id", False, "missing PLY")
        return 0, 0
    cloud = read_ply(path)
    has_object_id = "object_id" in cloud.fields
    exported_objects = len(np.unique(cloud.vertices["object_id"])) if has_object_id else 0
    add("viewer_ply_available", True, str(path))
    add(
        "viewer_ply_exports_object_id",
        has_object_id and cloud.count == gaussian_count and exported_objects == object_count,
        f"gaussians={cloud.count} objects={exported_objects}",
    )
    return cloud.count, exported_objects


def _frontend_asset_registered(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return (
        "nerf-lego-alpha-closure-local" in text
        and "/samples/lego_alpha_v1_objects.ply" in text
        and "/samples/lego_alpha_proxy.splat" in text
    )


def _resolve_manifest_path(value: object, manifest_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        return manifest_path.parent / "__missing__"
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    candidate = manifest_path.parent / path
    return candidate if candidate.exists() else path


def _optional_manifest_path(value: object, manifest_path: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return _resolve_manifest_path(value, manifest_path)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
