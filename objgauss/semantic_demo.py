from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import Any

import numpy as np

from objgauss.clustering import summarize_labels
from objgauss.demo import _demo_camera_transform
from objgauss.features import colors, positions
from objgauss.mask_voting import (
    project_points,
    train_object_field_from_votes,
    training_summary,
    vote_masks_to_gaussians,
)
from objgauss.object_field import ObjectField, load_object_field, object_field_label_delta, object_field_metrics, save_object_field
from objgauss.ply import read_ply, write_ply
from objgauss.segment import assign_object_ids


PLUSH_COLOR_SLOTS: tuple[dict[str, int | str], ...] = (
    {"slot": 0, "label": "red-subject"},
    {"slot": 1, "label": "straw-frame"},
    {"slot": 2, "label": "dark-detail"},
    {"slot": 3, "label": "other-surface"},
)


@dataclass(frozen=True)
class PlushSemanticClosureResult:
    manifest_path: Path
    mask_manifest_path: Path
    initial_field_path: Path
    trained_field_path: Path
    output_ply_path: Path
    public_ply_path: Path | None
    public_splat_path: Path | None
    gaussian_count: int
    slot_count: int
    object_count: int
    supervised_gaussians: int
    initial_loss: float
    final_loss: float


@dataclass(frozen=True)
class PlushSemanticClosureVerification:
    manifest_path: Path
    passed: bool
    checks: tuple[dict[str, object], ...]
    summary: dict[str, object]


def build_plush_semantic_closure_demo(
    *,
    input_ply: str | Path = "outputs/assets/converted/plush.ply",
    splat_path: str | Path = "public/samples/plush.splat",
    output_dir: str | Path = "outputs/demos/plush-semantic-closure",
    public_dir: str | Path | None = "public/samples",
    image_size: int = 512,
    iterations: int = 160,
    learning_rate: float = 1.0,
) -> PlushSemanticClosureResult:
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    input_ply = Path(input_ply)
    splat_path = Path(splat_path)
    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    cloud = read_ply(input_ply)
    cloud.require_fields(("x", "y", "z"))
    slot_count = len(PLUSH_COLOR_SLOTS)
    field = ObjectField(np.zeros((cloud.count, slot_count), dtype=np.float32))

    mask_manifest_path = output_dir / "mask-manifest.json"
    mask_summary = _write_projected_color_mask_manifest(
        cloud=cloud,
        path=mask_manifest_path,
        masks_dir=masks_dir,
        image_size=image_size,
    )
    votes = vote_masks_to_gaussians(cloud, mask_manifest_path, slots=slot_count)
    training = train_object_field_from_votes(
        field,
        votes,
        iterations=iterations,
        learning_rate=learning_rate,
    )
    field_delta = object_field_label_delta(field, training.field)

    initial_field_path = output_dir / "object_field_initial.npz"
    trained_field_path = output_dir / "object_field_trained.npz"
    output_ply_path = output_dir / "plush_semantic_objects.ply"
    save_object_field(initial_field_path, field)
    save_object_field(trained_field_path, training.field)

    labels = training.field.labels()
    object_count = int(np.unique(labels).shape[0])
    labeled = assign_object_ids(cloud, labels)
    write_ply(output_ply_path, labeled, fmt="binary_little_endian")

    public_ply_path = None
    public_splat_path = None
    if public_dir is not None:
        public_dir = Path(public_dir)
        public_dir.mkdir(parents=True, exist_ok=True)
        public_ply_path = public_dir / "plush_semantic_objects.ply"
        public_splat_path = public_dir / "plush_semantic.splat"
        copyfile(output_ply_path, public_ply_path)
        copyfile(splat_path, public_splat_path)

    manifest_path = output_dir / "plush-semantic-closure-manifest.json"
    public_assets = bool(
        public_ply_path
        and public_splat_path
        and public_ply_path.exists()
        and public_splat_path.exists()
    )
    manifest = {
        "demo": "ObjGauss Plush semantic closure",
        "input_ply": str(input_ply),
        "splat_path": str(splat_path),
        "gaussian_source": "external_3dgs_splat",
        "semantic_source": "projected_3dgs_color_masks",
        "note": (
            "This demo uses projected 2D color masks from a real 3DGS scene. "
            "It does not use KMeans labels, SAM, or CLIP."
        ),
        "gaussian_count": cloud.count,
        "slot_count": slot_count,
        "object_count": object_count,
        "slots": list(PLUSH_COLOR_SLOTS),
        "mask_manifest": str(mask_manifest_path),
        "initial_field": str(initial_field_path),
        "trained_field": str(trained_field_path),
        "output_ply": str(output_ply_path),
        "public_ply": str(public_ply_path) if public_ply_path else None,
        "public_splat": str(public_splat_path) if public_splat_path else None,
        "mask_summary": mask_summary,
        "trained_object_counts": summarize_labels(labels),
        "training": training_summary(training),
        "object_field_delta": field_delta.as_dict(),
        "acceptance": {
            "real_3dgs_scene_can_render": splat_path.exists(),
            "semantic_source_is_2d_masks": True,
            "object_field_saved": trained_field_path.exists(),
            "mask_votes_supervise_gaussians": training.supervised_gaussians > 0,
            "mask_guidance_changed_object_field": field_delta.changed_gaussians > 0,
            "projection_loss_decreased": training.final_loss < training.initial_loss,
            "viewer_ply_available": output_ply_path.exists(),
            "object_id_ply_available": output_ply_path.exists(),
            "public_assets_available": public_assets,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return PlushSemanticClosureResult(
        manifest_path=manifest_path,
        mask_manifest_path=mask_manifest_path,
        initial_field_path=initial_field_path,
        trained_field_path=trained_field_path,
        output_ply_path=output_ply_path,
        public_ply_path=public_ply_path,
        public_splat_path=public_splat_path,
        gaussian_count=cloud.count,
        slot_count=slot_count,
        object_count=object_count,
        supervised_gaussians=training.supervised_gaussians,
        initial_loss=training.initial_loss,
        final_loss=training.final_loss,
    )


def verify_plush_semantic_closure_demo(
    manifest_path: str | Path = "outputs/demos/plush-semantic-closure/plush-semantic-closure-manifest.json",
    *,
    asset_library_path: str | Path = "src/assetLibrary.js",
    require_public_copy: bool = True,
    min_views: int = 2,
) -> PlushSemanticClosureVerification:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise ValueError(f"Plush semantic closure manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    splat_path = _resolve_manifest_path(manifest.get("splat_path"), manifest_path)
    mask_manifest_path = _resolve_manifest_path(manifest.get("mask_manifest"), manifest_path)
    initial_field_path = _resolve_manifest_path(manifest.get("initial_field"), manifest_path)
    trained_field_path = _resolve_manifest_path(manifest.get("trained_field"), manifest_path)
    output_ply_path = _resolve_manifest_path(manifest.get("output_ply"), manifest_path)
    public_ply_path = _optional_manifest_path(manifest.get("public_ply"), manifest_path)
    public_splat_path = _optional_manifest_path(manifest.get("public_splat"), manifest_path)
    gaussian_count = int(manifest.get("gaussian_count", 0) or 0)
    slot_count = int(manifest.get("slot_count", 0) or 0)
    object_count = int(manifest.get("object_count", 0) or 0)

    add("real_3dgs_scene", splat_path.exists() and splat_path.stat().st_size > 0, str(splat_path))
    add(
        "gaussian_source_is_real_3dgs",
        manifest.get("gaussian_source") == "external_3dgs_splat",
        str(manifest.get("gaussian_source")),
    )
    add(
        "semantic_source_is_2d_color_masks",
        manifest.get("semantic_source") == "projected_3dgs_color_masks",
        str(manifest.get("semantic_source")),
    )

    frame_count, mask_count, missing_masks = _check_mask_manifest(mask_manifest_path)
    add("mask_manifest_exists", mask_manifest_path.exists(), str(mask_manifest_path))
    add("mask_manifest_uses_multiview", frame_count >= min_views and mask_count > 0, f"{frame_count} frames / {mask_count} masks")
    add("mask_files_exist", missing_masks == 0, f"missing_masks={missing_masks}")

    field_shape = None
    active_slots = 0
    changed_gaussians = 0
    changed_fraction = 0.0
    if trained_field_path.exists():
        field = load_object_field(trained_field_path)
        metrics = object_field_metrics(field)
        field_shape = tuple(int(value) for value in field.logits.shape)
        active_slots = metrics.active_slots
        add("object_field_saved", True, str(trained_field_path))
        add(
            "object_field_shape_matches_scene",
            field.gaussian_count == gaussian_count and field.slots == slot_count,
            f"logits={field_shape} scene=({gaussian_count}, {slot_count})",
        )
        add("object_field_has_active_slots", active_slots >= 2, f"active_slots={active_slots}")
    else:
        add("object_field_saved", False, str(trained_field_path))
        add("object_field_shape_matches_scene", False, "missing field")
        add("object_field_has_active_slots", False, "missing field")

    if initial_field_path.exists() and trained_field_path.exists():
        initial_field = load_object_field(initial_field_path)
        trained_field = load_object_field(trained_field_path)
        field_delta = object_field_label_delta(initial_field, trained_field)
        changed_gaussians = field_delta.changed_gaussians
        changed_fraction = field_delta.changed_fraction
        add(
            "mask_guidance_changed_object_field",
            changed_gaussians > 0,
            f"changed_gaussians={changed_gaussians} fraction={changed_fraction:.6f}",
        )
    else:
        add("mask_guidance_changed_object_field", False, "missing initial/trained field")

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

    exported_gaussians, exported_objects = _check_output_ply(output_ply_path, gaussian_count, object_count, add)
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

    add("frontend_asset_registered", _frontend_asset_registered(asset_library_path), str(asset_library_path))

    passed = all(bool(check["passed"]) for check in checks)
    return PlushSemanticClosureVerification(
        manifest_path=manifest_path,
        passed=passed,
        checks=tuple(checks),
        summary={
            "gaussians": gaussian_count,
            "slots": slot_count,
            "objects": object_count,
            "mask_frames": frame_count,
            "masks": mask_count,
            "field_shape": field_shape,
            "active_slots": active_slots,
            "changed_gaussians": changed_gaussians,
            "changed_fraction": changed_fraction,
            "supervised_gaussians": supervised,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "exported_gaussians": exported_gaussians,
            "exported_objects": exported_objects,
            "public_copy": public_ok,
            "missing_masks": missing_masks,
        },
    )


def _write_projected_color_mask_manifest(
    *,
    cloud,
    path: Path,
    masks_dir: Path,
    image_size: int,
) -> dict[str, Any]:
    xyz = positions(cloud)
    labels = _classify_plush_colors(colors(cloud))
    view_directions = (
        np.array([1.0, 0.55, 1.0], dtype=np.float32),
        np.array([-1.0, 0.55, 1.0], dtype=np.float32),
        np.array([0.15, 0.8, -1.0], dtype=np.float32),
    )
    slot_gaussian_counts = np.bincount(labels, minlength=len(PLUSH_COLOR_SLOTS)).astype(np.int64)
    slot_mask_pixels = np.zeros(len(PLUSH_COLOR_SLOTS), dtype=np.int64)
    frames = []
    visible_count = 0
    mask_count = 0

    for frame_index, direction in enumerate(view_directions):
        c2w = _demo_camera_transform(xyz, direction=direction)
        projection = project_points(
            xyz,
            transform_matrix=c2w,
            width=image_size,
            height=image_size,
            camera_angle_x=np.pi / 3.0,
        )
        masks = []
        for slot in range(len(PLUSH_COLOR_SLOTS)):
            selected = projection.visible & (labels == slot)
            if not np.any(selected):
                continue
            mask = np.zeros((image_size, image_size), dtype=bool)
            x = np.clip(np.floor(projection.u[selected]).astype(np.int64), 0, image_size - 1)
            y = np.clip(np.floor(projection.v[selected]).astype(np.int64), 0, image_size - 1)
            mask[y, x] = True
            pixels = int(np.count_nonzero(mask))
            if pixels == 0:
                continue
            slot_mask_pixels[slot] += pixels
            mask_path = masks_dir / f"frame_{frame_index:03d}_slot_{slot}.npy"
            np.save(mask_path, mask)
            masks.append(
                {
                    "slot": slot,
                    "label": str(PLUSH_COLOR_SLOTS[slot]["label"]),
                    "mask_path": str(mask_path.relative_to(path.parent)),
                }
            )
            visible_count += int(np.count_nonzero(selected))
            mask_count += 1
        frames.append(
            {
                "name": f"plush-color-view-{frame_index}",
                "transform_matrix": c2w.tolist(),
                "masks": masks,
            }
        )

    payload = {
        "width": image_size,
        "height": image_size,
        "camera_angle_x": float(np.pi / 3.0),
        "source_type": "projected_3dgs_color_masks",
        "slots": list(PLUSH_COLOR_SLOTS),
        "frames": frames,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "frames": len(frames),
        "masks": mask_count,
        "visible_gaussians": visible_count,
        "slot_gaussian_counts": _slot_count_summary(slot_gaussian_counts),
        "slot_mask_pixels": _slot_count_summary(slot_mask_pixels),
    }


def _classify_plush_colors(rgb: np.ndarray) -> np.ndarray:
    rgb255 = np.clip(rgb * 255.0, 0.0, 255.0)
    red = rgb255[:, 0]
    green = rgb255[:, 1]
    blue = rgb255[:, 2]
    brightness = (red + green + blue) / 3.0
    labels = np.full(rgb255.shape[0], 3, dtype=np.int32)
    red_subject = (red > 115) & (red > green * 1.25) & (red > blue * 1.25)
    dark_detail = brightness < 70
    straw_frame = (
        (red > 95)
        & (green > 70)
        & (blue < 160)
        & (red >= blue * 1.15)
        & ~red_subject
        & ~dark_detail
    )
    labels[red_subject] = 0
    labels[straw_frame] = 1
    labels[dark_detail] = 2
    return labels


def _slot_count_summary(counts: np.ndarray) -> list[dict[str, int | str]]:
    return [
        {
            "slot": int(slot["slot"]),
            "label": str(slot["label"]),
            "count": int(counts[int(slot["slot"])]),
        }
        for slot in PLUSH_COLOR_SLOTS
    ]


def _check_mask_manifest(path: Path) -> tuple[int, int, int]:
    if not path.exists():
        return 0, 0, 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return 0, 0, 0
    mask_count = 0
    missing_masks = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
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
    return len(frames), mask_count, missing_masks


def _check_output_ply(path: Path, gaussian_count: int, object_count: int, add) -> tuple[int, int]:
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
        "plush-semantic-closure-local" in text
        and "/samples/plush_semantic_objects.ply" in text
        and "/samples/plush_semantic.splat" in text
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
