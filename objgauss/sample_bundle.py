from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from objgauss.object_field import load_object_field
from objgauss.ply import read_ply


@dataclass(frozen=True)
class SampleBundleResult:
    sample_path: Path
    sample_id: str
    asset_id: str
    image_count: int
    mask_frame_count: int
    gaussian_count: int | None
    object_field_gaussian_count: int | None
    slot_count: int | None


def write_sample_bundle(
    *,
    output: str | Path,
    sample_id: str,
    asset_id: str,
    dataset: str | Path,
    masks: str | Path,
    training_manifest: str | Path,
    split: str = "train",
) -> SampleBundleResult:
    if not sample_id:
        raise ValueError("sample_id is required")
    if not asset_id:
        raise ValueError("asset_id is required")

    output = Path(output)
    dataset = Path(dataset)
    masks = Path(masks)
    training_manifest = Path(training_manifest)
    transforms_path = dataset / f"transforms_{split}.json"
    if not transforms_path.exists():
        raise ValueError(f"missing transforms file: {transforms_path}")
    if not masks.exists():
        raise ValueError(f"missing mask manifest: {masks}")
    if not training_manifest.exists():
        raise ValueError(f"missing training output manifest: {training_manifest}")

    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    mask_manifest = json.loads(masks.read_text(encoding="utf-8"))
    training = json.loads(training_manifest.read_text(encoding="utf-8"))
    frames = transforms.get("frames") if isinstance(transforms.get("frames"), list) else []
    mask_frames = mask_manifest.get("frames") if isinstance(mask_manifest.get("frames"), list) else []
    slots = _normalized_slots(mask_manifest)

    gaussian_ply = _path_from_manifest(training.get("gaussian_ply"))
    splat_path = _path_from_manifest(training.get("splat_path"))
    object_field_path = _path_from_manifest(training.get("trained_field"))
    object_ply = _path_from_manifest(training.get("object_ply"))
    gaussian_count = _optional_int(training.get("gaussian_count"))
    if gaussian_count is None and gaussian_ply and gaussian_ply.exists():
        gaussian_count = read_ply(gaussian_ply).count

    object_field_gaussian_count = None
    object_field_slot_count = None
    if object_field_path and object_field_path.exists():
        field = load_object_field(object_field_path)
        object_field_gaussian_count = field.gaussian_count
        object_field_slot_count = field.slots

    object_ply_count = None
    if object_ply and object_ply.exists():
        object_ply_count = read_ply(object_ply).count

    base = output.parent
    image_dir = dataset / split
    sample = {
        "schema": "objgauss-sample-bundle-v1",
        "sample_id": sample_id,
        "asset_id": asset_id,
        "image_set": split,
        "image_count": len(frames),
        "image_width": mask_manifest.get("image_width", mask_manifest.get("width")),
        "image_height": mask_manifest.get("image_height", mask_manifest.get("height")),
        "dataset": _rel(dataset, base),
        "transforms": _rel(transforms_path, base),
        "mask_run_id": mask_manifest.get("mask_run_id") or mask_manifest.get("source_type"),
        "mask_manifest": _rel(masks, base),
        "mask_frame_count": len(mask_frames),
        "slot_count": len(slots) if slots else _optional_int(training.get("slots")),
        "slots": slots,
        "splat_run_id": training.get("asset_id"),
        "gaussian_ply": _rel(gaussian_ply, base) if gaussian_ply else None,
        "splat_path": _rel(splat_path, base) if splat_path else None,
        "gaussian_count": gaussian_count,
        "splat_sha256": _sha256(gaussian_ply) if gaussian_ply and gaussian_ply.exists() else None,
        "object_field_path": _rel(object_field_path, base) if object_field_path else None,
        "object_ply": _rel(object_ply, base) if object_ply else None,
        "object_ply_gaussian_count": object_ply_count,
        "object_field_gaussian_count": object_field_gaussian_count,
        "object_field_slot_count": object_field_slot_count,
        "training_manifest": _rel(training_manifest, base),
        "training": {
            "supervised_gaussians": _nested(training, ("training", "supervised_gaussians")),
            "initial_loss": _nested(training, ("training", "initial_loss")),
            "final_loss": _nested(training, ("training", "final_loss")),
            "background_training": training.get("background_training"),
            "object_label_policy": training.get("object_label_policy"),
        },
        "created_from": {
            "images": _rel(image_dir, base) if image_dir.exists() else _rel(dataset, base),
            "poses": _rel(transforms_path, base),
            "masks": _rel(masks, base),
            "splat": _rel(gaussian_ply, base) if gaussian_ply else None,
        },
        "consistency": {
            "object_field_matches_gaussians": (
                object_field_gaussian_count == gaussian_count
                if object_field_gaussian_count is not None and gaussian_count is not None
                else None
            ),
            "object_ply_matches_gaussians": (
                object_ply_count == gaussian_count
                if object_ply_count is not None and gaussian_count is not None
                else None
            ),
            "mask_slots_match_object_field": (
                len(slots) == object_field_slot_count
                if slots and object_field_slot_count is not None
                else None
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return SampleBundleResult(
        sample_path=output,
        sample_id=sample_id,
        asset_id=asset_id,
        image_count=len(frames),
        mask_frame_count=len(mask_frames),
        gaussian_count=gaussian_count,
        object_field_gaussian_count=object_field_gaussian_count,
        slot_count=int(sample["slot_count"]) if sample.get("slot_count") is not None else None,
    )


def _normalized_slots(mask_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    slots = mask_manifest.get("slots")
    if not isinstance(slots, list):
        return []
    normalized: list[dict[str, Any]] = []
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        slot_id = int(slot.get("slot", slot.get("slot_id")))
        normalized.append(
            {
                "slot_id": slot_id,
                "name": slot.get("name") or slot.get("label") or f"slot_{slot_id}",
                "type": slot.get("type") or "object",
            }
        )
    normalized.sort(key=lambda item: int(item["slot_id"]))
    return normalized


def _path_from_manifest(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def _rel(path: str | Path, base: Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        path = path.resolve()
    return os.path.relpath(path, base.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _nested(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
