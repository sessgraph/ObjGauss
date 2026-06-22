from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import Any

import numpy as np

from objgauss.features import colors
from objgauss.gaussians import GaussianCloud
from objgauss.mask_voting import (
    train_object_field_from_votes,
    training_summary,
    vote_masks_to_gaussians,
)
from objgauss.object_field import ObjectField, object_field_label_delta, save_object_field
from objgauss.ply import append_or_replace_property, read_ply, write_ply
from objgauss.segment import apply_object_colors, assign_object_ids
from objgauss.splat import read_splat, write_splat


@dataclass(frozen=True)
class TrainingOutputRegistration:
    manifest_path: Path
    gaussian_ply_path: Path
    splat_path: Path
    object_field_path: Path | None
    object_ply_path: Path | None
    public_splat_path: Path | None
    public_object_ply_path: Path | None
    gaussian_count: int
    slots: int | None
    supervised_gaussians: int | None
    initial_loss: float | None
    final_loss: float | None


def register_training_output(
    input_path: str | Path,
    *,
    output_dir: str | Path,
    asset_id: str,
    dataset: str | Path | None = None,
    masks: str | Path | None = None,
    slots: int | None = None,
    public_dir: str | Path | None = "public/samples",
    public_name: str | None = None,
    iterations: int = 100,
    learning_rate: float = 0.5,
    colorize: bool = True,
) -> TrainingOutputRegistration:
    if not asset_id:
        raise ValueError("asset_id is required")
    if slots is not None and slots < 1:
        raise ValueError("slots must be >= 1")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cloud, input_format = _read_gaussian_output(input_path)
    cloud = _with_rgb_properties(cloud)
    cloud.require_fields(("x", "y", "z", "red", "green", "blue"))

    gaussian_ply_path = output_dir / "gaussians.ply"
    splat_path = output_dir / "gaussians.splat"
    write_ply(gaussian_ply_path, cloud, fmt="binary_little_endian")
    write_splat(splat_path, cloud)

    object_field_initial_path = None
    object_field_path = None
    object_ply_path = None
    training = None
    field_delta = None
    inferred_slots = slots
    if masks is not None:
        masks = Path(masks)
        inferred_slots = slots or _slots_from_mask_manifest(masks)
        field = ObjectField(np.zeros((cloud.count, inferred_slots), dtype=np.float32))
        object_field_initial_path = output_dir / "object_field_initial.npz"
        object_field_path = output_dir / "object_field_trained.npz"
        save_object_field(object_field_initial_path, field)
        votes = vote_masks_to_gaussians(cloud, masks, slots=inferred_slots)
        training = train_object_field_from_votes(
            field,
            votes,
            iterations=iterations,
            learning_rate=learning_rate,
        )
        field_delta = object_field_label_delta(field, training.field)
        save_object_field(object_field_path, training.field)
        labeled = assign_object_ids(cloud, training.field.labels())
        if colorize:
            labeled = apply_object_colors(labeled)
        object_ply_path = output_dir / "object_aware_gaussians.ply"
        write_ply(object_ply_path, labeled, fmt="binary_little_endian")
        summary_path = output_dir / "mask-training-summary.json"
        summary_path.write_text(
            json.dumps(training_summary(training), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    public_splat_path = None
    public_object_ply_path = None
    if public_dir is not None:
        public_dir = Path(public_dir)
        public_dir.mkdir(parents=True, exist_ok=True)
        public_stem = public_name or asset_id.replace("-", "_")
        public_splat_path = public_dir / f"{public_stem}.splat"
        copyfile(splat_path, public_splat_path)
        if object_ply_path is not None:
            public_object_ply_path = public_dir / f"{public_stem}_objects.ply"
            copyfile(object_ply_path, public_object_ply_path)

    manifest_path = output_dir / "training-output-manifest.json"
    manifest = {
        "asset_id": asset_id,
        "dataset": str(dataset) if dataset is not None else None,
        "input": str(input_path),
        "input_format": input_format,
        "gaussian_source": "external_3dgs_training_output",
        "note": (
            "This command registers an output produced by an external 3DGS trainer; "
            "it does not train 3DGS inside ObjGauss."
        ),
        "gaussian_count": cloud.count,
        "fields": list(cloud.fields),
        "gaussian_ply": str(gaussian_ply_path),
        "splat_path": str(splat_path),
        "mask_manifest": str(masks) if masks is not None else None,
        "slots": inferred_slots,
        "initial_field": str(object_field_initial_path) if object_field_initial_path else None,
        "trained_field": str(object_field_path) if object_field_path else None,
        "object_ply": str(object_ply_path) if object_ply_path else None,
        "public_splat": str(public_splat_path) if public_splat_path else None,
        "public_object_ply": str(public_object_ply_path) if public_object_ply_path else None,
        "training": training_summary(training) if training is not None else None,
        "object_field_delta": field_delta.as_dict() if field_delta is not None else None,
        "acceptance": {
            "external_gaussian_loaded": cloud.count > 0,
            "viewer_splat_available": splat_path.exists(),
            "object_field_trained": training is not None,
            "mask_votes_supervise_gaussians": (
                training.supervised_gaussians > 0 if training is not None else False
            ),
            "mask_vote_quality_audit_available": training is not None,
            "mask_guidance_changed_object_field": (
                field_delta.changed_gaussians > 0 if field_delta is not None else False
            ),
            "projection_loss_decreased": (
                training.final_loss < training.initial_loss if training is not None else False
            ),
            "object_id_ply_available": object_ply_path is not None and object_ply_path.exists(),
            "public_assets_available": (
                public_splat_path is not None
                and public_splat_path.exists()
                and (
                    public_object_ply_path is None
                    or public_object_ply_path.exists()
                )
            ),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return TrainingOutputRegistration(
        manifest_path=manifest_path,
        gaussian_ply_path=gaussian_ply_path,
        splat_path=splat_path,
        object_field_path=object_field_path,
        object_ply_path=object_ply_path,
        public_splat_path=public_splat_path,
        public_object_ply_path=public_object_ply_path,
        gaussian_count=cloud.count,
        slots=inferred_slots,
        supervised_gaussians=training.supervised_gaussians if training else None,
        initial_loss=training.initial_loss if training else None,
        final_loss=training.final_loss if training else None,
    )


def _read_gaussian_output(path: Path) -> tuple[GaussianCloud, str]:
    if not path.exists():
        raise ValueError(f"missing training output: {path}")
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return read_ply(path), "ply"
    if suffix == ".splat":
        return read_splat(path), "splat"
    raise ValueError("training output must be a .ply or .splat file")


def _with_rgb_properties(cloud: GaussianCloud) -> GaussianCloud:
    rgb = np.clip(colors(cloud) * 255.0, 0, 255).astype(np.uint8)
    vertices = cloud.vertices
    for channel, name in enumerate(("red", "green", "blue")):
        vertices = append_or_replace_property(vertices, name, rgb[:, channel], "u1")
    return cloud.with_vertices(vertices)


def _slots_from_mask_manifest(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    max_slot = -1
    slots = payload.get("slots")
    if isinstance(slots, list):
        for slot in slots:
            if isinstance(slot, dict) and "slot" in slot:
                max_slot = max(max_slot, int(slot["slot"]))
    frames = payload.get("frames")
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            masks = frame.get("masks")
            if not isinstance(masks, list):
                continue
            for mask in masks:
                if isinstance(mask, dict) and "slot" in mask:
                    max_slot = max(max_slot, int(mask["slot"]))
    if max_slot < 0:
        raise ValueError("mask manifest does not declare any slots")
    return max_slot + 1
