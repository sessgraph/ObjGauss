from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = (
    "NerfDatasetSummary",
    "NerfSplitSummary",
    "inspect_nerf_dataset",
)


@dataclass(frozen=True)
class NerfSplitSummary:
    name: str
    frames: int
    missing_images: int
    invalid_transforms: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "frames": self.frames,
            "missing_images": self.missing_images,
            "invalid_transforms": self.invalid_transforms,
        }


@dataclass(frozen=True)
class NerfDatasetSummary:
    root: Path
    splits: tuple[NerfSplitSummary, ...]

    @property
    def total_frames(self) -> int:
        return sum(split.frames for split in self.splits)

    @property
    def missing_images(self) -> int:
        return sum(split.missing_images for split in self.splits)

    @property
    def invalid_transforms(self) -> int:
        return sum(split.invalid_transforms for split in self.splits)

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "splits": [split.as_dict() for split in self.splits],
            "total_frames": self.total_frames,
            "missing_images": self.missing_images,
            "invalid_transforms": self.invalid_transforms,
        }


def inspect_nerf_dataset(root: str | Path) -> NerfDatasetSummary:
    root = Path(root)
    if not root.exists():
        raise ValueError(f"NeRF dataset path does not exist: {root}")

    splits: list[NerfSplitSummary] = []
    for path in sorted(root.glob("transforms_*.json")):
        split_name = path.stem.removeprefix("transforms_")
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames = payload.get("frames", [])
        if not isinstance(frames, list):
            raise ValueError(f"{path} has no frames list")

        missing_images = 0
        invalid_transforms = 0
        for frame in frames:
            if not isinstance(frame, dict):
                invalid_transforms += 1
                continue
            if not _has_valid_transform(frame.get("transform_matrix")):
                invalid_transforms += 1
            file_path = frame.get("file_path")
            if not isinstance(file_path, str) or not _frame_image_exists(root, file_path):
                missing_images += 1
        splits.append(
            NerfSplitSummary(
                name=split_name,
                frames=len(frames),
                missing_images=missing_images,
                invalid_transforms=invalid_transforms,
            )
        )

    if not splits:
        raise ValueError(f"no transforms_*.json files found under {root}")
    return NerfDatasetSummary(root=root, splits=tuple(splits))


def _frame_image_exists(root: Path, file_path: str) -> bool:
    raw = file_path[2:] if file_path.startswith("./") else file_path
    candidate = root / raw
    if candidate.exists():
        return True
    if candidate.suffix:
        return False
    return any(candidate.with_suffix(suffix).exists() for suffix in (".png", ".jpg", ".jpeg"))


def _has_valid_transform(value: object) -> bool:
    try:
        matrix = np.asarray(value, dtype=np.float32)
    except Exception:
        return False
    return matrix.shape == (4, 4) and bool(np.isfinite(matrix).all())
