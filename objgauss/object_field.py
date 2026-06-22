from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from objgauss.clustering import ClusteringResult, cluster_features, summarize_labels
from objgauss.features import extract_features, positions
from objgauss.gaussians import GaussianCloud
from objgauss.segment import assign_object_ids

_EPS = 1e-8
_FIELD_VERSION = 1


@dataclass(frozen=True)
class ObjectField:
    """Soft object-slot assignment over a Gaussian cloud."""

    logits: np.ndarray

    def __post_init__(self) -> None:
        logits = np.asarray(self.logits, dtype=np.float32)
        if logits.ndim != 2:
            raise ValueError("object field logits must be a 2D array")
        if logits.shape[0] == 0:
            raise ValueError("object field needs at least one Gaussian")
        if logits.shape[1] == 0:
            raise ValueError("object field needs at least one object slot")
        object.__setattr__(self, "logits", logits)

    @property
    def gaussian_count(self) -> int:
        return int(self.logits.shape[0])

    @property
    def slots(self) -> int:
        return int(self.logits.shape[1])

    def probabilities(self) -> np.ndarray:
        return softmax(self.logits, axis=1)

    def labels(self) -> np.ndarray:
        return np.argmax(self.logits, axis=1).astype(np.int32, copy=False)


@dataclass(frozen=True)
class ObjectFieldInit:
    field: ObjectField
    clustering: ClusteringResult


@dataclass(frozen=True)
class ObjectFieldMetrics:
    entropy: float
    normalized_entropy: float
    sharpness: float
    active_slots: int
    smoothness: float | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "entropy": self.entropy,
            "normalized_entropy": self.normalized_entropy,
            "sharpness": self.sharpness,
            "active_slots": self.active_slots,
            "smoothness": self.smoothness,
        }


@dataclass(frozen=True)
class ObjectFieldLabelDelta:
    changed_gaussians: int
    changed_fraction: float
    initial_active_slots: int
    trained_active_slots: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "changed_gaussians": self.changed_gaussians,
            "changed_fraction": self.changed_fraction,
            "initial_active_slots": self.initial_active_slots,
            "trained_active_slots": self.trained_active_slots,
        }


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


def initialize_object_field(
    cloud: GaussianCloud,
    *,
    slots: int,
    seed: int = 0,
    max_iter: int = 100,
    confidence: float = 0.92,
    spatial_weight: float = 1.0,
    color_weight: float = 0.5,
    opacity_weight: float = 0.2,
    normalize: bool = True,
) -> ObjectFieldInit:
    features = extract_features(
        cloud,
        spatial_weight=spatial_weight,
        color_weight=color_weight,
        opacity_weight=opacity_weight,
        normalize=normalize,
    )
    clustering = cluster_features(features, clusters=slots, seed=seed, max_iter=max_iter)
    field = field_from_labels(clustering.labels, slots=slots, confidence=confidence)
    return ObjectFieldInit(field=field, clustering=clustering)


def field_from_labels(
    labels: np.ndarray,
    *,
    slots: int | None = None,
    confidence: float = 0.92,
) -> ObjectField:
    labels = np.asarray(labels, dtype=np.int32)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if labels.shape[0] == 0:
        raise ValueError("cannot initialize an object field from empty labels")
    if np.any(labels < 0):
        raise ValueError("object labels must be non-negative")
    slots = int(labels.max()) + 1 if slots is None else int(slots)
    if slots < 1:
        raise ValueError("slots must be >= 1")
    if int(labels.max()) >= slots:
        raise ValueError("labels cannot exceed slot count")
    if not 0.0 < confidence <= 1.0:
        raise ValueError("confidence must be in (0, 1]")

    if slots == 1:
        probabilities = np.ones((labels.shape[0], 1), dtype=np.float32)
    else:
        rest = (1.0 - confidence) / (slots - 1)
        probabilities = np.full((labels.shape[0], slots), rest, dtype=np.float32)
        probabilities[np.arange(labels.shape[0]), labels] = confidence
    return ObjectField(np.log(np.clip(probabilities, _EPS, 1.0)).astype(np.float32))


def softmax(values: np.ndarray, *, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def attach_hard_labels(
    cloud: GaussianCloud,
    field: ObjectField,
    *,
    object_id_field: str = "object_id",
) -> GaussianCloud:
    if cloud.count != field.gaussian_count:
        raise ValueError(
            f"field has {field.gaussian_count} gaussians for cloud with {cloud.count}"
        )
    if object_id_field != "object_id":
        vertices = _assign_custom_object_ids(cloud, field.labels(), object_id_field)
        return cloud.with_vertices(vertices)
    return assign_object_ids(cloud, field.labels())


def object_field_metrics(
    field: ObjectField,
    *,
    positions_xyz: np.ndarray | None = None,
    neighbors: int = 4,
    max_smooth_points: int = 1024,
) -> ObjectFieldMetrics:
    probabilities = field.probabilities()
    entropy_per_gaussian = -np.sum(probabilities * np.log(np.clip(probabilities, _EPS, 1.0)), axis=1)
    entropy = float(np.mean(entropy_per_gaussian))
    normalized_entropy = 0.0 if field.slots == 1 else float(entropy / np.log(field.slots))
    sharpness = float(np.mean(probabilities * (1.0 - probabilities)))
    active_slots = len(summarize_labels(field.labels()))
    smoothness = None
    if positions_xyz is not None:
        smoothness = local_smoothness_loss(
            probabilities,
            positions_xyz,
            neighbors=neighbors,
            max_points=max_smooth_points,
        )
    return ObjectFieldMetrics(
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        sharpness=sharpness,
        active_slots=active_slots,
        smoothness=smoothness,
    )


def object_field_label_delta(
    initial: ObjectField,
    trained: ObjectField,
) -> ObjectFieldLabelDelta:
    if initial.logits.shape != trained.logits.shape:
        raise ValueError(
            f"object field shapes differ: {initial.logits.shape} vs {trained.logits.shape}"
        )
    initial_labels = initial.labels()
    trained_labels = trained.labels()
    changed = int(np.count_nonzero(initial_labels != trained_labels))
    return ObjectFieldLabelDelta(
        changed_gaussians=changed,
        changed_fraction=float(changed / max(initial.gaussian_count, 1)),
        initial_active_slots=len(summarize_labels(initial_labels)),
        trained_active_slots=len(summarize_labels(trained_labels)),
    )


def local_smoothness_loss(
    probabilities: np.ndarray,
    positions_xyz: np.ndarray,
    *,
    neighbors: int = 4,
    max_points: int = 1024,
) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    positions_xyz = np.asarray(positions_xyz, dtype=np.float32)
    if probabilities.ndim != 2 or positions_xyz.ndim != 2 or positions_xyz.shape[1] != 3:
        raise ValueError("probabilities must be NxK and positions must be Nx3")
    if probabilities.shape[0] != positions_xyz.shape[0]:
        raise ValueError("probabilities and positions must have the same row count")
    if probabilities.shape[0] <= 1:
        return 0.0

    count = min(probabilities.shape[0], max_points)
    sample_indices = _even_sample(probabilities.shape[0], count)
    sampled_positions = positions_xyz[sample_indices]
    sampled_probabilities = probabilities[sample_indices]
    k = min(max(1, neighbors), count - 1)

    distances = _squared_distances(sampled_positions)
    nearest = np.argsort(distances, axis=1)[:, 1 : k + 1]
    diffs = sampled_probabilities[:, None, :] - sampled_probabilities[nearest]
    return float(np.mean(np.sum(diffs * diffs, axis=2)))


def save_object_field(path: str | Path, field: ObjectField) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez_compressed(
            file,
            version=np.array([_FIELD_VERSION], dtype=np.int32),
            logits=field.logits,
        )


def load_object_field(path: str | Path) -> ObjectField:
    with np.load(Path(path)) as payload:
        version = int(payload["version"][0]) if "version" in payload else 0
        if version != _FIELD_VERSION:
            raise ValueError(f"unsupported object field version: {version}")
        return ObjectField(payload["logits"])


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


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cloud_positions_for_metrics(cloud: GaussianCloud) -> np.ndarray:
    return positions(cloud)


def _assign_custom_object_ids(
    cloud: GaussianCloud,
    labels: np.ndarray,
    object_id_field: str,
) -> np.ndarray:
    from objgauss.ply import append_or_replace_property

    return append_or_replace_property(cloud.vertices, object_id_field, labels, np.int32)


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


def _even_sample(total: int, count: int) -> np.ndarray:
    if count >= total:
        return np.arange(total)
    return np.linspace(0, total - 1, count, dtype=np.int64)


def _squared_distances(values: np.ndarray) -> np.ndarray:
    diff = values[:, None, :] - values[None, :, :]
    return np.sum(diff * diff, axis=2)
