from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from objgauss.core.object_field import softmax

_EPS = 1e-8
_COMPAT_EXPORTS = {
    "MaskTrainingResult": ("objgauss.pipelines.mask_voting", "MaskTrainingResult"),
    "depth_visibility_diagnostic": (
        "objgauss.pipelines.mask_voting",
        "depth_visibility_diagnostic",
    ),
    "mask_vote_quality_check": (
        "objgauss.evaluation.mask_vote_quality",
        "mask_vote_quality_check",
    ),
    "train_object_field_from_votes": (
        "objgauss.pipelines.mask_voting",
        "train_object_field_from_votes",
    ),
    "training_summary": ("objgauss.pipelines.mask_voting", "training_summary"),
    "vote_masks_to_gaussians": (
        "objgauss.pipelines.mask_voting",
        "vote_masks_to_gaussians",
    ),
}

__all__ = (
    "MaskTrainingResult",
    "MaskVoteResult",
    "Projection",
    "depth_visibility_diagnostic",
    "mask_vote_quality_audit",
    "mask_vote_quality_check",
    "mask_vote_targets",
    "project_points",
    "projection_loss",
    "train_object_field_from_votes",
    "training_summary",
    "vote_masks_to_gaussians",
)


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _COMPAT_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


@dataclass(frozen=True)
class MaskVoteResult:
    votes: np.ndarray
    observations: np.ndarray
    frames: int
    projected: int
    matched: int
    background_slot: int | None = None
    background_weight: float | None = None
    background_matched: int = 0
    visibility_mode: str = "projected"
    depth_tolerance: float = 0.0
    raw_projected: int = 0
    depth_visible: int = 0
    depth_culled: int = 0
    depth_culled_matched: int = 0

    @property
    def supervised_gaussians(self) -> int:
        return int(np.count_nonzero(self.observations > 0))

    def as_dict(self) -> dict[str, Any]:
        result = {
            "frames": self.frames,
            "projected": self.projected,
            "matched": self.matched,
            "supervised_gaussians": self.supervised_gaussians,
            "visibility": {
                "mode": self.visibility_mode,
                "depth_tolerance": float(self.depth_tolerance),
                "raw_projected": int(self.raw_projected or self.projected),
                "depth_visible": int(self.depth_visible or self.projected),
                "depth_culled": int(self.depth_culled),
                "depth_culled_matched": int(self.depth_culled_matched),
            },
            "vote_quality": mask_vote_quality_audit(self),
        }
        if self.background_slot is not None:
            result["background_training"] = {
                "type": "projected_unmatched_mask_vote",
                "slot": int(self.background_slot),
                "weight": float(self.background_weight or 0.0),
                "matched": int(self.background_matched),
                "matched_projected_fraction": _safe_fraction(
                    int(self.background_matched),
                    int(self.projected),
                ),
            }
        return result


@dataclass(frozen=True)
class Projection:
    u: np.ndarray
    v: np.ndarray
    visible: np.ndarray
    depth: np.ndarray


def project_points(
    xyz: np.ndarray,
    *,
    transform_matrix: np.ndarray,
    width: int,
    height: int,
    camera_angle_x: float,
    near: float = 1e-4,
) -> Projection:
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be an Nx3 array")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    c2w = np.asarray(transform_matrix, dtype=np.float32)
    if c2w.shape != (4, 4):
        raise ValueError("transform_matrix must be 4x4")

    homogeneous = np.concatenate(
        [xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1
    )
    world_to_camera = np.linalg.inv(c2w)
    camera = homogeneous @ world_to_camera.T
    forward = -camera[:, 2]
    focal = 0.5 * width / np.tan(0.5 * camera_angle_x)
    u = focal * (camera[:, 0] / np.maximum(forward, near)) + width * 0.5
    v = height * 0.5 - focal * (camera[:, 1] / np.maximum(forward, near))
    visible = (
        (forward > near)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
        & np.isfinite(u)
        & np.isfinite(v)
    )
    return Projection(
        u=u.astype(np.float32),
        v=v.astype(np.float32),
        visible=visible,
        depth=forward.astype(np.float32, copy=False),
    )


def mask_vote_quality_audit(vote_result: MaskVoteResult) -> dict[str, Any]:
    votes = np.asarray(vote_result.votes, dtype=np.float32)
    if votes.ndim != 2:
        raise ValueError("votes must be an NxK array")
    gaussian_count, slots = votes.shape
    vote_sum = votes.sum(axis=1)
    supervised = vote_sum > 0
    supervised_count = int(np.count_nonzero(supervised))
    unsupervised_count = int(gaussian_count - supervised_count)
    supervised_fraction = _safe_fraction(supervised_count, gaussian_count)

    observations = np.asarray(vote_result.observations, dtype=np.float32)
    observed_weights = (
        observations[supervised]
        if observations.shape == (gaussian_count,)
        else vote_sum[supervised]
    )
    slot_support = votes > 0
    support_counts = slot_support.sum(axis=1)
    conflicted = supervised & (support_counts > 1)
    conflict_count = int(np.count_nonzero(conflicted))

    target_entropy = 0.0
    normalized_target_entropy = 0.0
    target_confidence_mean = 0.0
    target_confidence_min = 0.0
    winners = np.zeros(gaussian_count, dtype=np.int64)
    targets = np.zeros_like(votes, dtype=np.float32)
    if supervised_count > 0:
        targets[supervised] = votes[supervised] / vote_sum[supervised, None]
        entropy_per_gaussian = -np.sum(
            targets[supervised] * np.log(np.clip(targets[supervised], _EPS, 1.0)),
            axis=1,
        )
        target_entropy = float(np.mean(entropy_per_gaussian))
        normalized_target_entropy = (
            0.0 if slots <= 1 else _clamp_unit(float(target_entropy / np.log(slots)))
        )
        target_confidence = targets[supervised].max(axis=1)
        target_confidence_mean = float(np.mean(target_confidence))
        target_confidence_min = float(np.min(target_confidence))
        winners = np.argmax(votes, axis=1)

    return {
        "gaussian_count": int(gaussian_count),
        "slots": int(slots),
        "supervised_gaussians": supervised_count,
        "unsupervised_gaussians": unsupervised_count,
        "supervised_fraction": supervised_fraction,
        "projected": int(vote_result.projected),
        "matched": int(vote_result.matched),
        "matched_projected_fraction": _safe_fraction(
            int(vote_result.matched),
            int(vote_result.projected),
        ),
        "observation_weight": _weight_stats(observed_weights),
        "vote_conflict": {
            "gaussians": conflict_count,
            "fraction": _safe_fraction(conflict_count, supervised_count),
            "target_entropy": target_entropy,
            "normalized_target_entropy": normalized_target_entropy,
        },
        "target_confidence": {
            "mean": target_confidence_mean,
            "min": target_confidence_min,
        },
        "per_slot": [
            {
                "slot": int(slot),
                "vote_weight": float(np.sum(votes[:, slot])),
                "supervised_gaussians": int(np.count_nonzero(slot_support[:, slot])),
                "winner_gaussians": int(
                    np.count_nonzero(supervised & (winners == slot))
                ),
                "supervised_fraction": _safe_fraction(
                    int(np.count_nonzero(slot_support[:, slot])),
                    gaussian_count,
                ),
                "winner_fraction": _safe_fraction(
                    int(np.count_nonzero(supervised & (winners == slot))),
                    supervised_count,
                ),
            }
            for slot in range(slots)
        ],
    }


def _weight_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _safe_fraction(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator / denominator)


def _clamp_unit(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def mask_vote_targets(vote_result: MaskVoteResult) -> tuple[np.ndarray, np.ndarray]:
    vote_sum = vote_result.votes.sum(axis=1)
    supervised = vote_sum > 0
    targets = np.zeros_like(vote_result.votes, dtype=np.float32)
    targets[supervised] = vote_result.votes[supervised] / vote_sum[supervised, None]
    weights = np.zeros(vote_result.votes.shape[0], dtype=np.float32)
    weights[supervised] = vote_sum[supervised] / max(
        float(np.max(vote_sum[supervised])), _EPS
    )
    return targets, weights


def projection_loss(logits: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
    supervised = weights > 0
    probabilities = softmax(logits, axis=1)
    cross_entropy = -np.sum(
        targets * np.log(np.clip(probabilities, _EPS, 1.0)), axis=1
    )
    return float(
        np.sum(cross_entropy[supervised] * weights[supervised])
        / np.sum(weights[supervised])
    )
