from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from objgauss.features import positions
from objgauss.gaussians import GaussianCloud
from objgauss.object_field import ObjectField, object_field_metrics, softmax

_EPS = 1e-8


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
class MaskTrainingResult:
    field: ObjectField
    initial_loss: float
    final_loss: float
    iterations: int
    supervised_gaussians: int
    vote_summary: MaskVoteResult

    def as_dict(self) -> dict[str, float | int]:
        return {
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "iterations": self.iterations,
            "supervised_gaussians": self.supervised_gaussians,
            **self.vote_summary.as_dict(),
        }


def vote_masks_to_gaussians(
    cloud: GaussianCloud,
    manifest_path: str | Path,
    *,
    slots: int,
    max_frames: int | None = None,
    background_slot: int | None = None,
    background_weight: float = 1.0,
    visibility_mode: str = "projected",
    depth_tolerance: float = 0.0,
) -> MaskVoteResult:
    if slots < 1:
        raise ValueError("slots must be >= 1")
    visibility_mode = _validate_visibility_mode(visibility_mode)
    if not np.isfinite(depth_tolerance) or depth_tolerance < 0:
        raise ValueError("depth_tolerance must be a finite value >= 0")
    if background_slot is not None:
        if background_slot < 0 or background_slot >= slots:
            raise ValueError(f"background slot {background_slot} is outside [0, {slots})")
        if background_weight <= 0:
            raise ValueError("background_weight must be > 0")
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("mask manifest must contain a non-empty frames list")

    root = manifest_path.parent
    default_width = _optional_int(payload.get("width"))
    default_height = _optional_int(payload.get("height"))
    default_angle_x = _optional_float(payload.get("camera_angle_x"))
    xyz = positions(cloud)
    votes = np.zeros((cloud.count, slots), dtype=np.float32)
    observations = np.zeros(cloud.count, dtype=np.float32)
    projected_total = 0
    raw_projected_total = 0
    depth_visible_total = 0
    depth_culled_total = 0
    depth_culled_matched_total = 0
    matched_total = 0
    background_matched_total = 0
    used_frames = 0

    for frame in frames[:max_frames]:
        if not isinstance(frame, dict):
            raise ValueError("each mask frame must be an object")
        width = _required_int(frame.get("width", default_width), "frame width")
        height = _required_int(frame.get("height", default_height), "frame height")
        camera_angle_x = _required_float(
            frame.get("camera_angle_x", default_angle_x),
            "camera_angle_x",
        )
        transform = _required_matrix(frame.get("transform_matrix"))
        masks = frame.get("masks")
        if not isinstance(masks, list) or not masks:
            raise ValueError("each mask frame must contain a non-empty masks list")

        projection = project_points(
            xyz,
            transform_matrix=transform,
            width=width,
            height=height,
            camera_angle_x=camera_angle_x,
        )
        raw_visible = projection.visible
        visible = _visibility_mask(
            projection,
            width=width,
            height=height,
            mode=visibility_mode,
            depth_tolerance=depth_tolerance,
        )
        raw_projected = int(np.count_nonzero(raw_visible))
        depth_visible = int(np.count_nonzero(visible))
        raw_projected_total += raw_projected
        depth_visible_total += depth_visible
        depth_culled_total += int(np.count_nonzero(raw_visible & ~visible))
        projected_total += depth_visible
        frame_matched = np.zeros(cloud.count, dtype=bool)

        for mask in masks:
            if not isinstance(mask, dict):
                raise ValueError("mask entries must be objects")
            slot = _required_int(mask.get("slot"), "mask slot")
            if slot < 0 or slot >= slots:
                raise ValueError(f"mask slot {slot} is outside [0, {slots})")
            if background_slot is not None and slot == background_slot:
                raise ValueError("background slot must not be used by foreground masks")
            confidence = float(mask.get("confidence", 1.0))
            if confidence <= 0:
                continue
            contained = _mask_contains(mask, projection, width, height, root)
            if visibility_mode == "depth-buffer":
                depth_culled_matched_total += int(np.count_nonzero(raw_visible & contained & ~visible))
            selected = visible & contained
            if not np.any(selected):
                continue
            votes[selected, slot] += confidence
            observations[selected] += confidence
            frame_matched |= selected

        matched_total += int(np.count_nonzero(frame_matched))
        if background_slot is not None:
            background_selected = visible & ~frame_matched
            if np.any(background_selected):
                votes[background_selected, background_slot] += float(background_weight)
                observations[background_selected] += float(background_weight)
                background_matched_total += int(np.count_nonzero(background_selected))
        used_frames += 1

    return MaskVoteResult(
        votes=votes,
        observations=observations,
        frames=used_frames,
        projected=projected_total,
        matched=matched_total,
        background_slot=background_slot,
        background_weight=float(background_weight) if background_slot is not None else None,
        background_matched=background_matched_total,
        visibility_mode=visibility_mode,
        depth_tolerance=float(depth_tolerance),
        raw_projected=raw_projected_total,
        depth_visible=depth_visible_total,
        depth_culled=depth_culled_total,
        depth_culled_matched=depth_culled_matched_total,
    )


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

    homogeneous = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
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


def depth_visibility_diagnostic(
    cloud: GaussianCloud,
    manifest_path: str | Path,
    *,
    slots: int,
    max_frames: int | None = None,
    background_slot: int | None = None,
    background_weight: float = 1.0,
    depth_tolerance: float = 0.0,
) -> dict[str, Any]:
    baseline = vote_masks_to_gaussians(
        cloud,
        manifest_path,
        slots=slots,
        max_frames=max_frames,
        background_slot=background_slot,
        background_weight=background_weight,
        visibility_mode="projected",
    )
    depth_aware = vote_masks_to_gaussians(
        cloud,
        manifest_path,
        slots=slots,
        max_frames=max_frames,
        background_slot=background_slot,
        background_weight=background_weight,
        visibility_mode="depth-buffer",
        depth_tolerance=depth_tolerance,
    )
    baseline_summary = _vote_diagnostic_summary(baseline)
    depth_summary = _vote_diagnostic_summary(depth_aware)
    conflict_delta = (
        baseline_summary["vote_conflict_fraction"] - depth_summary["vote_conflict_fraction"]
    )
    balance_delta = depth_summary["slot_balance_score"] - baseline_summary["slot_balance_score"]
    supervised_delta = depth_summary["supervised_fraction"] - baseline_summary["supervised_fraction"]
    return {
        "kind": "objgauss-depth-visibility-vote-diagnostic-v1",
        "config": {
            "slots": int(slots),
            "max_frames": max_frames,
            "background_slot": background_slot,
            "background_weight": float(background_weight) if background_slot is not None else None,
            "depth_tolerance": float(depth_tolerance),
        },
        "baseline": baseline_summary,
        "depth_aware": depth_summary,
        "deltas": {
            "vote_conflict_fraction_reduction": float(conflict_delta),
            "slot_balance_score_delta": float(balance_delta),
            "supervised_fraction_delta": float(supervised_delta),
            "matched_delta": int(depth_aware.matched - baseline.matched),
            "depth_culled_matched": int(depth_aware.depth_culled_matched),
        },
        "recommendation": _depth_vote_recommendation(conflict_delta, balance_delta, depth_aware),
    }


def train_object_field_from_votes(
    field: ObjectField,
    vote_result: MaskVoteResult,
    *,
    iterations: int = 100,
    learning_rate: float = 0.5,
) -> MaskTrainingResult:
    if vote_result.votes.shape != field.logits.shape:
        raise ValueError(
            f"votes shape {vote_result.votes.shape} does not match field shape {field.logits.shape}"
        )
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    targets, weights = mask_vote_targets(vote_result)
    supervised = weights > 0
    if not np.any(supervised):
        raise ValueError("mask votes did not supervise any Gaussian")

    logits = field.logits.astype(np.float32, copy=True)
    initial_loss = projection_loss(logits, targets, weights)
    for _ in range(iterations):
        probabilities = softmax(logits, axis=1)
        gradient = (probabilities - targets) * weights[:, None]
        logits -= learning_rate * gradient.astype(np.float32, copy=False)
    trained = ObjectField(logits)
    final_loss = projection_loss(trained.logits, targets, weights)
    return MaskTrainingResult(
        field=trained,
        initial_loss=initial_loss,
        final_loss=final_loss,
        iterations=iterations,
        supervised_gaussians=int(np.count_nonzero(supervised)),
        vote_summary=vote_result,
    )


def training_summary(result: MaskTrainingResult) -> dict[str, Any]:
    metrics = object_field_metrics(result.field).as_dict()
    return {
        **result.as_dict(),
        "metrics": metrics,
    }


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
                "winner_gaussians": int(np.count_nonzero(supervised & (winners == slot))),
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


def mask_vote_quality_check(
    training: dict[str, Any],
    *,
    expected_slots: int | None = None,
) -> tuple[bool, str]:
    quality = training.get("vote_quality") if isinstance(training, dict) else None
    if not isinstance(quality, dict):
        return False, "missing vote_quality"
    per_slot = quality.get("per_slot")
    if not isinstance(per_slot, list) or not per_slot:
        return False, "missing per_slot coverage"
    if expected_slots is not None and len(per_slot) != expected_slots:
        return False, f"per_slot={len(per_slot)} expected={expected_slots}"
    supervised = int(quality.get("supervised_gaussians", 0) or 0)
    supervised_fraction = float(quality.get("supervised_fraction", 0.0) or 0.0)
    conflict = (
        quality.get("vote_conflict")
        if isinstance(quality.get("vote_conflict"), dict)
        else {}
    )
    conflict_fraction = float(conflict.get("fraction", 0.0) or 0.0)
    entropy = float(conflict.get("normalized_target_entropy", 0.0) or 0.0)
    ok = supervised > 0 and supervised_fraction > 0.0
    detail = (
        f"supervised_gaussians={supervised} "
        f"supervised_fraction={supervised_fraction:.6f} "
        f"conflict_fraction={conflict_fraction:.6f} "
        f"normalized_target_entropy={entropy:.6f} "
        f"slots={len(per_slot)}"
    )
    return ok, detail


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


def _validate_visibility_mode(mode: str) -> str:
    if mode not in {"projected", "depth-buffer"}:
        raise ValueError("visibility_mode must be 'projected' or 'depth-buffer'")
    return mode


def _visibility_mask(
    projection: Projection,
    *,
    width: int,
    height: int,
    mode: str,
    depth_tolerance: float,
) -> np.ndarray:
    if mode == "projected":
        return projection.visible
    if mode == "depth-buffer":
        return _depth_buffer_visible(
            projection,
            width=width,
            height=height,
            depth_tolerance=depth_tolerance,
        )
    raise ValueError("visibility mode must be 'projected' or 'depth-buffer'")


def _depth_buffer_visible(
    projection: Projection,
    *,
    width: int,
    height: int,
    depth_tolerance: float,
) -> np.ndarray:
    visible_indices = np.flatnonzero(projection.visible)
    result = np.zeros_like(projection.visible, dtype=bool)
    if visible_indices.size == 0:
        return result

    x = np.clip(np.floor(projection.u[visible_indices]).astype(np.int64), 0, width - 1)
    y = np.clip(np.floor(projection.v[visible_indices]).astype(np.int64), 0, height - 1)
    pixel = y * width + x
    depth = projection.depth[visible_indices]
    min_depth = np.full(width * height, np.inf, dtype=np.float32)
    np.minimum.at(min_depth, pixel, depth)
    result[visible_indices] = depth <= (min_depth[pixel] + float(depth_tolerance))
    return result


def _vote_diagnostic_summary(vote_result: MaskVoteResult) -> dict[str, Any]:
    quality = mask_vote_quality_audit(vote_result)
    conflict = quality["vote_conflict"]
    slot_balance = _slot_balance_summary(quality)
    return {
        "frames": int(vote_result.frames),
        "projected": int(vote_result.projected),
        "matched": int(vote_result.matched),
        "supervised_gaussians": int(quality["supervised_gaussians"]),
        "supervised_fraction": float(quality["supervised_fraction"]),
        "matched_projected_fraction": float(quality["matched_projected_fraction"]),
        "vote_conflict_gaussians": int(conflict["gaussians"]),
        "vote_conflict_fraction": float(conflict["fraction"]),
        "normalized_target_entropy": float(conflict["normalized_target_entropy"]),
        "target_confidence_mean": float(quality["target_confidence"]["mean"]),
        "slot_balance": slot_balance,
        "slot_balance_score": float(slot_balance["score"]),
        "per_slot": quality["per_slot"],
        "visibility": vote_result.as_dict()["visibility"],
    }


def _slot_balance_summary(quality: dict[str, Any]) -> dict[str, Any]:
    per_slot = quality.get("per_slot")
    if not isinstance(per_slot, list):
        return {"score": 0.0, "active_slots": 0, "min_winners": 0, "max_winners": 0}
    winners = [int(slot.get("winner_gaussians", 0) or 0) for slot in per_slot]
    active = [count for count in winners if count > 0]
    if not active:
        return {
            "score": 0.0,
            "active_slots": 0,
            "min_winners": 0,
            "max_winners": 0,
            "winner_gaussians": winners,
        }
    min_winners = min(active)
    max_winners = max(active)
    return {
        "score": 0.0 if max_winners <= 0 else float(min_winners / max_winners),
        "active_slots": int(len(active)),
        "min_winners": int(min_winners),
        "max_winners": int(max_winners),
        "winner_gaussians": winners,
    }


def _depth_vote_recommendation(
    conflict_delta: float,
    balance_delta: float,
    depth_aware: MaskVoteResult,
) -> str:
    if depth_aware.depth_culled_matched <= 0:
        return "no-depth-occlusion-signal"
    if conflict_delta > 0.0 or balance_delta > 0.0:
        return "depth-aware-diagnostic-improved"
    return "depth-aware-diagnostic-review"


def mask_vote_targets(vote_result: MaskVoteResult) -> tuple[np.ndarray, np.ndarray]:
    vote_sum = vote_result.votes.sum(axis=1)
    supervised = vote_sum > 0
    targets = np.zeros_like(vote_result.votes, dtype=np.float32)
    targets[supervised] = vote_result.votes[supervised] / vote_sum[supervised, None]
    weights = np.zeros(vote_result.votes.shape[0], dtype=np.float32)
    weights[supervised] = vote_sum[supervised] / max(float(np.max(vote_sum[supervised])), _EPS)
    return targets, weights


def projection_loss(logits: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
    supervised = weights > 0
    probabilities = softmax(logits, axis=1)
    cross_entropy = -np.sum(targets * np.log(np.clip(probabilities, _EPS, 1.0)), axis=1)
    return float(np.sum(cross_entropy[supervised] * weights[supervised]) / np.sum(weights[supervised]))


def _mask_contains(
    mask: dict[str, Any],
    projection: Projection,
    width: int,
    height: int,
    root: Path,
) -> np.ndarray:
    if "rect" in mask:
        x0, y0, x1, y1 = _required_rect(mask["rect"])
        return (projection.u >= x0) & (projection.u < x1) & (projection.v >= y0) & (projection.v < y1)
    if "mask_path" in mask:
        mask_path = root / str(mask["mask_path"])
        mask_array = np.load(mask_path)
        if mask_array.shape != (height, width):
            raise ValueError(f"mask {mask_path} shape {mask_array.shape} does not match {height}x{width}")
        x = np.clip(np.floor(projection.u).astype(np.int64), 0, width - 1)
        y = np.clip(np.floor(projection.v).astype(np.int64), 0, height - 1)
        return mask_array[y, x].astype(bool, copy=False)
    raise ValueError("mask entry must include rect or mask_path")


def _required_rect(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError("mask rect must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (float(part) for part in value)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("mask rect must have x1 > x0 and y1 > y0")
    return x0, y0, x1, y1


def _required_matrix(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform_matrix must be a finite 4x4 matrix")
    return matrix


def _required_int(value: object, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    return int(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _required_float(value: object, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    return float(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
