from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from objgauss.clustering import summarize_labels
from objgauss.gaussians import GaussianCloud
from objgauss.mask_voting import MaskVoteResult, mask_vote_targets, projection_loss
from objgauss.object_field import ObjectField, softmax
from objgauss.render_probe import RenderProbeFrame, render_occlusion_delta

_EPS = 1e-8
_RENDER_OCCLUSION_KIND = "scale_aware_cpu_splat_l1"


def object_emergence_metrics(
    field: ObjectField,
    *,
    positions_xyz: np.ndarray | None = None,
    reference: ObjectField | None = None,
) -> dict[str, Any]:
    """Compute conservative object-emergence observability metrics.

    This intentionally reports a partial Object Emergence Score. Current ObjGauss
    can measure assignment, spatial compactness, and reference stability; true
    occlusion/render-delta and gradient-coherence probes are left explicit.
    """

    probabilities = field.probabilities()
    labels = field.labels()
    assignment = _assignment_metrics(probabilities, labels)
    spatial = _spatial_metrics(labels, positions_xyz, field.slots) if positions_xyz is not None else None
    stability = _stability_metrics(reference, field) if reference is not None else None
    score = _object_emergence_score(
        assignment_confidence=assignment["assignment_confidence"],
        spatial_compactness_score=spatial["compactness_score"] if spatial else None,
        stability_score=stability["stability_score"] if stability else None,
    )
    return {
        "gaussians": field.gaussian_count,
        "slots": field.slots,
        "assignment": assignment,
        "spatial": spatial,
        "stability": stability,
        "object_emergence_score": score,
    }


def object_emergence_curve(
    field: ObjectField,
    vote_result: MaskVoteResult,
    *,
    positions_xyz: np.ndarray | None = None,
    cloud: GaussianCloud | None = None,
    render_frames: list[RenderProbeFrame] | None = None,
    heldout_vote_result: MaskVoteResult | None = None,
    heldout_render_frames: list[RenderProbeFrame] | None = None,
    iterations: int = 100,
    learning_rate: float = 0.5,
    eval_every: int = 10,
) -> dict[str, Any]:
    if vote_result.votes.shape != field.logits.shape:
        raise ValueError(
            f"votes shape {vote_result.votes.shape} does not match field shape {field.logits.shape}"
        )
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if eval_every < 1:
        raise ValueError("eval_every must be >= 1")

    targets, weights = mask_vote_targets(vote_result)
    if not np.any(weights > 0):
        raise ValueError("mask votes did not supervise any Gaussian")
    heldout_targets = None
    heldout_weights = None
    if heldout_vote_result is not None:
        if heldout_vote_result.votes.shape != field.logits.shape:
            raise ValueError(
                f"heldout votes shape {heldout_vote_result.votes.shape} does not match field shape {field.logits.shape}"
            )
        heldout_targets, heldout_weights = mask_vote_targets(heldout_vote_result)

    logits = field.logits.astype(np.float32, copy=True)
    initial_labels = field.labels()
    previous_snapshot: ObjectField | None = None
    points: list[dict[str, Any]] = []

    def record(step: int) -> None:
        nonlocal previous_snapshot
        current = ObjectField(logits.copy())
        render_occlusion = (
            render_occlusion_delta(cloud, current, render_frames)
            if cloud is not None and render_frames
            else None
        )
        metrics = object_emergence_metrics(
            current,
            positions_xyz=positions_xyz,
            reference=previous_snapshot,
        )
        labels = current.labels()
        point = _curve_point(
            step=step,
            field=current,
            metrics=metrics,
            projection_loss_value=projection_loss(current.logits, targets, weights),
            mask_proxy_occlusion=mask_proxy_occlusion_delta(current, targets, weights),
            render_occlusion=render_occlusion,
            heldout=_heldout_curve_eval(
                current,
                vote_result=heldout_vote_result,
                targets=heldout_targets,
                weights=heldout_weights,
                cloud=cloud,
                render_frames=heldout_render_frames,
            ),
            initial_labels=initial_labels,
            ari_to_initial=adjusted_rand_index(initial_labels, labels),
            ari_to_previous=(
                None
                if previous_snapshot is None
                else adjusted_rand_index(previous_snapshot.labels(), labels)
            ),
        )
        points.append(point)
        previous_snapshot = current

    record(0)
    for step in range(1, iterations + 1):
        probabilities = softmax(logits, axis=1)
        gradient = (probabilities - targets) * weights[:, None]
        logits -= learning_rate * gradient.astype(np.float32, copy=False)
        if step % eval_every == 0 or step == iterations:
            record(step)

    return {
        "kind": "object_emergence_curve",
        "occlusion_delta_kind": (
            _RENDER_OCCLUSION_KIND
            if cloud is not None and render_frames
            else "mask_proxy_projection_loss"
        ),
        "mask_proxy_occlusion_delta_kind": "mask_proxy_projection_loss",
        "render_occlusion_delta_kind": (
            _RENDER_OCCLUSION_KIND if cloud is not None and render_frames else None
        ),
        "heldout": None if heldout_vote_result is None else heldout_vote_result.as_dict(),
        "iterations": iterations,
        "learning_rate": learning_rate,
        "eval_every": eval_every,
        "gaussians": field.gaussian_count,
        "slots": field.slots,
        "vote_summary": vote_result.as_dict(),
        "points": points,
    }


def mask_proxy_occlusion_delta(
    field: ObjectField,
    targets: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    probabilities = field.probabilities()
    base_loss = _projection_loss_from_probabilities(probabilities, targets, weights)
    per_slot: list[dict[str, float | int]] = []
    deltas: list[float] = []
    for slot in range(field.slots):
        masked = probabilities.copy()
        masked[:, slot] = _EPS
        masked /= np.sum(masked, axis=1, keepdims=True)
        masked_loss = _projection_loss_from_probabilities(masked, targets, weights)
        delta = float(masked_loss - base_loss)
        deltas.append(delta)
        per_slot.append(
            {
                "slot": slot,
                "delta_loss": delta,
                "relative_delta": float(delta / max(base_loss, _EPS)),
            }
        )
    return {
        "base_projection_loss": base_loss,
        "mean_delta_loss": float(np.mean(deltas)) if deltas else 0.0,
        "max_delta_loss": float(np.max(deltas)) if deltas else 0.0,
        "min_delta_loss": float(np.min(deltas)) if deltas else 0.0,
        "per_slot": per_slot,
    }


def _heldout_curve_eval(
    field: ObjectField,
    *,
    vote_result: MaskVoteResult | None,
    targets: np.ndarray | None,
    weights: np.ndarray | None,
    cloud: GaussianCloud | None,
    render_frames: list[RenderProbeFrame] | None,
) -> dict[str, Any] | None:
    if vote_result is None or targets is None or weights is None:
        return None
    supervised = weights > 0
    loss = projection_loss(field.logits, targets, weights) if np.any(supervised) else None
    render_occlusion = (
        render_occlusion_delta(cloud, field, render_frames)
        if cloud is not None and render_frames
        else None
    )
    return {
        "projection_loss": loss,
        "frames": vote_result.frames,
        "projected": vote_result.projected,
        "matched": vote_result.matched,
        "supervised_gaussians": vote_result.supervised_gaussians,
        "render_occlusion_delta": render_occlusion,
    }


def write_emergence_curve_csv(path: str | Path, curve: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_curve_csv_row(point) for point in curve.get("points", [])]
    fieldnames = [
        "step",
        "projection_loss",
        "assignment_confidence",
        "mean_normalized_entropy",
        "effective_slots",
        "spatial_compactness_score",
        "ari_to_initial",
        "ari_to_previous",
        "mask_proxy_occlusion_mean_delta_loss",
        "mask_proxy_occlusion_max_delta_loss",
        "render_occlusion_mean_delta_l1",
        "render_occlusion_mean_relative_delta_l1",
        "render_occlusion_mean_affected_fraction",
        "render_occlusion_mean_target_delta_l1",
        "render_occlusion_mean_non_target_delta_l1",
        "render_occlusion_mean_locality_score",
        "render_occlusion_effect_score",
        "heldout_projection_loss",
        "heldout_supervised_gaussians",
        "heldout_render_occlusion_effect_score",
        "object_emergence_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    labels_a = np.asarray(labels_a, dtype=np.int64)
    labels_b = np.asarray(labels_b, dtype=np.int64)
    if labels_a.ndim != 1 or labels_b.ndim != 1:
        raise ValueError("labels must be 1D arrays")
    if labels_a.shape != labels_b.shape:
        raise ValueError("label arrays must have the same shape")
    if labels_a.size < 2:
        return 1.0

    _, rows = np.unique(labels_a, return_inverse=True)
    _, cols = np.unique(labels_b, return_inverse=True)
    contingency = np.zeros((int(rows.max()) + 1, int(cols.max()) + 1), dtype=np.int64)
    np.add.at(contingency, (rows, cols), 1)

    sum_comb = float(_comb2(contingency).sum())
    row_comb = float(_comb2(contingency.sum(axis=1)).sum())
    col_comb = float(_comb2(contingency.sum(axis=0)).sum())
    total_comb = float(_comb2(np.array([labels_a.size], dtype=np.int64))[0])
    if total_comb <= 0:
        return 1.0

    expected = row_comb * col_comb / total_comb
    max_index = 0.5 * (row_comb + col_comb)
    denominator = max_index - expected
    if abs(denominator) <= _EPS:
        return 1.0 if np.array_equal(labels_a, labels_b) else 0.0
    return float((sum_comb - expected) / denominator)


def _curve_point(
    *,
    step: int,
    field: ObjectField,
    metrics: dict[str, Any],
    projection_loss_value: float,
    mask_proxy_occlusion: dict[str, Any],
    render_occlusion: dict[str, Any] | None,
    heldout: dict[str, Any] | None,
    initial_labels: np.ndarray,
    ari_to_initial: float,
    ari_to_previous: float | None,
) -> dict[str, Any]:
    assignment = metrics["assignment"]
    spatial = metrics["spatial"]
    score = _object_emergence_score(
        assignment_confidence=assignment["assignment_confidence"],
        spatial_compactness_score=spatial["compactness_score"] if spatial else None,
        stability_score=metrics["stability"]["stability_score"] if metrics["stability"] else None,
        occlusion_effect_score=(
            render_occlusion["occlusion_effect_score"] if render_occlusion else None
        ),
    )
    return {
        "step": step,
        "projection_loss": projection_loss_value,
        "assignment_confidence": assignment["assignment_confidence"],
        "mean_normalized_entropy": assignment["mean_normalized_entropy"],
        "effective_slots": assignment["effective_slots"],
        "active_slots": assignment["active_slots"],
        "spatial_compactness_score": spatial["compactness_score"] if spatial else None,
        "ari_to_initial": ari_to_initial,
        "ari_to_previous": ari_to_previous,
        "mask_proxy_occlusion_delta": mask_proxy_occlusion,
        "render_occlusion_delta": render_occlusion,
        "heldout": heldout,
        "object_emergence_score": score,
        "labels_changed_from_initial_fraction": float(
            np.count_nonzero(field.labels() != initial_labels) / max(field.gaussian_count, 1)
        ),
    }


def _assignment_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    slots = probabilities.shape[1]
    entropy_per_gaussian = -np.sum(
        probabilities * np.log(np.clip(probabilities, _EPS, 1.0)),
        axis=1,
    )
    normalized_entropy = (
        np.zeros_like(entropy_per_gaussian)
        if slots == 1
        else entropy_per_gaussian / np.log(slots)
    )
    slot_mass = probabilities.mean(axis=0)
    mass_entropy = _entropy(slot_mass)
    hard_counts = np.bincount(labels, minlength=slots).astype(np.int64)
    return {
        "mean_entropy": float(np.mean(entropy_per_gaussian)),
        "mean_normalized_entropy": float(np.mean(normalized_entropy)),
        "assignment_confidence": float(1.0 - np.mean(normalized_entropy)),
        "low_entropy_fraction": _safe_fraction(np.count_nonzero(normalized_entropy <= 0.2), labels.size),
        "high_entropy_fraction": _safe_fraction(np.count_nonzero(normalized_entropy >= 0.8), labels.size),
        "effective_slots": float(np.exp(mass_entropy)),
        "slot_mass": [float(value) for value in slot_mass],
        "hard_slot_counts": [int(value) for value in hard_counts],
        "active_slots": len(summarize_labels(labels)),
    }


def _projection_loss_from_probabilities(
    probabilities: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> float:
    supervised = weights > 0
    if not np.any(supervised):
        raise ValueError("weights must supervise at least one Gaussian")
    cross_entropy = -np.sum(targets * np.log(np.clip(probabilities, _EPS, 1.0)), axis=1)
    return float(np.sum(cross_entropy[supervised] * weights[supervised]) / np.sum(weights[supervised]))


def _curve_csv_row(point: dict[str, Any]) -> dict[str, float | int | None]:
    occlusion = point["mask_proxy_occlusion_delta"]
    render_occlusion = point.get("render_occlusion_delta")
    heldout = point.get("heldout")
    heldout_render = heldout.get("render_occlusion_delta") if isinstance(heldout, dict) else None
    score = point["object_emergence_score"]
    return {
        "step": int(point["step"]),
        "projection_loss": float(point["projection_loss"]),
        "assignment_confidence": float(point["assignment_confidence"]),
        "mean_normalized_entropy": float(point["mean_normalized_entropy"]),
        "effective_slots": float(point["effective_slots"]),
        "spatial_compactness_score": _optional_float(point.get("spatial_compactness_score")),
        "ari_to_initial": float(point["ari_to_initial"]),
        "ari_to_previous": _optional_float(point.get("ari_to_previous")),
        "mask_proxy_occlusion_mean_delta_loss": float(occlusion["mean_delta_loss"]),
        "mask_proxy_occlusion_max_delta_loss": float(occlusion["max_delta_loss"]),
        "render_occlusion_mean_delta_l1": (
            None if render_occlusion is None else float(render_occlusion["mean_delta_l1"])
        ),
        "render_occlusion_mean_relative_delta_l1": (
            None
            if render_occlusion is None
            else float(render_occlusion["mean_relative_delta_l1"])
        ),
        "render_occlusion_mean_affected_fraction": (
            None
            if render_occlusion is None
            else float(render_occlusion["mean_affected_fraction"])
        ),
        "render_occlusion_mean_target_delta_l1": (
            None if render_occlusion is None else float(render_occlusion["mean_target_delta_l1"])
        ),
        "render_occlusion_mean_non_target_delta_l1": (
            None if render_occlusion is None else float(render_occlusion["mean_non_target_delta_l1"])
        ),
        "render_occlusion_mean_locality_score": (
            None if render_occlusion is None else float(render_occlusion["mean_locality_score"])
        ),
        "render_occlusion_effect_score": (
            None
            if render_occlusion is None
            else float(render_occlusion["occlusion_effect_score"])
        ),
        "heldout_projection_loss": (
            None if not isinstance(heldout, dict) else _optional_float(heldout.get("projection_loss"))
        ),
        "heldout_supervised_gaussians": (
            None if not isinstance(heldout, dict) else int(heldout["supervised_gaussians"])
        ),
        "heldout_render_occlusion_effect_score": (
            None
            if not isinstance(heldout_render, dict)
            else float(heldout_render["occlusion_effect_score"])
        ),
        "object_emergence_score": _optional_float(score.get("score")),
    }


def _spatial_metrics(labels: np.ndarray, positions_xyz: np.ndarray, slots: int) -> dict[str, Any]:
    positions_xyz = np.asarray(positions_xyz, dtype=np.float32)
    if positions_xyz.ndim != 2 or positions_xyz.shape[1] != 3:
        raise ValueError("positions must be an Nx3 array")
    if positions_xyz.shape[0] != labels.shape[0]:
        raise ValueError("positions and labels must have the same row count")

    extent = positions_xyz.max(axis=0) - positions_xyz.min(axis=0)
    scene_radius_sq = float(np.sum(extent * extent))
    if scene_radius_sq <= _EPS:
        scene_radius_sq = 1.0

    per_slot: list[dict[str, Any]] = []
    weighted_normalized_radius = 0.0
    for slot in range(slots):
        selected = labels == slot
        count = int(np.count_nonzero(selected))
        if count == 0:
            per_slot.append(
                {
                    "slot": slot,
                    "gaussians": 0,
                    "mean_squared_radius": None,
                    "normalized_mean_squared_radius": None,
                }
            )
            continue
        points = positions_xyz[selected]
        centroid = points.mean(axis=0)
        mean_squared_radius = float(np.mean(np.sum((points - centroid) ** 2, axis=1)))
        normalized = float(mean_squared_radius / scene_radius_sq)
        weighted_normalized_radius += normalized * count
        per_slot.append(
            {
                "slot": slot,
                "gaussians": count,
                "mean_squared_radius": mean_squared_radius,
                "normalized_mean_squared_radius": normalized,
            }
        )

    overall = float(weighted_normalized_radius / max(labels.size, 1))
    return {
        "overall_normalized_compactness": overall,
        "compactness_score": float(1.0 - min(1.0, overall)),
        "per_slot": per_slot,
    }


def _stability_metrics(reference: ObjectField, current: ObjectField) -> dict[str, Any]:
    if reference.gaussian_count != current.gaussian_count:
        raise ValueError("reference and current fields must have the same Gaussian count")
    reference_labels = reference.labels()
    current_labels = current.labels()
    ari = adjusted_rand_index(reference_labels, current_labels)
    matching, matched = _greedy_slot_matching(reference_labels, current_labels, reference.slots, current.slots)
    return {
        "adjusted_rand_index": ari,
        "stability_score": float(max(0.0, min(1.0, ari))),
        "matched_label_agreement": _safe_fraction(matched, reference.gaussian_count),
        "slot_matching": matching,
    }


def _greedy_slot_matching(
    reference_labels: np.ndarray,
    current_labels: np.ndarray,
    reference_slots: int,
    current_slots: int,
) -> tuple[list[dict[str, int]], int]:
    confusion = np.zeros((reference_slots, current_slots), dtype=np.int64)
    np.add.at(confusion, (reference_labels, current_labels), 1)
    candidates = [
        (int(confusion[row, col]), row, col)
        for row in range(reference_slots)
        for col in range(current_slots)
        if confusion[row, col] > 0
    ]
    candidates.sort(reverse=True)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    matching: list[dict[str, int]] = []
    matched = 0
    for overlap, row, col in candidates:
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        matched += overlap
        matching.append(
            {
                "reference_slot": row,
                "current_slot": col,
                "overlap_gaussians": overlap,
            }
        )
    matching.sort(key=lambda item: item["reference_slot"])
    return matching, matched


def _object_emergence_score(
    *,
    assignment_confidence: float,
    spatial_compactness_score: float | None,
    stability_score: float | None,
    occlusion_effect_score: float | None = None,
) -> dict[str, Any]:
    weights = {
        "assignment_confidence": 0.25,
        "stability": 0.35,
        "spatial_compactness": 0.20,
        "occlusion_effect": 0.20,
    }
    components: dict[str, float | None] = {
        "assignment_confidence": float(max(0.0, min(1.0, assignment_confidence))),
        "stability": stability_score,
        "spatial_compactness": spatial_compactness_score,
        "occlusion_effect": (
            None
            if occlusion_effect_score is None
            else float(max(0.0, min(1.0, occlusion_effect_score)))
        ),
    }
    missing = [name for name, value in components.items() if value is None]
    available = {name: value for name, value in components.items() if value is not None}
    weight_sum = sum(weights[name] for name in available)
    score = None
    if weight_sum > 0:
        score = float(sum(weights[name] * float(value) for name, value in available.items()) / weight_sum)
    return {
        "score": score,
        "complete": len(missing) == 0,
        "components": components,
        "weights": weights,
        "missing_components": missing,
        "unsupported_components": ["gradient_coherence"],
    }


def _entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float32)
    return float(-np.sum(values * np.log(np.clip(values, _EPS, 1.0))))


def _comb2(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values * (values - 1.0) / 2.0


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
