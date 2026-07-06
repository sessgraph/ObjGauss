from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.real_sample_v2_full_cloud_purity import (
    RealSampleV2FullCloudPurityReport,
    real_sample_v2_full_cloud_purity_from_cloud,
    validate_real_sample_v2_full_cloud_purity_summary,
)
from objgauss.core.trainable_kernel import (
    TRAINING_IMAGE_RENDERER_POINT,
    TRAINING_IMAGE_RENDERERS,
)

REAL_SAMPLE_V2_SEGMENTATION_QUALITY_SCHEMA = (
    "objgauss-real-sample-v2-segmentation-quality-v1"
)
_STATUS_PASS = "real_sample_v2_segmentation_quality_pass"
_STATUS_DIAGNOSTIC = "real_sample_v2_segmentation_quality_diagnostic"

_DIRECT_SLOT_MATCH_FLOOR = 0.98
_TARGET_RECALL_FLOOR = 0.90
_PREDICTED_PURITY_FLOOR = 0.95
_WEAK_TARGET_RECALL_FLOOR = 0.95
_WEAK_PREDICTED_PURITY_FLOOR = 0.98
_LOW_CONFIDENCE_THRESHOLD = 0.50
_LOW_MEAN_CONFIDENCE_FLOOR = 0.65
_HIGH_ENTROPY_THRESHOLD = 0.50


@dataclass(frozen=True)
class RealSampleV2SegmentationQualityReport:
    projected_cloud: GaussianCloud
    sample_source: str
    max_points: int
    solver_temperature: float | None
    training_quality: dict[str, Any] | None = None
    viewer_path: str | None = None
    schema: str = REAL_SAMPLE_V2_SEGMENTATION_QUALITY_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        analysis = _segmentation_analysis(self.projected_cloud)
        status = _STATUS_PASS if analysis["passes_hard_gate"] else _STATUS_DIAGNOSTIC
        payload = {
            "schema": self.schema,
            "kind": "real_sample_v2_segmentation_quality",
            "status": status,
            "source": {
                "input": self.sample_source,
                "source_gaussians": int(self.projected_cloud.count),
                "required_fields": [
                    "object_id",
                    "target_object_id",
                    "target_slot",
                    "assignment_confidence",
                    "assignment_entropy",
                ],
            },
            "segmentation_target": {
                "target": "object_id_one_hot_segmentation",
                "max_points": int(self.max_points),
                "solver_temperature": (
                    None if self.solver_temperature is None else float(self.solver_temperature)
                ),
                "export_object_id": "argmax_assignment_slot",
                "quality_focus": "per_object_confusion_and_assignment_uncertainty",
            },
            "training_quality": dict(self.training_quality or {}),
            "global_quality": {
                "direct_slot_match": analysis["direct_slot_match"],
                "hard_argmax_object_purity": analysis["hard_argmax_object_purity"],
                "min_predicted_object_purity": analysis["min_predicted_object_purity"],
                "min_target_recall": analysis["min_target_recall"],
                "mixed_gaussians": analysis["mixed_gaussians"],
                "predicted_object_count": len(analysis["predicted_objects"]),
                "target_object_count": len(analysis["target_objects"]),
                "diagnostics": analysis["diagnostics"],
            },
            "confusion": {
                "target_slots": analysis["target_slots"],
                "predicted_object_ids": analysis["predicted_object_ids"],
                "matrix": analysis["confusion_matrix"],
                "rows": analysis["confusion_rows"],
            },
            "per_predicted_object": analysis["predicted_objects"],
            "per_target_object": analysis["target_objects"],
            "recommendation": _recommendation(analysis),
            "viewer": {
                "route_param": "ply",
                "viewer_path": self.viewer_path,
                "debug_route": f"/?ply={self.viewer_path}" if self.viewer_path else None,
                "load_mode": "url-object-aware-ply",
            },
            "output_policy": {
                "preview_ply": "write to /tmp or ignored outputs; do not commit generated preview PLY",
                "summary": "write to /tmp or ignored outputs for segmentation quality diagnostics",
                "screenshots": "write browser evidence to /tmp only",
            },
            "non_goals": {
                "uses_gpu": False,
                "unfreezes_gaussian_geometry": False,
                "unfreezes_camera": False,
                "mutates_dynamic_k": False,
                "uses_rollout_model": False,
                "uses_replay_buffer": False,
                "uses_diffusion": False,
                "claims_public_demo_release": False,
            },
        }
        return validate_real_sample_v2_segmentation_quality_summary(payload)


def real_sample_v2_segmentation_quality_from_cloud(
    cloud: GaussianCloud,
    *,
    sample_source: str = "memory://gaussian-cloud",
    object_id_field: str = "object_id",
    slots: int | None = None,
    max_points: int = 128,
    frame_count: int = 2,
    temporal_offset: float = 0.01,
    image_width: int = 12,
    image_height: int = 12,
    point_radius: int = 1,
    visibility_policy: str = "covered_pixels",
    seed: int = 4,
    iterations: int = 100,
    learning_rate: float = 0.4,
    temperature_candidates: Sequence[float] = (1.0, 0.75, 0.5, 0.35, 0.25),
    baseline_temperature: float = 1.0,
    image_renderer: str = TRAINING_IMAGE_RENDERER_POINT,
    vram_reserve_gb: int = 1,
    rewrite_sh: bool = False,
    viewer_path: str | None = None,
) -> RealSampleV2SegmentationQualityReport:
    if image_renderer not in TRAINING_IMAGE_RENDERERS:
        raise ValueError(f"image_renderer must be one of: {', '.join(TRAINING_IMAGE_RENDERERS)}")
    if max_points < 1:
        raise ValueError("max_points must be >= 1")
    coverage = real_sample_v2_full_cloud_purity_from_cloud(
        cloud,
        sample_source=sample_source,
        object_id_field=object_id_field,
        slots=slots,
        max_point_candidates=(int(max_points),),
        frame_count=frame_count,
        temporal_offset=temporal_offset,
        image_width=image_width,
        image_height=image_height,
        point_radius=point_radius,
        visibility_policy=visibility_policy,
        seed=seed,
        iterations=iterations,
        learning_rate=learning_rate,
        temperature_candidates=temperature_candidates,
        baseline_temperature=baseline_temperature,
        image_renderer=image_renderer,
        vram_reserve_gb=vram_reserve_gb,
        rewrite_sh=rewrite_sh,
        viewer_path=viewer_path,
    )
    return real_sample_v2_segmentation_quality_from_purity_report(
        coverage,
        sample_source=sample_source,
        viewer_path=viewer_path,
    )


def real_sample_v2_segmentation_quality_from_purity_report(
    report: RealSampleV2FullCloudPurityReport,
    *,
    sample_source: str | None = None,
    viewer_path: str | None = None,
) -> RealSampleV2SegmentationQualityReport:
    summary = validate_real_sample_v2_full_cloud_purity_summary(report.as_dict())
    best = summary["best_candidate"]
    return RealSampleV2SegmentationQualityReport(
        projected_cloud=report.best_candidate.projected_cloud,
        sample_source=str(sample_source or summary["source"]["input"]),
        max_points=int(summary["segmentation_target"]["selected_max_points"]),
        solver_temperature=float(summary["segmentation_target"]["selected_solver_temperature"]),
        training_quality=dict(best["quality"]),
        viewer_path=viewer_path if viewer_path is not None else summary["viewer"]["viewer_path"],
    )


def real_sample_v2_segmentation_quality_from_projected_cloud(
    cloud: GaussianCloud,
    *,
    sample_source: str = "memory://projected-cloud",
    max_points: int = 128,
    solver_temperature: float | None = None,
    viewer_path: str | None = None,
) -> RealSampleV2SegmentationQualityReport:
    return RealSampleV2SegmentationQualityReport(
        projected_cloud=cloud,
        sample_source=str(sample_source),
        max_points=int(max_points),
        solver_temperature=solver_temperature,
        viewer_path=viewer_path,
    )


def validate_real_sample_v2_segmentation_quality_summary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("real sample v2 segmentation quality summary must be a dict")
    if payload.get("schema") != REAL_SAMPLE_V2_SEGMENTATION_QUALITY_SCHEMA:
        raise ValueError(
            f"unsupported real sample v2 segmentation quality schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "real_sample_v2_segmentation_quality":
        raise ValueError("real sample v2 segmentation quality kind is unsupported")
    if payload.get("status") not in {_STATUS_PASS, _STATUS_DIAGNOSTIC}:
        raise ValueError("real sample v2 segmentation quality status is unsupported")
    for key in (
        "source",
        "segmentation_target",
        "global_quality",
        "confusion",
        "per_predicted_object",
        "per_target_object",
        "recommendation",
        "viewer",
        "output_policy",
        "non_goals",
    ):
        if key not in payload:
            raise ValueError(f"real sample v2 segmentation quality summary missing {key}")
    source_count = int(payload["source"]["source_gaussians"])
    if source_count < 1:
        raise ValueError("segmentation quality summary requires source gaussians")
    if int(payload["segmentation_target"]["max_points"]) < 1:
        raise ValueError("segmentation quality max_points must be >= 1")
    if payload["segmentation_target"].get("export_object_id") != "argmax_assignment_slot":
        raise ValueError("segmentation quality export object_id must be argmax assignment slot")
    global_quality = payload["global_quality"]
    for metric in (
        "direct_slot_match",
        "hard_argmax_object_purity",
        "min_predicted_object_purity",
        "min_target_recall",
    ):
        value = float(global_quality[metric])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"segmentation quality {metric} must be in [0, 1]")
    if int(global_quality["mixed_gaussians"]) < 0:
        raise ValueError("segmentation quality mixed_gaussians must be >= 0")
    confusion = payload["confusion"]
    matrix = confusion.get("matrix")
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("segmentation quality confusion matrix is required")
    row_total = 0
    for row in matrix:
        if not isinstance(row, list) or not row:
            raise ValueError("segmentation quality confusion rows must be non-empty lists")
        row_total += sum(int(value) for value in row)
    if row_total != source_count:
        raise ValueError("segmentation quality confusion total must match source gaussians")
    if not payload["per_predicted_object"]:
        raise ValueError("segmentation quality requires predicted object summaries")
    if not payload["per_target_object"]:
        raise ValueError("segmentation quality requires target object summaries")
    recommendation = payload["recommendation"]
    if recommendation.get("requires_geometry_unfreeze") is not False:
        raise ValueError("segmentation quality must not recommend geometry unfreeze")
    if recommendation.get("requires_diffusion_replay_or_rollout") is not False:
        raise ValueError("segmentation quality must not recommend diffusion/replay/rollout")
    non_goals = payload["non_goals"]
    if (
        non_goals.get("uses_gpu")
        or non_goals.get("unfreezes_gaussian_geometry")
        or non_goals.get("unfreezes_camera")
        or non_goals.get("mutates_dynamic_k")
        or non_goals.get("uses_rollout_model")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("claims_public_demo_release")
    ):
        raise ValueError("real sample v2 segmentation quality summary violates non-goals")
    return payload


def _segmentation_analysis(cloud: GaussianCloud) -> dict[str, Any]:
    _require_fields(
        cloud,
        (
            "x",
            "y",
            "z",
            "object_id",
            "target_object_id",
            "target_slot",
            "assignment_confidence",
            "assignment_entropy",
        ),
    )
    vertices = cloud.vertices
    predicted = np.asarray(vertices["object_id"], dtype=np.int64)
    target_slots = np.asarray(vertices["target_slot"], dtype=np.int64)
    target_object_ids = np.asarray(vertices["target_object_id"], dtype=np.int64)
    confidence = np.asarray(vertices["assignment_confidence"], dtype=np.float64)
    entropy = np.asarray(vertices["assignment_entropy"], dtype=np.float64)
    positions = np.column_stack(
        [
            np.asarray(vertices["x"], dtype=np.float64),
            np.asarray(vertices["y"], dtype=np.float64),
            np.asarray(vertices["z"], dtype=np.float64),
        ]
    )
    if predicted.size == 0:
        raise ValueError("segmentation quality requires at least one gaussian")
    if np.any(predicted < 0) or np.any(target_slots < 0):
        raise ValueError("segmentation quality object ids and target slots must be non-negative")

    predicted_ids = tuple(int(value) for value in np.unique(predicted))
    target_ids = tuple(int(value) for value in np.unique(target_slots))
    matrix = _confusion_matrix(target_slots, predicted, target_ids, predicted_ids)
    total = int(predicted.shape[0])
    diagonal = _aligned_diagonal_sum(matrix, target_ids, predicted_ids)
    direct_match = 0.0 if total == 0 else float(diagonal / total)
    predicted_objects = _predicted_object_records(
        predicted=predicted,
        target_slots=target_slots,
        target_object_ids=target_object_ids,
        confidence=confidence,
        entropy=entropy,
        positions=positions,
        predicted_ids=predicted_ids,
    )
    target_objects = _target_object_records(
        predicted=predicted,
        target_slots=target_slots,
        target_object_ids=target_object_ids,
        target_ids=target_ids,
    )
    min_predicted_purity = min(record["purity"] for record in predicted_objects)
    min_target_recall = min(record["recall"] for record in target_objects)
    hard_purity = _weighted_average(
        [(record["purity"], record["gaussian_count"]) for record in predicted_objects]
    )
    diagnostics = _diagnostics(
        direct_slot_match=direct_match,
        min_predicted_purity=min_predicted_purity,
        min_target_recall=min_target_recall,
        predicted_objects=predicted_objects,
        target_objects=target_objects,
    )
    passes_hard_gate = (
        direct_match >= _DIRECT_SLOT_MATCH_FLOOR
        and min_predicted_purity >= _PREDICTED_PURITY_FLOOR
        and min_target_recall >= _TARGET_RECALL_FLOOR
        and len(predicted_ids) == len(target_ids)
    )
    return {
        "passes_hard_gate": passes_hard_gate,
        "direct_slot_match": direct_match,
        "hard_argmax_object_purity": hard_purity,
        "min_predicted_object_purity": min_predicted_purity,
        "min_target_recall": min_target_recall,
        "mixed_gaussians": int(total - diagonal),
        "target_slots": list(target_ids),
        "predicted_object_ids": list(predicted_ids),
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_rows": _confusion_rows(matrix, target_ids, predicted_ids),
        "predicted_objects": predicted_objects,
        "target_objects": target_objects,
        "diagnostics": diagnostics,
    }


def _predicted_object_records(
    *,
    predicted: np.ndarray,
    target_slots: np.ndarray,
    target_object_ids: np.ndarray,
    confidence: np.ndarray,
    entropy: np.ndarray,
    positions: np.ndarray,
    predicted_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for object_id in predicted_ids:
        mask = predicted == object_id
        count = int(mask.sum())
        slot_counts = _value_counts(target_slots[mask])
        object_counts = _value_counts(target_object_ids[mask])
        dominant_target_slot, dominant_count = _dominant(slot_counts)
        dominant_target_object_id, _ = _dominant(object_counts)
        purity = 0.0 if count == 0 else float(dominant_count / count)
        low_confidence_count = int(np.sum(confidence[mask] < _LOW_CONFIDENCE_THRESHOLD))
        high_entropy_count = int(np.sum(entropy[mask] > _HIGH_ENTROPY_THRESHOLD))
        diagnostics: list[str] = []
        if purity < _WEAK_PREDICTED_PURITY_FLOOR:
            diagnostics.append("mixed_predicted_object")
        if float(np.mean(confidence[mask])) < _LOW_MEAN_CONFIDENCE_FLOOR:
            diagnostics.append("low_confidence_predicted_object")
        if float(np.mean(entropy[mask])) > _HIGH_ENTROPY_THRESHOLD:
            diagnostics.append("high_entropy_predicted_object")
        records.append(
            {
                "object_id": int(object_id),
                "gaussian_count": count,
                "dominant_target_slot": int(dominant_target_slot),
                "dominant_target_object_id": int(dominant_target_object_id),
                "dominant_target_count": int(dominant_count),
                "purity": purity,
                "mixed_count": int(count - dominant_count),
                "confidence": _distribution(confidence[mask]),
                "entropy": _distribution(entropy[mask]),
                "low_confidence_count": low_confidence_count,
                "low_confidence_fraction": float(low_confidence_count / count),
                "high_entropy_count": high_entropy_count,
                "high_entropy_fraction": float(high_entropy_count / count),
                "centroid": _vector_mean(positions[mask]),
                "bbox": _bbox(positions[mask]),
                "target_slot_counts": [
                    {"target_slot": int(slot), "count": int(value)}
                    for slot, value in sorted(slot_counts.items())
                ],
                "diagnostics": diagnostics,
            }
        )
    return records


def _target_object_records(
    *,
    predicted: np.ndarray,
    target_slots: np.ndarray,
    target_object_ids: np.ndarray,
    target_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for target_slot in target_ids:
        mask = target_slots == target_slot
        count = int(mask.sum())
        predicted_counts = _value_counts(predicted[mask])
        target_object_counts = _value_counts(target_object_ids[mask])
        dominant_predicted_id, dominant_count = _dominant(predicted_counts)
        dominant_target_object_id, _ = _dominant(target_object_counts)
        recall = 0.0 if count == 0 else float(dominant_count / count)
        leakage = [
            {"object_id": int(object_id), "count": int(value)}
            for object_id, value in sorted(predicted_counts.items())
            if int(object_id) != int(dominant_predicted_id)
        ]
        diagnostics = []
        if recall < _WEAK_TARGET_RECALL_FLOOR:
            diagnostics.append("weak_target_recall")
        records.append(
            {
                "target_slot": int(target_slot),
                "target_object_id": int(dominant_target_object_id),
                "gaussian_count": count,
                "dominant_predicted_object_id": int(dominant_predicted_id),
                "dominant_predicted_count": int(dominant_count),
                "recall": recall,
                "missed_count": int(count - dominant_count),
                "leakage": leakage,
                "diagnostics": diagnostics,
            }
        )
    return records


def _confusion_matrix(
    target_slots: np.ndarray,
    predicted: np.ndarray,
    target_ids: tuple[int, ...],
    predicted_ids: tuple[int, ...],
) -> np.ndarray:
    target_index = {target: index for index, target in enumerate(target_ids)}
    predicted_index = {object_id: index for index, object_id in enumerate(predicted_ids)}
    matrix = np.zeros((len(target_ids), len(predicted_ids)), dtype=np.int64)
    for target, object_id in zip(target_slots, predicted, strict=True):
        matrix[target_index[int(target)], predicted_index[int(object_id)]] += 1
    return matrix


def _aligned_diagonal_sum(
    matrix: np.ndarray,
    target_ids: tuple[int, ...],
    predicted_ids: tuple[int, ...],
) -> int:
    predicted_index = {object_id: index for index, object_id in enumerate(predicted_ids)}
    total = 0
    for row_index, target_id in enumerate(target_ids):
        column_index = predicted_index.get(target_id)
        if column_index is not None:
            total += int(matrix[row_index, column_index])
    return total


def _confusion_rows(
    matrix: np.ndarray,
    target_ids: tuple[int, ...],
    predicted_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, target_slot in enumerate(target_ids):
        counts = [
            {"object_id": int(object_id), "count": int(matrix[row_index, column_index])}
            for column_index, object_id in enumerate(predicted_ids)
        ]
        rows.append(
            {
                "target_slot": int(target_slot),
                "predicted_counts": counts,
                "total": int(matrix[row_index].sum()),
            }
        )
    return rows


def _diagnostics(
    *,
    direct_slot_match: float,
    min_predicted_purity: float,
    min_target_recall: float,
    predicted_objects: list[dict[str, Any]],
    target_objects: list[dict[str, Any]],
) -> list[str]:
    diagnostics: list[str] = []
    if direct_slot_match < _DIRECT_SLOT_MATCH_FLOOR:
        diagnostics.append("direct_slot_match_below_gate")
    if min_predicted_purity < _PREDICTED_PURITY_FLOOR:
        diagnostics.append("predicted_object_purity_below_gate")
    if min_target_recall < _TARGET_RECALL_FLOOR:
        diagnostics.append("target_recall_below_gate")
    for record in target_objects:
        if "weak_target_recall" in record["diagnostics"]:
            diagnostics.append(f"weak_target_recall:{record['target_slot']}")
    for record in predicted_objects:
        if "mixed_predicted_object" in record["diagnostics"]:
            diagnostics.append(f"mixed_predicted_object:{record['object_id']}")
        if "low_confidence_predicted_object" in record["diagnostics"]:
            diagnostics.append(f"low_confidence_predicted_object:{record['object_id']}")
        if "high_entropy_predicted_object" in record["diagnostics"]:
            diagnostics.append(f"high_entropy_predicted_object:{record['object_id']}")
    return diagnostics


def _recommendation(analysis: dict[str, Any]) -> dict[str, Any]:
    weak_targets = [
        record["target_slot"]
        for record in analysis["target_objects"]
        if "weak_target_recall" in record["diagnostics"]
    ]
    mixed_objects = [
        record["object_id"]
        for record in analysis["predicted_objects"]
        if "mixed_predicted_object" in record["diagnostics"]
    ]
    low_confidence_objects = [
        record["object_id"]
        for record in analysis["predicted_objects"]
        if "low_confidence_predicted_object" in record["diagnostics"]
    ]
    high_entropy_objects = [
        record["object_id"]
        for record in analysis["predicted_objects"]
        if "high_entropy_predicted_object" in record["diagnostics"]
    ]
    if analysis["passes_hard_gate"] and (weak_targets or mixed_objects or high_entropy_objects):
        decision = "keep_128_target_and_inspect_weak_boundaries"
        action = "inspect_confusion_slots_before_evidence_normalization"
        evidence_normalization = "candidate_next_if_visual_boundary_issue_persists"
    elif analysis["passes_hard_gate"]:
        decision = "segmentation_quality_sufficient_for_current_sample"
        action = "keep_128_target"
        evidence_normalization = "not_required_for_current_sample"
    else:
        decision = "segmentation_quality_needs_normalization_or_more_coverage"
        action = "diagnose_evidence_normalization"
        evidence_normalization = "required_next"
    return {
        "decision": decision,
        "action": action,
        "weak_target_slots": weak_targets,
        "mixed_predicted_objects": mixed_objects,
        "low_confidence_predicted_objects": low_confidence_objects,
        "high_entropy_predicted_objects": high_entropy_objects,
        "evidence_normalization": evidence_normalization,
        "export_policy": "do_not_hide_or_recolor_low_quality_objects",
        "requires_more_coverage": False,
        "requires_geometry_unfreeze": False,
        "requires_diffusion_replay_or_rollout": False,
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "p10": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "min": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "mean": float(np.mean(values)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def _value_counts(values: np.ndarray) -> dict[int, int]:
    keys, counts = np.unique(values.astype(np.int64, copy=False), return_counts=True)
    return {int(key): int(count) for key, count in zip(keys, counts, strict=True)}


def _dominant(counts: dict[int, int]) -> tuple[int, int]:
    if not counts:
        return 0, 0
    return max(counts.items(), key=lambda item: (item[1], -item[0]))


def _weighted_average(values: list[tuple[float, int]]) -> float:
    weight = sum(count for _, count in values)
    if weight <= 0:
        return 0.0
    return float(sum(value * count for value, count in values) / weight)


def _vector_mean(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.0, 0.0, 0.0]
    return [float(value) for value in np.mean(values, axis=0)]


def _bbox(values: np.ndarray) -> dict[str, list[float]]:
    if values.size == 0:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    return {
        "min": [float(value) for value in np.min(values, axis=0)],
        "max": [float(value) for value in np.max(values, axis=0)],
    }


def _require_fields(cloud: GaussianCloud, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in cloud.fields]
    if missing:
        raise ValueError(f"segmentation quality cloud missing fields: {', '.join(missing)}")
