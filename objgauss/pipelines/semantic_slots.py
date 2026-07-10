from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from objgauss.core.features import positions
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.projection import project_points

DEFAULT_SLOT_BACKGROUND_LABELS = (
    "background",
    "white background",
    "table surface",
    "floor",
    "wall",
    "shadow",
    "cast shadow",
    "object shadow",
)

__all__ = (
    "DEFAULT_SLOT_BACKGROUND_LABELS",
    "SlotAlignmentResult",
    "align_mask_manifest_slots",
)


@dataclass(frozen=True)
class SlotAlignmentResult:
    source_manifest: Path
    output_manifest: Path
    frames: int
    masks: int
    aligned_slots: int
    remapped_masks: int
    dropped_masks: int
    named_slots: int
    clusters: tuple[dict[str, Any], ...]
    slot_naming_quality: dict[str, Any]
    record_filters: dict[str, Any]
    slot_rebalance: dict[str, Any]
    foreground_coverage_recovery: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "objgauss-cross-view-slot-alignment-v1",
            "source_manifest": str(self.source_manifest),
            "output_manifest": str(self.output_manifest),
            "frames": int(self.frames),
            "masks": int(self.masks),
            "aligned_slots": int(self.aligned_slots),
            "remapped_masks": int(self.remapped_masks),
            "dropped_masks": int(self.dropped_masks),
            "named_slots": int(self.named_slots),
            "clusters": list(self.clusters),
            "slot_naming_quality": self.slot_naming_quality,
            "record_filters": self.record_filters,
            "slot_rebalance": self.slot_rebalance,
            "foreground_coverage_recovery": self.foreground_coverage_recovery,
        }


@dataclass
class _MaskRecord:
    frame_index: int
    mask_index: int
    source_slot: int
    source_label: str
    confidence: float
    area: int
    gaussian_indices: np.ndarray
    clip_scores: dict[str, float]


@dataclass
class _SlotCluster:
    records: list[_MaskRecord]
    gaussian_indices: set[int]
    clip_scores: dict[str, float]
    label_weights: dict[str, float]


@dataclass(frozen=True)
class _MaskRecordExtraction:
    records: list[_MaskRecord]
    source_masks: int
    projected_masks: int
    filtered_low_area: int
    filtered_top_label: int
    filtered_empty_projection: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "objgauss-slot-record-filters-v1",
            "source_masks": int(self.source_masks),
            "projected_masks": int(self.projected_masks),
            "kept_records": int(len(self.records)),
            "filtered_low_area": int(self.filtered_low_area),
            "filtered_top_label": int(self.filtered_top_label),
            "filtered_empty_projection": int(self.filtered_empty_projection),
        }


@dataclass(frozen=True)
class _SlotRebalanceResult:
    clusters: tuple[_SlotCluster, ...]
    dropped_clusters: tuple[dict[str, Any], ...]
    min_slot_support_gaussians: int
    min_slot_support_ratio: float
    min_balanced_slots: int

    @property
    def dropped_slots(self) -> int:
        return len(self.dropped_clusters)

    @property
    def dropped_masks(self) -> int:
        return sum(int(cluster["mask_count"]) for cluster in self.dropped_clusters)

    def as_dict(self) -> dict[str, Any]:
        support_counts = [len(cluster.gaussian_indices) for cluster in self.clusters]
        return {
            "kind": "objgauss-slot-support-rebalance-v1",
            "enabled": bool(self.min_slot_support_gaussians > 0 or self.min_slot_support_ratio > 0),
            "min_slot_support_gaussians": int(self.min_slot_support_gaussians),
            "min_slot_support_ratio": float(self.min_slot_support_ratio),
            "min_balanced_slots": int(self.min_balanced_slots),
            "kept_slots": int(len(self.clusters)),
            "dropped_slots": int(self.dropped_slots),
            "dropped_masks": int(self.dropped_masks),
            "kept_support_gaussians": support_counts,
            "support_balance_score": _support_balance_score(support_counts),
            "dropped_clusters": list(self.dropped_clusters),
        }


def align_mask_manifest_slots(
    cloud: GaussianCloud,
    manifest_path: str | Path,
    *,
    output: str | Path,
    min_iou: float = 0.05,
    min_shared_gaussians: int = 1,
    max_slots: int | None = None,
    max_frames: int | None = None,
    min_mask_area: int = 0,
    min_mask_area_fraction: float = 0.0,
    exclude_top_labels: list[str] | tuple[str, ...] | None = None,
    exclude_background_top_labels: bool = False,
    background_labels: list[str] | tuple[str, ...] | None = None,
    min_named_slots: int = 1,
    min_unique_slot_labels: int = 2,
    max_slot_label_fraction: float = 0.5,
    max_background_slot_fraction: float = 0.25,
    foreground_only_slot_names: bool = False,
    unique_slot_names: bool = False,
    slot_name_diversity_penalty: float = 0.0,
    min_slot_support_gaussians: int = 0,
    min_slot_support_ratio: float = 0.0,
    min_balanced_slots: int = 1,
    recover_foreground_coverage: bool = False,
) -> SlotAlignmentResult:
    if min_iou < 0 or min_iou > 1:
        raise ValueError("min_iou must be in [0, 1]")
    if min_shared_gaussians < 0:
        raise ValueError("min_shared_gaussians must be >= 0")
    if max_slots is not None and max_slots < 1:
        raise ValueError("max_slots must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if min_mask_area < 0:
        raise ValueError("min_mask_area must be >= 0")
    if not 0.0 <= min_mask_area_fraction <= 1.0:
        raise ValueError("min_mask_area_fraction must be in [0, 1]")
    if min_named_slots < 0:
        raise ValueError("min_named_slots must be >= 0")
    if min_unique_slot_labels < 0:
        raise ValueError("min_unique_slot_labels must be >= 0")
    if not 0.0 < max_slot_label_fraction <= 1.0:
        raise ValueError("max_slot_label_fraction must be in (0, 1]")
    if not 0.0 <= max_background_slot_fraction <= 1.0:
        raise ValueError("max_background_slot_fraction must be in [0, 1]")
    if slot_name_diversity_penalty < 0.0:
        raise ValueError("slot_name_diversity_penalty must be >= 0")
    if min_slot_support_gaussians < 0:
        raise ValueError("min_slot_support_gaussians must be >= 0")
    if not 0.0 <= min_slot_support_ratio <= 1.0:
        raise ValueError("min_slot_support_ratio must be in [0, 1]")
    if min_balanced_slots < 1:
        raise ValueError("min_balanced_slots must be >= 1")

    clean_background_labels = _clean_label_set(background_labels or DEFAULT_SLOT_BACKGROUND_LABELS)
    excluded_labels = _clean_label_set(exclude_top_labels or ())
    if exclude_background_top_labels:
        excluded_labels |= clean_background_labels

    manifest_path = Path(manifest_path)
    output = Path(output)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("mask manifest must contain a non-empty frames list")

    root = manifest_path.parent
    extraction = _mask_records_for_manifest(
        cloud,
        payload,
        root=root,
        max_frames=max_frames,
        min_mask_area=min_mask_area,
        min_mask_area_fraction=min_mask_area_fraction,
        exclude_top_labels=excluded_labels,
    )
    records = extraction.records
    if not records:
        raise ValueError("mask manifest did not produce any kept projected mask records")

    clusters = _cluster_mask_records(
        records,
        min_iou=float(min_iou),
        min_shared_gaussians=int(min_shared_gaussians),
    )
    candidate_clusters = _order_clusters(clusters)
    if max_slots is not None:
        candidate_clusters = candidate_clusters[:max_slots]
    rebalance = _rebalance_slot_support(
        candidate_clusters,
        min_slot_support_gaussians=min_slot_support_gaussians,
        min_slot_support_ratio=min_slot_support_ratio,
        min_balanced_slots=min_balanced_slots,
    )
    ordered_clusters = list(rebalance.clusters)
    kept_cluster_ids = {id(cluster) for cluster in ordered_clusters}
    dropped_clusters = [
        cluster
        for cluster in candidate_clusters
        if id(cluster) not in kept_cluster_ids
    ]
    if not ordered_clusters:
        raise ValueError("slot support rebalance removed all clusters")
    record_to_slot = {
        (record.frame_index, record.mask_index): slot
        for slot, cluster in enumerate(ordered_clusters)
        for record in cluster.records
    }
    naming_policy = _slot_naming_policy(
        background_labels=clean_background_labels,
        foreground_only_slot_names=foreground_only_slot_names,
        unique_slot_names=unique_slot_names,
        slot_name_diversity_penalty=slot_name_diversity_penalty,
    )
    summaries = tuple(
        _cluster_summaries_with_naming_policy(
            ordered_clusters,
            background_labels=clean_background_labels,
            foreground_only_slot_names=foreground_only_slot_names,
            unique_slot_names=unique_slot_names,
            slot_name_diversity_penalty=slot_name_diversity_penalty,
        )
    )
    coverage_recovery, coverage_record_metadata = _recover_foreground_coverage_records(
        dropped_clusters,
        kept_clusters=ordered_clusters,
        slot_summaries=summaries,
        record_to_slot=record_to_slot,
        background_labels=clean_background_labels,
        enabled=recover_foreground_coverage,
        min_shared_gaussians=max(1, int(min_shared_gaussians)),
    )
    for key, metadata in coverage_record_metadata.items():
        record_to_slot[key] = int(metadata["target_slot"])
    slot_naming_quality = _slot_naming_quality(
        summaries,
        background_labels=clean_background_labels,
        min_named_slots=min_named_slots,
        min_unique_slot_labels=min_unique_slot_labels,
        max_slot_label_fraction=max_slot_label_fraction,
        max_background_slot_fraction=max_background_slot_fraction,
    )

    aligned_payload = _rewrite_aligned_manifest(
        payload,
        record_to_slot=record_to_slot,
        slot_summaries=summaries,
        max_frames=max_frames,
        source_record_count=extraction.projected_masks,
        record_filters=extraction.as_dict(),
        slot_rebalance=rebalance.as_dict(),
        slot_naming_quality=slot_naming_quality,
        naming_policy=naming_policy,
        coverage_record_metadata=coverage_record_metadata,
        foreground_coverage_recovery=coverage_recovery,
        source_root=manifest_path.parent,
        output_root=output.parent,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(aligned_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_frames = aligned_payload.get("frames")
    output_frame_count = len(output_frames) if isinstance(output_frames, list) else 0

    slot_naming_quality = aligned_payload["slot_alignment"]["slot_naming_quality"]
    record_filters = aligned_payload["slot_alignment"]["record_filters"]
    slot_rebalance = aligned_payload["slot_alignment"]["slot_rebalance"]
    foreground_coverage_recovery = aligned_payload["slot_alignment"]["foreground_coverage_recovery"]
    named_slots = sum(1 for cluster in summaries if cluster["semantic_name_source"] != "unnamed")
    remapped_masks = sum(
        1
        for record in records
        if (record.frame_index, record.mask_index) in record_to_slot
        and record_to_slot[(record.frame_index, record.mask_index)] != record.source_slot
    )
    dropped_masks = len(records) - len(record_to_slot)
    return SlotAlignmentResult(
        source_manifest=manifest_path,
        output_manifest=output,
        frames=output_frame_count,
        masks=len(record_to_slot),
        aligned_slots=len(ordered_clusters),
        remapped_masks=remapped_masks,
        dropped_masks=dropped_masks,
        named_slots=named_slots,
        clusters=summaries,
        slot_naming_quality=slot_naming_quality,
        record_filters=record_filters,
        slot_rebalance=slot_rebalance,
        foreground_coverage_recovery=foreground_coverage_recovery,
    )


def _mask_records_for_manifest(
    cloud: GaussianCloud,
    payload: dict[str, Any],
    *,
    root: Path,
    max_frames: int | None,
    min_mask_area: int,
    min_mask_area_fraction: float,
    exclude_top_labels: set[str],
) -> _MaskRecordExtraction:
    default_width = _optional_int(payload.get("width") or payload.get("image_width"))
    default_height = _optional_int(payload.get("height") or payload.get("image_height"))
    default_angle_x = _optional_float(payload.get("camera_angle_x"))
    xyz = positions(cloud)
    records: list[_MaskRecord] = []
    frames = payload["frames"]
    source_masks = 0
    projected_masks = 0
    filtered_low_area = 0
    filtered_top_label = 0
    filtered_empty_projection = 0

    for frame_index, frame in enumerate(frames[:max_frames]):
        if not isinstance(frame, dict):
            raise ValueError("each mask frame must be an object")
        width = _required_int(frame.get("width", default_width), "frame width")
        height = _required_int(frame.get("height", default_height), "frame height")
        camera_angle_x = _required_float(
            frame.get("camera_angle_x", default_angle_x),
            "camera_angle_x",
        )
        transform = _required_matrix(frame.get("transform_matrix"))
        projection = project_points(
            xyz,
            transform_matrix=transform,
            width=width,
            height=height,
            camera_angle_x=camera_angle_x,
        )
        masks = frame.get("masks")
        if not isinstance(masks, list) or not masks:
            raise ValueError("each mask frame must contain a non-empty masks list")
        for mask_index, mask in enumerate(masks):
            if not isinstance(mask, dict):
                raise ValueError("mask entries must be objects")
            source_masks += 1
            area = _mask_pixel_area(mask, root=root, width=width, height=height)
            min_area = max(int(min_mask_area), int(np.ceil(width * height * min_mask_area_fraction)))
            if min_area > 0 and area < min_area:
                filtered_low_area += 1
                continue
            top_label = _top_clip_label(mask)
            if top_label and top_label.lower() in exclude_top_labels:
                filtered_top_label += 1
                continue
            contained = _mask_contains(mask, projection, width, height, root)
            selected = np.flatnonzero(projection.visible & contained)
            if selected.size == 0:
                filtered_empty_projection += 1
                continue
            projected_masks += 1
            source_slot = _required_int(mask.get("slot", mask.get("slot_id")), "mask slot")
            label = str(mask.get("label") or mask.get("name") or f"slot-{source_slot}")
            confidence = float(mask.get("confidence", 1.0) or 1.0)
            records.append(
                _MaskRecord(
                    frame_index=frame_index,
                    mask_index=mask_index,
                    source_slot=source_slot,
                    source_label=label,
                    confidence=confidence,
                    area=area,
                    gaussian_indices=selected.astype(np.int64, copy=False),
                    clip_scores=_clip_scores(mask),
                )
            )
    return _MaskRecordExtraction(
        records=records,
        source_masks=source_masks,
        projected_masks=projected_masks,
        filtered_low_area=filtered_low_area,
        filtered_top_label=filtered_top_label,
        filtered_empty_projection=filtered_empty_projection,
    )


def _cluster_mask_records(
    records: list[_MaskRecord],
    *,
    min_iou: float,
    min_shared_gaussians: int,
) -> list[_SlotCluster]:
    clusters: list[_SlotCluster] = []
    for record in records:
        record_set = set(int(index) for index in record.gaussian_indices)
        best_index = -1
        best_iou = -1.0
        best_shared = 0
        for index, cluster in enumerate(clusters):
            shared = len(record_set & cluster.gaussian_indices)
            union = len(record_set | cluster.gaussian_indices)
            iou = 0.0 if union == 0 else float(shared / union)
            if iou > best_iou:
                best_index = index
                best_iou = iou
                best_shared = shared
        if best_index >= 0 and best_iou >= min_iou and best_shared >= min_shared_gaussians:
            _add_record_to_cluster(clusters[best_index], record, record_set)
        else:
            clusters.append(_new_cluster(record, record_set))
    return clusters


def _new_cluster(record: _MaskRecord, record_set: set[int]) -> _SlotCluster:
    cluster = _SlotCluster(
        records=[],
        gaussian_indices=set(),
        clip_scores={},
        label_weights={},
    )
    _add_record_to_cluster(cluster, record, record_set)
    return cluster


def _add_record_to_cluster(
    cluster: _SlotCluster,
    record: _MaskRecord,
    record_set: set[int],
) -> None:
    cluster.records.append(record)
    cluster.gaussian_indices.update(record_set)
    weight = max(1.0, float(record.area)) * max(0.0, float(record.confidence))
    if _is_semantic_label(record.source_label):
        cluster.label_weights[record.source_label] = (
            cluster.label_weights.get(record.source_label, 0.0) + weight
        )
    for label, score in record.clip_scores.items():
        cluster.clip_scores[label] = cluster.clip_scores.get(label, 0.0) + float(score) * weight


def _order_clusters(clusters: list[_SlotCluster]) -> list[_SlotCluster]:
    return sorted(
        clusters,
        key=lambda cluster: (
            -len(cluster.gaussian_indices),
            min(record.frame_index for record in cluster.records),
            min(record.mask_index for record in cluster.records),
        ),
    )


def _rebalance_slot_support(
    clusters: list[_SlotCluster],
    *,
    min_slot_support_gaussians: int,
    min_slot_support_ratio: float,
    min_balanced_slots: int,
) -> _SlotRebalanceResult:
    if not clusters:
        return _SlotRebalanceResult(
            clusters=(),
            dropped_clusters=(),
            min_slot_support_gaussians=min_slot_support_gaussians,
            min_slot_support_ratio=min_slot_support_ratio,
            min_balanced_slots=min_balanced_slots,
        )
    if min_slot_support_gaussians <= 0 and min_slot_support_ratio <= 0:
        return _SlotRebalanceResult(
            clusters=tuple(clusters),
            dropped_clusters=(),
            min_slot_support_gaussians=min_slot_support_gaussians,
            min_slot_support_ratio=min_slot_support_ratio,
            min_balanced_slots=min_balanced_slots,
        )
    max_support = max(len(cluster.gaussian_indices) for cluster in clusters)
    ratio_floor = int(np.ceil(float(max_support) * float(min_slot_support_ratio)))
    support_floor = max(int(min_slot_support_gaussians), ratio_floor)
    kept: list[_SlotCluster] = []
    dropped: list[dict[str, Any]] = []
    for slot, cluster in enumerate(clusters):
        support = len(cluster.gaussian_indices)
        if support >= support_floor or len(kept) < int(min_balanced_slots):
            kept.append(cluster)
            continue
        dropped.append(
            {
                "source_order": int(slot),
                "support_gaussians": int(support),
                "mask_count": int(len(cluster.records)),
                "frame_count": int(len({record.frame_index for record in cluster.records})),
                "source_slots": sorted({int(record.source_slot) for record in cluster.records}),
                "drop_reason": (
                    f"slot-support-below-threshold:{support}<"
                    f"{support_floor}"
                ),
            }
        )
    return _SlotRebalanceResult(
        clusters=tuple(kept),
        dropped_clusters=tuple(dropped),
        min_slot_support_gaussians=min_slot_support_gaussians,
        min_slot_support_ratio=min_slot_support_ratio,
        min_balanced_slots=min_balanced_slots,
    )


def _recover_foreground_coverage_records(
    dropped_clusters: list[_SlotCluster],
    *,
    kept_clusters: list[_SlotCluster],
    slot_summaries: tuple[dict[str, Any], ...],
    record_to_slot: dict[tuple[int, int], int],
    background_labels: set[str],
    enabled: bool,
    min_shared_gaussians: int,
) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
    summary: dict[str, Any] = {
        "kind": "objgauss-foreground-coverage-recovery-v1",
        "enabled": bool(enabled),
        "candidate_clusters": 0,
        "candidate_masks": 0,
        "recovered_clusters": 0,
        "recovered_masks": 0,
        "recovered_gaussian_support": 0,
        "skipped_background_clusters": 0,
        "skipped_unmatched_clusters": 0,
        "min_shared_gaussians": int(min_shared_gaussians),
        "records": [],
    }
    if not enabled or not dropped_clusters or not kept_clusters:
        return summary, {}

    label_to_slot = _foreground_label_to_slot(
        slot_summaries,
        background_labels=background_labels,
    )
    recovered: dict[tuple[int, int], dict[str, Any]] = {}
    recovered_gaussians: set[int] = set()
    for cluster in dropped_clusters:
        foreground_label = _cluster_foreground_label(cluster, background_labels=background_labels)
        if foreground_label is None:
            summary["skipped_background_clusters"] += 1
            continue
        summary["candidate_clusters"] += 1
        summary["candidate_masks"] += len(cluster.records)
        target_slot = label_to_slot.get(foreground_label.lower())
        reason = "semantic-label-match"
        overlap = _best_overlap_slot(cluster, kept_clusters)
        if target_slot is None:
            if overlap is None or overlap["shared_gaussians"] < min_shared_gaussians:
                summary["skipped_unmatched_clusters"] += 1
                continue
            target_slot = int(overlap["slot"])
            reason = "gaussian-overlap"
        if overlap is None:
            overlap = {"shared_gaussians": 0, "iou": 0.0}
        cluster_gaussians = set(int(index) for index in cluster.gaussian_indices)
        recovered_gaussians.update(cluster_gaussians)
        summary["recovered_clusters"] += 1
        for record in cluster.records:
            key = (record.frame_index, record.mask_index)
            if key in record_to_slot:
                continue
            metadata = {
                "target_slot": int(target_slot),
                "coverage_recovery": {
                    "kind": "foreground-coverage-only-mask-v1",
                    "source_slot": int(record.source_slot),
                    "source_label": record.source_label,
                    "foreground_label": foreground_label,
                    "target_slot": int(target_slot),
                    "reason": reason,
                    "shared_gaussians": int(overlap["shared_gaussians"]),
                    "iou": float(overlap["iou"]),
                    "support_gaussians": int(len(record.gaussian_indices)),
                },
            }
            recovered[key] = metadata
            summary["recovered_masks"] += 1
            summary["records"].append(metadata["coverage_recovery"])
    summary["recovered_gaussian_support"] = int(len(recovered_gaussians))
    return summary, recovered


def _foreground_label_to_slot(
    slot_summaries: tuple[dict[str, Any], ...],
    *,
    background_labels: set[str],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for summary in slot_summaries:
        label = str(summary.get("semantic_label", "")).strip()
        if not label or label.lower() in background_labels:
            continue
        if summary.get("semantic_name_source") == "unnamed":
            continue
        result.setdefault(label.lower(), int(summary["slot"]))
    return result


def _cluster_foreground_label(
    cluster: _SlotCluster,
    *,
    background_labels: set[str],
) -> str | None:
    candidates = _ranked_items(cluster.clip_scores, limit=None)
    if not candidates:
        candidates = _ranked_items(cluster.label_weights, limit=None)
    if not candidates:
        return None
    label = str(candidates[0]["label"]).strip()
    if not label or label.lower() in background_labels:
        return None
    return label


def _best_overlap_slot(
    cluster: _SlotCluster,
    kept_clusters: list[_SlotCluster],
) -> dict[str, Any] | None:
    cluster_set = set(int(index) for index in cluster.gaussian_indices)
    if not cluster_set:
        return None
    best: dict[str, Any] | None = None
    for slot, kept in enumerate(kept_clusters):
        kept_set = kept.gaussian_indices
        shared = len(cluster_set & kept_set)
        union = len(cluster_set | kept_set)
        iou = 0.0 if union == 0 else float(shared / union)
        candidate = {
            "slot": int(slot),
            "shared_gaussians": int(shared),
            "iou": iou,
            "kept_support_gaussians": int(len(kept_set)),
        }
        if best is None or (
            candidate["shared_gaussians"],
            candidate["iou"],
            candidate["kept_support_gaussians"],
            -candidate["slot"],
        ) > (
            best["shared_gaussians"],
            best["iou"],
            best["kept_support_gaussians"],
            -best["slot"],
        ):
            best = candidate
    return best


def _rewrite_aligned_manifest(
    payload: dict[str, Any],
    *,
    record_to_slot: dict[tuple[int, int], int],
    slot_summaries: tuple[dict[str, Any], ...],
    max_frames: int | None,
    source_record_count: int,
    record_filters: dict[str, Any],
    slot_rebalance: dict[str, Any],
    slot_naming_quality: dict[str, Any],
    naming_policy: dict[str, Any],
    coverage_record_metadata: dict[tuple[int, int], dict[str, Any]],
    foreground_coverage_recovery: dict[str, Any],
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    aligned = json.loads(json.dumps(payload))
    frames = aligned.get("frames")
    if not isinstance(frames, list):
        raise ValueError("mask manifest must contain frames")
    source_frames = frames[:max_frames]
    aligned_frames = []
    for frame_index, frame in enumerate(source_frames):
        masks = frame.get("masks") if isinstance(frame, dict) else None
        if not isinstance(masks, list):
            continue
        kept_masks = []
        for mask_index, mask in enumerate(masks):
            stable_slot = record_to_slot.get((frame_index, mask_index))
            if stable_slot is None:
                continue
            source_slot = mask.get("slot", mask.get("slot_id"))
            source_label = mask.get("label", mask.get("name"))
            slot_summary = slot_summaries[stable_slot]
            mask["source_slot"] = source_slot
            mask["source_label"] = source_label
            mask["aligned_slot"] = int(stable_slot)
            mask["slot"] = int(stable_slot)
            mask["slot_id"] = int(stable_slot)
            mask["label"] = slot_summary["semantic_label"]
            mask["semantic_name_source"] = slot_summary["semantic_name_source"]
            mask["semantic_name_policy"] = slot_summary["semantic_name_policy"]
            recovery_metadata = coverage_record_metadata.get((frame_index, mask_index))
            if recovery_metadata is not None:
                mask["coverage_only"] = True
                mask["coverage_recovery"] = recovery_metadata["coverage_recovery"]
            _rewrite_relative_path(
                mask,
                "mask_path",
                source_root=source_root,
                output_root=output_root,
            )
            _rewrite_relative_path(
                mask,
                "ignore_mask_path",
                source_root=source_root,
                output_root=output_root,
            )
            kept_masks.append(mask)
        if not kept_masks:
            continue
        _rewrite_relative_path(
            frame,
            "ignore_mask_path",
            source_root=source_root,
            output_root=output_root,
        )
        frame["masks"] = kept_masks
        aligned_frames.append(frame)

    aligned["frames"] = aligned_frames
    aligned["slots"] = [
        _slot_definition(summary)
        for summary in slot_summaries
    ]
    aligned["slot_count"] = len(slot_summaries)
    aligned["slot_alignment"] = {
        "kind": "objgauss-cross-view-slot-alignment-v1",
        "source_slot_semantics": "per-frame-mask-rank",
        "target_slot_semantics": "cross-view-gaussian-support",
        "source_frames": len(source_frames),
        "frames": len(aligned_frames),
        "masks": len(record_to_slot),
        "dropped_masks": int(source_record_count - len(record_to_slot)),
        "record_filters": record_filters,
        "slot_rebalance": slot_rebalance,
        "foreground_coverage_recovery": foreground_coverage_recovery,
        "naming_policy": naming_policy,
        "slot_naming_quality": slot_naming_quality,
    }
    return aligned


def _rewrite_relative_path(
    entry: dict[str, Any],
    key: str,
    *,
    source_root: Path,
    output_root: Path,
) -> None:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        return
    path = Path(value)
    if path.is_absolute():
        return
    absolute = (source_root / path).resolve()
    entry[key] = os.path.relpath(absolute, output_root.resolve())


def _slot_definition(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "slot": int(summary["slot"]),
        "slot_id": int(summary["slot"]),
        "label": summary["semantic_label"],
        "name": summary["semantic_label"],
        "semantic_name_source": summary["semantic_name_source"],
        "semantic_name_policy": summary["semantic_name_policy"],
        "mask_count": summary["mask_count"],
        "frame_count": summary["frame_count"],
        "support_gaussians": summary["support_gaussians"],
        "source_slots": summary["source_slots"],
        "clip_candidates": summary["clip_candidates"],
        "name_candidates": summary["name_candidates"],
    }


def _cluster_summaries_with_naming_policy(
    clusters: list[_SlotCluster],
    *,
    background_labels: set[str],
    foreground_only_slot_names: bool,
    unique_slot_names: bool,
    slot_name_diversity_penalty: float,
) -> tuple[dict[str, Any], ...]:
    used_labels: Counter[str] = Counter()
    summaries = []
    for slot, cluster in enumerate(clusters):
        selection = _select_semantic_label(
            cluster,
            fallback=f"slot-{slot}",
            background_labels=background_labels,
            foreground_only_slot_names=foreground_only_slot_names,
            unique_slot_names=unique_slot_names,
            slot_name_diversity_penalty=slot_name_diversity_penalty,
            used_labels=used_labels,
        )
        if selection["semantic_name_source"] != "unnamed":
            used_labels[selection["semantic_label"].lower()] += 1
        summaries.append(_cluster_summary(slot, cluster, selection=selection))
    return tuple(summaries)


def _cluster_summary(
    slot: int,
    cluster: _SlotCluster,
    *,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if selection is None:
        selection = _select_semantic_label(
            cluster,
            fallback=f"slot-{slot}",
            background_labels=set(),
            foreground_only_slot_names=False,
            unique_slot_names=False,
            slot_name_diversity_penalty=0.0,
            used_labels=Counter(),
        )
    return {
        "slot": int(slot),
        "semantic_label": selection["semantic_label"],
        "semantic_name_source": selection["semantic_name_source"],
        "semantic_name_policy": selection["semantic_name_policy"],
        "semantic_name_score": selection["semantic_name_score"],
        "semantic_name_adjusted_score": selection["semantic_name_adjusted_score"],
        "semantic_name_raw_rank": selection["semantic_name_raw_rank"],
        "mask_count": len(cluster.records),
        "frame_count": len({record.frame_index for record in cluster.records}),
        "support_gaussians": len(cluster.gaussian_indices),
        "source_slots": sorted({int(record.source_slot) for record in cluster.records}),
        "source_labels": _ranked_items(cluster.label_weights),
        "clip_candidates": _ranked_items(cluster.clip_scores),
        "name_candidates": selection["name_candidates"],
    }


def _select_semantic_label(
    cluster: _SlotCluster,
    *,
    fallback: str,
    background_labels: set[str],
    foreground_only_slot_names: bool,
    unique_slot_names: bool,
    slot_name_diversity_penalty: float,
    used_labels: Counter[str],
) -> dict[str, Any]:
    candidates = _semantic_label_candidates(
        cluster,
        background_labels=background_labels,
        foreground_only_slot_names=foreground_only_slot_names,
        used_labels=used_labels,
        slot_name_diversity_penalty=slot_name_diversity_penalty,
    )
    if unique_slot_names:
        unused = [candidate for candidate in candidates if candidate["used_count"] == 0]
        if unused:
            candidates = unused
    if not candidates:
        return {
            "semantic_label": fallback,
            "semantic_name_source": "unnamed",
            "semantic_name_policy": _semantic_name_policy_label(
                foreground_only_slot_names=foreground_only_slot_names,
                unique_slot_names=unique_slot_names,
                slot_name_diversity_penalty=slot_name_diversity_penalty,
            ),
            "semantic_name_score": 0.0,
            "semantic_name_adjusted_score": 0.0,
            "semantic_name_raw_rank": None,
            "name_candidates": [],
        }
    selected = sorted(
        candidates,
        key=lambda item: (-item["adjusted_score"], item["raw_rank"], item["label"]),
    )[0]
    return {
        "semantic_label": str(selected["label"]),
        "semantic_name_source": str(selected["source"]),
        "semantic_name_policy": _semantic_name_policy_label(
            foreground_only_slot_names=foreground_only_slot_names,
            unique_slot_names=unique_slot_names,
            slot_name_diversity_penalty=slot_name_diversity_penalty,
        ),
        "semantic_name_score": float(selected["score"]),
        "semantic_name_adjusted_score": float(selected["adjusted_score"]),
        "semantic_name_raw_rank": int(selected["raw_rank"]),
        "name_candidates": candidates[:5],
    }


def _semantic_label_candidates(
    cluster: _SlotCluster,
    *,
    background_labels: set[str],
    foreground_only_slot_names: bool,
    used_labels: Counter[str],
    slot_name_diversity_penalty: float,
) -> list[dict[str, Any]]:
    clip_candidates = [
        {
            "label": str(item["label"]),
            "score": float(item["score"]),
            "source": "clip_scores",
            "raw_rank": index,
        }
        for index, item in enumerate(_ranked_items(cluster.clip_scores, limit=None))
    ]
    label_candidates = [
        {
            "label": str(item["label"]),
            "score": float(item["score"]),
            "source": "mask_label_majority",
            "raw_rank": index,
        }
        for index, item in enumerate(_ranked_items(cluster.label_weights, limit=None))
    ]
    raw_candidates = clip_candidates or label_candidates
    if foreground_only_slot_names:
        raw_candidates = [
            candidate
            for candidate in raw_candidates
            if candidate["label"].lower() not in background_labels
        ]
        if not raw_candidates and clip_candidates:
            raw_candidates = [
                candidate
                for candidate in label_candidates
                if candidate["label"].lower() not in background_labels
            ]

    candidates = []
    for candidate in raw_candidates:
        used_count = int(used_labels.get(candidate["label"].lower(), 0))
        adjusted = float(candidate["score"]) / (1.0 + slot_name_diversity_penalty * used_count)
        candidates.append(
            {
                **candidate,
                "used_count": used_count,
                "adjusted_score": adjusted,
            }
        )
    return sorted(candidates, key=lambda item: (-item["adjusted_score"], item["raw_rank"], item["label"]))


def _slot_naming_policy(
    *,
    background_labels: set[str],
    foreground_only_slot_names: bool,
    unique_slot_names: bool,
    slot_name_diversity_penalty: float,
) -> dict[str, Any]:
    return {
        "kind": "objgauss-slot-naming-policy-v1",
        "foreground_only_slot_names": bool(foreground_only_slot_names),
        "unique_slot_names": bool(unique_slot_names),
        "slot_name_diversity_penalty": float(slot_name_diversity_penalty),
        "background_labels": sorted(background_labels),
    }


def _semantic_name_policy_label(
    *,
    foreground_only_slot_names: bool,
    unique_slot_names: bool,
    slot_name_diversity_penalty: float,
) -> str:
    parts = ["clip-slot-naming"]
    if foreground_only_slot_names:
        parts.append("foreground-only")
    if unique_slot_names:
        parts.append("unique")
    if slot_name_diversity_penalty > 0:
        parts.append(f"diversity-penalty:{slot_name_diversity_penalty:g}")
    return ":".join(parts)


def _support_balance_score(counts: list[int]) -> float:
    active = [int(count) for count in counts if int(count) > 0]
    if not active:
        return 0.0
    max_count = max(active)
    if max_count <= 0:
        return 0.0
    return float(min(active) / max_count)


def _slot_naming_quality(
    slot_summaries: tuple[dict[str, Any], ...],
    *,
    background_labels: set[str],
    min_named_slots: int,
    min_unique_slot_labels: int,
    max_slot_label_fraction: float,
    max_background_slot_fraction: float,
) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    background_count = 0
    for summary in slot_summaries:
        label = str(summary["semantic_label"])
        source = str(summary["semantic_name_source"])
        source_counts[source] += 1
        if source == "unnamed":
            continue
        label_counts[label] += 1
        if label.lower() in background_labels:
            background_count += 1

    named_slots = int(sum(label_counts.values()))
    unique_slot_labels = int(len(label_counts))
    max_label = ""
    max_count = 0
    if label_counts:
        max_label, max_count = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[0]
    max_fraction = float(max_count / named_slots) if named_slots else 0.0
    background_fraction = float(background_count / named_slots) if named_slots else 0.0

    blockers: list[str] = []
    if named_slots < min_named_slots:
        blockers.append("not-enough-named-slots")
    if unique_slot_labels < min_unique_slot_labels:
        blockers.append("not-enough-unique-slot-labels")
    if max_fraction > max_slot_label_fraction:
        blockers.append(f"slot-label-dominant:{max_label}")
    if background_fraction > max_background_slot_fraction:
        blockers.append("background-slot-dominant")

    return {
        "kind": "objgauss-slot-naming-quality-v1",
        "passed": not blockers,
        "blockers": blockers,
        "slots": int(len(slot_summaries)),
        "named_slots": named_slots,
        "unique_slot_labels": unique_slot_labels,
        "slot_label_counts": dict(sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))),
        "semantic_name_sources": dict(sorted(source_counts.items())),
        "max_slot_label": max_label,
        "max_slot_label_count": int(max_count),
        "max_slot_label_fraction": max_fraction,
        "background_labels": sorted(background_labels),
        "background_slot_count": int(background_count),
        "background_slot_fraction": background_fraction,
        "thresholds": {
            "min_named_slots": int(min_named_slots),
            "min_unique_slot_labels": int(min_unique_slot_labels),
            "max_slot_label_fraction": float(max_slot_label_fraction),
            "max_background_slot_fraction": float(max_background_slot_fraction),
        },
    }


def _ranked_items(scores: dict[str, float], *, limit: int | None = 5) -> list[dict[str, Any]]:
    items = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        items = items[:limit]
    return [
        {"label": label, "score": float(score)}
        for label, score in items
    ]


def _top_clip_label(mask: dict[str, Any]) -> str | None:
    scores = _clip_scores(mask)
    if not scores:
        return None
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _is_semantic_label(label: str) -> bool:
    text = label.strip().lower()
    return bool(text) and not text.startswith("sam-area-rank-")


def _clip_scores(mask: dict[str, Any]) -> dict[str, float]:
    for key in ("clip_scores", "clip_label_scores", "semantic_scores"):
        value = mask.get(key)
        if isinstance(value, dict):
            return _numeric_scores(value)
    clip = mask.get("clip")
    if isinstance(clip, dict) and isinstance(clip.get("scores"), dict):
        return _numeric_scores(clip["scores"])
    return {}


def _numeric_scores(value: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label, score in value.items():
        try:
            numeric = float(score)
        except Exception:
            continue
        if np.isfinite(numeric):
            scores[str(label)] = numeric
    return scores


def _clean_label_set(labels: list[str] | tuple[str, ...]) -> set[str]:
    return {str(label).strip().lower() for label in labels if str(label).strip()}


def _mask_pixel_area(mask: dict[str, Any], *, root: Path, width: int, height: int) -> int:
    raw_area = mask.get("area", mask.get("pixels"))
    if raw_area is not None:
        return int(raw_area)
    if "rect" in mask:
        x0, y0, x1, y1 = _required_rect(mask["rect"])
        x0 = max(0.0, min(float(width), x0))
        x1 = max(0.0, min(float(width), x1))
        y0 = max(0.0, min(float(height), y0))
        y1 = max(0.0, min(float(height), y1))
        return int(max(0.0, x1 - x0) * max(0.0, y1 - y0))
    if "mask_path" in mask:
        array = np.load(root / str(mask["mask_path"]))
        if array.shape != (height, width):
            raise ValueError(f"mask {mask['mask_path']} shape {array.shape} does not match {height}x{width}")
        return int(np.count_nonzero(array))
    raise ValueError("mask entry must include rect or mask_path")


def _mask_contains(
    mask: dict[str, Any],
    projection: Any,
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
