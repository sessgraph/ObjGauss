from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from objgauss.features import positions
from objgauss.gaussians import GaussianCloud
from objgauss.mask_voting import project_points


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


def align_mask_manifest_slots(
    cloud: GaussianCloud,
    manifest_path: str | Path,
    *,
    output: str | Path,
    min_iou: float = 0.05,
    min_shared_gaussians: int = 1,
    max_slots: int | None = None,
    max_frames: int | None = None,
) -> SlotAlignmentResult:
    if min_iou < 0 or min_iou > 1:
        raise ValueError("min_iou must be in [0, 1]")
    if min_shared_gaussians < 0:
        raise ValueError("min_shared_gaussians must be >= 0")
    if max_slots is not None and max_slots < 1:
        raise ValueError("max_slots must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")

    manifest_path = Path(manifest_path)
    output = Path(output)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("mask manifest must contain a non-empty frames list")

    root = manifest_path.parent
    records = _mask_records_for_manifest(cloud, payload, root=root, max_frames=max_frames)
    if not records:
        raise ValueError("mask manifest did not produce any projected mask records")

    clusters = _cluster_mask_records(
        records,
        min_iou=float(min_iou),
        min_shared_gaussians=int(min_shared_gaussians),
    )
    ordered_clusters = _order_clusters(clusters)
    if max_slots is not None:
        ordered_clusters = ordered_clusters[:max_slots]
    record_to_slot = {
        (record.frame_index, record.mask_index): slot
        for slot, cluster in enumerate(ordered_clusters)
        for record in cluster.records
    }

    aligned_payload = _rewrite_aligned_manifest(
        payload,
        record_to_slot=record_to_slot,
        clusters=ordered_clusters,
        max_frames=max_frames,
        source_record_count=len(records),
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

    summaries = tuple(
        _cluster_summary(slot, cluster)
        for slot, cluster in enumerate(ordered_clusters)
    )
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
    )


def _mask_records_for_manifest(
    cloud: GaussianCloud,
    payload: dict[str, Any],
    *,
    root: Path,
    max_frames: int | None,
) -> list[_MaskRecord]:
    default_width = _optional_int(payload.get("width") or payload.get("image_width"))
    default_height = _optional_int(payload.get("height") or payload.get("image_height"))
    default_angle_x = _optional_float(payload.get("camera_angle_x"))
    xyz = positions(cloud)
    records: list[_MaskRecord] = []
    frames = payload["frames"]

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
            contained = _mask_contains(mask, projection, width, height, root)
            selected = np.flatnonzero(projection.visible & contained)
            if selected.size == 0:
                continue
            source_slot = _required_int(mask.get("slot", mask.get("slot_id")), "mask slot")
            label = str(mask.get("label") or mask.get("name") or f"slot-{source_slot}")
            confidence = float(mask.get("confidence", 1.0) or 1.0)
            area = int(mask.get("area", selected.size) or selected.size)
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
    return records


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


def _rewrite_aligned_manifest(
    payload: dict[str, Any],
    *,
    record_to_slot: dict[tuple[int, int], int],
    clusters: list[_SlotCluster],
    max_frames: int | None,
    source_record_count: int,
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
            slot_summary = _slot_definition(stable_slot, clusters[stable_slot])
            mask["source_slot"] = source_slot
            mask["source_label"] = source_label
            mask["aligned_slot"] = int(stable_slot)
            mask["slot"] = int(stable_slot)
            mask["slot_id"] = int(stable_slot)
            mask["label"] = slot_summary["label"]
            mask["semantic_name_source"] = slot_summary["semantic_name_source"]
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
        _slot_definition(slot, cluster)
        for slot, cluster in enumerate(clusters)
    ]
    aligned["slot_count"] = len(clusters)
    aligned["slot_alignment"] = {
        "kind": "objgauss-cross-view-slot-alignment-v1",
        "source_slot_semantics": "per-frame-mask-rank",
        "target_slot_semantics": "cross-view-gaussian-support",
        "source_frames": len(source_frames),
        "frames": len(aligned_frames),
        "masks": len(record_to_slot),
        "dropped_masks": int(source_record_count - len(record_to_slot)),
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


def _slot_definition(slot: int, cluster: _SlotCluster) -> dict[str, Any]:
    summary = _cluster_summary(slot, cluster)
    return {
        "slot": int(slot),
        "slot_id": int(slot),
        "label": summary["semantic_label"],
        "name": summary["semantic_label"],
        "semantic_name_source": summary["semantic_name_source"],
        "mask_count": summary["mask_count"],
        "frame_count": summary["frame_count"],
        "support_gaussians": summary["support_gaussians"],
        "source_slots": summary["source_slots"],
        "clip_candidates": summary["clip_candidates"],
    }


def _cluster_summary(slot: int, cluster: _SlotCluster) -> dict[str, Any]:
    semantic_label, source = _semantic_label(cluster, fallback=f"slot-{slot}")
    return {
        "slot": int(slot),
        "semantic_label": semantic_label,
        "semantic_name_source": source,
        "mask_count": len(cluster.records),
        "frame_count": len({record.frame_index for record in cluster.records}),
        "support_gaussians": len(cluster.gaussian_indices),
        "source_slots": sorted({int(record.source_slot) for record in cluster.records}),
        "source_labels": _ranked_items(cluster.label_weights),
        "clip_candidates": _ranked_items(cluster.clip_scores),
    }


def _semantic_label(cluster: _SlotCluster, *, fallback: str) -> tuple[str, str]:
    clip_candidates = _ranked_items(cluster.clip_scores)
    if clip_candidates:
        return str(clip_candidates[0]["label"]), "clip_scores"
    label_candidates = _ranked_items(cluster.label_weights)
    if label_candidates:
        return str(label_candidates[0]["label"]), "mask_label_majority"
    return fallback, "unnamed"


def _ranked_items(scores: dict[str, float], *, limit: int = 5) -> list[dict[str, Any]]:
    return [
        {"label": label, "score": float(score)}
        for label, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


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
