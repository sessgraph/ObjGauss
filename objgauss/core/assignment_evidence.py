from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.object_state import validate_assignment_matrix

ASSIGNMENT_EVIDENCE_BATCH_SCHEMA = "objgauss-assignment-evidence-batch-v1"


@dataclass(frozen=True)
class AssignmentEvidenceBatch:
    positions: np.ndarray
    features: np.ndarray
    frame_index: int = 0
    mask_votes: np.ndarray | None = None
    track_hints: np.ndarray | None = None
    target_assignment: np.ndarray | None = None
    source: str = "unknown"
    schema: str = ASSIGNMENT_EVIDENCE_BATCH_SCHEMA

    @property
    def evidence_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def as_dict(self) -> dict[str, Any]:
        checked = validate_assignment_evidence_batch(self)
        return {
            "schema": checked.schema,
            "kind": "assignment_evidence_batch",
            "source": checked.source,
            "frame_index": int(checked.frame_index),
            "evidence_count": int(checked.positions.shape[0]),
            "feature_dim": int(checked.features.shape[1]),
            "position_dim": int(checked.positions.shape[1]),
            "has_mask_votes": checked.mask_votes is not None,
            "mask_vote_dim": None if checked.mask_votes is None else int(checked.mask_votes.shape[1]),
            "has_track_hints": checked.track_hints is not None,
            "has_target_assignment": checked.target_assignment is not None,
            "target_slots": None
            if checked.target_assignment is None
            else int(checked.target_assignment.shape[1]),
            "shapes": {
                "positions": list(checked.positions.shape),
                "features": list(checked.features.shape),
                "mask_votes": None if checked.mask_votes is None else list(checked.mask_votes.shape),
                "track_hints": None if checked.track_hints is None else list(checked.track_hints.shape),
                "target_assignment": None
                if checked.target_assignment is None
                else list(checked.target_assignment.shape),
            },
        }


def assignment_evidence_from_trainable_frame(
    frame: Any,
    *,
    frame_index: int = 0,
    source: str = "trainable_kernel_frame",
    mask_votes: np.ndarray | None = None,
    track_hints: np.ndarray | None = None,
) -> AssignmentEvidenceBatch:
    return validate_assignment_evidence_batch(
        AssignmentEvidenceBatch(
            positions=np.asarray(frame.positions, dtype=np.float32),
            features=np.asarray(frame.features, dtype=np.float32),
            frame_index=int(frame_index),
            mask_votes=mask_votes,
            track_hints=track_hints,
            target_assignment=frame.target_assignment,
            source=source,
        )
    )


def assignment_evidence_sequence_from_trainable_frames(
    frames: Sequence[Any],
    *,
    source: str = "trainable_kernel_frame",
) -> tuple[AssignmentEvidenceBatch, ...]:
    return tuple(
        assignment_evidence_from_trainable_frame(
            frame,
            frame_index=index,
            source=source,
        )
        for index, frame in enumerate(frames)
    )


def assignment_evidence_from_object_emergence(
    evidence: Any,
    *,
    source: str | None = None,
    mask_votes: np.ndarray | None = None,
    track_hints: np.ndarray | None = None,
) -> AssignmentEvidenceBatch:
    return validate_assignment_evidence_batch(
        AssignmentEvidenceBatch(
            positions=np.asarray(evidence.positions, dtype=np.float32),
            features=np.asarray(evidence.features, dtype=np.float32),
            frame_index=int(evidence.frame_index),
            mask_votes=mask_votes,
            track_hints=track_hints,
            target_assignment=evidence.target_assignment,
            source=source or f"object_emergence_evidence:{evidence.source}",
        )
    )


def validate_assignment_evidence_batch(batch: AssignmentEvidenceBatch) -> AssignmentEvidenceBatch:
    if not isinstance(batch, AssignmentEvidenceBatch):
        raise TypeError("assignment evidence batch must be AssignmentEvidenceBatch")
    if batch.schema != ASSIGNMENT_EVIDENCE_BATCH_SCHEMA:
        raise ValueError(f"unsupported assignment evidence schema: {batch.schema}")
    positions = _array2d(batch.positions, "positions", columns=3)
    features = _array2d(batch.features, "features")
    if features.shape[0] != positions.shape[0]:
        raise ValueError("features rows must match positions")
    mask_votes = None if batch.mask_votes is None else _array2d(batch.mask_votes, "mask_votes")
    if mask_votes is not None:
        if mask_votes.shape[0] != positions.shape[0]:
            raise ValueError("mask_votes rows must match positions")
        if np.any(mask_votes < 0.0):
            raise ValueError("mask_votes must be non-negative")
    track_hints = None if batch.track_hints is None else _track_hints_array(batch.track_hints)
    if track_hints is not None and track_hints.shape[0] != positions.shape[0]:
        raise ValueError("track_hints length must match positions")
    target_assignment = None
    if batch.target_assignment is not None:
        target_assignment = validate_assignment_matrix(
            batch.target_assignment,
            evidence_count=positions.shape[0],
        )
    return AssignmentEvidenceBatch(
        positions=positions,
        features=features,
        frame_index=int(batch.frame_index),
        mask_votes=mask_votes,
        track_hints=track_hints,
        target_assignment=target_assignment,
        source=str(batch.source),
        schema=batch.schema,
    )


def validate_assignment_evidence_summary(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        raise TypeError("assignment evidence summary must be a dict")
    if payload.get("schema") != ASSIGNMENT_EVIDENCE_BATCH_SCHEMA:
        raise ValueError(f"unsupported assignment evidence schema: {payload.get('schema')}")
    for key in ("evidence_count", "feature_dim", "position_dim", "shapes"):
        if key not in payload:
            raise ValueError(f"assignment evidence summary missing {key}")
    if int(payload["evidence_count"]) < 1:
        raise ValueError("assignment evidence summary must contain evidence")
    if int(payload["feature_dim"]) < 1:
        raise ValueError("assignment evidence feature_dim must be >= 1")
    shapes = payload["shapes"]
    if not isinstance(shapes, dict):
        raise ValueError("assignment evidence summary shapes must be an object")
    return True


def _array2d(value: np.ndarray, label: str, *, columns: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array")
    if array.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one row")
    if array.shape[1] == 0:
        raise ValueError(f"{label} must contain at least one column")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{label} must have {columns} columns")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array.astype(np.float32, copy=False)


def _track_hints_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError("track_hints must be a 1D array")
    if array.shape[0] == 0:
        raise ValueError("track_hints must contain at least one value")
    if not np.isfinite(array.astype(np.float32)).all():
        raise ValueError("track_hints must contain only finite values")
    return array.astype(np.int64, copy=False)
