from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from objgauss.core.object_state import (
    ObjectStateProjection,
    track_object_state_projection,
)
from objgauss.pipelines.trainable_kernel import TrainableKernelResult, TrainableKernelSample

TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA = "objgauss-trainable-kernel-model-artifact-v1"

_CANONICAL_STATE_ADDRESS_KEYS = frozenset(
    ("persistent_id", "slot", "object_id")
)
_CANONICAL_STATE_LAYOUT = "canonical_persistent_id_and_renderer_slot"
_LEGACY_STATE_LAYOUT = "legacy_id_only"


def trainable_kernel_model_artifact(
    result: TrainableKernelResult,
    *,
    sample: TrainableKernelSample | None = None,
    input_path: str | Path | None = None,
    renderer_api: dict[str, Any] | None = None,
    label: str = "trainable-kernel-mvp",
) -> dict[str, Any]:
    tracked_projections = _tracked_object_state_projections(
        result.object_state_projections
    )
    training_summary = result.as_dict()
    training_summary["object_states"] = [
        [
            {
                "id": int(state.id),
                "persistent_id": int(state.id),
                "slot": int(state.slot),
                "object_id": int(state.slot),
                "slot_mass": float(state.slot_mass),
                "confidence": float(state.confidence),
                "centroid": _round_array(state.centroid),
                "status": state.status,
            }
            for state in projection.states
        ]
        for projection in tracked_projections
    ]
    artifact = {
        "schema": TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
        "kind": "trainable_kernel_mvp_model",
        "label": label,
        "source": {
            "input": str(input_path) if input_path is not None else None,
            "sample": sample.as_dict() if sample is not None else None,
        },
        "training": training_summary,
        "renderer_api": renderer_api,
        "learned_parameters": {
            "decoder_colors": _round_array(result.decoder_colors),
        },
        "assignments": [
            {
                "frame_index": frame_index,
                "shape": [int(value) for value in assignment.shape],
                "matrix": _round_array(assignment),
            }
            for frame_index, assignment in enumerate(result.assignments)
        ],
        "object_states": [
            {
                "frame_index": frame_index,
                "states": [_object_state_summary(state) for state in projection.states],
                "derived_object_ids": projection.derived_object_ids.astype(int).tolist(),
            }
            for frame_index, projection in enumerate(tracked_projections)
        ],
        "rendered_rgb": [
            {
                "frame_index": frame_index,
                "shape": [int(value) for value in rendered.shape],
                "rgb": _round_array(rendered),
            }
            for frame_index, rendered in enumerate(result.rendered_rgb)
        ],
        "artifact_policy": {
            "intended_location": "user-selected path or ignored outputs/",
            "git_policy": "do_not_commit_training_outputs_by_default",
            "viewer_policy": "not a browser artifact until explicitly converted",
        },
    }
    validate_trainable_kernel_model_artifact(artifact)
    return artifact


def write_trainable_kernel_model_artifact(
    path: str | Path,
    result: TrainableKernelResult,
    *,
    sample: TrainableKernelSample | None = None,
    input_path: str | Path | None = None,
    renderer_api: dict[str, Any] | None = None,
    label: str = "trainable-kernel-mvp",
) -> dict[str, Any]:
    artifact = trainable_kernel_model_artifact(
        result,
        sample=sample,
        input_path=input_path,
        renderer_api=renderer_api,
        label=label,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def validate_trainable_kernel_model_artifact(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        raise TypeError("trainable kernel model artifact must be a dict")
    if payload.get("schema") != TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA:
        raise ValueError(f"unsupported trainable model artifact schema: {payload.get('schema')}")
    required = (
        "kind",
        "source",
        "training",
        "learned_parameters",
        "assignments",
        "object_states",
        "artifact_policy",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"trainable model artifact missing keys: {', '.join(missing)}")
    if payload.get("kind") != "trainable_kernel_mvp_model":
        raise ValueError("trainable model artifact kind is unsupported")
    training = payload["training"]
    if not isinstance(training, dict) or training.get("schema") != "objgauss-v1-trainable-kernel-mvp-v1":
        raise ValueError("trainable model artifact training summary is invalid")
    assignments = payload["assignments"]
    if not isinstance(assignments, list) or len(assignments) != int(training.get("frame_count", -1)):
        raise ValueError("trainable model artifact assignments must match frame_count")
    states = payload["object_states"]
    if not isinstance(states, list) or len(states) != int(training.get("frame_count", -1)):
        raise ValueError("trainable model artifact object_states must match frame_count")
    state_layout = _validate_artifact_object_state_frames(states)
    training_states = training.get("object_states")
    if training_states is not None:
        if (
            not isinstance(training_states, list)
            or len(training_states) != int(training.get("frame_count", -1))
        ):
            raise ValueError(
                "trainable model artifact training object_states must match frame_count"
            )
        training_state_layout = _validate_training_object_state_frames(
            training_states
        )
        if (
            state_layout is not None
            and training_state_layout is not None
            and training_state_layout != state_layout
        ):
            raise ValueError(
                "trainable model artifact object_states cannot mix canonical and legacy layouts"
            )
    return True


def _validate_artifact_object_state_frames(
    frames: list[Any],
) -> str | None:
    layout: str | None = None
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise TypeError(
                "trainable model artifact object_states entries must be dicts"
            )
        states = frame.get("states")
        if not isinstance(states, list):
            raise TypeError(
                "trainable model artifact object_states frame states must be a list"
            )
        layout = _validate_object_state_frame(
            states,
            frame_index=frame_index,
            artifact_layout=layout,
        )
    return layout


def _validate_training_object_state_frames(
    frames: list[Any],
) -> str | None:
    layout: str | None = None
    for frame_index, states in enumerate(frames):
        if not isinstance(states, list):
            raise TypeError(
                "trainable model artifact training object_states frames must be lists"
            )
        layout = _validate_object_state_frame(
            states,
            frame_index=frame_index,
            artifact_layout=layout,
        )
    return layout


def _validate_object_state_frame(
    states: list[Any],
    *,
    frame_index: int,
    artifact_layout: str | None,
) -> str | None:
    persistent_ids: set[int] = set()
    slots: set[int] = set()
    layout = artifact_layout
    for state in states:
        state_layout, persistent_id, slot = _validate_object_state_abi(state)
        if layout is None:
            layout = state_layout
        elif state_layout != layout:
            raise ValueError(
                "trainable model artifact object_states cannot mix canonical and legacy layouts"
            )
        if persistent_id in persistent_ids:
            raise ValueError(
                "trainable model artifact ObjectState id must be unique within "
                f"frame {frame_index}"
            )
        persistent_ids.add(persistent_id)
        if slot is not None:
            if slot in slots:
                raise ValueError(
                    "trainable model artifact ObjectState slot must be unique within "
                    f"frame {frame_index}"
                )
            slots.add(slot)
    return layout


def _validate_object_state_abi(state: Any) -> tuple[str, int, int | None]:
    if not isinstance(state, dict):
        raise TypeError("trainable model artifact ObjectState must be a dict")
    persistent_id = _non_negative_state_integer(state.get("id"), "id")
    _validate_state_confidence(state)

    present_address_keys = _CANONICAL_STATE_ADDRESS_KEYS.intersection(state)
    if not present_address_keys:
        return _LEGACY_STATE_LAYOUT, persistent_id, None
    if present_address_keys != _CANONICAL_STATE_ADDRESS_KEYS:
        raise ValueError(
            "canonical trainable model artifact ObjectState requires "
            "persistent_id, slot, and object_id together"
        )

    persistent_alias = _non_negative_state_integer(
        state["persistent_id"],
        "persistent_id",
    )
    slot = _non_negative_state_integer(state["slot"], "slot")
    object_id = _non_negative_state_integer(state["object_id"], "object_id")
    if persistent_id != persistent_alias:
        raise ValueError(
            "canonical trainable model artifact ObjectState id must match persistent_id"
        )
    if slot != object_id:
        raise ValueError(
            "canonical trainable model artifact ObjectState slot must match object_id"
        )
    return _CANONICAL_STATE_LAYOUT, persistent_id, slot


def _non_negative_state_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"trainable model artifact ObjectState {name} must be an integer"
        )
    if value < 0:
        raise ValueError(
            f"trainable model artifact ObjectState {name} must be non-negative"
        )
    return int(value)


def _validate_state_confidence(state: dict[str, Any]) -> float:
    if "confidence" not in state:
        raise ValueError(
            "trainable model artifact ObjectState requires confidence"
        )
    confidence = state["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError(
            "trainable model artifact ObjectState confidence must be numeric"
        )
    value = float(confidence)
    if not np.isfinite(value):
        raise ValueError(
            "trainable model artifact ObjectState confidence must be finite"
        )
    if value < 0.0 or value > 1.0:
        raise ValueError(
            "trainable model artifact ObjectState confidence must be in [0, 1]"
        )
    return value


def _object_state_summary(state: Any) -> dict[str, Any]:
    return {
        "id": int(state.id),
        "persistent_id": int(state.id),
        "slot": int(state.slot),
        "object_id": int(state.slot),
        "slot_mass": float(state.slot_mass),
        "confidence": float(state.confidence),
        "mass_fraction": float(state.mass_fraction),
        "assignment_entropy": float(state.assignment_entropy),
        "normalized_assignment_entropy": float(state.normalized_assignment_entropy),
        "centroid": _round_array(state.centroid),
        "bbox": _round_array(state.bbox),
        "feature": _round_array(state.feature),
        "status": state.status,
        "diagnostics": list(state.diagnostics),
    }


def _tracked_object_state_projections(
    projections: tuple[ObjectStateProjection, ...],
) -> tuple[ObjectStateProjection, ...]:
    tracked: list[ObjectStateProjection] = []
    previous: ObjectStateProjection | None = None
    for projection in projections:
        current = (
            projection
            if previous is None
            else track_object_state_projection(
                projection,
                previous_state=previous,
            )
        )
        tracked.append(current)
        previous = current
    return tuple(tracked)


def _round_array(value: Any) -> list[Any]:
    return np.round(np.asarray(value), 6).tolist()


__all__ = (
    "TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA",
    "trainable_kernel_model_artifact",
    "write_trainable_kernel_model_artifact",
    "validate_trainable_kernel_model_artifact",
)
