from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from objgauss.core.objectstate_controlled_capture import (
    validate_objectstate_controlled_capture_manifest,
)
from objgauss.core.objectstate_controlled_identity_eval import (
    OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA,
    validate_objectstate_controlled_identity_predictions,
)
from objgauss.core.trainable_artifact import validate_trainable_kernel_model_artifact


def read_trainable_kernel_identity_source(path: str | Path) -> dict[str, Any]:
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("trainable kernel identity source JSON must be an object")
    validate_trainable_kernel_model_artifact(payload)
    return payload


def objectstate_identity_predictions_from_trainable_artifact(
    capture_manifest: Mapping[str, Any],
    model_artifact: Mapping[str, Any],
    *,
    candidate_id: str | None = None,
    source: str = "trainable_kernel_objectstate_nearest_pose_adapter",
    artifact_refs: Sequence[str] | None = None,
    max_centroid_distance: float | None = None,
) -> dict[str, Any]:
    capture = validate_objectstate_controlled_capture_manifest(capture_manifest)
    artifact = _validate_trainable_artifact_mapping(model_artifact)
    max_distance = _validate_max_distance(max_centroid_distance)
    frames = capture["frames"]
    artifact_frames = _artifact_frames(artifact)
    if len(artifact_frames) != len(frames):
        raise ValueError(
            "trainable artifact object_states frame count must match capture frames"
        )

    predictions: list[dict[str, Any]] = []
    for frame_index, (capture_frame, artifact_frame) in enumerate(
        zip(frames, artifact_frames)
    ):
        artifact_index = int(artifact_frame.get("frame_index", frame_index))
        if artifact_index != frame_index:
            raise ValueError(
                "trainable artifact object_states frame_index must match capture frame order"
            )
        states = _frame_states(artifact_frame, frame_index=frame_index)
        for frame_object in capture_frame["objects"]:
            position = _frame_object_position(frame_object)
            match = _nearest_state(position, states)
            if max_distance is not None and match["distance"] > max_distance:
                continue
            predictions.append(
                {
                    "frame_id": capture_frame["frame_id"],
                    "object_id": frame_object["object_id"],
                    "predicted_identity": f"slot-{match['state_id']}",
                    "confidence": match["confidence"],
                }
            )

    candidate = {
        "candidate_id": _candidate_id(candidate_id, artifact),
        "source": str(source),
        "artifact_refs": _artifact_refs(artifact_refs, artifact),
    }
    identity_evidence = _identity_evidence(artifact.get("identity_evidence"))
    if identity_evidence is not None:
        candidate["identity_evidence"] = identity_evidence

    payload = {
        "schema": OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA,
        "sample_id": capture["sample"]["sample_id"],
        "candidate": candidate,
        "predictions": predictions,
    }
    return validate_objectstate_controlled_identity_predictions(payload)


def _validate_trainable_artifact_mapping(model_artifact: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(model_artifact, Mapping):
        raise TypeError("trainable artifact must be a mapping")
    artifact = dict(model_artifact)
    validate_trainable_kernel_model_artifact(artifact)
    return artifact


def _validate_max_distance(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("max_centroid_distance must be numeric")
    distance = float(value)
    if distance < 0.0:
        raise ValueError("max_centroid_distance must be >= 0")
    return distance


def _artifact_frames(artifact: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    frames = artifact.get("object_states")
    if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
        raise TypeError("trainable artifact object_states must be a sequence")
    return tuple(_mapping(item, "trainable artifact object_states entries") for item in frames)


def _frame_states(
    artifact_frame: Mapping[str, Any],
    *,
    frame_index: int,
) -> tuple[dict[str, Any], ...]:
    states = artifact_frame.get("states")
    if isinstance(states, (str, bytes)) or not isinstance(states, Sequence):
        raise TypeError("trainable artifact object_states frame states must be a sequence")
    normalized = tuple(_state(item) for item in states)
    if not normalized:
        raise ValueError(f"trainable artifact frame {frame_index} requires states")
    return normalized


def _state(value: Any) -> dict[str, Any]:
    item = _mapping(value, "trainable artifact ObjectState")
    state_id = item.get("id")
    if isinstance(state_id, bool) or not isinstance(state_id, int):
        raise ValueError("trainable artifact ObjectState id must be an integer")
    centroid = _vector(item.get("centroid"), "trainable artifact ObjectState centroid")
    confidence = item.get("confidence", 1.0)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("trainable artifact ObjectState confidence must be numeric")
    confidence_value = float(confidence)
    if confidence_value < 0.0 or confidence_value > 1.0:
        raise ValueError("trainable artifact ObjectState confidence must be in [0, 1]")
    return {
        "id": int(state_id),
        "centroid": centroid,
        "confidence": confidence_value,
    }


def _frame_object_position(frame_object: Mapping[str, Any]) -> np.ndarray:
    pose = frame_object.get("pose")
    if not isinstance(pose, Mapping):
        raise ValueError(
            "trainable artifact identity adapter requires pose.position for every capture object"
        )
    return _vector(pose.get("position"), "capture object pose.position")


def _nearest_state(
    position: np.ndarray,
    states: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    distances = [
        float(np.linalg.norm(position - np.asarray(state["centroid"], dtype=np.float32)))
        for state in states
    ]
    state_index = int(np.argmin(np.asarray(distances, dtype=np.float32)))
    state = states[state_index]
    return {
        "state_id": int(state["id"]),
        "confidence": float(state["confidence"]),
        "distance": distances[state_index],
    }


def _candidate_id(candidate_id: str | None, artifact: Mapping[str, Any]) -> str:
    if candidate_id is not None:
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        return candidate_id
    label = artifact.get("label")
    if isinstance(label, str) and label:
        return label
    return "trainable-kernel-objectstate-candidate"


def _artifact_refs(
    artifact_refs: Sequence[str] | None,
    artifact: Mapping[str, Any],
) -> tuple[str, ...]:
    if artifact_refs is not None:
        if isinstance(artifact_refs, (str, bytes)) or not isinstance(
            artifact_refs, Sequence
        ):
            raise TypeError("artifact_refs must be a sequence of strings")
        refs = tuple(str(item) for item in artifact_refs)
    else:
        source = artifact.get("source")
        source_input = source.get("input") if isinstance(source, Mapping) else None
        refs = (
            (source_input,)
            if isinstance(source_input, str) and source_input
            else ("memory://trainable-kernel-model-artifact",)
        )
    if not refs or any(not item for item in refs):
        raise ValueError("artifact_refs must contain non-empty strings")
    return refs


def _identity_evidence(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("trainable artifact identity_evidence must be a mapping")
    result = dict(value)
    result.setdefault("source", "trainable_kernel_model_artifact.identity_evidence")
    return result


def _vector(value: Any, name: str) -> np.ndarray:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a length-3 sequence")
    if len(value) != 3:
        raise ValueError(f"{name} must have length 3")
    vector = np.asarray(value, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain finite values")
    return vector


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value
