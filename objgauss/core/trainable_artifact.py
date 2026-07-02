from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from objgauss.core.trainable_kernel import TrainableKernelResult, TrainableKernelSample

TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA = "objgauss-trainable-kernel-model-artifact-v1"


def trainable_kernel_model_artifact(
    result: TrainableKernelResult,
    *,
    sample: TrainableKernelSample | None = None,
    input_path: str | Path | None = None,
    renderer_api: dict[str, Any] | None = None,
    label: str = "trainable-kernel-mvp",
) -> dict[str, Any]:
    training_summary = result.as_dict()
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
            for frame_index, projection in enumerate(result.object_state_projections)
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
    return True


def _object_state_summary(state: Any) -> dict[str, Any]:
    return {
        "id": int(state.id),
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


def _round_array(value: Any) -> list[Any]:
    return np.round(np.asarray(value), 6).tolist()
