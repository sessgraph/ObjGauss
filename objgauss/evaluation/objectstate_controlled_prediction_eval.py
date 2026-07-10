from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.datasets.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    objectstate_controlled_capture_summary,
    objectstate_controlled_real_manifest_from_capture_manifest,
    validate_objectstate_controlled_capture_manifest,
)
from objgauss.datasets.objectstate_controlled_real_manifest import (
    validate_objectstate_controlled_real_manifest,
)

OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA = (
    "objgauss-objectstate-controlled-prediction-candidates-v1"
)
OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA = (
    "objgauss-objectstate-controlled-prediction-eval-v1"
)


@dataclass(frozen=True)
class ObjectStateControlledPredictionThresholds:
    max_state_ade: float = 0.05
    max_prediction_gap_vs_history_model: float = 0.02
    max_error_ratio_vs_history_model: float = 1.25
    min_prediction_count: int = 1

    def as_dict(self) -> dict[str, Any]:
        return validate_objectstate_controlled_prediction_thresholds(self)


def read_objectstate_controlled_prediction_candidates(path: str | Path) -> dict[str, Any]:
    candidate_path = Path(path)
    with candidate_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("controlled prediction candidates JSON must be an object")
    return validate_objectstate_controlled_prediction_candidates(payload)


def evaluate_objectstate_controlled_prediction_candidates(
    capture_manifest: Mapping[str, Any],
    candidates: Mapping[str, Any],
    *,
    thresholds: ObjectStateControlledPredictionThresholds | None = None,
) -> dict[str, Any]:
    checked_capture = validate_objectstate_controlled_capture_manifest(capture_manifest)
    checked_candidates = validate_objectstate_controlled_prediction_candidates(candidates)
    checked_thresholds = thresholds or ObjectStateControlledPredictionThresholds()
    checked_thresholds.as_dict()
    if checked_candidates["sample_id"] != checked_capture["sample"]["sample_id"]:
        raise ValueError("controlled prediction candidates sample_id must match capture sample_id")
    capture_summary = objectstate_controlled_capture_summary(checked_capture)
    if not capture_summary["readiness"]["prediction_stage_ready"]:
        raise ValueError("controlled capture manifest is not prediction-stage ready")
    pose_map = _pose_map(checked_capture)
    prediction_records = _prediction_records(checked_candidates, pose_map)
    metrics = _prediction_metrics(prediction_records)
    pass_gates = _prediction_pass_gates(metrics, checked_thresholds)
    passed = all(pass_gates.values())
    controlled_real_manifest = _controlled_real_manifest_with_prediction_row(
        checked_capture,
        checked_candidates,
        metrics=metrics,
        passed=passed,
        pass_gates=pass_gates,
    )
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA,
        "kind": "objectstate_controlled_prediction_eval",
        "capture_schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "prediction_schema": OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
        "status": (
            "objectstate_controlled_prediction_eval_pass"
            if passed
            else "objectstate_controlled_prediction_eval_fail"
        ),
        "sample": dict(checked_capture["sample"]),
        "candidate": dict(checked_candidates["candidate"]),
        "thresholds": checked_thresholds.as_dict(),
        "metrics": metrics,
        "pass_gates": pass_gates,
        "prediction_records": prediction_records,
        "controlled_real_manifest": controlled_real_manifest,
        "claim_policy": {
            "capture_pose_ground_truth_required": True,
            "candidate_future_predictions_required": True,
            "history_baseline_required": True,
            "prediction_metrics_drive_pass_fail": True,
            "does_not_claim_identity_or_intervention": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "runs_prediction_model": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_controlled_prediction_eval_summary(payload)


def validate_objectstate_controlled_prediction_candidates(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled prediction candidates must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA:
        raise ValueError(
            "unsupported controlled prediction candidates schema: "
            f"{payload.get('schema')}"
        )
    candidate = _validate_candidate(payload.get("candidate"))
    predictions = tuple(
        _validate_prediction(item)
        for item in _sequence(payload.get("predictions"), "predictions")
    )
    if not predictions:
        raise ValueError("controlled prediction candidates require at least one prediction")
    return {
        "schema": OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
        "sample_id": _required_string(payload, "sample_id"),
        "candidate": candidate,
        "predictions": predictions,
    }


def validate_objectstate_controlled_prediction_eval_summary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("controlled prediction eval summary must be a dict")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA:
        raise ValueError(
            f"unsupported controlled prediction eval schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_prediction_eval":
        raise ValueError("controlled prediction eval summary kind is unsupported")
    if payload.get("capture_schema") != OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA:
        raise ValueError("controlled prediction eval summary has unsupported capture_schema")
    if payload.get("prediction_schema") != OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA:
        raise ValueError("controlled prediction eval summary has unsupported prediction_schema")
    if payload.get("status") not in {
        "objectstate_controlled_prediction_eval_pass",
        "objectstate_controlled_prediction_eval_fail",
    }:
        raise ValueError("controlled prediction eval status is unsupported")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("controlled prediction eval summary requires metrics")
    for key in (
        "state_ade",
        "history_ade",
        "prediction_gap_vs_history_model",
        "error_ratio_vs_history_model",
        "prediction_count",
        "object_count",
        "mean_horizon_seconds",
        "max_horizon_seconds",
    ):
        if key not in metrics:
            raise ValueError(f"controlled prediction metrics missing {key}")
    pass_gates = payload.get("pass_gates")
    if not isinstance(pass_gates, dict) or any(
        not isinstance(value, bool) for value in pass_gates.values()
    ):
        raise ValueError("controlled prediction pass_gates must be bool")
    expected_status = (
        "objectstate_controlled_prediction_eval_pass"
        if all(pass_gates.values())
        else "objectstate_controlled_prediction_eval_fail"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled prediction status must match pass gates")
    records = payload.get("prediction_records")
    if not isinstance(records, list) or not records:
        raise ValueError("controlled prediction eval summary requires prediction_records")
    validate_objectstate_controlled_real_manifest(payload.get("controlled_real_manifest"))
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("capture_pose_ground_truth_required")
        or not claim_policy.get("candidate_future_predictions_required")
        or not claim_policy.get("history_baseline_required")
        or not claim_policy.get("prediction_metrics_drive_pass_fail")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled prediction eval summary must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("runs_prediction_model")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError(
            "controlled prediction eval cannot claim capture, GT, prediction run, "
            "training, replay, diffusion, or viewer mutation"
        )
    return payload


def validate_objectstate_controlled_prediction_thresholds(
    thresholds: ObjectStateControlledPredictionThresholds,
) -> dict[str, Any]:
    if not isinstance(thresholds, ObjectStateControlledPredictionThresholds):
        raise TypeError("thresholds must be ObjectStateControlledPredictionThresholds")
    max_state_ade = float(thresholds.max_state_ade)
    max_gap = float(thresholds.max_prediction_gap_vs_history_model)
    max_ratio = float(thresholds.max_error_ratio_vs_history_model)
    min_count = thresholds.min_prediction_count
    if max_state_ade < 0.0:
        raise ValueError("max_state_ade must be non-negative")
    if max_gap < 0.0:
        raise ValueError("max_prediction_gap_vs_history_model must be non-negative")
    if max_ratio < 0.0:
        raise ValueError("max_error_ratio_vs_history_model must be non-negative")
    if isinstance(min_count, bool) or not isinstance(min_count, int):
        raise TypeError("min_prediction_count must be an integer")
    if min_count < 1:
        raise ValueError("min_prediction_count must be >= 1")
    return {
        "max_state_ade": max_state_ade,
        "max_prediction_gap_vs_history_model": max_gap,
        "max_error_ratio_vs_history_model": max_ratio,
        "min_prediction_count": int(min_count),
    }


def _controlled_real_manifest_with_prediction_row(
    capture_manifest: Mapping[str, Any],
    candidates: Mapping[str, Any],
    *,
    metrics: Mapping[str, Any],
    passed: bool,
    pass_gates: Mapping[str, bool],
) -> dict[str, Any]:
    manifest = objectstate_controlled_real_manifest_from_capture_manifest(capture_manifest)
    prediction_row = {
        "evidence_kind": "prediction",
        "status": "pass" if passed else "fail",
        "metrics": {
            "state_ade": metrics["state_ade"],
            "history_ade": metrics["history_ade"],
            "prediction_gap_vs_history_model": metrics[
                "prediction_gap_vs_history_model"
            ],
            "error_ratio_vs_history_model": metrics["error_ratio_vs_history_model"],
            "prediction_count": metrics["prediction_count"],
            "mean_horizon_seconds": metrics["mean_horizon_seconds"],
            "max_horizon_seconds": metrics["max_horizon_seconds"],
        },
        "artifact_refs": tuple(
            list(capture_manifest["sample"]["artifact_refs"])
            + list(candidates["candidate"]["artifact_refs"])
        ),
    }
    if not passed:
        failed = ", ".join(key for key, value in pass_gates.items() if not value)
        prediction_row["failure_reason"] = (
            f"controlled prediction metrics failed: {failed}"
        )
    rows = list(manifest["evidence_rows"])
    rows[1] = prediction_row
    manifest["evidence_rows"] = rows
    return validate_objectstate_controlled_real_manifest(manifest)


def _pose_map(
    capture_manifest: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    timestamp_by_frame = {
        frame["frame_id"]: float(frame["timestamp"])
        for frame in capture_manifest["frames"]
    }
    for frame in capture_manifest["frames"]:
        frame_id = frame["frame_id"]
        for item in frame["objects"]:
            pose = item.get("pose")
            if pose is None:
                continue
            result[(frame_id, item["object_id"])] = {
                "frame_id": frame_id,
                "object_id": item["object_id"],
                "timestamp": timestamp_by_frame[frame_id],
                "position": tuple(float(value) for value in pose["position"]),
            }
    return result


def _prediction_records(
    candidates: Mapping[str, Any],
    pose_map: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in candidates["predictions"]:
        key = (item["source_frame_id"], item["target_frame_id"], item["object_id"])
        if key in seen:
            raise ValueError(
                "controlled prediction candidates contain duplicate "
                "source/target/object tuple: "
                f"{key[0]} / {key[1]} / {key[2]}"
            )
        seen.add(key)
        source_key = (item["source_frame_id"], item["object_id"])
        target_key = (item["target_frame_id"], item["object_id"])
        if source_key not in pose_map:
            raise ValueError(
                "controlled prediction references unknown source frame/object pose: "
                f"{source_key[0]} / {source_key[1]}"
            )
        if target_key not in pose_map:
            raise ValueError(
                "controlled prediction references unknown target frame/object pose: "
                f"{target_key[0]} / {target_key[1]}"
            )
        source = pose_map[source_key]
        target = pose_map[target_key]
        horizon = float(target["timestamp"]) - float(source["timestamp"])
        if horizon <= 0.0:
            raise ValueError("controlled prediction target frame must be after source frame")
        predicted = item["predicted_position"]
        baseline = item["history_baseline_position"]
        target_position = target["position"]
        state_error = _distance(predicted, target_position)
        history_error = _distance(baseline, target_position)
        result.append(
            {
                "source_frame_id": item["source_frame_id"],
                "target_frame_id": item["target_frame_id"],
                "object_id": item["object_id"],
                "horizon_seconds": horizon,
                "predicted_position": list(predicted),
                "history_baseline_position": list(baseline),
                "target_position": list(target_position),
                "state_error": state_error,
                "history_error": history_error,
                "prediction_gap_vs_history_model": state_error - history_error,
            }
        )
    return result


def _prediction_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("controlled prediction eval requires at least one record")
    state_errors = tuple(float(item["state_error"]) for item in records)
    history_errors = tuple(float(item["history_error"]) for item in records)
    horizons = tuple(float(item["horizon_seconds"]) for item in records)
    state_ade = sum(state_errors) / len(state_errors)
    history_ade = sum(history_errors) / len(history_errors)
    gap = state_ade - history_ade
    if history_ade == 0.0:
        ratio = 1.0 if state_ade == 0.0 else math.inf
    else:
        ratio = state_ade / history_ade
    object_ids = {str(item["object_id"]) for item in records}
    return {
        "state_ade": state_ade,
        "history_ade": history_ade,
        "prediction_gap_vs_history_model": gap,
        "error_ratio_vs_history_model": ratio,
        "prediction_count": len(records),
        "object_count": len(object_ids),
        "mean_horizon_seconds": sum(horizons) / len(horizons),
        "max_horizon_seconds": max(horizons),
        "max_state_error": max(state_errors),
        "max_history_error": max(history_errors),
    }


def _prediction_pass_gates(
    metrics: Mapping[str, Any],
    thresholds: ObjectStateControlledPredictionThresholds,
) -> dict[str, bool]:
    return {
        "prediction_count_at_or_above_threshold": (
            int(metrics["prediction_count"]) >= int(thresholds.min_prediction_count)
        ),
        "state_ade_at_or_below_threshold": (
            float(metrics["state_ade"]) <= float(thresholds.max_state_ade)
        ),
        "prediction_gap_at_or_below_threshold": (
            float(metrics["prediction_gap_vs_history_model"])
            <= float(thresholds.max_prediction_gap_vs_history_model)
        ),
        "error_ratio_at_or_below_threshold": (
            float(metrics["error_ratio_vs_history_model"])
            <= float(thresholds.max_error_ratio_vs_history_model)
        ),
    }


def _validate_candidate(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled prediction candidate must be a mapping")
    return {
        "candidate_id": _required_string(value, "candidate_id"),
        "source": str(value.get("source", "unknown")),
        "artifact_refs": _string_tuple(value.get("artifact_refs"), "artifact_refs"),
    }


def _validate_prediction(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled prediction entries must be mappings")
    result = {
        "source_frame_id": _required_string(value, "source_frame_id"),
        "target_frame_id": _required_string(value, "target_frame_id"),
        "object_id": _required_string(value, "object_id"),
        "predicted_position": _vector(
            value.get("predicted_position"),
            "predicted_position",
        ),
        "history_baseline_position": _vector(
            value.get("history_baseline_position"),
            "history_baseline_position",
        ),
    }
    if result["source_frame_id"] == result["target_frame_id"]:
        raise ValueError(
            "controlled prediction source_frame_id and target_frame_id must differ"
        )
    if "confidence" in value:
        confidence = _number(value["confidence"], "confidence")
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence must be in [0, 1]")
        result["confidence"] = confidence
    return result


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be a non-empty string")
    return result


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    normalized = tuple(str(item) for item in value)
    if not normalized or any(not item for item in normalized):
        raise ValueError(f"{name} must contain non-empty strings")
    return normalized


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _vector(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(value) != 3:
        raise ValueError(f"{name} must have length 3")
    return tuple(_number(item, name) for item in value)  # type: ignore[return-value]


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("distance vectors must have same length")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))
