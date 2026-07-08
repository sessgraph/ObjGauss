from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    objectstate_controlled_capture_summary,
    objectstate_controlled_real_manifest_from_capture_manifest,
    validate_objectstate_controlled_capture_manifest,
)
from objgauss.core.objectstate_controlled_real_rows import (
    validate_objectstate_controlled_real_manifest,
)

OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA = (
    "objgauss-objectstate-controlled-intervention-candidates-v1"
)
OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA = (
    "objgauss-objectstate-controlled-intervention-eval-v1"
)
_EPS = 1e-8


@dataclass(frozen=True)
class ObjectStateControlledInterventionThresholds:
    max_action_conditioned_ade: float = 0.05
    min_counterfactual_outcome_accuracy: float = 0.95
    max_wrong_direction_rate: float = 0.0
    min_intervention_gain: float = 0.0
    min_intervention_count: int = 1

    def as_dict(self) -> dict[str, Any]:
        return validate_objectstate_controlled_intervention_thresholds(self)


def read_objectstate_controlled_intervention_candidates(
    path: str | Path,
) -> dict[str, Any]:
    candidate_path = Path(path)
    with candidate_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("controlled intervention candidates JSON must be an object")
    return validate_objectstate_controlled_intervention_candidates(payload)


def evaluate_objectstate_controlled_intervention_candidates(
    capture_manifest: Mapping[str, Any],
    candidates: Mapping[str, Any],
    *,
    thresholds: ObjectStateControlledInterventionThresholds | None = None,
) -> dict[str, Any]:
    checked_capture = validate_objectstate_controlled_capture_manifest(capture_manifest)
    checked_candidates = validate_objectstate_controlled_intervention_candidates(
        candidates
    )
    checked_thresholds = thresholds or ObjectStateControlledInterventionThresholds()
    checked_thresholds.as_dict()
    if checked_candidates["sample_id"] != checked_capture["sample"]["sample_id"]:
        raise ValueError(
            "controlled intervention candidates sample_id must match capture sample_id"
        )
    capture_summary = objectstate_controlled_capture_summary(checked_capture)
    if not capture_summary["readiness"]["intervention_stage_ready"]:
        raise ValueError("controlled capture manifest is not intervention-stage ready")
    pose_map = _pose_map(checked_capture)
    action_map = _action_map(checked_capture)
    intervention_records = _intervention_records(
        checked_candidates,
        pose_map=pose_map,
        action_map=action_map,
    )
    metrics = _intervention_metrics(intervention_records)
    pass_gates = _intervention_pass_gates(metrics, checked_thresholds)
    passed = all(pass_gates.values())
    controlled_real_manifest = _controlled_real_manifest_with_intervention_row(
        checked_capture,
        checked_candidates,
        metrics=metrics,
        passed=passed,
        pass_gates=pass_gates,
    )
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA,
        "kind": "objectstate_controlled_intervention_eval",
        "capture_schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "candidate_schema": OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
        "status": (
            "objectstate_controlled_intervention_eval_pass"
            if passed
            else "objectstate_controlled_intervention_eval_fail"
        ),
        "sample": dict(checked_capture["sample"]),
        "candidate": dict(checked_candidates["candidate"]),
        "thresholds": checked_thresholds.as_dict(),
        "metrics": metrics,
        "pass_gates": pass_gates,
        "intervention_records": intervention_records,
        "controlled_real_manifest": controlled_real_manifest,
        "claim_policy": {
            "capture_pose_and_action_ground_truth_required": True,
            "candidate_action_conditioned_predictions_required": True,
            "no_action_baseline_required": True,
            "action_vector_required_for_wrong_direction": True,
            "intervention_metrics_drive_pass_fail": True,
            "does_not_claim_identity_or_prediction": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "runs_intervention_model": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_controlled_intervention_eval_summary(payload)


def validate_objectstate_controlled_intervention_candidates(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled intervention candidates must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA:
        raise ValueError(
            "unsupported controlled intervention candidates schema: "
            f"{payload.get('schema')}"
        )
    candidate = _validate_candidate(payload.get("candidate"))
    interventions = tuple(
        _validate_intervention(item)
        for item in _sequence(payload.get("interventions"), "interventions")
    )
    if not interventions:
        raise ValueError(
            "controlled intervention candidates require at least one intervention"
        )
    return {
        "schema": OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
        "sample_id": _required_string(payload, "sample_id"),
        "candidate": candidate,
        "interventions": interventions,
    }


def validate_objectstate_controlled_intervention_eval_summary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("controlled intervention eval summary must be a dict")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA:
        raise ValueError(
            f"unsupported controlled intervention eval schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_intervention_eval":
        raise ValueError("controlled intervention eval summary kind is unsupported")
    if payload.get("capture_schema") != OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA:
        raise ValueError("controlled intervention eval summary has unsupported capture_schema")
    if payload.get("candidate_schema") != OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA:
        raise ValueError("controlled intervention eval summary has unsupported candidate_schema")
    if payload.get("status") not in {
        "objectstate_controlled_intervention_eval_pass",
        "objectstate_controlled_intervention_eval_fail",
    }:
        raise ValueError("controlled intervention eval status is unsupported")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("controlled intervention eval summary requires metrics")
    for key in (
        "action_conditioned_ade",
        "no_action_ade",
        "intervention_gain",
        "counterfactual_outcome_accuracy",
        "wrong_direction_rate",
        "intervention_count",
        "mean_horizon_seconds",
        "max_horizon_seconds",
    ):
        if key not in metrics:
            raise ValueError(f"controlled intervention metrics missing {key}")
    pass_gates = payload.get("pass_gates")
    if not isinstance(pass_gates, dict) or any(
        not isinstance(value, bool) for value in pass_gates.values()
    ):
        raise ValueError("controlled intervention pass_gates must be bool")
    expected_status = (
        "objectstate_controlled_intervention_eval_pass"
        if all(pass_gates.values())
        else "objectstate_controlled_intervention_eval_fail"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled intervention status must match pass gates")
    records = payload.get("intervention_records")
    if not isinstance(records, list) or not records:
        raise ValueError("controlled intervention eval summary requires records")
    validate_objectstate_controlled_real_manifest(payload.get("controlled_real_manifest"))
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("capture_pose_and_action_ground_truth_required")
        or not claim_policy.get("candidate_action_conditioned_predictions_required")
        or not claim_policy.get("no_action_baseline_required")
        or not claim_policy.get("action_vector_required_for_wrong_direction")
        or not claim_policy.get("intervention_metrics_drive_pass_fail")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled intervention eval summary must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("runs_intervention_model")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError(
            "controlled intervention eval cannot claim capture, GT, model run, "
            "training, replay, diffusion, or viewer mutation"
        )
    return payload


def validate_objectstate_controlled_intervention_thresholds(
    thresholds: ObjectStateControlledInterventionThresholds,
) -> dict[str, Any]:
    if not isinstance(thresholds, ObjectStateControlledInterventionThresholds):
        raise TypeError("thresholds must be ObjectStateControlledInterventionThresholds")
    max_ade = float(thresholds.max_action_conditioned_ade)
    min_accuracy = float(thresholds.min_counterfactual_outcome_accuracy)
    max_wrong = float(thresholds.max_wrong_direction_rate)
    min_gain = float(thresholds.min_intervention_gain)
    min_count = thresholds.min_intervention_count
    if max_ade < 0.0:
        raise ValueError("max_action_conditioned_ade must be non-negative")
    if min_accuracy < 0.0 or min_accuracy > 1.0:
        raise ValueError("min_counterfactual_outcome_accuracy must be in [0, 1]")
    if max_wrong < 0.0 or max_wrong > 1.0:
        raise ValueError("max_wrong_direction_rate must be in [0, 1]")
    if min_gain < 0.0:
        raise ValueError("min_intervention_gain must be non-negative")
    if isinstance(min_count, bool) or not isinstance(min_count, int):
        raise TypeError("min_intervention_count must be an integer")
    if min_count < 1:
        raise ValueError("min_intervention_count must be >= 1")
    return {
        "max_action_conditioned_ade": max_ade,
        "min_counterfactual_outcome_accuracy": min_accuracy,
        "max_wrong_direction_rate": max_wrong,
        "min_intervention_gain": min_gain,
        "min_intervention_count": int(min_count),
    }


def _controlled_real_manifest_with_intervention_row(
    capture_manifest: Mapping[str, Any],
    candidates: Mapping[str, Any],
    *,
    metrics: Mapping[str, Any],
    passed: bool,
    pass_gates: Mapping[str, bool],
) -> dict[str, Any]:
    manifest = objectstate_controlled_real_manifest_from_capture_manifest(capture_manifest)
    intervention_row = {
        "evidence_kind": "intervention",
        "status": "pass" if passed else "fail",
        "metrics": {
            "action_conditioned_ade": metrics["action_conditioned_ade"],
            "counterfactual_outcome_accuracy": metrics[
                "counterfactual_outcome_accuracy"
            ],
            "wrong_direction_rate": metrics["wrong_direction_rate"],
            "no_action_ade": metrics["no_action_ade"],
            "intervention_gain": metrics["intervention_gain"],
            "intervention_count": metrics["intervention_count"],
        },
        "artifact_refs": tuple(
            list(capture_manifest["sample"]["artifact_refs"])
            + list(candidates["candidate"]["artifact_refs"])
        ),
    }
    if not passed:
        failed = ", ".join(key for key, value in pass_gates.items() if not value)
        intervention_row["failure_reason"] = (
            f"controlled intervention metrics failed: {failed}"
        )
    rows = list(manifest["evidence_rows"])
    rows[2] = intervention_row
    manifest["evidence_rows"] = rows
    return validate_objectstate_controlled_real_manifest(manifest)


def _pose_map(
    capture_manifest: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for frame in capture_manifest["frames"]:
        frame_id = frame["frame_id"]
        for item in frame["objects"]:
            pose = item.get("pose")
            if pose is None:
                continue
            result[(frame_id, item["object_id"])] = {
                "frame_id": frame_id,
                "object_id": item["object_id"],
                "timestamp": float(frame["timestamp"]),
                "position": tuple(float(value) for value in pose["position"]),
            }
    return result


def _action_map(capture_manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for action in capture_manifest["actions"]:
        vector = action.get("vector")
        if vector is None:
            raise ValueError(
                "controlled intervention action requires vector for wrong_direction_rate"
            )
        vector_tuple = tuple(float(value) for value in vector)
        if _norm(vector_tuple) <= _EPS:
            raise ValueError("controlled intervention action vector must be non-zero")
        result[action["action_id"]] = dict(action, vector=vector_tuple)
    return result


def _intervention_records(
    candidates: Mapping[str, Any],
    *,
    pose_map: Mapping[tuple[str, str], Mapping[str, Any]],
    action_map: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in candidates["interventions"]:
        key = (
            item["source_frame_id"],
            item["target_frame_id"],
            item["object_id"],
            item["action_id"],
        )
        if key in seen:
            raise ValueError(
                "controlled intervention candidates contain duplicate "
                "source/target/object/action tuple: "
                f"{key[0]} / {key[1]} / {key[2]} / {key[3]}"
            )
        seen.add(key)
        action = action_map.get(item["action_id"])
        if action is None:
            raise ValueError(
                "controlled intervention references unknown action_id: "
                f"{item['action_id']}"
            )
        if action["object_id"] != item["object_id"]:
            raise ValueError("controlled intervention action object_id mismatch")
        source_key = (item["source_frame_id"], item["object_id"])
        target_key = (item["target_frame_id"], item["object_id"])
        if source_key not in pose_map:
            raise ValueError(
                "controlled intervention references unknown source frame/object pose: "
                f"{source_key[0]} / {source_key[1]}"
            )
        if target_key not in pose_map:
            raise ValueError(
                "controlled intervention references unknown target frame/object pose: "
                f"{target_key[0]} / {target_key[1]}"
            )
        source = pose_map[source_key]
        target = pose_map[target_key]
        horizon = float(target["timestamp"]) - float(source["timestamp"])
        if horizon <= 0.0:
            raise ValueError("controlled intervention target frame must be after source frame")
        if (
            float(action["start_timestamp"]) < float(source["timestamp"])
            or float(action["end_timestamp"]) > float(target["timestamp"])
        ):
            raise ValueError(
                "controlled intervention action interval must fit within "
                "source/target frame timestamps"
            )
        action_position = item["action_conditioned_position"]
        no_action_position = item["no_action_baseline_position"]
        target_position = target["position"]
        source_position = source["position"]
        action_error = _distance(action_position, target_position)
        no_action_error = _distance(no_action_position, target_position)
        movement = tuple(
            float(a) - float(b) for a, b in zip(action_position, source_position)
        )
        wrong_direction = _dot(movement, action["vector"]) <= _EPS
        counterfactual_correct = action_error <= no_action_error and not wrong_direction
        result.append(
            {
                "source_frame_id": item["source_frame_id"],
                "target_frame_id": item["target_frame_id"],
                "object_id": item["object_id"],
                "action_id": item["action_id"],
                "action_type": action["action_type"],
                "horizon_seconds": horizon,
                "action_vector": list(action["vector"]),
                "action_conditioned_position": list(action_position),
                "no_action_baseline_position": list(no_action_position),
                "target_position": list(target_position),
                "action_conditioned_error": action_error,
                "no_action_error": no_action_error,
                "wrong_direction": bool(wrong_direction),
                "counterfactual_correct": bool(counterfactual_correct),
            }
        )
    return result


def _intervention_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("controlled intervention eval requires at least one record")
    action_errors = tuple(float(item["action_conditioned_error"]) for item in records)
    no_action_errors = tuple(float(item["no_action_error"]) for item in records)
    horizons = tuple(float(item["horizon_seconds"]) for item in records)
    action_ade = sum(action_errors) / len(action_errors)
    no_action_ade = sum(no_action_errors) / len(no_action_errors)
    correct = sum(1 for item in records if item["counterfactual_correct"])
    wrong = sum(1 for item in records if item["wrong_direction"])
    action_type_counts: dict[str, int] = {}
    for item in records:
        action_type = str(item["action_type"])
        action_type_counts[action_type] = action_type_counts.get(action_type, 0) + 1
    return {
        "action_conditioned_ade": action_ade,
        "no_action_ade": no_action_ade,
        "intervention_gain": no_action_ade - action_ade,
        "action_error_ratio": _safe_ratio(action_ade, no_action_ade),
        "counterfactual_outcome_accuracy": correct / len(records),
        "wrong_direction_rate": wrong / len(records),
        "wrong_direction_count": int(wrong),
        "intervention_count": len(records),
        "mean_horizon_seconds": sum(horizons) / len(horizons),
        "max_horizon_seconds": max(horizons),
        "max_action_conditioned_error": max(action_errors),
        "max_no_action_error": max(no_action_errors),
        "action_type_counts": action_type_counts,
    }


def _intervention_pass_gates(
    metrics: Mapping[str, Any],
    thresholds: ObjectStateControlledInterventionThresholds,
) -> dict[str, bool]:
    return {
        "intervention_count_at_or_above_threshold": (
            int(metrics["intervention_count"]) >= int(thresholds.min_intervention_count)
        ),
        "action_conditioned_ade_at_or_below_threshold": (
            float(metrics["action_conditioned_ade"])
            <= float(thresholds.max_action_conditioned_ade)
        ),
        "counterfactual_accuracy_at_or_above_threshold": (
            float(metrics["counterfactual_outcome_accuracy"])
            >= float(thresholds.min_counterfactual_outcome_accuracy)
        ),
        "wrong_direction_rate_at_or_below_threshold": (
            float(metrics["wrong_direction_rate"])
            <= float(thresholds.max_wrong_direction_rate)
        ),
        "intervention_gain_above_threshold": (
            float(metrics["intervention_gain"])
            > float(thresholds.min_intervention_gain)
        ),
    }


def _validate_candidate(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled intervention candidate must be a mapping")
    return {
        "candidate_id": _required_string(value, "candidate_id"),
        "source": str(value.get("source", "unknown")),
        "artifact_refs": _string_tuple(value.get("artifact_refs"), "artifact_refs"),
    }


def _validate_intervention(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled intervention entries must be mappings")
    result = {
        "source_frame_id": _required_string(value, "source_frame_id"),
        "target_frame_id": _required_string(value, "target_frame_id"),
        "object_id": _required_string(value, "object_id"),
        "action_id": _required_string(value, "action_id"),
        "action_conditioned_position": _vector(
            value.get("action_conditioned_position"),
            "action_conditioned_position",
        ),
        "no_action_baseline_position": _vector(
            value.get("no_action_baseline_position"),
            "no_action_baseline_position",
        ),
    }
    if result["source_frame_id"] == result["target_frame_id"]:
        raise ValueError(
            "controlled intervention source_frame_id and target_frame_id must differ"
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


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("dot vectors must have same length")
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) ** 2 for item in value))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) < _EPS:
        return 0.0 if abs(float(numerator)) < _EPS else math.inf
    return float(numerator) / float(denominator)
