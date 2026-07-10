from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.evaluation.objectstate_controlled_prediction_eval import (
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
    validate_objectstate_controlled_prediction_candidates,
)
from objgauss.datasets.objectstate_transition_dataset import (
    OBJECTSTATE_TRANSITION_DATASET_SCHEMA,
    read_objectstate_transition_dataset,
    validate_objectstate_transition_dataset,
)

OBJECTSTATE_TRANSITION_PREDICTION_CANDIDATES_SCHEMA = (
    "objgauss-objectstate-transition-prediction-candidates-v1"
)

OBJECTSTATE_TRANSITION_PREDICTION_POLICIES = (
    "hold",
    "constant_velocity",
    "action_delta",
)


def objectstate_transition_prediction_candidates(
    transition_dataset: Mapping[str, Any],
    *,
    policy: str = "constant_velocity",
    candidate_id: str = "transition-prediction-baseline-constant-velocity",
    candidate_source: str | None = None,
    artifact_ref: str = "generated-transition-prediction-candidates",
    confidence: float = 0.5,
    require_action_transition: bool = False,
) -> dict[str, Any]:
    dataset = validate_objectstate_transition_dataset(transition_dataset)
    checked_policy = _validate_policy(policy)
    checked_confidence = _confidence(confidence)
    transitions = _sequence(dataset["transitions"], "transitions")
    histories = _object_pose_histories(transitions)
    candidate = {
        "candidate_id": _required_string(candidate_id, "candidate_id"),
        "source": _required_string(
            candidate_source
            if candidate_source is not None
            else f"transition_prediction_baseline:{checked_policy}",
            "candidate_source",
        ),
        "artifact_refs": [_required_string(artifact_ref, "artifact_ref")],
    }
    predictions, records = _prediction_rows(
        transitions,
        histories=histories,
        policy=checked_policy,
        confidence=checked_confidence,
    )
    action_conditioned_rows = sum(1 for item in records if item["has_action"])
    if require_action_transition and action_conditioned_rows < 1:
        raise ValueError(
            "transition prediction candidates require at least one "
            "action-conditioned transition"
        )
    candidate_payload = validate_objectstate_controlled_prediction_candidates(
        {
            "schema": OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
            "sample_id": dataset["sample"]["sample_id"],
            "candidate": candidate,
            "predictions": predictions,
        }
    )
    payload = {
        "schema": OBJECTSTATE_TRANSITION_PREDICTION_CANDIDATES_SCHEMA,
        "kind": "objectstate_transition_prediction_candidates",
        "status": "objectstate_transition_prediction_candidates_ready",
        "transition_dataset_schema": OBJECTSTATE_TRANSITION_DATASET_SCHEMA,
        "target_eval_schema": OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
        "sample_id": dataset["sample"]["sample_id"],
        "candidate": candidate,
        "requirements": {
            "action_transition_required": bool(require_action_transition),
        },
        "policy": {
            "name": checked_policy,
            "history_baseline": "hold_source_pose",
            "prediction": _policy_description(checked_policy),
            "fallback_without_previous_pose_or_action": "hold_source_pose",
            "uses_source_pose": True,
            "uses_prior_pose": checked_policy == "constant_velocity",
            "uses_action_vector": checked_policy == "action_delta",
            "uses_target_timestamp_only": True,
            "uses_target_pose_values": False,
        },
        "row_counts": {
            "prediction_candidates": len(records),
            "constant_velocity_rows": sum(
                1 for item in records if item["prediction_mode"] == "constant_velocity"
            ),
            "action_delta_rows": sum(
                1 for item in records if item["prediction_mode"] == "action_delta"
            ),
            "hold_rows": sum(1 for item in records if item["prediction_mode"] == "hold"),
            "action_conditioned_rows": action_conditioned_rows,
            "no_action_rows": len(records) - action_conditioned_rows,
        },
        "prediction_candidates": candidate_payload,
        "row_records": records,
        "claim_policy": {
            "transition_dataset_candidate_generator": True,
            "uses_object_level_transition_dataset": True,
            "uses_source_pose": True,
            "uses_prior_pose_or_action_only": True,
            "uses_target_timestamp_only": True,
            "does_not_read_target_pose_values_for_prediction": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_identity": True,
            "does_not_run_prediction_eval": True,
            "does_not_train_dynamics_model": True,
            "does_not_create_replay_buffer": True,
            "does_not_create_reality_rows": True,
            "does_not_claim_metric_pass": True,
            "does_not_claim_learned_model": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "downloads_dataset": False,
            "creates_ground_truth": False,
            "infers_identity": False,
            "reconstructs_gaussians": False,
            "runs_tracking_model": False,
            "runs_prediction_model": False,
            "runs_intervention_model": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "creates_replay_buffer": False,
            "uses_diffusion": False,
            "creates_reality_rows": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_transition_prediction_candidates_summary(payload)


def objectstate_transition_prediction_candidates_summary(
    transition_dataset: Mapping[str, Any],
    *,
    policy: str = "constant_velocity",
    candidate_id: str = "transition-prediction-baseline-constant-velocity",
    candidate_source: str | None = None,
    artifact_ref: str = "generated-transition-prediction-candidates",
    confidence: float = 0.5,
    require_action_transition: bool = False,
) -> dict[str, Any]:
    return objectstate_transition_prediction_candidates(
        transition_dataset,
        policy=policy,
        candidate_id=candidate_id,
        candidate_source=candidate_source,
        artifact_ref=artifact_ref,
        confidence=confidence,
        require_action_transition=require_action_transition,
    )


def write_objectstate_transition_prediction_candidates(
    transition_dataset: str | Path,
    output: str | Path,
    *,
    policy: str = "constant_velocity",
    candidate_id: str = "transition-prediction-baseline-constant-velocity",
    candidate_source: str | None = None,
    artifact_ref: str = "generated-transition-prediction-candidates",
    confidence: float = 0.5,
    require_action_transition: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    transition_path = Path(transition_dataset)
    output_path = Path(output)
    dataset = read_objectstate_transition_dataset(transition_path)
    summary = objectstate_transition_prediction_candidates(
        dataset,
        policy=policy,
        candidate_id=candidate_id,
        candidate_source=candidate_source,
        artifact_ref=artifact_ref,
        confidence=confidence,
        require_action_transition=require_action_transition,
    )
    _ensure_can_write(output_path, force=force)
    _write_json(output_path, summary["prediction_candidates"])
    next_commands = {
        "eval_prediction": (
            "uv run objgauss object-state eval-controlled-prediction "
            f"{dataset.get('source_capture_manifest', '<capture-manifest.json>')} "
            f"{output_path}"
        ),
    }
    checked = validate_objectstate_transition_prediction_candidates_summary(
        {
            **summary,
            "source_transition_dataset": str(transition_path),
            "output": str(output_path),
            "files": {
                "prediction_candidates": str(output_path),
            },
            "next_commands": next_commands,
        }
    )
    return checked


def validate_objectstate_transition_prediction_candidates_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("transition prediction candidate summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_TRANSITION_PREDICTION_CANDIDATES_SCHEMA:
        raise ValueError(
            "unsupported transition prediction candidates schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_transition_prediction_candidates":
        raise ValueError("transition prediction candidates kind is unsupported")
    if payload.get("status") != "objectstate_transition_prediction_candidates_ready":
        raise ValueError("transition prediction candidates status is unsupported")
    if payload.get("transition_dataset_schema") != OBJECTSTATE_TRANSITION_DATASET_SCHEMA:
        raise ValueError("transition prediction candidates dataset schema mismatch")
    if payload.get("target_eval_schema") != OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA:
        raise ValueError("transition prediction candidates target schema mismatch")
    _required_string(payload.get("sample_id"), "sample_id")
    requirements = payload.get("requirements")
    if not isinstance(requirements, Mapping):
        raise ValueError("transition prediction candidates require requirements")
    if not isinstance(requirements.get("action_transition_required"), bool):
        raise ValueError(
            "transition prediction candidates missing action_transition_required"
        )
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("transition prediction candidates require candidate")
    _required_string(candidate.get("candidate_id"), "candidate_id")
    _required_string(candidate.get("source"), "candidate_source")
    refs = candidate.get("artifact_refs")
    if isinstance(refs, (str, bytes)) or not isinstance(refs, Sequence) or not refs:
        raise ValueError("transition prediction candidates artifact_refs must be non-empty")
    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("transition prediction candidates require policy")
    if _validate_policy(policy.get("name")) != policy.get("name"):
        raise ValueError("transition prediction candidates policy mismatch")
    if policy.get("uses_target_pose_values") is not False:
        raise ValueError(
            "transition prediction candidates cannot use target pose values"
        )
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("transition prediction candidates require row_counts")
    count = row_counts.get("prediction_candidates")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("transition prediction candidates require prediction rows")
    for key in (
        "constant_velocity_rows",
        "action_delta_rows",
        "hold_rows",
        "action_conditioned_rows",
        "no_action_rows",
    ):
        value = row_counts.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"transition prediction candidates invalid count {key}")
    if (
        row_counts["constant_velocity_rows"]
        + row_counts["action_delta_rows"]
        + row_counts["hold_rows"]
        != count
    ):
        raise ValueError("transition prediction candidate mode counts mismatch")
    if row_counts["action_conditioned_rows"] + row_counts["no_action_rows"] != count:
        raise ValueError("transition prediction candidate action counts mismatch")
    if (
        requirements["action_transition_required"]
        and row_counts["action_conditioned_rows"] < 1
    ):
        raise ValueError(
            "transition prediction candidates required action-conditioned rows"
        )
    predictions = validate_objectstate_controlled_prediction_candidates(
        payload.get("prediction_candidates")
    )
    if predictions["sample_id"] != payload["sample_id"]:
        raise ValueError("transition prediction candidates sample_id mismatch")
    if len(predictions["predictions"]) != count:
        raise ValueError("transition prediction candidates row count mismatch")
    rows = payload.get("row_records")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or len(rows) != count:
        raise ValueError("transition prediction candidates row_records mismatch")
    for row in rows:
        _validate_row_record(row)
    claim_policy = payload.get("claim_policy", {})
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("transition_dataset_candidate_generator")
        or not claim_policy.get("uses_object_level_transition_dataset")
        or not claim_policy.get("uses_source_pose")
        or not claim_policy.get("uses_prior_pose_or_action_only")
        or not claim_policy.get("uses_target_timestamp_only")
        or not claim_policy.get("does_not_read_target_pose_values_for_prediction")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_infer_identity")
        or not claim_policy.get("does_not_run_prediction_eval")
        or not claim_policy.get("does_not_train_dynamics_model")
        or not claim_policy.get("does_not_create_replay_buffer")
        or not claim_policy.get("does_not_create_reality_rows")
        or not claim_policy.get("does_not_claim_metric_pass")
        or not claim_policy.get("does_not_claim_learned_model")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError(
            "transition prediction candidates must preserve claim policy"
        )
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "transition prediction candidates cannot claim capture, download, GT, "
            "identity inference, reconstruction, model runs, training, replay, "
            "diffusion, reality rows, or viewer mutation"
        )
    return dict(payload)


def _prediction_rows(
    transitions: Sequence[Any],
    *,
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
    policy: str,
    confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for transition in transitions:
        if not isinstance(transition, Mapping):
            raise ValueError("transition prediction candidates require transition rows")
        object_id = _required_string(transition.get("object_id"), "object_id")
        source = _state_pose_record(transition["state_t"], object_id=object_id)
        previous = _previous_pose(histories, object_id, source["timestamp"])
        horizon = float(transition["target_timestamp"]) - float(
            transition["source_timestamp"]
        )
        if horizon <= 0.0:
            raise ValueError("transition prediction horizon must be positive")
        action_delta = _action_delta(transition.get("action_context", ()))
        predicted, mode = _predict_position(
            source,
            previous,
            action_delta=action_delta,
            horizon_seconds=horizon,
            policy=policy,
        )
        baseline = list(source["position"])
        prediction = {
            "source_frame_id": _required_string(
                transition.get("source_frame_id"),
                "source_frame_id",
            ),
            "target_frame_id": _required_string(
                transition.get("target_frame_id"),
                "target_frame_id",
            ),
            "object_id": object_id,
            "predicted_position": predicted,
            "history_baseline_position": baseline,
            "confidence": confidence,
        }
        predictions.append(prediction)
        records.append(
            {
                "transition_id": _required_string(
                    transition.get("transition_id"),
                    "transition_id",
                ),
                "source_frame_id": prediction["source_frame_id"],
                "target_frame_id": prediction["target_frame_id"],
                "object_id": object_id,
                "horizon_seconds": horizon,
                "prediction_mode": mode,
                "has_action": bool(transition.get("has_action")),
                "action_ids": list(transition.get("action_ids", ())),
                "predicted_position": predicted,
                "history_baseline_position": baseline,
            }
        )
    return predictions, records


def _object_pose_histories(
    transitions: Sequence[Any],
) -> dict[str, list[dict[str, Any]]]:
    keyed: dict[tuple[str, str], dict[str, Any]] = {}
    for transition in transitions:
        if not isinstance(transition, Mapping):
            continue
        object_id = str(transition["object_id"])
        for key in ("state_t", "state_t1"):
            state = transition[key]
            if not isinstance(state, Mapping) or "pose" not in state:
                continue
            record = _state_pose_record(state, object_id=object_id)
            keyed[(object_id, record["frame_id"])] = record
    histories: dict[str, list[dict[str, Any]]] = {}
    for (object_id, _frame_id), record in keyed.items():
        histories.setdefault(object_id, []).append(record)
    for object_id in histories:
        histories[object_id].sort(
            key=lambda item: (float(item["timestamp"]), str(item["frame_id"]))
        )
    return histories


def _state_pose_record(
    state: Mapping[str, Any],
    *,
    object_id: str,
) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise ValueError("transition prediction state must be a mapping")
    if "pose" not in state:
        raise ValueError(
            "transition prediction candidates require pose in transition states"
        )
    pose = state["pose"]
    if not isinstance(pose, Mapping):
        raise ValueError("transition prediction pose must be a mapping")
    return {
        "frame_id": _required_string(state.get("frame_id"), "frame_id"),
        "object_id": object_id,
        "timestamp": _number(state.get("timestamp"), "timestamp"),
        "position": _vector(pose.get("position"), "pose.position"),
    }


def _previous_pose(
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
    object_id: str,
    source_timestamp: float,
) -> Mapping[str, Any] | None:
    previous = None
    for item in histories.get(object_id, ()):
        if float(item["timestamp"]) < float(source_timestamp):
            previous = item
        else:
            break
    return previous


def _predict_position(
    source: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    *,
    action_delta: Sequence[float] | None,
    horizon_seconds: float,
    policy: str,
) -> tuple[list[float], str]:
    source_position = tuple(float(value) for value in source["position"])
    if policy == "hold":
        return list(source_position), "hold"
    if policy == "action_delta":
        if action_delta is None:
            return list(source_position), "hold"
        return [
            source_position[index] + float(action_delta[index])
            for index in range(3)
        ], "action_delta"
    if policy == "constant_velocity":
        if previous is None:
            return list(source_position), "hold"
        delta_t = float(source["timestamp"]) - float(previous["timestamp"])
        if delta_t <= 0.0:
            return list(source_position), "hold"
        previous_position = tuple(float(value) for value in previous["position"])
        return [
            source_position[index]
            + ((source_position[index] - previous_position[index]) / delta_t)
            * float(horizon_seconds)
            for index in range(3)
        ], "constant_velocity"
    raise ValueError(f"unsupported transition prediction policy: {policy}")


def _action_delta(action_context: Any) -> tuple[float, float, float] | None:
    if isinstance(action_context, (str, bytes)) or not isinstance(action_context, Sequence):
        return None
    total = [0.0, 0.0, 0.0]
    found = False
    for action in action_context:
        if not isinstance(action, Mapping) or "vector" not in action:
            continue
        vector = _vector(action["vector"], "action.vector")
        for index in range(3):
            total[index] += float(vector[index])
        found = True
    if not found:
        return None
    return (total[0], total[1], total[2])


def _validate_row_record(row: Any) -> None:
    if not isinstance(row, Mapping):
        raise ValueError("transition prediction row record must be a mapping")
    for key in (
        "transition_id",
        "source_frame_id",
        "target_frame_id",
        "object_id",
    ):
        _required_string(row.get(key), key)
    mode = row.get("prediction_mode")
    if mode not in {"hold", "constant_velocity", "action_delta"}:
        raise ValueError("transition prediction row record has unsupported mode")
    _number(row.get("horizon_seconds"), "horizon_seconds")
    _vector(row.get("predicted_position"), "predicted_position")
    _vector(row.get("history_baseline_position"), "history_baseline_position")
    if not isinstance(row.get("has_action"), bool):
        raise ValueError("transition prediction row record missing has_action")
    action_ids = row.get("action_ids")
    if isinstance(action_ids, (str, bytes)) or not isinstance(action_ids, Sequence):
        raise ValueError("transition prediction row action_ids must be a sequence")
    if any(not isinstance(item, str) or not item for item in action_ids):
        raise ValueError("transition prediction row action_ids must be strings")


def _validate_policy(policy: Any) -> str:
    if not isinstance(policy, str) or policy not in OBJECTSTATE_TRANSITION_PREDICTION_POLICIES:
        raise ValueError(
            "transition prediction policy must be one of: "
            + ", ".join(OBJECTSTATE_TRANSITION_PREDICTION_POLICIES)
        )
    return policy


def _policy_description(policy: str) -> str:
    if policy == "hold":
        return "hold_source_pose"
    if policy == "constant_velocity":
        return "source_pose_plus_previous_velocity"
    if policy == "action_delta":
        return "source_pose_plus_action_vector"
    raise ValueError(f"unsupported transition prediction policy: {policy}")


def _confidence(value: Any) -> float:
    result = _number(value, "confidence")
    if result < 0.0 or result > 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return result


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
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


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _ensure_can_write(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            "transition prediction candidates refuse to overwrite existing file: "
            f"{path}"
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


__all__ = (
    "OBJECTSTATE_TRANSITION_PREDICTION_CANDIDATES_SCHEMA",
    "OBJECTSTATE_TRANSITION_PREDICTION_POLICIES",
    "objectstate_transition_prediction_candidates",
    "objectstate_transition_prediction_candidates_summary",
    "write_objectstate_transition_prediction_candidates",
    "validate_objectstate_transition_prediction_candidates_summary",
)
