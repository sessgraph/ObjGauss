from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.evaluation.objectstate_controlled_intervention_eval import (
    OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
    validate_objectstate_controlled_intervention_candidates,
)
from objgauss.datasets.objectstate_transition_dataset import (
    OBJECTSTATE_TRANSITION_DATASET_SCHEMA,
    read_objectstate_transition_dataset,
    validate_objectstate_transition_dataset,
)

OBJECTSTATE_TRANSITION_INTERVENTION_CANDIDATES_SCHEMA = (
    "objgauss-objectstate-transition-intervention-candidates-v1"
)

OBJECTSTATE_TRANSITION_INTERVENTION_POLICIES = (
    "action_delta",
    "hold_action",
)


def objectstate_transition_intervention_candidates(
    transition_dataset: Mapping[str, Any],
    *,
    policy: str = "action_delta",
    candidate_id: str = "transition-intervention-baseline-action-delta",
    candidate_source: str | None = None,
    artifact_ref: str = "generated-transition-intervention-candidates",
    confidence: float = 0.5,
    require_intervention: bool = True,
) -> dict[str, Any]:
    dataset = validate_objectstate_transition_dataset(transition_dataset)
    checked_policy = _validate_policy(policy)
    checked_confidence = _confidence(confidence)
    candidate = {
        "candidate_id": _required_string(candidate_id, "candidate_id"),
        "source": _required_string(
            candidate_source
            if candidate_source is not None
            else f"transition_intervention_baseline:{checked_policy}",
            "candidate_source",
        ),
        "artifact_refs": [_required_string(artifact_ref, "artifact_ref")],
    }
    interventions, records, skipped = _intervention_rows(
        _sequence(dataset["transitions"], "transitions"),
        policy=checked_policy,
        confidence=checked_confidence,
    )
    if require_intervention and not interventions:
        raise ValueError(
            "transition intervention candidates require at least one "
            "action-conditioned transition with a valid action vector"
        )
    candidate_payload = validate_objectstate_controlled_intervention_candidates(
        {
            "schema": OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
            "sample_id": dataset["sample"]["sample_id"],
            "candidate": candidate,
            "interventions": interventions,
        }
    )
    payload = {
        "schema": OBJECTSTATE_TRANSITION_INTERVENTION_CANDIDATES_SCHEMA,
        "kind": "objectstate_transition_intervention_candidates",
        "status": "objectstate_transition_intervention_candidates_ready",
        "transition_dataset_schema": OBJECTSTATE_TRANSITION_DATASET_SCHEMA,
        "target_eval_schema": OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
        "sample_id": dataset["sample"]["sample_id"],
        "candidate": candidate,
        "requirements": {
            "intervention_required": bool(require_intervention),
        },
        "policy": {
            "name": checked_policy,
            "action_conditioned_prediction": _policy_description(checked_policy),
            "no_action_baseline": "hold_source_pose",
            "uses_source_pose": True,
            "uses_action_vector": checked_policy == "action_delta",
            "uses_target_timestamp_only": True,
            "uses_target_pose_values": False,
        },
        "row_counts": {
            "intervention_candidates": len(records),
            "action_delta_rows": sum(
                1 for item in records if item["prediction_mode"] == "action_delta"
            ),
            "hold_action_rows": sum(
                1 for item in records if item["prediction_mode"] == "hold_action"
            ),
            "skipped_action_contexts": len(skipped),
        },
        "intervention_candidates": candidate_payload,
        "row_records": records,
        "skipped_action_contexts": skipped,
        "claim_policy": {
            "transition_dataset_candidate_generator": True,
            "uses_object_level_transition_dataset": True,
            "uses_source_pose": True,
            "uses_action_vector_only_for_action_delta": True,
            "uses_target_timestamp_only": True,
            "does_not_read_target_pose_values_for_prediction": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_identity": True,
            "does_not_run_intervention_eval": True,
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
    return validate_objectstate_transition_intervention_candidates_summary(payload)


def objectstate_transition_intervention_candidates_summary(
    transition_dataset: Mapping[str, Any],
    *,
    policy: str = "action_delta",
    candidate_id: str = "transition-intervention-baseline-action-delta",
    candidate_source: str | None = None,
    artifact_ref: str = "generated-transition-intervention-candidates",
    confidence: float = 0.5,
    require_intervention: bool = True,
) -> dict[str, Any]:
    return objectstate_transition_intervention_candidates(
        transition_dataset,
        policy=policy,
        candidate_id=candidate_id,
        candidate_source=candidate_source,
        artifact_ref=artifact_ref,
        confidence=confidence,
        require_intervention=require_intervention,
    )


def write_objectstate_transition_intervention_candidates(
    transition_dataset: str | Path,
    output: str | Path,
    *,
    policy: str = "action_delta",
    candidate_id: str = "transition-intervention-baseline-action-delta",
    candidate_source: str | None = None,
    artifact_ref: str = "generated-transition-intervention-candidates",
    confidence: float = 0.5,
    require_intervention: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    transition_path = Path(transition_dataset)
    output_path = Path(output)
    dataset = read_objectstate_transition_dataset(transition_path)
    summary = objectstate_transition_intervention_candidates(
        dataset,
        policy=policy,
        candidate_id=candidate_id,
        candidate_source=candidate_source,
        artifact_ref=artifact_ref,
        confidence=confidence,
        require_intervention=require_intervention,
    )
    _ensure_can_write(output_path, force=force)
    _write_json(output_path, summary["intervention_candidates"])
    checked = validate_objectstate_transition_intervention_candidates_summary(
        {
            **summary,
            "source_transition_dataset": str(transition_path),
            "output": str(output_path),
            "files": {
                "intervention_candidates": str(output_path),
            },
            "next_commands": {
                "eval_intervention": (
                    "uv run objgauss object-state eval-controlled-intervention "
                    f"{dataset.get('source_capture_manifest', '<capture-manifest.json>')} "
                    f"{output_path}"
                ),
            },
        }
    )
    return checked


def validate_objectstate_transition_intervention_candidates_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("transition intervention candidate summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_TRANSITION_INTERVENTION_CANDIDATES_SCHEMA:
        raise ValueError(
            "unsupported transition intervention candidates schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_transition_intervention_candidates":
        raise ValueError("transition intervention candidates kind is unsupported")
    if payload.get("status") != "objectstate_transition_intervention_candidates_ready":
        raise ValueError("transition intervention candidates status is unsupported")
    if payload.get("transition_dataset_schema") != OBJECTSTATE_TRANSITION_DATASET_SCHEMA:
        raise ValueError("transition intervention candidates dataset schema mismatch")
    if payload.get("target_eval_schema") != OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA:
        raise ValueError("transition intervention candidates target schema mismatch")
    _required_string(payload.get("sample_id"), "sample_id")
    requirements = payload.get("requirements")
    if not isinstance(requirements, Mapping):
        raise ValueError("transition intervention candidates require requirements")
    if not isinstance(requirements.get("intervention_required"), bool):
        raise ValueError("transition intervention candidates missing intervention_required")
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("transition intervention candidates require candidate")
    _required_string(candidate.get("candidate_id"), "candidate_id")
    _required_string(candidate.get("source"), "candidate_source")
    refs = candidate.get("artifact_refs")
    if isinstance(refs, (str, bytes)) or not isinstance(refs, Sequence) or not refs:
        raise ValueError("transition intervention candidates artifact_refs must be non-empty")
    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("transition intervention candidates require policy")
    if _validate_policy(policy.get("name")) != policy.get("name"):
        raise ValueError("transition intervention candidates policy mismatch")
    if policy.get("uses_target_pose_values") is not False:
        raise ValueError(
            "transition intervention candidates cannot use target pose values"
        )
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("transition intervention candidates require row_counts")
    count = row_counts.get("intervention_candidates")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("transition intervention candidates invalid row count")
    if requirements["intervention_required"] and count < 1:
        raise ValueError("transition intervention candidates required rows")
    for key in ("action_delta_rows", "hold_action_rows", "skipped_action_contexts"):
        value = row_counts.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"transition intervention candidates invalid count {key}")
    if row_counts["action_delta_rows"] + row_counts["hold_action_rows"] != count:
        raise ValueError("transition intervention candidate mode counts mismatch")
    interventions = validate_objectstate_controlled_intervention_candidates(
        payload.get("intervention_candidates")
    )
    if interventions["sample_id"] != payload["sample_id"]:
        raise ValueError("transition intervention candidates sample_id mismatch")
    if len(interventions["interventions"]) != count:
        raise ValueError("transition intervention candidates row count mismatch")
    rows = payload.get("row_records")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or len(rows) != count:
        raise ValueError("transition intervention candidates row_records mismatch")
    for row in rows:
        _validate_row_record(row)
    skipped = payload.get("skipped_action_contexts")
    if isinstance(skipped, (str, bytes)) or not isinstance(skipped, Sequence):
        raise ValueError("transition intervention candidates skipped rows must be a sequence")
    claim_policy = payload.get("claim_policy", {})
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("transition_dataset_candidate_generator")
        or not claim_policy.get("uses_object_level_transition_dataset")
        or not claim_policy.get("uses_source_pose")
        or not claim_policy.get("uses_action_vector_only_for_action_delta")
        or not claim_policy.get("uses_target_timestamp_only")
        or not claim_policy.get("does_not_read_target_pose_values_for_prediction")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_infer_identity")
        or not claim_policy.get("does_not_run_intervention_eval")
        or not claim_policy.get("does_not_train_dynamics_model")
        or not claim_policy.get("does_not_create_replay_buffer")
        or not claim_policy.get("does_not_create_reality_rows")
        or not claim_policy.get("does_not_claim_metric_pass")
        or not claim_policy.get("does_not_claim_learned_model")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError(
            "transition intervention candidates must preserve claim policy"
        )
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "transition intervention candidates cannot claim capture, download, "
            "GT, identity inference, reconstruction, model runs, training, "
            "replay, diffusion, reality rows, or viewer mutation"
        )
    return dict(payload)


def _intervention_rows(
    transitions: Sequence[Any],
    *,
    policy: str,
    confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    interventions: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for transition in transitions:
        if not isinstance(transition, Mapping):
            raise ValueError("transition intervention candidates require transition rows")
        if not transition.get("has_action"):
            continue
        object_id = _required_string(transition.get("object_id"), "object_id")
        source = _state_pose_record(transition["state_t"], object_id=object_id)
        source_position = tuple(float(value) for value in source["position"])
        for action in _sequence(transition.get("action_context", ()), "action_context"):
            if not isinstance(action, Mapping):
                continue
            action_id = _required_string(action.get("action_id"), "action_id")
            if action.get("object_id") != object_id and action.get("target_object_id") != object_id:
                skipped.append(
                    _skipped(action_id, transition, "action does not reference object")
                )
                continue
            if not _action_interval_fits(transition, action):
                skipped.append(
                    _skipped(
                        action_id,
                        transition,
                        "action interval does not fit source/target timestamps",
                    )
                )
                continue
            vector = action.get("vector")
            if vector is None:
                skipped.append(_skipped(action_id, transition, "action vector missing"))
                continue
            action_vector = _vector(vector, "action.vector")
            if _norm(action_vector) <= 0.0:
                skipped.append(_skipped(action_id, transition, "action vector is zero"))
                continue
            action_position, mode = _action_conditioned_position(
                source_position,
                action_vector=action_vector,
                policy=policy,
            )
            no_action = list(source_position)
            row = {
                "source_frame_id": _required_string(
                    transition.get("source_frame_id"),
                    "source_frame_id",
                ),
                "target_frame_id": _required_string(
                    transition.get("target_frame_id"),
                    "target_frame_id",
                ),
                "object_id": object_id,
                "action_id": action_id,
                "action_conditioned_position": action_position,
                "no_action_baseline_position": no_action,
                "confidence": confidence,
            }
            interventions.append(row)
            records.append(
                {
                    "transition_id": _required_string(
                        transition.get("transition_id"),
                        "transition_id",
                    ),
                    "source_frame_id": row["source_frame_id"],
                    "target_frame_id": row["target_frame_id"],
                    "object_id": object_id,
                    "action_id": action_id,
                    "action_type": _required_string(
                        action.get("action_type"),
                        "action_type",
                    ),
                    "horizon_seconds": float(transition["delta_t"]),
                    "prediction_mode": mode,
                    "action_vector": list(action_vector),
                    "action_conditioned_position": action_position,
                    "no_action_baseline_position": no_action,
                }
            )
    return interventions, records, skipped


def _action_conditioned_position(
    source_position: Sequence[float],
    *,
    action_vector: Sequence[float],
    policy: str,
) -> tuple[list[float], str]:
    if policy == "hold_action":
        return list(source_position), "hold_action"
    if policy == "action_delta":
        return [
            float(source_position[index]) + float(action_vector[index])
            for index in range(3)
        ], "action_delta"
    raise ValueError(f"unsupported transition intervention policy: {policy}")


def _state_pose_record(
    state: Mapping[str, Any],
    *,
    object_id: str,
) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise ValueError("transition intervention state must be a mapping")
    if "pose" not in state:
        raise ValueError(
            "transition intervention candidates require pose in transition states"
        )
    pose = state["pose"]
    if not isinstance(pose, Mapping):
        raise ValueError("transition intervention pose must be a mapping")
    return {
        "frame_id": _required_string(state.get("frame_id"), "frame_id"),
        "object_id": object_id,
        "timestamp": _number(state.get("timestamp"), "timestamp"),
        "position": _vector(pose.get("position"), "pose.position"),
    }


def _action_interval_fits(
    transition: Mapping[str, Any],
    action: Mapping[str, Any],
) -> bool:
    start = float(transition["source_timestamp"])
    end = float(transition["target_timestamp"])
    return (
        _number(action.get("start_timestamp"), "action.start_timestamp") >= start
        and _number(action.get("end_timestamp"), "action.end_timestamp") <= end
    )


def _skipped(
    action_id: str,
    transition: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "action_id": action_id,
        "transition_id": str(transition.get("transition_id", "")),
        "object_id": str(transition.get("object_id", "")),
        "reason": reason,
    }


def _validate_row_record(row: Any) -> None:
    if not isinstance(row, Mapping):
        raise ValueError("transition intervention row record must be a mapping")
    for key in (
        "transition_id",
        "source_frame_id",
        "target_frame_id",
        "object_id",
        "action_id",
        "action_type",
    ):
        _required_string(row.get(key), key)
    mode = row.get("prediction_mode")
    if mode not in {"action_delta", "hold_action"}:
        raise ValueError("transition intervention row record has unsupported mode")
    _number(row.get("horizon_seconds"), "horizon_seconds")
    _vector(row.get("action_vector"), "action_vector")
    _vector(row.get("action_conditioned_position"), "action_conditioned_position")
    _vector(row.get("no_action_baseline_position"), "no_action_baseline_position")


def _validate_policy(policy: Any) -> str:
    if not isinstance(policy, str) or policy not in OBJECTSTATE_TRANSITION_INTERVENTION_POLICIES:
        raise ValueError(
            "transition intervention policy must be one of: "
            + ", ".join(OBJECTSTATE_TRANSITION_INTERVENTION_POLICIES)
        )
    return policy


def _policy_description(policy: str) -> str:
    if policy == "action_delta":
        return "source_pose_plus_action_vector"
    if policy == "hold_action":
        return "hold_source_pose_even_when_action_present"
    raise ValueError(f"unsupported transition intervention policy: {policy}")


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


def _norm(value: Sequence[float]) -> float:
    return sum(float(item) ** 2 for item in value) ** 0.5


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _ensure_can_write(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            "transition intervention candidates refuse to overwrite existing file: "
            f"{path}"
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


__all__ = (
    "OBJECTSTATE_TRANSITION_INTERVENTION_CANDIDATES_SCHEMA",
    "OBJECTSTATE_TRANSITION_INTERVENTION_POLICIES",
    "objectstate_transition_intervention_candidates",
    "objectstate_transition_intervention_candidates_summary",
    "write_objectstate_transition_intervention_candidates",
    "validate_objectstate_transition_intervention_candidates_summary",
)
