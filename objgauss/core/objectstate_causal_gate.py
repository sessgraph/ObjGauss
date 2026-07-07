from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.v2_stability_foundation import (
    SyntheticStabilityScenarioFixture,
    SyntheticWorldObject,
    make_synthetic_stability_scenario_suite,
    validate_synthetic_stability_scenario_fixture,
)

OBJECTSTATE_CAUSAL_GATE_SCHEMA = "objgauss-objectstate-causal-gate-v1"
OBJECTSTATE_ACTION_SCHEMA = "objgauss-objectstate-action-v1"
OBJECTSTATE_CAUSAL_ACTIONS = ("push_left", "push_right", "hold")
_GATE_STATUS_PASS = "objectstate_causal_gate_pass"
_GATE_STATUS_FAIL = "objectstate_causal_gate_fail"
_EPS = 1e-8


@dataclass(frozen=True)
class ObjectStateAction:
    action_id: str
    action_type: str
    delta_position: np.ndarray
    expected_direction: str
    schema: str = OBJECTSTATE_ACTION_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        action = validate_objectstate_action(self)
        return {
            "schema": action.schema,
            "action_id": action.action_id,
            "action_type": action.action_type,
            "delta_position": np.round(action.delta_position, 6).tolist(),
            "expected_direction": action.expected_direction,
        }


@dataclass(frozen=True)
class ObjectStateCausalGateThresholds:
    action_conditioned_ade_max: float = 1e-5
    counterfactual_accuracy_min: float = 1.0
    wrong_direction_rate_max: float = 0.0
    intervention_gain_min: float = 0.0
    identity_consistency_min: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return validate_objectstate_causal_gate_thresholds(self)


@dataclass(frozen=True)
class ObjectStateCausalRow:
    scenario_id: str
    scenario_kind: str
    oracle_object_id: int
    lineage_id: str
    frame_index: int
    current_pose: np.ndarray
    velocity: np.ndarray
    action: ObjectStateAction
    target_pose: np.ndarray
    action_conditioned_prediction: np.ndarray
    no_action_prediction: np.ndarray
    action_error: float
    no_action_error: float
    wrong_direction: bool
    counterfactual_correct: bool
    identity_consistent: bool

    def as_dict(self) -> dict[str, Any]:
        row = validate_objectstate_causal_row(self)
        return {
            "scenario_id": row.scenario_id,
            "scenario_kind": row.scenario_kind,
            "oracle_object_id": int(row.oracle_object_id),
            "lineage_id": row.lineage_id,
            "frame_index": int(row.frame_index),
            "objectstate_t": {
                "pose": np.round(row.current_pose, 6).tolist(),
                "velocity": np.round(row.velocity, 6).tolist(),
            },
            "action": row.action.as_dict(),
            "target": {
                "pose": np.round(row.target_pose, 6).tolist(),
            },
            "action_conditioned_prediction": np.round(
                row.action_conditioned_prediction,
                6,
            ).tolist(),
            "no_action_prediction": np.round(row.no_action_prediction, 6).tolist(),
            "action_error": float(row.action_error),
            "no_action_error": float(row.no_action_error),
            "wrong_direction": bool(row.wrong_direction),
            "counterfactual_correct": bool(row.counterfactual_correct),
            "identity_consistent": bool(row.identity_consistent),
        }


@dataclass(frozen=True)
class ObjectStateCausalGateReport:
    rows: tuple[ObjectStateCausalRow, ...]
    thresholds: ObjectStateCausalGateThresholds
    metrics: dict[str, Any]
    hard_gates: dict[str, bool]
    hard_blockers: tuple[str, ...]
    action_source: str
    schema: str = OBJECTSTATE_CAUSAL_GATE_SCHEMA

    @property
    def passed(self) -> bool:
        return all(bool(value) for value in self.hard_gates.values())

    def as_dict(self) -> dict[str, Any]:
        summary = {
            "schema": self.schema,
            "kind": "objectstate_causal_gate",
            "status": _GATE_STATUS_PASS if self.passed else _GATE_STATUS_FAIL,
            "gate_role": "objectstate_action_conditioned_causal_smoke_gate",
            "action_source": self.action_source,
            "thresholds": self.thresholds.as_dict(),
            "row_count": len(self.rows),
            "metrics": self.metrics,
            "hard_gates": {key: bool(value) for key, value in self.hard_gates.items()},
            "hard_blockers": list(self.hard_blockers),
            "rows": [row.as_dict() for row in self.rows],
            "claim_policy": {
                "synthetic_controlled_actions_only": True,
                "does_not_claim_real_world_causality": True,
                "requires_action_conditioned_comparison": True,
                "controlled_real_rows_required_later": True,
            },
            "non_goals": {
                "trains_dynamics_model": False,
                "uses_replay_buffer": False,
                "uses_diffusion": False,
                "uses_renderer_loss": False,
                "mutates_viewer_defaults": False,
            },
        }
        return validate_objectstate_causal_gate_summary(summary)


def evaluate_objectstate_causal_gate(
    fixtures: Sequence[SyntheticStabilityScenarioFixture] | None = None,
    *,
    action_effect_scale: float = 1.0,
    candidate_action_scale: float = 1.0,
    thresholds: ObjectStateCausalGateThresholds | None = None,
) -> ObjectStateCausalGateReport:
    resolved_fixtures = tuple(
        make_synthetic_stability_scenario_suite()
        if fixtures is None
        else fixtures
    )
    if not resolved_fixtures:
        raise ValueError("fixtures must contain at least one scenario")
    if not np.isfinite(float(action_effect_scale)):
        raise ValueError("action_effect_scale must be finite")
    if not np.isfinite(float(candidate_action_scale)):
        raise ValueError("candidate_action_scale must be finite")
    checked_thresholds = thresholds or ObjectStateCausalGateThresholds()
    checked_thresholds.as_dict()
    rows: list[ObjectStateCausalRow] = []
    for fixture in resolved_fixtures:
        checked_fixture = validate_synthetic_stability_scenario_fixture(fixture)
        rows.extend(
            _causal_rows_for_fixture(
                checked_fixture,
                action_effect_scale=float(action_effect_scale),
                candidate_action_scale=float(candidate_action_scale),
            )
        )
    if not rows:
        raise ValueError("causal gate produced no rows")
    metrics = _causal_metrics(tuple(rows))
    hard_gates, hard_blockers = _hard_gate_result(metrics, checked_thresholds)
    return ObjectStateCausalGateReport(
        rows=tuple(rows),
        thresholds=checked_thresholds,
        metrics=metrics,
        hard_gates=hard_gates,
        hard_blockers=hard_blockers,
        action_source="synthetic_controlled_action_oracle",
    )


def validate_objectstate_causal_gate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("objectstate causal gate summary must be a dict")
    if payload.get("schema") != OBJECTSTATE_CAUSAL_GATE_SCHEMA:
        raise ValueError(f"unsupported objectstate causal gate schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_causal_gate":
        raise ValueError("objectstate causal gate kind must be objectstate_causal_gate")
    if payload.get("status") not in {_GATE_STATUS_PASS, _GATE_STATUS_FAIL}:
        raise ValueError("objectstate causal gate status is unsupported")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("objectstate causal gate requires rows")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("objectstate causal gate requires metrics")
    for key in (
        "action_conditioned_ade",
        "action_conditioned_fde",
        "no_action_ade",
        "intervention_gain",
        "counterfactual_outcome_accuracy",
        "wrong_direction_rate",
        "identity_consistency_rate",
    ):
        if key not in metrics:
            raise ValueError(f"objectstate causal metrics missing {key}")
    hard_gates = payload.get("hard_gates")
    if not isinstance(hard_gates, dict) or not hard_gates:
        raise ValueError("objectstate causal gate requires hard_gates")
    for value in hard_gates.values():
        if not isinstance(value, bool):
            raise ValueError("objectstate causal hard gates must be bool")
    expected_status = _GATE_STATUS_PASS if all(hard_gates.values()) else _GATE_STATUS_FAIL
    if payload["status"] != expected_status:
        raise ValueError("objectstate causal status must match hard gates")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("trains_dynamics_model")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("uses_renderer_loss")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError("objectstate causal gate cannot train, replay, diffuse, render, or mutate viewer policy")
    return payload


def validate_objectstate_causal_gate_thresholds(
    thresholds: ObjectStateCausalGateThresholds,
) -> dict[str, float]:
    if not isinstance(thresholds, ObjectStateCausalGateThresholds):
        raise TypeError("thresholds must be ObjectStateCausalGateThresholds")
    payload = {
        "action_conditioned_ade_max": float(thresholds.action_conditioned_ade_max),
        "counterfactual_accuracy_min": float(thresholds.counterfactual_accuracy_min),
        "wrong_direction_rate_max": float(thresholds.wrong_direction_rate_max),
        "intervention_gain_min": float(thresholds.intervention_gain_min),
        "identity_consistency_min": float(thresholds.identity_consistency_min),
    }
    for key, value in payload.items():
        if not np.isfinite(float(value)):
            raise ValueError(f"{key} must be finite")
    if payload["action_conditioned_ade_max"] < 0.0:
        raise ValueError("action_conditioned_ade_max must be >= 0")
    if not 0.0 <= payload["counterfactual_accuracy_min"] <= 1.0:
        raise ValueError("counterfactual_accuracy_min must be in [0, 1]")
    if not 0.0 <= payload["wrong_direction_rate_max"] <= 1.0:
        raise ValueError("wrong_direction_rate_max must be in [0, 1]")
    if payload["intervention_gain_min"] < 0.0:
        raise ValueError("intervention_gain_min must be >= 0")
    if not 0.0 <= payload["identity_consistency_min"] <= 1.0:
        raise ValueError("identity_consistency_min must be in [0, 1]")
    return payload


def validate_objectstate_action(action: ObjectStateAction) -> ObjectStateAction:
    if not isinstance(action, ObjectStateAction):
        raise TypeError("action must be ObjectStateAction")
    if action.schema != OBJECTSTATE_ACTION_SCHEMA:
        raise ValueError(f"unsupported objectstate action schema: {action.schema}")
    if action.action_type not in OBJECTSTATE_CAUSAL_ACTIONS:
        raise ValueError(f"unsupported action_type: {action.action_type}")
    delta = _vector3(action.delta_position, "delta_position")
    if action.expected_direction not in {"left", "right", "none"}:
        raise ValueError("expected_direction must be left, right, or none")
    return ObjectStateAction(
        action_id=str(action.action_id),
        action_type=action.action_type,
        delta_position=delta,
        expected_direction=action.expected_direction,
        schema=action.schema,
    )


def validate_objectstate_causal_row(row: ObjectStateCausalRow) -> ObjectStateCausalRow:
    if not isinstance(row, ObjectStateCausalRow):
        raise TypeError("row must be ObjectStateCausalRow")
    current = _vector3(row.current_pose, "current_pose")
    velocity = _vector3(row.velocity, "velocity")
    action = validate_objectstate_action(row.action)
    target = _vector3(row.target_pose, "target_pose")
    action_prediction = _vector3(
        row.action_conditioned_prediction,
        "action_conditioned_prediction",
    )
    no_action_prediction = _vector3(row.no_action_prediction, "no_action_prediction")
    return ObjectStateCausalRow(
        scenario_id=str(row.scenario_id),
        scenario_kind=str(row.scenario_kind),
        oracle_object_id=int(row.oracle_object_id),
        lineage_id=str(row.lineage_id),
        frame_index=int(row.frame_index),
        current_pose=current,
        velocity=velocity,
        action=action,
        target_pose=target,
        action_conditioned_prediction=action_prediction,
        no_action_prediction=no_action_prediction,
        action_error=float(row.action_error),
        no_action_error=float(row.no_action_error),
        wrong_direction=bool(row.wrong_direction),
        counterfactual_correct=bool(row.counterfactual_correct),
        identity_consistent=bool(row.identity_consistent),
    )


def _causal_rows_for_fixture(
    fixture: SyntheticStabilityScenarioFixture,
    *,
    action_effect_scale: float,
    candidate_action_scale: float,
) -> tuple[ObjectStateCausalRow, ...]:
    rows = []
    max_start = max(0, fixture.world.frame_count - 1)
    for frame_index in range(max_start):
        frame = fixture.world.frames[frame_index]
        for obj in frame.objects:
            if not obj.visible:
                continue
            action = _action_for_object(
                obj,
                frame_index=frame_index,
                effect_scale=float(action_effect_scale),
            )
            rows.append(
                _causal_row(
                    fixture,
                    current=obj,
                    action=action,
                    candidate_action_scale=float(candidate_action_scale),
                )
            )
    return tuple(rows)


def _causal_row(
    fixture: SyntheticStabilityScenarioFixture,
    *,
    current: SyntheticWorldObject,
    action: ObjectStateAction,
    candidate_action_scale: float,
) -> ObjectStateCausalRow:
    current_pose = np.asarray(current.pose_center, dtype=np.float32)
    velocity = np.asarray(current.trajectory, dtype=np.float32)
    action = validate_objectstate_action(action)
    target_pose = current_pose + velocity + action.delta_position
    no_action_prediction = current_pose + velocity
    action_conditioned_prediction = (
        current_pose + velocity + action.delta_position * float(candidate_action_scale)
    )
    movement = action_conditioned_prediction - no_action_prediction
    wrong_direction = _wrong_direction(
        movement,
        expected_direction=action.expected_direction,
    )
    action_error = float(np.linalg.norm(action_conditioned_prediction - target_pose))
    no_action_error = float(np.linalg.norm(no_action_prediction - target_pose))
    counterfactual_correct = action_error <= 1e-5 and not wrong_direction
    return validate_objectstate_causal_row(
        ObjectStateCausalRow(
            scenario_id=fixture.scenario_id,
            scenario_kind=fixture.scenario_kind,
            oracle_object_id=int(current.oracle_object_id),
            lineage_id=current.lineage_id,
            frame_index=int(current.frame_index),
            current_pose=current_pose,
            velocity=velocity,
            action=action,
            target_pose=target_pose,
            action_conditioned_prediction=action_conditioned_prediction,
            no_action_prediction=no_action_prediction,
            action_error=action_error,
            no_action_error=no_action_error,
            wrong_direction=wrong_direction,
            counterfactual_correct=counterfactual_correct,
            identity_consistent=True,
        )
    )


def _action_for_object(
    obj: SyntheticWorldObject,
    *,
    frame_index: int,
    effect_scale: float,
) -> ObjectStateAction:
    if int(obj.oracle_object_id) % 2 == 0:
        action_type = "push_left"
        direction = "left"
        delta = np.asarray([-0.06, 0.0, 0.0], dtype=np.float32)
    elif int(frame_index) % 2 == 0:
        action_type = "push_right"
        direction = "right"
        delta = np.asarray([0.06, 0.0, 0.0], dtype=np.float32)
    else:
        action_type = "hold"
        direction = "none"
        delta = np.zeros(3, dtype=np.float32)
    return validate_objectstate_action(
        ObjectStateAction(
            action_id=(
                f"{action_type}:frame-{int(frame_index):04d}:"
                f"object-{int(obj.oracle_object_id):04d}"
            ),
            action_type=action_type,
            delta_position=delta * float(effect_scale),
            expected_direction=direction,
        )
    )


def _causal_metrics(rows: tuple[ObjectStateCausalRow, ...]) -> dict[str, Any]:
    action_errors = np.asarray([row.action_error for row in rows], dtype=np.float32)
    no_action_errors = np.asarray([row.no_action_error for row in rows], dtype=np.float32)
    action_ade = float(np.mean(action_errors))
    no_action_ade = float(np.mean(no_action_errors))
    correct_count = sum(1 for row in rows if row.counterfactual_correct)
    wrong_direction_count = sum(1 for row in rows if row.wrong_direction)
    identity_count = sum(1 for row in rows if row.identity_consistent)
    return {
        "row_count": int(len(rows)),
        "action_conditioned_ade": action_ade,
        "action_conditioned_fde": float(action_errors[-1]),
        "no_action_ade": no_action_ade,
        "no_action_fde": float(no_action_errors[-1]),
        "intervention_gain": float(no_action_ade - action_ade),
        "action_error_ratio": _safe_ratio(action_ade, no_action_ade),
        "counterfactual_outcome_accuracy": float(correct_count / len(rows)),
        "wrong_direction_rate": float(wrong_direction_count / len(rows)),
        "wrong_direction_count": int(wrong_direction_count),
        "identity_consistency_rate": float(identity_count / len(rows)),
        "action_type_counts": _action_type_counts(rows),
    }


def _hard_gate_result(
    metrics: dict[str, Any],
    thresholds: ObjectStateCausalGateThresholds,
) -> tuple[dict[str, bool], tuple[str, ...]]:
    hard_gates = {
        "action_conditioned_ade_pass": float(metrics["action_conditioned_ade"])
        <= float(thresholds.action_conditioned_ade_max),
        "counterfactual_accuracy_pass": float(metrics["counterfactual_outcome_accuracy"])
        >= float(thresholds.counterfactual_accuracy_min),
        "wrong_direction_rate_pass": float(metrics["wrong_direction_rate"])
        <= float(thresholds.wrong_direction_rate_max),
        "intervention_gain_pass": float(metrics["intervention_gain"])
        > float(thresholds.intervention_gain_min),
        "identity_consistency_pass": float(metrics["identity_consistency_rate"])
        >= float(thresholds.identity_consistency_min),
    }
    blocker_by_gate = {
        "action_conditioned_ade_pass": "action_conditioned_ade_above_threshold",
        "counterfactual_accuracy_pass": "counterfactual_accuracy_below_threshold",
        "wrong_direction_rate_pass": "wrong_direction_rate_above_threshold",
        "intervention_gain_pass": "intervention_gain_not_positive",
        "identity_consistency_pass": "identity_consistency_below_threshold",
    }
    blockers = [
        blocker_by_gate[gate]
        for gate, passed in hard_gates.items()
        if not passed
    ]
    return hard_gates, tuple(blockers)


def _action_type_counts(rows: Sequence[ObjectStateCausalRow]) -> dict[str, int]:
    counts = {action_type: 0 for action_type in OBJECTSTATE_CAUSAL_ACTIONS}
    for row in rows:
        counts[row.action.action_type] += 1
    return counts


def _wrong_direction(movement: np.ndarray, *, expected_direction: str) -> bool:
    x_delta = float(movement[0])
    if expected_direction == "left":
        return x_delta >= -_EPS
    if expected_direction == "right":
        return x_delta <= _EPS
    return abs(x_delta) > _EPS


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) < _EPS:
        return 0.0 if abs(float(numerator)) < _EPS else float("inf")
    return float(numerator) / float(denominator)


def _vector3(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or array.shape[0] != 3:
        raise ValueError(f"{label} must be a 3D vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite values")
    return array.astype(np.float32, copy=False)
