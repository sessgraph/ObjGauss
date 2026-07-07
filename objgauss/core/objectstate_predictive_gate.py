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

OBJECTSTATE_PREDICTIVE_GATE_SCHEMA = "objgauss-objectstate-predictive-gate-v1"
_GATE_STATUS_PASS = "objectstate_predictive_gate_pass"
_GATE_STATUS_FAIL = "objectstate_predictive_gate_fail"
_EPS = 1e-8


@dataclass(frozen=True)
class ObjectStatePredictiveGateThresholds:
    state_ade_max: float = 1e-5
    prediction_error_ratio_max: float = 1.05
    identity_consistency_min: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return validate_objectstate_predictive_gate_thresholds(self)


@dataclass(frozen=True)
class ObjectStatePredictiveRow:
    scenario_id: str
    scenario_kind: str
    oracle_object_id: int
    lineage_id: str
    frame_index: int
    horizon: int
    current_pose: np.ndarray
    velocity: np.ndarray
    target_pose: np.ndarray
    state_prediction: np.ndarray
    history_prediction: np.ndarray
    state_error: float
    history_error: float
    identity_consistent: bool
    history_available: bool

    def as_dict(self) -> dict[str, Any]:
        row = validate_objectstate_predictive_row(self)
        return {
            "scenario_id": row.scenario_id,
            "scenario_kind": row.scenario_kind,
            "oracle_object_id": int(row.oracle_object_id),
            "lineage_id": row.lineage_id,
            "frame_index": int(row.frame_index),
            "horizon": int(row.horizon),
            "objectstate_t": {
                "pose": np.round(row.current_pose, 6).tolist(),
                "velocity": np.round(row.velocity, 6).tolist(),
            },
            "target": {
                "pose": np.round(row.target_pose, 6).tolist(),
            },
            "state_prediction": np.round(row.state_prediction, 6).tolist(),
            "history_prediction": np.round(row.history_prediction, 6).tolist(),
            "state_error": float(row.state_error),
            "history_error": float(row.history_error),
            "identity_consistent": bool(row.identity_consistent),
            "history_available": bool(row.history_available),
        }


@dataclass(frozen=True)
class ObjectStatePredictiveGateReport:
    rows: tuple[ObjectStatePredictiveRow, ...]
    thresholds: ObjectStatePredictiveGateThresholds
    metrics: dict[str, Any]
    hard_gates: dict[str, bool]
    hard_blockers: tuple[str, ...]
    velocity_source: str
    schema: str = OBJECTSTATE_PREDICTIVE_GATE_SCHEMA

    @property
    def passed(self) -> bool:
        return all(bool(value) for value in self.hard_gates.values())

    def as_dict(self) -> dict[str, Any]:
        summary = {
            "schema": self.schema,
            "kind": "objectstate_predictive_gate",
            "status": _GATE_STATUS_PASS if self.passed else _GATE_STATUS_FAIL,
            "gate_role": "objectstate_predictive_sufficiency_smoke_gate",
            "velocity_source": self.velocity_source,
            "thresholds": self.thresholds.as_dict(),
            "row_count": len(self.rows),
            "metrics": self.metrics,
            "hard_gates": {key: bool(value) for key, value in self.hard_gates.items()},
            "hard_blockers": list(self.hard_blockers),
            "rows": [row.as_dict() for row in self.rows],
            "claim_policy": {
                "synthetic_smoke_only": True,
                "does_not_claim_real_world_markov_state": True,
                "requires_history_baseline_comparison": True,
                "counterfactual_required_later": True,
            },
            "non_goals": {
                "trains_dynamics_model": False,
                "uses_replay_buffer": False,
                "uses_diffusion": False,
                "uses_renderer_loss": False,
                "mutates_viewer_defaults": False,
            },
        }
        return validate_objectstate_predictive_gate_summary(summary)


def evaluate_objectstate_predictive_gate(
    fixtures: Sequence[SyntheticStabilityScenarioFixture] | None = None,
    *,
    horizon: int = 1,
    velocity_scale: float = 1.0,
    thresholds: ObjectStatePredictiveGateThresholds | None = None,
) -> ObjectStatePredictiveGateReport:
    resolved_fixtures = tuple(
        make_synthetic_stability_scenario_suite()
        if fixtures is None
        else fixtures
    )
    if not resolved_fixtures:
        raise ValueError("fixtures must contain at least one scenario")
    if int(horizon) < 1:
        raise ValueError("horizon must be >= 1")
    if not np.isfinite(float(velocity_scale)):
        raise ValueError("velocity_scale must be finite")
    checked_thresholds = thresholds or ObjectStatePredictiveGateThresholds()
    checked_thresholds.as_dict()
    rows: list[ObjectStatePredictiveRow] = []
    for fixture in resolved_fixtures:
        checked_fixture = validate_synthetic_stability_scenario_fixture(fixture)
        rows.extend(
            _predictive_rows_for_fixture(
                checked_fixture,
                horizon=int(horizon),
                velocity_scale=float(velocity_scale),
            )
        )
    if not rows:
        raise ValueError("predictive gate produced no rows")
    metrics = _predictive_metrics(tuple(rows))
    hard_gates, hard_blockers = _hard_gate_result(metrics, checked_thresholds)
    return ObjectStatePredictiveGateReport(
        rows=tuple(rows),
        thresholds=checked_thresholds,
        metrics=metrics,
        hard_gates=hard_gates,
        hard_blockers=hard_blockers,
        velocity_source="synthetic_world_object_trajectory",
    )


def validate_objectstate_predictive_gate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("objectstate predictive gate summary must be a dict")
    if payload.get("schema") != OBJECTSTATE_PREDICTIVE_GATE_SCHEMA:
        raise ValueError(f"unsupported objectstate predictive gate schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_predictive_gate":
        raise ValueError("objectstate predictive gate kind must be objectstate_predictive_gate")
    if payload.get("status") not in {_GATE_STATUS_PASS, _GATE_STATUS_FAIL}:
        raise ValueError("objectstate predictive gate status is unsupported")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("objectstate predictive gate requires rows")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("objectstate predictive gate requires metrics")
    for key in (
        "state_ade",
        "state_fde",
        "history_ade",
        "prediction_error_ratio",
        "state_sufficiency_score",
        "identity_consistency_rate",
    ):
        if key not in metrics:
            raise ValueError(f"objectstate predictive metrics missing {key}")
    hard_gates = payload.get("hard_gates")
    if not isinstance(hard_gates, dict) or not hard_gates:
        raise ValueError("objectstate predictive gate requires hard_gates")
    for value in hard_gates.values():
        if not isinstance(value, bool):
            raise ValueError("objectstate predictive hard gates must be bool")
    expected_status = _GATE_STATUS_PASS if all(hard_gates.values()) else _GATE_STATUS_FAIL
    if payload["status"] != expected_status:
        raise ValueError("objectstate predictive status must match hard gates")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("trains_dynamics_model")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("uses_renderer_loss")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError("objectstate predictive gate cannot train, replay, diffuse, render, or mutate viewer policy")
    return payload


def validate_objectstate_predictive_gate_thresholds(
    thresholds: ObjectStatePredictiveGateThresholds,
) -> dict[str, float]:
    if not isinstance(thresholds, ObjectStatePredictiveGateThresholds):
        raise TypeError("thresholds must be ObjectStatePredictiveGateThresholds")
    payload = {
        "state_ade_max": float(thresholds.state_ade_max),
        "prediction_error_ratio_max": float(thresholds.prediction_error_ratio_max),
        "identity_consistency_min": float(thresholds.identity_consistency_min),
    }
    for key, value in payload.items():
        if not np.isfinite(float(value)):
            raise ValueError(f"{key} must be finite")
    if payload["state_ade_max"] < 0.0:
        raise ValueError("state_ade_max must be >= 0")
    if payload["prediction_error_ratio_max"] < 0.0:
        raise ValueError("prediction_error_ratio_max must be >= 0")
    if not 0.0 <= payload["identity_consistency_min"] <= 1.0:
        raise ValueError("identity_consistency_min must be in [0, 1]")
    return payload


def validate_objectstate_predictive_row(row: ObjectStatePredictiveRow) -> ObjectStatePredictiveRow:
    if not isinstance(row, ObjectStatePredictiveRow):
        raise TypeError("row must be ObjectStatePredictiveRow")
    current = _vector3(row.current_pose, "current_pose")
    velocity = _vector3(row.velocity, "velocity")
    target = _vector3(row.target_pose, "target_pose")
    state_prediction = _vector3(row.state_prediction, "state_prediction")
    history_prediction = _vector3(row.history_prediction, "history_prediction")
    if int(row.horizon) < 1:
        raise ValueError("horizon must be >= 1")
    return ObjectStatePredictiveRow(
        scenario_id=str(row.scenario_id),
        scenario_kind=str(row.scenario_kind),
        oracle_object_id=int(row.oracle_object_id),
        lineage_id=str(row.lineage_id),
        frame_index=int(row.frame_index),
        horizon=int(row.horizon),
        current_pose=current,
        velocity=velocity,
        target_pose=target,
        state_prediction=state_prediction,
        history_prediction=history_prediction,
        state_error=float(row.state_error),
        history_error=float(row.history_error),
        identity_consistent=bool(row.identity_consistent),
        history_available=bool(row.history_available),
    )


def _predictive_rows_for_fixture(
    fixture: SyntheticStabilityScenarioFixture,
    *,
    horizon: int,
    velocity_scale: float,
) -> tuple[ObjectStatePredictiveRow, ...]:
    by_frame_object = {
        (int(frame.frame_index), int(obj.oracle_object_id)): obj
        for frame in fixture.world.frames
        for obj in frame.objects
    }
    rows = []
    max_start = fixture.world.frame_count - int(horizon)
    for frame_index in range(max_start):
        for identity in fixture.world.oracle.identities:
            object_id = int(identity.oracle_object_id)
            current = by_frame_object[(frame_index, object_id)]
            target = by_frame_object[(frame_index + int(horizon), object_id)]
            previous = by_frame_object.get((frame_index - 1, object_id))
            rows.append(
                _predictive_row(
                    fixture,
                    current=current,
                    target=target,
                    previous=previous,
                    horizon=int(horizon),
                    velocity_scale=float(velocity_scale),
                )
            )
    return tuple(rows)


def _predictive_row(
    fixture: SyntheticStabilityScenarioFixture,
    *,
    current: SyntheticWorldObject,
    target: SyntheticWorldObject,
    previous: SyntheticWorldObject | None,
    horizon: int,
    velocity_scale: float,
) -> ObjectStatePredictiveRow:
    current_pose = np.asarray(current.pose_center, dtype=np.float32)
    target_pose = np.asarray(target.pose_center, dtype=np.float32)
    velocity = np.asarray(current.trajectory, dtype=np.float32) * float(velocity_scale)
    state_prediction = current_pose + velocity * float(horizon)
    if previous is None:
        history_velocity = np.zeros(3, dtype=np.float32)
        history_available = False
    else:
        history_velocity = current_pose - np.asarray(previous.pose_center, dtype=np.float32)
        history_available = True
    history_prediction = current_pose + history_velocity * float(horizon)
    return validate_objectstate_predictive_row(
        ObjectStatePredictiveRow(
            scenario_id=fixture.scenario_id,
            scenario_kind=fixture.scenario_kind,
            oracle_object_id=int(current.oracle_object_id),
            lineage_id=current.lineage_id,
            frame_index=int(current.frame_index),
            horizon=int(horizon),
            current_pose=current_pose,
            velocity=velocity,
            target_pose=target_pose,
            state_prediction=state_prediction,
            history_prediction=history_prediction,
            state_error=float(np.linalg.norm(state_prediction - target_pose)),
            history_error=float(np.linalg.norm(history_prediction - target_pose)),
            identity_consistent=int(current.oracle_object_id) == int(target.oracle_object_id),
            history_available=history_available,
        )
    )


def _predictive_metrics(rows: tuple[ObjectStatePredictiveRow, ...]) -> dict[str, Any]:
    state_errors = np.asarray([row.state_error for row in rows], dtype=np.float32)
    history_errors = np.asarray([row.history_error for row in rows], dtype=np.float32)
    state_ade = float(np.mean(state_errors))
    history_ade = float(np.mean(history_errors))
    ratio = _safe_ratio(state_ade, history_ade)
    consistency = sum(1 for row in rows if row.identity_consistent) / float(len(rows))
    return {
        "row_count": int(len(rows)),
        "history_available_count": int(sum(1 for row in rows if row.history_available)),
        "state_ade": state_ade,
        "state_fde": float(state_errors[-1]),
        "history_ade": history_ade,
        "history_fde": float(history_errors[-1]),
        "prediction_error_ratio": ratio,
        "state_sufficiency_score": _safe_ratio(history_ade, state_ade),
        "identity_consistency_rate": float(consistency),
        "max_state_error": float(np.max(state_errors)),
        "max_history_error": float(np.max(history_errors)),
    }


def _hard_gate_result(
    metrics: dict[str, Any],
    thresholds: ObjectStatePredictiveGateThresholds,
) -> tuple[dict[str, bool], tuple[str, ...]]:
    hard_gates = {
        "state_ade_pass": float(metrics["state_ade"]) <= float(thresholds.state_ade_max),
        "prediction_error_ratio_pass": float(metrics["prediction_error_ratio"])
        <= float(thresholds.prediction_error_ratio_max),
        "identity_consistency_pass": float(metrics["identity_consistency_rate"])
        >= float(thresholds.identity_consistency_min),
    }
    blockers = []
    blocker_by_gate = {
        "state_ade_pass": "state_ade_above_threshold",
        "prediction_error_ratio_pass": "prediction_error_ratio_above_threshold",
        "identity_consistency_pass": "identity_consistency_below_threshold",
    }
    for gate, passed in hard_gates.items():
        if not passed:
            blockers.append(blocker_by_gate[gate])
    return hard_gates, tuple(blockers)


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
