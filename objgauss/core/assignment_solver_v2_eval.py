from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.assignment_solver_v2 import (
    ASSIGNMENT_SOLVER_V2_TRAINING_SCHEMA,
    AssignmentSolverV2State,
    AssignmentSolverV2TrainingResult,
    assignment_solver_v2_state_from_dict,
    predict_assignment_solver_v2,
    validate_assignment_solver_v2_state,
    validate_assignment_solver_v2_training_summary,
)
from objgauss.core.v2_stability_diagnostics import V2_STABILITY_FAILURE_MODES
from objgauss.core.v2_stability_foundation import (
    SyntheticStabilityScenarioFixture,
    make_synthetic_stability_scenario_suite,
    validate_synthetic_stability_scenario_fixture,
)
from objgauss.core.v2_stability_gate import (
    V2_STABILITY_GATE_SUITE_SCHEMA,
    evaluate_synthetic_stability_suite_gate,
    validate_synthetic_stability_suite_gate_summary,
)

ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA = "objgauss-assignment-solver-v2-checkpoint"
ASSIGNMENT_SOLVER_V2_STABILITY_EVAL_SCHEMA = "objgauss-assignment-solver-v2-stability-eval-v1"
_EVAL_STATUS_PASS = "assignment_solver_v2_stability_eval_pass"
_EVAL_STATUS_FAIL = "assignment_solver_v2_stability_eval_fail"


@dataclass(frozen=True)
class AssignmentSolverV2StabilityEvalReport:
    training_result: AssignmentSolverV2TrainingResult
    fixtures: tuple[SyntheticStabilityScenarioFixture, ...]
    before_gate: dict[str, Any]
    after_gate: dict[str, Any]
    checkpoint: dict[str, Any]
    restored_state: AssignmentSolverV2State
    checkpoint_roundtrip: dict[str, Any]
    schema: str = ASSIGNMENT_SOLVER_V2_STABILITY_EVAL_SCHEMA

    @property
    def passed(self) -> bool:
        return (
            self.training_result.final_loss.total_loss < self.training_result.initial_loss.total_loss
            and self.after_gate["status"] == "synthetic_stability_suite_gate_pass"
            and bool(self.checkpoint_roundtrip["pass"])
        )

    def as_dict(self) -> dict[str, Any]:
        training_summary = self.training_result.as_dict()
        summary = {
            "schema": self.schema,
            "kind": "assignment_solver_v2_stability_eval",
            "status": _EVAL_STATUS_PASS if self.passed else _EVAL_STATUS_FAIL,
            "training_schema": training_summary["schema"],
            "checkpoint_schema": self.checkpoint["schema"],
            "fixture_count": len(self.fixtures),
            "training_loss": {
                "initial_total_loss": float(self.training_result.initial_loss.total_loss),
                "final_total_loss": float(self.training_result.final_loss.total_loss),
                "loss_decreased": bool(
                    self.training_result.final_loss.total_loss
                    < self.training_result.initial_loss.total_loss
                ),
                "initial_supervised_loss": float(self.training_result.initial_loss.supervised_loss),
                "final_supervised_loss": float(self.training_result.final_loss.supervised_loss),
                "supervised_loss_decreased": bool(
                    self.training_result.final_loss.supervised_loss
                    < self.training_result.initial_loss.supervised_loss
                ),
            },
            "hard_gate": {
                "before_status": self.before_gate["status"],
                "after_status": self.after_gate["status"],
                "before_passed": self.before_gate["status"] == "synthetic_stability_suite_gate_pass",
                "after_passed": self.after_gate["status"] == "synthetic_stability_suite_gate_pass",
                "loss_decrease_does_not_override_identity_gate": True,
            },
            "diagnostics_delta": _diagnostics_delta(self.before_gate, self.after_gate),
            "checkpoint_roundtrip": self.checkpoint_roundtrip,
            "before_gate": self.before_gate,
            "after_gate": self.after_gate,
            "checkpoint": self.checkpoint,
            "training": training_summary,
            "non_goals": {
                "uses_renderer_loss": False,
                "uses_gpu": False,
                "uses_rollout_model": False,
                "uses_replay_buffer": False,
                "mutates_dynamic_k": False,
            },
        }
        return validate_assignment_solver_v2_stability_eval_summary(summary)


def assignment_solver_v2_checkpoint(
    result: AssignmentSolverV2TrainingResult,
    *,
    source: str = "synthetic_stability_training",
) -> dict[str, Any]:
    if result.schema != ASSIGNMENT_SOLVER_V2_TRAINING_SCHEMA:
        raise ValueError(f"unsupported assignment solver v2 training schema: {result.schema}")
    payload = {
        "schema": ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA,
        "kind": "assignment_solver_v2_checkpoint",
        "source": {
            "source": source,
            "training_schema": result.schema,
        },
        "training": result.as_dict(),
        "solver_state": result.final_state.as_dict(include_arrays=True),
        "gpu_policy": {
            "uses_gpu": False,
            "renderer_loss": "not_used",
        },
        "export_policy": {
            "repository_write": "do_not_commit_training_checkpoints",
            "intended_locations": ["/tmp", "ignored outputs/"],
            "large_artifacts": "keep_out_of_git",
        },
    }
    return validate_assignment_solver_v2_checkpoint(payload)


def assignment_solver_v2_state_from_checkpoint(payload: dict[str, Any]) -> AssignmentSolverV2State:
    checked = validate_assignment_solver_v2_checkpoint(payload)
    return assignment_solver_v2_state_from_dict(checked["solver_state"])


def evaluate_assignment_solver_v2_stability(
    training_result: AssignmentSolverV2TrainingResult,
    fixtures: Sequence[SyntheticStabilityScenarioFixture] | None = None,
    *,
    source: str = "synthetic_stability_suite",
) -> AssignmentSolverV2StabilityEvalReport:
    if fixtures is None:
        resolved_fixtures = make_synthetic_stability_scenario_suite()
    else:
        resolved_fixtures = tuple(
            validate_synthetic_stability_scenario_fixture(fixture)
            for fixture in fixtures
        )
    if not resolved_fixtures:
        raise ValueError("fixtures must contain at least one scenario")
    checkpoint = assignment_solver_v2_checkpoint(training_result, source=source)
    restored_state = assignment_solver_v2_state_from_checkpoint(checkpoint)
    before_predictions = _predictions_by_fixture(
        resolved_fixtures,
        training_result.initial_state,
    )
    after_predictions = _predictions_by_fixture(
        resolved_fixtures,
        training_result.final_state,
    )
    restored_predictions = _predictions_by_fixture(resolved_fixtures, restored_state)
    before_gate = evaluate_synthetic_stability_suite_gate(
        resolved_fixtures,
        predicted_assignments_by_fixture=before_predictions,
    ).as_dict()
    after_gate = evaluate_synthetic_stability_suite_gate(
        resolved_fixtures,
        predicted_assignments_by_fixture=after_predictions,
    ).as_dict()
    roundtrip = _checkpoint_roundtrip_summary(
        final_predictions=after_predictions,
        restored_predictions=restored_predictions,
        final_state=training_result.final_state,
        restored_state=restored_state,
    )
    return AssignmentSolverV2StabilityEvalReport(
        training_result=training_result,
        fixtures=resolved_fixtures,
        before_gate=before_gate,
        after_gate=after_gate,
        checkpoint=checkpoint,
        restored_state=restored_state,
        checkpoint_roundtrip=roundtrip,
    )


def validate_assignment_solver_v2_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("assignment solver v2 checkpoint must be a dict")
    if payload.get("schema") != ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported assignment solver v2 checkpoint schema: {payload.get('schema')}")
    if payload.get("kind") != "assignment_solver_v2_checkpoint":
        raise ValueError("assignment solver v2 checkpoint kind must be assignment_solver_v2_checkpoint")
    if not isinstance(payload.get("training"), dict):
        raise ValueError("assignment solver v2 checkpoint missing training")
    validate_assignment_solver_v2_training_summary(payload["training"])
    if not isinstance(payload.get("solver_state"), dict):
        raise ValueError("assignment solver v2 checkpoint missing solver_state")
    assignment_solver_v2_state_from_dict(payload["solver_state"])
    gpu_policy = payload.get("gpu_policy")
    if not isinstance(gpu_policy, dict) or gpu_policy.get("uses_gpu") is not False:
        raise ValueError("assignment solver v2 checkpoint must record uses_gpu=false")
    if gpu_policy.get("renderer_loss") != "not_used":
        raise ValueError("assignment solver v2 checkpoint must not use renderer loss")
    return payload


def validate_assignment_solver_v2_stability_eval_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("assignment solver v2 stability eval summary must be a dict")
    if payload.get("schema") != ASSIGNMENT_SOLVER_V2_STABILITY_EVAL_SCHEMA:
        raise ValueError(f"unsupported assignment solver v2 eval schema: {payload.get('schema')}")
    if payload.get("kind") != "assignment_solver_v2_stability_eval":
        raise ValueError("assignment solver v2 eval kind must be assignment_solver_v2_stability_eval")
    if payload.get("status") not in {_EVAL_STATUS_PASS, _EVAL_STATUS_FAIL}:
        raise ValueError("assignment solver v2 eval status is unsupported")
    for key in (
        "training_loss",
        "hard_gate",
        "diagnostics_delta",
        "checkpoint_roundtrip",
        "before_gate",
        "after_gate",
        "checkpoint",
        "training",
        "non_goals",
    ):
        if key not in payload:
            raise ValueError(f"assignment solver v2 eval missing {key}")
    if payload["before_gate"].get("schema") != V2_STABILITY_GATE_SUITE_SCHEMA:
        raise ValueError("assignment solver v2 eval before_gate schema is unsupported")
    if payload["after_gate"].get("schema") != V2_STABILITY_GATE_SUITE_SCHEMA:
        raise ValueError("assignment solver v2 eval after_gate schema is unsupported")
    validate_synthetic_stability_suite_gate_summary(payload["before_gate"])
    validate_synthetic_stability_suite_gate_summary(payload["after_gate"])
    validate_assignment_solver_v2_checkpoint(payload["checkpoint"])
    validate_assignment_solver_v2_training_summary(payload["training"])
    hard_gate = payload["hard_gate"]
    if not isinstance(hard_gate, dict):
        raise ValueError("assignment solver v2 eval hard_gate must be a dict")
    if hard_gate.get("loss_decrease_does_not_override_identity_gate") is not True:
        raise ValueError("assignment solver v2 eval must record loss does not override identity gate")
    checkpoint_roundtrip = payload["checkpoint_roundtrip"]
    if not isinstance(checkpoint_roundtrip, dict) or not isinstance(
        checkpoint_roundtrip.get("pass"),
        bool,
    ):
        raise ValueError("assignment solver v2 eval checkpoint_roundtrip.pass must be a bool")
    expected_pass = (
        bool(payload["training_loss"]["loss_decreased"])
        and bool(hard_gate["after_passed"])
        and bool(checkpoint_roundtrip["pass"])
    )
    expected_status = _EVAL_STATUS_PASS if expected_pass else _EVAL_STATUS_FAIL
    if payload["status"] != expected_status:
        raise ValueError("assignment solver v2 eval status must match loss, gate, and checkpoint")
    non_goals = payload["non_goals"]
    if (
        non_goals.get("uses_renderer_loss")
        or non_goals.get("uses_gpu")
        or non_goals.get("uses_rollout_model")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("mutates_dynamic_k")
    ):
        raise ValueError("assignment solver v2 eval violates non-goals")
    return payload


def _predictions_by_fixture(
    fixtures: Sequence[SyntheticStabilityScenarioFixture],
    state: AssignmentSolverV2State,
) -> tuple[tuple[np.ndarray, ...], ...]:
    state = validate_assignment_solver_v2_state(state)
    by_fixture = []
    for fixture in fixtures:
        frame_predictions = []
        for observation in fixture.observations:
            frame_predictions.append(
                predict_assignment_solver_v2(observation.evidence, state).assignment
            )
        by_fixture.append(tuple(frame_predictions))
    return tuple(by_fixture)


def _checkpoint_roundtrip_summary(
    *,
    final_predictions: Sequence[Sequence[np.ndarray]],
    restored_predictions: Sequence[Sequence[np.ndarray]],
    final_state: AssignmentSolverV2State,
    restored_state: AssignmentSolverV2State,
) -> dict[str, Any]:
    state_arrays_pass = (
        np.allclose(final_state.feature_centers, restored_state.feature_centers, atol=1e-5)
        and np.allclose(final_state.position_centers, restored_state.position_centers, atol=1e-5)
        and np.allclose(final_state.slot_bias, restored_state.slot_bias, atol=1e-5)
        and int(final_state.step) == int(restored_state.step)
    )
    prediction_pass = True
    max_assignment_delta = 0.0
    for final_fixture, restored_fixture in zip(final_predictions, restored_predictions, strict=True):
        for final_frame, restored_frame in zip(final_fixture, restored_fixture, strict=True):
            delta = float(np.max(np.abs(final_frame - restored_frame)))
            max_assignment_delta = max(max_assignment_delta, delta)
            if delta > 1e-5:
                prediction_pass = False
    return {
        "pass": bool(state_arrays_pass and prediction_pass),
        "state_arrays_pass": bool(state_arrays_pass),
        "prediction_roundtrip_pass": bool(prediction_pass),
        "max_assignment_delta": float(max_assignment_delta),
    }


def _diagnostics_delta(before_gate: dict[str, Any], after_gate: dict[str, Any]) -> dict[str, Any]:
    before = before_gate["aggregate_failure_mode_counts"]
    after = after_gate["aggregate_failure_mode_counts"]
    return {
        "before_failure_mode_counts": before,
        "after_failure_mode_counts": after,
        "delta_failure_mode_counts": {
            mode: int(after[mode]) - int(before[mode])
            for mode in V2_STABILITY_FAILURE_MODES
        },
        "before_hard_blockers": before_gate["hard_blockers"],
        "after_hard_blockers": after_gate["hard_blockers"],
    }
