from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from objgauss.core.assignment_solver_v2 import (
    validate_assignment_solver_v2_training_summary,
)
from objgauss.core.assignment_solver_v2_eval import (
    validate_assignment_solver_v2_stability_eval_summary,
)
from objgauss.core.assignment_v2_renderer_validation import (
    validate_assignment_v2_renderer_joint_summary,
)
from objgauss.core.renderer_loss import (
    renderer_loss_boundary_report,
    validate_renderer_loss_boundary_summary,
)

CORE_MODEL_TRAIN_VALIDATE_SCHEMA = "objgauss-core-model-train-validate-v1"
_STATUS_PASS = "core_model_train_validate_pass"
_STATUS_FAIL = "core_model_train_validate_fail"


@dataclass(frozen=True)
class CoreModelTrainValidateReport:
    assignment_training: dict[str, Any]
    stability_eval: dict[str, Any]
    renderer_joint: dict[str, Any]
    renderer_boundary: dict[str, Any]
    schema: str = CORE_MODEL_TRAIN_VALIDATE_SCHEMA

    @property
    def passed(self) -> bool:
        gates = _milestone_gates(
            self.assignment_training,
            self.stability_eval,
            self.renderer_joint,
            self.renderer_boundary,
        )
        return all(bool(value) for value in gates.values())

    def as_dict(self) -> dict[str, Any]:
        gates = _milestone_gates(
            self.assignment_training,
            self.stability_eval,
            self.renderer_joint,
            self.renderer_boundary,
        )
        payload = {
            "schema": self.schema,
            "kind": "core_model_train_validate",
            "status": _STATUS_PASS if all(gates.values()) else _STATUS_FAIL,
            "gates": gates,
            "evidence": {
                "assignment_training": _assignment_training_evidence(
                    self.assignment_training
                ),
                "stability_eval": _stability_eval_evidence(self.stability_eval),
                "renderer_joint": _renderer_joint_evidence(self.renderer_joint),
                "renderer_boundary": _renderer_boundary_evidence(
                    self.renderer_boundary
                ),
            },
            "small_sample_smoke": {
                "status": "pass"
                if gates["small_sample_smoke_not_degraded"]
                else "fail",
                "source": self.renderer_joint["source"].get("target_source"),
                "image_render_loss_decreased": bool(
                    self.renderer_joint["image_render_loss_decreased"]
                ),
                "real_public_sample": False,
                "promotion_requires_real_sample_repeat": True,
            },
            "failure_diagnostics": {
                "available": gates["failure_diagnostics_available"],
                "before_hard_blockers": self.stability_eval["diagnostics_delta"][
                    "before_hard_blockers"
                ],
                "after_hard_blockers": self.stability_eval["diagnostics_delta"][
                    "after_hard_blockers"
                ],
                "delta_failure_mode_counts": self.stability_eval[
                    "diagnostics_delta"
                ]["delta_failure_mode_counts"],
            },
            "checkpoint_roundtrip": {
                "assignment_stability_eval_pass": bool(
                    self.stability_eval["checkpoint_roundtrip"]["pass"]
                ),
                "renderer_joint_pass": bool(
                    self.renderer_joint["checkpoint_roundtrip"]["pass"]
                ),
            },
            "non_goals": {
                "starts_long_gpu_training": False,
                "uses_rollout_model": False,
                "uses_replay_buffer": False,
                "uses_diffusion_world_model": False,
                "mutates_dynamic_k": False,
                "unfreezes_gaussian_geometry": False,
            },
            "next_steps": _next_steps(gates),
            "inputs": {
                "assignment_training_schema": self.assignment_training["schema"],
                "stability_eval_schema": self.stability_eval["schema"],
                "renderer_joint_schema": self.renderer_joint["schema"],
                "renderer_boundary_schema": self.renderer_boundary["schema"],
            },
        }
        return validate_core_model_train_validate_summary(payload)


def core_model_train_validate_report(
    *,
    assignment_training: dict[str, Any],
    stability_eval: dict[str, Any],
    renderer_joint: dict[str, Any],
    renderer_boundary: dict[str, Any] | None = None,
) -> CoreModelTrainValidateReport:
    checked_training = validate_assignment_solver_v2_training_summary(
        assignment_training
    )
    checked_stability = validate_assignment_solver_v2_stability_eval_summary(
        stability_eval
    )
    checked_renderer = validate_assignment_v2_renderer_joint_summary(renderer_joint)
    checked_boundary = (
        renderer_loss_boundary_report(checked_renderer).as_dict()
        if renderer_boundary is None
        else renderer_boundary
    )
    validate_renderer_loss_boundary_summary(checked_boundary)
    return CoreModelTrainValidateReport(
        assignment_training=checked_training,
        stability_eval=checked_stability,
        renderer_joint=checked_renderer,
        renderer_boundary=checked_boundary,
    )


def validate_core_model_train_validate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("core model train/validate summary must be a dict")
    if payload.get("schema") != CORE_MODEL_TRAIN_VALIDATE_SCHEMA:
        raise ValueError(f"unsupported core model validation schema: {payload.get('schema')}")
    if payload.get("kind") != "core_model_train_validate":
        raise ValueError("core model train/validate kind is unsupported")
    if payload.get("status") not in {_STATUS_PASS, _STATUS_FAIL}:
        raise ValueError("core model train/validate status is unsupported")
    gates = payload.get("gates")
    if not isinstance(gates, dict) or not gates:
        raise ValueError("core model train/validate requires gates")
    expected_status = _STATUS_PASS if all(bool(value) for value in gates.values()) else _STATUS_FAIL
    if payload["status"] != expected_status:
        raise ValueError("core model train/validate status must match gates")
    for key in (
        "evidence",
        "small_sample_smoke",
        "failure_diagnostics",
        "checkpoint_roundtrip",
        "non_goals",
        "inputs",
    ):
        if key not in payload:
            raise ValueError(f"core model train/validate missing {key}")
    non_goals = payload["non_goals"]
    if (
        non_goals.get("starts_long_gpu_training")
        or non_goals.get("uses_rollout_model")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion_world_model")
        or non_goals.get("mutates_dynamic_k")
        or non_goals.get("unfreezes_gaussian_geometry")
    ):
        raise ValueError("core model train/validate violates non-goals")
    return payload


def _milestone_gates(
    training: dict[str, Any],
    stability: dict[str, Any],
    renderer: dict[str, Any],
    boundary: dict[str, Any],
) -> dict[str, bool]:
    diagnostics = stability["diagnostics_delta"]
    return {
        "assignment_training_loss_decreased": bool(training["loss_decreased"]),
        "synthetic_stability_hard_gate_passed": bool(
            stability["hard_gate"]["after_passed"]
        ),
        "failure_diagnostics_available": bool(
            "delta_failure_mode_counts" in diagnostics
            and "before_hard_blockers" in diagnostics
            and "after_hard_blockers" in diagnostics
        ),
        "assignment_checkpoint_roundtrip_passed": bool(
            stability["checkpoint_roundtrip"]["pass"]
        ),
        "objectstate_eval_passed": renderer["object_state_eval"]["status"]
        == "objectstate_eval_pass",
        "renderer_joint_smoke_passed": renderer["status"]
        == "assignment_v2_renderer_joint_validation_pass",
        "small_sample_smoke_not_degraded": bool(
            renderer["loss_decreased"]
            and renderer["image_render_loss_decreased"]
            and renderer["object_loss_decreased"]
        ),
        "renderer_checkpoint_roundtrip_passed": bool(
            renderer["checkpoint_roundtrip"]["pass"]
        ),
        "renderer_loss_boundary_consumes_summary": boundary["status"]
        in {
            "assignment_v2_renderer_joint_validation_ready",
            "full_3dgs_assignment_v2_renderer_joint_validation_ready",
        },
        "identity_gate_preserved": bool(
            stability["hard_gate"]["loss_decrease_does_not_override_identity_gate"]
            and renderer["identity_gate"][
                "renderer_loss_does_not_override_identity_gate"
            ]
        ),
    }


def _assignment_training_evidence(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": summary["schema"],
        "iterations": int(summary["iterations"]),
        "loss_decreased": bool(summary["loss_decreased"]),
        "initial_total_loss": float(summary["initial_loss"]["total_loss"]),
        "final_total_loss": float(summary["final_loss"]["total_loss"]),
        "renderer_loss": summary["renderer_loss"],
        "dynamic_k": summary["dynamic_k"],
    }


def _stability_eval_evidence(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": summary["schema"],
        "status": summary["status"],
        "before_status": summary["hard_gate"]["before_status"],
        "after_status": summary["hard_gate"]["after_status"],
        "checkpoint_roundtrip_pass": bool(
            summary["checkpoint_roundtrip"]["pass"]
        ),
        "after_hard_blockers": summary["diagnostics_delta"][
            "after_hard_blockers"
        ],
    }


def _renderer_joint_evidence(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": summary["schema"],
        "status": summary["status"],
        "loss_decreased": bool(summary["loss_decreased"]),
        "image_render_loss_decreased": bool(summary["image_render_loss_decreased"]),
        "object_loss_decreased": bool(summary["object_loss_decreased"]),
        "objectstate_status": summary["object_state_eval"]["status"],
        "checkpoint_roundtrip_pass": bool(
            summary["checkpoint_roundtrip"]["pass"]
        ),
    }


def _renderer_boundary_evidence(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": summary["schema"],
        "status": summary["status"],
        "source_evidence": summary["decoder_handoff_contract"]["source_evidence"],
        "decoder_handoff_status": summary["decoder_handoff_contract"]["status"],
        "upgrade_blockers": summary["upgrade_blockers"],
    }


def _next_steps(gates: dict[str, bool]) -> list[str]:
    if all(gates.values()):
        return [
            "repeat renderer validation on a small real/public sample before promotion",
            "turn v2 assignment renderer validation into controlled solver-decoder joint training",
            "keep diffusion, replay buffer, and dynamic-K mutation behind later ADRs",
        ]
    failed = [name for name, passed in gates.items() if not passed]
    return [
        "fix failed gate(s): " + ", ".join(failed),
        "classify assignment failure before changing renderer or model family",
        "do not bypass identity hard gate with renderer loss",
    ]
