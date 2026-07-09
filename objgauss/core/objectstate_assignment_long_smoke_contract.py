from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from objgauss.core.objectstate_teacher_evidence_leakage_audit import (
    OBJECTSTATE_TEACHER_EVIDENCE_LEAKAGE_AUDIT_SCHEMA,
    validate_objectstate_teacher_evidence_leakage_audit_summary,
)

OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SCHEMA = (
    "objgauss-objectstate-assignment-long-smoke-contract-v1"
)
OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SUMMARY_SCHEMA = (
    "objgauss-objectstate-assignment-long-smoke-contract-summary-v1"
)
OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_REQUIRED_POLICY = "semantic"
OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SUCCESS_CRITERIA = (
    "held_out_identity_retrieval_at_1_not_decrease",
    "identity_margin_improves",
    "occlusion_recovery_not_decrease",
    "generalization_gap_not_expand",
    "slot_swap_rate_interpretable",
    "checkpoint_roundtrip",
)
_CLAIM_POLICY_KEYS = (
    "semantic_policy_only",
    "requires_teacher_evidence_contract",
    "requires_passed_teacher_evidence_leakage_audit",
    "fixed_seed_required",
    "before_after_identity_benchmark_required",
    "checkpoint_roundtrip_required",
    "does_not_claim_training_run",
    "does_not_claim_identity_gate_pass",
    "does_not_claim_world_model",
)
_NON_GOAL_KEYS = (
    "runs_training",
    "runs_long_smoke",
    "uses_native_gaussian_policy",
    "uses_renderer_loss",
    "uses_temporal_loss",
    "uses_dynamics",
    "uses_diffusion",
    "uses_replay_buffer",
    "uses_gpu",
    "ingests_real_capture",
    "mutates_viewer_defaults",
)


@dataclass(frozen=True)
class ObjectStateAssignmentLongSmokeContractThresholds:
    max_duration_seconds: int = 600
    max_identity_retrieval_drop: float = 0.0
    min_identity_margin_delta: float = 0.0
    max_occlusion_recovery_drop: float = 0.0
    max_generalization_gap_increase: float = 0.0
    max_slot_swap_rate: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "max_duration_seconds": int(self.max_duration_seconds),
            "max_identity_retrieval_drop": float(self.max_identity_retrieval_drop),
            "min_identity_margin_delta": float(self.min_identity_margin_delta),
            "max_occlusion_recovery_drop": float(self.max_occlusion_recovery_drop),
            "max_generalization_gap_increase": float(
                self.max_generalization_gap_increase
            ),
            "max_slot_swap_rate": float(self.max_slot_swap_rate),
        }
        if payload["max_duration_seconds"] < 1 or payload["max_duration_seconds"] > 600:
            raise ValueError("assignment long smoke max_duration_seconds must be in [1,600]")
        for key in (
            "max_identity_retrieval_drop",
            "min_identity_margin_delta",
            "max_occlusion_recovery_drop",
            "max_generalization_gap_increase",
        ):
            if payload[key] < 0.0 or not np.isfinite(payload[key]):
                raise ValueError(f"assignment long smoke {key} must be finite and >= 0")
        if not 0.0 <= payload["max_slot_swap_rate"] <= 1.0:
            raise ValueError("assignment long smoke max_slot_swap_rate must be in [0,1]")
        return payload


def objectstate_assignment_long_smoke_contract_summary(
    *,
    sample_id: str = "objectstate-assignment-long-smoke-contract-001",
    evidence_policy: str = OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_REQUIRED_POLICY,
    teacher_evidence_leakage_audit: Mapping[str, Any] | None = None,
    thresholds: ObjectStateAssignmentLongSmokeContractThresholds | None = None,
) -> dict[str, Any]:
    policy = _semantic_policy(evidence_policy)
    threshold_payload = (
        thresholds or ObjectStateAssignmentLongSmokeContractThresholds()
    ).as_dict()
    audit_digest = _leakage_audit_digest(teacher_evidence_leakage_audit)
    blockers = _readiness_blockers(audit_digest)
    payload = {
        "schema": OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SUMMARY_SCHEMA,
        "kind": "objectstate_assignment_long_smoke_contract_summary",
        "contract_schema": OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SCHEMA,
        "status": (
            "objectstate_assignment_long_smoke_contract_ready"
            if not blockers
            else "objectstate_assignment_long_smoke_contract_blocked"
        ),
        "sample_id": str(sample_id),
        "evidence_policy": {
            "policy": policy,
            "requires_teacher_evidence_batch": True,
            "requires_inference_time_teacher_source": True,
            "native_gaussian_long_training_allowed": False,
        },
        "duration_policy": {
            "max_duration_seconds": threshold_payload["max_duration_seconds"],
            "max_duration_minutes": threshold_payload["max_duration_seconds"] / 60.0,
            "bounded": True,
        },
        "preconditions": {
            "teacher_evidence_contract_required": True,
            "teacher_evidence_leakage_audit_required": True,
            "teacher_evidence_leakage_audit": audit_digest,
            "teacher_evidence_leakage_audit_passed": audit_digest["status"] == "pass",
            "semantic_teacher_evidence_training_allowed": bool(
                audit_digest["semantic_teacher_evidence_training_allowed"]
            ),
        },
        "training_constraints": {
            "fixed_seed_required": True,
            "checkpoint_roundtrip_required": True,
            "before_after_identity_benchmark_required": True,
            "held_out_generalization_required": True,
            "renderer_loss_allowed": False,
            "temporal_loss_allowed": False,
            "dynamics_allowed": False,
            "diffusion_allowed": False,
            "replay_buffer_allowed": False,
            "native_gaussian_policy_allowed": False,
        },
        "success_criteria": _success_criteria(threshold_payload),
        "readiness_gate": {
            "long_smoke_contract_ready": not blockers,
            "blocked_reasons": blockers,
            "next_allowed_pr": (
                "OBJECTSTATE-ASSIGNMENT-LONG-SMOKE-001"
                if not blockers
                else None
            ),
            "blocked_default_fixture_note": (
                "synthetic one-hot semantic fixtures do not clear this gate unless "
                "a training-allowed inference-time source split audit passes"
            ),
        },
        "required_run_artifacts": {
            "before_identity_benchmark_summary": True,
            "after_identity_benchmark_summary": True,
            "checkpoint_ref": True,
            "checkpoint_roundtrip_summary": True,
            "held_out_generalization_summary": True,
            "loss_curve": True,
            "seed": True,
        },
        "claim_policy": {key: True for key in _CLAIM_POLICY_KEYS},
        "non_goals": {key: False for key in _NON_GOAL_KEYS},
    }
    return validate_objectstate_assignment_long_smoke_contract_summary(payload)


def validate_objectstate_assignment_long_smoke_contract_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("assignment long smoke contract summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SUMMARY_SCHEMA:
        raise ValueError(
            "unsupported assignment long smoke contract summary schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_assignment_long_smoke_contract_summary":
        raise ValueError("assignment long smoke contract summary kind is unsupported")
    if payload.get("contract_schema") != OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SCHEMA:
        raise ValueError("assignment long smoke contract schema mismatch")
    if payload.get("status") not in {
        "objectstate_assignment_long_smoke_contract_ready",
        "objectstate_assignment_long_smoke_contract_blocked",
    }:
        raise ValueError("assignment long smoke contract status is unsupported")
    if not isinstance(payload.get("sample_id"), str) or not payload["sample_id"]:
        raise ValueError("assignment long smoke contract requires sample_id")
    evidence_policy = _mapping(payload, "evidence_policy")
    if evidence_policy.get("policy") != OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_REQUIRED_POLICY:
        raise ValueError("assignment long smoke contract is semantic-policy only")
    for key in (
        "requires_teacher_evidence_batch",
        "requires_inference_time_teacher_source",
    ):
        if evidence_policy.get(key) is not True:
            raise ValueError(f"assignment long smoke evidence policy must set {key}")
    if evidence_policy.get("native_gaussian_long_training_allowed") is not False:
        raise ValueError("assignment long smoke cannot allow native Gaussian long training")
    duration = _mapping(payload, "duration_policy")
    max_duration = _positive_int(duration.get("max_duration_seconds"), "max_duration_seconds")
    if max_duration > 600 or duration.get("bounded") is not True:
        raise ValueError("assignment long smoke duration must be bounded to <= 600s")
    preconditions = _mapping(payload, "preconditions")
    audit = _mapping(preconditions, "teacher_evidence_leakage_audit")
    _validate_leakage_audit_digest(audit)
    if preconditions.get("teacher_evidence_contract_required") is not True:
        raise ValueError("assignment long smoke must require teacher evidence contract")
    if preconditions.get("teacher_evidence_leakage_audit_required") is not True:
        raise ValueError("assignment long smoke must require leakage audit")
    constraints = _mapping(payload, "training_constraints")
    for key in (
        "fixed_seed_required",
        "checkpoint_roundtrip_required",
        "before_after_identity_benchmark_required",
        "held_out_generalization_required",
    ):
        if constraints.get(key) is not True:
            raise ValueError(f"assignment long smoke constraint {key} must be true")
    for key in (
        "renderer_loss_allowed",
        "temporal_loss_allowed",
        "dynamics_allowed",
        "diffusion_allowed",
        "replay_buffer_allowed",
        "native_gaussian_policy_allowed",
    ):
        if constraints.get(key) is not False:
            raise ValueError(f"assignment long smoke constraint {key} must be false")
    criteria = _mapping(payload, "success_criteria")
    if set(criteria) != set(OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SUCCESS_CRITERIA):
        raise ValueError("assignment long smoke success criteria mismatch")
    for name in OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SUCCESS_CRITERIA:
        _validate_criterion(criteria[name], name)
    gate = _mapping(payload, "readiness_gate")
    blockers = gate.get("blocked_reasons")
    if not isinstance(blockers, list) or any(not isinstance(item, str) for item in blockers):
        raise ValueError("assignment long smoke blocked_reasons must be strings")
    ready = not blockers
    if gate.get("long_smoke_contract_ready") is not ready:
        raise ValueError("assignment long smoke readiness gate contradicts blockers")
    if payload["status"] != (
        "objectstate_assignment_long_smoke_contract_ready"
        if ready
        else "objectstate_assignment_long_smoke_contract_blocked"
    ):
        raise ValueError("assignment long smoke status contradicts readiness")
    if ready:
        if audit["status"] != "pass" or not bool(audit["semantic_teacher_evidence_training_allowed"]):
            raise ValueError("ready assignment long smoke requires passed leakage audit")
        if gate.get("next_allowed_pr") != "OBJECTSTATE-ASSIGNMENT-LONG-SMOKE-001":
            raise ValueError("ready assignment long smoke next PR mismatch")
    elif gate.get("next_allowed_pr") is not None:
        raise ValueError("blocked assignment long smoke cannot name next PR")
    artifacts = _mapping(payload, "required_run_artifacts")
    if any(value is not True for value in artifacts.values()):
        raise ValueError("assignment long smoke required artifacts must be true")
    claim_policy = _mapping(payload, "claim_policy")
    if any(not bool(claim_policy.get(key)) for key in _CLAIM_POLICY_KEYS):
        raise ValueError("assignment long smoke must preserve claim policy")
    non_goals = _mapping(payload, "non_goals")
    if any(bool(non_goals.get(key)) for key in _NON_GOAL_KEYS):
        raise ValueError("assignment long smoke cannot claim non-goals")
    return dict(payload)


def _success_criteria(thresholds: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "held_out_identity_retrieval_at_1_not_decrease": {
            "metric": "identity_retrieval_at_1",
            "comparison": "after >= before - max_identity_retrieval_drop",
            "max_drop": float(thresholds["max_identity_retrieval_drop"]),
            "scope": "held_out_identity_benchmark",
        },
        "identity_margin_improves": {
            "metric": "identity_margin",
            "comparison": "after > before + min_identity_margin_delta",
            "min_delta": float(thresholds["min_identity_margin_delta"]),
            "scope": "held_out_identity_benchmark",
        },
        "occlusion_recovery_not_decrease": {
            "metric": "occlusion_recovery",
            "comparison": "after >= before - max_occlusion_recovery_drop",
            "max_drop": float(thresholds["max_occlusion_recovery_drop"]),
            "scope": "held_out_identity_benchmark",
        },
        "generalization_gap_not_expand": {
            "metric": "generalization_gap",
            "comparison": "after <= before + max_generalization_gap_increase",
            "max_increase": float(thresholds["max_generalization_gap_increase"]),
            "scope": "train_vs_held_out_assignment_and_identity",
        },
        "slot_swap_rate_interpretable": {
            "metric": "slot_swap_rate",
            "comparison": "finite and <= max_slot_swap_rate",
            "max_value": float(thresholds["max_slot_swap_rate"]),
            "scope": "identity_benchmark",
        },
        "checkpoint_roundtrip": {
            "metric": "checkpoint_roundtrip_ok",
            "comparison": "must_be_true",
            "scope": "assignment_solver_v2_checkpoint",
        },
    }


def _leakage_audit_digest(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "schema": OBJECTSTATE_TEACHER_EVIDENCE_LEAKAGE_AUDIT_SCHEMA,
            "status": "missing",
            "sample_id": None,
            "semantic_teacher_evidence_training_allowed": False,
            "blocked_checks": ["teacher_evidence_leakage_audit_missing"],
            "check_statuses": {},
            "artifact_refs": {},
        }
    checked = validate_objectstate_teacher_evidence_leakage_audit_summary(payload)
    return {
        "schema": checked["schema"],
        "status": (
            "pass"
            if checked["status"] == "objectstate_teacher_evidence_leakage_audit_pass"
            else "blocked"
        ),
        "sample_id": checked["sample_id"],
        "semantic_teacher_evidence_training_allowed": bool(
            checked["training_gate"]["semantic_teacher_evidence_training_allowed"]
        ),
        "blocked_checks": list(checked["training_gate"]["blocked_checks"]),
        "check_statuses": {
            name: checked["audit_checks"][name]["status"]
            for name in sorted(checked["audit_checks"])
        },
        "artifact_refs": dict(checked["artifact_refs"]),
    }


def _readiness_blockers(audit: Mapping[str, Any]) -> list[str]:
    blockers = []
    if audit["status"] == "missing":
        blockers.append("teacher_evidence_leakage_audit_missing")
    elif audit["status"] != "pass":
        blockers.append("teacher_evidence_leakage_audit_not_passed")
    if not bool(audit["semantic_teacher_evidence_training_allowed"]):
        blockers.append("semantic_teacher_evidence_training_not_allowed")
    blockers.extend(str(item) for item in audit["blocked_checks"])
    return _dedupe(blockers)


def _semantic_policy(value: str) -> str:
    policy = str(value).strip()
    if policy != OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_REQUIRED_POLICY:
        raise ValueError("assignment long smoke contract only supports policy=semantic")
    return policy


def _validate_leakage_audit_digest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != OBJECTSTATE_TEACHER_EVIDENCE_LEAKAGE_AUDIT_SCHEMA:
        raise ValueError("assignment long smoke leakage audit schema mismatch")
    if payload.get("status") not in {"missing", "pass", "blocked"}:
        raise ValueError("assignment long smoke leakage audit status unsupported")
    if payload["status"] != "missing" and not isinstance(payload.get("sample_id"), str):
        raise ValueError("assignment long smoke leakage audit digest requires sample_id")
    if not isinstance(payload.get("semantic_teacher_evidence_training_allowed"), bool):
        raise ValueError("assignment long smoke leakage audit training flag must be bool")
    if not isinstance(payload.get("blocked_checks"), list):
        raise ValueError("assignment long smoke leakage audit blocked_checks must be list")
    if not isinstance(payload.get("check_statuses"), Mapping):
        raise ValueError("assignment long smoke leakage audit check_statuses must be mapping")
    if not isinstance(payload.get("artifact_refs"), Mapping):
        raise ValueError("assignment long smoke leakage audit artifact_refs must be mapping")


def _validate_criterion(payload: Any, name: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"assignment long smoke criterion {name} must be a mapping")
    if not isinstance(payload.get("metric"), str) or not payload["metric"]:
        raise ValueError(f"assignment long smoke criterion {name} requires metric")
    if not isinstance(payload.get("comparison"), str) or not payload["comparison"]:
        raise ValueError(f"assignment long smoke criterion {name} requires comparison")
    if not isinstance(payload.get("scope"), str) or not payload["scope"]:
        raise ValueError(f"assignment long smoke criterion {name} requires scope")
    for numeric_key in ("max_drop", "min_delta", "max_increase", "max_value"):
        if numeric_key in payload:
            _finite(payload[numeric_key], f"{name}.{numeric_key}")


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"assignment long smoke contract requires {key}")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"assignment long smoke {label} must be a positive integer")
    return int(value)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"assignment long smoke {label} must be a number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"assignment long smoke {label} must be finite")
    return number


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
