from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from objgauss.pipelines.objectstate_assignment_long_smoke import (
    OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SCHEMA,
    validate_objectstate_assignment_long_smoke_summary,
)

OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SCHEMA = (
    "objgauss-objectstate-temporal-assignment-contract-v1"
)
OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SUMMARY_SCHEMA = (
    "objgauss-objectstate-temporal-assignment-contract-summary-v1"
)
OBJECTSTATE_TEMPORAL_ASSIGNMENT_REQUIRED_POLICY = "semantic"
OBJECTSTATE_TEMPORAL_ASSIGNMENT_INPUTS = (
    "gaussian_t",
    "teacher_evidence_t",
    "assignment_t",
    "objectstate_t",
    "gaussian_t_plus_1",
    "teacher_evidence_t_plus_1",
    "assignment_t_plus_1",
    "objectstate_t_plus_1",
)
OBJECTSTATE_TEMPORAL_ASSIGNMENT_LOSS_TERMS = (
    "assignment_consistency",
    "objectstate_embedding_consistency",
    "slot_transition_smoothness",
)
OBJECTSTATE_TEMPORAL_ASSIGNMENT_METRICS = (
    "temporal_assignment_consistency",
    "identity_retrieval_at_1",
    "identity_margin",
    "slot_swap_rate",
    "occlusion_recovery",
    "track_fragmentation_rate",
    "checkpoint_roundtrip",
)
_CLAIM_POLICY_KEYS = (
    "semantic_policy_only",
    "requires_passed_assignment_long_smoke",
    "teacher_evidence_layer_remains_required",
    "assignment_matrix_is_single_source_of_truth",
    "physical_identity_labels_are_evaluation_only",
    "defines_temporal_consistency_without_running_training",
    "does_not_claim_temporal_assignment_implementation",
    "does_not_claim_real_data_identity_pass",
    "does_not_claim_world_model",
)
_NON_GOAL_KEYS = (
    "runs_temporal_training",
    "enables_assignment_solver_temporal_policy",
    "uses_renderer_loss",
    "uses_dynamics",
    "uses_diffusion",
    "uses_replay_buffer",
    "uses_gpu",
    "downloads_teacher_weights",
    "ingests_real_capture",
    "mutates_viewer_defaults",
)


@dataclass(frozen=True)
class ObjectStateTemporalAssignmentContractThresholds:
    max_duration_seconds: int = 600
    min_temporal_assignment_consistency: float = 0.95
    min_identity_retrieval_at_1: float = 0.95
    min_identity_margin: float = 0.0
    max_slot_swap_rate: float = 0.05
    max_occlusion_recovery_drop: float = 0.0
    max_track_fragmentation_rate: float = 0.10

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "max_duration_seconds": int(self.max_duration_seconds),
            "min_temporal_assignment_consistency": float(
                self.min_temporal_assignment_consistency
            ),
            "min_identity_retrieval_at_1": float(self.min_identity_retrieval_at_1),
            "min_identity_margin": float(self.min_identity_margin),
            "max_slot_swap_rate": float(self.max_slot_swap_rate),
            "max_occlusion_recovery_drop": float(self.max_occlusion_recovery_drop),
            "max_track_fragmentation_rate": float(self.max_track_fragmentation_rate),
        }
        if payload["max_duration_seconds"] < 1 or payload["max_duration_seconds"] > 600:
            raise ValueError("temporal assignment max_duration_seconds must be in [1,600]")
        for key in (
            "min_temporal_assignment_consistency",
            "min_identity_retrieval_at_1",
            "min_identity_margin",
            "max_slot_swap_rate",
            "max_occlusion_recovery_drop",
            "max_track_fragmentation_rate",
        ):
            value = payload[key]
            if not np.isfinite(value):
                raise ValueError(f"temporal assignment {key} must be finite")
        for key in (
            "min_temporal_assignment_consistency",
            "min_identity_retrieval_at_1",
            "max_slot_swap_rate",
            "max_occlusion_recovery_drop",
            "max_track_fragmentation_rate",
        ):
            if not 0.0 <= payload[key] <= 1.0:
                raise ValueError(f"temporal assignment {key} must be in [0,1]")
        if payload["min_identity_margin"] < 0.0:
            raise ValueError("temporal assignment min_identity_margin must be >= 0")
        return payload


def objectstate_temporal_assignment_contract_summary(
    *,
    sample_id: str = "objectstate-temporal-assignment-contract-001",
    evidence_policy: str = OBJECTSTATE_TEMPORAL_ASSIGNMENT_REQUIRED_POLICY,
    assignment_long_smoke_summary: Mapping[str, Any] | None = None,
    thresholds: ObjectStateTemporalAssignmentContractThresholds | None = None,
) -> dict[str, Any]:
    policy = _semantic_policy(evidence_policy)
    threshold_payload = (
        thresholds or ObjectStateTemporalAssignmentContractThresholds()
    ).as_dict()
    long_smoke = _long_smoke_digest(assignment_long_smoke_summary)
    blockers = _readiness_blockers(long_smoke)
    payload = {
        "schema": OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SUMMARY_SCHEMA,
        "kind": "objectstate_temporal_assignment_contract_summary",
        "contract_schema": OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SCHEMA,
        "status": (
            "objectstate_temporal_assignment_contract_ready"
            if not blockers
            else "objectstate_temporal_assignment_contract_blocked"
        ),
        "sample_id": str(sample_id),
        "evidence_policy": {
            "policy": policy,
            "requires_teacher_evidence_batch": True,
            "requires_passed_teacher_evidence_leakage_audit": True,
            "native_gaussian_temporal_training_allowed": False,
        },
        "preconditions": {
            "assignment_long_smoke_required": True,
            "assignment_long_smoke": long_smoke,
            "assignment_long_smoke_passed": long_smoke["status"] == "pass",
            "temporal_contract_unlocked": bool(
                long_smoke["temporal_assignment_contract_allowed"]
            ),
        },
        "duration_policy": {
            "max_duration_seconds": threshold_payload["max_duration_seconds"],
            "max_duration_minutes": threshold_payload["max_duration_seconds"] / 60.0,
            "bounded": True,
        },
        "temporal_inputs": _temporal_inputs(),
        "loss_contract": _loss_contract(),
        "success_metrics": _success_metrics(threshold_payload),
        "training_constraints": {
            "fixed_seed_required": True,
            "checkpoint_roundtrip_required": True,
            "before_after_identity_benchmark_required": True,
            "temporal_consistency_metrics_required": True,
            "slot_match_manifest_required": True,
            "renderer_loss_allowed": False,
            "dynamics_allowed": False,
            "diffusion_allowed": False,
            "replay_buffer_allowed": False,
            "native_gaussian_policy_allowed": False,
            "real_capture_required_for_first_smoke": False,
        },
        "readiness_gate": {
            "temporal_assignment_contract_ready": not blockers,
            "blocked_reasons": blockers,
            "next_allowed_pr": (
                "OBJECTSTATE-TEMPORAL-ASSIGNMENT-001" if not blockers else None
            ),
            "blocked_default_note": (
                "temporal assignment remains blocked until the semantic long "
                "smoke passes and explicitly allows this contract"
            ),
        },
        "required_run_artifacts": {
            "temporal_assignment_summary": True,
            "before_identity_benchmark_summary": True,
            "after_identity_benchmark_summary": True,
            "temporal_consistency_summary": True,
            "slot_match_manifest": True,
            "checkpoint_roundtrip_summary": True,
            "seed": True,
        },
        "claim_policy": {key: True for key in _CLAIM_POLICY_KEYS},
        "non_goals": {key: False for key in _NON_GOAL_KEYS},
    }
    return validate_objectstate_temporal_assignment_contract_summary(payload)


def validate_objectstate_temporal_assignment_contract_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("temporal assignment contract summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SUMMARY_SCHEMA:
        raise ValueError(
            "unsupported temporal assignment contract summary schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_temporal_assignment_contract_summary":
        raise ValueError("temporal assignment contract summary kind is unsupported")
    if payload.get("contract_schema") != OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SCHEMA:
        raise ValueError("temporal assignment contract schema mismatch")
    if payload.get("status") not in {
        "objectstate_temporal_assignment_contract_ready",
        "objectstate_temporal_assignment_contract_blocked",
    }:
        raise ValueError("temporal assignment contract status is unsupported")
    if not isinstance(payload.get("sample_id"), str) or not payload["sample_id"]:
        raise ValueError("temporal assignment contract requires sample_id")
    evidence_policy = _mapping(payload, "evidence_policy")
    if evidence_policy.get("policy") != OBJECTSTATE_TEMPORAL_ASSIGNMENT_REQUIRED_POLICY:
        raise ValueError("temporal assignment contract is semantic-policy only")
    if evidence_policy.get("native_gaussian_temporal_training_allowed") is not False:
        raise ValueError("temporal assignment cannot allow native Gaussian temporal training")
    preconditions = _mapping(payload, "preconditions")
    long_smoke = _mapping(preconditions, "assignment_long_smoke")
    _validate_long_smoke_digest(long_smoke)
    if preconditions.get("assignment_long_smoke_required") is not True:
        raise ValueError("temporal assignment must require assignment long smoke")
    duration = _mapping(payload, "duration_policy")
    max_duration = _positive_int(duration.get("max_duration_seconds"), "max_duration_seconds")
    if max_duration > 600 or duration.get("bounded") is not True:
        raise ValueError("temporal assignment duration must be bounded to <= 600s")
    inputs = _mapping(payload, "temporal_inputs")
    if tuple(inputs.get("required_inputs", ())) != OBJECTSTATE_TEMPORAL_ASSIGNMENT_INPUTS:
        raise ValueError("temporal assignment required inputs mismatch")
    if inputs.get("assignment_source") != "A[N,K]":
        raise ValueError("temporal assignment source must remain A[N,K]")
    if inputs.get("physical_identity_labels") != "evaluation_only":
        raise ValueError("physical identity labels must be evaluation only")
    losses = _mapping(payload, "loss_contract")
    if tuple(losses.get("allowed_loss_terms", ())) != OBJECTSTATE_TEMPORAL_ASSIGNMENT_LOSS_TERMS:
        raise ValueError("temporal assignment allowed loss terms mismatch")
    if losses.get("contract_only_no_training_run") is not True:
        raise ValueError("temporal assignment contract must not run training")
    metrics = _mapping(payload, "success_metrics")
    if set(metrics) != set(OBJECTSTATE_TEMPORAL_ASSIGNMENT_METRICS):
        raise ValueError("temporal assignment success metrics mismatch")
    for name in OBJECTSTATE_TEMPORAL_ASSIGNMENT_METRICS:
        _validate_metric(metrics[name], name)
    constraints = _mapping(payload, "training_constraints")
    for key in (
        "fixed_seed_required",
        "checkpoint_roundtrip_required",
        "before_after_identity_benchmark_required",
        "temporal_consistency_metrics_required",
        "slot_match_manifest_required",
    ):
        if constraints.get(key) is not True:
            raise ValueError(f"temporal assignment constraint {key} must be true")
    for key in (
        "renderer_loss_allowed",
        "dynamics_allowed",
        "diffusion_allowed",
        "replay_buffer_allowed",
        "native_gaussian_policy_allowed",
        "real_capture_required_for_first_smoke",
    ):
        if constraints.get(key) is not False:
            raise ValueError(f"temporal assignment constraint {key} must be false")
    gate = _mapping(payload, "readiness_gate")
    blockers = gate.get("blocked_reasons")
    if not isinstance(blockers, list) or any(not isinstance(item, str) for item in blockers):
        raise ValueError("temporal assignment blocked_reasons must be strings")
    ready = not blockers
    if gate.get("temporal_assignment_contract_ready") is not ready:
        raise ValueError("temporal assignment readiness contradicts blockers")
    if payload["status"] != (
        "objectstate_temporal_assignment_contract_ready"
        if ready
        else "objectstate_temporal_assignment_contract_blocked"
    ):
        raise ValueError("temporal assignment status contradicts readiness")
    if ready and gate.get("next_allowed_pr") != "OBJECTSTATE-TEMPORAL-ASSIGNMENT-001":
        raise ValueError("ready temporal assignment next PR mismatch")
    if not ready and gate.get("next_allowed_pr") is not None:
        raise ValueError("blocked temporal assignment cannot name next PR")
    artifacts = _mapping(payload, "required_run_artifacts")
    if any(value is not True for value in artifacts.values()):
        raise ValueError("temporal assignment required artifacts must be true")
    claim_policy = _mapping(payload, "claim_policy")
    if any(not bool(claim_policy.get(key)) for key in _CLAIM_POLICY_KEYS):
        raise ValueError("temporal assignment must preserve claim policy")
    non_goals = _mapping(payload, "non_goals")
    if any(bool(non_goals.get(key)) for key in _NON_GOAL_KEYS):
        raise ValueError("temporal assignment cannot claim non-goals")
    return dict(payload)


def _temporal_inputs() -> dict[str, Any]:
    return {
        "frame_pair_required": True,
        "required_inputs": list(OBJECTSTATE_TEMPORAL_ASSIGNMENT_INPUTS),
        "assignment_source": "A[N,K]",
        "objectstate_source": "ObjectStateProjection",
        "teacher_evidence_source": "TeacherEvidenceBatch",
        "track_hints": "optional_evidence_not_ground_truth",
        "physical_identity_labels": "evaluation_only",
        "hard_object_id_policy": "derived_from_assignment_or_slot_match_only",
    }


def _loss_contract() -> dict[str, Any]:
    return {
        "contract_only_no_training_run": True,
        "allowed_loss_terms": list(OBJECTSTATE_TEMPORAL_ASSIGNMENT_LOSS_TERMS),
        "term_definitions": {
            "assignment_consistency": {
                "target": "A_t and A_t_plus_1 agree for matched Gaussian or track evidence",
                "uses_ground_truth_identity": False,
            },
            "objectstate_embedding_consistency": {
                "target": "same matched slot keeps nearby ObjectState embedding",
                "uses_ground_truth_identity": False,
            },
            "slot_transition_smoothness": {
                "target": "slot prototypes move smoothly under bounded frame interval",
                "uses_ground_truth_identity": False,
            },
        },
        "forbidden_loss_terms": [
            "renderer_loss",
            "dynamics_loss",
            "diffusion_loss",
            "replay_buffer_loss",
            "oracle_identity_loss",
        ],
    }


def _success_metrics(thresholds: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "temporal_assignment_consistency": {
            "comparison": ">= min_temporal_assignment_consistency",
            "threshold": float(thresholds["min_temporal_assignment_consistency"]),
            "scope": "matched_frame_pair_assignments",
        },
        "identity_retrieval_at_1": {
            "comparison": ">= min_identity_retrieval_at_1",
            "threshold": float(thresholds["min_identity_retrieval_at_1"]),
            "scope": "held_out_identity_benchmark",
        },
        "identity_margin": {
            "comparison": "> min_identity_margin",
            "threshold": float(thresholds["min_identity_margin"]),
            "scope": "held_out_identity_benchmark",
        },
        "slot_swap_rate": {
            "comparison": "<= max_slot_swap_rate",
            "threshold": float(thresholds["max_slot_swap_rate"]),
            "scope": "temporal_slot_matching",
        },
        "occlusion_recovery": {
            "comparison": "not_decrease_beyond_max_occlusion_recovery_drop",
            "threshold": float(thresholds["max_occlusion_recovery_drop"]),
            "scope": "occlusion_perturbation_frames",
        },
        "track_fragmentation_rate": {
            "comparison": "<= max_track_fragmentation_rate",
            "threshold": float(thresholds["max_track_fragmentation_rate"]),
            "scope": "matched_tracks_or_synthetic_correspondence",
        },
        "checkpoint_roundtrip": {
            "comparison": "must_be_true",
            "threshold": True,
            "scope": "assignment_solver_v2_checkpoint",
        },
    }


def _long_smoke_digest(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "schema": OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SCHEMA,
            "status": "missing",
            "sample_id": None,
            "evidence_policy": None,
            "temporal_assignment_contract_allowed": False,
            "blocked_checks": ["assignment_long_smoke_missing"],
        }
    try:
        checked = validate_objectstate_assignment_long_smoke_summary(payload)
    except (TypeError, ValueError) as exc:
        return {
            "schema": str(payload.get("schema", "")),
            "status": "invalid",
            "sample_id": payload.get("sample_id"),
            "evidence_policy": None,
            "temporal_assignment_contract_allowed": False,
            "blocked_checks": ["assignment_long_smoke_invalid"],
            "error": str(exc),
        }
    gate = _mapping(checked, "next_stage_gate")
    run_config = _mapping(checked, "run_config")
    blocked = [
        str(item)
        for item in gate.get("blocked_reasons", [])
        if isinstance(item, str)
    ]
    if checked["status"] != "objectstate_assignment_long_smoke_pass":
        blocked.append("assignment_long_smoke_not_passed")
    if gate.get("temporal_assignment_contract_allowed") is not True:
        blocked.append("temporal_assignment_contract_not_allowed")
    return {
        "schema": checked["schema"],
        "status": (
            "pass"
            if checked["status"] == "objectstate_assignment_long_smoke_pass"
            else "reviewable"
        ),
        "sample_id": checked["sample_id"],
        "evidence_policy": run_config["evidence_policy"],
        "temporal_assignment_contract_allowed": bool(
            gate.get("temporal_assignment_contract_allowed")
        ),
        "blocked_checks": sorted(set(blocked)),
        "summary_path": checked.get("summary_path"),
    }


def _readiness_blockers(long_smoke: Mapping[str, Any]) -> list[str]:
    blockers = []
    if long_smoke["status"] != "pass":
        blockers.append(f"assignment_long_smoke_{long_smoke['status']}")
    if long_smoke["evidence_policy"] != OBJECTSTATE_TEMPORAL_ASSIGNMENT_REQUIRED_POLICY:
        blockers.append("assignment_long_smoke_policy_not_semantic")
    if long_smoke["temporal_assignment_contract_allowed"] is not True:
        blockers.append("assignment_long_smoke_did_not_allow_temporal_contract")
    blockers.extend(str(item) for item in long_smoke.get("blocked_checks", ()))
    return sorted(set(blockers))


def _validate_long_smoke_digest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SCHEMA:
        raise ValueError("temporal assignment long-smoke digest schema mismatch")
    if payload.get("status") not in {"missing", "invalid", "reviewable", "pass"}:
        raise ValueError("temporal assignment long-smoke digest status unsupported")
    if payload.get("status") == "pass":
        if payload.get("evidence_policy") != OBJECTSTATE_TEMPORAL_ASSIGNMENT_REQUIRED_POLICY:
            raise ValueError("temporal assignment requires semantic long-smoke policy")
        if payload.get("temporal_assignment_contract_allowed") is not True:
            raise ValueError("temporal assignment requires long-smoke next-stage allow")
    blocked = payload.get("blocked_checks")
    if not isinstance(blocked, list) or any(not isinstance(item, str) for item in blocked):
        raise ValueError("temporal assignment long-smoke blocked checks must be strings")


def _validate_metric(payload: Mapping[str, Any], name: str) -> None:
    if payload.get("scope") in {None, ""}:
        raise ValueError(f"temporal assignment metric {name} requires scope")
    comparison = payload.get("comparison")
    if not isinstance(comparison, str) or not comparison:
        raise ValueError(f"temporal assignment metric {name} requires comparison")
    threshold = payload.get("threshold")
    if name == "checkpoint_roundtrip":
        if threshold is not True:
            raise ValueError("temporal assignment checkpoint threshold must be true")
        return
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(f"temporal assignment metric {name} threshold must be numeric")
    if not np.isfinite(float(threshold)):
        raise ValueError(f"temporal assignment metric {name} threshold must be finite")


def _semantic_policy(value: str) -> str:
    policy = str(value).strip()
    if policy != OBJECTSTATE_TEMPORAL_ASSIGNMENT_REQUIRED_POLICY:
        raise ValueError("temporal assignment contract requires policy=semantic")
    return policy


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"temporal assignment contract requires {key}")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"temporal assignment {label} must be a positive integer")
    return int(value)


__all__ = (
    "OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SCHEMA",
    "OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SUMMARY_SCHEMA",
    "OBJECTSTATE_TEMPORAL_ASSIGNMENT_REQUIRED_POLICY",
    "OBJECTSTATE_TEMPORAL_ASSIGNMENT_INPUTS",
    "OBJECTSTATE_TEMPORAL_ASSIGNMENT_LOSS_TERMS",
    "OBJECTSTATE_TEMPORAL_ASSIGNMENT_METRICS",
    "ObjectStateTemporalAssignmentContractThresholds",
    "objectstate_temporal_assignment_contract_summary",
    "validate_objectstate_temporal_assignment_contract_summary",
)
