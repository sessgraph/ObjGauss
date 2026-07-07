from __future__ import annotations

from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_identity_eval import (
    OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA,
    OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA,
    ObjectStateControlledIdentityThresholds,
    evaluate_objectstate_controlled_identity_predictions,
    validate_objectstate_controlled_identity_eval_summary,
    validate_objectstate_controlled_identity_predictions,
)
from objgauss.core.objectstate_controlled_real_rows import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
    OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
    objectstate_controlled_real_rows_summary,
    validate_objectstate_controlled_real_manifest,
    validate_objectstate_controlled_real_rows_summary,
)
from objgauss.core.objectstate_identity_prediction_adapter import (
    objectstate_identity_predictions_from_trainable_artifact,
)
from objgauss.core.objectstate_reality_gate import ObjectStateRealityGateThresholds

OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA = (
    "objgauss-objectstate-controlled-identity-handoff-v1"
)


def objectstate_controlled_identity_handoff(
    capture_manifest: Mapping[str, Any],
    model_artifact: Mapping[str, Any],
    *,
    candidate_id: str | None = None,
    source: str = "trainable_kernel_objectstate_nearest_pose_adapter",
    artifact_refs: Sequence[str] | None = None,
    max_centroid_distance: float | None = None,
    identity_thresholds: ObjectStateControlledIdentityThresholds | None = None,
    synthetic_smoke_passed: bool = True,
    min_real_or_public_rows: int = 1,
) -> dict[str, Any]:
    predictions = objectstate_identity_predictions_from_trainable_artifact(
        capture_manifest,
        model_artifact,
        candidate_id=candidate_id,
        source=source,
        artifact_refs=artifact_refs,
        max_centroid_distance=max_centroid_distance,
    )
    identity_eval = evaluate_objectstate_controlled_identity_predictions(
        capture_manifest,
        predictions,
        thresholds=identity_thresholds,
    )
    controlled_real_manifest = identity_eval["controlled_real_manifest"]
    controlled_real_summary = objectstate_controlled_real_rows_summary(
        controlled_real_manifest,
        synthetic_smoke_passed=bool(synthetic_smoke_passed),
        thresholds=ObjectStateRealityGateThresholds(
            min_real_or_public_rows=int(min_real_or_public_rows),
            require_identity_pass_row=True,
            require_prediction_pass_row=False,
            require_intervention_pass_row=False,
            fail_on_failed_rows=True,
        ),
    )
    passed = (
        identity_eval["status"] == "objectstate_controlled_identity_eval_pass"
        and controlled_real_summary["gate"]["status"] == "objectstate_reality_gate_pass"
    )
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA,
        "kind": "objectstate_controlled_identity_handoff",
        "status": (
            "objectstate_controlled_identity_handoff_pass"
            if passed
            else "objectstate_controlled_identity_handoff_fail"
        ),
        "prediction_schema": OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA,
        "identity_eval_schema": OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA,
        "controlled_real_manifest_schema": OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
        "controlled_real_rows_schema": OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
        "sample": dict(identity_eval["sample"]),
        "candidate": dict(identity_eval["candidate"]),
        "identity_predictions": predictions,
        "identity_eval": identity_eval,
        "controlled_real_manifest": controlled_real_manifest,
        "controlled_real_summary": controlled_real_summary,
        "handoff_contract": {
            "writes_predictions": True,
            "writes_identity_eval": True,
            "writes_controlled_real_manifest": True,
            "writes_identity_only_gate_summary": True,
            "prediction_and_intervention_rows_remain_visible": True,
        },
        "claim_policy": {
            "capture_ground_truth_required": True,
            "candidate_artifact_required": True,
            "identity_only_stage1_gate": True,
            "does_not_claim_prediction_or_intervention": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "runs_tracking_model": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_controlled_identity_handoff_summary(payload)


def validate_objectstate_controlled_identity_handoff_summary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("controlled identity handoff summary must be a dict")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA:
        raise ValueError(
            f"unsupported controlled identity handoff schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_identity_handoff":
        raise ValueError("controlled identity handoff kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_identity_handoff_pass",
        "objectstate_controlled_identity_handoff_fail",
    }:
        raise ValueError("controlled identity handoff status is unsupported")
    if payload.get("prediction_schema") != OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA:
        raise ValueError("controlled identity handoff has unsupported prediction_schema")
    if payload.get("identity_eval_schema") != OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA:
        raise ValueError("controlled identity handoff has unsupported identity_eval_schema")
    if payload.get("controlled_real_manifest_schema") != OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA:
        raise ValueError("controlled identity handoff has unsupported controlled_real_manifest_schema")
    if payload.get("controlled_real_rows_schema") != OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA:
        raise ValueError("controlled identity handoff has unsupported controlled_real_rows_schema")

    predictions = validate_objectstate_controlled_identity_predictions(
        payload.get("identity_predictions")
    )
    identity_eval = validate_objectstate_controlled_identity_eval_summary(
        payload.get("identity_eval")
    )
    controlled_real_manifest = validate_objectstate_controlled_real_manifest(
        payload.get("controlled_real_manifest")
    )
    controlled_real_summary = validate_objectstate_controlled_real_rows_summary(
        payload.get("controlled_real_summary")
    )
    if predictions["sample_id"] != identity_eval["sample"]["sample_id"]:
        raise ValueError("controlled identity handoff sample ids must match")
    if identity_eval["controlled_real_manifest"] != controlled_real_manifest:
        raise ValueError("controlled identity handoff manifest must come from identity eval")
    if controlled_real_summary["sample"]["sample_id"] != identity_eval["sample"]["sample_id"]:
        raise ValueError("controlled identity handoff controlled-real summary sample mismatch")
    hard_blockers = set(controlled_real_summary["gate"].get("hard_blockers", ()))
    if "prediction_pass_rows_present" in hard_blockers:
        raise ValueError("controlled identity handoff must not require prediction pass rows")
    if "intervention_pass_rows_present" in hard_blockers:
        raise ValueError("controlled identity handoff must not require intervention pass rows")
    expected_status = (
        "objectstate_controlled_identity_handoff_pass"
        if identity_eval["status"] == "objectstate_controlled_identity_eval_pass"
        and controlled_real_summary["gate"]["status"] == "objectstate_reality_gate_pass"
        else "objectstate_controlled_identity_handoff_fail"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled identity handoff status must match embedded gates")
    handoff_contract = payload.get("handoff_contract", {})
    if (
        not handoff_contract.get("writes_predictions")
        or not handoff_contract.get("writes_identity_eval")
        or not handoff_contract.get("writes_controlled_real_manifest")
        or not handoff_contract.get("writes_identity_only_gate_summary")
        or not handoff_contract.get("prediction_and_intervention_rows_remain_visible")
    ):
        raise ValueError("controlled identity handoff contract is incomplete")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("capture_ground_truth_required")
        or not claim_policy.get("candidate_artifact_required")
        or not claim_policy.get("identity_only_stage1_gate")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled identity handoff must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("runs_tracking_model")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("writes_public_samples")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError("controlled identity handoff cannot claim capture, GT, tracking, training, replay, diffusion, or viewer mutation")
    return payload
