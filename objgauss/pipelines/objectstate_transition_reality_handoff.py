from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.datasets.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    objectstate_controlled_real_manifest_from_capture_manifest,
    read_objectstate_controlled_capture_manifest,
    validate_objectstate_controlled_capture_manifest,
)
from objgauss.datasets.objectstate_controlled_capture_intervention_action_gt import (
    objectstate_controlled_capture_intervention_action_gt_readiness,
    validate_objectstate_controlled_capture_intervention_action_gt_readiness,
)
from objgauss.evaluation.objectstate_controlled_intervention_eval import (
    OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA,
    ObjectStateControlledInterventionThresholds,
    evaluate_objectstate_controlled_intervention_candidates,
    validate_objectstate_controlled_intervention_eval_summary,
)
from objgauss.evaluation.objectstate_controlled_prediction_eval import (
    OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA,
    ObjectStateControlledPredictionThresholds,
    evaluate_objectstate_controlled_prediction_candidates,
    validate_objectstate_controlled_prediction_eval_summary,
)
from objgauss.datasets.objectstate_controlled_real_manifest import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
    validate_objectstate_controlled_real_manifest,
)
from objgauss.evaluation.objectstate_controlled_real_rows import (
    OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
    objectstate_controlled_real_rows_summary,
    validate_objectstate_controlled_real_rows_summary,
)
from objgauss.evaluation.objectstate_reality_gate import ObjectStateRealityGateThresholds
from objgauss.datasets.objectstate_transition_dataset import (
    OBJECTSTATE_TRANSITION_DATASET_AUDIT_SCHEMA,
    OBJECTSTATE_TRANSITION_DATASET_SCHEMA,
    objectstate_transition_dataset_audit,
    read_objectstate_transition_dataset,
    validate_objectstate_transition_dataset,
    validate_objectstate_transition_dataset_audit,
)
from objgauss.pipelines.objectstate_transition_intervention_candidates import (
    OBJECTSTATE_TRANSITION_INTERVENTION_CANDIDATES_SCHEMA,
    objectstate_transition_intervention_candidates_summary,
    validate_objectstate_transition_intervention_candidates_summary,
)
from objgauss.pipelines.objectstate_transition_prediction_candidates import (
    OBJECTSTATE_TRANSITION_PREDICTION_CANDIDATES_SCHEMA,
    objectstate_transition_prediction_candidates_summary,
    validate_objectstate_transition_prediction_candidates_summary,
)

OBJECTSTATE_TRANSITION_REALITY_HANDOFF_SCHEMA = (
    "objgauss-objectstate-transition-reality-handoff-v1"
)


def objectstate_transition_reality_handoff(
    capture_manifest: Mapping[str, Any],
    transition_dataset: Mapping[str, Any],
    *,
    prediction_policy: str = "action_delta",
    intervention_policy: str = "action_delta",
    prediction_candidate_id: str = "transition-prediction-baseline-action-delta",
    intervention_candidate_id: str = "transition-intervention-baseline-action-delta",
    prediction_candidate_source: str | None = None,
    intervention_candidate_source: str | None = None,
    prediction_artifact_ref: str = "generated-transition-prediction-candidates",
    intervention_artifact_ref: str = "generated-transition-intervention-candidates",
    confidence: float = 0.5,
    min_object_episodes: int = 1,
    min_transitions: int = 1,
    min_action_conditioned_transitions: int = 1,
    min_horizon_seconds: float = 0.0,
    require_gaussian_refs: bool = False,
    prediction_thresholds: ObjectStateControlledPredictionThresholds | None = None,
    intervention_thresholds: ObjectStateControlledInterventionThresholds | None = None,
    synthetic_smoke_passed: bool = True,
) -> dict[str, Any]:
    capture = validate_objectstate_controlled_capture_manifest(capture_manifest)
    dataset = validate_objectstate_transition_dataset(transition_dataset)
    if capture["sample"]["sample_id"] != dataset["sample"]["sample_id"]:
        raise ValueError("transition handoff capture and dataset sample_id mismatch")
    intervention_action_gt = (
        objectstate_controlled_capture_intervention_action_gt_readiness(capture)
    )
    _require_intervention_action_gt_ready(intervention_action_gt)
    audit = objectstate_transition_dataset_audit(
        dataset,
        min_object_episodes=min_object_episodes,
        min_transitions=min_transitions,
        min_action_conditioned_transitions=min_action_conditioned_transitions,
        min_horizon_seconds=min_horizon_seconds,
        require_pose=True,
        require_action_transition=True,
        require_gaussian_refs=require_gaussian_refs,
    )
    prediction_summary = objectstate_transition_prediction_candidates_summary(
        dataset,
        policy=prediction_policy,
        candidate_id=prediction_candidate_id,
        candidate_source=prediction_candidate_source,
        artifact_ref=prediction_artifact_ref,
        confidence=confidence,
        require_action_transition=True,
    )
    intervention_summary = objectstate_transition_intervention_candidates_summary(
        dataset,
        policy=intervention_policy,
        candidate_id=intervention_candidate_id,
        candidate_source=intervention_candidate_source,
        artifact_ref=intervention_artifact_ref,
        confidence=confidence,
        require_intervention=True,
    )
    prediction_eval = evaluate_objectstate_controlled_prediction_candidates(
        capture,
        prediction_summary["prediction_candidates"],
        thresholds=prediction_thresholds,
    )
    intervention_eval = evaluate_objectstate_controlled_intervention_candidates(
        capture,
        intervention_summary["intervention_candidates"],
        thresholds=intervention_thresholds,
    )
    controlled_real_manifest = _merged_transition_controlled_real_manifest(
        objectstate_controlled_real_manifest_from_capture_manifest(capture),
        prediction_eval["controlled_real_manifest"],
        intervention_eval["controlled_real_manifest"],
    )
    controlled_real_summary = objectstate_controlled_real_rows_summary(
        controlled_real_manifest,
        synthetic_smoke_passed=bool(synthetic_smoke_passed),
        thresholds=ObjectStateRealityGateThresholds(
            min_real_or_public_rows=3,
            require_identity_pass_row=False,
            require_prediction_pass_row=True,
            require_intervention_pass_row=True,
            fail_on_failed_rows=True,
        ),
    )
    gates = {
        "intervention_action_gt_ready": bool(intervention_action_gt["ready"]),
        "transition_dataset_ready": bool(audit["readiness"]["transition_dataset_ready"]),
        "action_transition_ready": bool(
            audit["readiness"]["action_transition_count_ready"]
        ),
        "prediction_eval_pass": (
            prediction_eval["status"] == "objectstate_controlled_prediction_eval_pass"
        ),
        "intervention_eval_pass": (
            intervention_eval["status"]
            == "objectstate_controlled_intervention_eval_pass"
        ),
        "prediction_generator_target_pose_guard": bool(
            prediction_summary["claim_policy"][
                "does_not_read_target_pose_values_for_prediction"
            ]
        ),
        "intervention_generator_target_pose_guard": bool(
            intervention_summary["claim_policy"][
                "does_not_read_target_pose_values_for_prediction"
            ]
        ),
        "partial_reality_gate_pass": (
            controlled_real_summary["gate"]["status"] == "objectstate_reality_gate_pass"
        ),
    }
    payload = {
        "schema": OBJECTSTATE_TRANSITION_REALITY_HANDOFF_SCHEMA,
        "kind": "objectstate_transition_reality_handoff",
        "status": (
            "objectstate_transition_reality_handoff_pass"
            if all(gates.values())
            else "objectstate_transition_reality_handoff_fail"
        ),
        "capture_schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "transition_dataset_schema": OBJECTSTATE_TRANSITION_DATASET_SCHEMA,
        "transition_audit_schema": OBJECTSTATE_TRANSITION_DATASET_AUDIT_SCHEMA,
        "transition_prediction_candidates_schema": (
            OBJECTSTATE_TRANSITION_PREDICTION_CANDIDATES_SCHEMA
        ),
        "transition_intervention_candidates_schema": (
            OBJECTSTATE_TRANSITION_INTERVENTION_CANDIDATES_SCHEMA
        ),
        "prediction_eval_schema": OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA,
        "intervention_eval_schema": OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA,
        "controlled_real_manifest_schema": OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
        "controlled_real_rows_schema": OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
        "sample": dict(capture["sample"]),
        "candidate": {
            "prediction_candidate_id": prediction_eval["candidate"]["candidate_id"],
            "intervention_candidate_id": intervention_eval["candidate"]["candidate_id"],
            "prediction_artifact_refs": list(
                prediction_eval["candidate"]["artifact_refs"]
            ),
            "intervention_artifact_refs": list(
                intervention_eval["candidate"]["artifact_refs"]
            ),
        },
        "handoff_gates": gates,
        "issues": _handoff_issues(
            audit,
            prediction_eval,
            intervention_eval,
            controlled_real_summary,
        ),
        "intervention_action_gt": intervention_action_gt,
        "transition_audit": audit,
        "prediction_candidate_summary": prediction_summary,
        "intervention_candidate_summary": intervention_summary,
        "prediction_eval": prediction_eval,
        "intervention_eval": intervention_eval,
        "controlled_real_manifest": controlled_real_manifest,
        "controlled_real_summary": controlled_real_summary,
        "handoff_contract": {
            "uses_existing_controlled_capture_manifest": True,
            "uses_objectstate_transition_dataset": True,
            "requires_intervention_action_gt_ready": True,
            "audits_transition_dataset_before_eval": True,
            "exports_prediction_candidates": True,
            "exports_intervention_candidates": True,
            "runs_prediction_eval": True,
            "runs_intervention_eval": True,
            "merges_prediction_intervention_rows": True,
            "identity_row_stays_blocked_seed": True,
            "partial_gate_does_not_require_identity_pass": True,
        },
        "claim_policy": {
            "controlled_capture_ground_truth_required": True,
            "intervention_action_gt_preflight_required": True,
            "transition_dataset_required": True,
            "baseline_candidates_not_learned_model": True,
            "candidate_future_predictions_required": True,
            "candidate_action_predictions_required": True,
            "creates_transition_backed_prediction_row": True,
            "creates_transition_backed_intervention_row": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_identity": True,
            "does_not_evaluate_identity": True,
            "does_not_train_dynamics_model": True,
            "does_not_create_replay_buffer": True,
            "does_not_claim_counterfactual_proof": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "downloads_dataset": False,
            "creates_ground_truth": False,
            "infers_identity": False,
            "reconstructs_gaussians": False,
            "runs_tracking_model": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "creates_replay_buffer": False,
            "uses_diffusion": False,
            "writes_public_samples": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_transition_reality_handoff_summary(payload)


def write_objectstate_transition_reality_handoff(
    capture_manifest: str | Path,
    transition_dataset: str | Path,
    output_dir: str | Path,
    *,
    prediction_policy: str = "action_delta",
    intervention_policy: str = "action_delta",
    prediction_candidate_id: str = "transition-prediction-baseline-action-delta",
    intervention_candidate_id: str = "transition-intervention-baseline-action-delta",
    prediction_candidate_source: str | None = None,
    intervention_candidate_source: str | None = None,
    prediction_artifact_ref: str | None = None,
    intervention_artifact_ref: str | None = None,
    confidence: float = 0.5,
    min_object_episodes: int = 1,
    min_transitions: int = 1,
    min_action_conditioned_transitions: int = 1,
    min_horizon_seconds: float = 0.0,
    require_gaussian_refs: bool = False,
    prediction_thresholds: ObjectStateControlledPredictionThresholds | None = None,
    intervention_thresholds: ObjectStateControlledInterventionThresholds | None = None,
    synthetic_smoke_passed: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    capture_path = Path(capture_manifest)
    transition_path = Path(transition_dataset)
    output_root = Path(output_dir)
    files = _handoff_files(output_root)
    _ensure_can_write_many(files.values(), force=force)
    capture = read_objectstate_controlled_capture_manifest(capture_path)
    dataset = read_objectstate_transition_dataset(transition_path)
    prediction_ref = (
        str(files["prediction_candidates"])
        if prediction_artifact_ref is None
        else prediction_artifact_ref
    )
    intervention_ref = (
        str(files["intervention_candidates"])
        if intervention_artifact_ref is None
        else intervention_artifact_ref
    )
    summary = objectstate_transition_reality_handoff(
        capture,
        dataset,
        prediction_policy=prediction_policy,
        intervention_policy=intervention_policy,
        prediction_candidate_id=prediction_candidate_id,
        intervention_candidate_id=intervention_candidate_id,
        prediction_candidate_source=prediction_candidate_source,
        intervention_candidate_source=intervention_candidate_source,
        prediction_artifact_ref=prediction_ref,
        intervention_artifact_ref=intervention_ref,
        confidence=confidence,
        min_object_episodes=min_object_episodes,
        min_transitions=min_transitions,
        min_action_conditioned_transitions=min_action_conditioned_transitions,
        min_horizon_seconds=min_horizon_seconds,
        require_gaussian_refs=require_gaussian_refs,
        prediction_thresholds=prediction_thresholds,
        intervention_thresholds=intervention_thresholds,
        synthetic_smoke_passed=synthetic_smoke_passed,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(files["transition_audit"], summary["transition_audit"])
    _write_json(
        files["prediction_candidates"],
        summary["prediction_candidate_summary"]["prediction_candidates"],
    )
    _write_json(
        files["prediction_candidate_summary"],
        summary["prediction_candidate_summary"],
    )
    _write_json(
        files["intervention_candidates"],
        summary["intervention_candidate_summary"]["intervention_candidates"],
    )
    _write_json(
        files["intervention_candidate_summary"],
        summary["intervention_candidate_summary"],
    )
    _write_json(files["prediction_eval"], summary["prediction_eval"])
    _write_json(
        files["prediction_controlled_real"],
        summary["prediction_eval"]["controlled_real_manifest"],
    )
    _write_json(files["intervention_eval"], summary["intervention_eval"])
    _write_json(
        files["intervention_controlled_real"],
        summary["intervention_eval"]["controlled_real_manifest"],
    )
    _write_json(files["controlled_real"], summary["controlled_real_manifest"])
    _write_json(files["controlled_real_summary"], summary["controlled_real_summary"])
    files["blocked_rows"].write_text(
        summary["controlled_real_summary"]["blocked_rows_markdown"],
        encoding="utf-8",
    )
    checked = validate_objectstate_transition_reality_handoff_summary(
        {
            **summary,
            "source_capture_manifest": str(capture_path),
            "source_transition_dataset": str(transition_path),
            "output_dir": str(output_root),
            "files": {key: str(value) for key, value in files.items()},
        }
    )
    _write_json(files["handoff_summary"], checked)
    return checked


def validate_objectstate_transition_reality_handoff_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("transition reality handoff summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_TRANSITION_REALITY_HANDOFF_SCHEMA:
        raise ValueError(
            "unsupported transition reality handoff schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_transition_reality_handoff":
        raise ValueError("transition reality handoff kind is unsupported")
    if payload.get("status") not in {
        "objectstate_transition_reality_handoff_pass",
        "objectstate_transition_reality_handoff_fail",
    }:
        raise ValueError("transition reality handoff status is unsupported")
    if payload.get("capture_schema") != OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA:
        raise ValueError("transition reality handoff capture schema mismatch")
    if payload.get("transition_dataset_schema") != OBJECTSTATE_TRANSITION_DATASET_SCHEMA:
        raise ValueError("transition reality handoff dataset schema mismatch")
    if payload.get("transition_audit_schema") != OBJECTSTATE_TRANSITION_DATASET_AUDIT_SCHEMA:
        raise ValueError("transition reality handoff audit schema mismatch")
    if (
        payload.get("transition_prediction_candidates_schema")
        != OBJECTSTATE_TRANSITION_PREDICTION_CANDIDATES_SCHEMA
    ):
        raise ValueError("transition reality handoff prediction schema mismatch")
    if (
        payload.get("transition_intervention_candidates_schema")
        != OBJECTSTATE_TRANSITION_INTERVENTION_CANDIDATES_SCHEMA
    ):
        raise ValueError("transition reality handoff intervention schema mismatch")
    if payload.get("prediction_eval_schema") != OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA:
        raise ValueError("transition reality handoff prediction eval schema mismatch")
    if (
        payload.get("intervention_eval_schema")
        != OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA
    ):
        raise ValueError("transition reality handoff intervention eval schema mismatch")
    if (
        payload.get("controlled_real_manifest_schema")
        != OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA
    ):
        raise ValueError("transition reality handoff manifest schema mismatch")
    if payload.get("controlled_real_rows_schema") != OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA:
        raise ValueError("transition reality handoff rows schema mismatch")

    transition_audit = validate_objectstate_transition_dataset_audit(
        payload.get("transition_audit")
    )
    intervention_action_gt = (
        validate_objectstate_controlled_capture_intervention_action_gt_readiness(
            payload.get("intervention_action_gt")
        )
    )
    prediction_summary = validate_objectstate_transition_prediction_candidates_summary(
        payload.get("prediction_candidate_summary")
    )
    intervention_summary = validate_objectstate_transition_intervention_candidates_summary(
        payload.get("intervention_candidate_summary")
    )
    prediction_eval = validate_objectstate_controlled_prediction_eval_summary(
        payload.get("prediction_eval")
    )
    intervention_eval = validate_objectstate_controlled_intervention_eval_summary(
        payload.get("intervention_eval")
    )
    controlled_real_manifest = validate_objectstate_controlled_real_manifest(
        payload.get("controlled_real_manifest")
    )
    controlled_real_summary = validate_objectstate_controlled_real_rows_summary(
        payload.get("controlled_real_summary")
    )
    sample_id = transition_audit["sample"]["sample_id"]
    for name, child_sample_id in (
        ("prediction candidate", prediction_summary["sample_id"]),
        ("intervention candidate", intervention_summary["sample_id"]),
        ("prediction eval", prediction_eval["sample"]["sample_id"]),
        ("intervention eval", intervention_eval["sample"]["sample_id"]),
        ("controlled manifest", controlled_real_manifest["sample"]["sample_id"]),
        ("controlled summary", controlled_real_summary["sample"]["sample_id"]),
    ):
        if child_sample_id != sample_id:
            raise ValueError(f"transition reality handoff {name} sample mismatch")
    sample = payload.get("sample")
    if not isinstance(sample, Mapping) or sample.get("sample_id") != sample_id:
        raise ValueError("transition reality handoff sample mismatch")
    expected_manifest = _merged_transition_controlled_real_manifest(
        _identity_seed_manifest(controlled_real_manifest),
        prediction_eval["controlled_real_manifest"],
        intervention_eval["controlled_real_manifest"],
    )
    if _json_normalize(controlled_real_manifest) != _json_normalize(expected_manifest):
        raise ValueError("transition reality handoff controlled manifest mismatch")
    identity_row = _single_evidence_row(controlled_real_manifest, "identity")
    if identity_row["status"] != "blocked":
        raise ValueError("transition reality handoff must leave identity row blocked")
    summary_thresholds = controlled_real_summary["gate"]["thresholds"]
    if summary_thresholds["require_identity_pass_row"] is not False:
        raise ValueError(
            "transition reality handoff partial gate cannot require identity pass"
        )
    if summary_thresholds["require_prediction_pass_row"] is not True:
        raise ValueError(
            "transition reality handoff partial gate must require prediction pass"
        )
    if summary_thresholds["require_intervention_pass_row"] is not True:
        raise ValueError(
            "transition reality handoff partial gate must require intervention pass"
        )
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("transition reality handoff requires candidate")
    if (
        candidate.get("prediction_candidate_id")
        != prediction_eval["candidate"]["candidate_id"]
        or candidate.get("intervention_candidate_id")
        != intervention_eval["candidate"]["candidate_id"]
    ):
        raise ValueError("transition reality handoff candidate id mismatch")
    gates = payload.get("handoff_gates")
    if not isinstance(gates, Mapping) or any(
        not isinstance(value, bool) for value in gates.values()
    ):
        raise ValueError("transition reality handoff gates must be bools")
    expected_gates = {
        "intervention_action_gt_ready": bool(intervention_action_gt["ready"]),
        "transition_dataset_ready": bool(
            transition_audit["readiness"]["transition_dataset_ready"]
        ),
        "action_transition_ready": bool(
            transition_audit["readiness"]["action_transition_count_ready"]
        ),
        "prediction_eval_pass": (
            prediction_eval["status"] == "objectstate_controlled_prediction_eval_pass"
        ),
        "intervention_eval_pass": (
            intervention_eval["status"]
            == "objectstate_controlled_intervention_eval_pass"
        ),
        "prediction_generator_target_pose_guard": bool(
            prediction_summary["claim_policy"][
                "does_not_read_target_pose_values_for_prediction"
            ]
        ),
        "intervention_generator_target_pose_guard": bool(
            intervention_summary["claim_policy"][
                "does_not_read_target_pose_values_for_prediction"
            ]
        ),
        "partial_reality_gate_pass": (
            controlled_real_summary["gate"]["status"] == "objectstate_reality_gate_pass"
        ),
    }
    if dict(gates) != expected_gates:
        raise ValueError("transition reality handoff gates must match children")
    expected_status = (
        "objectstate_transition_reality_handoff_pass"
        if all(expected_gates.values())
        else "objectstate_transition_reality_handoff_fail"
    )
    if payload["status"] != expected_status:
        raise ValueError("transition reality handoff status must match gates")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("transition reality handoff requires issues")
    handoff_contract = payload.get("handoff_contract", {})
    if (
        not handoff_contract.get("uses_existing_controlled_capture_manifest")
        or not handoff_contract.get("uses_objectstate_transition_dataset")
        or not handoff_contract.get("requires_intervention_action_gt_ready")
        or not handoff_contract.get("audits_transition_dataset_before_eval")
        or not handoff_contract.get("exports_prediction_candidates")
        or not handoff_contract.get("exports_intervention_candidates")
        or not handoff_contract.get("runs_prediction_eval")
        or not handoff_contract.get("runs_intervention_eval")
        or not handoff_contract.get("merges_prediction_intervention_rows")
        or not handoff_contract.get("identity_row_stays_blocked_seed")
        or not handoff_contract.get("partial_gate_does_not_require_identity_pass")
    ):
        raise ValueError("transition reality handoff contract is incomplete")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("controlled_capture_ground_truth_required")
        or not claim_policy.get("intervention_action_gt_preflight_required")
        or not claim_policy.get("transition_dataset_required")
        or not claim_policy.get("baseline_candidates_not_learned_model")
        or not claim_policy.get("candidate_future_predictions_required")
        or not claim_policy.get("candidate_action_predictions_required")
        or not claim_policy.get("creates_transition_backed_prediction_row")
        or not claim_policy.get("creates_transition_backed_intervention_row")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_infer_identity")
        or not claim_policy.get("does_not_evaluate_identity")
        or not claim_policy.get("does_not_train_dynamics_model")
        or not claim_policy.get("does_not_create_replay_buffer")
        or not claim_policy.get("does_not_claim_counterfactual_proof")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("transition reality handoff must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("downloads_dataset")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("infers_identity")
        or non_goals.get("reconstructs_gaussians")
        or non_goals.get("runs_tracking_model")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("creates_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("writes_public_samples")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError(
            "transition reality handoff cannot capture, download, create GT, infer "
            "identity, reconstruct, track, train, replay, diffuse, write public "
            "samples, or mutate viewer defaults"
        )
    return dict(payload)


def _merged_transition_controlled_real_manifest(
    seed_manifest: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any],
    intervention_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    seed = validate_objectstate_controlled_real_manifest(seed_manifest)
    prediction = validate_objectstate_controlled_real_manifest(prediction_manifest)
    intervention = validate_objectstate_controlled_real_manifest(intervention_manifest)
    sample_id = seed["sample"]["sample_id"]
    for child in (prediction, intervention):
        if child["sample"]["sample_id"] != sample_id:
            raise ValueError("transition handoff child manifest sample mismatch")
        if child["sample"] != seed["sample"]:
            raise ValueError("transition handoff child manifest sample mismatch")
        if child["ground_truth"] != seed["ground_truth"]:
            raise ValueError("transition handoff child manifest ground truth mismatch")
    return validate_objectstate_controlled_real_manifest(
        {
            "schema": OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
            "sample": seed["sample"],
            "ground_truth": seed["ground_truth"],
            "evidence_rows": [
                _single_evidence_row(seed, "identity"),
                _single_evidence_row(prediction, "prediction"),
                _single_evidence_row(intervention, "intervention"),
            ],
        }
    )


def _identity_seed_manifest(
    controlled_real_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    checked = validate_objectstate_controlled_real_manifest(controlled_real_manifest)
    return validate_objectstate_controlled_real_manifest(
        {
            "schema": OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
            "sample": checked["sample"],
            "ground_truth": checked["ground_truth"],
            "evidence_rows": [
                _single_evidence_row(checked, "identity"),
                {
                    "evidence_kind": "prediction",
                    "status": "blocked",
                    "metrics": {},
                    "block_reason": "prediction row supplied by child eval",
                },
                {
                    "evidence_kind": "intervention",
                    "status": "blocked",
                    "metrics": {},
                    "block_reason": "intervention row supplied by child eval",
                },
            ],
        }
    )


def _single_evidence_row(
    manifest: Mapping[str, Any],
    evidence_kind: str,
) -> dict[str, Any]:
    rows = [
        row for row in manifest["evidence_rows"] if row["evidence_kind"] == evidence_kind
    ]
    if len(rows) != 1:
        raise ValueError(
            "transition handoff manifest must contain exactly one "
            f"{evidence_kind} row"
        )
    return dict(rows[0])


def _handoff_issues(
    transition_audit: Mapping[str, Any],
    prediction_eval: Mapping[str, Any],
    intervention_eval: Mapping[str, Any],
    controlled_real_summary: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []
    if not transition_audit["readiness"]["transition_dataset_ready"]:
        issues.append("transition dataset audit did not pass")
        issues.extend(str(item) for item in transition_audit["hard_blockers"])
    if prediction_eval["status"] != "objectstate_controlled_prediction_eval_pass":
        issues.append("controlled prediction eval did not pass")
        issues.extend(
            f"prediction gate failed: {key}"
            for key, value in prediction_eval["pass_gates"].items()
            if not value
        )
    if (
        intervention_eval["status"]
        != "objectstate_controlled_intervention_eval_pass"
    ):
        issues.append("controlled intervention eval did not pass")
        issues.extend(
            f"intervention gate failed: {key}"
            for key, value in intervention_eval["pass_gates"].items()
            if not value
        )
    if controlled_real_summary["gate"]["status"] != "objectstate_reality_gate_pass":
        issues.append("partial transition reality gate did not pass")
        issues.extend(
            f"partial reality gate failed: {key}"
            for key in controlled_real_summary["gate"]["hard_blockers"]
        )
    return issues


def _require_intervention_action_gt_ready(intervention_action_gt: Mapping[str, Any]) -> None:
    if intervention_action_gt.get("ready") is True:
        return
    issues = intervention_action_gt.get("issues", ())
    detail = "; ".join(str(item) for item in issues) if issues else "unknown issue"
    raise ValueError(
        "transition reality handoff requires intervention action GT readiness: "
        f"{detail}"
    )


def _handoff_files(output_dir: Path) -> dict[str, Path]:
    return {
        "transition_audit": output_dir / "transition-dataset-audit.json",
        "prediction_candidates": output_dir / "prediction-candidates.json",
        "prediction_candidate_summary": output_dir
        / "transition-prediction-summary.json",
        "prediction_eval": output_dir / "prediction-eval-summary.json",
        "prediction_controlled_real": output_dir / "prediction-controlled-real.json",
        "intervention_candidates": output_dir / "intervention-candidates.json",
        "intervention_candidate_summary": output_dir
        / "transition-intervention-summary.json",
        "intervention_eval": output_dir / "intervention-eval-summary.json",
        "intervention_controlled_real": output_dir / "intervention-controlled-real.json",
        "controlled_real": output_dir / "controlled-real.json",
        "controlled_real_summary": output_dir / "controlled-real-summary.json",
        "blocked_rows": output_dir / "blocked-rows.md",
        "handoff_summary": output_dir / "transition-reality-handoff-summary.json",
    }


def _ensure_can_write_many(paths: Sequence[Path], *, force: bool) -> None:
    for path in paths:
        if path.exists() and not force:
            raise FileExistsError(f"{path} already exists; pass force=True to overwrite")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    return value


__all__ = (
    "OBJECTSTATE_TRANSITION_REALITY_HANDOFF_SCHEMA",
    "objectstate_transition_reality_handoff",
    "write_objectstate_transition_reality_handoff",
    "validate_objectstate_transition_reality_handoff_summary",
)
