"""Controlled dataset contract helpers."""

from __future__ import annotations

from typing import Any, Mapping

from objgauss.datasets.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    objectstate_controlled_capture_summary,
    validate_objectstate_controlled_capture_manifest,
)
from objgauss.datasets.objectstate_controlled_capture_intervention_action_gt import (
    objectstate_controlled_capture_intervention_action_gt_readiness,
)

OBJECTSTATE_CONTROLLED_DATASET_CONTRACT_SCHEMA = (
    "objgauss-objectstate-controlled-dataset-contract-v1"
)
OBJECTSTATE_CONTROLLED_DATASET_CONTRACT_SUMMARY_SCHEMA = (
    "objgauss-objectstate-controlled-dataset-contract-summary-v1"
)


def objectstate_controlled_dataset_contract_summary(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    checked = validate_objectstate_controlled_capture_manifest(manifest)
    capture_summary = objectstate_controlled_capture_summary(checked)
    intervention_action_gt = (
        objectstate_controlled_capture_intervention_action_gt_readiness(checked)
    )
    invariant_status = {
        "identity_invariant": bool(capture_summary["readiness"]["identity_stage_ready"]),
        "prediction_invariant": bool(
            capture_summary["readiness"]["prediction_stage_ready"]
        ),
        "causal_invariant": bool(intervention_action_gt["ready"]),
    }
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_DATASET_CONTRACT_SUMMARY_SCHEMA,
        "kind": "objectstate_controlled_dataset_contract_summary",
        "contract_schema": OBJECTSTATE_CONTROLLED_DATASET_CONTRACT_SCHEMA,
        "source_schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": dict(checked["sample"]),
        "dataset_language": {
            "episode": {
                "sample_id": checked["sample"]["sample_id"],
                "scene_id": checked["sample"]["scenario"],
                "camera": checked["sample"]["capture_device"],
                "object_count": len(checked["objects"]),
                "action_count": len(checked["actions"]),
                "frame_count": len(checked["frames"]),
            },
            "object_instances": tuple(dict(item) for item in checked["objects"]),
            "action_events": tuple(dict(item) for item in checked["actions"]),
            "state_transition_source": {
                "type": "consecutive timestamped 6DoF pose rows",
                "requires_pose_gt": True,
                "transition_count": max(0, len(checked["frames"]) - 1)
                * len(checked["objects"]),
            },
        },
        "invariants": {
            "identity": {
                "ready": invariant_status["identity_invariant"],
                "definition": "same physical object_id must persist across timestamps",
            },
            "prediction": {
                "ready": invariant_status["prediction_invariant"],
                "definition": "S(t) and S(t+n) must be available from timestamped pose GT",
            },
            "causal": {
                "ready": invariant_status["causal_invariant"],
                "definition": (
                    "a non-zero ActionEvent must fit inside a referenced object's "
                    "consecutive pose transition interval"
                ),
                "intervention_action_gt": intervention_action_gt,
            },
        },
        "readiness": {
            "controlled_dataset_contract_ready": all(invariant_status.values()),
            "identity_ready": invariant_status["identity_invariant"],
            "prediction_ready": invariant_status["prediction_invariant"],
            "causal_ready": invariant_status["causal_invariant"],
        },
        "metrics": {
            "frame_count": len(checked["frames"]),
            "object_count": len(checked["objects"]),
            "action_count": len(checked["actions"]),
            "identity_track_count": len(capture_summary["object_track_counts"]),
            "usable_action_transition_count": intervention_action_gt["metrics"][
                "usable_action_transition_count"
            ],
        },
        "hard_blockers": _contract_blockers(capture_summary, intervention_action_gt),
        "claim_policy": {
            "reuses_controlled_capture_manifest": True,
            "does_not_create_ground_truth": True,
            "does_not_score_candidate_model": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "reconstructs_gaussians": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "runs_identity_eval": False,
            "runs_prediction_eval": False,
            "runs_intervention_eval": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_controlled_dataset_contract_summary(payload)


def validate_objectstate_controlled_dataset_contract_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled dataset contract summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_DATASET_CONTRACT_SUMMARY_SCHEMA:
        raise ValueError(
            "unsupported controlled dataset contract summary schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_dataset_contract_summary":
        raise ValueError("controlled dataset contract summary kind is unsupported")
    if payload.get("contract_schema") != OBJECTSTATE_CONTROLLED_DATASET_CONTRACT_SCHEMA:
        raise ValueError("controlled dataset contract summary has unsupported contract_schema")
    if payload.get("source_schema") != OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA:
        raise ValueError("controlled dataset contract summary has unsupported source_schema")
    if not isinstance(payload.get("sample"), Mapping):
        raise ValueError("controlled dataset contract summary requires sample")
    dataset_language = payload.get("dataset_language")
    if not isinstance(dataset_language, Mapping):
        raise ValueError("controlled dataset contract summary requires dataset_language")
    for key in (
        "episode",
        "object_instances",
        "action_events",
        "state_transition_source",
    ):
        if key not in dataset_language:
            raise ValueError(f"controlled dataset contract summary requires {key}")
    invariants = payload.get("invariants")
    if not isinstance(invariants, Mapping):
        raise ValueError("controlled dataset contract summary requires invariants")
    for key in ("identity", "prediction", "causal"):
        item = invariants.get(key)
        if not isinstance(item, Mapping) or not isinstance(item.get("ready"), bool):
            raise ValueError(f"controlled dataset contract invariant requires bool {key}")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("controlled dataset contract summary requires readiness")
    expected_ready = (
        invariants["identity"]["ready"]
        and invariants["prediction"]["ready"]
        and invariants["causal"]["ready"]
    )
    if readiness.get("controlled_dataset_contract_ready") is not expected_ready:
        raise ValueError("controlled dataset contract readiness must match invariants")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("controlled dataset contract summary requires metrics")
    for key in (
        "frame_count",
        "object_count",
        "action_count",
        "identity_track_count",
        "usable_action_transition_count",
    ):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"controlled dataset contract metric requires int {key}")
    hard_blockers = payload.get("hard_blockers")
    if not isinstance(hard_blockers, list) or any(
        not isinstance(item, str) for item in hard_blockers
    ):
        raise ValueError("controlled dataset contract hard_blockers must be strings")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("reuses_controlled_capture_manifest")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_score_candidate_model")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled dataset contract summary must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("reconstructs_gaussians")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("runs_identity_eval")
        or non_goals.get("runs_prediction_eval")
        or non_goals.get("runs_intervention_eval")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError("controlled dataset contract summary cannot claim non-goals")
    return dict(payload)


def _contract_blockers(
    capture_summary: Mapping[str, Any],
    intervention_action_gt: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    readiness = capture_summary["readiness"]
    if not readiness["identity_stage_ready"]:
        blockers.append("identity invariant requires timestamped object_id tracks")
    if not readiness["prediction_stage_ready"]:
        blockers.append("prediction invariant requires timestamped 6DoF pose tracks")
    if not intervention_action_gt["ready"]:
        blockers.extend(intervention_action_gt["issues"])
    return _dedupe(blockers)


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
