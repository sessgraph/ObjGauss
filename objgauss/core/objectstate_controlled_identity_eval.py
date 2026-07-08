from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    objectstate_controlled_capture_summary,
    objectstate_controlled_real_manifest_from_capture_manifest,
    read_objectstate_controlled_capture_manifest,
    validate_objectstate_controlled_capture_manifest,
)
from objgauss.core.objectstate_controlled_real_rows import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
    validate_objectstate_controlled_real_manifest,
)

OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA = (
    "objgauss-objectstate-controlled-identity-predictions-v1"
)
OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA = (
    "objgauss-objectstate-controlled-identity-eval-v1"
)


@dataclass(frozen=True)
class ObjectStateControlledIdentityThresholds:
    min_idf1: float = 0.95
    min_track_retrieval_recall_at_1: float = 0.95
    max_fragmentation_rate: float = 0.05
    max_long_term_drift_rate: float = 0.05
    max_swap_rate: float = 0.0
    min_reconstruction_noise_robustness: float = 0.95
    min_reconstruction_noise_variants: int = 2
    require_no_identity_collapse: bool = True

    def as_dict(self) -> dict[str, Any]:
        return validate_objectstate_controlled_identity_thresholds(self)


def read_objectstate_controlled_identity_predictions(path: str | Path) -> dict[str, Any]:
    prediction_path = Path(path)
    with prediction_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("controlled identity predictions JSON must be an object")
    return validate_objectstate_controlled_identity_predictions(payload)


def evaluate_objectstate_controlled_identity_predictions(
    capture_manifest: Mapping[str, Any],
    predictions: Mapping[str, Any],
    *,
    thresholds: ObjectStateControlledIdentityThresholds | None = None,
) -> dict[str, Any]:
    checked_capture = validate_objectstate_controlled_capture_manifest(capture_manifest)
    checked_predictions = validate_objectstate_controlled_identity_predictions(predictions)
    checked_thresholds = thresholds or ObjectStateControlledIdentityThresholds()
    checked_thresholds.as_dict()
    if checked_predictions["sample_id"] != checked_capture["sample"]["sample_id"]:
        raise ValueError("controlled identity predictions sample_id must match capture sample_id")
    capture_summary = objectstate_controlled_capture_summary(checked_capture)
    if not capture_summary["readiness"]["identity_stage_ready"]:
        raise ValueError("controlled capture manifest is not identity-stage ready")
    gt_pairs = _ground_truth_pairs(checked_capture)
    prediction_map = _prediction_map(checked_predictions, gt_pairs)
    metrics = _identity_metrics(
        checked_capture,
        gt_pairs,
        prediction_map,
        checked_predictions["candidate"],
    )
    pass_gates = _identity_pass_gates(metrics, checked_thresholds)
    passed = all(pass_gates.values())
    controlled_real_manifest = _controlled_real_manifest_with_identity_row(
        checked_capture,
        checked_predictions,
        metrics=metrics,
        passed=passed,
        pass_gates=pass_gates,
    )
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA,
        "kind": "objectstate_controlled_identity_eval",
        "capture_schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "prediction_schema": OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA,
        "status": (
            "objectstate_controlled_identity_eval_pass"
            if passed
            else "objectstate_controlled_identity_eval_fail"
        ),
        "sample": dict(checked_capture["sample"]),
        "candidate": dict(checked_predictions["candidate"]),
        "thresholds": checked_thresholds.as_dict(),
        "metrics": metrics,
        "pass_gates": pass_gates,
        "controlled_real_manifest": controlled_real_manifest,
        "claim_policy": {
            "capture_ground_truth_required": True,
            "candidate_predictions_required": True,
            "identity_metrics_drive_pass_fail": True,
            "does_not_claim_prediction_or_intervention": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "runs_tracking_model": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_controlled_identity_eval_summary(payload)


def validate_objectstate_controlled_identity_predictions(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled identity predictions must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA:
        raise ValueError(
            f"unsupported controlled identity predictions schema: {payload.get('schema')}"
        )
    candidate = _validate_candidate(payload.get("candidate"))
    predictions = tuple(
        _validate_prediction(item)
        for item in _sequence(payload.get("predictions"), "predictions")
    )
    if not predictions:
        raise ValueError("controlled identity predictions require at least one prediction")
    return {
        "schema": OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA,
        "sample_id": _required_string(payload, "sample_id"),
        "candidate": candidate,
        "predictions": predictions,
    }


def validate_objectstate_controlled_identity_eval_summary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("controlled identity eval summary must be a dict")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA:
        raise ValueError(f"unsupported controlled identity eval schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_controlled_identity_eval":
        raise ValueError("controlled identity eval summary kind is unsupported")
    if payload.get("capture_schema") != OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA:
        raise ValueError("controlled identity eval summary has unsupported capture_schema")
    if payload.get("prediction_schema") != OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA:
        raise ValueError("controlled identity eval summary has unsupported prediction_schema")
    if payload.get("status") not in {
        "objectstate_controlled_identity_eval_pass",
        "objectstate_controlled_identity_eval_fail",
    }:
        raise ValueError("controlled identity eval status is unsupported")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("controlled identity eval summary requires metrics")
    for key in (
        "idf1",
        "track_retrieval_recall_at_1",
        "long_term_drift_rate",
        "fragmentation_rate",
        "swap_rate",
        "identity_collapse",
        "reconstruction_noise_robustness",
        "reconstruction_noise_variant_count",
        "reconstruction_noise_evidence_present",
        "track_coverage",
        "evaluated_pairs",
        "missing_prediction_count",
    ):
        if key not in metrics:
            raise ValueError(f"controlled identity metrics missing {key}")
    pass_gates = payload.get("pass_gates")
    if not isinstance(pass_gates, dict) or any(
        not isinstance(value, bool) for value in pass_gates.values()
    ):
        raise ValueError("controlled identity pass_gates must be bool")
    expected_status = (
        "objectstate_controlled_identity_eval_pass"
        if all(pass_gates.values())
        else "objectstate_controlled_identity_eval_fail"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled identity status must match pass gates")
    validate_objectstate_controlled_real_manifest(payload.get("controlled_real_manifest"))
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("capture_ground_truth_required")
        or not claim_policy.get("candidate_predictions_required")
        or not claim_policy.get("identity_metrics_drive_pass_fail")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled identity eval summary must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("runs_tracking_model")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError("controlled identity eval cannot claim capture, GT, tracking, training, replay, diffusion, or viewer mutation")
    return payload


def validate_objectstate_controlled_identity_thresholds(
    thresholds: ObjectStateControlledIdentityThresholds,
) -> dict[str, Any]:
    if not isinstance(thresholds, ObjectStateControlledIdentityThresholds):
        raise TypeError("thresholds must be ObjectStateControlledIdentityThresholds")
    min_idf1 = float(thresholds.min_idf1)
    min_retrieval = float(thresholds.min_track_retrieval_recall_at_1)
    max_fragmentation = float(thresholds.max_fragmentation_rate)
    max_drift = float(thresholds.max_long_term_drift_rate)
    max_swap = float(thresholds.max_swap_rate)
    min_robustness = float(thresholds.min_reconstruction_noise_robustness)
    min_variants = thresholds.min_reconstruction_noise_variants
    if min_idf1 < 0.0 or min_idf1 > 1.0:
        raise ValueError("min_idf1 must be in [0, 1]")
    if min_retrieval < 0.0 or min_retrieval > 1.0:
        raise ValueError("min_track_retrieval_recall_at_1 must be in [0, 1]")
    if max_fragmentation < 0.0 or max_fragmentation > 1.0:
        raise ValueError("max_fragmentation_rate must be in [0, 1]")
    if max_drift < 0.0 or max_drift > 1.0:
        raise ValueError("max_long_term_drift_rate must be in [0, 1]")
    if max_swap < 0.0 or max_swap > 1.0:
        raise ValueError("max_swap_rate must be in [0, 1]")
    if min_robustness < 0.0 or min_robustness > 1.0:
        raise ValueError("min_reconstruction_noise_robustness must be in [0, 1]")
    if isinstance(min_variants, bool) or not isinstance(min_variants, int):
        raise TypeError("min_reconstruction_noise_variants must be an integer")
    if min_variants < 1:
        raise ValueError("min_reconstruction_noise_variants must be >= 1")
    return {
        "min_idf1": min_idf1,
        "min_track_retrieval_recall_at_1": min_retrieval,
        "max_fragmentation_rate": max_fragmentation,
        "max_long_term_drift_rate": max_drift,
        "max_swap_rate": max_swap,
        "min_reconstruction_noise_robustness": min_robustness,
        "min_reconstruction_noise_variants": int(min_variants),
        "require_no_identity_collapse": bool(thresholds.require_no_identity_collapse),
    }


def _controlled_real_manifest_with_identity_row(
    capture_manifest: Mapping[str, Any],
    predictions: Mapping[str, Any],
    *,
    metrics: Mapping[str, Any],
    passed: bool,
    pass_gates: Mapping[str, bool],
) -> dict[str, Any]:
    manifest = objectstate_controlled_real_manifest_from_capture_manifest(capture_manifest)
    identity_row = {
        "evidence_kind": "identity",
        "status": "pass" if passed else "fail",
        "metrics": {
            "idf1": metrics["idf1"],
            "track_retrieval_recall_at_1": metrics["track_retrieval_recall_at_1"],
            "long_term_drift_rate": metrics["long_term_drift_rate"],
            "fragmentation_rate": metrics["fragmentation_rate"],
            "swap_rate": metrics["swap_rate"],
            "identity_collapse": metrics["identity_collapse"],
            "reconstruction_noise_robustness": metrics[
                "reconstruction_noise_robustness"
            ]
            if metrics["reconstruction_noise_robustness"] is not None
            else 0.0,
            "reconstruction_noise_variant_count": metrics[
                "reconstruction_noise_variant_count"
            ],
            "reconstruction_noise_evidence_present": metrics[
                "reconstruction_noise_evidence_present"
            ],
        },
        "artifact_refs": tuple(
            list(capture_manifest["sample"]["artifact_refs"])
            + list(predictions["candidate"]["artifact_refs"])
        ),
    }
    if not passed:
        failed = ", ".join(key for key, value in pass_gates.items() if not value)
        identity_row["failure_reason"] = f"controlled identity metrics failed: {failed}"
    rows = list(manifest["evidence_rows"])
    rows[0] = identity_row
    manifest["evidence_rows"] = rows
    return validate_objectstate_controlled_real_manifest(manifest)


def _identity_metrics(
    capture_manifest: Mapping[str, Any],
    gt_pairs: tuple[tuple[str, str], ...],
    prediction_map: Mapping[tuple[str, str], str],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    object_ids = tuple(item["object_id"] for item in capture_manifest["objects"])
    frame_ids = tuple(frame["frame_id"] for frame in capture_manifest["frames"])
    predictions_by_object: dict[str, list[str]] = {object_id: [] for object_id in object_ids}
    correct = 0
    missing = 0
    majority_by_object: dict[str, str | None] = {}
    for object_id in object_ids:
        identities = [
            prediction_map[(frame_id, object_id)]
            for frame_id, pair_object_id in gt_pairs
            if pair_object_id == object_id and (frame_id, object_id) in prediction_map
        ]
        predictions_by_object[object_id] = identities
        majority_by_object[object_id] = _majority_identity(identities)
    for frame_id, object_id in gt_pairs:
        predicted = prediction_map.get((frame_id, object_id))
        if predicted is None:
            missing += 1
        elif predicted == majority_by_object[object_id]:
            correct += 1
    drift = _long_term_drift_summary(predictions_by_object)
    retrieval = _track_retrieval_summary(gt_pairs, prediction_map)
    swap_events, swap_denominator = _swap_events(
        frame_ids,
        object_ids,
        prediction_map,
    )
    identity_collapse = _identity_collapse(frame_ids, object_ids, prediction_map)
    reconstruction_noise = _reconstruction_noise_robustness_summary(candidate)
    evaluated = len(gt_pairs)
    predicted_count = len(prediction_map)
    return {
        "idf1": correct / evaluated if evaluated else 0.0,
        "track_retrieval_recall_at_1": retrieval["recall_at_1"],
        "track_retrieval_evaluated_count": retrieval["evaluated_count"],
        "track_retrieval_correct_count": retrieval["correct_count"],
        "long_term_drift_rate": drift["rate"],
        "long_term_drift_transition_count": drift["transition_count"],
        "long_term_drift_count": drift["drift_count"],
        "fragmentation_rate": drift["rate"],
        "swap_rate": swap_events / swap_denominator if swap_denominator else 0.0,
        "identity_collapse": bool(identity_collapse),
        "reconstruction_noise_robustness": reconstruction_noise["score"],
        "reconstruction_noise_variant_count": reconstruction_noise["variant_count"],
        "reconstruction_noise_evidence_present": reconstruction_noise["evidence_present"],
        "reconstruction_noise_source": reconstruction_noise["source"],
        "track_coverage": predicted_count / evaluated if evaluated else 0.0,
        "evaluated_pairs": evaluated,
        "correct_identity_pairs": correct,
        "missing_prediction_count": missing,
        "predicted_pair_count": predicted_count,
        "object_majority_identity": dict(majority_by_object),
    }


def _identity_pass_gates(
    metrics: Mapping[str, Any],
    thresholds: ObjectStateControlledIdentityThresholds,
) -> dict[str, bool]:
    return {
        "idf1_at_or_above_threshold": float(metrics["idf1"]) >= float(thresholds.min_idf1),
        "track_retrieval_at_or_above_threshold": (
            int(metrics["track_retrieval_evaluated_count"]) > 0
            and float(metrics["track_retrieval_recall_at_1"])
            >= float(thresholds.min_track_retrieval_recall_at_1)
        ),
        "fragmentation_at_or_below_threshold": (
            float(metrics["fragmentation_rate"]) <= float(thresholds.max_fragmentation_rate)
        ),
        "long_term_drift_at_or_below_threshold": (
            float(metrics["long_term_drift_rate"])
            <= float(thresholds.max_long_term_drift_rate)
        ),
        "swap_at_or_below_threshold": float(metrics["swap_rate"]) <= float(thresholds.max_swap_rate),
        "reconstruction_noise_evidence_present": bool(
            metrics["reconstruction_noise_evidence_present"]
        ),
        "reconstruction_noise_robustness_at_or_above_threshold": (
            metrics["reconstruction_noise_robustness"] is not None
            and float(metrics["reconstruction_noise_robustness"])
            >= float(thresholds.min_reconstruction_noise_robustness)
        ),
        "reconstruction_noise_variants_at_or_above_threshold": (
            int(metrics["reconstruction_noise_variant_count"])
            >= int(thresholds.min_reconstruction_noise_variants)
        ),
        "identity_collapse_absent": (
            not bool(thresholds.require_no_identity_collapse)
            or not bool(metrics["identity_collapse"])
        ),
    }


def _ground_truth_pairs(capture_manifest: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    pairs = []
    for frame in capture_manifest["frames"]:
        frame_id = frame["frame_id"]
        for item in frame["objects"]:
            pairs.append((frame_id, item["object_id"]))
    return tuple(pairs)


def _prediction_map(
    predictions: Mapping[str, Any],
    gt_pairs: tuple[tuple[str, str], ...],
) -> dict[tuple[str, str], str]:
    allowed_pairs = set(gt_pairs)
    result: dict[tuple[str, str], str] = {}
    for item in predictions["predictions"]:
        pair = (item["frame_id"], item["object_id"])
        if pair not in allowed_pairs:
            raise ValueError(
                "controlled identity prediction references unknown frame/object pair: "
                f"{pair[0]} / {pair[1]}"
            )
        if pair in result:
            raise ValueError(
                "controlled identity predictions contain duplicate frame/object pair: "
                f"{pair[0]} / {pair[1]}"
            )
        result[pair] = item["predicted_identity"]
    return result


def _majority_identity(identities: Sequence[str]) -> str | None:
    if not identities:
        return None
    counts: dict[str, int] = {}
    for identity in identities:
        counts[identity] = counts.get(identity, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _long_term_drift_summary(
    predictions_by_object: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    transitions = 0
    drifts = 0
    for identities in predictions_by_object.values():
        if len(identities) < 2:
            continue
        transitions += len(identities) - 1
        drifts += sum(
            1
            for previous, current in zip(identities, identities[1:])
            if previous != current
        )
    return {
        "transition_count": int(transitions),
        "drift_count": int(drifts),
        "rate": float(drifts / transitions) if transitions else 0.0,
    }


def _track_retrieval_summary(
    gt_pairs: Sequence[tuple[str, str]],
    prediction_map: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    owners_by_identity: dict[str, dict[str, int]] = {}
    for frame_id, object_id in gt_pairs:
        identity = prediction_map.get((frame_id, object_id))
        if identity is None:
            continue
        owners = owners_by_identity.setdefault(identity, {})
        owners[object_id] = owners.get(object_id, 0) + 1
    majority_owner_by_identity = {
        identity: sorted(owners.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for identity, owners in owners_by_identity.items()
        if owners
    }
    evaluated = 0
    correct = 0
    for frame_id, object_id in gt_pairs:
        identity = prediction_map.get((frame_id, object_id))
        if identity is None:
            continue
        evaluated += 1
        if majority_owner_by_identity.get(identity) == object_id:
            correct += 1
    return {
        "evaluated_count": int(evaluated),
        "correct_count": int(correct),
        "recall_at_1": float(correct / evaluated) if evaluated else 0.0,
    }


def _reconstruction_noise_robustness_summary(
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = candidate.get("identity_evidence")
    if evidence is None:
        return {
            "evidence_present": False,
            "score": None,
            "variant_count": 0,
            "source": None,
        }
    return {
        "evidence_present": True,
        "score": evidence["reconstruction_noise_robustness"],
        "variant_count": evidence["reconstruction_noise_variant_count"],
        "source": evidence["source"],
    }


def _swap_events(
    frame_ids: Sequence[str],
    object_ids: Sequence[str],
    prediction_map: Mapping[tuple[str, str], str],
) -> tuple[int, int]:
    swaps = 0
    denominator = 0
    for previous_frame, current_frame in zip(frame_ids, frame_ids[1:]):
        common = tuple(
            object_id
            for object_id in object_ids
            if (previous_frame, object_id) in prediction_map
            and (current_frame, object_id) in prediction_map
        )
        denominator += len(common)
        swapped_objects: set[str] = set()
        for left_index, left in enumerate(common):
            for right in common[left_index + 1:]:
                left_previous = prediction_map[(previous_frame, left)]
                left_current = prediction_map[(current_frame, left)]
                right_previous = prediction_map[(previous_frame, right)]
                right_current = prediction_map[(current_frame, right)]
                if (
                    left_previous == right_current
                    and right_previous == left_current
                    and left_previous != right_previous
                ):
                    swapped_objects.add(left)
                    swapped_objects.add(right)
        swaps += len(swapped_objects)
    return swaps, denominator


def _identity_collapse(
    frame_ids: Sequence[str],
    object_ids: Sequence[str],
    prediction_map: Mapping[tuple[str, str], str],
) -> bool:
    for frame_id in frame_ids:
        identities = [
            prediction_map[(frame_id, object_id)]
            for object_id in object_ids
            if (frame_id, object_id) in prediction_map
        ]
        if len(identities) != len(set(identities)):
            return True
    return False


def _validate_candidate(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled identity candidate must be a mapping")
    result = {
        "candidate_id": _required_string(value, "candidate_id"),
        "source": str(value.get("source", "unknown")),
        "artifact_refs": _string_tuple(value.get("artifact_refs"), "artifact_refs"),
    }
    if "identity_evidence" in value:
        result["identity_evidence"] = _validate_candidate_identity_evidence(
            value["identity_evidence"]
        )
    return result


def _validate_candidate_identity_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled identity candidate identity_evidence must be a mapping")
    robustness = _number(
        value.get("reconstruction_noise_robustness"),
        "reconstruction_noise_robustness",
    )
    if robustness < 0.0 or robustness > 1.0:
        raise ValueError("reconstruction_noise_robustness must be in [0, 1]")
    variant_count = _integer(
        value.get("reconstruction_noise_variant_count"),
        "reconstruction_noise_variant_count",
    )
    if variant_count < 1:
        raise ValueError("reconstruction_noise_variant_count must be >= 1")
    return {
        "reconstruction_noise_robustness": robustness,
        "reconstruction_noise_variant_count": variant_count,
        "source": _required_string(value, "source"),
    }


def _validate_prediction(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled identity prediction entries must be mappings")
    result = {
        "frame_id": _required_string(value, "frame_id"),
        "object_id": _required_string(value, "object_id"),
        "predicted_identity": _required_string(value, "predicted_identity"),
    }
    if "confidence" in value:
        confidence = _number(value["confidence"], "confidence")
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence must be in [0, 1]")
        result["confidence"] = confidence
    return result


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be a non-empty string")
    return result


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    normalized = tuple(str(item) for item in value)
    if not normalized or any(not item for item in normalized):
        raise ValueError(f"{name} must contain non-empty strings")
    return normalized


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)
