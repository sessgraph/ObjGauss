from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from objgauss.core.assignment_evidence import AssignmentEvidenceBatch
from objgauss.core.assignment_solver_v2 import (
    ASSIGNMENT_SOLVER_V2_STATE_SCHEMA,
    AssignmentSolverV2State,
    predict_assignment_solver_v2,
    validate_assignment_solver_v2_state,
)
from objgauss.core.features import extract_features, positions
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.io import append_or_replace_property, write_ply
from objgauss.core.object_state import project_object_states, validate_assignment_matrix
from objgauss.core.objectstate_assignment_mvp import _projection_summary

OBJECTSTATE_MODEL_IDENTITY_GATE_SCHEMA = (
    "objgauss-objectstate-model-identity-gate-v1"
)
OBJECTSTATE_MODEL_IDENTITY_BASELINES = (
    "random_assignment",
    "xyz_centroid",
    "oracle_target_assignment",
    "assignment_solver_v2",
)


@dataclass(frozen=True)
class ObjectStateModelIdentityGateThresholds:
    identity_retrieval_at_1_min: float = 0.95
    identity_margin_min: float = 0.0
    assignment_consistency_min: float = 0.95
    objectstate_drift_max: float = 0.25

    def as_dict(self) -> dict[str, float]:
        payload = {
            "identity_retrieval_at_1_min": float(self.identity_retrieval_at_1_min),
            "identity_margin_min": float(self.identity_margin_min),
            "assignment_consistency_min": float(self.assignment_consistency_min),
            "objectstate_drift_max": float(self.objectstate_drift_max),
        }
        for key, value in payload.items():
            if key.endswith("_min") and not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} must be in [0,1]")
            if key.endswith("_max") and value < 0.0:
                raise ValueError(f"{key} must be >= 0")
        return payload


def objectstate_model_identity_gate_summary(
    frame0_cloud: GaussianCloud,
    frame0_identity_labels: np.ndarray,
    frame1_cloud: GaussianCloud,
    frame1_identity_labels: np.ndarray,
    solver_state: AssignmentSolverV2State,
    *,
    output_dir: str | Path,
    sample_id: str = "model-identity-gate-001",
    frame0_id: str = "t0",
    frame1_id: str = "t1",
    frame0_features: np.ndarray | None = None,
    frame1_features: np.ndarray | None = None,
    thresholds: ObjectStateModelIdentityGateThresholds | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    state = validate_assignment_solver_v2_state(solver_state)
    labels0 = _identity_labels(frame0_identity_labels, frame0_cloud.count, "frame0_identity_labels")
    labels1 = _identity_labels(frame1_identity_labels, frame1_cloud.count, "frame1_identity_labels")
    identities = _common_identities(labels0, labels1)
    if len(identities) < 2:
        raise ValueError("model identity gate requires at least two shared physical identities")
    features0 = _features(frame0_cloud, frame0_features, state.config.feature_dim, "frame0_features")
    features1 = _features(frame1_cloud, frame1_features, state.config.feature_dim, "frame1_features")
    if features0.shape[1] != state.config.feature_dim or features1.shape[1] != state.config.feature_dim:
        raise ValueError("frame feature_dim must match solver state")
    checked_thresholds = thresholds or ObjectStateModelIdentityGateThresholds()
    threshold_payload = checked_thresholds.as_dict()

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    assignments = _baseline_assignments(
        frame0_cloud,
        labels0,
        frame1_cloud,
        labels1,
        state,
        features0,
        features1,
        identities=identities,
        seed=seed,
    )
    baseline_summaries = {}
    for name, (assignment0, assignment1) in assignments.items():
        baseline_summaries[name] = _evaluate_identity_candidate(
            name,
            frame0_cloud,
            labels0,
            frame1_cloud,
            labels1,
            assignment0,
            assignment1,
            features0,
            features1,
            identities=identities,
        )
    candidate = baseline_summaries["assignment_solver_v2"]
    gate = _gate_status(candidate["metrics"], threshold_payload)
    artifact_refs = _write_artifacts(
        output_root,
        candidate,
        frame0_cloud,
        frame1_cloud,
        np.asarray(candidate["frame0"]["derived_object_ids"], dtype=np.int32),
        np.asarray(candidate["frame1"]["derived_object_ids"], dtype=np.int32),
    )
    payload = {
        "schema": OBJECTSTATE_MODEL_IDENTITY_GATE_SCHEMA,
        "kind": "objectstate_model_identity_gate",
        "status": gate["status"],
        "gate_status": gate["gate_status"],
        "sample_id": str(sample_id),
        "frames": {
            "frame0_id": str(frame0_id),
            "frame1_id": str(frame1_id),
            "frame0_gaussian_count": int(frame0_cloud.count),
            "frame1_gaussian_count": int(frame1_cloud.count),
            "shared_identity_count": len(identities),
            "shared_identities": [int(item) for item in identities],
        },
        "solver": {
            "schema": ASSIGNMENT_SOLVER_V2_STATE_SCHEMA,
            "family": state.config.solver_family,
            "slots": int(state.config.slots),
            "feature_dim": int(state.config.feature_dim),
            "step": int(state.step),
            "source": state.source,
        },
        "thresholds": threshold_payload,
        "candidate": candidate,
        "metrics": candidate["metrics"],
        "baselines": baseline_summaries,
        "baseline_comparison": _baseline_comparison(baseline_summaries),
        "hard_gates": gate["hard_gates"],
        "hard_blockers": gate["hard_blockers"],
        "artifact_refs": artifact_refs,
        "claim_policy": {
            "uses_permutation_aware_identity_matching": True,
            "hard_object_id_is_derived": True,
            "assignment_matrix_is_single_source_of_truth": True,
            "tests_identity_state_only": True,
            "does_not_claim_prediction_gate_pass": True,
            "does_not_claim_causal_gate_pass": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "trains_model": False,
            "uses_renderer_loss": False,
            "uses_temporal_loss": False,
            "uses_hungarian_dependency": False,
            "uses_transformer": False,
            "uses_slot_attention": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "uses_dynamics_model": False,
            "runs_long_training": False,
            "mutates_viewer_defaults": False,
        },
    }
    checked = validate_objectstate_model_identity_gate_summary(payload)
    summary_path = output_root / "identity-summary.json"
    summary_path.write_text(json.dumps(checked, indent=2, sort_keys=True), encoding="utf-8")
    checked["summary_path"] = str(summary_path)
    return validate_objectstate_model_identity_gate_summary(checked)


def validate_objectstate_model_identity_gate_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("model identity gate summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_MODEL_IDENTITY_GATE_SCHEMA:
        raise ValueError(f"unsupported model identity gate schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_model_identity_gate":
        raise ValueError("model identity gate kind is unsupported")
    if payload.get("status") not in {
        "objectstate_model_identity_gate_pass",
        "objectstate_model_identity_gate_fail",
    }:
        raise ValueError("model identity gate status is unsupported")
    metrics = _mapping(payload, "metrics")
    for key in (
        "identity_retrieval_at_1",
        "identity_margin",
        "slot_swap_rate",
        "objectstate_drift",
        "assignment_consistency",
        "occlusion_recovery",
    ):
        _finite(metrics.get(key), f"metrics.{key}")
    baselines = _mapping(payload, "baselines")
    for name in OBJECTSTATE_MODEL_IDENTITY_BASELINES:
        if name not in baselines:
            raise ValueError(f"model identity gate missing baseline {name}")
    hard_gates = _mapping(payload, "hard_gates")
    if not hard_gates or any(not isinstance(value, bool) for value in hard_gates.values()):
        raise ValueError("model identity gate hard_gates must be booleans")
    expected = (
        "objectstate_model_identity_gate_pass"
        if all(bool(value) for value in hard_gates.values())
        else "objectstate_model_identity_gate_fail"
    )
    if payload["status"] != expected:
        raise ValueError("model identity gate status must match hard_gates")
    claim_policy = _mapping(payload, "claim_policy")
    if (
        not claim_policy.get("uses_permutation_aware_identity_matching")
        or not claim_policy.get("hard_object_id_is_derived")
        or not claim_policy.get("assignment_matrix_is_single_source_of_truth")
        or not claim_policy.get("tests_identity_state_only")
        or not claim_policy.get("does_not_claim_prediction_gate_pass")
        or not claim_policy.get("does_not_claim_causal_gate_pass")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("model identity gate must preserve claim policy")
    non_goals = _mapping(payload, "non_goals")
    if any(bool(value) for value in non_goals.values()):
        raise ValueError("model identity gate cannot claim non-goals")
    if "summary_path" in payload and not isinstance(payload["summary_path"], str):
        raise ValueError("model identity gate summary_path must be a string")
    return dict(payload)


def _baseline_assignments(
    frame0_cloud: GaussianCloud,
    labels0: np.ndarray,
    frame1_cloud: GaussianCloud,
    labels1: np.ndarray,
    state: AssignmentSolverV2State,
    features0: np.ndarray,
    features1: np.ndarray,
    *,
    identities: tuple[int, ...],
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    slots = len(identities)
    solver0 = predict_assignment_solver_v2(
        AssignmentEvidenceBatch(positions(frame0_cloud), features0, source="model-identity:t0"),
        state,
    ).assignment
    solver1 = predict_assignment_solver_v2(
        AssignmentEvidenceBatch(positions(frame1_cloud), features1, source="model-identity:t1"),
        state,
    ).assignment
    return {
        "random_assignment": (
            _random_assignment(frame0_cloud.count, slots, seed=seed),
            _random_assignment(frame1_cloud.count, slots, seed=seed + 1),
        ),
        "xyz_centroid": (
            _xyz_centroid_assignment(frame0_cloud, slots),
            _xyz_centroid_assignment(frame1_cloud, slots),
        ),
        "oracle_target_assignment": (
            _oracle_assignment(labels0, identities),
            _oracle_assignment(labels1, identities),
        ),
        "assignment_solver_v2": (solver0, solver1),
    }


def _evaluate_identity_candidate(
    name: str,
    frame0_cloud: GaussianCloud,
    labels0: np.ndarray,
    frame1_cloud: GaussianCloud,
    labels1: np.ndarray,
    assignment0: np.ndarray,
    assignment1: np.ndarray,
    features0: np.ndarray,
    features1: np.ndarray,
    *,
    identities: tuple[int, ...],
) -> dict[str, Any]:
    assignment0 = validate_assignment_matrix(assignment0, evidence_count=frame0_cloud.count)
    assignment1 = validate_assignment_matrix(assignment1, evidence_count=frame1_cloud.count)
    projection0 = project_object_states(frame0_cloud, assignment0, evidence_features=features0)
    projection1 = project_object_states(frame1_cloud, assignment1, evidence_features=features1)
    identity_slots0 = _identity_slots(assignment0, labels0, identities)
    identity_slots1 = _identity_slots(assignment1, labels1, identities)
    embeddings0 = _identity_embeddings(projection0, identity_slots0, identities)
    embeddings1 = _identity_embeddings(projection1, identity_slots1, identities)
    distance_rows, retrieval_rows = _pairwise_identity_distances(embeddings0, embeddings1)
    metrics = _identity_metrics(
        identity_slots0,
        identity_slots1,
        distance_rows,
        retrieval_rows,
        identities=identities,
    )
    return {
        "name": name,
        "frame0": {
            "projection": _projection_summary(projection0),
            "identity_slots": {str(key): int(value) for key, value in identity_slots0.items()},
            "derived_object_ids": projection0.derived_object_ids.astype(int).tolist(),
        },
        "frame1": {
            "projection": _projection_summary(projection1),
            "identity_slots": {str(key): int(value) for key, value in identity_slots1.items()},
            "derived_object_ids": projection1.derived_object_ids.astype(int).tolist(),
        },
        "matching": retrieval_rows,
        "pairwise_distances": distance_rows,
        "metrics": metrics,
    }


def _identity_slots(
    assignment: np.ndarray,
    labels: np.ndarray,
    identities: tuple[int, ...],
) -> dict[int, int]:
    slots = {}
    for identity in identities:
        mask = labels == int(identity)
        if not bool(np.any(mask)):
            continue
        mass = assignment[mask].sum(axis=0)
        slots[int(identity)] = int(np.argmax(mass))
    return slots


def _identity_embeddings(
    projection: Any,
    identity_slots: Mapping[int, int],
    identities: tuple[int, ...],
) -> dict[int, np.ndarray]:
    embeddings = {}
    for identity in identities:
        slot = int(identity_slots[int(identity)])
        state = projection.states[slot]
        embeddings[int(identity)] = np.concatenate(
            [
                np.asarray(state.feature, dtype=np.float32),
                np.asarray([state.confidence, state.mass_fraction], dtype=np.float32),
            ]
        )
    return embeddings


def _pairwise_identity_distances(
    embeddings0: Mapping[int, np.ndarray],
    embeddings1: Mapping[int, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    distance_rows = []
    retrieval_rows = []
    ids0 = sorted(embeddings0)
    ids1 = sorted(embeddings1)
    for left in ids0:
        distances = []
        for right in ids1:
            distance = float(np.linalg.norm(embeddings0[left] - embeddings1[right]))
            distances.append((right, distance))
            distance_rows.append(
                {
                    "frame0_identity": int(left),
                    "frame1_identity": int(right),
                    "distance": distance,
                    "same_identity": bool(left == right),
                }
            )
        nearest_identity, nearest_distance = min(distances, key=lambda item: item[1])
        positive = next(distance for right, distance in distances if right == left)
        negative = min((distance for right, distance in distances if right != left), default=positive)
        retrieval_rows.append(
            {
                "frame0_identity": int(left),
                "nearest_frame1_identity": int(nearest_identity),
                "nearest_distance": float(nearest_distance),
                "positive_distance": float(positive),
                "nearest_negative_distance": float(negative),
                "identity_margin": float(negative - positive),
                "correct": bool(nearest_identity == left),
            }
        )
    return distance_rows, retrieval_rows


def _identity_metrics(
    identity_slots0: Mapping[int, int],
    identity_slots1: Mapping[int, int],
    distance_rows: Sequence[Mapping[str, Any]],
    retrieval_rows: Sequence[Mapping[str, Any]],
    *,
    identities: tuple[int, ...],
) -> dict[str, Any]:
    correct = sum(1 for row in retrieval_rows if row["correct"])
    evaluated = len(retrieval_rows)
    margins = [float(row["identity_margin"]) for row in retrieval_rows]
    positives = [float(row["positive_distance"]) for row in retrieval_rows]
    swaps = sum(
        1
        for identity in identities
        if int(identity_slots0[int(identity)]) != int(identity_slots1[int(identity)])
    )
    consistency = _assignment_consistency(identity_slots0, identity_slots1, identities)
    return {
        "identity_retrieval_at_1": float(correct / evaluated) if evaluated else 0.0,
        "identity_retrieval_evaluated_count": int(evaluated),
        "identity_retrieval_correct_count": int(correct),
        "identity_margin": float(np.mean(margins)) if margins else 0.0,
        "slot_swap_rate": float(swaps / len(identities)) if identities else 0.0,
        "slot_swap_count": int(swaps),
        "objectstate_drift": float(np.mean(positives)) if positives else 0.0,
        "assignment_consistency": consistency,
        "occlusion_recovery": float(correct / evaluated) if evaluated else 0.0,
        "occlusion_recovery_check_count": int(evaluated),
        "pairwise_distance_count": int(len(distance_rows)),
    }


def _assignment_consistency(
    identity_slots0: Mapping[int, int],
    identity_slots1: Mapping[int, int],
    identities: tuple[int, ...],
) -> float:
    if not identities:
        return 0.0
    pairs0 = {
        (left, right): int(identity_slots0[left]) == int(identity_slots0[right])
        for index, left in enumerate(identities)
        for right in identities[index + 1 :]
    }
    if not pairs0:
        return 1.0
    preserved = 0
    for pair, same_slot0 in pairs0.items():
        left, right = pair
        same_slot1 = int(identity_slots1[left]) == int(identity_slots1[right])
        if same_slot0 == same_slot1:
            preserved += 1
    return float(preserved / len(pairs0))


def _gate_status(metrics: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    hard_gates = {
        "identity_retrieval_at_1_pass": float(metrics["identity_retrieval_at_1"])
        >= float(thresholds["identity_retrieval_at_1_min"]),
        "identity_margin_pass": float(metrics["identity_margin"])
        > float(thresholds["identity_margin_min"]),
        "assignment_consistency_pass": float(metrics["assignment_consistency"])
        >= float(thresholds["assignment_consistency_min"]),
        "objectstate_drift_pass": float(metrics["objectstate_drift"])
        <= float(thresholds["objectstate_drift_max"]),
        "occlusion_recovery_pass": int(metrics["occlusion_recovery_check_count"]) > 0
        and float(metrics["occlusion_recovery"]) >= float(thresholds["identity_retrieval_at_1_min"]),
    }
    blockers = [
        name.removesuffix("_pass")
        for name, passed in hard_gates.items()
        if not bool(passed)
    ]
    passed = all(hard_gates.values())
    return {
        "status": "objectstate_model_identity_gate_pass"
        if passed
        else "objectstate_model_identity_gate_fail",
        "gate_status": "pass" if passed else "fail",
        "hard_gates": hard_gates,
        "hard_blockers": blockers,
    }


def _baseline_comparison(baselines: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    candidate = baselines["assignment_solver_v2"]["metrics"]
    kmeans = baselines["xyz_centroid"]["metrics"]
    oracle = baselines["oracle_target_assignment"]["metrics"]
    random = baselines["random_assignment"]["metrics"]
    return {
        "candidate_retrieval_lift_vs_random": float(
            candidate["identity_retrieval_at_1"] - random["identity_retrieval_at_1"]
        ),
        "candidate_retrieval_lift_vs_xyz_centroid": float(
            candidate["identity_retrieval_at_1"] - kmeans["identity_retrieval_at_1"]
        ),
        "candidate_margin_lift_vs_xyz_centroid": float(
            candidate["identity_margin"] - kmeans["identity_margin"]
        ),
        "candidate_retrieval_gap_to_oracle": float(
            oracle["identity_retrieval_at_1"] - candidate["identity_retrieval_at_1"]
        ),
    }


def _write_artifacts(
    output_root: Path,
    candidate: Mapping[str, Any],
    frame0_cloud: GaussianCloud,
    frame1_cloud: GaussianCloud,
    ids0: np.ndarray,
    ids1: np.ndarray,
) -> dict[str, str]:
    matching_path = output_root / "identity-matching.json"
    retrieval_path = output_root / "objectstate-retrieval.json"
    distances_path = output_root / "identity-pairwise-distances.csv"
    assignment0_path = output_root / "assignment-t0.ply"
    assignment1_path = output_root / "assignment-t1.ply"
    matching_path.write_text(
        json.dumps(candidate["matching"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    retrieval_path.write_text(
        json.dumps(
            {
                "metrics": candidate["metrics"],
                "matching": candidate["matching"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with distances_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("frame0_identity", "frame1_identity", "distance", "same_identity"),
        )
        writer.writeheader()
        writer.writerows(candidate["pairwise_distances"])
    _write_assignment_ply(assignment0_path, frame0_cloud, ids0)
    _write_assignment_ply(assignment1_path, frame1_cloud, ids1)
    return {
        "identity_matching": str(matching_path),
        "objectstate_retrieval": str(retrieval_path),
        "identity_pairwise_distances": str(distances_path),
        "assignment_t0_ply": str(assignment0_path),
        "assignment_t1_ply": str(assignment1_path),
    }


def _write_assignment_ply(path: Path, cloud: GaussianCloud, derived_ids: np.ndarray) -> None:
    vertices = append_or_replace_property(
        cloud.vertices,
        "predicted_object_id",
        np.asarray(derived_ids, dtype=np.int32),
        "i4",
    )
    write_ply(
        path,
        GaussianCloud(vertices, comments=cloud.comments, source_format=cloud.source_format),
        fmt="ascii",
        comments=("ObjGauss model identity gate assignment",),
    )


def _random_assignment(count: int, slots: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, slots, size=count)
    return _one_hot(labels, slots)


def _xyz_centroid_assignment(cloud: GaussianCloud, slots: int) -> np.ndarray:
    xyz = positions(cloud)
    labels = _kmeans_labels(xyz, slots=slots, iterations=12)
    return _one_hot(labels, slots)


def _oracle_assignment(labels: np.ndarray, identities: tuple[int, ...]) -> np.ndarray:
    index = {identity: offset for offset, identity in enumerate(identities)}
    mapped = np.asarray([index[int(label)] for label in labels], dtype=np.int64)
    return _one_hot(mapped, len(identities))


def _one_hot(labels: np.ndarray, slots: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    matrix = np.zeros((labels.shape[0], slots), dtype=np.float32)
    matrix[np.arange(labels.shape[0]), labels] = 1.0
    return matrix


def _kmeans_labels(values: np.ndarray, *, slots: int, iterations: int) -> np.ndarray:
    if values.shape[0] < slots:
        raise ValueError("xyz centroid baseline requires at least K points")
    order = np.argsort(values[:, 0])
    centers = values[order[np.linspace(0, values.shape[0] - 1, slots).round().astype(int)]].copy()
    labels = np.zeros(values.shape[0], dtype=np.int64)
    for _ in range(iterations):
        distances = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1).astype(np.int64)
        for slot in range(slots):
            mask = labels == slot
            if np.any(mask):
                centers[slot] = values[mask].mean(axis=0)
    return labels


def _features(
    cloud: GaussianCloud,
    features: np.ndarray | None,
    expected_dim: int,
    label: str,
) -> np.ndarray:
    array = extract_features(cloud) if features is None else np.asarray(features, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D feature matrix")
    if array.shape[0] != cloud.count:
        raise ValueError(f"{label} rows must match Gaussian count")
    if array.shape[1] != expected_dim:
        raise ValueError(f"{label} feature_dim must match solver state")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array.astype(np.float32, copy=False)


def _identity_labels(value: np.ndarray, count: int, label: str) -> np.ndarray:
    labels = np.asarray(value, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError(f"{label} must be a 1D array")
    if labels.shape[0] != count:
        raise ValueError(f"{label} length must match Gaussian count")
    if labels.size and int(labels.min()) < 0:
        raise ValueError(f"{label} must contain non-negative physical identity ids")
    return labels


def _common_identities(labels0: np.ndarray, labels1: np.ndarray) -> tuple[int, ...]:
    return tuple(
        sorted(set(int(value) for value in np.unique(labels0)) & set(int(value) for value in np.unique(labels1)))
    )


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"model identity gate requires {key}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number
