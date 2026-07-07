from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.v2_stability_diagnostics import (
    IdentitySlotObservation,
    diagnose_synthetic_stability_fixture,
)
from objgauss.core.v2_stability_foundation import (
    SyntheticStabilityScenarioFixture,
    make_synthetic_stability_scenario_suite,
    validate_synthetic_stability_scenario_fixture,
)

OBJECTSTATE_IDENTITY_GATE_SCHEMA = "objgauss-objectstate-identity-gate-v1"
OBJECTSTATE_IDENTITY_DATASET_SCHEMA = "objgauss-objectstate-identity-dataset-v1"
_GATE_STATUS_PASS = "objectstate_identity_gate_pass"
_GATE_STATUS_FAIL = "objectstate_identity_gate_fail"


@dataclass(frozen=True)
class ObjectStateIdentityGateThresholds:
    id_accuracy_min: float = 0.95
    idf1_min: float = 0.95
    embedding_retrieval_recall_at_1_min: float = 0.95
    long_term_drift_rate_max: float = 0.02
    fragmentation_rate_max: float = 0.02
    occlusion_recovery_rate_min: float = 0.95
    contrastive_margin_min: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return validate_objectstate_identity_gate_thresholds(self)


@dataclass(frozen=True)
class ObjectStateIdentityRow:
    row_id: str
    scenario_id: str
    scenario_kind: str
    frame_index: int
    view_id: str
    oracle_object_id: int
    lineage_id: str
    expected_identity: int
    predicted_identity: int
    confidence: float
    support_count: int
    pose_center: np.ndarray
    appearance_embedding: np.ndarray
    geometry_embedding: np.ndarray
    objectstate_embedding: np.ndarray
    transformation: str

    @property
    def matches_expected(self) -> bool:
        return int(self.expected_identity) == int(self.predicted_identity)

    @property
    def identity_key(self) -> tuple[str, int]:
        return (self.scenario_id, int(self.oracle_object_id))

    def as_dict(self) -> dict[str, Any]:
        row = validate_objectstate_identity_row(self)
        return {
            "row_id": row.row_id,
            "object_instance": {
                "oracle_object_id": int(row.oracle_object_id),
                "lineage_id": row.lineage_id,
            },
            "observation": {
                "scenario_id": row.scenario_id,
                "scenario_kind": row.scenario_kind,
                "frame_index": int(row.frame_index),
                "view_id": row.view_id,
                "support_count": int(row.support_count),
            },
            "transformation": {
                "kind": row.transformation,
            },
            "ground_truth_identity": {
                "expected_identity": int(row.expected_identity),
            },
            "candidate_objectstate": {
                "predicted_identity": int(row.predicted_identity),
                "confidence": float(row.confidence),
                "pose": np.round(row.pose_center, 6).tolist(),
                "geometry_embedding": np.round(row.geometry_embedding, 6).tolist(),
                "appearance_embedding": np.round(row.appearance_embedding, 6).tolist(),
                "identity_embedding": np.round(row.objectstate_embedding, 6).tolist(),
            },
            "matches_expected": bool(row.matches_expected),
        }


@dataclass(frozen=True)
class ObjectStateIdentityGateReport:
    rows: tuple[ObjectStateIdentityRow, ...]
    fixture_summaries: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    hard_gates: dict[str, bool]
    hard_blockers: tuple[str, ...]
    thresholds: ObjectStateIdentityGateThresholds
    embedding_source: str
    schema: str = OBJECTSTATE_IDENTITY_GATE_SCHEMA

    @property
    def passed(self) -> bool:
        return all(bool(value) for value in self.hard_gates.values())

    def as_dict(self) -> dict[str, Any]:
        summary = {
            "schema": self.schema,
            "kind": "objectstate_identity_gate",
            "status": _GATE_STATUS_PASS if self.passed else _GATE_STATUS_FAIL,
            "gate_role": "objectstate_identity_state_variable_smoke_gate",
            "dataset": {
                "schema": OBJECTSTATE_IDENTITY_DATASET_SCHEMA,
                "kind": "objectstate_identity_dataset",
                "row_count": len(self.rows),
                "fixture_count": len(self.fixture_summaries),
                "contract": {
                    "inputs": [
                        "ObjectInstance",
                        "Observation",
                        "Transformation",
                        "GroundTruthIdentity",
                    ],
                    "candidate_outputs": [
                        "object_id",
                        "geometry_embedding",
                        "appearance_embedding",
                        "pose",
                        "uncertainty",
                    ],
                },
                "rows": [row.as_dict() for row in self.rows],
            },
            "fixture_summaries": list(self.fixture_summaries),
            "embedding_source": self.embedding_source,
            "thresholds": self.thresholds.as_dict(),
            "metrics": self.metrics,
            "hard_gates": {key: bool(value) for key, value in self.hard_gates.items()},
            "hard_blockers": list(self.hard_blockers),
            "claim_policy": {
                "does_not_claim_world_state": True,
                "tests_identity_state_only": True,
                "predictive_sufficiency_required_later": True,
                "counterfactual_required_later": True,
            },
            "non_goals": {
                "trains_encoder": False,
                "uses_renderer_loss": False,
                "uses_diffusion": False,
                "uses_replay_buffer": False,
                "mutates_viewer_defaults": False,
            },
        }
        return validate_objectstate_identity_gate_summary(summary)


def evaluate_objectstate_identity_gate(
    fixtures: Sequence[SyntheticStabilityScenarioFixture] | None = None,
    *,
    predicted_slots_by_fixture: Sequence[Sequence[np.ndarray | Sequence[int]] | None] | None = None,
    predicted_assignments_by_fixture: Sequence[Sequence[np.ndarray] | None] | None = None,
    thresholds: ObjectStateIdentityGateThresholds | None = None,
) -> ObjectStateIdentityGateReport:
    resolved_fixtures = tuple(
        make_synthetic_stability_scenario_suite()
        if fixtures is None
        else fixtures
    )
    if not resolved_fixtures:
        raise ValueError("fixtures must contain at least one scenario")
    if predicted_slots_by_fixture is not None and predicted_assignments_by_fixture is not None:
        raise ValueError("provide predicted_slots_by_fixture or predicted_assignments_by_fixture, not both")
    if predicted_slots_by_fixture is None and predicted_assignments_by_fixture is None:
        raise ValueError(
            "objectstate identity gate requires explicit predicted_slots_by_fixture "
            "or predicted_assignments_by_fixture"
        )
    checked_thresholds = thresholds or ObjectStateIdentityGateThresholds()
    checked_thresholds.as_dict()
    slot_predictions = _prediction_sequence(predicted_slots_by_fixture, len(resolved_fixtures))
    assignment_predictions = _prediction_sequence(predicted_assignments_by_fixture, len(resolved_fixtures))
    rows: list[ObjectStateIdentityRow] = []
    fixture_summaries: list[dict[str, Any]] = []
    embedding_source = "predicted_assignments" if predicted_assignments_by_fixture is not None else "predicted_slots_one_hot"
    for index, fixture in enumerate(resolved_fixtures):
        checked_fixture = validate_synthetic_stability_scenario_fixture(fixture)
        diagnostics = diagnose_synthetic_stability_fixture(
            checked_fixture,
            predicted_slots=slot_predictions[index],
            predicted_assignments=assignment_predictions[index],
        )
        frame_embeddings = _candidate_frame_embeddings(
            checked_fixture,
            predicted_slots=slot_predictions[index],
            predicted_assignments=assignment_predictions[index],
        )
        fixture_rows = _rows_from_fixture(
            checked_fixture,
            diagnostics.identity_observations,
            frame_embeddings,
        )
        rows.extend(fixture_rows)
        fixture_summaries.append(
            {
                "scenario_id": checked_fixture.scenario_id,
                "scenario_kind": checked_fixture.scenario_kind,
                "object_count": checked_fixture.object_count,
                "frame_count": checked_fixture.frame_count,
                "row_count": len(fixture_rows),
                "failure_mode_counts": diagnostics.as_dict()["failure_mode_counts"],
            }
        )
    normalized_rows = _pad_row_embeddings(tuple(rows))
    metrics = _identity_gate_metrics(normalized_rows, resolved_fixtures)
    hard_gates, hard_blockers = _hard_gate_result(metrics, checked_thresholds)
    return ObjectStateIdentityGateReport(
        rows=normalized_rows,
        fixture_summaries=tuple(fixture_summaries),
        metrics=metrics,
        hard_gates=hard_gates,
        hard_blockers=hard_blockers,
        thresholds=checked_thresholds,
        embedding_source=embedding_source,
    )


def validate_objectstate_identity_gate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("objectstate identity gate summary must be a dict")
    if payload.get("schema") != OBJECTSTATE_IDENTITY_GATE_SCHEMA:
        raise ValueError(f"unsupported objectstate identity gate schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_identity_gate":
        raise ValueError("objectstate identity gate kind must be objectstate_identity_gate")
    if payload.get("status") not in {_GATE_STATUS_PASS, _GATE_STATUS_FAIL}:
        raise ValueError("objectstate identity gate status is unsupported")
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("schema") != OBJECTSTATE_IDENTITY_DATASET_SCHEMA:
        raise ValueError("objectstate identity gate dataset schema is unsupported")
    rows = dataset.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("objectstate identity dataset must contain rows")
    if dataset.get("row_count") != len(rows):
        raise ValueError("objectstate identity dataset row_count must match rows")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("objectstate identity gate missing metrics")
    for key in (
        "id_accuracy",
        "idf1",
        "embedding_retrieval_recall_at_1",
        "long_term_drift_rate",
        "fragmentation_rate",
        "occlusion_recovery_rate",
        "contrastive_margin",
    ):
        if key not in metrics:
            raise ValueError(f"objectstate identity metrics missing {key}")
    hard_gates = payload.get("hard_gates")
    if not isinstance(hard_gates, dict) or not hard_gates:
        raise ValueError("objectstate identity gate missing hard_gates")
    for value in hard_gates.values():
        if not isinstance(value, bool):
            raise ValueError("objectstate identity hard gate values must be bool")
    expected_status = _GATE_STATUS_PASS if all(hard_gates.values()) else _GATE_STATUS_FAIL
    if payload["status"] != expected_status:
        raise ValueError("objectstate identity gate status must match hard gates")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("trains_encoder")
        or non_goals.get("uses_renderer_loss")
        or non_goals.get("uses_diffusion")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError("objectstate identity gate cannot train or mutate runtime policy")
    return payload


def validate_objectstate_identity_gate_thresholds(
    thresholds: ObjectStateIdentityGateThresholds,
) -> dict[str, float]:
    if not isinstance(thresholds, ObjectStateIdentityGateThresholds):
        raise TypeError("thresholds must be ObjectStateIdentityGateThresholds")
    payload = {
        "id_accuracy_min": float(thresholds.id_accuracy_min),
        "idf1_min": float(thresholds.idf1_min),
        "embedding_retrieval_recall_at_1_min": float(
            thresholds.embedding_retrieval_recall_at_1_min
        ),
        "long_term_drift_rate_max": float(thresholds.long_term_drift_rate_max),
        "fragmentation_rate_max": float(thresholds.fragmentation_rate_max),
        "occlusion_recovery_rate_min": float(thresholds.occlusion_recovery_rate_min),
        "contrastive_margin_min": float(thresholds.contrastive_margin_min),
    }
    for key, value in payload.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{key} must be in [0, 1]")
    return payload


def validate_objectstate_identity_row(row: ObjectStateIdentityRow) -> ObjectStateIdentityRow:
    if not isinstance(row, ObjectStateIdentityRow):
        raise TypeError("row must be ObjectStateIdentityRow")
    if not row.row_id:
        raise ValueError("row_id must be non-empty")
    if not row.scenario_id:
        raise ValueError("scenario_id must be non-empty")
    if int(row.support_count) < 1:
        raise ValueError("support_count must be >= 1")
    pose = _float_vector(row.pose_center, "pose_center")
    appearance = _float_vector(row.appearance_embedding, "appearance_embedding")
    geometry = _float_vector(row.geometry_embedding, "geometry_embedding")
    embedding = _float_vector(row.objectstate_embedding, "objectstate_embedding")
    if pose.shape[0] != 3:
        raise ValueError("pose_center must have 3 values")
    if geometry.shape[0] != 3:
        raise ValueError("geometry_embedding must have 3 values")
    return ObjectStateIdentityRow(
        row_id=str(row.row_id),
        scenario_id=str(row.scenario_id),
        scenario_kind=str(row.scenario_kind),
        frame_index=int(row.frame_index),
        view_id=str(row.view_id),
        oracle_object_id=int(row.oracle_object_id),
        lineage_id=str(row.lineage_id),
        expected_identity=int(row.expected_identity),
        predicted_identity=int(row.predicted_identity),
        confidence=float(row.confidence),
        support_count=int(row.support_count),
        pose_center=pose,
        appearance_embedding=appearance,
        geometry_embedding=geometry,
        objectstate_embedding=embedding,
        transformation=str(row.transformation),
    )


def _rows_from_fixture(
    fixture: SyntheticStabilityScenarioFixture,
    observations: Sequence[IdentitySlotObservation],
    frame_embeddings: Sequence[np.ndarray],
) -> tuple[ObjectStateIdentityRow, ...]:
    by_frame = {int(frame.frame_index): frame for frame in fixture.observations}
    rows = []
    for observation in observations:
        frame = by_frame[int(observation.frame_index)]
        mask = frame.oracle_object_ids.astype(int) == int(observation.oracle_object_id)
        if not bool(np.any(mask)):
            raise ValueError("identity observation must have visible support in source frame")
        embeddings = frame_embeddings[int(observation.frame_index)]
        row_id = (
            f"{fixture.scenario_id}:frame-{int(observation.frame_index):04d}:"
            f"object-{int(observation.oracle_object_id):04d}"
        )
        rows.append(
            validate_objectstate_identity_row(
                ObjectStateIdentityRow(
                    row_id=row_id,
                    scenario_id=fixture.scenario_id,
                    scenario_kind=fixture.scenario_kind,
                    frame_index=int(observation.frame_index),
                    view_id=frame.view_id,
                    oracle_object_id=int(observation.oracle_object_id),
                    lineage_id=observation.lineage_id,
                    expected_identity=int(observation.expected_slot),
                    predicted_identity=int(observation.predicted_slot),
                    confidence=float(observation.mean_confidence),
                    support_count=int(observation.evidence_count),
                    pose_center=np.mean(frame.evidence.positions[mask], axis=0),
                    appearance_embedding=np.mean(frame.evidence.features[mask], axis=0),
                    geometry_embedding=np.mean(frame.evidence.positions[mask], axis=0),
                    objectstate_embedding=np.mean(embeddings[mask], axis=0),
                    transformation=_transformation_label(fixture, frame.frame_index),
                )
            )
        )
    return tuple(rows)


def _candidate_frame_embeddings(
    fixture: SyntheticStabilityScenarioFixture,
    *,
    predicted_slots: Sequence[np.ndarray | Sequence[int]] | None,
    predicted_assignments: Sequence[np.ndarray] | None,
) -> tuple[np.ndarray, ...]:
    if predicted_slots is not None and predicted_assignments is not None:
        raise ValueError("provide predicted_slots or predicted_assignments, not both")
    if predicted_slots is None and predicted_assignments is None:
        raise ValueError("candidate frame embeddings require explicit predictions")
    if predicted_assignments is not None:
        if len(predicted_assignments) != len(fixture.observations):
            raise ValueError("predicted_assignments must cover every observation frame")
        matrices = []
        for frame_index, (frame, assignment) in enumerate(zip(fixture.observations, predicted_assignments)):
            matrix = _assignment_array(assignment, f"predicted_assignments[{frame_index}]")
            if matrix.shape[0] != frame.evidence.evidence_count:
                raise ValueError("predicted assignment rows must match observation evidence rows")
            if matrix.shape[1] != fixture.world.oracle.slots:
                raise ValueError("predicted assignment columns must match fixture slot count")
            matrices.append(matrix)
        return tuple(matrices)
    if predicted_slots is None:
        raise ValueError("predicted_slots must be provided")
    if len(predicted_slots) != len(fixture.observations):
        raise ValueError("predicted_slots must cover every observation frame")
    slot_arrays = []
    max_slot = fixture.world.oracle.slots - 1
    for frame_index, (frame, slots) in enumerate(zip(fixture.observations, predicted_slots)):
        slot_array = np.asarray(slots, dtype=np.int64)
        if slot_array.ndim != 1:
            raise ValueError(f"predicted_slots[{frame_index}] must be a 1D array")
        if slot_array.shape[0] != frame.evidence.evidence_count:
            raise ValueError("predicted slot rows must match observation evidence rows")
        if slot_array.size:
            max_slot = max(max_slot, int(np.max(slot_array)))
        slot_arrays.append(slot_array)
    dim = max(1, max_slot + 1)
    embeddings = []
    for slot_array in slot_arrays:
        matrix = np.zeros((slot_array.shape[0], dim), dtype=np.float32)
        valid = slot_array >= 0
        if np.any(valid):
            matrix[np.arange(slot_array.shape[0])[valid], slot_array[valid]] = 1.0
        embeddings.append(matrix)
    return tuple(embeddings)


def _identity_gate_metrics(
    rows: tuple[ObjectStateIdentityRow, ...],
    fixtures: Sequence[SyntheticStabilityScenarioFixture],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("objectstate identity gate requires at least one row")
    total = len(rows)
    correct = sum(1 for row in rows if row.matches_expected)
    wrong = total - correct
    id_accuracy = float(correct / total)
    idf1 = float((2 * correct) / ((2 * correct) + wrong + wrong)) if total else 0.0
    fragmentation = _fragmentation_summary(rows)
    drift = _long_term_drift_summary(rows)
    retrieval = _embedding_retrieval_summary(rows)
    distances = _embedding_distance_summary(rows)
    occlusion = _occlusion_recovery_summary(rows, fixtures)
    return {
        "row_count": int(total),
        "correct_identity_count": int(correct),
        "wrong_identity_count": int(wrong),
        "id_accuracy": id_accuracy,
        "idf1": idf1,
        "idtp": int(correct),
        "idfp": int(wrong),
        "idfn": int(wrong),
        "embedding_retrieval_recall_at_1": retrieval["recall_at_1"],
        "embedding_retrieval_evaluated_count": retrieval["evaluated_count"],
        "embedding_retrieval_correct_count": retrieval["correct_count"],
        "same_object_distance_mean": distances["same_object_distance_mean"],
        "different_object_distance_mean": distances["different_object_distance_mean"],
        "contrastive_margin": distances["contrastive_margin"],
        "long_term_drift_rate": drift["rate"],
        "long_term_drift_transition_count": drift["transition_count"],
        "long_term_drift_count": drift["drift_count"],
        "fragmentation_rate": fragmentation["rate"],
        "fragmented_identity_count": fragmentation["fragmented_identity_count"],
        "identity_count": fragmentation["identity_count"],
        "occlusion_recovery_rate": occlusion["rate"],
        "occlusion_recovery_check_count": occlusion["check_count"],
        "occlusion_recovery_pass_count": occlusion["pass_count"],
        "occlusion_recovery_checks": occlusion["checks"],
    }


def _hard_gate_result(
    metrics: dict[str, Any],
    thresholds: ObjectStateIdentityGateThresholds,
) -> tuple[dict[str, bool], tuple[str, ...]]:
    hard_gates = {
        "id_accuracy_pass": float(metrics["id_accuracy"]) >= float(thresholds.id_accuracy_min),
        "idf1_pass": float(metrics["idf1"]) >= float(thresholds.idf1_min),
        "embedding_retrieval_recall_at_1_pass": (
            int(metrics["embedding_retrieval_evaluated_count"]) > 0
            and float(metrics["embedding_retrieval_recall_at_1"])
            >= float(thresholds.embedding_retrieval_recall_at_1_min)
        ),
        "long_term_drift_rate_pass": float(metrics["long_term_drift_rate"])
        <= float(thresholds.long_term_drift_rate_max),
        "fragmentation_rate_pass": float(metrics["fragmentation_rate"])
        <= float(thresholds.fragmentation_rate_max),
        "occlusion_recovery_rate_pass": (
            int(metrics["occlusion_recovery_check_count"]) == 0
            or float(metrics["occlusion_recovery_rate"]) >= float(thresholds.occlusion_recovery_rate_min)
        ),
        "contrastive_margin_positive_pass": float(metrics["contrastive_margin"])
        > float(thresholds.contrastive_margin_min),
    }
    blockers = []
    blocker_by_gate = {
        "id_accuracy_pass": "id_accuracy_below_threshold",
        "idf1_pass": "idf1_below_threshold",
        "embedding_retrieval_recall_at_1_pass": "embedding_retrieval_below_threshold",
        "long_term_drift_rate_pass": "long_term_drift_above_threshold",
        "fragmentation_rate_pass": "fragmentation_above_threshold",
        "occlusion_recovery_rate_pass": "occlusion_recovery_below_threshold",
        "contrastive_margin_positive_pass": "contrastive_margin_not_positive",
    }
    for gate, passed in hard_gates.items():
        if not passed:
            blockers.append(blocker_by_gate[gate])
    if int(metrics["wrong_identity_count"]) > 0:
        blockers.append("expected_identity_mismatch")
    return hard_gates, tuple(sorted(set(blockers)))


def _fragmentation_summary(rows: Sequence[ObjectStateIdentityRow]) -> dict[str, Any]:
    by_identity = _rows_by_identity(rows)
    fragmented = 0
    for identity_rows in by_identity.values():
        predictions = {int(row.predicted_identity) for row in identity_rows}
        if len(predictions) > 1:
            fragmented += 1
    identity_count = len(by_identity)
    return {
        "fragmented_identity_count": int(fragmented),
        "identity_count": int(identity_count),
        "rate": float(fragmented / identity_count) if identity_count else 0.0,
    }


def _long_term_drift_summary(rows: Sequence[ObjectStateIdentityRow]) -> dict[str, Any]:
    transitions = 0
    drifts = 0
    for identity_rows in _rows_by_identity(rows).values():
        ordered = sorted(identity_rows, key=lambda row: int(row.frame_index))
        for previous, current in zip(ordered, ordered[1:]):
            transitions += 1
            if int(previous.predicted_identity) != int(current.predicted_identity):
                drifts += 1
    return {
        "transition_count": int(transitions),
        "drift_count": int(drifts),
        "rate": float(drifts / transitions) if transitions else 0.0,
    }


def _embedding_retrieval_summary(rows: Sequence[ObjectStateIdentityRow]) -> dict[str, Any]:
    evaluated = 0
    correct = 0
    by_scenario = _rows_by_scenario(rows)
    for scenario_rows in by_scenario.values():
        embeddings = np.vstack([row.objectstate_embedding for row in scenario_rows])
        for index, row in enumerate(scenario_rows):
            positive_exists = any(
                other_index != index and other.identity_key == row.identity_key
                for other_index, other in enumerate(scenario_rows)
            )
            if not positive_exists:
                continue
            distances = np.linalg.norm(embeddings - embeddings[index], axis=1)
            distances[index] = np.inf
            nearest_index = int(np.argmin(distances))
            evaluated += 1
            if scenario_rows[nearest_index].identity_key == row.identity_key:
                correct += 1
    return {
        "evaluated_count": int(evaluated),
        "correct_count": int(correct),
        "recall_at_1": float(correct / evaluated) if evaluated else 0.0,
    }


def _embedding_distance_summary(rows: Sequence[ObjectStateIdentityRow]) -> dict[str, Any]:
    same = []
    different = []
    for scenario_rows in _rows_by_scenario(rows).values():
        for left_index, left in enumerate(scenario_rows):
            for right in scenario_rows[left_index + 1 :]:
                distance = float(np.linalg.norm(left.objectstate_embedding - right.objectstate_embedding))
                if left.identity_key == right.identity_key:
                    same.append(distance)
                else:
                    different.append(distance)
    same_mean = float(np.mean(same)) if same else 0.0
    different_mean = float(np.mean(different)) if different else 0.0
    return {
        "same_object_distance_mean": same_mean,
        "different_object_distance_mean": different_mean,
        "contrastive_margin": float(different_mean - same_mean),
    }


def _occlusion_recovery_summary(
    rows: Sequence[ObjectStateIdentityRow],
    fixtures: Sequence[SyntheticStabilityScenarioFixture],
) -> dict[str, Any]:
    row_by_key = {
        (row.scenario_id, int(row.frame_index), int(row.oracle_object_id)): row
        for row in rows
    }
    checks = []
    for fixture in fixtures:
        checked_fixture = validate_synthetic_stability_scenario_fixture(fixture)
        for frame_index in range(1, checked_fixture.world.oracle.frame_count):
            previous_oracle = {
                int(item.oracle_object_id): item
                for item in checked_fixture.world.oracle.frames[frame_index - 1]
            }
            current_oracle = checked_fixture.world.oracle.frames[frame_index]
            for current in current_oracle:
                previous = previous_oracle[int(current.oracle_object_id)]
                if previous.visible or not current.visible:
                    continue
                previous_visible_row = _latest_visible_row_before(
                    rows,
                    scenario_id=checked_fixture.scenario_id,
                    object_id=int(current.oracle_object_id),
                    frame_index=frame_index,
                )
                current_row = row_by_key.get(
                    (checked_fixture.scenario_id, frame_index, int(current.oracle_object_id))
                )
                passed = (
                    previous_visible_row is not None
                    and current_row is not None
                    and previous_visible_row.matches_expected
                    and current_row.matches_expected
                    and int(previous_visible_row.predicted_identity)
                    == int(current_row.predicted_identity)
                )
                checks.append(
                    {
                        "scenario_id": checked_fixture.scenario_id,
                        "frame_index": int(frame_index),
                        "oracle_object_id": int(current.oracle_object_id),
                        "predicted_before": None
                        if previous_visible_row is None
                        else int(previous_visible_row.predicted_identity),
                        "predicted_after": None
                        if current_row is None
                        else int(current_row.predicted_identity),
                        "expected_identity": int(current.expected_slot),
                        "pass": bool(passed),
                    }
                )
    pass_count = sum(1 for check in checks if check["pass"])
    return {
        "check_count": len(checks),
        "pass_count": int(pass_count),
        "rate": float(pass_count / len(checks)) if checks else 1.0,
        "checks": checks,
    }


def _latest_visible_row_before(
    rows: Sequence[ObjectStateIdentityRow],
    *,
    scenario_id: str,
    object_id: int,
    frame_index: int,
) -> ObjectStateIdentityRow | None:
    candidates = [
        row
        for row in rows
        if row.scenario_id == scenario_id
        and int(row.oracle_object_id) == int(object_id)
        and int(row.frame_index) < int(frame_index)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: int(row.frame_index))[-1]


def _pad_row_embeddings(rows: tuple[ObjectStateIdentityRow, ...]) -> tuple[ObjectStateIdentityRow, ...]:
    if not rows:
        return rows
    dim = max(int(row.objectstate_embedding.shape[0]) for row in rows)
    padded = []
    for row in rows:
        embedding = row.objectstate_embedding
        if embedding.shape[0] < dim:
            embedding = np.pad(embedding, (0, dim - embedding.shape[0]), mode="constant")
        padded.append(
            ObjectStateIdentityRow(
                row_id=row.row_id,
                scenario_id=row.scenario_id,
                scenario_kind=row.scenario_kind,
                frame_index=row.frame_index,
                view_id=row.view_id,
                oracle_object_id=row.oracle_object_id,
                lineage_id=row.lineage_id,
                expected_identity=row.expected_identity,
                predicted_identity=row.predicted_identity,
                confidence=row.confidence,
                support_count=row.support_count,
                pose_center=row.pose_center,
                appearance_embedding=row.appearance_embedding,
                geometry_embedding=row.geometry_embedding,
                objectstate_embedding=embedding.astype(np.float32, copy=False),
                transformation=row.transformation,
            )
        )
    return tuple(validate_objectstate_identity_row(row) for row in padded)


def _rows_by_identity(
    rows: Sequence[ObjectStateIdentityRow],
) -> dict[tuple[str, int], tuple[ObjectStateIdentityRow, ...]]:
    grouped: dict[tuple[str, int], list[ObjectStateIdentityRow]] = {}
    for row in rows:
        grouped.setdefault(row.identity_key, []).append(row)
    return {key: tuple(value) for key, value in sorted(grouped.items())}


def _rows_by_scenario(
    rows: Sequence[ObjectStateIdentityRow],
) -> dict[str, tuple[ObjectStateIdentityRow, ...]]:
    grouped: dict[str, list[ObjectStateIdentityRow]] = {}
    for row in rows:
        grouped.setdefault(row.scenario_id, []).append(row)
    return {key: tuple(value) for key, value in sorted(grouped.items())}


def _prediction_sequence(value: Sequence[Any] | None, expected_length: int) -> tuple[Any, ...]:
    if value is None:
        return tuple(None for _ in range(expected_length))
    if len(value) != expected_length:
        raise ValueError("prediction sequence must cover every fixture")
    return tuple(value)


def _transformation_label(fixture: SyntheticStabilityScenarioFixture, frame_index: int) -> str:
    frame = fixture.world.frames[int(frame_index)]
    perturbation = frame.perturbation.get("kind", "none")
    return f"{fixture.scenario_kind}:{frame.view_id}:{perturbation}"


def _assignment_array(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{label} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite values")
    if np.any(array < 0.0):
        raise ValueError(f"{label} must be non-negative")
    row_sums = np.sum(array, axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError(f"{label} rows must have positive mass")
    return (array / row_sums[:, None]).astype(np.float32, copy=False)


def _float_vector(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a 1D array")
    if array.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one value")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite values")
    return array.astype(np.float32, copy=False)
