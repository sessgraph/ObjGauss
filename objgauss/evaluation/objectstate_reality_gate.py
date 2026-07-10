from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

OBJECTSTATE_REALITY_GATE_SCHEMA = "objgauss-objectstate-reality-gate-v1"
OBJECTSTATE_REALITY_ROW_SCHEMA = "objgauss-objectstate-real-public-row-v1"
OBJECTSTATE_REALITY_EVIDENCE_KINDS = ("identity", "prediction", "intervention")
OBJECTSTATE_REALITY_SOURCE_KINDS = (
    "controlled_real",
    "public_replay",
    "open_world_real",
)
OBJECTSTATE_REALITY_ROW_STATUSES = ("pass", "fail", "blocked")
_GATE_STATUS_PASS = "objectstate_reality_gate_pass"
_GATE_STATUS_FAIL = "objectstate_reality_gate_fail"
_REAL_OR_PUBLIC_SOURCE_KINDS = {"controlled_real", "public_replay"}
_REQUIRED_METRICS_BY_EVIDENCE = {
    "identity": (
        "idf1",
        "fragmentation_rate",
        "swap_rate",
        "identity_collapse",
    ),
    "prediction": ("state_ade", "history_ade"),
    "intervention": (
        "action_conditioned_ade",
        "no_action_ade",
        "counterfactual_outcome_accuracy",
        "wrong_direction_rate",
    ),
}


@dataclass(frozen=True)
class ObjectStateRealityGateThresholds:
    min_real_or_public_rows: int = 1
    require_identity_pass_row: bool = True
    require_prediction_pass_row: bool = True
    require_intervention_pass_row: bool = True
    fail_on_failed_rows: bool = True
    min_identity_idf1: float = 0.95
    max_identity_fragmentation_rate: float = 0.05
    max_identity_swap_rate: float = 0.0
    require_no_identity_collapse: bool = True
    max_prediction_state_ade: float = 0.05
    min_prediction_improvement_margin: float = 0.0
    max_intervention_action_conditioned_ade: float = 0.05
    min_intervention_counterfactual_outcome_accuracy: float = 0.95
    max_intervention_wrong_direction_rate: float = 0.0
    min_intervention_gain: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return validate_objectstate_reality_gate_thresholds(self)


@dataclass(frozen=True)
class ObjectStateRealityRow:
    row_id: str
    sample_id: str
    source_kind: str
    evidence_kind: str
    status: str
    object_category: str
    scenario: str
    observation_modalities: tuple[str, ...]
    artifact_refs: tuple[str, ...]
    metrics: Mapping[str, Any]
    has_identity_gt: bool
    has_pose_gt: bool
    has_action_gt: bool
    has_timestamp: bool
    license: str = "unknown"
    block_reason: str | None = None
    failure_reason: str | None = None
    schema: str = OBJECTSTATE_REALITY_ROW_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        row = validate_objectstate_reality_row(self)
        return {
            "schema": row.schema,
            "row_id": row.row_id,
            "sample_id": row.sample_id,
            "source_kind": row.source_kind,
            "evidence_kind": row.evidence_kind,
            "status": row.status,
            "object_category": row.object_category,
            "scenario": row.scenario,
            "observation_modalities": list(row.observation_modalities),
            "artifact_refs": list(row.artifact_refs),
            "license": row.license,
            "ground_truth": {
                "identity": bool(row.has_identity_gt),
                "pose": bool(row.has_pose_gt),
                "action": bool(row.has_action_gt),
                "timestamp": bool(row.has_timestamp),
            },
            "metrics": dict(row.metrics),
            "block_reason": row.block_reason,
            "failure_reason": row.failure_reason,
        }


@dataclass(frozen=True)
class ObjectStateRealityGateReport:
    rows: tuple[ObjectStateRealityRow, ...]
    synthetic_smoke_passed: bool
    thresholds: ObjectStateRealityGateThresholds
    metrics: dict[str, Any]
    hard_gates: dict[str, bool]
    hard_blockers: tuple[str, ...]
    declaration_diagnostics: dict[str, Any]
    schema: str = OBJECTSTATE_REALITY_GATE_SCHEMA

    @property
    def passed(self) -> bool:
        return all(bool(value) for value in self.hard_gates.values())

    @property
    def pass_rows(self) -> tuple[ObjectStateRealityRow, ...]:
        return tuple(row for row in self.rows if row.status == "pass")

    @property
    def fail_rows(self) -> tuple[ObjectStateRealityRow, ...]:
        return tuple(row for row in self.rows if row.status == "fail")

    @property
    def blocked_rows(self) -> tuple[ObjectStateRealityRow, ...]:
        return tuple(row for row in self.rows if row.status == "blocked")

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "kind": "objectstate_reality_gate",
            "status": _GATE_STATUS_PASS if self.passed else _GATE_STATUS_FAIL,
            "gate_role": "controlled_real_public_state_variable_candidate_gate",
            "row_schema": OBJECTSTATE_REALITY_ROW_SCHEMA,
            "synthetic_smoke_passed": bool(self.synthetic_smoke_passed),
            "thresholds": self.thresholds.as_dict(),
            "row_count": len(self.rows),
            "pass_row_count": len(self.pass_rows),
            "fail_row_count": len(self.fail_rows),
            "blocked_row_count": len(self.blocked_rows),
            "metrics": dict(self.metrics),
            "declaration_diagnostics": dict(self.declaration_diagnostics),
            "hard_gates": {key: bool(value) for key, value in self.hard_gates.items()},
            "hard_blockers": list(self.hard_blockers),
            "rows": [row.as_dict() for row in self.rows],
            "pass_rows": [row.as_dict() for row in self.pass_rows],
            "fail_rows": [row.as_dict() for row in self.fail_rows],
            "blocked_rows": [row.as_dict() for row in self.blocked_rows],
            "required_outputs": {
                "state_variable_candidate_summary.json": "this payload",
                "blocked_rows.md": "derive from blocked_rows",
                "identity_public_rows": "pass/fail/blocked rows with evidence_kind=identity",
                "prediction_public_rows": "pass/fail/blocked rows with evidence_kind=prediction",
                "intervention_public_rows": "pass/fail/blocked rows with evidence_kind=intervention",
            },
            "claim_policy": {
                "real_public_rows_required": True,
                "does_not_claim_world_model": True,
                "does_not_claim_open_world_pass": True,
                "blocked_rows_are_not_pass_rows": True,
                "synthetic_smoke_is_prerequisite_not_reality_proof": True,
                "caller_declared_status_is_diagnostic_only": True,
                "derived_metrics_recomputed_from_primitives": True,
                "metric_level_independence_only": True,
                "does_not_reconstruct_metrics_from_artifact_files": True,
                "identity_pass_requires_explicit_raw_prediction_observations": True,
            },
            "non_goals": {
                "trains_dynamics_model": False,
                "uses_replay_buffer": False,
                "uses_diffusion": False,
                "uses_renderer_loss": False,
                "mutates_viewer_defaults": False,
                "submits_training_outputs": False,
            },
        }
        return validate_objectstate_reality_gate_summary(payload)


def evaluate_objectstate_reality_gate(
    rows: Sequence[ObjectStateRealityRow],
    *,
    synthetic_smoke_passed: bool,
    thresholds: ObjectStateRealityGateThresholds | None = None,
) -> ObjectStateRealityGateReport:
    checked_thresholds = thresholds or ObjectStateRealityGateThresholds()
    checked_thresholds.as_dict()
    declared_rows = tuple(validate_objectstate_reality_row(row) for row in rows)
    if not declared_rows:
        raise ValueError("objectstate reality gate requires at least one row")
    checked_rows = tuple(
        _derive_reality_row(row, checked_thresholds) for row in declared_rows
    )
    declaration_diagnostics = _declaration_diagnostics(
        declared_rows,
        checked_rows,
    )
    metrics = _reality_metrics(checked_rows)
    hard_gates, hard_blockers = _hard_gate_result(
        metrics,
        synthetic_smoke_passed=bool(synthetic_smoke_passed),
        thresholds=checked_thresholds,
    )
    return ObjectStateRealityGateReport(
        rows=checked_rows,
        synthetic_smoke_passed=bool(synthetic_smoke_passed),
        thresholds=checked_thresholds,
        metrics=metrics,
        hard_gates=hard_gates,
        hard_blockers=hard_blockers,
        declaration_diagnostics=declaration_diagnostics,
    )


def objectstate_reality_blocked_rows_markdown(
    report: ObjectStateRealityGateReport,
) -> str:
    if not isinstance(report, ObjectStateRealityGateReport):
        raise TypeError("report must be ObjectStateRealityGateReport")
    if not report.blocked_rows:
        return "No blocked ObjectState reality rows.\n"
    lines = [
        "| row_id | sample_id | evidence_kind | block_reason | artifact_refs |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report.blocked_rows:
        refs = ", ".join(row.artifact_refs)
        lines.append(
            "| "
            + " | ".join(
                (
                    row.row_id,
                    row.sample_id,
                    row.evidence_kind,
                    row.block_reason or "",
                    refs,
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def validate_objectstate_reality_gate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("objectstate reality gate summary must be a dict")
    if payload.get("schema") != OBJECTSTATE_REALITY_GATE_SCHEMA:
        raise ValueError(f"unsupported objectstate reality gate schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_reality_gate":
        raise ValueError("objectstate reality gate kind must be objectstate_reality_gate")
    if payload.get("status") not in {_GATE_STATUS_PASS, _GATE_STATUS_FAIL}:
        raise ValueError("objectstate reality gate status is unsupported")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("objectstate reality gate requires rows")
    if payload.get("row_count") != len(rows):
        raise ValueError("objectstate reality row_count must match rows")
    for group_name, status in (
        ("pass_rows", "pass"),
        ("fail_rows", "fail"),
        ("blocked_rows", "blocked"),
    ):
        group = payload.get(group_name)
        if not isinstance(group, list):
            raise ValueError(f"objectstate reality gate requires {group_name}")
        if any(row.get("status") != status for row in group):
            raise ValueError(f"{group_name} contains a non-{status} row")
    if payload.get("pass_row_count") != len(payload["pass_rows"]):
        raise ValueError("objectstate reality pass_row_count must match pass_rows")
    if payload.get("fail_row_count") != len(payload["fail_rows"]):
        raise ValueError("objectstate reality fail_row_count must match fail_rows")
    if payload.get("blocked_row_count") != len(payload["blocked_rows"]):
        raise ValueError("objectstate reality blocked_row_count must match blocked_rows")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("objectstate reality gate requires metrics")
    for key in (
        "controlled_real_or_public_row_count",
        "source_kind_counts",
        "evidence_kind_counts",
        "status_counts",
        "controlled_real_identity_collapse",
        "controlled_real_fragmentation_rate",
        "controlled_real_swap_rate",
        "short_horizon_prediction_gap_vs_history_model",
        "intervention_counterfactual_outcome_accuracy",
        "blocked_rows_separated_from_pass_rows",
    ):
        if key not in metrics:
            raise ValueError(f"objectstate reality metrics missing {key}")
    diagnostics = payload.get("declaration_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("objectstate reality gate requires declaration_diagnostics")
    if not isinstance(diagnostics.get("caller_status_mismatches"), list):
        raise ValueError(
            "objectstate reality gate declaration diagnostics require "
            "caller_status_mismatches"
        )
    if diagnostics.get("caller_status_mismatch_count") != len(
        diagnostics["caller_status_mismatches"]
    ):
        raise ValueError(
            "objectstate reality caller_status_mismatch_count must match mismatches"
        )
    if not isinstance(diagnostics.get("derived_metric_mismatches"), list):
        raise ValueError(
            "objectstate reality gate declaration diagnostics require "
            "derived_metric_mismatches"
        )
    if diagnostics.get("derived_metric_mismatch_count") != len(
        diagnostics["derived_metric_mismatches"]
    ):
        raise ValueError(
            "objectstate reality derived_metric_mismatch_count must match mismatches"
        )
    hard_gates = payload.get("hard_gates")
    if not isinstance(hard_gates, dict) or not hard_gates:
        raise ValueError("objectstate reality gate requires hard_gates")
    if any(not isinstance(value, bool) for value in hard_gates.values()):
        raise ValueError("objectstate reality hard gates must be bool")
    expected_status = _GATE_STATUS_PASS if all(hard_gates.values()) else _GATE_STATUS_FAIL
    if payload["status"] != expected_status:
        raise ValueError("objectstate reality status must match hard gates")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("trains_dynamics_model")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("uses_renderer_loss")
        or non_goals.get("mutates_viewer_defaults")
        or non_goals.get("submits_training_outputs")
    ):
        raise ValueError("objectstate reality gate cannot train, replay, diffuse, render, mutate viewer, or submit outputs")
    return payload


def validate_objectstate_reality_gate_thresholds(
    thresholds: ObjectStateRealityGateThresholds,
) -> dict[str, Any]:
    if not isinstance(thresholds, ObjectStateRealityGateThresholds):
        raise TypeError("thresholds must be ObjectStateRealityGateThresholds")
    min_rows = int(thresholds.min_real_or_public_rows)
    if min_rows < 1:
        raise ValueError("min_real_or_public_rows must be >= 1")
    bounded_thresholds = {
        "min_identity_idf1": float(thresholds.min_identity_idf1),
        "max_identity_fragmentation_rate": float(
            thresholds.max_identity_fragmentation_rate
        ),
        "max_identity_swap_rate": float(thresholds.max_identity_swap_rate),
        "min_intervention_counterfactual_outcome_accuracy": float(
            thresholds.min_intervention_counterfactual_outcome_accuracy
        ),
        "max_intervention_wrong_direction_rate": float(
            thresholds.max_intervention_wrong_direction_rate
        ),
    }
    for name, value in bounded_thresholds.items():
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    nonnegative_thresholds = {
        "max_prediction_state_ade": float(thresholds.max_prediction_state_ade),
        "min_prediction_improvement_margin": float(
            thresholds.min_prediction_improvement_margin
        ),
        "max_intervention_action_conditioned_ade": float(
            thresholds.max_intervention_action_conditioned_ade
        ),
        "min_intervention_gain": float(thresholds.min_intervention_gain),
    }
    for name, value in nonnegative_thresholds.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    return {
        "min_real_or_public_rows": min_rows,
        "require_identity_pass_row": bool(thresholds.require_identity_pass_row),
        "require_prediction_pass_row": bool(thresholds.require_prediction_pass_row),
        "require_intervention_pass_row": bool(thresholds.require_intervention_pass_row),
        "fail_on_failed_rows": bool(thresholds.fail_on_failed_rows),
        **bounded_thresholds,
        "require_no_identity_collapse": bool(
            thresholds.require_no_identity_collapse
        ),
        **nonnegative_thresholds,
    }


def validate_objectstate_reality_row(
    row: ObjectStateRealityRow,
) -> ObjectStateRealityRow:
    if not isinstance(row, ObjectStateRealityRow):
        raise TypeError("row must be ObjectStateRealityRow")
    if row.schema != OBJECTSTATE_REALITY_ROW_SCHEMA:
        raise ValueError(f"unsupported objectstate reality row schema: {row.schema}")
    if not row.row_id:
        raise ValueError("row_id must be non-empty")
    if not row.sample_id:
        raise ValueError("sample_id must be non-empty")
    if row.source_kind not in OBJECTSTATE_REALITY_SOURCE_KINDS:
        raise ValueError(f"unsupported source_kind: {row.source_kind}")
    if row.evidence_kind not in OBJECTSTATE_REALITY_EVIDENCE_KINDS:
        raise ValueError(f"unsupported evidence_kind: {row.evidence_kind}")
    if row.status not in OBJECTSTATE_REALITY_ROW_STATUSES:
        raise ValueError(f"unsupported status: {row.status}")
    if not row.object_category:
        raise ValueError("object_category must be non-empty")
    if not row.scenario:
        raise ValueError("scenario must be non-empty")
    modalities = _non_empty_string_tuple(row.observation_modalities, "observation_modalities")
    artifact_refs = _non_empty_string_tuple(row.artifact_refs, "artifact_refs")
    metrics = _normalize_metrics(row.metrics)
    block_reason = None if row.block_reason is None else str(row.block_reason)
    failure_reason = None if row.failure_reason is None else str(row.failure_reason)
    if row.status == "blocked" and not block_reason:
        raise ValueError("blocked reality rows require block_reason")
    if row.status != "blocked" and block_reason:
        raise ValueError("non-blocked reality rows cannot carry block_reason")
    if row.status != "blocked":
        _validate_ground_truth_for_evidence(row)
        _validate_required_metrics(row.evidence_kind, metrics)
    return ObjectStateRealityRow(
        row_id=str(row.row_id),
        sample_id=str(row.sample_id),
        source_kind=row.source_kind,
        evidence_kind=row.evidence_kind,
        status=row.status,
        object_category=str(row.object_category),
        scenario=str(row.scenario),
        observation_modalities=modalities,
        artifact_refs=artifact_refs,
        metrics=metrics,
        has_identity_gt=bool(row.has_identity_gt),
        has_pose_gt=bool(row.has_pose_gt),
        has_action_gt=bool(row.has_action_gt),
        has_timestamp=bool(row.has_timestamp),
        license=str(row.license),
        block_reason=block_reason,
        failure_reason=failure_reason,
        schema=row.schema,
    )


def _validate_ground_truth_for_evidence(row: ObjectStateRealityRow) -> None:
    if not bool(row.has_timestamp):
        raise ValueError("non-blocked reality rows require timestamp ground truth")
    if row.evidence_kind == "identity" and not bool(row.has_identity_gt):
        raise ValueError("identity reality rows require identity ground truth")
    if row.evidence_kind == "prediction" and not bool(row.has_pose_gt):
        raise ValueError("prediction reality rows require pose ground truth")
    if row.evidence_kind == "intervention" and not (
        bool(row.has_pose_gt) and bool(row.has_action_gt)
    ):
        raise ValueError("intervention reality rows require pose and action ground truth")


def _validate_required_metrics(evidence_kind: str, metrics: Mapping[str, Any]) -> None:
    for key in _REQUIRED_METRICS_BY_EVIDENCE[evidence_kind]:
        if key not in metrics:
            raise ValueError(f"{evidence_kind} reality row missing metric {key}")


def _derive_reality_row(
    row: ObjectStateRealityRow,
    thresholds: ObjectStateRealityGateThresholds,
) -> ObjectStateRealityRow:
    if row.status == "blocked":
        return row

    metrics = dict(row.metrics)
    failures: list[str]
    if row.evidence_kind == "identity":
        failures = _identity_metric_failures(metrics, thresholds)
    elif row.evidence_kind == "prediction":
        failures = _prediction_metric_failures(metrics, thresholds)
    elif row.evidence_kind == "intervention":
        failures = _intervention_metric_failures(metrics, thresholds)
    else:  # pragma: no cover - validated before dispatch
        raise ValueError(f"unsupported evidence_kind: {row.evidence_kind}")

    if row.source_kind == "open_world_real":
        failures.append("open_world_real_not_eligible_for_pass")
    status = "pass" if not failures else "fail"
    failure_reason = None
    if failures:
        failure_reason = (
            f"derived {row.evidence_kind} gate failed: " + ", ".join(failures)
        )
    return replace(
        row,
        status=status,
        metrics=metrics,
        block_reason=None,
        failure_reason=failure_reason,
    )


def _identity_metric_failures(
    metrics: dict[str, Any],
    thresholds: ObjectStateRealityGateThresholds,
) -> list[str]:
    idf1 = _bounded_rate_metric(metrics, "idf1")
    fragmentation = _bounded_rate_metric(metrics, "fragmentation_rate")
    swap = _bounded_rate_metric(metrics, "swap_rate")
    collapse = metrics["identity_collapse"]
    if not isinstance(collapse, bool):
        raise TypeError("metric identity_collapse must be bool")
    raw_predictions = metrics.get("raw_prediction_observations", False)
    if not isinstance(raw_predictions, bool):
        raise TypeError("metric raw_prediction_observations must be bool")
    metrics["raw_prediction_observations"] = raw_predictions
    failures = []
    if not raw_predictions:
        failures.append("raw_prediction_observations_required")
    if idf1 < float(thresholds.min_identity_idf1):
        failures.append("idf1_below_minimum")
    if fragmentation > float(thresholds.max_identity_fragmentation_rate):
        failures.append("fragmentation_rate_above_maximum")
    if swap > float(thresholds.max_identity_swap_rate):
        failures.append("swap_rate_above_maximum")
    if thresholds.require_no_identity_collapse and collapse:
        failures.append("identity_collapse_detected")
    return failures


def _prediction_metric_failures(
    metrics: dict[str, Any],
    thresholds: ObjectStateRealityGateThresholds,
) -> list[str]:
    state_ade = _nonnegative_metric(metrics, "state_ade")
    history_ade = _nonnegative_metric(metrics, "history_ade")
    prediction_gap = float(state_ade - history_ade)
    metrics["prediction_gap_vs_history_model"] = prediction_gap
    failures = []
    if state_ade > float(thresholds.max_prediction_state_ade):
        failures.append("state_ade_above_maximum")
    if prediction_gap >= -float(thresholds.min_prediction_improvement_margin):
        failures.append("state_does_not_strictly_beat_history")
    return failures


def _intervention_metric_failures(
    metrics: dict[str, Any],
    thresholds: ObjectStateRealityGateThresholds,
) -> list[str]:
    action_ade = _nonnegative_metric(metrics, "action_conditioned_ade")
    no_action_ade = _nonnegative_metric(metrics, "no_action_ade")
    accuracy = _bounded_rate_metric(metrics, "counterfactual_outcome_accuracy")
    wrong_direction = _bounded_rate_metric(metrics, "wrong_direction_rate")
    gain = float(no_action_ade - action_ade)
    metrics["intervention_gain"] = gain
    failures = []
    if action_ade > float(thresholds.max_intervention_action_conditioned_ade):
        failures.append("action_conditioned_ade_above_maximum")
    if accuracy < float(
        thresholds.min_intervention_counterfactual_outcome_accuracy
    ):
        failures.append("counterfactual_outcome_accuracy_below_minimum")
    if wrong_direction > float(thresholds.max_intervention_wrong_direction_rate):
        failures.append("wrong_direction_rate_above_maximum")
    if gain <= float(thresholds.min_intervention_gain):
        failures.append("intervention_gain_not_positive")
    return failures


def _declaration_diagnostics(
    declared_rows: Sequence[ObjectStateRealityRow],
    derived_rows: Sequence[ObjectStateRealityRow],
) -> dict[str, Any]:
    status_mismatches = []
    derived_metric_mismatches = []
    for declared, derived in zip(declared_rows, derived_rows, strict=True):
        if declared.status != derived.status:
            status_mismatches.append(
                {
                    "row_id": derived.row_id,
                    "caller_status": declared.status,
                    "derived_status": derived.status,
                    "caller_failure_reason": declared.failure_reason,
                    "derived_failure_reason": derived.failure_reason,
                }
            )
        derived_metric = {
            "prediction": "prediction_gap_vs_history_model",
            "intervention": "intervention_gain",
        }.get(derived.evidence_kind)
        if derived.status == "blocked" or derived_metric is None:
            continue
        caller_value = declared.metrics.get(derived_metric)
        if caller_value is None:
            continue
        canonical_value = derived.metrics[derived_metric]
        if isinstance(caller_value, bool) or not np.isclose(
            float(caller_value),
            float(canonical_value),
            rtol=0.0,
            atol=1e-12,
        ):
            derived_metric_mismatches.append(
                {
                    "row_id": derived.row_id,
                    "metric": derived_metric,
                    "caller_value": caller_value,
                    "derived_value": canonical_value,
                }
            )
    return {
        "caller_status_is_diagnostic_only": True,
        "caller_status_mismatch_count": len(status_mismatches),
        "caller_status_mismatches": status_mismatches,
        "derived_metrics_are_recomputed": True,
        "derived_metric_mismatch_count": len(derived_metric_mismatches),
        "derived_metric_mismatches": derived_metric_mismatches,
    }


def _nonnegative_metric(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics[key]
    if isinstance(value, bool):
        raise TypeError(f"metric {key} must be numeric")
    result = float(value)
    if result < 0.0:
        raise ValueError(f"metric {key} must be non-negative")
    return result


def _bounded_rate_metric(metrics: Mapping[str, Any], key: str) -> float:
    result = _nonnegative_metric(metrics, key)
    if result > 1.0:
        raise ValueError(f"metric {key} must be in [0, 1]")
    return result


def _hard_gate_result(
    metrics: Mapping[str, Any],
    *,
    synthetic_smoke_passed: bool,
    thresholds: ObjectStateRealityGateThresholds,
) -> tuple[dict[str, bool], tuple[str, ...]]:
    hard_gates = {
        "synthetic_smoke_passed": bool(synthetic_smoke_passed),
        "real_or_public_rows_present": (
            int(metrics["controlled_real_or_public_row_count"])
            >= int(thresholds.min_real_or_public_rows)
        ),
        "identity_pass_rows_present": (
            not thresholds.require_identity_pass_row
            or int(metrics["pass_evidence_kind_counts"].get("identity", 0)) > 0
        ),
        "prediction_pass_rows_present": (
            not thresholds.require_prediction_pass_row
            or int(metrics["pass_evidence_kind_counts"].get("prediction", 0)) > 0
        ),
        "intervention_pass_rows_present": (
            not thresholds.require_intervention_pass_row
            or int(metrics["pass_evidence_kind_counts"].get("intervention", 0)) > 0
        ),
        "controlled_real_identity_collapse_absent": not bool(
            metrics["controlled_real_identity_collapse"]
        ),
        "non_blocked_rows_have_required_metrics": bool(
            metrics["non_blocked_rows_have_required_metrics"]
        ),
        "blocked_rows_separated_from_pass_rows": bool(
            metrics["blocked_rows_separated_from_pass_rows"]
        ),
        "no_open_world_pass_rows": int(metrics["open_world_pass_row_count"]) == 0,
        "failed_rows_absent": (
            not thresholds.fail_on_failed_rows or int(metrics["status_counts"].get("fail", 0)) == 0
        ),
    }
    hard_blockers = tuple(
        key for key, value in hard_gates.items() if not bool(value)
    )
    return hard_gates, hard_blockers


def _reality_metrics(rows: Sequence[ObjectStateRealityRow]) -> dict[str, Any]:
    checked_rows = tuple(validate_objectstate_reality_row(row) for row in rows)
    source_counts = _counts(row.source_kind for row in checked_rows)
    evidence_counts = _counts(row.evidence_kind for row in checked_rows)
    status_counts = _counts(row.status for row in checked_rows)
    pass_rows = tuple(row for row in checked_rows if row.status == "pass")
    non_blocked_rows = tuple(row for row in checked_rows if row.status != "blocked")
    real_public_rows = tuple(
        row for row in checked_rows if row.source_kind in _REAL_OR_PUBLIC_SOURCE_KINDS
    )
    identity_rows = tuple(
        row
        for row in real_public_rows
        if row.evidence_kind == "identity" and row.status != "blocked"
    )
    prediction_rows = tuple(
        row
        for row in real_public_rows
        if row.evidence_kind == "prediction" and row.status != "blocked"
    )
    intervention_rows = tuple(
        row
        for row in real_public_rows
        if row.evidence_kind == "intervention" and row.status != "blocked"
    )
    return {
        "controlled_real_or_public_row_count": len(real_public_rows),
        "source_kind_counts": source_counts,
        "evidence_kind_counts": evidence_counts,
        "status_counts": status_counts,
        "pass_evidence_kind_counts": _counts(row.evidence_kind for row in pass_rows),
        "controlled_real_identity_collapse": any(
            bool(row.metrics.get("identity_collapse", False)) for row in identity_rows
        ),
        "controlled_real_fragmentation_rate": _mean_metric(
            identity_rows,
            "fragmentation_rate",
        ),
        "controlled_real_swap_rate": _mean_metric(identity_rows, "swap_rate"),
        "short_horizon_prediction_gap_vs_history_model": _mean_metric(
            prediction_rows,
            "prediction_gap_vs_history_model",
        ),
        "intervention_counterfactual_outcome_accuracy": _mean_metric(
            intervention_rows,
            "counterfactual_outcome_accuracy",
        ),
        "blocked_rows_separated_from_pass_rows": all(
            row.status != "pass" or not row.block_reason for row in checked_rows
        ),
        "non_blocked_rows_have_required_metrics": all(
            _has_required_metrics(row) for row in non_blocked_rows
        ),
        "open_world_pass_row_count": sum(
            1
            for row in checked_rows
            if row.source_kind == "open_world_real" and row.status == "pass"
        ),
    }


def _has_required_metrics(row: ObjectStateRealityRow) -> bool:
    return all(
        key in row.metrics for key in _REQUIRED_METRICS_BY_EVIDENCE[row.evidence_kind]
    )


def _mean_metric(rows: Sequence[ObjectStateRealityRow], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.metrics.get(key)
        if isinstance(value, bool) or value is None:
            continue
        values.append(float(value))
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _counts(values: Sequence[str] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        text = str(value)
        counts[text] = counts.get(text, 0) + 1
    return counts


def _normalize_metrics(metrics: Mapping[str, Any]) -> dict[str, float | bool]:
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    normalized: dict[str, float | bool] = {}
    for key, value in metrics.items():
        metric_key = str(key)
        if not metric_key:
            raise ValueError("metric keys must be non-empty")
        if isinstance(value, (bool, np.bool_)):
            normalized[metric_key] = bool(value)
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            metric_value = float(value)
            if not np.isfinite(metric_value):
                raise ValueError(f"metric {metric_key} must be finite")
            normalized[metric_key] = metric_value
            continue
        raise TypeError(f"metric {metric_key} must be numeric or bool")
    return normalized


def _non_empty_string_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    normalized = tuple(str(value) for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{name} must contain non-empty strings")
    return normalized
