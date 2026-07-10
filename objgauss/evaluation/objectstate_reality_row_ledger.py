from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.evaluation.objectstate_bop_reality_rows import (
    OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA,
    validate_objectstate_bop_reality_rows_summary,
)
from objgauss.evaluation.objectstate_controlled_real_rows import (
    OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
    validate_objectstate_controlled_real_rows_summary,
)
from objgauss.evaluation.objectstate_public_interaction_reality_rows import (
    OBJECTSTATE_PUBLIC_INTERACTION_REALITY_ROWS_SCHEMA,
    validate_objectstate_public_interaction_reality_rows_summary,
)
from objgauss.evaluation.objectstate_real_identity_rows import (
    OBJECTSTATE_REAL_IDENTITY_ROWS_SCHEMA,
    validate_objectstate_real_identity_rows_summary,
)
from objgauss.evaluation.objectstate_real_intervention_rows import (
    OBJECTSTATE_REAL_INTERVENTION_ROWS_SCHEMA,
    validate_objectstate_real_intervention_rows_summary,
)
from objgauss.evaluation.objectstate_real_prediction_rows import (
    OBJECTSTATE_REAL_PREDICTION_ROWS_SCHEMA,
    validate_objectstate_real_prediction_rows_summary,
)
from objgauss.evaluation.objectstate_reality_gate import (
    OBJECTSTATE_REALITY_GATE_SCHEMA,
    ObjectStateRealityGateThresholds,
    ObjectStateRealityRow,
    evaluate_objectstate_reality_gate,
    objectstate_reality_blocked_rows_markdown,
    validate_objectstate_reality_gate_summary,
    validate_objectstate_reality_row,
)
from objgauss.evaluation.objectstate_reality_public_rows import (
    OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA,
    validate_objectstate_reality_public_rows_summary,
)

OBJECTSTATE_REALITY_ROW_LEDGER_SCHEMA = (
    "objgauss-objectstate-reality-row-ledger-v1"
)

__all__ = (
    "OBJECTSTATE_REALITY_ROW_LEDGER_SCHEMA",
    "read_objectstate_reality_row_summary",
    "objectstate_reality_rows_from_summary",
    "objectstate_reality_row_ledger",
    "validate_objectstate_reality_row_ledger_summary",
)
_SUPPORTED_SUMMARY_SCHEMAS = {
    OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA,
    OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
    OBJECTSTATE_PUBLIC_INTERACTION_REALITY_ROWS_SCHEMA,
    OBJECTSTATE_REAL_IDENTITY_ROWS_SCHEMA,
    OBJECTSTATE_REAL_INTERVENTION_ROWS_SCHEMA,
    OBJECTSTATE_REAL_PREDICTION_ROWS_SCHEMA,
    OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA,
    OBJECTSTATE_REALITY_GATE_SCHEMA,
}
_STATE_VARIABLE_EXPERIMENTS = (
    {
        "experiment": "identity_persistence",
        "evidence_kind": "identity",
        "required_metrics": (
            "idf1",
            "fragmentation_rate",
            "swap_rate",
            "identity_collapse",
        ),
        "challenge_metrics": (),
    },
    {
        "experiment": "occlusion_recovery",
        "evidence_kind": "identity",
        "required_metrics": ("occlusion_recovery_rate",),
        "challenge_metrics": ("occlusion_challenge_present",),
    },
    {
        "experiment": "view_invariance",
        "evidence_kind": "identity",
        "required_metrics": ("contrastive_margin",),
        "challenge_metrics": ("view_challenge_present",),
    },
    {
        "experiment": "predictive_sufficiency",
        "evidence_kind": "prediction",
        "required_metrics": (
            "state_ade",
            "history_ade",
            "prediction_gap_vs_history_model",
        ),
        "challenge_metrics": (),
    },
    {
        "experiment": "counterfactual_action_interface",
        "evidence_kind": "intervention",
        "required_metrics": (
            "action_conditioned_ade",
            "counterfactual_outcome_accuracy",
            "wrong_direction_rate",
        ),
        "challenge_metrics": ("action_challenge_present",),
    },
)


def read_objectstate_reality_row_summary(path: str | Path) -> dict[str, Any]:
    summary_path = Path(path)
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("ObjectState reality row summary JSON must be an object")
    return _validate_supported_summary(payload)


def objectstate_reality_rows_from_summary(
    summary: Mapping[str, Any],
) -> tuple[ObjectStateRealityRow, ...]:
    checked = _validate_supported_summary(summary)
    rows = _rows_payloads_from_summary(checked)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("ObjectState reality row summary requires rows")
    return tuple(_row_from_payload(row) for row in rows)


def objectstate_reality_row_ledger(
    summary_paths: Sequence[str | Path],
    *,
    synthetic_smoke_passed: bool = True,
    thresholds: ObjectStateRealityGateThresholds | None = None,
) -> dict[str, Any]:
    paths = tuple(Path(path) for path in summary_paths)
    records: list[dict[str, Any]] = []
    rows: list[ObjectStateRealityRow] = []
    for path in paths:
        record, record_rows = _record_from_path(path)
        records.append(record)
        rows.extend(record_rows)

    duplicate_row_ids = _duplicates(row.row_id for row in rows)
    ledger_gates = {
        "summaries_present": bool(paths),
        "all_files_present": all(record["is_file"] for record in records),
        "all_json_schemas_valid": all(record["schema_ok"] for record in records),
        "all_summary_validators_passed": all(
            record["validator_ok"] for record in records
        ),
        "rows_present": bool(rows),
        "duplicate_row_ids_absent": not duplicate_row_ids,
    }
    gate_payload: dict[str, Any] | None = None
    blocked_rows_markdown = "No ObjectState reality rows were available.\n"
    evaluated_rows = list(rows)
    if rows:
        gate = evaluate_objectstate_reality_gate(
            tuple(rows),
            synthetic_smoke_passed=bool(synthetic_smoke_passed),
            thresholds=thresholds,
        )
        gate_payload = gate.as_dict()
        evaluated_rows = list(gate.rows)
        blocked_rows_markdown = objectstate_reality_blocked_rows_markdown(gate)
    next_actions = _next_actions(gate_payload, evaluated_rows)
    experiment_matrix = _state_variable_evidence_matrix(evaluated_rows)

    payload = {
        "schema": OBJECTSTATE_REALITY_ROW_LEDGER_SCHEMA,
        "kind": "objectstate_reality_row_ledger",
        "status": (
            "objectstate_reality_row_ledger_reviewable"
            if all(ledger_gates.values())
            else "objectstate_reality_row_ledger_incomplete"
        ),
        "summary_count": len(paths),
        "records": records,
        "row_schema": "objgauss-objectstate-real-public-row-v1",
        "row_count": len(evaluated_rows),
        "pass_row_count": _row_status_count(evaluated_rows, "pass"),
        "fail_row_count": _row_status_count(evaluated_rows, "fail"),
        "blocked_row_count": _row_status_count(evaluated_rows, "blocked"),
        "sample_scope": _sample_scope(evaluated_rows),
        "row_counts": {
            "by_source_kind": _counts(row.source_kind for row in evaluated_rows),
            "by_evidence_kind": _counts(row.evidence_kind for row in evaluated_rows),
            "by_status": _counts(row.status for row in evaluated_rows),
        },
        "duplicate_row_ids": duplicate_row_ids,
        "rows": [row.as_dict() for row in evaluated_rows],
        "gate": gate_payload,
        "gap_summary": _gap_summary(gate_payload, evaluated_rows),
        "state_variable_evidence_matrix": experiment_matrix,
        "state_variable_evidence_matrix_markdown": (
            _state_variable_evidence_matrix_markdown(experiment_matrix)
        ),
        "next_actions": next_actions,
        "next_actions_markdown": _next_actions_markdown(next_actions),
        "blocked_rows_markdown": blocked_rows_markdown,
        "ledger_gates": ledger_gates,
        "issues": _issues(records, ledger_gates, gate_payload, duplicate_row_ids),
        "claim_policy": {
            "read_only_audit": True,
            "checks_existing_reality_row_summaries": True,
            "does_not_create_ground_truth": True,
            "does_not_run_bop_handoff": True,
            "does_not_run_identity_eval": True,
            "does_not_run_prediction_eval": True,
            "does_not_run_intervention_eval": True,
            "does_not_train_model": True,
            "does_not_claim_world_model": True,
            "full_gate_status_is_authoritative": True,
            "blocked_rows_are_not_pass_rows": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "captures_video": False,
            "creates_ground_truth": False,
            "reconstructs_gaussians": False,
            "runs_tracking_model": False,
            "runs_identity_model": False,
            "runs_prediction_model": False,
            "runs_intervention_model": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_reality_row_ledger_summary(payload)


def validate_objectstate_reality_row_ledger_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("ObjectState reality row ledger must be a mapping")
    if payload.get("schema") != OBJECTSTATE_REALITY_ROW_LEDGER_SCHEMA:
        raise ValueError(
            f"unsupported ObjectState reality row ledger schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_reality_row_ledger":
        raise ValueError("ObjectState reality row ledger kind is unsupported")
    if payload.get("status") not in {
        "objectstate_reality_row_ledger_reviewable",
        "objectstate_reality_row_ledger_incomplete",
    }:
        raise ValueError("ObjectState reality row ledger status is unsupported")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("ObjectState reality row ledger requires records")
    if payload.get("summary_count") != len(records):
        raise ValueError("ObjectState reality row ledger summary_count mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("ObjectState reality row ledger requires rows")
    if payload.get("row_count") != len(rows):
        raise ValueError("ObjectState reality row ledger row_count mismatch")
    checked_rows = tuple(_row_from_payload(row) for row in rows)
    for status, count_key in (
        ("pass", "pass_row_count"),
        ("fail", "fail_row_count"),
        ("blocked", "blocked_row_count"),
    ):
        if payload.get(count_key) != _row_status_count(checked_rows, status):
            raise ValueError(
                f"ObjectState reality row ledger {count_key} must match rows"
            )
    gate = payload.get("gate")
    if gate is not None:
        if not isinstance(gate, Mapping):
            raise ValueError("ObjectState reality row ledger gate must be a mapping")
        validate_objectstate_reality_gate_summary(dict(gate))
        if rows != gate.get("rows"):
            raise ValueError(
                "ObjectState reality row ledger rows must match gate-derived rows"
            )
        if payload["row_count"] != gate.get("row_count"):
            raise ValueError("ObjectState reality row ledger row_count must match gate")
        if payload["pass_row_count"] != gate.get("pass_row_count"):
            raise ValueError("ObjectState reality row ledger pass count must match gate")
        if payload["fail_row_count"] != gate.get("fail_row_count"):
            raise ValueError("ObjectState reality row ledger fail count must match gate")
        if payload["blocked_row_count"] != gate.get("blocked_row_count"):
            raise ValueError("ObjectState reality row ledger blocked count must match gate")
    ledger_gates = payload.get("ledger_gates")
    if not isinstance(ledger_gates, Mapping) or not ledger_gates:
        raise ValueError("ObjectState reality row ledger requires ledger_gates")
    if any(not isinstance(value, bool) for value in ledger_gates.values()):
        raise ValueError("ObjectState reality row ledger gates must be bool")
    expected_status = (
        "objectstate_reality_row_ledger_reviewable"
        if all(ledger_gates.values())
        else "objectstate_reality_row_ledger_incomplete"
    )
    if payload["status"] != expected_status:
        raise ValueError("ObjectState reality row ledger status mismatch")
    if not isinstance(payload.get("sample_scope"), Mapping):
        raise ValueError("ObjectState reality row ledger requires sample_scope")
    if payload["sample_scope"] != _sample_scope(checked_rows):
        raise ValueError("ObjectState reality row ledger sample_scope must match rows")
    if not isinstance(payload.get("row_counts"), Mapping):
        raise ValueError("ObjectState reality row ledger requires row_counts")
    expected_row_counts = {
        "by_source_kind": _counts(row.source_kind for row in checked_rows),
        "by_evidence_kind": _counts(row.evidence_kind for row in checked_rows),
        "by_status": _counts(row.status for row in checked_rows),
    }
    if payload["row_counts"] != expected_row_counts:
        raise ValueError("ObjectState reality row ledger row_counts must match rows")
    if not isinstance(payload.get("duplicate_row_ids"), list):
        raise ValueError("ObjectState reality row ledger requires duplicate_row_ids")
    if not isinstance(payload.get("gap_summary"), Mapping):
        raise ValueError("ObjectState reality row ledger requires gap_summary")
    if payload["gap_summary"] != _gap_summary(gate, checked_rows):
        raise ValueError("ObjectState reality row ledger gap_summary must match rows")
    experiment_matrix = payload.get("state_variable_evidence_matrix")
    if not isinstance(experiment_matrix, list):
        raise ValueError(
            "ObjectState reality row ledger requires state_variable_evidence_matrix"
        )
    for record in experiment_matrix:
        _validate_experiment_record(record)
    if experiment_matrix != _state_variable_evidence_matrix(checked_rows):
        raise ValueError(
            "ObjectState reality row ledger evidence matrix must match rows"
        )
    if not isinstance(payload.get("state_variable_evidence_matrix_markdown"), str):
        raise ValueError(
            "ObjectState reality row ledger requires "
            "state_variable_evidence_matrix_markdown"
        )
    next_actions = payload.get("next_actions")
    if not isinstance(next_actions, list):
        raise ValueError("ObjectState reality row ledger requires next_actions")
    for action in next_actions:
        if not isinstance(action, Mapping):
            raise ValueError("ObjectState reality row ledger action must be a mapping")
        for key in (
            "evidence_kind",
            "status",
            "priority",
            "required_evidence",
            "minimum_metrics",
            "recommended_route",
            "commands",
            "claim_boundary",
        ):
            if key not in action:
                raise ValueError(
                    f"ObjectState reality row ledger action requires {key}"
                )
    if next_actions != _next_actions(gate, checked_rows):
        raise ValueError("ObjectState reality row ledger next_actions must match rows")
    if not isinstance(payload.get("next_actions_markdown"), str):
        raise ValueError(
            "ObjectState reality row ledger requires next_actions_markdown"
        )
    if not isinstance(payload.get("blocked_rows_markdown"), str):
        raise ValueError("ObjectState reality row ledger requires blocked_rows_markdown")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("ObjectState reality row ledger requires issues")
    claim_policy = payload.get("claim_policy", {})
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("read_only_audit")
        or not claim_policy.get("checks_existing_reality_row_summaries")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_run_bop_handoff")
        or not claim_policy.get("does_not_run_identity_eval")
        or not claim_policy.get("does_not_run_prediction_eval")
        or not claim_policy.get("does_not_run_intervention_eval")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_claim_world_model")
        or not claim_policy.get("full_gate_status_is_authoritative")
        or not claim_policy.get("blocked_rows_are_not_pass_rows")
    ):
        raise ValueError("ObjectState reality row ledger must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "ObjectState reality row ledger cannot download, capture, create GT, "
            "reconstruct, run models, train, write public samples, replay, diffuse, "
            "or mutate viewer policy"
        )
    return dict(payload)


def _record_from_path(path: Path) -> tuple[dict[str, Any], tuple[ObjectStateRealityRow, ...]]:
    record = {
        "path": str(path),
        "is_file": path.is_file(),
        "schema": None,
        "kind": None,
        "schema_ok": False,
        "validator_ok": False,
        "sample_ids": [],
        "row_count": 0,
        "issues": [],
    }
    if not path.is_file():
        record["issues"].append("summary file is missing")
        return record, ()
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # pragma: no cover - exact JSON errors are runtime-specific.
        record["issues"].append(f"failed to read summary JSON: {exc}")
        return record, ()
    if not isinstance(payload, dict):
        record["issues"].append("summary JSON is not an object")
        return record, ()
    record["schema"] = payload.get("schema")
    record["kind"] = payload.get("kind")
    record["schema_ok"] = record["schema"] in _SUPPORTED_SUMMARY_SCHEMAS
    if not record["schema_ok"]:
        record["issues"].append(f"unsupported summary schema: {record['schema']}")
        return record, ()
    try:
        checked = _validate_supported_summary(payload)
        rows = objectstate_reality_rows_from_summary(checked)
    except Exception as exc:
        record["issues"].append(f"summary validator failed: {exc}")
        return record, ()
    record["validator_ok"] = True
    record["row_count"] = len(rows)
    record["sample_ids"] = sorted({row.sample_id for row in rows})
    return record, rows


def _validate_supported_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        raise TypeError("ObjectState reality row summary must be a mapping")
    schema = summary.get("schema")
    if schema == OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA:
        return validate_objectstate_bop_reality_rows_summary(summary)
    if schema == OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA:
        return validate_objectstate_controlled_real_rows_summary(dict(summary))
    if schema == OBJECTSTATE_PUBLIC_INTERACTION_REALITY_ROWS_SCHEMA:
        return validate_objectstate_public_interaction_reality_rows_summary(summary)
    if schema == OBJECTSTATE_REAL_IDENTITY_ROWS_SCHEMA:
        return validate_objectstate_real_identity_rows_summary(summary)
    if schema == OBJECTSTATE_REAL_INTERVENTION_ROWS_SCHEMA:
        return validate_objectstate_real_intervention_rows_summary(summary)
    if schema == OBJECTSTATE_REAL_PREDICTION_ROWS_SCHEMA:
        return validate_objectstate_real_prediction_rows_summary(summary)
    if schema == OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA:
        return validate_objectstate_reality_public_rows_summary(dict(summary))
    if schema == OBJECTSTATE_REALITY_GATE_SCHEMA:
        return validate_objectstate_reality_gate_summary(dict(summary))
    raise ValueError(f"unsupported ObjectState reality row summary schema: {schema}")


def _rows_payloads_from_summary(summary: Mapping[str, Any]) -> Any:
    gate = summary.get("gate")
    if not isinstance(gate, Mapping):
        for gate_key in ("identity_gate", "prediction_gate", "intervention_gate"):
            candidate = summary.get(gate_key)
            if isinstance(candidate, Mapping):
                gate = candidate
                break
    if isinstance(gate, Mapping) and "rows" in gate:
        return gate.get("rows")
    for key in ("rows", "identity_rows", "prediction_rows", "intervention_rows"):
        if key in summary:
            return summary.get(key)
    return None


def _row_from_payload(payload: Mapping[str, Any]) -> ObjectStateRealityRow:
    if not isinstance(payload, Mapping):
        raise TypeError("ObjectState reality row payload must be a mapping")
    ground_truth = payload.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        raise ValueError("ObjectState reality row payload requires ground_truth")
    return validate_objectstate_reality_row(
        ObjectStateRealityRow(
            row_id=str(payload.get("row_id", "")),
            sample_id=str(payload.get("sample_id", "")),
            source_kind=str(payload.get("source_kind", "")),
            evidence_kind=str(payload.get("evidence_kind", "")),
            status=str(payload.get("status", "")),
            object_category=str(payload.get("object_category", "")),
            scenario=str(payload.get("scenario", "")),
            observation_modalities=tuple(payload.get("observation_modalities", ())),
            artifact_refs=tuple(payload.get("artifact_refs", ())),
            metrics=payload.get("metrics", {}),
            has_identity_gt=bool(ground_truth.get("identity", False)),
            has_pose_gt=bool(ground_truth.get("pose", False)),
            has_action_gt=bool(ground_truth.get("action", False)),
            has_timestamp=bool(ground_truth.get("timestamp", False)),
            license=str(payload.get("license", "unknown")),
            block_reason=payload.get("block_reason"),
            failure_reason=payload.get("failure_reason"),
            schema=str(payload.get("schema", "")),
        )
    )


def _gap_summary(
    gate: Mapping[str, Any] | None,
    rows: Sequence[ObjectStateRealityRow],
) -> dict[str, Any]:
    pass_kinds = sorted({row.evidence_kind for row in rows if row.status == "pass"})
    fail_kinds = sorted({row.evidence_kind for row in rows if row.status == "fail"})
    blocked_kinds = sorted(
        {row.evidence_kind for row in rows if row.status == "blocked"}
    )
    required = ("identity", "prediction", "intervention")
    return {
        "required_pass_evidence_kinds": list(required),
        "present_pass_evidence_kinds": pass_kinds,
        "missing_pass_evidence_kinds": [
            kind for kind in required if kind not in pass_kinds
        ],
        "failed_evidence_kinds": fail_kinds,
        "blocked_evidence_kinds": blocked_kinds,
        "full_gate_status": None if gate is None else gate.get("status"),
        "hard_blockers": [] if gate is None else list(gate.get("hard_blockers", ())),
    }


def _state_variable_evidence_matrix(
    rows: Sequence[ObjectStateRealityRow],
) -> list[dict[str, Any]]:
    records = []
    for spec in _STATE_VARIABLE_EXPERIMENTS:
        evidence_kind = str(spec["evidence_kind"])
        required_metrics = tuple(str(metric) for metric in spec["required_metrics"])
        challenge_metrics = tuple(str(metric) for metric in spec["challenge_metrics"])
        experiment_rows = [row for row in rows if row.evidence_kind == evidence_kind]
        metric_rows = [
            row
            for row in experiment_rows
            if all(metric in row.metrics for metric in required_metrics)
        ]
        challenge_status = _challenge_status(experiment_rows, challenge_metrics)
        status = _experiment_status(experiment_rows, metric_rows)
        records.append(
            {
                "experiment": str(spec["experiment"]),
                "evidence_kind": evidence_kind,
                "status": status,
                "challenge_status": challenge_status,
                "required_metrics": list(required_metrics),
                "challenge_metrics": list(challenge_metrics),
                "present_metrics": sorted(
                    {
                        str(metric)
                        for row in experiment_rows
                        for metric in row.metrics.keys()
                    }
                ),
                "missing_metrics": [
                    metric
                    for metric in required_metrics
                    if not any(metric in row.metrics for row in experiment_rows)
                ],
                "source_row_ids": [row.row_id for row in experiment_rows],
                "metric_row_ids": [row.row_id for row in metric_rows],
                "row_status_counts": {
                    "pass": _row_status_count(experiment_rows, "pass"),
                    "fail": _row_status_count(experiment_rows, "fail"),
                    "blocked": _row_status_count(experiment_rows, "blocked"),
                },
                "interpretation": _experiment_interpretation(
                    str(spec["experiment"]),
                    status,
                    evidence_kind=evidence_kind,
                    metric_rows=metric_rows,
                ),
            }
        )
    return records


def _experiment_status(
    experiment_rows: Sequence[ObjectStateRealityRow],
    metric_rows: Sequence[ObjectStateRealityRow],
) -> str:
    if any(row.status == "pass" for row in metric_rows):
        return "objectstate_state_variable_experiment_pass"
    if any(row.status == "fail" for row in metric_rows):
        return "objectstate_state_variable_experiment_fail"
    if any(row.status == "blocked" for row in experiment_rows):
        return "objectstate_state_variable_experiment_blocked"
    if experiment_rows:
        return "objectstate_state_variable_experiment_missing_metric"
    return "objectstate_state_variable_experiment_missing_row"


def _experiment_interpretation(
    experiment: str,
    status: str,
    *,
    evidence_kind: str,
    metric_rows: Sequence[ObjectStateRealityRow],
) -> str:
    if status == "objectstate_state_variable_experiment_pass":
        pass_rows = [row.row_id for row in metric_rows if row.status == "pass"]
        return f"{experiment} has pass evidence from rows: {', '.join(pass_rows)}"
    if status == "objectstate_state_variable_experiment_fail":
        fail_rows = [row.row_id for row in metric_rows if row.status == "fail"]
        return f"{experiment} has explicit fail evidence from rows: {', '.join(fail_rows)}"
    if status == "objectstate_state_variable_experiment_blocked":
        return f"{experiment} is blocked by existing {evidence_kind} rows"
    if status == "objectstate_state_variable_experiment_missing_metric":
        return (
            f"{experiment} has {evidence_kind} rows but lacks the experiment "
            "metric fields required for this state-variable claim"
        )
    return f"{experiment} has no {evidence_kind} rows in this ledger"


def _challenge_status(
    rows: Sequence[ObjectStateRealityRow],
    challenge_metrics: Sequence[str],
) -> str:
    if not challenge_metrics:
        return "objectstate_state_variable_challenge_not_required"
    if not rows:
        return "objectstate_state_variable_challenge_missing_row"
    present_values = []
    for metric in challenge_metrics:
        for row in rows:
            if metric in row.metrics:
                present_values.append(row.metrics[metric])
    if not present_values:
        return "objectstate_state_variable_challenge_unknown"
    if any(_truthy_metric(value) for value in present_values):
        return "objectstate_state_variable_challenge_present"
    return "objectstate_state_variable_challenge_absent"


def _truthy_metric(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _state_variable_evidence_matrix_markdown(
    matrix: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# ObjectState State-Variable Evidence Matrix",
        "",
        "| experiment | evidence_kind | status / challenge | missing_metrics | source_rows |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in matrix:
        lines.append(
            "| "
            + str(record.get("experiment", ""))
            + " | "
            + str(record.get("evidence_kind", ""))
            + " | "
            + str(record.get("status", ""))
            + " / "
            + str(record.get("challenge_status", ""))
            + " | "
            + ", ".join(str(item) for item in record.get("missing_metrics", ()))
            + " | "
            + ", ".join(str(item) for item in record.get("source_row_ids", ()))
            + " |"
        )
    lines.extend(
        [
            "",
            "This matrix is read-only. It maps existing reality rows to the five "
            "state-variable experiments and does not create ground truth, train "
            "models, relax gates, or claim a world-model pass.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_experiment_record(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("ObjectState reality row ledger experiment must be a mapping")
    for key in (
        "experiment",
        "evidence_kind",
        "status",
        "challenge_status",
        "required_metrics",
        "challenge_metrics",
        "present_metrics",
        "missing_metrics",
        "source_row_ids",
        "metric_row_ids",
        "row_status_counts",
        "interpretation",
    ):
        if key not in value:
            raise ValueError(
                f"ObjectState reality row ledger experiment requires {key}"
            )
    if value["status"] not in {
        "objectstate_state_variable_experiment_pass",
        "objectstate_state_variable_experiment_fail",
        "objectstate_state_variable_experiment_blocked",
        "objectstate_state_variable_experiment_missing_metric",
        "objectstate_state_variable_experiment_missing_row",
    }:
        raise ValueError("ObjectState reality row ledger experiment status unsupported")
    if value["challenge_status"] not in {
        "objectstate_state_variable_challenge_not_required",
        "objectstate_state_variable_challenge_missing_row",
        "objectstate_state_variable_challenge_unknown",
        "objectstate_state_variable_challenge_present",
        "objectstate_state_variable_challenge_absent",
    }:
        raise ValueError(
            "ObjectState reality row ledger experiment challenge status unsupported"
        )
    for key in (
        "required_metrics",
        "challenge_metrics",
        "present_metrics",
        "missing_metrics",
        "source_row_ids",
        "metric_row_ids",
    ):
        if not isinstance(value.get(key), list):
            raise ValueError(
                f"ObjectState reality row ledger experiment {key} must be list"
            )
    counts = value.get("row_status_counts")
    if not isinstance(counts, Mapping) or any(
        not isinstance(counts.get(status), int)
        for status in ("pass", "fail", "blocked")
    ):
        raise ValueError(
            "ObjectState reality row ledger experiment requires row_status_counts"
        )


def _next_actions(
    gate: Mapping[str, Any] | None,
    rows: Sequence[ObjectStateRealityRow],
) -> list[dict[str, Any]]:
    pass_kinds = {row.evidence_kind for row in rows if row.status == "pass"}
    status_counts = {
        kind: {
            "pass": _row_status_count(
                [row for row in rows if row.evidence_kind == kind],
                "pass",
            ),
            "fail": _row_status_count(
                [row for row in rows if row.evidence_kind == kind],
                "fail",
            ),
            "blocked": _row_status_count(
                [row for row in rows if row.evidence_kind == kind],
                "blocked",
            ),
        }
        for kind in ("identity", "prediction", "intervention")
    }
    actions: list[dict[str, Any]] = []
    for kind in ("identity", "prediction", "intervention"):
        if kind in pass_kinds:
            continue
        actions.append(
            _next_action_for_kind(
                kind,
                gate_status=None if gate is None else str(gate.get("status")),
                counts=status_counts[kind],
            )
        )
    return actions


def _next_action_for_kind(
    kind: str,
    *,
    gate_status: str | None,
    counts: Mapping[str, int],
) -> dict[str, Any]:
    common_boundary = [
        "Do not mark blocked rows as pass rows.",
        "Do not claim ObjectState is a world model from this ledger.",
        "Do not use renderer-facing object_id as physical identity ground truth.",
    ]
    if kind == "identity":
        return {
            "evidence_kind": "identity",
            "status": "pass_evidence_missing",
            "priority": "p0",
            "reason": (
                "The ledger has no identity pass row. Existing identity rows "
                f"include fail={counts['fail']} and blocked={counts['blocked']}."
            ),
            "required_evidence": [
                "timestamped physical object identity ground truth",
                "clear-visible / occluded / reappeared frames",
                "view, lighting and camera-pose condition metadata",
                "RGB and per-frame Gaussian evidence files",
                "candidate ObjectState artifact bound to the capture manifest",
                "reconstruction-noise robustness evidence",
            ],
            "minimum_metrics": [
                "idf1",
                "fragmentation_rate",
                "swap_rate",
                "identity_collapse=false",
                "retrieval_recall_at_1",
                "long_term_drift_rate",
                "reconstruction_noise_robustness",
            ],
            "recommended_route": "controlled_real_identity_handoff",
            "commands": [
                "uv run objgauss object-state audit-controlled-capture-environment --summary-output outputs/captures/controlled-capture-environment.json",
                "uv run objgauss object-state init-controlled-capture-bundle outputs/captures/controlled-tabletop-cup-box-001 --sample-id controlled-tabletop-cup-box-001 --object-category cup_box --capture-device local-camera --object cup-001:cup:\"blue cup\" --object box-001:box:\"red box\"",
                "uv run objgauss object-state audit-controlled-capture-bundle-readiness outputs/captures/controlled-tabletop-cup-box-001 --summary-output outputs/captures/controlled-tabletop-cup-box-001/readiness-summary.json",
                "uv run objgauss object-state controlled-identity-bundle-handoff outputs/captures/controlled-tabletop-cup-box-001 outputs/captures/controlled-tabletop-cup-box-001/objectstates.json --output-dir outputs/captures/controlled-tabletop-cup-box-001/identity-handoff --hash-files --require-pass",
            ],
            "claim_boundary": common_boundary
            + [
                "A reviewable identity package is not a pass unless its identity row status is pass.",
            ],
            "gate_status": gate_status,
        }
    if kind == "prediction":
        return {
            "evidence_kind": "prediction",
            "status": "pass_evidence_missing",
            "priority": "p1",
            "reason": (
                "The ledger has no prediction pass row. Existing prediction rows "
                f"include fail={counts['fail']} and blocked={counts['blocked']}."
            ),
            "required_evidence": [
                "timestamped pose ground truth",
                "ObjectState future-pose candidate predictions",
                "history or no-state baseline predictions",
                "shared capture manifest binding for GT and candidates",
            ],
            "minimum_metrics": [
                "state_ade",
                "history_ade",
                "prediction_gap_vs_history_model",
            ],
            "recommended_route": "controlled_or_public_prediction_eval",
            "commands": [
                "uv run objgauss object-state init-controlled-reality-candidates outputs/captures/controlled-tabletop-cup-box-001 --output-dir outputs/captures/controlled-tabletop-cup-box-001/reality-candidates --candidate-id controlled-tabletop-cup-box-001-candidate-v1 --candidate-source \"external objectstate predictor\" --artifact-ref outputs/captures/controlled-tabletop-cup-box-001/objectstates.json --summary-output outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/template-summary.json",
                "uv run objgauss object-state finalize-controlled-reality-candidates outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/prediction-candidates.template.json outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/intervention-candidates.template.json --output-dir outputs/captures/controlled-tabletop-cup-box-001/reality-candidates --bundle-root outputs/captures/controlled-tabletop-cup-box-001 --summary-output outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/finalize-summary.json",
            ],
            "claim_boundary": common_boundary
            + [
                "Prediction sufficiency is only comparative: ObjectState must be measured against the declared history baseline.",
            ],
            "gate_status": gate_status,
        }
    if kind == "intervention":
        return {
            "evidence_kind": "intervention",
            "status": "pass_evidence_missing",
            "priority": "p0",
            "reason": (
                "The ledger has no intervention pass row. Existing intervention "
                f"rows include fail={counts['fail']} and blocked={counts['blocked']}."
            ),
            "required_evidence": [
                "timestamped pose ground truth",
                "timestamped action ground truth",
                "action-conditioned candidate predictions",
                "no-action baseline predictions",
                "counterfactual outcome labels or measurable outcome rule",
            ],
            "minimum_metrics": [
                "action_conditioned_ade",
                "counterfactual_outcome_accuracy",
                "wrong_direction_rate",
            ],
            "recommended_route": "controlled_reality_bundle_handoff",
            "commands": [
                "uv run objgauss object-state audit-controlled-reality-bundle-readiness outputs/captures/controlled-tabletop-cup-box-001 outputs/captures/controlled-tabletop-cup-box-001/objectstates.json outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/prediction-candidates.json outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/intervention-candidates.json --summary-output outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/full-readiness-summary.json --require-ready",
                "uv run objgauss object-state controlled-reality-bundle-handoff outputs/captures/controlled-tabletop-cup-box-001 outputs/captures/controlled-tabletop-cup-box-001/objectstates.json outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/prediction-candidates.json outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/intervention-candidates.json --output-dir outputs/captures/controlled-tabletop-cup-box-001/reality-handoff --hash-files --require-pass",
            ],
            "claim_boundary": common_boundary
            + [
                "BOP pose replay cannot satisfy intervention evidence without action GT and counterfactual outcome evidence.",
            ],
            "gate_status": gate_status,
        }
    raise ValueError(f"unsupported ObjectState reality evidence kind: {kind}")


def _next_actions_markdown(actions: Sequence[Mapping[str, Any]]) -> str:
    if not actions:
        return "# ObjectState Reality Row Ledger Next Actions\n\nNo missing pass evidence kinds.\n"
    lines = [
        "# ObjectState Reality Row Ledger Next Actions",
        "",
        "| evidence_kind | priority | recommended_route | reason |",
        "| --- | --- | --- | --- |",
    ]
    for action in actions:
        reason = str(action.get("reason", "")).replace("|", "\\|")
        lines.append(
            "| "
            + str(action.get("evidence_kind", ""))
            + " | "
            + str(action.get("priority", ""))
            + " | "
            + str(action.get("recommended_route", ""))
            + " | "
            + reason
            + " |"
        )
    lines.extend(["", "## Commands", ""])
    for action in actions:
        lines.append(f"### {action['evidence_kind']}")
        lines.append("")
        for command in action.get("commands", ()):
            lines.append("```bash")
            lines.append(str(command))
            lines.append("```")
            lines.append("")
    lines.extend(
        [
            "## Claim Boundary",
            "",
            "- This ledger is read-only.",
            "- Missing pass evidence must be filled by real controlled/public rows.",
            "- It must not be used to claim a world-model pass.",
            "",
        ]
    )
    return "\n".join(lines)


def _sample_scope(rows: Sequence[ObjectStateRealityRow]) -> dict[str, Any]:
    sample_ids = sorted({row.sample_id for row in rows})
    return {
        "sample_ids": sample_ids,
        "sample_count": len(sample_ids),
        "object_categories": sorted({row.object_category for row in rows}),
        "scenarios": sorted({row.scenario for row in rows}),
    }


def _issues(
    records: Sequence[Mapping[str, Any]],
    ledger_gates: Mapping[str, bool],
    gate: Mapping[str, Any] | None,
    duplicate_row_ids: Sequence[str],
) -> list[str]:
    issues: list[str] = []
    for record in records:
        for issue in record.get("issues", ()):
            issues.append(f"{record['path']}: {issue}")
    if duplicate_row_ids:
        issues.append("duplicate row ids: " + ", ".join(duplicate_row_ids))
    if not ledger_gates["summaries_present"]:
        issues.append("no ObjectState reality row summaries were provided")
    if gate is not None and gate.get("status") != "objectstate_reality_gate_pass":
        blockers = ", ".join(str(item) for item in gate.get("hard_blockers", ()))
        issues.append(f"full ObjectState reality gate did not pass: {blockers}")
    return _dedupe(issues)


def _row_status_count(rows: Sequence[ObjectStateRealityRow], status: str) -> int:
    return sum(1 for row in rows if row.status == status)


def _counts(values: Sequence[str] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        text = str(value)
        counts[text] = counts.get(text, 0) + 1
    return counts


def _duplicates(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            duplicates.add(text)
        seen.add(text)
    return sorted(duplicates)


def _dedupe(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped
