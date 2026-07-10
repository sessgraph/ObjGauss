from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.datasets.objectstate_real_evidence_bundle import (
    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
    read_objectstate_real_evidence_bundle,
    validate_objectstate_real_evidence_bundle,
)
from objgauss.evaluation.objectstate_reality_gate import (
    OBJECTSTATE_REALITY_GATE_SCHEMA,
    OBJECTSTATE_REALITY_ROW_SCHEMA,
    ObjectStateRealityGateThresholds,
    ObjectStateRealityRow,
    evaluate_objectstate_reality_gate,
    validate_objectstate_reality_gate_summary,
    validate_objectstate_reality_row,
)

OBJECTSTATE_REAL_PREDICTION_ROWS_SCHEMA = (
    "objgauss-objectstate-real-prediction-rows-v1"
)

_PREDICTION_ACCOUNTING_STATUSES = {
    "pass",
    "fail",
    "evidence_incomplete",
    "unsupported",
}
_BLOCKED_ACCOUNTING_STATUSES = {"evidence_incomplete", "unsupported"}
_ACCOUNTING_STATUSES_REQUIRING_TRANSITION = {"pass", "fail"}

__all__ = (
    "OBJECTSTATE_REAL_PREDICTION_ROWS_SCHEMA",
    "read_objectstate_real_prediction_rows_summary",
    "objectstate_real_prediction_rows_summary",
    "objectstate_real_prediction_rows_from_bundle",
    "validate_objectstate_real_prediction_rows_summary",
)


def read_objectstate_real_prediction_rows_summary(path: str | Path) -> dict[str, Any]:
    bundle = read_objectstate_real_evidence_bundle(path)
    return objectstate_real_prediction_rows_summary(bundle)


def objectstate_real_prediction_rows_summary(
    bundle: Mapping[str, Any],
    *,
    synthetic_smoke_passed: bool = True,
    min_real_or_public_rows: int = 1,
) -> dict[str, Any]:
    checked = validate_objectstate_real_evidence_bundle(bundle)
    rows = objectstate_real_prediction_rows_from_bundle(checked)
    thresholds = ObjectStateRealityGateThresholds(
        min_real_or_public_rows=int(min_real_or_public_rows),
        require_identity_pass_row=False,
        require_prediction_pass_row=True,
        require_intervention_pass_row=False,
        fail_on_failed_rows=True,
    )
    gate = None
    gate_status = "not_run"
    gate_hard_blockers: list[str] = []
    if rows:
        report = evaluate_objectstate_reality_gate(
            rows,
            synthetic_smoke_passed=bool(synthetic_smoke_passed),
            thresholds=thresholds,
        )
        rows = report.rows
        gate = report.as_dict()
        gate_status = gate["status"]
        gate_hard_blockers = list(gate["hard_blockers"])
    status = _summary_status(rows, gate_status)
    payload = {
        "schema": OBJECTSTATE_REAL_PREDICTION_ROWS_SCHEMA,
        "kind": "objectstate_real_prediction_rows",
        "status": status,
        "source_bundle_schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
        "reality_gate_schema": OBJECTSTATE_REALITY_GATE_SCHEMA,
        "reality_row_schema": OBJECTSTATE_REALITY_ROW_SCHEMA,
        "sample": dict(checked["sample"]),
        "row_counts": {
            "prediction_rows": len(rows),
            "prediction_pass_rows": sum(1 for row in rows if row.status == "pass"),
            "prediction_fail_rows": sum(1 for row in rows if row.status == "fail"),
            "prediction_blocked_rows": sum(
                1 for row in rows if row.status == "blocked"
            ),
            "state_transition_rows": len(checked["state_transition_rows"]),
            "referenced_transition_rows": len(_referenced_prediction_transitions(checked)),
            "object_pose_rows": len(checked["object_pose_rows"]),
            "observation_rows": len(checked["observation_rows"]),
        },
        "metrics": _prediction_bundle_metrics(checked, rows),
        "prediction_rows": [row.as_dict() for row in rows],
        "prediction_gate": gate,
        "blocked_rows_markdown": _blocked_rows_markdown(rows),
        "hard_blockers": _summary_hard_blockers(rows, gate_hard_blockers),
        "claim_policy": {
            "prediction_rows_enter_pass_fail_accounting": True,
            "evidence_incomplete_is_mapped_to_blocked": True,
            "unsupported_is_mapped_to_blocked": True,
            "history_baseline_comparison_required": True,
            "identity_rows_out_of_scope": True,
            "intervention_rows_out_of_scope": True,
            "does_not_claim_reality_gate_full_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "reconstructs_gaussians": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "runs_prediction_model": False,
            "runs_identity_eval": False,
            "runs_intervention_eval": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_real_prediction_rows_summary(payload)


def objectstate_real_prediction_rows_from_bundle(
    bundle: Mapping[str, Any],
) -> tuple[ObjectStateRealityRow, ...]:
    checked = validate_objectstate_real_evidence_bundle(bundle)
    sample = checked["sample"]
    rows = []
    for accounting in checked["gate_accounting_rows"]:
        if accounting["evidence_kind"] != "prediction":
            continue
        rows.append(_prediction_reality_row(sample, accounting, checked))
    return tuple(validate_objectstate_reality_row(row) for row in rows)


def validate_objectstate_real_prediction_rows_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("real prediction rows summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_REAL_PREDICTION_ROWS_SCHEMA:
        raise ValueError(
            f"unsupported real prediction rows schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_real_prediction_rows":
        raise ValueError("real prediction rows kind is unsupported")
    if payload.get("status") not in {
        "objectstate_real_prediction_rows_pass",
        "objectstate_real_prediction_rows_fail",
        "objectstate_real_prediction_rows_incomplete",
    }:
        raise ValueError("real prediction rows status is unsupported")
    if payload.get("source_bundle_schema") != OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA:
        raise ValueError("real prediction rows source bundle schema is unsupported")
    if payload.get("reality_gate_schema") != OBJECTSTATE_REALITY_GATE_SCHEMA:
        raise ValueError("real prediction rows reality gate schema is unsupported")
    if payload.get("reality_row_schema") != OBJECTSTATE_REALITY_ROW_SCHEMA:
        raise ValueError("real prediction rows reality row schema is unsupported")
    sample = payload.get("sample")
    if not isinstance(sample, Mapping) or not sample.get("sample_id"):
        raise ValueError("real prediction rows summary requires sample")
    rows = payload.get("prediction_rows")
    if not isinstance(rows, list):
        raise ValueError("real prediction rows summary requires prediction_rows")
    for row in rows:
        _validate_reality_row_payload(row)
    counts = payload.get("row_counts")
    if not isinstance(counts, Mapping):
        raise ValueError("real prediction rows summary requires row_counts")
    for key in (
        "prediction_rows",
        "prediction_pass_rows",
        "prediction_fail_rows",
        "prediction_blocked_rows",
        "state_transition_rows",
        "referenced_transition_rows",
        "object_pose_rows",
        "observation_rows",
    ):
        _non_negative_int(counts.get(key), f"row_counts.{key}")
    if counts["prediction_rows"] != len(rows):
        raise ValueError("prediction row count must match prediction_rows")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("real prediction rows summary requires metrics")
    for key in (
        "pose_transition_coverage",
        "prediction_accounting_status_counts",
        "reality_status_counts",
        "mean_state_ade",
        "mean_history_ade",
        "mean_state_vs_history_error_ratio",
        "mean_prediction_gap_vs_history_model",
    ):
        if key not in metrics:
            raise ValueError(f"real prediction rows metrics missing {key}")
    _finite_fraction(metrics["pose_transition_coverage"], "pose_transition_coverage")
    gate = payload.get("prediction_gate")
    if gate is not None:
        validate_objectstate_reality_gate_summary(gate)
        if rows != gate.get("rows"):
            raise ValueError("prediction_rows must match gate-derived rows")
    if not isinstance(payload.get("blocked_rows_markdown"), str):
        raise ValueError("real prediction rows summary requires blocked_rows_markdown")
    blockers = payload.get("hard_blockers")
    if not isinstance(blockers, list) or any(
        not isinstance(item, str) or not item for item in blockers
    ):
        raise ValueError("real prediction rows summary requires hard_blockers")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("prediction_rows_enter_pass_fail_accounting")
        or not claim_policy.get("evidence_incomplete_is_mapped_to_blocked")
        or not claim_policy.get("unsupported_is_mapped_to_blocked")
        or not claim_policy.get("history_baseline_comparison_required")
        or not claim_policy.get("identity_rows_out_of_scope")
        or not claim_policy.get("intervention_rows_out_of_scope")
        or not claim_policy.get("does_not_claim_reality_gate_full_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("real prediction rows summary must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("reconstructs_gaussians")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("runs_prediction_model")
        or non_goals.get("runs_identity_eval")
        or non_goals.get("runs_intervention_eval")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError(
            "real prediction rows summary cannot claim capture, GT creation, "
            "reconstruction, training, eval, replay, diffusion or viewer mutation"
        )
    return dict(payload)


def _prediction_reality_row(
    sample: Mapping[str, Any],
    accounting: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> ObjectStateRealityRow:
    accounting_status = str(accounting["accounting_status"])
    if accounting_status not in _PREDICTION_ACCOUNTING_STATUSES:
        raise ValueError(
            f"unsupported prediction accounting status: {accounting_status}"
        )
    transition = _referenced_transition(accounting, bundle)
    gt = accounting["gt_requirements"]
    metrics = _prediction_metrics(accounting)
    if transition is not None:
        metrics.update(_transition_scope_metrics(transition))
    status = _reality_status(accounting_status)
    return ObjectStateRealityRow(
        row_id=str(accounting["row_id"]),
        sample_id=str(sample["sample_id"]),
        source_kind=str(sample["source_kind"]),
        evidence_kind="prediction",
        status=status,
        object_category=str(sample["object_category"]),
        scenario=str(sample["scenario"]),
        observation_modalities=tuple(sample["observation_modalities"]),
        artifact_refs=tuple(accounting["artifact_refs"] or sample["artifact_refs"]),
        metrics=metrics,
        has_identity_gt=bool(gt["identity"]),
        has_pose_gt=bool(gt["pose"]),
        has_action_gt=bool(gt["action"]),
        has_timestamp=bool(gt["timestamp"]),
        license=str(sample["license"]),
        block_reason=_block_reason(accounting) if status == "blocked" else None,
        failure_reason=_failure_reason(accounting) if status == "fail" else None,
    )


def _referenced_transition(
    accounting: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    transition_id = accounting.get("transition_id")
    status = str(accounting["accounting_status"])
    if status in _ACCOUNTING_STATUSES_REQUIRING_TRANSITION and not transition_id:
        raise ValueError("prediction pass/fail accounting rows require transition_id")
    if not transition_id:
        return None
    transitions = {
        row["transition_id"]: row for row in bundle["state_transition_rows"]
    }
    transition = transitions.get(str(transition_id))
    if transition is None:
        if status in _ACCOUNTING_STATUSES_REQUIRING_TRANSITION:
            raise ValueError(
                f"prediction accounting references unknown transition_id: {transition_id}"
            )
        return None
    object_id = accounting.get("object_id")
    if object_id is not None and str(object_id) != str(transition["object_id"]):
        raise ValueError(
            "prediction accounting object_id must match referenced transition object_id"
        )
    return transition


def _prediction_metrics(accounting: Mapping[str, Any]) -> dict[str, float | bool]:
    metrics = dict(accounting["metrics"])
    state_ade = metrics.get("state_ade")
    history_ade = metrics.get("history_ade")
    if (
        "prediction_gap_vs_history_model" not in metrics
        and _is_number(state_ade)
        and _is_number(history_ade)
    ):
        metrics["prediction_gap_vs_history_model"] = float(state_ade) - float(history_ade)
    if (
        "state_vs_history_error_ratio" not in metrics
        and _is_number(state_ade)
        and _is_number(history_ade)
        and abs(float(history_ade)) > 0.0
    ):
        metrics["state_vs_history_error_ratio"] = float(state_ade) / float(history_ade)
    return metrics


def _transition_scope_metrics(transition: Mapping[str, Any]) -> dict[str, float]:
    duration = float(transition["target_timestamp"]) - float(
        transition["source_timestamp"]
    )
    return {
        "referenced_transition_count": 1.0,
        "transition_duration_s": duration,
    }


def _prediction_bundle_metrics(
    bundle: Mapping[str, Any],
    rows: Sequence[ObjectStateRealityRow],
) -> dict[str, Any]:
    transition_count = len(bundle["state_transition_rows"])
    referenced = _referenced_prediction_transitions(bundle)
    return {
        "pose_transition_coverage": (
            0.0 if transition_count == 0 else float(len(referenced) / transition_count)
        ),
        "prediction_accounting_status_counts": _counts(
            row["accounting_status"]
            for row in bundle["gate_accounting_rows"]
            if row["evidence_kind"] == "prediction"
        ),
        "reality_status_counts": _counts(row.status for row in rows),
        "mean_state_ade": _mean_metric(rows, "state_ade"),
        "mean_history_ade": _mean_metric(rows, "history_ade"),
        "mean_state_vs_history_error_ratio": _mean_metric(
            rows,
            "state_vs_history_error_ratio",
        ),
        "mean_prediction_gap_vs_history_model": _mean_metric(
            rows,
            "prediction_gap_vs_history_model",
        ),
    }


def _referenced_prediction_transitions(bundle: Mapping[str, Any]) -> set[str]:
    transition_ids = {row["transition_id"] for row in bundle["state_transition_rows"]}
    referenced = set()
    for row in bundle["gate_accounting_rows"]:
        if row["evidence_kind"] != "prediction":
            continue
        if row["accounting_status"] not in _ACCOUNTING_STATUSES_REQUIRING_TRANSITION:
            continue
        transition_id = row.get("transition_id")
        if isinstance(transition_id, str) and transition_id in transition_ids:
            referenced.add(transition_id)
    return referenced


def _summary_status(rows: Sequence[ObjectStateRealityRow], gate_status: str) -> str:
    if not rows:
        return "objectstate_real_prediction_rows_incomplete"
    if all(row.status == "blocked" for row in rows):
        return "objectstate_real_prediction_rows_incomplete"
    if any(row.status == "fail" for row in rows):
        return "objectstate_real_prediction_rows_fail"
    if gate_status == "objectstate_reality_gate_pass":
        return "objectstate_real_prediction_rows_pass"
    return "objectstate_real_prediction_rows_fail"


def _summary_hard_blockers(
    rows: Sequence[ObjectStateRealityRow],
    gate_hard_blockers: Sequence[str],
) -> list[str]:
    blockers = list(gate_hard_blockers)
    if not rows:
        blockers.append("missing prediction accounting rows")
    return blockers


def _reality_status(accounting_status: str) -> str:
    if accounting_status in {"pass", "fail"}:
        return accounting_status
    if accounting_status in _BLOCKED_ACCOUNTING_STATUSES:
        return "blocked"
    raise ValueError(f"unsupported prediction accounting status: {accounting_status}")


def _block_reason(accounting: Mapping[str, Any]) -> str:
    status = accounting["accounting_status"]
    reason = str(accounting.get("reason") or "").strip()
    if reason:
        return f"{status}: {reason}"
    return f"{status}: prediction evidence cannot enter pass/fail accounting"


def _failure_reason(accounting: Mapping[str, Any]) -> str:
    reason = str(accounting.get("reason") or "").strip()
    return reason or "prediction accounting row reported fail"


def _blocked_rows_markdown(rows: Sequence[ObjectStateRealityRow]) -> str:
    blocked = [row for row in rows if row.status == "blocked"]
    if not blocked:
        return "No blocked real prediction rows.\n"
    lines = [
        "| row_id | sample_id | block_reason | artifact_refs |",
        "| --- | --- | --- | --- |",
    ]
    for row in blocked:
        lines.append(
            "| "
            + " | ".join(
                (
                    row.row_id,
                    row.sample_id,
                    row.block_reason or "",
                    ", ".join(row.artifact_refs),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _validate_reality_row_payload(row: Any) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError("prediction row payload must be a mapping")
    ground_truth = row.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        raise ValueError("prediction row payload requires ground_truth")
    checked = validate_objectstate_reality_row(
        ObjectStateRealityRow(
            row_id=str(row.get("row_id", "")),
            sample_id=str(row.get("sample_id", "")),
            source_kind=str(row.get("source_kind", "")),
            evidence_kind=str(row.get("evidence_kind", "")),
            status=str(row.get("status", "")),
            object_category=str(row.get("object_category", "")),
            scenario=str(row.get("scenario", "")),
            observation_modalities=tuple(row.get("observation_modalities", ())),
            artifact_refs=tuple(row.get("artifact_refs", ())),
            metrics=row.get("metrics", {}),
            has_identity_gt=bool(ground_truth.get("identity")),
            has_pose_gt=bool(ground_truth.get("pose")),
            has_action_gt=bool(ground_truth.get("action")),
            has_timestamp=bool(ground_truth.get("timestamp")),
            license=str(row.get("license", "unknown")),
            block_reason=row.get("block_reason"),
            failure_reason=row.get("failure_reason"),
            schema=str(row.get("schema", "")),
        )
    )
    return checked.as_dict()


def _mean_metric(rows: Sequence[ObjectStateRealityRow], key: str) -> float | None:
    values = []
    for row in rows:
        if row.status == "blocked":
            continue
        value = row.metrics.get(key)
        if _is_number(value):
            values.append(float(value))
    if not values:
        return None
    return float(sum(values) / len(values))


def _counts(values: Sequence[str] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        text = str(value)
        counts[text] = counts.get(text, 0) + 1
    return counts


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return int(value)


def _finite_fraction(value: Any, name: str) -> float:
    if not _is_number(value):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
