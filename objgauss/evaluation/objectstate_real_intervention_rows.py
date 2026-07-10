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

OBJECTSTATE_REAL_INTERVENTION_ROWS_SCHEMA = (
    "objgauss-objectstate-real-intervention-rows-v1"
)

_INTERVENTION_ACCOUNTING_STATUSES = {
    "pass",
    "fail",
    "evidence_incomplete",
    "unsupported",
}
_BLOCKED_ACCOUNTING_STATUSES = {"evidence_incomplete", "unsupported"}
_ACCOUNTING_STATUSES_REQUIRING_ACTION_TRANSITION = {"pass", "fail"}
_REQUIRED_INTERVENTION_METRICS = (
    "action_conditioned_ade",
    "no_action_ade",
    "counterfactual_outcome_accuracy",
    "wrong_direction_rate",
    "identity_consistency_rate",
)

__all__ = (
    "OBJECTSTATE_REAL_INTERVENTION_ROWS_SCHEMA",
    "read_objectstate_real_intervention_rows_summary",
    "objectstate_real_intervention_rows_summary",
    "objectstate_real_intervention_rows_from_bundle",
    "validate_objectstate_real_intervention_rows_summary",
)


def read_objectstate_real_intervention_rows_summary(path: str | Path) -> dict[str, Any]:
    bundle = read_objectstate_real_evidence_bundle(path)
    return objectstate_real_intervention_rows_summary(bundle)


def objectstate_real_intervention_rows_summary(
    bundle: Mapping[str, Any],
    *,
    synthetic_smoke_passed: bool = True,
    min_real_or_public_rows: int = 1,
) -> dict[str, Any]:
    checked = validate_objectstate_real_evidence_bundle(bundle)
    rows = objectstate_real_intervention_rows_from_bundle(checked)
    thresholds = ObjectStateRealityGateThresholds(
        min_real_or_public_rows=int(min_real_or_public_rows),
        require_identity_pass_row=False,
        require_prediction_pass_row=False,
        require_intervention_pass_row=True,
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
    action_transition_pairs = _referenced_intervention_pairs(checked)
    payload = {
        "schema": OBJECTSTATE_REAL_INTERVENTION_ROWS_SCHEMA,
        "kind": "objectstate_real_intervention_rows",
        "status": status,
        "source_bundle_schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
        "reality_gate_schema": OBJECTSTATE_REALITY_GATE_SCHEMA,
        "reality_row_schema": OBJECTSTATE_REALITY_ROW_SCHEMA,
        "sample": dict(checked["sample"]),
        "row_counts": {
            "intervention_rows": len(rows),
            "intervention_pass_rows": sum(1 for row in rows if row.status == "pass"),
            "intervention_fail_rows": sum(1 for row in rows if row.status == "fail"),
            "intervention_blocked_rows": sum(
                1 for row in rows if row.status == "blocked"
            ),
            "action_interval_rows": len(checked["action_interval_rows"]),
            "state_transition_rows": len(checked["state_transition_rows"]),
            "referenced_action_transition_pairs": len(action_transition_pairs),
            "identity_link_rows": len(checked["identity_link_rows"]),
        },
        "metrics": _intervention_bundle_metrics(checked, rows),
        "intervention_rows": [row.as_dict() for row in rows],
        "intervention_gate": gate,
        "blocked_rows_markdown": _blocked_rows_markdown(rows),
        "hard_blockers": _summary_hard_blockers(rows, gate_hard_blockers),
        "claim_policy": {
            "intervention_rows_enter_pass_fail_accounting": True,
            "evidence_incomplete_is_mapped_to_blocked": True,
            "unsupported_is_mapped_to_blocked": True,
            "action_transition_overlap_required": True,
            "identity_stability_required_across_transition": True,
            "identity_rows_out_of_scope": True,
            "prediction_rows_out_of_scope": True,
            "does_not_claim_reality_gate_full_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "reconstructs_gaussians": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "runs_intervention_model": False,
            "runs_identity_eval": False,
            "runs_prediction_eval": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_real_intervention_rows_summary(payload)


def objectstate_real_intervention_rows_from_bundle(
    bundle: Mapping[str, Any],
) -> tuple[ObjectStateRealityRow, ...]:
    checked = validate_objectstate_real_evidence_bundle(bundle)
    sample = checked["sample"]
    rows = []
    for accounting in checked["gate_accounting_rows"]:
        if accounting["evidence_kind"] != "intervention":
            continue
        rows.append(_intervention_reality_row(sample, accounting, checked))
    return tuple(validate_objectstate_reality_row(row) for row in rows)


def validate_objectstate_real_intervention_rows_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("real intervention rows summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_REAL_INTERVENTION_ROWS_SCHEMA:
        raise ValueError(
            f"unsupported real intervention rows schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_real_intervention_rows":
        raise ValueError("real intervention rows kind is unsupported")
    if payload.get("status") not in {
        "objectstate_real_intervention_rows_pass",
        "objectstate_real_intervention_rows_fail",
        "objectstate_real_intervention_rows_incomplete",
    }:
        raise ValueError("real intervention rows status is unsupported")
    if payload.get("source_bundle_schema") != OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA:
        raise ValueError("real intervention rows source bundle schema is unsupported")
    if payload.get("reality_gate_schema") != OBJECTSTATE_REALITY_GATE_SCHEMA:
        raise ValueError("real intervention rows reality gate schema is unsupported")
    if payload.get("reality_row_schema") != OBJECTSTATE_REALITY_ROW_SCHEMA:
        raise ValueError("real intervention rows reality row schema is unsupported")
    sample = payload.get("sample")
    if not isinstance(sample, Mapping) or not sample.get("sample_id"):
        raise ValueError("real intervention rows summary requires sample")
    rows = payload.get("intervention_rows")
    if not isinstance(rows, list):
        raise ValueError("real intervention rows summary requires intervention_rows")
    for row in rows:
        _validate_reality_row_payload(row)
    counts = payload.get("row_counts")
    if not isinstance(counts, Mapping):
        raise ValueError("real intervention rows summary requires row_counts")
    for key in (
        "intervention_rows",
        "intervention_pass_rows",
        "intervention_fail_rows",
        "intervention_blocked_rows",
        "action_interval_rows",
        "state_transition_rows",
        "referenced_action_transition_pairs",
        "identity_link_rows",
    ):
        _non_negative_int(counts.get(key), f"row_counts.{key}")
    if counts["intervention_rows"] != len(rows):
        raise ValueError("intervention row count must match intervention_rows")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("real intervention rows summary requires metrics")
    for key in (
        "action_transition_coverage_rate",
        "intervention_accounting_status_counts",
        "reality_status_counts",
        "mean_action_conditioned_ade",
        "mean_no_action_ade",
        "mean_intervention_gain",
        "mean_counterfactual_outcome_accuracy",
        "mean_wrong_direction_rate",
        "mean_identity_consistency_rate",
    ):
        if key not in metrics:
            raise ValueError(f"real intervention rows metrics missing {key}")
    _finite_fraction(
        metrics["action_transition_coverage_rate"],
        "action_transition_coverage_rate",
    )
    gate = payload.get("intervention_gate")
    if gate is not None:
        validate_objectstate_reality_gate_summary(gate)
        if rows != gate.get("rows"):
            raise ValueError("intervention_rows must match gate-derived rows")
    if not isinstance(payload.get("blocked_rows_markdown"), str):
        raise ValueError("real intervention rows summary requires blocked_rows_markdown")
    blockers = payload.get("hard_blockers")
    if not isinstance(blockers, list) or any(
        not isinstance(item, str) or not item for item in blockers
    ):
        raise ValueError("real intervention rows summary requires hard_blockers")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("intervention_rows_enter_pass_fail_accounting")
        or not claim_policy.get("evidence_incomplete_is_mapped_to_blocked")
        or not claim_policy.get("unsupported_is_mapped_to_blocked")
        or not claim_policy.get("action_transition_overlap_required")
        or not claim_policy.get("identity_stability_required_across_transition")
        or not claim_policy.get("identity_rows_out_of_scope")
        or not claim_policy.get("prediction_rows_out_of_scope")
        or not claim_policy.get("does_not_claim_reality_gate_full_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("real intervention rows summary must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("reconstructs_gaussians")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("runs_intervention_model")
        or non_goals.get("runs_identity_eval")
        or non_goals.get("runs_prediction_eval")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError(
            "real intervention rows summary cannot claim capture, GT creation, "
            "reconstruction, training, eval, replay, diffusion or viewer mutation"
        )
    return dict(payload)


def _intervention_reality_row(
    sample: Mapping[str, Any],
    accounting: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> ObjectStateRealityRow:
    accounting_status = str(accounting["accounting_status"])
    if accounting_status not in _INTERVENTION_ACCOUNTING_STATUSES:
        raise ValueError(
            f"unsupported intervention accounting status: {accounting_status}"
        )
    action, transition = _referenced_action_transition(accounting, bundle)
    gt = accounting["gt_requirements"]
    metrics = _intervention_metrics(accounting)
    if action is not None and transition is not None:
        _require_identity_stability(transition, bundle)
        metrics.update(_action_transition_scope_metrics(action, transition))
    status = _reality_status(accounting_status)
    return ObjectStateRealityRow(
        row_id=str(accounting["row_id"]),
        sample_id=str(sample["sample_id"]),
        source_kind=str(sample["source_kind"]),
        evidence_kind="intervention",
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


def _referenced_action_transition(
    accounting: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    action_id = accounting.get("action_id")
    transition_id = accounting.get("transition_id")
    status = str(accounting["accounting_status"])
    requires_refs = status in _ACCOUNTING_STATUSES_REQUIRING_ACTION_TRANSITION
    if requires_refs and not action_id:
        raise ValueError("intervention pass/fail accounting rows require action_id")
    if requires_refs and not transition_id:
        raise ValueError("intervention pass/fail accounting rows require transition_id")
    if not action_id or not transition_id:
        return None, None
    actions = {row["action_id"]: row for row in bundle["action_interval_rows"]}
    transitions = {
        row["transition_id"]: row for row in bundle["state_transition_rows"]
    }
    action = actions.get(str(action_id))
    transition = transitions.get(str(transition_id))
    if action is None:
        if requires_refs:
            raise ValueError(
                f"intervention accounting references unknown action_id: {action_id}"
            )
        return None, transition
    if transition is None:
        if requires_refs:
            raise ValueError(
                "intervention accounting references unknown transition_id: "
                f"{transition_id}"
            )
        return action, None
    if not _action_references_transition_object(action, transition):
        raise ValueError(
            "intervention accounting action object must match referenced transition object"
        )
    if not _intervals_overlap(
        float(action["action_start_ts"]),
        float(action["action_end_ts"]),
        float(transition["source_timestamp"]),
        float(transition["target_timestamp"]),
    ):
        raise ValueError(
            "intervention accounting action interval must overlap state transition"
        )
    return action, transition


def _intervention_metrics(accounting: Mapping[str, Any]) -> dict[str, float | bool]:
    metrics = dict(accounting["metrics"])
    status = str(accounting["accounting_status"])
    if status in _ACCOUNTING_STATUSES_REQUIRING_ACTION_TRANSITION:
        for key in _REQUIRED_INTERVENTION_METRICS:
            if key not in metrics:
                raise ValueError(f"intervention accounting row missing metric {key}")
    action_ade = metrics.get("action_conditioned_ade")
    no_action_ade = metrics.get("no_action_ade")
    if (
        "intervention_gain" not in metrics
        and _is_number(no_action_ade)
        and _is_number(action_ade)
    ):
        metrics["intervention_gain"] = float(no_action_ade) - float(action_ade)
    return metrics


def _require_identity_stability(
    transition: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> None:
    source_identity = _physical_identity_for_pose(
        transition["object_id"],
        transition["source_frame_id"],
        bundle["identity_link_rows"],
    )
    target_identity = _physical_identity_for_pose(
        transition["object_id"],
        transition["target_frame_id"],
        bundle["identity_link_rows"],
    )
    if source_identity is None or target_identity is None:
        raise ValueError("intervention accounting requires identity links across transition")
    if source_identity != target_identity:
        raise ValueError("intervention accounting requires stable identity across transition")


def _physical_identity_for_pose(
    object_id: str,
    frame_id: str,
    identity_links: Sequence[Mapping[str, Any]],
) -> str | None:
    matches = [
        row["physical_identity_id"]
        for row in identity_links
        if row["object_id"] == object_id and row["frame_id"] == frame_id
    ]
    if not matches:
        return None
    unique = set(matches)
    if len(unique) != 1:
        raise ValueError("multiple physical identities for object/frame")
    return matches[0]


def _action_transition_scope_metrics(
    action: Mapping[str, Any],
    transition: Mapping[str, Any],
) -> dict[str, float]:
    return {
        "referenced_action_transition_count": 1.0,
        "action_duration_s": float(action["action_end_ts"]) - float(action["action_start_ts"]),
        "transition_duration_s": float(transition["target_timestamp"])
        - float(transition["source_timestamp"]),
    }


def _intervention_bundle_metrics(
    bundle: Mapping[str, Any],
    rows: Sequence[ObjectStateRealityRow],
) -> dict[str, Any]:
    action_count = len(bundle["action_interval_rows"])
    pairs = _referenced_intervention_pairs(bundle)
    return {
        "action_transition_coverage_rate": (
            0.0 if action_count == 0 else float(len({item[0] for item in pairs}) / action_count)
        ),
        "intervention_accounting_status_counts": _counts(
            row["accounting_status"]
            for row in bundle["gate_accounting_rows"]
            if row["evidence_kind"] == "intervention"
        ),
        "reality_status_counts": _counts(row.status for row in rows),
        "mean_action_conditioned_ade": _mean_metric(rows, "action_conditioned_ade"),
        "mean_no_action_ade": _mean_metric(rows, "no_action_ade"),
        "mean_intervention_gain": _mean_metric(rows, "intervention_gain"),
        "mean_counterfactual_outcome_accuracy": _mean_metric(
            rows,
            "counterfactual_outcome_accuracy",
        ),
        "mean_wrong_direction_rate": _mean_metric(rows, "wrong_direction_rate"),
        "mean_identity_consistency_rate": _mean_metric(
            rows,
            "identity_consistency_rate",
        ),
    }


def _referenced_intervention_pairs(bundle: Mapping[str, Any]) -> set[tuple[str, str]]:
    action_ids = {row["action_id"] for row in bundle["action_interval_rows"]}
    transition_ids = {row["transition_id"] for row in bundle["state_transition_rows"]}
    referenced = set()
    for row in bundle["gate_accounting_rows"]:
        if row["evidence_kind"] != "intervention":
            continue
        if row["accounting_status"] not in _ACCOUNTING_STATUSES_REQUIRING_ACTION_TRANSITION:
            continue
        action_id = row.get("action_id")
        transition_id = row.get("transition_id")
        if (
            isinstance(action_id, str)
            and isinstance(transition_id, str)
            and action_id in action_ids
            and transition_id in transition_ids
        ):
            referenced.add((action_id, transition_id))
    return referenced


def _summary_status(rows: Sequence[ObjectStateRealityRow], gate_status: str) -> str:
    if not rows:
        return "objectstate_real_intervention_rows_incomplete"
    if all(row.status == "blocked" for row in rows):
        return "objectstate_real_intervention_rows_incomplete"
    if any(row.status == "fail" for row in rows):
        return "objectstate_real_intervention_rows_fail"
    if gate_status == "objectstate_reality_gate_pass":
        return "objectstate_real_intervention_rows_pass"
    return "objectstate_real_intervention_rows_fail"


def _summary_hard_blockers(
    rows: Sequence[ObjectStateRealityRow],
    gate_hard_blockers: Sequence[str],
) -> list[str]:
    blockers = list(gate_hard_blockers)
    if not rows:
        blockers.append("missing intervention accounting rows")
    return blockers


def _reality_status(accounting_status: str) -> str:
    if accounting_status in {"pass", "fail"}:
        return accounting_status
    if accounting_status in _BLOCKED_ACCOUNTING_STATUSES:
        return "blocked"
    raise ValueError(f"unsupported intervention accounting status: {accounting_status}")


def _block_reason(accounting: Mapping[str, Any]) -> str:
    status = accounting["accounting_status"]
    reason = str(accounting.get("reason") or "").strip()
    if reason:
        return f"{status}: {reason}"
    return f"{status}: intervention evidence cannot enter pass/fail accounting"


def _failure_reason(accounting: Mapping[str, Any]) -> str:
    reason = str(accounting.get("reason") or "").strip()
    return reason or "intervention accounting row reported fail"


def _blocked_rows_markdown(rows: Sequence[ObjectStateRealityRow]) -> str:
    blocked = [row for row in rows if row.status == "blocked"]
    if not blocked:
        return "No blocked real intervention rows.\n"
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
        raise ValueError("intervention row payload must be a mapping")
    ground_truth = row.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        raise ValueError("intervention row payload requires ground_truth")
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


def _action_references_transition_object(
    action: Mapping[str, Any],
    transition: Mapping[str, Any],
) -> bool:
    referenced = {str(action["object_id"])}
    if action.get("target_object_id"):
        referenced.add(str(action["target_object_id"]))
    return str(transition["object_id"]) in referenced


def _intervals_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> bool:
    return start_a <= end_b and end_a >= start_b


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
