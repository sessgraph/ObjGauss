from __future__ import annotations

from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_real_evidence_bundle import (
    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
    OBJECTSTATE_REAL_GATE_ACCOUNTING_STATUSES,
    read_objectstate_real_evidence_bundle,
    validate_objectstate_real_evidence_bundle,
)
from objgauss.core.objectstate_reality_gate import (
    OBJECTSTATE_REALITY_GATE_SCHEMA,
    OBJECTSTATE_REALITY_ROW_SCHEMA,
    ObjectStateRealityGateThresholds,
    ObjectStateRealityRow,
    evaluate_objectstate_reality_gate,
    validate_objectstate_reality_gate_summary,
    validate_objectstate_reality_row,
)

OBJECTSTATE_REAL_IDENTITY_ROWS_SCHEMA = (
    "objgauss-objectstate-real-identity-rows-v1"
)

_IDENTITY_ACCOUNTING_STATUSES = {"pass", "fail", "evidence_incomplete", "unsupported"}
_BLOCKED_ACCOUNTING_STATUSES = {"evidence_incomplete", "unsupported"}


def read_objectstate_real_identity_rows_summary(path: str) -> dict[str, Any]:
    bundle = read_objectstate_real_evidence_bundle(path)
    return objectstate_real_identity_rows_summary(bundle)


def objectstate_real_identity_rows_summary(
    bundle: Mapping[str, Any],
    *,
    synthetic_smoke_passed: bool = True,
    min_real_or_public_rows: int = 1,
) -> dict[str, Any]:
    checked = validate_objectstate_real_evidence_bundle(bundle)
    rows = objectstate_real_identity_rows_from_bundle(checked)
    thresholds = ObjectStateRealityGateThresholds(
        min_real_or_public_rows=int(min_real_or_public_rows),
        require_identity_pass_row=True,
        require_prediction_pass_row=False,
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
        gate = report.as_dict()
        gate_status = gate["status"]
        gate_hard_blockers = list(gate["hard_blockers"])
    status = _summary_status(rows, gate_status)
    payload = {
        "schema": OBJECTSTATE_REAL_IDENTITY_ROWS_SCHEMA,
        "kind": "objectstate_real_identity_rows",
        "status": status,
        "source_bundle_schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
        "reality_gate_schema": OBJECTSTATE_REALITY_GATE_SCHEMA,
        "reality_row_schema": OBJECTSTATE_REALITY_ROW_SCHEMA,
        "sample": dict(checked["sample"]),
        "row_counts": {
            "identity_rows": len(rows),
            "identity_pass_rows": sum(1 for row in rows if row.status == "pass"),
            "identity_fail_rows": sum(1 for row in rows if row.status == "fail"),
            "identity_blocked_rows": sum(
                1 for row in rows if row.status == "blocked"
            ),
            "identity_link_rows": len(checked["identity_link_rows"]),
            "object_pose_rows": len(checked["object_pose_rows"]),
            "observation_rows": len(checked["observation_rows"]),
        },
        "metrics": _identity_bundle_metrics(checked, rows),
        "identity_rows": [row.as_dict() for row in rows],
        "identity_gate": gate,
        "blocked_rows_markdown": _blocked_rows_markdown(rows),
        "hard_blockers": _summary_hard_blockers(rows, gate_hard_blockers),
        "claim_policy": {
            "identity_rows_enter_pass_fail_accounting": True,
            "evidence_incomplete_is_mapped_to_blocked": True,
            "unsupported_is_mapped_to_blocked": True,
            "prediction_rows_out_of_scope": True,
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
            "runs_identity_model": False,
            "runs_prediction_eval": False,
            "runs_intervention_eval": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_real_identity_rows_summary(payload)


def objectstate_real_identity_rows_from_bundle(
    bundle: Mapping[str, Any],
) -> tuple[ObjectStateRealityRow, ...]:
    checked = validate_objectstate_real_evidence_bundle(bundle)
    sample = checked["sample"]
    rows = []
    for accounting in checked["gate_accounting_rows"]:
        if accounting["evidence_kind"] != "identity":
            continue
        rows.append(_identity_reality_row(sample, accounting, checked))
    return tuple(validate_objectstate_reality_row(row) for row in rows)


def validate_objectstate_real_identity_rows_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("real identity rows summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_REAL_IDENTITY_ROWS_SCHEMA:
        raise ValueError(
            f"unsupported real identity rows schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_real_identity_rows":
        raise ValueError("real identity rows kind is unsupported")
    if payload.get("status") not in {
        "objectstate_real_identity_rows_pass",
        "objectstate_real_identity_rows_fail",
        "objectstate_real_identity_rows_incomplete",
    }:
        raise ValueError("real identity rows status is unsupported")
    if payload.get("source_bundle_schema") != OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA:
        raise ValueError("real identity rows source bundle schema is unsupported")
    if payload.get("reality_gate_schema") != OBJECTSTATE_REALITY_GATE_SCHEMA:
        raise ValueError("real identity rows reality gate schema is unsupported")
    if payload.get("reality_row_schema") != OBJECTSTATE_REALITY_ROW_SCHEMA:
        raise ValueError("real identity rows reality row schema is unsupported")
    sample = payload.get("sample")
    if not isinstance(sample, Mapping) or not sample.get("sample_id"):
        raise ValueError("real identity rows summary requires sample")
    rows = payload.get("identity_rows")
    if not isinstance(rows, list):
        raise ValueError("real identity rows summary requires identity_rows")
    for row in rows:
        _validate_reality_row_payload(row)
    counts = payload.get("row_counts")
    if not isinstance(counts, Mapping):
        raise ValueError("real identity rows summary requires row_counts")
    for key in (
        "identity_rows",
        "identity_pass_rows",
        "identity_fail_rows",
        "identity_blocked_rows",
        "identity_link_rows",
        "object_pose_rows",
        "observation_rows",
    ):
        _non_negative_int(counts.get(key), f"row_counts.{key}")
    if counts["identity_rows"] != len(rows):
        raise ValueError("identity row count must match identity_rows")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("real identity rows summary requires metrics")
    for key in (
        "physical_identity_count",
        "object_id_count",
        "identity_frame_count",
        "identity_accounting_status_counts",
        "reality_status_counts",
    ):
        if key not in metrics:
            raise ValueError(f"real identity rows metrics missing {key}")
    gate = payload.get("identity_gate")
    if gate is not None:
        validate_objectstate_reality_gate_summary(gate)
    if not isinstance(payload.get("blocked_rows_markdown"), str):
        raise ValueError("real identity rows summary requires blocked_rows_markdown")
    blockers = payload.get("hard_blockers")
    if not isinstance(blockers, list) or any(
        not isinstance(item, str) or not item for item in blockers
    ):
        raise ValueError("real identity rows summary requires hard_blockers")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("identity_rows_enter_pass_fail_accounting")
        or not claim_policy.get("evidence_incomplete_is_mapped_to_blocked")
        or not claim_policy.get("unsupported_is_mapped_to_blocked")
        or not claim_policy.get("prediction_rows_out_of_scope")
        or not claim_policy.get("intervention_rows_out_of_scope")
        or not claim_policy.get("does_not_claim_reality_gate_full_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("real identity rows summary must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("reconstructs_gaussians")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("runs_identity_model")
        or non_goals.get("runs_prediction_eval")
        or non_goals.get("runs_intervention_eval")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError(
            "real identity rows summary cannot claim capture, GT creation, "
            "reconstruction, training, eval, replay, diffusion or viewer mutation"
        )
    return dict(payload)


def _identity_reality_row(
    sample: Mapping[str, Any],
    accounting: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> ObjectStateRealityRow:
    accounting_status = str(accounting["accounting_status"])
    if accounting_status not in _IDENTITY_ACCOUNTING_STATUSES:
        raise ValueError(f"unsupported identity accounting status: {accounting_status}")
    gt = accounting["gt_requirements"]
    object_id = accounting.get("object_id")
    metrics = dict(accounting["metrics"])
    metrics.update(_identity_scope_metrics(bundle["identity_link_rows"], object_id))
    status = _reality_status(accounting_status)
    return ObjectStateRealityRow(
        row_id=str(accounting["row_id"]),
        sample_id=str(sample["sample_id"]),
        source_kind=str(sample["source_kind"]),
        evidence_kind="identity",
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
        failure_reason=(
            _failure_reason(accounting) if status == "fail" else None
        ),
    )


def _identity_scope_metrics(
    identity_links: Sequence[Mapping[str, Any]],
    object_id: Any,
) -> dict[str, float]:
    scoped_links = tuple(
        row
        for row in identity_links
        if object_id is None or row["object_id"] == object_id
    )
    physical_ids = {row["physical_identity_id"] for row in scoped_links}
    object_ids = {row["object_id"] for row in scoped_links}
    frame_ids = {row["frame_id"] for row in scoped_links}
    return {
        "identity_link_count": float(len(scoped_links)),
        "physical_identity_count": float(len(physical_ids)),
        "object_id_count": float(len(object_ids)),
        "identity_frame_count": float(len(frame_ids)),
    }


def _identity_bundle_metrics(
    bundle: Mapping[str, Any],
    rows: Sequence[ObjectStateRealityRow],
) -> dict[str, Any]:
    links = bundle["identity_link_rows"]
    return {
        "physical_identity_count": len({row["physical_identity_id"] for row in links}),
        "object_id_count": len({row["object_id"] for row in links}),
        "identity_frame_count": len({row["frame_id"] for row in links}),
        "identity_accounting_status_counts": _counts(
            row["accounting_status"]
            for row in bundle["gate_accounting_rows"]
            if row["evidence_kind"] == "identity"
        ),
        "reality_status_counts": _counts(row.status for row in rows),
    }


def _summary_status(rows: Sequence[ObjectStateRealityRow], gate_status: str) -> str:
    if not rows:
        return "objectstate_real_identity_rows_incomplete"
    if all(row.status == "blocked" for row in rows):
        return "objectstate_real_identity_rows_incomplete"
    if any(row.status == "fail" for row in rows):
        return "objectstate_real_identity_rows_fail"
    if gate_status == "objectstate_reality_gate_pass":
        return "objectstate_real_identity_rows_pass"
    return "objectstate_real_identity_rows_fail"


def _summary_hard_blockers(
    rows: Sequence[ObjectStateRealityRow],
    gate_hard_blockers: Sequence[str],
) -> list[str]:
    blockers = list(gate_hard_blockers)
    if not rows:
        blockers.append("missing identity accounting rows")
    return blockers


def _reality_status(accounting_status: str) -> str:
    if accounting_status in {"pass", "fail"}:
        return accounting_status
    if accounting_status in _BLOCKED_ACCOUNTING_STATUSES:
        return "blocked"
    raise ValueError(f"unsupported identity accounting status: {accounting_status}")


def _block_reason(accounting: Mapping[str, Any]) -> str:
    status = accounting["accounting_status"]
    reason = str(accounting.get("reason") or "").strip()
    if reason:
        return f"{status}: {reason}"
    return f"{status}: identity evidence cannot enter pass/fail accounting"


def _failure_reason(accounting: Mapping[str, Any]) -> str:
    reason = str(accounting.get("reason") or "").strip()
    return reason or "identity accounting row reported fail"


def _blocked_rows_markdown(rows: Sequence[ObjectStateRealityRow]) -> str:
    blocked = [row for row in rows if row.status == "blocked"]
    if not blocked:
        return "No blocked real identity rows.\n"
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
        raise ValueError("identity row payload must be a mapping")
    ground_truth = row.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        raise ValueError("identity row payload requires ground_truth")
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
