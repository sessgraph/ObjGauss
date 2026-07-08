from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_bop_reality_rows import (
    OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA,
    validate_objectstate_bop_reality_rows_summary,
)
from objgauss.core.objectstate_controlled_real_rows import (
    OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
    validate_objectstate_controlled_real_rows_summary,
)
from objgauss.core.objectstate_reality_gate import (
    OBJECTSTATE_REALITY_GATE_SCHEMA,
    ObjectStateRealityGateThresholds,
    ObjectStateRealityRow,
    evaluate_objectstate_reality_gate,
    objectstate_reality_blocked_rows_markdown,
    validate_objectstate_reality_gate_summary,
    validate_objectstate_reality_row,
)
from objgauss.core.objectstate_reality_public_rows import (
    OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA,
    validate_objectstate_reality_public_rows_summary,
)

OBJECTSTATE_REALITY_ROW_LEDGER_SCHEMA = (
    "objgauss-objectstate-reality-row-ledger-v1"
)
_SUPPORTED_SUMMARY_SCHEMAS = {
    OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA,
    OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
    OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA,
    OBJECTSTATE_REALITY_GATE_SCHEMA,
}


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
    rows = checked.get("rows")
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
    if rows:
        gate = evaluate_objectstate_reality_gate(
            tuple(rows),
            synthetic_smoke_passed=bool(synthetic_smoke_passed),
            thresholds=thresholds,
        )
        gate_payload = gate.as_dict()
        blocked_rows_markdown = objectstate_reality_blocked_rows_markdown(gate)

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
        "row_count": len(rows),
        "pass_row_count": _row_status_count(rows, "pass"),
        "fail_row_count": _row_status_count(rows, "fail"),
        "blocked_row_count": _row_status_count(rows, "blocked"),
        "sample_scope": _sample_scope(rows),
        "row_counts": {
            "by_source_kind": _counts(row.source_kind for row in rows),
            "by_evidence_kind": _counts(row.evidence_kind for row in rows),
            "by_status": _counts(row.status for row in rows),
        },
        "duplicate_row_ids": duplicate_row_ids,
        "rows": [row.as_dict() for row in rows],
        "gate": gate_payload,
        "gap_summary": _gap_summary(gate_payload, rows),
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
    for row in rows:
        _row_from_payload(row)
    gate = payload.get("gate")
    if gate is not None:
        if not isinstance(gate, Mapping):
            raise ValueError("ObjectState reality row ledger gate must be a mapping")
        validate_objectstate_reality_gate_summary(dict(gate))
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
    if not isinstance(payload.get("row_counts"), Mapping):
        raise ValueError("ObjectState reality row ledger requires row_counts")
    if not isinstance(payload.get("duplicate_row_ids"), list):
        raise ValueError("ObjectState reality row ledger requires duplicate_row_ids")
    if not isinstance(payload.get("gap_summary"), Mapping):
        raise ValueError("ObjectState reality row ledger requires gap_summary")
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
    if schema == OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA:
        return validate_objectstate_reality_public_rows_summary(dict(summary))
    if schema == OBJECTSTATE_REALITY_GATE_SCHEMA:
        return validate_objectstate_reality_gate_summary(dict(summary))
    raise ValueError(f"unsupported ObjectState reality row summary schema: {schema}")


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
