from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from objgauss.core.objectstate_real_evidence_bundle import (
    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SUMMARY_SCHEMA,
    validate_objectstate_real_evidence_bundle_summary,
)
from objgauss.core.objectstate_real_evidence_bundle_ledger import (
    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_LEDGER_SCHEMA,
    validate_objectstate_real_evidence_bundle_ledger_summary,
)
from objgauss.core.objectstate_real_identity_rows import (
    OBJECTSTATE_REAL_IDENTITY_ROWS_SCHEMA,
    validate_objectstate_real_identity_rows_summary,
)
from objgauss.core.objectstate_real_intervention_rows import (
    OBJECTSTATE_REAL_INTERVENTION_ROWS_SCHEMA,
    validate_objectstate_real_intervention_rows_summary,
)
from objgauss.core.objectstate_real_prediction_rows import (
    OBJECTSTATE_REAL_PREDICTION_ROWS_SCHEMA,
    validate_objectstate_real_prediction_rows_summary,
)
from objgauss.core.objectstate_reality_row_ledger import (
    OBJECTSTATE_REALITY_ROW_LEDGER_SCHEMA,
    validate_objectstate_reality_row_ledger_summary,
)

OBJECTSTATE_REAL_EVIDENCE_BUNDLE_LEDGER_PACKAGE_AUDIT_SCHEMA = (
    "objgauss-objectstate-real-evidence-bundle-ledger-package-audit-v1"
)

Validator = Callable[[Any], Mapping[str, Any]]
_ACCOUNTING_STATUSES = ("pass", "fail", "evidence_incomplete", "unsupported")
_CONTROLLED_OR_PUBLIC_SOURCE_KINDS = {"controlled_real", "public_replay"}


def objectstate_real_evidence_bundle_ledger_package_audit(
    package_root: str | Path,
    *,
    wrapper_file: str | Path = "real-evidence-bundle-ledger.json",
) -> dict[str, Any]:
    root = Path(package_root)
    wrapper_path = _resolve_package_path(root, wrapper_file)
    payloads: dict[str, Any] = {}
    files = [
        _json_file_record(
            "bundle_ledger_wrapper",
            wrapper_path,
            OBJECTSTATE_REAL_EVIDENCE_BUNDLE_LEDGER_SCHEMA,
            validate_objectstate_real_evidence_bundle_ledger_summary,
            required=True,
            payloads=payloads,
        )
    ]
    wrapper = payloads.get("bundle_ledger_wrapper")
    expected = _expected_files_from_wrapper(root, wrapper)
    payloads.update(expected["payloads"])
    files.extend(expected["files"])
    ledger = payloads.get("reality_row_ledger")
    gates = _reviewability_gates(root, wrapper, ledger, files)
    phase1 = _phase1_acceptance(wrapper, ledger, payloads, gates)
    status = (
        "objectstate_real_evidence_bundle_ledger_package_audit_reviewable"
        if all(gates.values())
        else "objectstate_real_evidence_bundle_ledger_package_audit_incomplete"
    )
    payload = {
        "schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_LEDGER_PACKAGE_AUDIT_SCHEMA,
        "kind": "objectstate_real_evidence_bundle_ledger_package_audit",
        "status": status,
        "package_root": str(root),
        "wrapper_file": str(wrapper_path),
        "files": files,
        "bundle_count": 0 if not isinstance(wrapper, Mapping) else wrapper["bundle_count"],
        "row_summary_count": (
            0 if not isinstance(wrapper, Mapping) else wrapper["row_summary_count"]
        ),
        "sample_ids": _sample_ids(wrapper),
        "row_counts": _row_counts(wrapper, ledger),
        "accounting_status_counts": _accounting_status_counts(wrapper),
        "evidence_accounts": _evidence_accounts(wrapper),
        "reviewability_gates": gates,
        "phase1_acceptance_status": phase1["status"],
        "phase1_acceptance_gates": phase1["gates"],
        "phase1_acceptance_counts": phase1["counts"],
        "phase1_acceptance_issues": phase1["issues"],
        "issues": _issues(files, gates) + list(phase1["issues"]),
        "claim_policy": {
            "read_only_audit": True,
            "checks_real_bundle_ledger_package": True,
            "full_reality_row_ledger_is_authoritative": True,
            "static_scene_evidence_is_separate_from_state_variable_evidence": True,
            "evidence_incomplete_is_not_model_fail": True,
            "phase1_acceptance_is_evidence_system_not_metric_pass": True,
            "does_not_claim_metric_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "captures_video": False,
            "creates_ground_truth": False,
            "reconstructs_gaussians": False,
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
    return validate_objectstate_real_evidence_bundle_ledger_package_audit(payload)


def validate_objectstate_real_evidence_bundle_ledger_package_audit(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("real evidence bundle ledger package audit must be a mapping")
    if (
        payload.get("schema")
        != OBJECTSTATE_REAL_EVIDENCE_BUNDLE_LEDGER_PACKAGE_AUDIT_SCHEMA
    ):
        raise ValueError(
            "unsupported real evidence bundle ledger package audit schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_real_evidence_bundle_ledger_package_audit":
        raise ValueError("real evidence bundle ledger package audit kind is unsupported")
    if payload.get("status") not in {
        "objectstate_real_evidence_bundle_ledger_package_audit_reviewable",
        "objectstate_real_evidence_bundle_ledger_package_audit_incomplete",
    }:
        raise ValueError("real evidence bundle ledger package audit status is unsupported")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("real evidence bundle ledger package audit requires files")
    for record in files:
        _validate_file_record(record)
    gates = payload.get("reviewability_gates")
    if not isinstance(gates, Mapping) or not gates:
        raise ValueError("real evidence bundle ledger package audit requires gates")
    if any(not isinstance(value, bool) for value in gates.values()):
        raise ValueError("real evidence bundle ledger package audit gates must be bool")
    expected_status = (
        "objectstate_real_evidence_bundle_ledger_package_audit_reviewable"
        if all(gates.values())
        else "objectstate_real_evidence_bundle_ledger_package_audit_incomplete"
    )
    if payload["status"] != expected_status:
        raise ValueError("real evidence bundle ledger package audit status mismatch")
    phase1_status = payload.get("phase1_acceptance_status")
    phase1_gates = payload.get("phase1_acceptance_gates")
    if not isinstance(phase1_gates, Mapping) or not phase1_gates:
        raise ValueError("real evidence bundle ledger package audit requires phase1 gates")
    if any(not isinstance(value, bool) for value in phase1_gates.values()):
        raise ValueError("real evidence bundle ledger package audit phase1 gates must be bool")
    expected_phase1_status = (
        "objectstate_phase1_evidence_system_acceptance_pass"
        if all(phase1_gates.values())
        else "objectstate_phase1_evidence_system_acceptance_incomplete"
    )
    if phase1_status != expected_phase1_status:
        raise ValueError(
            "real evidence bundle ledger package audit phase1 status mismatch"
        )
    phase1_counts = payload.get("phase1_acceptance_counts")
    if not isinstance(phase1_counts, Mapping) or not phase1_counts:
        raise ValueError("real evidence bundle ledger package audit requires phase1 counts")
    if any(not isinstance(value, int) or value < 0 for value in phase1_counts.values()):
        raise ValueError(
            "real evidence bundle ledger package audit phase1 counts must be non-negative ints"
        )
    phase1_issues = payload.get("phase1_acceptance_issues")
    if not isinstance(phase1_issues, list) or any(
        not isinstance(issue, str) for issue in phase1_issues
    ):
        raise ValueError("real evidence bundle ledger package audit phase1 issues invalid")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("read_only_audit")
        or not claim_policy.get("checks_real_bundle_ledger_package")
        or not claim_policy.get("full_reality_row_ledger_is_authoritative")
        or not claim_policy.get(
            "static_scene_evidence_is_separate_from_state_variable_evidence"
        )
        or not claim_policy.get("evidence_incomplete_is_not_model_fail")
        or not claim_policy.get(
            "phase1_acceptance_is_evidence_system_not_metric_pass"
        )
        or not claim_policy.get("does_not_claim_metric_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("real evidence bundle ledger package audit claim policy changed")
    non_goals = payload.get("non_goals", {})
    if any(bool(value) for value in non_goals.values()):
        raise ValueError("real evidence bundle ledger package audit cannot claim non-goals")
    return dict(payload)


def _expected_files_from_wrapper(root: Path, wrapper: Any) -> dict[str, Any]:
    if not isinstance(wrapper, Mapping):
        return {"files": []}
    payloads: dict[str, Any] = {}
    files = [
        _json_file_record(
            "reality_row_ledger",
            _resolve_record_path(root, wrapper.get("ledger_summary_path")),
            OBJECTSTATE_REALITY_ROW_LEDGER_SCHEMA,
            validate_objectstate_reality_row_ledger_summary,
            required=True,
            payloads=payloads,
        ),
        _markdown_file_record(
            "blocked_rows_markdown",
            _resolve_record_path(root, wrapper.get("blocked_rows_path")),
            required=True,
        ),
        _markdown_file_record(
            "state_variable_evidence_matrix_markdown",
            _resolve_record_path(root, wrapper.get("state_variable_evidence_matrix_path")),
            required=True,
        ),
        _markdown_file_record(
            "next_actions_markdown",
            _resolve_record_path(root, wrapper.get("next_actions_path")),
            required=True,
        ),
    ]
    for index, record in enumerate(wrapper.get("records", [])):
        if not isinstance(record, Mapping):
            continue
        prefix = f"record_{index:03d}"
        files.extend(
            [
                _json_file_record(
                    f"{prefix}_bundle_summary",
                    _resolve_record_path(root, record.get("bundle_summary_path")),
                    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SUMMARY_SCHEMA,
                    validate_objectstate_real_evidence_bundle_summary,
                    required=True,
                    payloads=payloads,
                ),
                _json_file_record(
                    f"{prefix}_identity_summary",
                    _resolve_record_path(root, record.get("identity_summary_path")),
                    OBJECTSTATE_REAL_IDENTITY_ROWS_SCHEMA,
                    validate_objectstate_real_identity_rows_summary,
                    required=True,
                    payloads=payloads,
                ),
                _json_file_record(
                    f"{prefix}_prediction_summary",
                    _resolve_record_path(root, record.get("prediction_summary_path")),
                    OBJECTSTATE_REAL_PREDICTION_ROWS_SCHEMA,
                    validate_objectstate_real_prediction_rows_summary,
                    required=True,
                    payloads=payloads,
                ),
                _json_file_record(
                    f"{prefix}_intervention_summary",
                    _resolve_record_path(root, record.get("intervention_summary_path")),
                    OBJECTSTATE_REAL_INTERVENTION_ROWS_SCHEMA,
                    validate_objectstate_real_intervention_rows_summary,
                    required=True,
                    payloads=payloads,
                ),
            ]
        )
    return {"files": files, "payloads": payloads}


def _reviewability_gates(
    root: Path,
    wrapper: Any,
    ledger: Any,
    files: list[dict[str, Any]],
) -> dict[str, bool]:
    wrapper_ok = isinstance(wrapper, Mapping)
    ledger_ok = isinstance(ledger, Mapping)
    return {
        "required_files_present": all(
            bool(record["is_file"]) for record in files if record["required"]
        ),
        "required_json_schemas_valid": all(
            bool(record.get("schema_ok")) and bool(record.get("validator_ok"))
            for record in files
            if record["required"] and record["kind"] == "json"
        ),
        "wrapper_output_root_matches_package_root": (
            wrapper_ok and _same_path(root, wrapper.get("output_root"))
        ),
        "wrapper_embedded_ledger_matches_file": (
            wrapper_ok and ledger_ok and wrapper.get("ledger") == ledger
        ),
        "row_counts_match_ledger": _row_counts_match(wrapper, ledger),
        "record_summaries_match_wrapper_records": _record_summaries_match(wrapper),
        "static_state_evidence_split_preserved": _evidence_split_preserved(wrapper),
        "blocked_rows_are_not_pass_rows": (
            wrapper_ok
            and bool(wrapper.get("claim_policy", {}).get("blocked_rows_are_not_pass_rows"))
        ),
        "full_reality_row_ledger_authoritative": (
            wrapper_ok
            and bool(
                wrapper.get("claim_policy", {}).get(
                    "full_reality_row_ledger_is_authoritative"
                )
            )
        ),
        "markdown_outputs_present": all(
            bool(record["is_file"]) and int(record.get("size_bytes") or 0) > 0
            for record in files
            if record["kind"] == "markdown" and record["required"]
        ),
    }


def _phase1_acceptance(
    wrapper: Any,
    ledger: Any,
    payloads: Mapping[str, Any],
    reviewability_gates: Mapping[str, bool],
) -> dict[str, Any]:
    counts = _phase1_counts(wrapper, ledger, payloads)
    gates = {
        "package_reviewability_gates_pass": all(reviewability_gates.values()),
        "controlled_or_public_real_bundle_loaded": (
            counts["controlled_public_bundle_count"] >= 1
        ),
        "identity_rows_enter_accounting": counts["identity_accounting_row_count"] >= 1,
        "prediction_rows_enter_accounting": (
            counts["prediction_accounting_row_count"] >= 1
        ),
        "intervention_rows_enter_accounting": (
            counts["intervention_accounting_row_count"] >= 1
        ),
        "evaluable_real_accounting_rows_present": counts["pass_fail_row_count"] >= 1,
        "missing_gt_accounting_is_separate_from_fail": (
            _missing_gt_accounting_is_separate_from_fail(wrapper, ledger)
        ),
        "synthetic_and_real_gate_split": _synthetic_and_real_gate_split(
            wrapper,
            ledger,
        ),
        "static_scene_and_state_variable_evidence_split": _evidence_split_preserved(
            wrapper
        ),
    }
    issues = [
        f"phase1 acceptance gate failed: {gate}"
        for gate, passed in gates.items()
        if not passed
    ]
    return {
        "status": (
            "objectstate_phase1_evidence_system_acceptance_pass"
            if all(gates.values())
            else "objectstate_phase1_evidence_system_acceptance_incomplete"
        ),
        "gates": gates,
        "counts": counts,
        "issues": issues,
    }


def _phase1_counts(
    wrapper: Any,
    ledger: Any,
    payloads: Mapping[str, Any],
) -> dict[str, int]:
    accounting = _accounting_status_counts(wrapper)
    row_counts = _row_counts(wrapper, ledger)
    source_kind_counts = _phase1_bundle_source_kind_counts(wrapper, payloads)
    controlled_public_bundle_count = sum(
        source_kind_counts.get(source_kind, 0)
        for source_kind in _CONTROLLED_OR_PUBLIC_SOURCE_KINDS
    )
    return {
        "bundle_count": int(wrapper.get("bundle_count", 0))
        if isinstance(wrapper, Mapping)
        else 0,
        "controlled_public_bundle_count": controlled_public_bundle_count,
        "identity_accounting_row_count": _accounting_total(accounting, "identity"),
        "prediction_accounting_row_count": _accounting_total(accounting, "prediction"),
        "intervention_accounting_row_count": _accounting_total(
            accounting,
            "intervention",
        ),
        "pass_row_count": int(row_counts.get("pass_row_count", 0)),
        "fail_row_count": int(row_counts.get("fail_row_count", 0)),
        "pass_fail_row_count": int(row_counts.get("pass_row_count", 0))
        + int(row_counts.get("fail_row_count", 0)),
        "evidence_incomplete_row_count": int(
            row_counts.get("evidence_incomplete_row_count", 0)
        ),
        "unsupported_row_count": int(row_counts.get("unsupported_row_count", 0)),
        "static_scene_available_bundle_count": int(
            _evidence_accounts(wrapper)
            .get("static_scene_evidence", {})
            .get("available_bundle_count", 0)
        ),
        "state_variable_ready_bundle_count": int(
            _evidence_accounts(wrapper)
            .get("state_variable_evidence", {})
            .get("ready_bundle_count", 0)
        ),
        "state_variable_intervention_ready_bundle_count": int(
            _evidence_accounts(wrapper)
            .get("state_variable_evidence", {})
            .get("intervention_ready_bundle_count", 0)
        ),
    }


def _phase1_bundle_source_kind_counts(
    wrapper: Any,
    payloads: Mapping[str, Any],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(wrapper, Mapping):
        return counts
    for index, record in enumerate(wrapper.get("records", [])):
        if not isinstance(record, Mapping):
            continue
        bundle_summary = payloads.get(f"record_{index:03d}_bundle_summary")
        source_kind = None
        if isinstance(bundle_summary, Mapping):
            sample = bundle_summary.get("sample", {})
            if isinstance(sample, Mapping):
                source_kind = sample.get("source_kind")
        if not isinstance(source_kind, str) or not source_kind:
            source_kind = record.get("source_kind")
        if isinstance(source_kind, str) and source_kind:
            counts[source_kind] = counts.get(source_kind, 0) + 1
    return counts


def _accounting_total(accounting: Mapping[str, Any], evidence_kind: str) -> int:
    counts = accounting.get(evidence_kind, {})
    if not isinstance(counts, Mapping):
        return 0
    return sum(int(counts.get(status, 0)) for status in _ACCOUNTING_STATUSES)


def _accounting_count(
    accounting: Mapping[str, Any],
    evidence_kind: str,
    status: str,
) -> int:
    counts = accounting.get(evidence_kind, {})
    if not isinstance(counts, Mapping):
        return 0
    return int(counts.get(status, 0))


def _missing_gt_accounting_is_separate_from_fail(wrapper: Any, ledger: Any) -> bool:
    if not isinstance(wrapper, Mapping):
        return False
    accounting = _accounting_status_counts(wrapper)
    row_counts = _row_counts(wrapper, ledger)
    return (
        int(row_counts.get("fail_row_count", 0))
        == _accounting_count(accounting, "all", "fail")
        and int(row_counts.get("evidence_incomplete_row_count", 0))
        == _accounting_count(accounting, "all", "evidence_incomplete")
        and int(row_counts.get("unsupported_row_count", 0))
        == _accounting_count(accounting, "all", "unsupported")
        and bool(
            wrapper.get("claim_policy", {}).get("evidence_incomplete_is_not_model_fail")
        )
    )


def _synthetic_and_real_gate_split(wrapper: Any, ledger: Any) -> bool:
    if not isinstance(wrapper, Mapping) or not isinstance(ledger, Mapping):
        return False
    gate = ledger.get("gate", {})
    gate_claim = gate.get("claim_policy", {}) if isinstance(gate, Mapping) else {}
    ledger_claim = ledger.get("claim_policy", {})
    wrapper_claim = wrapper.get("claim_policy", {})
    source_counts = ledger.get("row_counts", {}).get("by_source_kind", {})
    controlled_public_rows = 0
    if isinstance(source_counts, Mapping):
        controlled_public_rows = sum(
            int(source_counts.get(source_kind, 0))
            for source_kind in _CONTROLLED_OR_PUBLIC_SOURCE_KINDS
        )
    return (
        controlled_public_rows >= 1
        and bool(wrapper_claim.get("full_reality_row_ledger_is_authoritative"))
        and bool(ledger_claim.get("full_gate_status_is_authoritative"))
        and bool(gate_claim.get("synthetic_smoke_is_prerequisite_not_reality_proof"))
    )


def _json_file_record(
    key: str,
    path: Path,
    expected_schema: str,
    validator: Validator,
    *,
    required: bool,
    payloads: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = _base_file_record(
        key,
        path,
        kind="json",
        required=required,
    )
    if not record["is_file"]:
        if required:
            record["issues"].append("required file is missing")
        return record
    try:
        payload = _read_json(path)
    except Exception as exc:
        record["issues"].append(f"json read failed: {exc}")
        return record
    record["schema"] = payload.get("schema")
    record["schema_ok"] = record["schema"] == expected_schema
    payloads[key] = payload
    if not record["schema_ok"]:
        record["issues"].append(
            f"schema mismatch: expected {expected_schema}, got {record['schema']}"
        )
    try:
        payloads[key] = validator(payload)
    except Exception as exc:
        record["issues"].append(f"validator failed: {exc}")
        return record
    record["validator_ok"] = True
    return record


def _markdown_file_record(key: str, path: Path, *, required: bool) -> dict[str, Any]:
    record = _base_file_record(key, path, kind="markdown", required=required)
    if not record["is_file"] and required:
        record["issues"].append("required file is missing")
    return record


def _base_file_record(
    key: str,
    path: Path,
    *,
    kind: str,
    required: bool,
) -> dict[str, Any]:
    is_file = path.is_file()
    return {
        "key": key,
        "path": str(path),
        "kind": kind,
        "required": bool(required),
        "is_file": is_file,
        "size_bytes": path.stat().st_size if is_file else 0,
        "schema": None,
        "schema_ok": False,
        "validator_ok": False,
        "issues": [],
    }


def _record_summaries_match(wrapper: Any) -> bool:
    if not isinstance(wrapper, Mapping):
        return False
    for record in wrapper.get("records", []):
        if not isinstance(record, Mapping):
            return False
        bundle_summary = _maybe_read(record.get("bundle_summary_path"))
        identity_summary = _maybe_read(record.get("identity_summary_path"))
        prediction_summary = _maybe_read(record.get("prediction_summary_path"))
        intervention_summary = _maybe_read(record.get("intervention_summary_path"))
        if not all(
            isinstance(item, Mapping)
            for item in (
                bundle_summary,
                identity_summary,
                prediction_summary,
                intervention_summary,
            )
        ):
            return False
        sample_id = record.get("sample_id")
        if bundle_summary["sample"]["sample_id"] != sample_id:
            return False
        for summary in (identity_summary, prediction_summary, intervention_summary):
            if summary["sample"]["sample_id"] != sample_id:
                return False
    return True


def _row_counts_match(wrapper: Any, ledger: Any) -> bool:
    if not isinstance(wrapper, Mapping) or not isinstance(ledger, Mapping):
        return False
    row_counts = wrapper.get("row_counts", {})
    return (
        row_counts.get("row_count") == ledger.get("row_count")
        and row_counts.get("pass_row_count") == ledger.get("pass_row_count")
        and row_counts.get("fail_row_count") == ledger.get("fail_row_count")
        and row_counts.get("blocked_row_count") == ledger.get("blocked_row_count")
    )


def _evidence_split_preserved(wrapper: Any) -> bool:
    if not isinstance(wrapper, Mapping):
        return False
    accounts = wrapper.get("evidence_accounts", {})
    static = accounts.get("static_scene_evidence", {})
    state = accounts.get("state_variable_evidence", {})
    claim = wrapper.get("claim_policy", {})
    return (
        static.get("usable_for_state_variable_gate") is False
        and state.get("requires_full_reality_row_ledger") is True
        and claim.get("static_scene_evidence_is_separate_from_state_variable_evidence")
        is True
        and claim.get("evidence_incomplete_is_not_model_fail") is True
    )


def _row_counts(wrapper: Any, ledger: Any) -> dict[str, int]:
    if isinstance(wrapper, Mapping) and isinstance(wrapper.get("row_counts"), Mapping):
        return {key: int(value) for key, value in wrapper["row_counts"].items()}
    if isinstance(ledger, Mapping):
        return {
            "row_count": int(ledger.get("row_count", 0)),
            "pass_row_count": int(ledger.get("pass_row_count", 0)),
            "fail_row_count": int(ledger.get("fail_row_count", 0)),
            "blocked_row_count": int(ledger.get("blocked_row_count", 0)),
        }
    return {
        "row_count": 0,
        "pass_row_count": 0,
        "fail_row_count": 0,
        "blocked_row_count": 0,
        "evidence_incomplete_row_count": 0,
        "unsupported_row_count": 0,
    }


def _accounting_status_counts(wrapper: Any) -> dict[str, Any]:
    if isinstance(wrapper, Mapping) and isinstance(
        wrapper.get("accounting_status_counts"), Mapping
    ):
        return dict(wrapper["accounting_status_counts"])
    return {
        "identity": {},
        "prediction": {},
        "intervention": {},
        "bundle": {},
        "all": {},
    }


def _evidence_accounts(wrapper: Any) -> dict[str, Any]:
    if isinstance(wrapper, Mapping) and isinstance(wrapper.get("evidence_accounts"), Mapping):
        return dict(wrapper["evidence_accounts"])
    return {
        "static_scene_evidence": {
            "available_bundle_count": 0,
            "usable_for_state_variable_gate": False,
        },
        "state_variable_evidence": {
            "ready_bundle_count": 0,
            "intervention_ready_bundle_count": 0,
            "requires_full_reality_row_ledger": True,
        },
    }


def _sample_ids(wrapper: Any) -> list[str]:
    if not isinstance(wrapper, Mapping):
        return []
    return sorted(
        str(record["sample_id"])
        for record in wrapper.get("records", [])
        if isinstance(record, Mapping) and record.get("sample_id")
    )


def _issues(files: list[dict[str, Any]], gates: Mapping[str, bool]) -> list[str]:
    issues = []
    for record in files:
        for issue in record["issues"]:
            issues.append(f"{record['key']}: {issue}")
    for gate, passed in gates.items():
        if not passed:
            issues.append(f"reviewability gate failed: {gate}")
    return issues


def _validate_file_record(record: Any) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("file record must be a mapping")
    for key in ("key", "path", "kind"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"file record missing {key}")
    for key in ("required", "is_file", "schema_ok", "validator_ok"):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"file record {record.get('key')} missing bool {key}")
    if not isinstance(record.get("size_bytes"), int) or record["size_bytes"] < 0:
        raise ValueError("file record requires non-negative size_bytes")
    if not isinstance(record.get("issues"), list):
        raise ValueError("file record requires issues")


def _resolve_package_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _resolve_record_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        return root / "__missing__"
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = root / path.name
    if candidate.exists():
        return candidate
    return path


def _same_path(root: Path, value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    if not path.is_absolute() and not path.exists():
        path = root if value == str(root) or Path(value).name == root.name else path
    try:
        return path.resolve() == root.resolve()
    except OSError:
        return False


def _maybe_read(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except Exception:
        return None


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON: {path}")
    return payload
