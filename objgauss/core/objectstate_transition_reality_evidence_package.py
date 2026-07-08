from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from objgauss.core.objectstate_controlled_intervention_eval import (
    OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
    OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA,
    validate_objectstate_controlled_intervention_candidates,
    validate_objectstate_controlled_intervention_eval_summary,
)
from objgauss.core.objectstate_controlled_prediction_eval import (
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
    OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA,
    validate_objectstate_controlled_prediction_candidates,
    validate_objectstate_controlled_prediction_eval_summary,
)
from objgauss.core.objectstate_controlled_real_rows import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
    OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
    validate_objectstate_controlled_real_manifest,
    validate_objectstate_controlled_real_rows_summary,
)
from objgauss.core.objectstate_transition_dataset import (
    OBJECTSTATE_TRANSITION_DATASET_AUDIT_SCHEMA,
    validate_objectstate_transition_dataset_audit,
)
from objgauss.core.objectstate_transition_intervention_candidates import (
    OBJECTSTATE_TRANSITION_INTERVENTION_CANDIDATES_SCHEMA,
    validate_objectstate_transition_intervention_candidates_summary,
)
from objgauss.core.objectstate_transition_prediction_candidates import (
    OBJECTSTATE_TRANSITION_PREDICTION_CANDIDATES_SCHEMA,
    validate_objectstate_transition_prediction_candidates_summary,
)
from objgauss.core.objectstate_transition_reality_handoff import (
    OBJECTSTATE_TRANSITION_REALITY_HANDOFF_SCHEMA,
    validate_objectstate_transition_reality_handoff_summary,
)

OBJECTSTATE_TRANSITION_REALITY_EVIDENCE_PACKAGE_SCHEMA = (
    "objgauss-objectstate-transition-reality-evidence-package-v1"
)

Validator = Callable[[Any], Mapping[str, Any]]


def objectstate_transition_reality_evidence_package(
    package_root: str | Path,
    *,
    handoff_dir: str | Path = "transition-reality-handoff",
) -> dict[str, Any]:
    root = Path(package_root)
    handoff_root = _resolve_package_path(root, handoff_dir)
    payloads: dict[str, Any] = {}
    files = [
        _json_file_record(
            "transition_dataset_audit",
            handoff_root / "transition-dataset-audit.json",
            OBJECTSTATE_TRANSITION_DATASET_AUDIT_SCHEMA,
            validate_objectstate_transition_dataset_audit,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "prediction_candidates",
            handoff_root / "prediction-candidates.json",
            OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
            validate_objectstate_controlled_prediction_candidates,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "transition_prediction_summary",
            handoff_root / "transition-prediction-summary.json",
            OBJECTSTATE_TRANSITION_PREDICTION_CANDIDATES_SCHEMA,
            validate_objectstate_transition_prediction_candidates_summary,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "prediction_eval_summary",
            handoff_root / "prediction-eval-summary.json",
            OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA,
            validate_objectstate_controlled_prediction_eval_summary,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "prediction_controlled_real",
            handoff_root / "prediction-controlled-real.json",
            OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
            validate_objectstate_controlled_real_manifest,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "intervention_candidates",
            handoff_root / "intervention-candidates.json",
            OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
            validate_objectstate_controlled_intervention_candidates,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "transition_intervention_summary",
            handoff_root / "transition-intervention-summary.json",
            OBJECTSTATE_TRANSITION_INTERVENTION_CANDIDATES_SCHEMA,
            validate_objectstate_transition_intervention_candidates_summary,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "intervention_eval_summary",
            handoff_root / "intervention-eval-summary.json",
            OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA,
            validate_objectstate_controlled_intervention_eval_summary,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "intervention_controlled_real",
            handoff_root / "intervention-controlled-real.json",
            OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
            validate_objectstate_controlled_real_manifest,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "controlled_real_manifest",
            handoff_root / "controlled-real.json",
            OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
            validate_objectstate_controlled_real_manifest,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "controlled_real_summary",
            handoff_root / "controlled-real-summary.json",
            OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
            validate_objectstate_controlled_real_rows_summary,
            required=True,
            payloads=payloads,
        ),
        _markdown_file_record(
            "blocked_rows_markdown",
            handoff_root / "blocked-rows.md",
            required=True,
        ),
        _json_file_record(
            "transition_reality_handoff_summary",
            handoff_root / "transition-reality-handoff-summary.json",
            OBJECTSTATE_TRANSITION_REALITY_HANDOFF_SCHEMA,
            validate_objectstate_transition_reality_handoff_summary,
            required=True,
            payloads=payloads,
        ),
    ]
    sample_consistency = _sample_consistency(files)
    output_consistency = _standalone_output_consistency(payloads)
    row_accounting = _row_accounting(payloads.get("controlled_real_summary"))
    handoff = _handoff_summary(payloads.get("transition_reality_handoff_summary"))
    transition = _transition_summary(payloads.get("transition_dataset_audit"))
    gates = {
        "required_files_present": all(
            bool(record["is_file"]) for record in files if record["required"]
        ),
        "required_json_schemas_valid": all(
            bool(record["schema_ok"]) and bool(record["validator_ok"])
            for record in files
            if record["required"] and record["kind"] == "json"
        ),
        "sample_ids_consistent": bool(sample_consistency["consistent"]),
        "transition_dataset_ready": bool(transition["transition_dataset_ready"]),
        "handoff_summary_present": "transition_reality_handoff_summary" in payloads,
        "standalone_outputs_match_handoff_summary": bool(output_consistency["matches"]),
        "controlled_real_summary_present": "controlled_real_summary" in payloads,
        "row_accounting_present": bool(row_accounting["present"]),
        "identity_row_blocked": bool(row_accounting["identity_row_blocked"]),
        "prediction_row_is_pass_or_fail": bool(
            row_accounting["prediction_row_status"] in {"pass", "fail"}
        ),
        "intervention_row_is_pass_or_fail": bool(
            row_accounting["intervention_row_status"] in {"pass", "fail"}
        ),
        "partial_gate_does_not_claim_identity_pass": bool(
            row_accounting["identity_row_blocked"]
            and not handoff["requires_identity_pass_row"]
        ),
        "blocked_rows_markdown_present": bool(
            next(
                record
                for record in files
                if record["key"] == "blocked_rows_markdown"
            )["is_file"]
        ),
    }
    status = (
        "objectstate_transition_reality_evidence_package_reviewable"
        if all(gates.values())
        else "objectstate_transition_reality_evidence_package_incomplete"
    )
    payload = {
        "schema": OBJECTSTATE_TRANSITION_REALITY_EVIDENCE_PACKAGE_SCHEMA,
        "kind": "objectstate_transition_reality_evidence_package",
        "status": status,
        "package_root": str(root),
        "handoff_dir": str(handoff_root),
        "sample_id": sample_consistency["sample_id"],
        "files": files,
        "sample_consistency": sample_consistency,
        "transition": transition,
        "handoff": handoff,
        "row_accounting": row_accounting,
        "output_consistency": output_consistency,
        "reviewability_gates": gates,
        "issues": _package_issues(files, gates, sample_consistency, output_consistency),
        "claim_policy": {
            "read_only_audit": True,
            "checks_local_transition_reality_evidence_package": True,
            "transition_handoff_required": True,
            "requires_identity_row_blocked": True,
            "does_not_create_ground_truth": True,
            "does_not_run_identity_handoff": True,
            "does_not_run_prediction_eval": True,
            "does_not_run_intervention_eval": True,
            "does_not_train_dynamics_model": True,
            "does_not_create_replay_buffer": True,
            "does_not_claim_metric_pass": True,
            "does_not_claim_full_reality_gate": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "reconstructs_gaussians": False,
            "runs_tracking_model": False,
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
    return validate_objectstate_transition_reality_evidence_package_summary(payload)


def validate_objectstate_transition_reality_evidence_package_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("transition reality evidence package summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_TRANSITION_REALITY_EVIDENCE_PACKAGE_SCHEMA:
        raise ValueError(
            "unsupported transition reality evidence package schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_transition_reality_evidence_package":
        raise ValueError("transition reality evidence package kind is unsupported")
    if payload.get("status") not in {
        "objectstate_transition_reality_evidence_package_reviewable",
        "objectstate_transition_reality_evidence_package_incomplete",
    }:
        raise ValueError("transition reality evidence package status is unsupported")
    for key in ("package_root", "handoff_dir"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"transition reality evidence package requires {key}")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("transition reality evidence package requires files")
    for record in files:
        _validate_file_record(record)
    sample_consistency = payload.get("sample_consistency")
    if not isinstance(sample_consistency, Mapping):
        raise ValueError("transition reality evidence package requires sample consistency")
    if not isinstance(sample_consistency.get("values"), Mapping):
        raise ValueError("transition reality evidence package sample values invalid")
    gates = payload.get("reviewability_gates")
    if not isinstance(gates, Mapping) or not gates:
        raise ValueError("transition reality evidence package requires gates")
    if any(not isinstance(value, bool) for value in gates.values()):
        raise ValueError("transition reality evidence package gates must be bool")
    expected_status = (
        "objectstate_transition_reality_evidence_package_reviewable"
        if all(gates.values())
        else "objectstate_transition_reality_evidence_package_incomplete"
    )
    if payload["status"] != expected_status:
        raise ValueError("transition reality evidence package status mismatch")
    row_accounting = payload.get("row_accounting")
    if not isinstance(row_accounting, Mapping):
        raise ValueError("transition reality evidence package requires row accounting")
    for key in (
        "present",
        "identity_prediction_intervention_rows_present",
        "identity_row_blocked",
    ):
        if not isinstance(row_accounting.get(key), bool):
            raise ValueError(f"transition reality row accounting invalid: {key}")
    handoff = payload.get("handoff")
    if not isinstance(handoff, Mapping):
        raise ValueError("transition reality evidence package requires handoff summary")
    for key in (
        "status",
        "prediction_eval_status",
        "intervention_eval_status",
        "partial_reality_gate_status",
    ):
        if handoff.get(key) is not None and not isinstance(handoff.get(key), str):
            raise ValueError(f"transition reality handoff field invalid: {key}")
    if not isinstance(handoff.get("requires_identity_pass_row"), bool):
        raise ValueError("transition reality handoff identity gate flag invalid")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("transition reality evidence package issues must be list")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("read_only_audit")
        or not claim_policy.get("checks_local_transition_reality_evidence_package")
        or not claim_policy.get("transition_handoff_required")
        or not claim_policy.get("requires_identity_row_blocked")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_run_identity_handoff")
        or not claim_policy.get("does_not_run_prediction_eval")
        or not claim_policy.get("does_not_run_intervention_eval")
        or not claim_policy.get("does_not_train_dynamics_model")
        or not claim_policy.get("does_not_create_replay_buffer")
        or not claim_policy.get("does_not_claim_metric_pass")
        or not claim_policy.get("does_not_claim_full_reality_gate")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("transition reality evidence package must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "transition reality evidence package cannot claim capture, GT, "
            "reconstruction, models, training, public samples, replay, diffusion, "
            "or viewer mutation"
        )
    return dict(payload)


def _json_file_record(
    key: str,
    path: Path,
    expected_schema: str,
    validator: Validator,
    *,
    required: bool,
    payloads: dict[str, Any],
) -> dict[str, Any]:
    base = _base_file_record(key, path, required=required, kind="json")
    issues: list[str] = []
    schema = None
    schema_ok = False
    validator_ok = False
    sample_id = None
    status = None
    if not base["is_file"]:
        if required:
            issues.append("required file is missing")
    else:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - audit reports malformed JSON.
            issues.append(f"invalid JSON: {exc}")
        else:
            if not isinstance(payload, Mapping):
                issues.append("JSON payload must be an object")
            else:
                schema = payload.get("schema")
                schema_ok = schema == expected_schema
                if not schema_ok:
                    issues.append(
                        f"schema mismatch: expected {expected_schema}, got {schema}"
                    )
                sample_id = _extract_sample_id(payload)
                status = _extract_status(payload)
                try:
                    validator(payload)
                except Exception as exc:  # noqa: BLE001 - keep audit tolerant.
                    issues.append(f"validator failed: {exc}")
                else:
                    validator_ok = True
                    payloads[key] = payload
    return {
        **base,
        "schema": schema,
        "expected_schema": expected_schema,
        "schema_ok": schema_ok,
        "validator_ok": validator_ok,
        "sample_id": sample_id,
        "status": status,
        "issues": issues,
    }


def _markdown_file_record(key: str, path: Path, *, required: bool) -> dict[str, Any]:
    base = _base_file_record(key, path, required=required, kind="markdown")
    issues = []
    if not base["is_file"] and required:
        issues.append("required file is missing")
    return {
        **base,
        "schema": None,
        "expected_schema": None,
        "schema_ok": base["is_file"] or not required,
        "validator_ok": base["is_file"] or not required,
        "sample_id": None,
        "status": None,
        "issues": issues,
    }


def _base_file_record(
    key: str,
    path: Path,
    *,
    required: bool,
    kind: str,
) -> dict[str, Any]:
    exists = path.exists()
    is_file = exists and path.is_file()
    return {
        "key": key,
        "path": str(path),
        "required": bool(required),
        "kind": kind,
        "exists": bool(exists),
        "is_file": bool(is_file),
        "size_bytes": int(path.stat().st_size) if is_file else 0,
    }


def _validate_file_record(record: Any) -> None:
    if not isinstance(record, Mapping):
        raise TypeError("transition reality evidence file records must be mappings")
    for key in ("key", "path", "kind"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"transition reality evidence file record requires {key}")
    for key in (
        "required",
        "exists",
        "is_file",
        "schema_ok",
        "validator_ok",
    ):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"transition reality evidence file record requires bool {key}")
    if isinstance(record.get("size_bytes"), bool) or not isinstance(
        record.get("size_bytes"), int
    ):
        raise ValueError("transition reality evidence file size must be int")
    if not isinstance(record.get("issues"), list):
        raise ValueError("transition reality evidence file issues must be list")


def _extract_sample_id(payload: Mapping[str, Any]) -> str | None:
    if isinstance(payload.get("sample_id"), str):
        return payload["sample_id"]
    sample = payload.get("sample")
    if isinstance(sample, Mapping) and isinstance(sample.get("sample_id"), str):
        return sample["sample_id"]
    controlled_summary = payload.get("controlled_real_summary")
    if isinstance(controlled_summary, Mapping):
        sample = controlled_summary.get("sample")
        if isinstance(sample, Mapping) and isinstance(sample.get("sample_id"), str):
            return sample["sample_id"]
    return None


def _extract_status(payload: Mapping[str, Any]) -> str | None:
    if isinstance(payload.get("status"), str):
        return payload["status"]
    gate = payload.get("gate")
    if isinstance(gate, Mapping) and isinstance(gate.get("status"), str):
        return gate["status"]
    return None


def _sample_consistency(files: list[dict[str, Any]]) -> dict[str, Any]:
    values = {
        record["key"]: record["sample_id"]
        for record in files
        if isinstance(record.get("sample_id"), str) and record["sample_id"]
    }
    unique = sorted(set(values.values()))
    return {
        "consistent": len(unique) == 1,
        "sample_id": unique[0] if len(unique) == 1 else None,
        "unique_sample_ids": unique,
        "values": values,
    }


def _transition_summary(payload: Any) -> dict[str, Any]:
    readiness = payload.get("readiness") if isinstance(payload, Mapping) else None
    metrics = payload.get("metrics") if isinstance(payload, Mapping) else None
    return {
        "status": _extract_status(payload) if isinstance(payload, Mapping) else None,
        "transition_dataset_ready": bool(
            isinstance(readiness, Mapping)
            and readiness.get("transition_dataset_ready") is True
        ),
        "transition_count": (
            int(metrics.get("transition_count", 0))
            if isinstance(metrics, Mapping)
            and not isinstance(metrics.get("transition_count"), bool)
            and isinstance(metrics.get("transition_count"), int)
            else 0
        ),
        "action_conditioned_transition_count": (
            int(metrics.get("action_conditioned_transition_count", 0))
            if isinstance(metrics, Mapping)
            and not isinstance(metrics.get("action_conditioned_transition_count"), bool)
            and isinstance(metrics.get("action_conditioned_transition_count"), int)
            else 0
        ),
    }


def _handoff_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {
            "status": None,
            "prediction_eval_status": None,
            "intervention_eval_status": None,
            "partial_reality_gate_status": None,
            "requires_identity_pass_row": True,
        }
    controlled_real_summary = payload.get("controlled_real_summary")
    gate = (
        controlled_real_summary.get("gate")
        if isinstance(controlled_real_summary, Mapping)
        else None
    )
    thresholds = gate.get("thresholds") if isinstance(gate, Mapping) else None
    return {
        "status": payload.get("status"),
        "prediction_eval_status": _extract_status(payload.get("prediction_eval", {})),
        "intervention_eval_status": _extract_status(
            payload.get("intervention_eval", {})
        ),
        "partial_reality_gate_status": (
            gate.get("status") if isinstance(gate, Mapping) else None
        ),
        "requires_identity_pass_row": bool(
            not isinstance(thresholds, Mapping)
            or thresholds.get("require_identity_pass_row") is not False
        ),
    }


def _row_accounting(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {
            "present": False,
            "row_count": 0,
            "pass_row_count": 0,
            "fail_row_count": 0,
            "blocked_row_count": 0,
            "evidence_kinds": [],
            "identity_prediction_intervention_rows_present": False,
            "identity_row_status": None,
            "prediction_row_status": None,
            "intervention_row_status": None,
            "identity_row_blocked": False,
        }
    rows = payload.get("rows")
    rows_list = rows if isinstance(rows, list) else []
    status_by_kind = {
        row["evidence_kind"]: row["status"]
        for row in rows_list
        if isinstance(row, Mapping)
        and isinstance(row.get("evidence_kind"), str)
        and isinstance(row.get("status"), str)
    }
    evidence_kinds = sorted(status_by_kind)
    required_kinds = {"identity", "prediction", "intervention"}
    return {
        "present": all(
            isinstance(payload.get(key), int)
            for key in (
                "row_count",
                "pass_row_count",
                "fail_row_count",
                "blocked_row_count",
            )
        ),
        "row_count": int(payload.get("row_count", 0)),
        "pass_row_count": int(payload.get("pass_row_count", 0)),
        "fail_row_count": int(payload.get("fail_row_count", 0)),
        "blocked_row_count": int(payload.get("blocked_row_count", 0)),
        "evidence_kinds": evidence_kinds,
        "identity_prediction_intervention_rows_present": required_kinds.issubset(
            set(evidence_kinds)
        ),
        "identity_row_status": status_by_kind.get("identity"),
        "prediction_row_status": status_by_kind.get("prediction"),
        "intervention_row_status": status_by_kind.get("intervention"),
        "identity_row_blocked": status_by_kind.get("identity") == "blocked",
    }


def _standalone_output_consistency(payloads: Mapping[str, Any]) -> dict[str, Any]:
    handoff = payloads.get("transition_reality_handoff_summary")
    issues: list[str] = []
    if not isinstance(handoff, Mapping):
        return {"matches": False, "issues": ["handoff summary is missing or invalid"]}
    _compare_status(
        issues,
        "prediction eval",
        handoff.get("prediction_eval"),
        payloads.get("prediction_eval_summary"),
    )
    _compare_status(
        issues,
        "intervention eval",
        handoff.get("intervention_eval"),
        payloads.get("intervention_eval_summary"),
    )
    _compare_controlled_real_summary(
        issues,
        handoff.get("controlled_real_summary"),
        payloads.get("controlled_real_summary"),
    )
    _compare_payload(
        issues,
        "prediction candidates",
        handoff.get("prediction_candidate_summary", {}).get("prediction_candidates")
        if isinstance(handoff.get("prediction_candidate_summary"), Mapping)
        else None,
        payloads.get("prediction_candidates"),
    )
    _compare_payload(
        issues,
        "intervention candidates",
        handoff.get("intervention_candidate_summary", {}).get("intervention_candidates")
        if isinstance(handoff.get("intervention_candidate_summary"), Mapping)
        else None,
        payloads.get("intervention_candidates"),
    )
    if handoff.get("controlled_real_manifest") != payloads.get("controlled_real_manifest"):
        issues.append("controlled real manifest does not match handoff summary")
    return {"matches": not issues, "issues": issues}


def _compare_status(
    issues: list[str],
    label: str,
    embedded: Any,
    standalone: Any,
) -> None:
    if not isinstance(embedded, Mapping) or not isinstance(standalone, Mapping):
        issues.append(f"{label} standalone output is missing or invalid")
        return
    if embedded.get("status") != standalone.get("status"):
        issues.append(f"{label} status does not match handoff summary")
    embedded_candidate = embedded.get("candidate")
    standalone_candidate = standalone.get("candidate")
    if isinstance(embedded_candidate, Mapping) and isinstance(
        standalone_candidate, Mapping
    ):
        if embedded_candidate.get("candidate_id") != standalone_candidate.get(
            "candidate_id"
        ):
            issues.append(f"{label} candidate_id does not match handoff summary")


def _compare_controlled_real_summary(
    issues: list[str],
    embedded: Any,
    standalone: Any,
) -> None:
    if not isinstance(embedded, Mapping) or not isinstance(standalone, Mapping):
        issues.append("controlled real summary standalone output is missing or invalid")
        return
    for key in ("row_count", "pass_row_count", "fail_row_count", "blocked_row_count"):
        if embedded.get(key) != standalone.get(key):
            issues.append(f"controlled real summary {key} does not match handoff")
    embedded_gate = embedded.get("gate")
    standalone_gate = standalone.get("gate")
    if isinstance(embedded_gate, Mapping) and isinstance(standalone_gate, Mapping):
        if embedded_gate.get("status") != standalone_gate.get("status"):
            issues.append("controlled real gate status does not match handoff")


def _compare_payload(
    issues: list[str],
    label: str,
    embedded: Any,
    standalone: Any,
) -> None:
    if embedded != standalone:
        issues.append(f"{label} does not match handoff summary")


def _package_issues(
    files: list[dict[str, Any]],
    gates: Mapping[str, bool],
    sample_consistency: Mapping[str, Any],
    output_consistency: Mapping[str, Any],
) -> list[str]:
    issues = []
    for record in files:
        for issue in record["issues"]:
            issues.append(f"{record['key']}: {issue}")
    if not sample_consistency["consistent"]:
        issues.append(
            "sample ids are not consistent: "
            f"{sample_consistency['unique_sample_ids']}"
        )
    for issue in output_consistency["issues"]:
        issues.append(f"output consistency: {issue}")
    for key, passed in gates.items():
        if not passed:
            issues.append(f"reviewability gate failed: {key}")
    return issues


def _resolve_package_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path
