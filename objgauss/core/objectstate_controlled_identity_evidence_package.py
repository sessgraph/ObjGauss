from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    validate_objectstate_controlled_capture_manifest,
)
from objgauss.core.objectstate_controlled_capture_files import (
    OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA,
    validate_objectstate_controlled_capture_file_audit_summary,
)
from objgauss.core.objectstate_controlled_identity_eval import (
    OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA,
    OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA,
    validate_objectstate_controlled_identity_eval_summary,
    validate_objectstate_controlled_identity_predictions,
)
from objgauss.core.objectstate_controlled_identity_handoff import (
    OBJECTSTATE_CONTROLLED_CANDIDATE_ARTIFACT_FILE_AUDIT_SCHEMA,
    OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA,
    OBJECTSTATE_CONTROLLED_IDENTITY_SCENARIO_AUDIT_SCHEMA,
    validate_objectstate_controlled_identity_handoff_summary,
)
from objgauss.core.objectstate_controlled_real_rows import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
    OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
    validate_objectstate_controlled_real_manifest,
    validate_objectstate_controlled_real_rows_summary,
)

OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA = (
    "objgauss-objectstate-controlled-identity-evidence-package-v1"
)

Validator = Callable[[Any], Mapping[str, Any]]


def objectstate_controlled_identity_evidence_package(
    package_root: str | Path,
    *,
    capture_manifest: str | Path = "capture-manifest.json",
    capture_file_audit: str | Path = "capture-file-audit.json",
    capture_missing_files_markdown: str | Path = "capture-missing-files.md",
    candidate_artifact_file_audit: str | Path = "candidate-artifact-file-audit.json",
    identity_scenario_audit: str | Path = "identity-scenario-audit.json",
    identity_predictions: str | Path = "identity-predictions.json",
    identity_eval: str | Path = "identity-eval-summary.json",
    controlled_real: str | Path = "controlled-real.json",
    controlled_real_summary: str | Path = "controlled-real-summary.json",
    blocked_rows_markdown: str | Path = "blocked-rows.md",
    handoff_summary: str | Path = "handoff-summary.json",
) -> dict[str, Any]:
    root = Path(package_root)
    payloads: dict[str, Any] = {}
    files = [
        _json_file_record(
            "capture_manifest",
            _resolve_package_path(root, capture_manifest),
            OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
            validate_objectstate_controlled_capture_manifest,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "capture_file_audit",
            _resolve_package_path(root, capture_file_audit),
            OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA,
            validate_objectstate_controlled_capture_file_audit_summary,
            required=True,
            payloads=payloads,
        ),
        _markdown_file_record(
            "capture_missing_files_markdown",
            _resolve_package_path(root, capture_missing_files_markdown),
            required=True,
        ),
        _json_file_record(
            "candidate_artifact_file_audit",
            _resolve_package_path(root, candidate_artifact_file_audit),
            OBJECTSTATE_CONTROLLED_CANDIDATE_ARTIFACT_FILE_AUDIT_SCHEMA,
            _schema_only_validator,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "identity_scenario_audit",
            _resolve_package_path(root, identity_scenario_audit),
            OBJECTSTATE_CONTROLLED_IDENTITY_SCENARIO_AUDIT_SCHEMA,
            _schema_only_validator,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "identity_predictions",
            _resolve_package_path(root, identity_predictions),
            OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA,
            validate_objectstate_controlled_identity_predictions,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "identity_eval_summary",
            _resolve_package_path(root, identity_eval),
            OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA,
            validate_objectstate_controlled_identity_eval_summary,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "controlled_real",
            _resolve_package_path(root, controlled_real),
            OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
            validate_objectstate_controlled_real_manifest,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "controlled_real_summary",
            _resolve_package_path(root, controlled_real_summary),
            OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
            validate_objectstate_controlled_real_rows_summary,
            required=True,
            payloads=payloads,
        ),
        _markdown_file_record(
            "blocked_rows_markdown",
            _resolve_package_path(root, blocked_rows_markdown),
            required=True,
        ),
        _json_file_record(
            "handoff_summary",
            _resolve_package_path(root, handoff_summary),
            OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA,
            validate_objectstate_controlled_identity_handoff_summary,
            required=True,
            payloads=payloads,
        ),
    ]
    sample_consistency = _sample_consistency(files)
    handoff_consistency = _handoff_consistency(payloads)
    identity = _identity_summary(
        payloads.get("identity_eval_summary"),
        payloads.get("controlled_real"),
    )
    evidence = _evidence_summary(payloads)
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
        "capture_file_audit_pass": bool(evidence["capture_file_audit_pass"]),
        "candidate_artifact_file_audit_pass": bool(
            evidence["candidate_artifact_file_audit_pass"]
        ),
        "candidate_artifact_ref_match": bool(evidence["candidate_artifact_ref_match"]),
        "identity_scenario_audit_pass": bool(evidence["identity_scenario_audit_pass"]),
        "identity_predictions_present": bool(identity["identity_prediction_count"] > 0),
        "identity_eval_present": bool(identity["identity_eval_present"]),
        "identity_row_present": bool(identity["identity_row_present"]),
        "identity_row_is_pass_or_fail": bool(
            identity["identity_row_status"] in {"pass", "fail"}
        ),
        "controlled_real_output_matches_eval": bool(
            handoff_consistency["controlled_real_matches_eval"]
        ),
        "standalone_outputs_match_handoff": bool(
            handoff_consistency["standalone_outputs_match_handoff"]
        ),
        "identity_only_gate_does_not_require_prediction_or_intervention": bool(
            evidence["identity_only_gate_scope"]
        ),
    }
    status = (
        "objectstate_controlled_identity_evidence_package_reviewable"
        if all(gates.values())
        else "objectstate_controlled_identity_evidence_package_incomplete"
    )
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA,
        "kind": "objectstate_controlled_identity_evidence_package",
        "status": status,
        "package_root": str(root),
        "sample_id": sample_consistency["sample_id"],
        "files": files,
        "sample_consistency": sample_consistency,
        "identity": identity,
        "evidence": evidence,
        "handoff_consistency": handoff_consistency,
        "reviewability_gates": gates,
        "issues": _package_issues(
            files,
            gates,
            sample_consistency,
            handoff_consistency,
        ),
        "claim_policy": {
            "read_only_audit": True,
            "checks_local_identity_evidence_package": True,
            "requires_real_gaussian_file_acceptance": True,
            "requires_candidate_artifact_audit": True,
            "requires_identity_scenario_audit": True,
            "reviewable_allows_identity_pass_or_fail": True,
            "does_not_create_ground_truth": True,
            "does_not_run_identity_eval": True,
            "does_not_claim_metric_pass": True,
            "does_not_claim_prediction_or_intervention_gate": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
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
    return validate_objectstate_controlled_identity_evidence_package_summary(payload)


def validate_objectstate_controlled_identity_evidence_package_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled identity evidence package summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA:
        raise ValueError(
            "unsupported controlled identity evidence package schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_identity_evidence_package":
        raise ValueError("controlled identity evidence package kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_identity_evidence_package_reviewable",
        "objectstate_controlled_identity_evidence_package_incomplete",
    }:
        raise ValueError("controlled identity evidence package status is unsupported")
    if not isinstance(payload.get("package_root"), str) or not payload["package_root"]:
        raise ValueError("controlled identity evidence package requires package_root")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("controlled identity evidence package requires files")
    for record in files:
        _validate_file_record(record)
    sample_consistency = payload.get("sample_consistency")
    if not isinstance(sample_consistency, Mapping):
        raise ValueError("controlled identity evidence package requires sample consistency")
    if not isinstance(sample_consistency.get("values"), Mapping):
        raise ValueError("controlled identity evidence package sample values invalid")
    for key in ("identity", "evidence", "handoff_consistency"):
        if not isinstance(payload.get(key), Mapping):
            raise ValueError(f"controlled identity evidence package requires {key}")
    gates = payload.get("reviewability_gates")
    if not isinstance(gates, Mapping) or not gates:
        raise ValueError("controlled identity evidence package requires gates")
    if any(not isinstance(value, bool) for value in gates.values()):
        raise ValueError("controlled identity evidence package gates must be bool")
    expected_status = (
        "objectstate_controlled_identity_evidence_package_reviewable"
        if all(gates.values())
        else "objectstate_controlled_identity_evidence_package_incomplete"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled identity evidence package status mismatch")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("controlled identity evidence package issues must be list")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("read_only_audit")
        or not claim_policy.get("checks_local_identity_evidence_package")
        or not claim_policy.get("requires_real_gaussian_file_acceptance")
        or not claim_policy.get("requires_candidate_artifact_audit")
        or not claim_policy.get("requires_identity_scenario_audit")
        or not claim_policy.get("reviewable_allows_identity_pass_or_fail")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_run_identity_eval")
        or not claim_policy.get("does_not_claim_metric_pass")
        or not claim_policy.get("does_not_claim_prediction_or_intervention_gate")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled identity evidence package must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "controlled identity evidence package cannot claim capture, GT, "
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
        raise TypeError("controlled identity evidence file records must be mappings")
    for key in ("key", "path", "kind"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"controlled identity evidence file record requires {key}")
    for key in ("required", "exists", "is_file", "schema_ok", "validator_ok"):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"controlled identity evidence file record requires bool {key}")
    if isinstance(record.get("size_bytes"), bool) or not isinstance(
        record.get("size_bytes"), int
    ):
        raise ValueError("controlled identity evidence file size must be int")
    if not isinstance(record.get("issues"), list):
        raise ValueError("controlled identity evidence file issues must be list")


def _schema_only_validator(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("schema-only identity evidence file must be a mapping")
    return payload


def _extract_sample_id(payload: Mapping[str, Any]) -> str | None:
    sample = payload.get("sample")
    if isinstance(sample, Mapping) and isinstance(sample.get("sample_id"), str):
        return sample["sample_id"]
    if isinstance(payload.get("sample_id"), str):
        return payload["sample_id"]
    manifest = payload.get("manifest")
    if isinstance(manifest, Mapping):
        sample = manifest.get("sample")
        if isinstance(sample, Mapping) and isinstance(sample.get("sample_id"), str):
            return sample["sample_id"]
    controlled_real = payload.get("controlled_real_manifest")
    if isinstance(controlled_real, Mapping):
        sample = controlled_real.get("sample")
        if isinstance(sample, Mapping) and isinstance(sample.get("sample_id"), str):
            return sample["sample_id"]
    return None


def _extract_status(payload: Any) -> str | None:
    if isinstance(payload, Mapping) and isinstance(payload.get("status"), str):
        return payload["status"]
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


def _identity_summary(identity_eval: Any, controlled_real: Any) -> dict[str, Any]:
    identity_row = None
    rows = controlled_real.get("evidence_rows") if isinstance(controlled_real, Mapping) else None
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping) and row.get("evidence_kind") == "identity":
                identity_row = row
                break
    metrics = identity_eval.get("metrics") if isinstance(identity_eval, Mapping) else None
    return {
        "identity_eval_present": isinstance(identity_eval, Mapping),
        "identity_eval_status": _extract_status(identity_eval),
        "identity_prediction_count": int(
            metrics.get("predicted_pair_count", metrics.get("prediction_count", 0))
        )
        if isinstance(metrics, Mapping)
        else 0,
        "idf1": metrics.get("idf1") if isinstance(metrics, Mapping) else None,
        "track_retrieval_recall_at_1": metrics.get("track_retrieval_recall_at_1")
        if isinstance(metrics, Mapping)
        else None,
        "fragmentation_rate": metrics.get("fragmentation_rate")
        if isinstance(metrics, Mapping)
        else None,
        "swap_rate": metrics.get("swap_rate") if isinstance(metrics, Mapping) else None,
        "reconstruction_noise_robustness": metrics.get(
            "reconstruction_noise_robustness"
        )
        if isinstance(metrics, Mapping)
        else None,
        "controlled_real_manifest_present": isinstance(controlled_real, Mapping),
        "identity_row_present": isinstance(identity_row, Mapping),
        "identity_row_status": identity_row.get("status")
        if isinstance(identity_row, Mapping)
        else None,
    }


def _evidence_summary(payloads: Mapping[str, Any]) -> dict[str, Any]:
    capture_file_audit = payloads.get("capture_file_audit")
    candidate_file_audit = payloads.get("candidate_artifact_file_audit")
    scenario_audit = payloads.get("identity_scenario_audit")
    handoff = payloads.get("handoff_summary")
    controlled_real_summary = payloads.get("controlled_real_summary")
    handoff_ref_match = (
        handoff.get("candidate_artifact_ref_match")
        if isinstance(handoff, Mapping)
        else None
    )
    hard_blockers = []
    if isinstance(controlled_real_summary, Mapping):
        gate = controlled_real_summary.get("gate")
        if isinstance(gate, Mapping) and isinstance(gate.get("hard_blockers"), list):
            hard_blockers = list(gate["hard_blockers"])
    return {
        "capture_file_audit_pass": bool(
            isinstance(capture_file_audit, Mapping)
            and capture_file_audit.get("status")
            == "objectstate_controlled_capture_file_audit_pass"
        ),
        "candidate_artifact_file_audit_pass": bool(
            isinstance(candidate_file_audit, Mapping)
            and candidate_file_audit.get("status")
            == "objectstate_controlled_candidate_artifact_file_audit_pass"
        ),
        "candidate_artifact_ref_match": bool(
            isinstance(handoff_ref_match, Mapping)
            and handoff_ref_match.get("matches") is True
        ),
        "identity_scenario_audit_pass": bool(
            isinstance(scenario_audit, Mapping)
            and scenario_audit.get("status")
            == "objectstate_controlled_identity_scenario_audit_pass"
        ),
        "identity_only_gate_scope": (
            "prediction_pass_rows_present" not in hard_blockers
            and "intervention_pass_rows_present" not in hard_blockers
        ),
    }


def _handoff_consistency(payloads: Mapping[str, Any]) -> dict[str, Any]:
    handoff = payloads.get("handoff_summary")
    issues = []
    if not isinstance(handoff, Mapping):
        return {
            "controlled_real_matches_eval": False,
            "standalone_outputs_match_handoff": False,
            "issues": ["handoff summary is missing or invalid"],
        }
    comparisons = {
        "capture_file_audit": "capture_file_audit",
        "candidate_artifact_file_audit": "candidate_artifact_file_audit",
        "identity_scenario_audit": "identity_scenario_audit",
        "identity_predictions": "identity_predictions",
        "identity_eval_summary": "identity_eval",
        "controlled_real": "controlled_real_manifest",
        "controlled_real_summary": "controlled_real_summary",
    }
    for standalone_key, handoff_key in comparisons.items():
        standalone = payloads.get(standalone_key)
        embedded = handoff.get(handoff_key)
        if standalone is None:
            issues.append(f"{standalone_key} is missing")
        elif not _json_equivalent(standalone, embedded):
            issues.append(f"{standalone_key} does not match handoff_summary.{handoff_key}")
    identity_eval = payloads.get("identity_eval_summary")
    controlled_real = payloads.get("controlled_real")
    controlled_real_matches_eval = bool(
        isinstance(identity_eval, Mapping)
        and isinstance(controlled_real, Mapping)
        and _json_equivalent(
            identity_eval.get("controlled_real_manifest"),
            controlled_real,
        )
    )
    if not controlled_real_matches_eval:
        issues.append("controlled-real manifest does not match identity eval")
    return {
        "controlled_real_matches_eval": controlled_real_matches_eval,
        "standalone_outputs_match_handoff": not issues,
        "issues": issues,
    }


def _json_equivalent(left: Any, right: Any) -> bool:
    return _json_normalize(left) == _json_normalize(right)


def _json_normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    return value


def _package_issues(
    files: list[dict[str, Any]],
    gates: Mapping[str, bool],
    sample_consistency: Mapping[str, Any],
    handoff_consistency: Mapping[str, Any],
) -> list[str]:
    issues = []
    for record in files:
        for issue in record["issues"]:
            issues.append(f"{record['key']}: {issue}")
    if not sample_consistency["consistent"]:
        issues.append(
            "sample_id mismatch across package files: "
            + ", ".join(sample_consistency["unique_sample_ids"])
        )
    for issue in handoff_consistency["issues"]:
        issues.append(issue)
    for gate, passed in gates.items():
        if not passed:
            issues.append(f"reviewability gate failed: {gate}")
    return issues


def _resolve_package_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
