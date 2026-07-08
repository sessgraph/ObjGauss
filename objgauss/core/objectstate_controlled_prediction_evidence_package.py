from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from objgauss.core.objectstate_bop_capture_adapter import (
    OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
    validate_objectstate_bop_capture_acceptance_summary,
)
from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    validate_objectstate_controlled_capture_manifest,
)
from objgauss.core.objectstate_controlled_capture_files import (
    OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA,
    validate_objectstate_controlled_capture_file_audit_summary,
)
from objgauss.core.objectstate_controlled_prediction_eval import (
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
    OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA,
    validate_objectstate_controlled_prediction_candidates,
    validate_objectstate_controlled_prediction_eval_summary,
)
from objgauss.core.objectstate_controlled_real_rows import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
    validate_objectstate_controlled_real_manifest,
)
from objgauss.core.objectstate_controlled_reality_candidate_template import (
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATE_FINALIZE_SCHEMA,
    OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA,
    validate_objectstate_controlled_prediction_candidate_finalize_summary,
    validate_objectstate_controlled_reality_candidate_template_summary,
)

OBJECTSTATE_CONTROLLED_PREDICTION_EVIDENCE_PACKAGE_SCHEMA = (
    "objgauss-objectstate-controlled-prediction-evidence-package-v1"
)

Validator = Callable[[Any], Mapping[str, Any]]


def objectstate_controlled_prediction_evidence_package(
    package_root: str | Path,
    *,
    candidate_dir: str | Path = "reality-candidates",
    capture_manifest: str | Path = "capture-manifest.json",
    acceptance_summary: str | Path = "bop-acceptance-summary.json",
    file_audit: str | Path = "bop-file-audit.json",
    missing_files_markdown: str | Path = "bop-missing-files.md",
) -> dict[str, Any]:
    root = Path(package_root)
    candidate_root = _resolve_package_path(root, candidate_dir)
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
            "bop_acceptance_summary",
            _resolve_package_path(root, acceptance_summary),
            OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
            validate_objectstate_bop_capture_acceptance_summary,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "capture_file_audit",
            _resolve_package_path(root, file_audit),
            OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA,
            validate_objectstate_controlled_capture_file_audit_summary,
            required=True,
            payloads=payloads,
        ),
        _markdown_file_record(
            "missing_files_markdown",
            _resolve_package_path(root, missing_files_markdown),
            required=True,
        ),
        _json_file_record(
            "template_summary",
            candidate_root / "template-summary.json",
            OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA,
            validate_objectstate_controlled_reality_candidate_template_summary,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "prediction_finalize_summary",
            candidate_root / "prediction-finalize-summary.json",
            OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATE_FINALIZE_SCHEMA,
            validate_objectstate_controlled_prediction_candidate_finalize_summary,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "prediction_candidates",
            candidate_root / "prediction-candidates.json",
            OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
            validate_objectstate_controlled_prediction_candidates,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "prediction_eval_summary",
            candidate_root / "prediction-eval-summary.json",
            OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA,
            validate_objectstate_controlled_prediction_eval_summary,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "controlled_real_prediction",
            candidate_root / "controlled-real-prediction.json",
            OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
            validate_objectstate_controlled_real_manifest,
            required=True,
            payloads=payloads,
        ),
    ]
    sample_consistency = _sample_consistency(files)
    acceptance = _acceptance_summary(payloads.get("bop_acceptance_summary"))
    prediction = _prediction_summary(
        payloads.get("prediction_eval_summary"),
        payloads.get("controlled_real_prediction"),
    )
    output_consistency = _output_consistency(
        payloads.get("prediction_eval_summary"),
        payloads.get("controlled_real_prediction"),
    )
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
        "bop_acceptance_pass": bool(acceptance["bop_acceptance_pass"]),
        "phase1_gaussian_evidence_ready": bool(
            acceptance["phase1_gaussian_evidence_ready"]
        ),
        "capture_file_audit_pass": bool(acceptance["capture_file_audit_pass"]),
        "prediction_candidates_present": bool(
            prediction["prediction_candidate_count"] > 0
        ),
        "prediction_eval_present": bool(prediction["prediction_eval_present"]),
        "prediction_row_present": bool(prediction["prediction_row_present"]),
        "prediction_row_is_pass_or_fail": bool(
            prediction["prediction_row_status"] in {"pass", "fail"}
        ),
        "controlled_real_output_matches_eval": bool(output_consistency["matches"]),
    }
    status = (
        "objectstate_controlled_prediction_evidence_package_reviewable"
        if all(gates.values())
        else "objectstate_controlled_prediction_evidence_package_incomplete"
    )
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_PREDICTION_EVIDENCE_PACKAGE_SCHEMA,
        "kind": "objectstate_controlled_prediction_evidence_package",
        "status": status,
        "package_root": str(root),
        "candidate_dir": str(candidate_root),
        "sample_id": sample_consistency["sample_id"],
        "files": files,
        "sample_consistency": sample_consistency,
        "acceptance": acceptance,
        "prediction": prediction,
        "output_consistency": output_consistency,
        "reviewability_gates": gates,
        "issues": _package_issues(
            files,
            gates,
            sample_consistency,
            output_consistency,
        ),
        "claim_policy": {
            "read_only_audit": True,
            "checks_local_prediction_evidence_package": True,
            "requires_real_gaussian_file_acceptance": True,
            "does_not_create_ground_truth": True,
            "does_not_run_prediction_eval": True,
            "does_not_claim_metric_pass": True,
            "does_not_claim_intervention_gate": True,
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
    return validate_objectstate_controlled_prediction_evidence_package_summary(
        payload
    )


def validate_objectstate_controlled_prediction_evidence_package_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled prediction evidence package summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_PREDICTION_EVIDENCE_PACKAGE_SCHEMA:
        raise ValueError(
            "unsupported controlled prediction evidence package schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_prediction_evidence_package":
        raise ValueError("controlled prediction evidence package kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_prediction_evidence_package_reviewable",
        "objectstate_controlled_prediction_evidence_package_incomplete",
    }:
        raise ValueError("controlled prediction evidence package status is unsupported")
    for key in ("package_root", "candidate_dir"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"controlled prediction evidence package requires {key}")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("controlled prediction evidence package requires files")
    for record in files:
        _validate_file_record(record)
    sample_consistency = payload.get("sample_consistency")
    if not isinstance(sample_consistency, Mapping):
        raise ValueError("controlled prediction evidence package requires sample consistency")
    if not isinstance(sample_consistency.get("values"), Mapping):
        raise ValueError("controlled prediction evidence package sample values invalid")
    gates = payload.get("reviewability_gates")
    if not isinstance(gates, Mapping) or not gates:
        raise ValueError("controlled prediction evidence package requires gates")
    if any(not isinstance(value, bool) for value in gates.values()):
        raise ValueError("controlled prediction evidence package gates must be bool")
    expected_status = (
        "objectstate_controlled_prediction_evidence_package_reviewable"
        if all(gates.values())
        else "objectstate_controlled_prediction_evidence_package_incomplete"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled prediction evidence package status mismatch")
    acceptance = payload.get("acceptance")
    prediction = payload.get("prediction")
    output_consistency = payload.get("output_consistency")
    if not isinstance(acceptance, Mapping) or not isinstance(prediction, Mapping):
        raise ValueError("controlled prediction evidence package requires summaries")
    if not isinstance(output_consistency, Mapping):
        raise ValueError("controlled prediction evidence package requires output consistency")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("controlled prediction evidence package issues must be list")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("read_only_audit")
        or not claim_policy.get("checks_local_prediction_evidence_package")
        or not claim_policy.get("requires_real_gaussian_file_acceptance")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_run_prediction_eval")
        or not claim_policy.get("does_not_claim_metric_pass")
        or not claim_policy.get("does_not_claim_intervention_gate")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled prediction evidence package must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("reconstructs_gaussians")
        or non_goals.get("runs_tracking_model")
        or non_goals.get("runs_prediction_model")
        or non_goals.get("runs_intervention_model")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("writes_public_samples")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError(
            "controlled prediction evidence package cannot claim capture, GT, "
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
                    issues.append(f"schema mismatch: expected {expected_schema}, got {schema}")
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
        raise TypeError("controlled prediction evidence file records must be mappings")
    for key in ("key", "path", "kind"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"controlled prediction evidence file record requires {key}")
    for key in ("required", "exists", "is_file", "schema_ok", "validator_ok"):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"controlled prediction evidence file record requires bool {key}")
    if isinstance(record.get("size_bytes"), bool) or not isinstance(
        record.get("size_bytes"), int
    ):
        raise ValueError("controlled prediction evidence file size must be int")
    if not isinstance(record.get("issues"), list):
        raise ValueError("controlled prediction evidence file issues must be list")


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
    return None


def _extract_status(payload: Mapping[str, Any]) -> str | None:
    if isinstance(payload.get("status"), str):
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


def _acceptance_summary(payload: Any) -> dict[str, Any]:
    readiness = payload.get("readiness") if isinstance(payload, Mapping) else None
    return {
        "status": _extract_status(payload) if isinstance(payload, Mapping) else None,
        "bop_acceptance_pass": bool(
            isinstance(payload, Mapping)
            and payload.get("status") == "objectstate_bop_capture_acceptance_pass"
        ),
        "capture_file_audit_pass": bool(
            isinstance(readiness, Mapping)
            and readiness.get("capture_file_audit_pass") is True
        ),
        "phase1_gaussian_evidence_ready": bool(
            isinstance(readiness, Mapping)
            and readiness.get("phase1_gaussian_evidence_ready") is True
        ),
        "identity_stage_ready": bool(
            isinstance(readiness, Mapping)
            and readiness.get("identity_stage_ready") is True
        ),
        "prediction_stage_ready": bool(
            isinstance(readiness, Mapping)
            and readiness.get("prediction_stage_ready") is True
        ),
    }


def _prediction_summary(
    prediction_eval: Any,
    controlled_real: Any,
) -> dict[str, Any]:
    prediction_row = None
    rows = controlled_real.get("evidence_rows") if isinstance(controlled_real, Mapping) else None
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping) and row.get("evidence_kind") == "prediction":
                prediction_row = row
                break
    metrics = prediction_eval.get("metrics") if isinstance(prediction_eval, Mapping) else None
    return {
        "prediction_eval_present": isinstance(prediction_eval, Mapping),
        "prediction_eval_status": _extract_status(prediction_eval)
        if isinstance(prediction_eval, Mapping)
        else None,
        "prediction_candidate_count": int(metrics.get("prediction_count", 0))
        if isinstance(metrics, Mapping)
        else 0,
        "state_ade": metrics.get("state_ade") if isinstance(metrics, Mapping) else None,
        "history_ade": metrics.get("history_ade") if isinstance(metrics, Mapping) else None,
        "prediction_gap_vs_history_model": metrics.get(
            "prediction_gap_vs_history_model"
        )
        if isinstance(metrics, Mapping)
        else None,
        "controlled_real_manifest_present": isinstance(controlled_real, Mapping),
        "prediction_row_present": isinstance(prediction_row, Mapping),
        "prediction_row_status": prediction_row.get("status")
        if isinstance(prediction_row, Mapping)
        else None,
    }


def _output_consistency(prediction_eval: Any, controlled_real: Any) -> dict[str, Any]:
    issues = []
    if not isinstance(prediction_eval, Mapping):
        issues.append("prediction eval summary is missing or invalid")
    if not isinstance(controlled_real, Mapping):
        issues.append("controlled-real prediction manifest is missing or invalid")
    if issues:
        return {"matches": False, "issues": issues}
    embedded = prediction_eval.get("controlled_real_manifest")
    if embedded != controlled_real:
        issues.append("controlled-real prediction manifest does not match prediction eval")
    return {"matches": not issues, "issues": issues}


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
            "sample_id mismatch across package files: "
            + ", ".join(sample_consistency["unique_sample_ids"])
        )
    for issue in output_consistency["issues"]:
        issues.append(issue)
    for gate, passed in gates.items():
        if not passed:
            issues.append(f"reviewability gate failed: {gate}")
    return issues


def _resolve_package_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
