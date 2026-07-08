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
from objgauss.core.objectstate_controlled_reality_bundle_handoff import (
    OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_HANDOFF_SCHEMA,
    validate_objectstate_controlled_reality_bundle_handoff_summary,
)
from objgauss.core.objectstate_controlled_reality_bundle_readiness import (
    OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_READINESS_SCHEMA,
    validate_objectstate_controlled_reality_bundle_readiness_summary,
)
from objgauss.core.objectstate_controlled_reality_candidate_template import (
    OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_FINALIZE_SCHEMA,
    OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA,
    validate_objectstate_controlled_reality_candidate_finalize_summary,
    validate_objectstate_controlled_reality_candidate_template_summary,
)

OBJECTSTATE_CONTROLLED_REALITY_EVIDENCE_PACKAGE_SCHEMA = (
    "objgauss-objectstate-controlled-reality-evidence-package-v1"
)

Validator = Callable[[Any], Mapping[str, Any]]


def objectstate_controlled_reality_evidence_package(
    package_root: str | Path,
    *,
    candidate_dir: str | Path = "reality-candidates",
    handoff_dir: str | Path = "reality-handoff",
) -> dict[str, Any]:
    root = Path(package_root)
    candidate_root = _resolve_package_path(root, candidate_dir)
    handoff_root = _resolve_package_path(root, handoff_dir)
    payloads: dict[str, Any] = {}
    files = [
        _json_file_record(
            "template_summary",
            candidate_root / "template-summary.json",
            OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA,
            validate_objectstate_controlled_reality_candidate_template_summary,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "finalize_summary",
            candidate_root / "finalize-summary.json",
            OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_FINALIZE_SCHEMA,
            validate_objectstate_controlled_reality_candidate_finalize_summary,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "full_readiness_summary",
            candidate_root / "full-readiness-summary.json",
            OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_READINESS_SCHEMA,
            validate_objectstate_controlled_reality_bundle_readiness_summary,
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
            "intervention_candidates",
            candidate_root / "intervention-candidates.json",
            OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
            validate_objectstate_controlled_intervention_candidates,
            required=True,
            payloads=payloads,
        ),
        _json_file_record(
            "reality_bundle_handoff_summary",
            handoff_root / "reality-bundle-handoff-summary.json",
            OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_HANDOFF_SCHEMA,
            validate_objectstate_controlled_reality_bundle_handoff_summary,
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
            "intervention_eval_summary",
            handoff_root / "intervention-eval-summary.json",
            OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA,
            validate_objectstate_controlled_intervention_eval_summary,
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
        _json_file_record(
            "controlled_real_manifest",
            handoff_root / "controlled-real.json",
            OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
            validate_objectstate_controlled_real_manifest,
            required=False,
            payloads=payloads,
        ),
        _markdown_file_record(
            "blocked_rows_markdown",
            handoff_root / "blocked-rows.md",
            required=True,
        ),
    ]
    sample_consistency = _sample_consistency(files)
    output_consistency = _standalone_output_consistency(payloads)
    row_accounting = _row_accounting(payloads.get("controlled_real_summary"))
    readiness = _readiness_summary(payloads.get("full_readiness_summary"))
    handoff = _handoff_summary(payloads.get("reality_bundle_handoff_summary"))
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
        "full_reality_handoff_ready_recorded": bool(
            readiness["full_reality_handoff_ready"]
        ),
        "handoff_summary_present": "reality_bundle_handoff_summary" in payloads,
        "standalone_outputs_match_handoff_summary": bool(output_consistency["matches"]),
        "controlled_real_summary_present": "controlled_real_summary" in payloads,
        "row_accounting_present": bool(row_accounting["present"]),
        "identity_prediction_intervention_rows_present": bool(
            row_accounting["identity_prediction_intervention_rows_present"]
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
        "objectstate_controlled_reality_evidence_package_reviewable"
        if all(gates.values())
        else "objectstate_controlled_reality_evidence_package_incomplete"
    )
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_REALITY_EVIDENCE_PACKAGE_SCHEMA,
        "kind": "objectstate_controlled_reality_evidence_package",
        "status": status,
        "package_root": str(root),
        "candidate_dir": str(candidate_root),
        "handoff_dir": str(handoff_root),
        "sample_id": sample_consistency["sample_id"],
        "files": files,
        "sample_consistency": sample_consistency,
        "readiness": readiness,
        "handoff": handoff,
        "row_accounting": row_accounting,
        "output_consistency": output_consistency,
        "reviewability_gates": gates,
        "issues": _package_issues(files, gates, sample_consistency, output_consistency),
        "claim_policy": {
            "read_only_audit": True,
            "checks_local_evidence_package": True,
            "does_not_create_ground_truth": True,
            "does_not_run_identity_handoff": True,
            "does_not_run_prediction_eval": True,
            "does_not_run_intervention_eval": True,
            "does_not_claim_metric_pass": True,
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
    return validate_objectstate_controlled_reality_evidence_package_summary(payload)


def validate_objectstate_controlled_reality_evidence_package_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled reality evidence package summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_REALITY_EVIDENCE_PACKAGE_SCHEMA:
        raise ValueError(
            "unsupported controlled reality evidence package schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_reality_evidence_package":
        raise ValueError("controlled reality evidence package kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_reality_evidence_package_reviewable",
        "objectstate_controlled_reality_evidence_package_incomplete",
    }:
        raise ValueError("controlled reality evidence package status is unsupported")
    for key in ("package_root", "candidate_dir", "handoff_dir"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"controlled reality evidence package requires {key}")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("controlled reality evidence package requires files")
    for record in files:
        _validate_file_record(record)
    gates = payload.get("reviewability_gates")
    if not isinstance(gates, Mapping) or not gates:
        raise ValueError("controlled reality evidence package requires gates")
    if any(not isinstance(value, bool) for value in gates.values()):
        raise ValueError("controlled reality evidence package gates must be bool")
    expected_status = (
        "objectstate_controlled_reality_evidence_package_reviewable"
        if all(gates.values())
        else "objectstate_controlled_reality_evidence_package_incomplete"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled reality evidence package status mismatch")
    sample_consistency = payload.get("sample_consistency")
    if not isinstance(sample_consistency, Mapping):
        raise ValueError("controlled reality evidence package requires sample consistency")
    if not isinstance(sample_consistency.get("values"), Mapping):
        raise ValueError("controlled reality evidence package sample values invalid")
    row_accounting = payload.get("row_accounting")
    if not isinstance(row_accounting, Mapping):
        raise ValueError("controlled reality evidence package requires row accounting")
    for key in ("present", "identity_prediction_intervention_rows_present"):
        if not isinstance(row_accounting.get(key), bool):
            raise ValueError(f"controlled reality evidence package row gate invalid: {key}")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("controlled reality evidence package issues must be list")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("read_only_audit")
        or not claim_policy.get("checks_local_evidence_package")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_run_identity_handoff")
        or not claim_policy.get("does_not_run_prediction_eval")
        or not claim_policy.get("does_not_run_intervention_eval")
        or not claim_policy.get("does_not_claim_metric_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled reality evidence package must preserve claim policy")
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
            "controlled reality evidence package cannot claim capture, GT, "
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
                except Exception as exc:  # noqa: BLE001 - keep audit read-only/tolerant.
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
        raise TypeError("controlled reality evidence file records must be mappings")
    for key in ("key", "path", "kind"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"controlled reality evidence file record requires {key}")
    for key in (
        "required",
        "exists",
        "is_file",
        "schema_ok",
        "validator_ok",
    ):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"controlled reality evidence file record requires bool {key}")
    if isinstance(record.get("size_bytes"), bool) or not isinstance(
        record.get("size_bytes"), int
    ):
        raise ValueError("controlled reality evidence file size must be int")
    if not isinstance(record.get("issues"), list):
        raise ValueError("controlled reality evidence file issues must be list")


def _extract_sample_id(payload: Mapping[str, Any]) -> str | None:
    sample = payload.get("sample")
    if isinstance(sample, Mapping) and isinstance(sample.get("sample_id"), str):
        return sample["sample_id"]
    if isinstance(payload.get("sample_id"), str):
        return payload["sample_id"]
    capture_readiness = payload.get("capture_readiness")
    if isinstance(capture_readiness, Mapping):
        imported = capture_readiness.get("import_summary")
        if isinstance(imported, Mapping):
            manifest = imported.get("manifest")
            if isinstance(manifest, Mapping):
                sample = manifest.get("sample")
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


def _readiness_summary(payload: Any) -> dict[str, Any]:
    readiness = payload.get("readiness") if isinstance(payload, Mapping) else None
    return {
        "status": _extract_status(payload) if isinstance(payload, Mapping) else None,
        "full_reality_handoff_ready": bool(
            isinstance(readiness, Mapping)
            and readiness.get("full_reality_handoff_ready") is True
        ),
    }


def _handoff_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {
            "status": None,
            "full_reality_gate_status": None,
            "prediction_eval_status": None,
            "intervention_eval_status": None,
        }
    controlled_real_summary = payload.get("controlled_real_summary")
    gate = (
        controlled_real_summary.get("gate")
        if isinstance(controlled_real_summary, Mapping)
        else None
    )
    return {
        "status": payload.get("status"),
        "full_reality_gate_status": (
            gate.get("status") if isinstance(gate, Mapping) else None
        ),
        "prediction_eval_status": _extract_status(payload.get("prediction_eval", {})),
        "intervention_eval_status": _extract_status(
            payload.get("intervention_eval", {})
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
        }
    rows = payload.get("rows")
    evidence_kinds = sorted(
        {
            row.get("evidence_kind")
            for row in rows
            if isinstance(row, Mapping) and isinstance(row.get("evidence_kind"), str)
        }
    ) if isinstance(rows, list) else []
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
    }


def _standalone_output_consistency(payloads: Mapping[str, Any]) -> dict[str, Any]:
    handoff = payloads.get("reality_bundle_handoff_summary")
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
    if "controlled_real_manifest" in payloads and handoff.get("controlled_real_manifest") != payloads.get("controlled_real_manifest"):
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
