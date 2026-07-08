from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from objgauss.core.objectstate_controlled_identity_evidence_package import (
    OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA,
    validate_objectstate_controlled_identity_evidence_package_summary,
)
from objgauss.core.objectstate_controlled_prediction_evidence_package import (
    OBJECTSTATE_CONTROLLED_PREDICTION_EVIDENCE_PACKAGE_SCHEMA,
    validate_objectstate_controlled_prediction_evidence_package_summary,
)
from objgauss.core.objectstate_controlled_reality_evidence_package import (
    OBJECTSTATE_CONTROLLED_REALITY_EVIDENCE_PACKAGE_SCHEMA,
    validate_objectstate_controlled_reality_evidence_package_summary,
)

OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA = (
    "objgauss-objectstate-phase1-evidence-ledger-v1"
)
IDENTITY_EVIDENCE_PACKAGE_SUMMARY_FILENAMES = (
    "identity-evidence-package-summary.json",
)
PREDICTION_EVIDENCE_PACKAGE_SUMMARY_FILENAMES = (
    "prediction-evidence-package-summary.json",
)
REALITY_EVIDENCE_PACKAGE_SUMMARY_FILENAMES = (
    "evidence-package-summary.json",
    "reality-evidence-package-summary.json",
)

Validator = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def objectstate_phase1_evidence_ledger(
    *,
    identity_summaries: Sequence[str | Path] = (),
    prediction_summaries: Sequence[str | Path] = (),
    reality_summaries: Sequence[str | Path] = (),
    discover_roots: Sequence[str | Path] = (),
    max_depth: int = 4,
) -> dict[str, Any]:
    discovery = _discover_summary_paths(discover_roots, max_depth=max_depth)
    identity_paths = _dedupe_paths(
        (*identity_summaries, *discovery["paths"]["identity"])
    )
    prediction_paths = _dedupe_paths(
        (*prediction_summaries, *discovery["paths"]["prediction"])
    )
    reality_paths = _dedupe_paths(
        (*reality_summaries, *discovery["paths"]["full_reality"])
    )
    records = [
        *(
            _summary_record(
                path,
                stage="identity",
                expected_schema=OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA,
                validator=validate_objectstate_controlled_identity_evidence_package_summary,
            )
            for path in identity_paths
        ),
        *(
            _summary_record(
                path,
                stage="prediction",
                expected_schema=OBJECTSTATE_CONTROLLED_PREDICTION_EVIDENCE_PACKAGE_SCHEMA,
                validator=validate_objectstate_controlled_prediction_evidence_package_summary,
            )
            for path in prediction_paths
        ),
        *(
            _summary_record(
                path,
                stage="full_reality",
                expected_schema=OBJECTSTATE_CONTROLLED_REALITY_EVIDENCE_PACKAGE_SCHEMA,
                validator=validate_objectstate_controlled_reality_evidence_package_summary,
            )
            for path in reality_paths
        ),
    ]
    discovery_summary = _discovery_summary(discovery)
    stage_summary = _stage_summary(records)
    sample_scope = _sample_scope(records)
    ledger_gates = {
        "discovery_roots_valid": not discovery_summary["issues"],
        "summaries_present": bool(records),
        "all_files_present": all(record["is_file"] for record in records),
        "all_json_schemas_valid": all(record["schema_ok"] for record in records),
        "all_summary_validators_passed": all(
            record["validator_ok"] for record in records
        ),
    }
    phase1_gates = {
        "identity_evidence_reviewable": bool(
            stage_summary["identity"]["reviewable_count"] > 0
            or stage_summary["full_reality"]["reviewable_count"] > 0
        ),
        "prediction_evidence_reviewable": bool(
            stage_summary["prediction"]["reviewable_count"] > 0
            or stage_summary["full_reality"]["reviewable_count"] > 0
        ),
        "full_reality_evidence_reviewable": bool(
            stage_summary["full_reality"]["reviewable_count"] > 0
        ),
        "intervention_evidence_reviewable": bool(
            stage_summary["full_reality"]["reviewable_count"] > 0
        ),
        "does_not_claim_metric_pass": True,
        "does_not_claim_world_model": True,
    }
    status = (
        "objectstate_phase1_evidence_ledger_reviewable"
        if all(ledger_gates.values())
        else "objectstate_phase1_evidence_ledger_incomplete"
    )
    payload = {
        "schema": OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA,
        "kind": "objectstate_phase1_evidence_ledger",
        "status": status,
        "maturity": _maturity(stage_summary),
        "sample_scope": sample_scope,
        "discovery": discovery_summary,
        "summaries": records,
        "stage_summary": stage_summary,
        "ledger_gates": ledger_gates,
        "phase1_evidence_gates": phase1_gates,
        "issues": _issues(records, ledger_gates, discovery=discovery_summary),
        "claim_policy": {
            "read_only_audit": True,
            "checks_existing_evidence_package_summaries": True,
            "discovers_existing_evidence_package_summaries": True,
            "does_not_create_ground_truth": True,
            "does_not_run_identity_handoff": True,
            "does_not_run_identity_eval": True,
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
    return validate_objectstate_phase1_evidence_ledger_summary(payload)


def validate_objectstate_phase1_evidence_ledger_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("phase1 evidence ledger summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA:
        raise ValueError(
            f"unsupported phase1 evidence ledger schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_phase1_evidence_ledger":
        raise ValueError("phase1 evidence ledger kind is unsupported")
    if payload.get("status") not in {
        "objectstate_phase1_evidence_ledger_reviewable",
        "objectstate_phase1_evidence_ledger_incomplete",
    }:
        raise ValueError("phase1 evidence ledger status is unsupported")
    if payload.get("maturity") not in {
        "no_evidence_summaries",
        "evidence_summaries_present_not_reviewable",
        "identity_reviewable",
        "identity_prediction_reviewable",
        "full_reality_reviewable",
    }:
        raise ValueError("phase1 evidence ledger maturity is unsupported")
    summaries = payload.get("summaries")
    if not isinstance(summaries, list):
        raise ValueError("phase1 evidence ledger requires summaries")
    for record in summaries:
        _validate_record(record)
    sample_scope = payload.get("sample_scope")
    if not isinstance(sample_scope, Mapping):
        raise ValueError("phase1 evidence ledger requires sample_scope")
    if not isinstance(sample_scope.get("sample_ids"), list):
        raise ValueError("phase1 evidence ledger sample_ids must be list")
    _validate_discovery(payload.get("discovery"))
    stage_summary = payload.get("stage_summary")
    if not isinstance(stage_summary, Mapping):
        raise ValueError("phase1 evidence ledger requires stage_summary")
    for stage in ("identity", "prediction", "full_reality"):
        _validate_stage(stage_summary.get(stage), stage)
    ledger_gates = payload.get("ledger_gates")
    if not isinstance(ledger_gates, Mapping) or not ledger_gates:
        raise ValueError("phase1 evidence ledger requires ledger_gates")
    if any(not isinstance(value, bool) for value in ledger_gates.values()):
        raise ValueError("phase1 evidence ledger ledger_gates must be bool")
    expected_status = (
        "objectstate_phase1_evidence_ledger_reviewable"
        if all(ledger_gates.values())
        else "objectstate_phase1_evidence_ledger_incomplete"
    )
    if payload["status"] != expected_status:
        raise ValueError("phase1 evidence ledger status mismatch")
    phase1_gates = payload.get("phase1_evidence_gates")
    if not isinstance(phase1_gates, Mapping) or not phase1_gates:
        raise ValueError("phase1 evidence ledger requires phase1 evidence gates")
    if any(not isinstance(value, bool) for value in phase1_gates.values()):
        raise ValueError("phase1 evidence ledger phase gates must be bool")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("phase1 evidence ledger issues must be list")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("read_only_audit")
        or not claim_policy.get("checks_existing_evidence_package_summaries")
        or not claim_policy.get("discovers_existing_evidence_package_summaries")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_run_identity_handoff")
        or not claim_policy.get("does_not_run_identity_eval")
        or not claim_policy.get("does_not_run_prediction_eval")
        or not claim_policy.get("does_not_run_intervention_eval")
        or not claim_policy.get("does_not_claim_metric_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("phase1 evidence ledger must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "phase1 evidence ledger cannot claim capture, GT, reconstruction, "
            "models, training, public samples, replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def _discover_summary_paths(
    roots: Sequence[str | Path],
    *,
    max_depth: int,
) -> dict[str, Any]:
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("phase1 evidence ledger max_depth must be a non-negative int")
    paths: dict[str, list[Path]] = {
        "identity": [],
        "prediction": [],
        "full_reality": [],
    }
    issues: list[str] = []
    root_strings: list[str] = []
    for root in roots:
        root_path = Path(root)
        root_strings.append(str(root_path))
        if not root_path.exists():
            issues.append(f"discovery root is missing: {root_path}")
            continue
        if not root_path.is_dir():
            issues.append(f"discovery root is not a directory: {root_path}")
            continue
        for path in sorted(root_path.rglob("*.json")):
            if not path.is_file():
                continue
            if _relative_file_depth(root_path, path) > max_depth:
                continue
            stage = _summary_stage_for_filename(path.name)
            if stage is not None:
                paths[stage].append(path)
    return {
        "roots": root_strings,
        "max_depth": max_depth,
        "paths": {
            stage: _dedupe_paths(stage_paths)
            for stage, stage_paths in paths.items()
        },
        "issues": issues,
    }


def _summary_stage_for_filename(filename: str) -> str | None:
    if filename in IDENTITY_EVIDENCE_PACKAGE_SUMMARY_FILENAMES:
        return "identity"
    if filename in PREDICTION_EVIDENCE_PACKAGE_SUMMARY_FILENAMES:
        return "prediction"
    if filename in REALITY_EVIDENCE_PACKAGE_SUMMARY_FILENAMES:
        return "full_reality"
    return None


def _relative_file_depth(root: Path, path: Path) -> int:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return 0
    return max(len(relative.parts) - 1, 0)


def _dedupe_paths(paths: Sequence[str | Path]) -> list[str | Path]:
    seen: set[str] = set()
    deduped: list[str | Path] = []
    for path in paths:
        key = _path_key(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _path_key(path: str | Path) -> str:
    try:
        return str(Path(path).resolve(strict=False))
    except OSError:
        return str(Path(path).absolute())


def _discovery_summary(discovery: Mapping[str, Any]) -> dict[str, Any]:
    paths = discovery["paths"]
    discovered_paths = {
        "identity": [str(path) for path in paths["identity"]],
        "prediction": [str(path) for path in paths["prediction"]],
        "full_reality": [str(path) for path in paths["full_reality"]],
    }
    return {
        "roots": list(discovery["roots"]),
        "max_depth": int(discovery["max_depth"]),
        "identity_summary_count": len(discovered_paths["identity"]),
        "prediction_summary_count": len(discovered_paths["prediction"]),
        "reality_summary_count": len(discovered_paths["full_reality"]),
        "discovered_paths": discovered_paths,
        "issues": list(discovery["issues"]),
    }


def _summary_record(
    path: str | Path,
    *,
    stage: str,
    expected_schema: str,
    validator: Validator,
) -> dict[str, Any]:
    summary_path = Path(path)
    exists = summary_path.exists()
    is_file = exists and summary_path.is_file()
    issues: list[str] = []
    schema = None
    status = None
    sample_id = None
    schema_ok = False
    validator_ok = False
    reviewable = False
    row_status = None
    row_accounting = None
    if not is_file:
        issues.append("summary file is missing")
    else:
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - ledger reports malformed inputs.
            issues.append(f"invalid JSON: {exc}")
        else:
            if not isinstance(payload, Mapping):
                issues.append("summary JSON must be an object")
            else:
                schema = payload.get("schema")
                schema_ok = schema == expected_schema
                if not schema_ok:
                    issues.append(
                        f"schema mismatch: expected {expected_schema}, got {schema}"
                    )
                try:
                    checked = validator(payload)
                except Exception as exc:  # noqa: BLE001 - keep ledger tolerant.
                    issues.append(f"validator failed: {exc}")
                else:
                    validator_ok = True
                    status = checked.get("status")
                    sample_id = checked.get("sample_id")
                    reviewable = bool(
                        isinstance(status, str) and status.endswith("_reviewable")
                    )
                    row_status = _row_status(stage, checked)
                    row_accounting = _row_accounting(stage, checked)
    return {
        "stage": stage,
        "path": str(summary_path),
        "exists": bool(exists),
        "is_file": bool(is_file),
        "size_bytes": int(summary_path.stat().st_size) if is_file else 0,
        "schema": schema,
        "expected_schema": expected_schema,
        "schema_ok": schema_ok,
        "validator_ok": validator_ok,
        "status": status,
        "reviewable": reviewable,
        "sample_id": sample_id,
        "row_status": row_status,
        "row_accounting": row_accounting,
        "issues": issues,
    }


def _row_status(stage: str, payload: Mapping[str, Any]) -> str | None:
    if stage == "identity":
        identity = payload.get("identity")
        return identity.get("identity_row_status") if isinstance(identity, Mapping) else None
    if stage == "prediction":
        prediction = payload.get("prediction")
        return (
            prediction.get("prediction_row_status")
            if isinstance(prediction, Mapping)
            else None
        )
    return None


def _row_accounting(stage: str, payload: Mapping[str, Any]) -> dict[str, int] | None:
    if stage != "full_reality":
        return None
    row_accounting = payload.get("row_accounting")
    if not isinstance(row_accounting, Mapping):
        return None
    return {
        "row_count": int(row_accounting.get("row_count", 0)),
        "pass_row_count": int(row_accounting.get("pass_row_count", 0)),
        "fail_row_count": int(row_accounting.get("fail_row_count", 0)),
        "blocked_row_count": int(row_accounting.get("blocked_row_count", 0)),
    }


def _stage_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        stage: _summarize_stage(stage, records)
        for stage in ("identity", "prediction", "full_reality")
    }


def _summarize_stage(
    stage: str,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stage_records = [record for record in records if record["stage"] == stage]
    pass_rows = 0
    fail_rows = 0
    blocked_rows = 0
    for record in stage_records:
        if record.get("row_status") == "pass":
            pass_rows += 1
        if record.get("row_status") == "fail":
            fail_rows += 1
        row_accounting = record.get("row_accounting")
        if isinstance(row_accounting, Mapping):
            pass_rows += int(row_accounting.get("pass_row_count", 0))
            fail_rows += int(row_accounting.get("fail_row_count", 0))
            blocked_rows += int(row_accounting.get("blocked_row_count", 0))
    return {
        "package_count": len(stage_records),
        "reviewable_count": sum(1 for record in stage_records if record["reviewable"]),
        "valid_count": sum(1 for record in stage_records if record["validator_ok"]),
        "invalid_count": sum(1 for record in stage_records if not record["validator_ok"]),
        "pass_row_count": pass_rows,
        "fail_row_count": fail_rows,
        "blocked_row_count": blocked_rows,
        "sample_ids": sorted(
            {
                record["sample_id"]
                for record in stage_records
                if isinstance(record.get("sample_id"), str) and record["sample_id"]
            }
        ),
    }


def _sample_scope(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sample_ids = sorted(
        {
            record["sample_id"]
            for record in records
            if isinstance(record.get("sample_id"), str) and record["sample_id"]
        }
    )
    return {
        "sample_ids": sample_ids,
        "sample_count": len(sample_ids),
        "contains_multiple_samples": len(sample_ids) > 1,
    }


def _maturity(stage_summary: Mapping[str, Mapping[str, Any]]) -> str:
    if (
        int(stage_summary["identity"]["package_count"]) == 0
        and int(stage_summary["prediction"]["package_count"]) == 0
        and int(stage_summary["full_reality"]["package_count"]) == 0
    ):
        return "no_evidence_summaries"
    if int(stage_summary["full_reality"]["reviewable_count"]) > 0:
        return "full_reality_reviewable"
    if (
        int(stage_summary["identity"]["reviewable_count"]) > 0
        and int(stage_summary["prediction"]["reviewable_count"]) > 0
    ):
        return "identity_prediction_reviewable"
    if int(stage_summary["identity"]["reviewable_count"]) > 0:
        return "identity_reviewable"
    return "evidence_summaries_present_not_reviewable"


def _issues(
    records: Sequence[Mapping[str, Any]],
    ledger_gates: Mapping[str, bool],
    *,
    discovery: Mapping[str, Any],
) -> list[str]:
    issues = []
    for issue in discovery.get("issues", []):
        issues.append(f"discovery: {issue}")
    for record in records:
        for issue in record["issues"]:
            issues.append(f"{record['stage']}:{record['path']}: {issue}")
    for gate, passed in ledger_gates.items():
        if not passed:
            issues.append(f"ledger gate failed: {gate}")
    return issues


def _validate_discovery(discovery: Any) -> None:
    if not isinstance(discovery, Mapping):
        raise ValueError("phase1 evidence ledger requires discovery")
    roots = discovery.get("roots")
    if not isinstance(roots, list) or any(
        not isinstance(root, str) for root in roots
    ):
        raise ValueError("phase1 evidence ledger discovery roots must be strings")
    max_depth = discovery.get("max_depth")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("phase1 evidence ledger discovery max_depth is invalid")
    discovered_paths = discovery.get("discovered_paths")
    if not isinstance(discovered_paths, Mapping):
        raise ValueError("phase1 evidence ledger requires discovered_paths")
    for stage in ("identity", "prediction", "full_reality"):
        paths = discovered_paths.get(stage)
        if not isinstance(paths, list) or any(
            not isinstance(path, str) or not path for path in paths
        ):
            raise ValueError(
                f"phase1 evidence ledger discovery paths invalid for {stage}"
            )
    count_fields = {
        "identity_summary_count": "identity",
        "prediction_summary_count": "prediction",
        "reality_summary_count": "full_reality",
    }
    for count_field, stage in count_fields.items():
        value = discovery.get(count_field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"phase1 evidence ledger discovery {count_field} is invalid"
            )
        if value != len(discovered_paths[stage]):
            raise ValueError(
                f"phase1 evidence ledger discovery {count_field} mismatch"
            )
    issues = discovery.get("issues")
    if not isinstance(issues, list) or any(
        not isinstance(issue, str) for issue in issues
    ):
        raise ValueError("phase1 evidence ledger discovery issues must be strings")


def _validate_record(record: Any) -> None:
    if not isinstance(record, Mapping):
        raise TypeError("phase1 evidence ledger records must be mappings")
    for key in ("stage", "path"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"phase1 evidence ledger record requires {key}")
    if record["stage"] not in {"identity", "prediction", "full_reality"}:
        raise ValueError("phase1 evidence ledger record stage is unsupported")
    for key in ("exists", "is_file", "schema_ok", "validator_ok", "reviewable"):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"phase1 evidence ledger record requires bool {key}")
    if isinstance(record.get("size_bytes"), bool) or not isinstance(
        record.get("size_bytes"), int
    ):
        raise ValueError("phase1 evidence ledger record size must be int")
    if not isinstance(record.get("issues"), list):
        raise ValueError("phase1 evidence ledger record issues must be list")


def _validate_stage(stage: Any, stage_name: str) -> None:
    if not isinstance(stage, Mapping):
        raise ValueError(f"phase1 evidence ledger requires {stage_name} summary")
    for key in (
        "package_count",
        "reviewable_count",
        "valid_count",
        "invalid_count",
        "pass_row_count",
        "fail_row_count",
        "blocked_row_count",
    ):
        value = stage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"phase1 evidence ledger stage {stage_name} invalid {key}")
    if not isinstance(stage.get("sample_ids"), list):
        raise ValueError(f"phase1 evidence ledger stage {stage_name} sample_ids invalid")
