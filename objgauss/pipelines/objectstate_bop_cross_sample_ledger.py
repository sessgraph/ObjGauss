from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.pipelines.objectstate_bop_local_row_handoff import (
    OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA,
    validate_objectstate_bop_local_row_handoff_summary,
)

OBJECTSTATE_BOP_CROSS_SAMPLE_LEDGER_SCHEMA = (
    "objgauss-objectstate-bop-cross-sample-ledger-v1"
)
BOP_LOCAL_ROW_HANDOFF_SUMMARY_FILENAMES = (
    "bop-local-row-handoff-summary.json",
)


def objectstate_bop_cross_sample_ledger(
    *,
    local_row_summaries: Sequence[str | Path] = (),
    discover_roots: Sequence[str | Path] = (),
    max_depth: int = 4,
    min_reviewable_samples: int = 3,
    min_scene_or_category_coverage: int = 3,
) -> dict[str, Any]:
    _require_non_negative_int(max_depth, "max_depth")
    _require_positive_int(min_reviewable_samples, "min_reviewable_samples")
    _require_positive_int(
        min_scene_or_category_coverage,
        "min_scene_or_category_coverage",
    )
    discovery = _discover_local_row_summaries(discover_roots, max_depth=max_depth)
    summary_paths = _dedupe_paths(
        (*local_row_summaries, *discovery["paths"]["local_row"])
    )
    records = [_local_row_record(path) for path in summary_paths]
    sample_summary = _sample_summary(records)
    audit_gates = {
        "discovery_roots_valid": not discovery["issues"],
        "local_row_summaries_present": bool(records),
        "all_files_present": all(record["is_file"] for record in records),
        "all_json_schemas_valid": all(record["schema_ok"] for record in records),
        "all_summary_validators_passed": all(
            record["validator_ok"] for record in records
        ),
    }
    candidate_gates = {
        "min_reviewable_samples_met": (
            sample_summary["reviewable_sample_count"] >= min_reviewable_samples
        ),
        "min_identity_prediction_reviewable_samples_met": (
            sample_summary["identity_prediction_reviewable_sample_count"]
            >= min_reviewable_samples
        ),
        "scene_or_category_coverage_met": (
            sample_summary["sample_count"] >= min_scene_or_category_coverage
            or sample_summary["scene_root_count"] >= min_scene_or_category_coverage
            or sample_summary["object_category_count"]
            >= min_scene_or_category_coverage
            or sample_summary["scenario_count"] >= min_scene_or_category_coverage
        ),
        "blocked_rows_separated_from_pass_rows": True,
        "intervention_not_claimed": True,
        "does_not_claim_world_model": True,
    }
    status = (
        "objectstate_bop_cross_sample_ledger_reviewable"
        if all(audit_gates.values())
        else "objectstate_bop_cross_sample_ledger_incomplete"
    )
    payload = {
        "schema": OBJECTSTATE_BOP_CROSS_SAMPLE_LEDGER_SCHEMA,
        "kind": "objectstate_bop_cross_sample_ledger",
        "status": status,
        "maturity": _maturity(sample_summary, candidate_gates),
        "local_row_handoff_schema": OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA,
        "thresholds": {
            "min_reviewable_samples": min_reviewable_samples,
            "min_scene_or_category_coverage": min_scene_or_category_coverage,
        },
        "discovery": _discovery_summary(discovery),
        "sample_summary": sample_summary,
        "records": records,
        "audit_gates": audit_gates,
        "candidate_gate": {
            "candidate_cross_sample_ready": all(candidate_gates.values()),
            "gates": candidate_gates,
            "interpretation": (
                "reviewable cross-sample BOP identity/prediction evidence"
                if all(candidate_gates.values())
                else "cross-sample BOP evidence remains incomplete"
            ),
        },
        "issues": _issues(
            records,
            audit_gates,
            candidate_gates,
            discovery=discovery,
        ),
        "sample_table_markdown": _sample_table_markdown(records),
        "claim_policy": {
            "read_only_audit": True,
            "checks_existing_bop_local_row_summaries": True,
            "discovers_existing_bop_local_row_summaries": True,
            "requires_identity_and_prediction_reviewability_per_sample": True,
            "reviewable_allows_metric_pass_or_fail": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_run_identity_handoff": True,
            "does_not_run_prediction_eval": True,
            "does_not_claim_intervention_gate": True,
            "does_not_claim_metric_pass": True,
            "does_not_claim_world_model": True,
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
    return validate_objectstate_bop_cross_sample_ledger_summary(payload)


def validate_objectstate_bop_cross_sample_ledger_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP cross-sample ledger summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_CROSS_SAMPLE_LEDGER_SCHEMA:
        raise ValueError(
            "unsupported BOP cross-sample ledger schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_cross_sample_ledger":
        raise ValueError("BOP cross-sample ledger kind is unsupported")
    if payload.get("status") not in {
        "objectstate_bop_cross_sample_ledger_reviewable",
        "objectstate_bop_cross_sample_ledger_incomplete",
    }:
        raise ValueError("BOP cross-sample ledger status is unsupported")
    if payload.get("maturity") not in {
        "no_local_row_summaries",
        "local_row_summaries_present_not_reviewable",
        "local_rows_reviewable",
        "identity_prediction_reviewable",
        "candidate_cross_sample_reviewable",
    }:
        raise ValueError("BOP cross-sample ledger maturity is unsupported")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("BOP cross-sample ledger requires thresholds")
    _require_positive_int(
        thresholds.get("min_reviewable_samples"),
        "min_reviewable_samples",
    )
    _require_positive_int(
        thresholds.get("min_scene_or_category_coverage"),
        "min_scene_or_category_coverage",
    )
    _validate_discovery(payload.get("discovery"))
    sample_summary = payload.get("sample_summary")
    if not isinstance(sample_summary, Mapping):
        raise ValueError("BOP cross-sample ledger requires sample_summary")
    for key in (
        "summary_count",
        "valid_summary_count",
        "reviewable_sample_count",
        "identity_prediction_reviewable_sample_count",
        "identity_reviewable_sample_count",
        "prediction_reviewable_sample_count",
        "identity_pass_sample_count",
        "prediction_pass_sample_count",
        "sample_count",
        "scene_root_count",
        "object_category_count",
        "scenario_count",
    ):
        _require_non_negative_int(sample_summary.get(key), key)
    for key in ("sample_ids", "scene_roots", "object_categories", "scenarios"):
        values = sample_summary.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"BOP cross-sample ledger {key} must be string list")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("BOP cross-sample ledger requires records")
    for record in records:
        _validate_record(record)
    audit_gates = payload.get("audit_gates")
    if not isinstance(audit_gates, Mapping) or not audit_gates:
        raise ValueError("BOP cross-sample ledger requires audit_gates")
    if any(not isinstance(value, bool) for value in audit_gates.values()):
        raise ValueError("BOP cross-sample ledger audit gates must be bool")
    expected_status = (
        "objectstate_bop_cross_sample_ledger_reviewable"
        if all(audit_gates.values())
        else "objectstate_bop_cross_sample_ledger_incomplete"
    )
    if payload.get("status") != expected_status:
        raise ValueError("BOP cross-sample ledger status mismatch")
    candidate_gate = payload.get("candidate_gate")
    if not isinstance(candidate_gate, Mapping):
        raise ValueError("BOP cross-sample ledger requires candidate_gate")
    if not isinstance(candidate_gate.get("candidate_cross_sample_ready"), bool):
        raise ValueError("BOP cross-sample ledger candidate readiness must be bool")
    candidate_gates = candidate_gate.get("gates")
    if not isinstance(candidate_gates, Mapping) or not candidate_gates:
        raise ValueError("BOP cross-sample ledger requires candidate gates")
    if any(not isinstance(value, bool) for value in candidate_gates.values()):
        raise ValueError("BOP cross-sample ledger candidate gates must be bool")
    if candidate_gate["candidate_cross_sample_ready"] != all(candidate_gates.values()):
        raise ValueError("BOP cross-sample ledger candidate readiness mismatch")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("BOP cross-sample ledger issues must be list")
    if not isinstance(payload.get("sample_table_markdown"), str):
        raise ValueError("BOP cross-sample ledger requires markdown table")
    claim_policy = payload.get("claim_policy")
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("read_only_audit")
        or not claim_policy.get("checks_existing_bop_local_row_summaries")
        or not claim_policy.get("discovers_existing_bop_local_row_summaries")
        or not claim_policy.get("requires_identity_and_prediction_reviewability_per_sample")
        or not claim_policy.get("reviewable_allows_metric_pass_or_fail")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_run_identity_handoff")
        or not claim_policy.get("does_not_run_prediction_eval")
        or not claim_policy.get("does_not_claim_intervention_gate")
        or not claim_policy.get("does_not_claim_metric_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP cross-sample ledger must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP cross-sample ledger cannot claim downloads, capture, GT, "
            "Gaussian reconstruction, models, training, public samples, replay, "
            "diffusion, or viewer mutation"
        )
    return dict(payload)


def _discover_local_row_summaries(
    roots: Sequence[str | Path],
    *,
    max_depth: int,
) -> dict[str, Any]:
    paths: dict[str, list[Path]] = {"local_row": []}
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
            if path.name in BOP_LOCAL_ROW_HANDOFF_SUMMARY_FILENAMES:
                paths["local_row"].append(path)
    return {
        "roots": root_strings,
        "max_depth": max_depth,
        "paths": {"local_row": _dedupe_paths(paths["local_row"])},
        "issues": issues,
    }


def _local_row_record(path: str | Path) -> dict[str, Any]:
    summary_path = Path(path)
    exists = summary_path.exists()
    is_file = exists and summary_path.is_file()
    issues: list[str] = []
    schema = None
    status = None
    sample_id = None
    scene_root = None
    output_root = None
    object_category = None
    scenario = None
    dataset_id = None
    schema_ok = False
    validator_ok = False
    reviewable = False
    identity_reviewable = False
    prediction_reviewable = False
    identity_prediction_reviewable = False
    identity_pass = False
    prediction_pass = False
    identity_row_status = None
    prediction_row_status = None
    ledger_maturity = None
    row_counts = {"identity_predictions": 0, "prediction_candidates": 0}
    if not is_file:
        issues.append("BOP local row handoff summary file is missing")
    else:
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - ledger reports malformed inputs.
            issues.append(f"invalid JSON: {exc}")
        else:
            if not isinstance(payload, Mapping):
                issues.append("BOP local row handoff summary JSON must be an object")
            else:
                schema = payload.get("schema")
                schema_ok = schema == OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA
                if not schema_ok:
                    issues.append(
                        "schema mismatch: expected "
                        f"{OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA}, got {schema}"
                    )
                try:
                    checked = validate_objectstate_bop_local_row_handoff_summary(payload)
                except Exception as exc:  # noqa: BLE001 - keep audit tolerant.
                    issues.append(f"validator failed: {exc}")
                else:
                    validator_ok = True
                    status = checked.get("status")
                    sample_id = checked.get("sample_id")
                    scene_root = checked.get("scene_root")
                    output_root = checked.get("output_root")
                    sample = _local_row_sample(checked)
                    object_category = sample.get("object_category")
                    scenario = sample.get("scenario")
                    dataset_id = sample.get("dataset_id")
                    reviewability = checked["reviewability_gates"]
                    identity_reviewable = bool(
                        reviewability["identity_handoff_reviewable"]
                    )
                    prediction_reviewable = bool(
                        reviewability["prediction_handoff_reviewable"]
                    )
                    identity_prediction_reviewable = bool(
                        identity_reviewable and prediction_reviewable
                    )
                    reviewable = bool(
                        status == "objectstate_bop_local_row_handoff_reviewable"
                    )
                    identity_pass = bool(checked["pass_gates"]["identity_handoff_pass"])
                    prediction_pass = bool(checked["pass_gates"]["prediction_eval_pass"])
                    identity_row_status = _identity_row_status(checked)
                    prediction_row_status = _prediction_row_status(checked)
                    ledger_maturity = checked["phase1_evidence_ledger_summary"][
                        "maturity"
                    ]
                    row_counts = {
                        "identity_predictions": int(
                            checked["row_counts"]["identity_predictions"]
                        ),
                        "prediction_candidates": int(
                            checked["row_counts"]["prediction_candidates"]
                        ),
                    }
    return {
        "path": str(summary_path),
        "exists": bool(exists),
        "is_file": bool(is_file),
        "size_bytes": int(summary_path.stat().st_size) if is_file else 0,
        "schema": schema,
        "expected_schema": OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA,
        "schema_ok": schema_ok,
        "validator_ok": validator_ok,
        "status": status,
        "reviewable": reviewable,
        "sample_id": sample_id,
        "scene_root": scene_root,
        "output_root": output_root,
        "dataset_id": dataset_id,
        "object_category": object_category,
        "scenario": scenario,
        "identity_reviewable": identity_reviewable,
        "prediction_reviewable": prediction_reviewable,
        "identity_prediction_reviewable": identity_prediction_reviewable,
        "identity_pass": identity_pass,
        "prediction_pass": prediction_pass,
        "identity_row_status": identity_row_status,
        "prediction_row_status": prediction_row_status,
        "phase1_evidence_ledger_maturity": ledger_maturity,
        "row_counts": row_counts,
        "issues": issues,
    }


def _local_row_sample(summary: Mapping[str, Any]) -> dict[str, str | None]:
    acceptance = summary.get("identity_handoff", {}).get("acceptance", {})
    manifest = acceptance.get("manifest") if isinstance(acceptance, Mapping) else None
    sample = manifest.get("sample") if isinstance(manifest, Mapping) else None
    source = acceptance.get("adapter_summary", {}).get("source", {})
    return {
        "dataset_id": source.get("dataset_id") if isinstance(source, Mapping) else None,
        "object_category": sample.get("object_category")
        if isinstance(sample, Mapping)
        else None,
        "scenario": sample.get("scenario") if isinstance(sample, Mapping) else None,
    }


def _identity_row_status(summary: Mapping[str, Any]) -> str | None:
    package = summary.get("identity_handoff", {}).get("identity_evidence_package", {})
    identity = package.get("identity") if isinstance(package, Mapping) else None
    if isinstance(identity, Mapping):
        value = identity.get("identity_row_status")
        return value if isinstance(value, str) else None
    return None


def _prediction_row_status(summary: Mapping[str, Any]) -> str | None:
    package = summary.get("prediction_handoff", {}).get("prediction_evidence_package", {})
    prediction = package.get("prediction") if isinstance(package, Mapping) else None
    if isinstance(prediction, Mapping):
        value = prediction.get("prediction_row_status")
        return value if isinstance(value, str) else None
    return None


def _sample_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid_records = [record for record in records if record["validator_ok"]]
    reviewable_sample_ids = _unique_values(
        record.get("sample_id")
        for record in valid_records
        if record["reviewable"]
    )
    identity_prediction_sample_ids = _unique_values(
        record.get("sample_id")
        for record in valid_records
        if record["identity_prediction_reviewable"]
    )
    identity_reviewable_sample_ids = _unique_values(
        record.get("sample_id")
        for record in valid_records
        if record["identity_reviewable"]
    )
    prediction_reviewable_sample_ids = _unique_values(
        record.get("sample_id")
        for record in valid_records
        if record["prediction_reviewable"]
    )
    identity_pass_sample_ids = _unique_values(
        record.get("sample_id")
        for record in valid_records
        if record["identity_pass"]
    )
    prediction_pass_sample_ids = _unique_values(
        record.get("sample_id")
        for record in valid_records
        if record["prediction_pass"]
    )
    sample_ids = _unique_values(record.get("sample_id") for record in valid_records)
    scene_roots = _unique_values(record.get("scene_root") for record in valid_records)
    object_categories = _unique_values(
        record.get("object_category") for record in valid_records
    )
    scenarios = _unique_values(record.get("scenario") for record in valid_records)
    return {
        "summary_count": len(records),
        "valid_summary_count": len(valid_records),
        "reviewable_sample_count": len(reviewable_sample_ids),
        "identity_prediction_reviewable_sample_count": len(
            identity_prediction_sample_ids
        ),
        "identity_reviewable_sample_count": len(identity_reviewable_sample_ids),
        "prediction_reviewable_sample_count": len(prediction_reviewable_sample_ids),
        "identity_pass_sample_count": len(identity_pass_sample_ids),
        "prediction_pass_sample_count": len(prediction_pass_sample_ids),
        "sample_count": len(sample_ids),
        "scene_root_count": len(scene_roots),
        "object_category_count": len(object_categories),
        "scenario_count": len(scenarios),
        "sample_ids": sample_ids,
        "scene_roots": scene_roots,
        "object_categories": object_categories,
        "scenarios": scenarios,
    }


def _maturity(
    sample_summary: Mapping[str, Any],
    candidate_gates: Mapping[str, bool],
) -> str:
    if int(sample_summary["summary_count"]) == 0:
        return "no_local_row_summaries"
    if all(candidate_gates.values()):
        return "candidate_cross_sample_reviewable"
    if int(sample_summary["identity_prediction_reviewable_sample_count"]) > 0:
        return "identity_prediction_reviewable"
    if int(sample_summary["reviewable_sample_count"]) > 0:
        return "local_rows_reviewable"
    return "local_row_summaries_present_not_reviewable"


def _discovery_summary(discovery: Mapping[str, Any]) -> dict[str, Any]:
    paths = [str(path) for path in discovery["paths"]["local_row"]]
    return {
        "roots": list(discovery["roots"]),
        "max_depth": int(discovery["max_depth"]),
        "local_row_summary_count": len(paths),
        "discovered_paths": {"local_row": paths},
        "issues": list(discovery["issues"]),
    }


def _issues(
    records: Sequence[Mapping[str, Any]],
    audit_gates: Mapping[str, bool],
    candidate_gates: Mapping[str, bool],
    *,
    discovery: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []
    for issue in discovery.get("issues", []):
        issues.append(f"discovery: {issue}")
    for record in records:
        for issue in record["issues"]:
            issues.append(f"{record['path']}: {issue}")
    for gate, passed in audit_gates.items():
        if not passed:
            issues.append(f"audit gate failed: {gate}")
    for gate, passed in candidate_gates.items():
        if not passed:
            issues.append(f"candidate gate incomplete: {gate}")
    return issues


def _sample_table_markdown(records: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| sample_id | identity_reviewable | prediction_reviewable | identity_row | prediction_row | identity_pass | prediction_pass | maturity |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for record in records:
        sample_id = _markdown_cell(record.get("sample_id") or "-")
        identity_row = _markdown_cell(record.get("identity_row_status") or "-")
        prediction_row = _markdown_cell(record.get("prediction_row_status") or "-")
        maturity = _markdown_cell(record.get("phase1_evidence_ledger_maturity") or "-")
        lines.append(
            "| "
            f"{sample_id} | "
            f"{_bool_text(record['identity_reviewable'])} | "
            f"{_bool_text(record['prediction_reviewable'])} | "
            f"{identity_row} | "
            f"{prediction_row} | "
            f"{_bool_text(record['identity_pass'])} | "
            f"{_bool_text(record['prediction_pass'])} | "
            f"{maturity} |"
        )
    return "\n".join(lines) + "\n"


def _validate_discovery(discovery: Any) -> None:
    if not isinstance(discovery, Mapping):
        raise ValueError("BOP cross-sample ledger requires discovery")
    roots = discovery.get("roots")
    if not isinstance(roots, list) or any(
        not isinstance(root, str) for root in roots
    ):
        raise ValueError("BOP cross-sample ledger discovery roots must be strings")
    max_depth = discovery.get("max_depth")
    _require_non_negative_int(max_depth, "max_depth")
    paths = discovery.get("discovered_paths")
    if not isinstance(paths, Mapping):
        raise ValueError("BOP cross-sample ledger requires discovered_paths")
    local_row_paths = paths.get("local_row")
    if not isinstance(local_row_paths, list) or any(
        not isinstance(path, str) or not path for path in local_row_paths
    ):
        raise ValueError("BOP cross-sample ledger discovered local row paths invalid")
    count = discovery.get("local_row_summary_count")
    _require_non_negative_int(count, "local_row_summary_count")
    if count != len(local_row_paths):
        raise ValueError("BOP cross-sample ledger local row discovery count mismatch")
    issues = discovery.get("issues")
    if not isinstance(issues, list) or any(
        not isinstance(issue, str) for issue in issues
    ):
        raise ValueError("BOP cross-sample ledger discovery issues must be strings")


def _validate_record(record: Any) -> None:
    if not isinstance(record, Mapping):
        raise TypeError("BOP cross-sample ledger records must be mappings")
    if not isinstance(record.get("path"), str) or not record["path"]:
        raise ValueError("BOP cross-sample ledger record requires path")
    for key in (
        "exists",
        "is_file",
        "schema_ok",
        "validator_ok",
        "reviewable",
        "identity_reviewable",
        "prediction_reviewable",
        "identity_prediction_reviewable",
        "identity_pass",
        "prediction_pass",
    ):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"BOP cross-sample ledger record requires bool {key}")
    _require_non_negative_int(record.get("size_bytes"), "size_bytes")
    row_counts = record.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("BOP cross-sample ledger record requires row_counts")
    for key in ("identity_predictions", "prediction_candidates"):
        _require_non_negative_int(row_counts.get(key), key)
    if not isinstance(record.get("issues"), list):
        raise ValueError("BOP cross-sample ledger record issues must be list")


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


def _unique_values(values: Any) -> list[str]:
    return sorted({value for value in values if isinstance(value, str) and value})


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _require_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"BOP cross-sample ledger requires positive int {name}")


def _require_non_negative_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"BOP cross-sample ledger requires non-negative int {name}")


__all__ = (
    "OBJECTSTATE_BOP_CROSS_SAMPLE_LEDGER_SCHEMA",
    "objectstate_bop_cross_sample_ledger",
    "validate_objectstate_bop_cross_sample_ledger_summary",
)
