from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from objgauss.core.objectstate_bop_baseline_local_row_handoff import (
    OBJECTSTATE_BOP_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA,
    validate_objectstate_bop_baseline_local_row_handoff_summary,
)
from objgauss.core.objectstate_bop_local_row_handoff import (
    OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA,
    validate_objectstate_bop_local_row_handoff_summary,
)
from objgauss.core.objectstate_bop_rgbd_baseline_local_row_handoff import (
    OBJECTSTATE_BOP_RGBD_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA,
    validate_objectstate_bop_rgbd_baseline_local_row_handoff_summary,
)
from objgauss.core.objectstate_controlled_real_rows import (
    validate_objectstate_controlled_real_manifest,
)
from objgauss.core.objectstate_reality_gate import (
    OBJECTSTATE_REALITY_SOURCE_KINDS,
    ObjectStateRealityGateThresholds,
    ObjectStateRealityRow,
    evaluate_objectstate_reality_gate,
    objectstate_reality_blocked_rows_markdown,
    validate_objectstate_reality_gate_summary,
    validate_objectstate_reality_row,
)

OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA = "objgauss-objectstate-bop-reality-rows-v1"
_ALLOWED_BOP_REALITY_SOURCE_KINDS = ("public_replay", "controlled_real")
_EVIDENCE_KINDS = ("identity", "prediction", "intervention")


def read_objectstate_bop_local_row_summary(path: str | Path) -> dict[str, Any]:
    summary_path = Path(path)
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("BOP local-row summary JSON must be an object")
    return _validate_supported_source_summary(payload)


def objectstate_bop_reality_rows_from_summary(
    summary: Mapping[str, Any],
    *,
    source_kind: str = "public_replay",
    source_summary_ref: str | None = None,
) -> tuple[ObjectStateRealityRow, ...]:
    checked_source = _validate_supported_source_summary(summary)
    checked_source_kind = _validate_source_kind(source_kind)
    local_row = _local_row_from_supported_summary(checked_source)
    identity_manifest = _identity_controlled_real_manifest(local_row)
    prediction_manifest = _prediction_controlled_real_manifest(local_row)

    rows: list[ObjectStateRealityRow] = []
    for evidence_kind in _EVIDENCE_KINDS:
        evidence, manifest, source_label = _select_evidence_row(
            evidence_kind,
            identity_manifest=identity_manifest,
            prediction_manifest=prediction_manifest,
        )
        rows.append(
            _reality_row_from_evidence(
                evidence,
                manifest=manifest,
                evidence_kind=evidence_kind,
                source_label=source_label,
                source_kind=checked_source_kind,
                source_summary_ref=source_summary_ref,
            )
        )
    return tuple(rows)


def objectstate_bop_reality_rows_summary(
    summary: Mapping[str, Any],
    *,
    source_kind: str = "public_replay",
    source_summary_ref: str | None = None,
    synthetic_smoke_passed: bool = True,
    thresholds: ObjectStateRealityGateThresholds | None = None,
) -> dict[str, Any]:
    checked_source = _validate_supported_source_summary(summary)
    checked_source_kind = _validate_source_kind(source_kind)
    local_row = _local_row_from_supported_summary(checked_source)
    rows = objectstate_bop_reality_rows_from_summary(
        checked_source,
        source_kind=checked_source_kind,
        source_summary_ref=source_summary_ref,
    )
    gate = evaluate_objectstate_reality_gate(
        rows,
        synthetic_smoke_passed=bool(synthetic_smoke_passed),
        thresholds=thresholds,
    )
    issues = _issues_from_source(checked_source, local_row, gate.as_dict())
    payload = {
        "schema": OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA,
        "kind": "objectstate_bop_reality_rows",
        "status": "objectstate_bop_reality_rows_reviewable",
        "source_schema": checked_source["schema"],
        "source_kind": checked_source_kind,
        "source_summary_ref": source_summary_ref,
        "sample_id": local_row["sample_id"],
        "row_schema": rows[0].schema,
        "row_count": len(rows),
        "pass_row_count": len(gate.pass_rows),
        "fail_row_count": len(gate.fail_rows),
        "blocked_row_count": len(gate.blocked_rows),
        "rows": [row.as_dict() for row in rows],
        "gate": gate.as_dict(),
        "blocked_rows_markdown": objectstate_reality_blocked_rows_markdown(gate),
        "source_reviewability": _source_reviewability(checked_source, local_row),
        "source_pass_gates": _source_pass_gates(checked_source, local_row),
        "row_sources": _row_sources(rows),
        "issues": issues,
        "claim_policy": {
            "imports_existing_bop_local_row_summary": True,
            "uses_existing_controlled_real_manifest_rows": True,
            "converts_bop_public_dataset_rows_to_public_replay": (
                checked_source_kind == "public_replay"
            ),
            "does_not_create_ground_truth": True,
            "does_not_run_bop_handoff": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_train_model": True,
            "does_not_create_public_samples": True,
            "does_not_claim_intervention_gate": True,
            "does_not_claim_world_model": True,
            "blocked_rows_are_not_pass_rows": True,
            "failed_rows_remain_failed": True,
            "prediction_baseline_is_not_learned_dynamics": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "captures_video": False,
            "creates_ground_truth": False,
            "runs_bop_handoff": False,
            "reconstructs_gaussians": False,
            "runs_tracking_model": False,
            "runs_learned_prediction_model": False,
            "runs_intervention_model": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_bop_reality_rows_summary(payload)


def validate_objectstate_bop_reality_rows_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP reality rows summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA:
        raise ValueError(
            f"unsupported BOP reality rows schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_reality_rows":
        raise ValueError("BOP reality rows kind is unsupported")
    if payload.get("status") != "objectstate_bop_reality_rows_reviewable":
        raise ValueError("BOP reality rows status is unsupported")
    if payload.get("source_schema") not in {
        OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA,
        OBJECTSTATE_BOP_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA,
        OBJECTSTATE_BOP_RGBD_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA,
    }:
        raise ValueError("BOP reality rows source_schema is unsupported")
    if payload.get("source_kind") not in _ALLOWED_BOP_REALITY_SOURCE_KINDS:
        raise ValueError("BOP reality rows source_kind is unsupported")
    if not isinstance(payload.get("sample_id"), str) or not payload["sample_id"]:
        raise ValueError("BOP reality rows require sample_id")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("BOP reality rows summary requires exactly three rows")
    if payload.get("row_count") != len(rows):
        raise ValueError("BOP reality row_count must match rows")
    if {row.get("evidence_kind") for row in rows} != set(_EVIDENCE_KINDS):
        raise ValueError("BOP reality rows must include identity, prediction, intervention")
    gate = payload.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError("BOP reality rows summary requires gate")
    validate_objectstate_reality_gate_summary(dict(gate))
    if payload.get("pass_row_count") != gate.get("pass_row_count"):
        raise ValueError("BOP reality pass_row_count must match gate")
    if payload.get("fail_row_count") != gate.get("fail_row_count"):
        raise ValueError("BOP reality fail_row_count must match gate")
    if payload.get("blocked_row_count") != gate.get("blocked_row_count"):
        raise ValueError("BOP reality blocked_row_count must match gate")
    if payload.get("row_count") != gate.get("row_count"):
        raise ValueError("BOP reality row_count must match gate")
    if not isinstance(payload.get("blocked_rows_markdown"), str):
        raise ValueError("BOP reality rows require blocked_rows_markdown")
    if not isinstance(payload.get("source_reviewability"), Mapping):
        raise ValueError("BOP reality rows require source_reviewability")
    if not isinstance(payload.get("source_pass_gates"), Mapping):
        raise ValueError("BOP reality rows require source_pass_gates")
    if not isinstance(payload.get("row_sources"), list):
        raise ValueError("BOP reality rows require row_sources")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("BOP reality rows require issues")
    claim_policy = payload.get("claim_policy", {})
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("imports_existing_bop_local_row_summary")
        or not claim_policy.get("uses_existing_controlled_real_manifest_rows")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_run_bop_handoff")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_create_public_samples")
        or not claim_policy.get("does_not_claim_intervention_gate")
        or not claim_policy.get("does_not_claim_world_model")
        or not claim_policy.get("blocked_rows_are_not_pass_rows")
        or not claim_policy.get("failed_rows_remain_failed")
        or not claim_policy.get("prediction_baseline_is_not_learned_dynamics")
    ):
        raise ValueError("BOP reality rows summary must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP reality rows cannot download, capture, create GT, run handoff, "
            "reconstruct, track, learn prediction, intervene, train, write public "
            "samples, replay, diffuse, or mutate viewer policy"
        )
    return dict(payload)


def _validate_supported_source_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        raise TypeError("BOP local-row source summary must be a mapping")
    schema = summary.get("schema")
    if schema == OBJECTSTATE_BOP_RGBD_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA:
        return validate_objectstate_bop_rgbd_baseline_local_row_handoff_summary(summary)
    if schema == OBJECTSTATE_BOP_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA:
        return validate_objectstate_bop_baseline_local_row_handoff_summary(summary)
    if schema == OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA:
        return validate_objectstate_bop_local_row_handoff_summary(summary)
    raise ValueError(f"unsupported BOP local-row source schema: {schema}")


def _local_row_from_supported_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    schema = summary["schema"]
    if schema == OBJECTSTATE_BOP_RGBD_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA:
        baseline = summary.get("baseline_local_row_handoff")
        if not isinstance(baseline, Mapping):
            raise ValueError("BOP RGB-D baseline summary has no baseline local row")
        return validate_objectstate_bop_local_row_handoff_summary(
            baseline.get("local_row_handoff")
        )
    if schema == OBJECTSTATE_BOP_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA:
        return validate_objectstate_bop_local_row_handoff_summary(
            summary.get("local_row_handoff")
        )
    return validate_objectstate_bop_local_row_handoff_summary(summary)


def _identity_controlled_real_manifest(local_row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        manifest = local_row["identity_handoff"]["identity_handoff"][
            "controlled_real_manifest"
        ]
    except KeyError as exc:
        raise ValueError("BOP local row summary is missing identity manifest") from exc
    return validate_objectstate_controlled_real_manifest(manifest)


def _prediction_controlled_real_manifest(local_row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        manifest = local_row["prediction_handoff"]["prediction_eval_summary"][
            "controlled_real_manifest"
        ]
    except KeyError as exc:
        raise ValueError("BOP local row summary is missing prediction manifest") from exc
    return validate_objectstate_controlled_real_manifest(manifest)


def _select_evidence_row(
    evidence_kind: str,
    *,
    identity_manifest: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    preferred = (
        (identity_manifest, "identity-handoff"),
        (prediction_manifest, "prediction-handoff"),
    )
    if evidence_kind == "prediction":
        preferred = tuple(reversed(preferred))
    if evidence_kind == "intervention":
        preferred = (
            (prediction_manifest, "prediction-handoff"),
            (identity_manifest, "identity-handoff"),
        )
    blocked_fallback: tuple[dict[str, Any], dict[str, Any], str] | None = None
    for manifest, source_label in preferred:
        evidence = _evidence_row(manifest, evidence_kind)
        if evidence is None:
            continue
        candidate = (evidence, dict(manifest), source_label)
        if evidence["status"] != "blocked":
            return candidate
        if blocked_fallback is None:
            blocked_fallback = candidate
    if blocked_fallback is not None:
        return blocked_fallback
    raise ValueError(f"BOP local row summary has no {evidence_kind} evidence row")


def _evidence_row(
    manifest: Mapping[str, Any],
    evidence_kind: str,
) -> dict[str, Any] | None:
    for row in manifest["evidence_rows"]:
        if row["evidence_kind"] == evidence_kind:
            return dict(row)
    return None


def _reality_row_from_evidence(
    evidence: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    evidence_kind: str,
    source_label: str,
    source_kind: str,
    source_summary_ref: str | None,
) -> ObjectStateRealityRow:
    sample = manifest["sample"]
    ground_truth = manifest["ground_truth"]
    row = ObjectStateRealityRow(
        row_id=f"{sample['sample_id']}:bop-reality:{evidence_kind}:{source_label}",
        sample_id=sample["sample_id"],
        source_kind=source_kind,
        evidence_kind=evidence_kind,
        status=evidence["status"],
        object_category=sample["object_category"],
        scenario=sample["scenario"],
        observation_modalities=tuple(sample["observation_modalities"]),
        artifact_refs=_artifact_refs(evidence, sample, source_summary_ref),
        metrics=_metrics(evidence.get("metrics", {})),
        has_identity_gt=bool(ground_truth["identity"]),
        has_pose_gt=bool(ground_truth["pose"]),
        has_action_gt=bool(ground_truth["action"]),
        has_timestamp=bool(ground_truth["timestamp"]),
        license=sample["license"],
        block_reason=evidence.get("block_reason"),
        failure_reason=evidence.get("failure_reason"),
    )
    return validate_objectstate_reality_row(row)


def _artifact_refs(
    evidence: Mapping[str, Any],
    sample: Mapping[str, Any],
    source_summary_ref: str | None,
) -> tuple[str, ...]:
    refs = list(evidence.get("artifact_refs") or sample["artifact_refs"])
    if source_summary_ref and source_summary_ref not in refs:
        refs.append(str(source_summary_ref))
    return tuple(str(ref) for ref in refs if str(ref))


def _metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        raise TypeError("BOP evidence metrics must be a mapping")
    normalized = {}
    for key, value in metrics.items():
        if value is None or isinstance(value, str):
            continue
        normalized[str(key)] = value
    return normalized


def _source_reviewability(
    summary: Mapping[str, Any],
    local_row: Mapping[str, Any],
) -> dict[str, Any]:
    reviewability: dict[str, Any] = {
        "source": dict(summary.get("reviewability_gates", {})),
        "local_row": dict(local_row.get("reviewability_gates", {})),
    }
    identity_handoff = local_row["identity_handoff"]
    prediction_handoff = local_row["prediction_handoff"]
    reviewability["identity_handoff"] = dict(
        identity_handoff.get("reviewability_gates", {})
    )
    reviewability["prediction_handoff"] = dict(
        prediction_handoff.get("readiness", {})
    )
    return reviewability


def _source_pass_gates(
    summary: Mapping[str, Any],
    local_row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source": dict(summary.get("pass_gates", {})),
        "local_row": dict(local_row.get("pass_gates", {})),
        "identity_handoff": dict(
            local_row["identity_handoff"].get("pass_gates", {})
        ),
    }


def _row_sources(rows: tuple[ObjectStateRealityRow, ...]) -> list[dict[str, str]]:
    sources = []
    for row in rows:
        source_label = row.row_id.rsplit(":", maxsplit=1)[-1]
        sources.append(
            {
                "row_id": row.row_id,
                "evidence_kind": row.evidence_kind,
                "status": row.status,
                "source": source_label,
            }
        )
    return sources


def _issues_from_source(
    summary: Mapping[str, Any],
    local_row: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []
    for source in (
        summary.get("issues", ()),
        local_row.get("issues", ()),
        local_row["identity_handoff"].get("issues", ()),
        local_row["prediction_handoff"].get("issues", ()),
    ):
        if isinstance(source, list):
            issues.extend(str(issue) for issue in source)
    if gate.get("status") != "objectstate_reality_gate_pass":
        blockers = ", ".join(str(item) for item in gate.get("hard_blockers", ()))
        issues.append(f"full ObjectState reality gate did not pass: {blockers}")
    return _dedupe(issues)


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def _validate_source_kind(source_kind: str) -> str:
    value = str(source_kind)
    if value not in _ALLOWED_BOP_REALITY_SOURCE_KINDS:
        raise ValueError("BOP reality rows source_kind must be public_replay or controlled_real")
    if value not in OBJECTSTATE_REALITY_SOURCE_KINDS:
        raise ValueError(f"unsupported ObjectState reality source_kind: {value}")
    return value
