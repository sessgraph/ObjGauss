from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_reality_gate import (
    ObjectStateRealityGateReport,
    ObjectStateRealityGateThresholds,
    ObjectStateRealityRow,
    evaluate_objectstate_reality_gate,
    objectstate_reality_blocked_rows_markdown,
    validate_objectstate_reality_gate_summary,
    validate_objectstate_reality_row,
)

OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA = (
    "objgauss-objectstate-controlled-real-manifest-v1"
)
OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA = "objgauss-objectstate-controlled-real-rows-v1"


def read_objectstate_controlled_real_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("controlled real manifest JSON must be an object")
    return validate_objectstate_controlled_real_manifest(payload)


def objectstate_reality_rows_from_controlled_real_manifest(
    manifest: Mapping[str, Any],
) -> tuple[ObjectStateRealityRow, ...]:
    checked_manifest = validate_objectstate_controlled_real_manifest(manifest)
    sample = checked_manifest["sample"]
    ground_truth = checked_manifest["ground_truth"]
    base_refs = tuple(sample["artifact_refs"])
    rows = []
    for index, evidence in enumerate(checked_manifest["evidence_rows"]):
        evidence_refs = tuple(evidence.get("artifact_refs", ()))
        artifact_refs = evidence_refs or base_refs
        rows.append(
            validate_objectstate_reality_row(
                ObjectStateRealityRow(
                    row_id=str(
                        evidence.get(
                            "row_id",
                            f"{sample['sample_id']}:{evidence['evidence_kind']}:{index:03d}",
                        )
                    ),
                    sample_id=sample["sample_id"],
                    source_kind="controlled_real",
                    evidence_kind=evidence["evidence_kind"],
                    status=evidence["status"],
                    object_category=sample["object_category"],
                    scenario=sample["scenario"],
                    observation_modalities=tuple(sample["observation_modalities"]),
                    artifact_refs=artifact_refs,
                    metrics=evidence.get("metrics", {}),
                    has_identity_gt=bool(ground_truth["identity"]),
                    has_pose_gt=bool(ground_truth["pose"]),
                    has_action_gt=bool(ground_truth["action"]),
                    has_timestamp=bool(ground_truth["timestamp"]),
                    license=sample["license"],
                    block_reason=evidence.get("block_reason"),
                    failure_reason=evidence.get("failure_reason"),
                )
            )
        )
    return tuple(rows)


def evaluate_controlled_real_manifest_reality_gate(
    manifest: Mapping[str, Any],
    *,
    synthetic_smoke_passed: bool = True,
    thresholds: ObjectStateRealityGateThresholds | None = None,
) -> ObjectStateRealityGateReport:
    return evaluate_objectstate_reality_gate(
        objectstate_reality_rows_from_controlled_real_manifest(manifest),
        synthetic_smoke_passed=bool(synthetic_smoke_passed),
        thresholds=thresholds,
    )


def objectstate_controlled_real_rows_summary(
    manifest: Mapping[str, Any],
    *,
    synthetic_smoke_passed: bool = True,
    thresholds: ObjectStateRealityGateThresholds | None = None,
) -> dict[str, Any]:
    checked_manifest = validate_objectstate_controlled_real_manifest(manifest)
    rows = objectstate_reality_rows_from_controlled_real_manifest(checked_manifest)
    report = evaluate_objectstate_reality_gate(
        rows,
        synthetic_smoke_passed=bool(synthetic_smoke_passed),
        thresholds=thresholds,
    )
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
        "kind": "objectstate_controlled_real_rows",
        "manifest_schema": OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
        "sample": dict(checked_manifest["sample"]),
        "ground_truth": dict(checked_manifest["ground_truth"]),
        "row_count": len(rows),
        "pass_row_count": len(report.pass_rows),
        "fail_row_count": len(report.fail_rows),
        "blocked_row_count": len(report.blocked_rows),
        "rows": [row.as_dict() for row in rows],
        "gate": report.as_dict(),
        "blocked_rows_markdown": objectstate_reality_blocked_rows_markdown(report),
        "claim_policy": {
            "controlled_real_manifest_required": True,
            "importer_does_not_create_ground_truth": True,
            "non_blocked_rows_require_manifest_ground_truth": True,
            "does_not_claim_open_world_generalization": True,
        },
        "non_goals": {
            "captures_video": False,
            "writes_public_samples": False,
            "submits_generated_outputs": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_controlled_real_rows_summary(payload)


def validate_objectstate_controlled_real_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise TypeError("controlled real manifest must be a mapping")
    if manifest.get("schema") != OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported controlled real manifest schema: {manifest.get('schema')}"
        )
    sample = _validate_sample(manifest.get("sample"))
    ground_truth = _validate_ground_truth(manifest.get("ground_truth"))
    evidence_rows = manifest.get("evidence_rows")
    if not isinstance(evidence_rows, Sequence) or isinstance(evidence_rows, (str, bytes)):
        raise TypeError("controlled real manifest evidence_rows must be a sequence")
    checked_rows = tuple(_validate_evidence_row(row) for row in evidence_rows)
    if not checked_rows:
        raise ValueError("controlled real manifest requires at least one evidence row")
    return {
        "schema": OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
        "sample": sample,
        "ground_truth": ground_truth,
        "evidence_rows": checked_rows,
    }


def validate_objectstate_controlled_real_rows_summary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("controlled real rows summary must be a dict")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA:
        raise ValueError(f"unsupported controlled real rows schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_controlled_real_rows":
        raise ValueError("controlled real rows kind is unsupported")
    if payload.get("manifest_schema") != OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA:
        raise ValueError("controlled real rows summary has unsupported manifest_schema")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("controlled real rows summary requires rows")
    if payload.get("row_count") != len(rows):
        raise ValueError("controlled real row_count must match rows")
    gate = payload.get("gate")
    if not isinstance(gate, dict):
        raise ValueError("controlled real rows summary requires gate")
    validate_objectstate_reality_gate_summary(gate)
    if payload.get("pass_row_count") != gate.get("pass_row_count"):
        raise ValueError("pass_row_count must match gate pass_row_count")
    if payload.get("fail_row_count") != gate.get("fail_row_count"):
        raise ValueError("fail_row_count must match gate fail_row_count")
    if payload.get("blocked_row_count") != gate.get("blocked_row_count"):
        raise ValueError("blocked_row_count must match gate blocked_row_count")
    if not isinstance(payload.get("blocked_rows_markdown"), str):
        raise ValueError("blocked_rows_markdown must be a string")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("controlled_real_manifest_required")
        or not claim_policy.get("importer_does_not_create_ground_truth")
        or not claim_policy.get("non_blocked_rows_require_manifest_ground_truth")
    ):
        raise ValueError("controlled real rows summary must preserve GT claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("writes_public_samples")
        or non_goals.get("submits_generated_outputs")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError("controlled real importer cannot capture, write outputs, train, replay, diffuse, or mutate viewer policy")
    return payload


def _validate_sample(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled real sample must be a mapping")
    sample_id = _required_string(value, "sample_id")
    object_category = _required_string(value, "object_category")
    scenario = _required_string(value, "scenario")
    license_text = _required_string(value, "license")
    source_kind = str(value.get("source_kind", "controlled_real"))
    if source_kind != "controlled_real":
        raise ValueError("controlled real manifest sample.source_kind must be controlled_real")
    return {
        "sample_id": sample_id,
        "source_kind": source_kind,
        "object_category": object_category,
        "scenario": scenario,
        "observation_modalities": _string_tuple(
            value.get("observation_modalities"),
            "observation_modalities",
        ),
        "artifact_refs": _string_tuple(value.get("artifact_refs"), "artifact_refs"),
        "license": license_text,
    }


def _validate_ground_truth(value: Any) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled real ground_truth must be a mapping")
    return {
        "identity": bool(value.get("identity", False)),
        "pose": bool(value.get("pose", False)),
        "action": bool(value.get("action", False)),
        "timestamp": bool(value.get("timestamp", False)),
    }


def _validate_evidence_row(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled real evidence rows must be mappings")
    evidence_kind = _required_string(value, "evidence_kind")
    status = _required_string(value, "status")
    if evidence_kind not in {"identity", "prediction", "intervention"}:
        raise ValueError(f"unsupported controlled real evidence_kind: {evidence_kind}")
    if status not in {"pass", "fail", "blocked"}:
        raise ValueError(f"unsupported controlled real row status: {status}")
    metrics = value.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise TypeError("controlled real evidence row metrics must be a mapping")
    block_reason = value.get("block_reason")
    failure_reason = value.get("failure_reason")
    if status == "blocked" and not block_reason:
        raise ValueError("blocked controlled real evidence rows require block_reason")
    if status == "fail" and not failure_reason:
        raise ValueError("failed controlled real evidence rows require failure_reason")
    if status == "pass" and (block_reason or failure_reason):
        raise ValueError("pass controlled real evidence rows cannot carry block or failure reasons")
    row = {
        "evidence_kind": evidence_kind,
        "status": status,
        "metrics": dict(metrics),
    }
    if "row_id" in value:
        row["row_id"] = _required_string(value, "row_id")
    if "artifact_refs" in value:
        row["artifact_refs"] = _string_tuple(value.get("artifact_refs"), "artifact_refs")
    if block_reason:
        row["block_reason"] = str(block_reason)
    if failure_reason:
        row["failure_reason"] = str(failure_reason)
    return row


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be a non-empty string")
    return result


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    normalized = tuple(str(item) for item in value)
    if not normalized or any(not item for item in normalized):
        raise ValueError(f"{name} must contain non-empty strings")
    return normalized
