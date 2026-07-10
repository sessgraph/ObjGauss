from __future__ import annotations

from typing import Any, Mapping

from objgauss.datasets.objectstate_controlled_real_manifest import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
    validate_objectstate_controlled_real_manifest,
)
from objgauss.evaluation.objectstate_reality_gate import (
    ObjectStateRealityGateReport,
    ObjectStateRealityGateThresholds,
    ObjectStateRealityRow,
    evaluate_objectstate_reality_gate,
    objectstate_reality_blocked_rows_markdown,
    validate_objectstate_reality_gate_summary,
    validate_objectstate_reality_row,
)

OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA = "objgauss-objectstate-controlled-real-rows-v1"


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
    rows = report.rows
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
    if rows != gate.get("rows"):
        raise ValueError("controlled real rows must match gate-derived rows")
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
