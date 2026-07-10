from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.pipelines.objectstate_controlled_reality_bundle_handoff import (
    OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_HANDOFF_SCHEMA,
    validate_objectstate_controlled_reality_bundle_handoff_summary,
)
from objgauss.pipelines.objectstate_public_dataset_candidates import (
    default_objectstate_public_dataset_candidates,
    validate_objectstate_public_dataset_candidate,
)
from objgauss.evaluation.objectstate_reality_gate import (
    ObjectStateRealityGateThresholds,
    ObjectStateRealityRow,
    evaluate_objectstate_reality_gate,
    objectstate_reality_blocked_rows_markdown,
    validate_objectstate_reality_gate_summary,
    validate_objectstate_reality_row,
)

OBJECTSTATE_PUBLIC_INTERACTION_REALITY_ROWS_SCHEMA = (
    "objgauss-objectstate-public-interaction-reality-rows-v1"
)
_PUBLIC_INTERACTION_SOURCE_KIND = "public_replay"
_EVIDENCE_KINDS = ("identity", "prediction", "intervention")


def read_objectstate_public_interaction_handoff_summary(
    path: str | Path,
) -> dict[str, Any]:
    summary_path = Path(path)
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("public interaction source handoff summary must be an object")
    return validate_objectstate_controlled_reality_bundle_handoff_summary(payload)


def objectstate_public_interaction_reality_rows_from_handoff(
    handoff_summary: Mapping[str, Any],
    *,
    candidate_id: str = "hot3d-clips",
    source_summary_ref: str | None = None,
) -> tuple[ObjectStateRealityRow, ...]:
    checked = validate_objectstate_controlled_reality_bundle_handoff_summary(
        handoff_summary
    )
    candidate = _public_interaction_candidate(candidate_id)
    controlled_rows = checked["controlled_real_summary"]["rows"]
    rows: list[ObjectStateRealityRow] = []
    for index, row in enumerate(controlled_rows):
        rows.append(
            _public_interaction_row_from_controlled_row(
                row,
                index=index,
                candidate_has_action_gt=bool(candidate.has_action_gt),
                source_summary_ref=source_summary_ref,
            )
        )
    if {row.evidence_kind for row in rows} != set(_EVIDENCE_KINDS):
        raise ValueError(
            "public interaction reality rows require identity, prediction and "
            "intervention rows"
        )
    return tuple(rows)


def objectstate_public_interaction_reality_rows_summary(
    handoff_summary: Mapping[str, Any],
    *,
    candidate_id: str = "hot3d-clips",
    source_summary_ref: str | None = None,
    synthetic_smoke_passed: bool = True,
    thresholds: ObjectStateRealityGateThresholds | None = None,
) -> dict[str, Any]:
    checked = validate_objectstate_controlled_reality_bundle_handoff_summary(
        handoff_summary
    )
    candidate = _public_interaction_candidate(candidate_id)
    rows = objectstate_public_interaction_reality_rows_from_handoff(
        checked,
        candidate_id=candidate_id,
        source_summary_ref=source_summary_ref,
    )
    gate = evaluate_objectstate_reality_gate(
        rows,
        synthetic_smoke_passed=bool(synthetic_smoke_passed),
        thresholds=thresholds
        or ObjectStateRealityGateThresholds(min_real_or_public_rows=len(rows)),
    )
    rows = gate.rows
    payload = {
        "schema": OBJECTSTATE_PUBLIC_INTERACTION_REALITY_ROWS_SCHEMA,
        "kind": "objectstate_public_interaction_reality_rows",
        "status": "objectstate_public_interaction_reality_rows_reviewable",
        "source_schema": OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_HANDOFF_SCHEMA,
        "source_kind": _PUBLIC_INTERACTION_SOURCE_KIND,
        "source_summary_ref": source_summary_ref,
        "candidate": candidate.as_dict(),
        "sample_id": checked["sample"]["sample_id"],
        "row_schema": rows[0].schema,
        "row_count": len(rows),
        "pass_row_count": len(gate.pass_rows),
        "fail_row_count": len(gate.fail_rows),
        "blocked_row_count": len(gate.blocked_rows),
        "rows": [row.as_dict() for row in rows],
        "gate": gate.as_dict(),
        "blocked_rows_markdown": objectstate_reality_blocked_rows_markdown(gate),
        "source_handoff_status": checked["status"],
        "source_handoff_gates": dict(checked["handoff_gates"]),
        "row_sources": _row_sources(rows, source_summary_ref=source_summary_ref),
        "issues": _issues_from_handoff(checked),
        "claim_policy": {
            "imports_existing_controlled_reality_handoff_summary": True,
            "converts_public_interaction_rows_to_public_replay": True,
            "does_not_create_ground_truth": True,
            "does_not_run_handoff": True,
            "does_not_run_eval": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_train_model": True,
            "does_not_create_public_samples": True,
            "does_not_claim_counterfactual_proof": True,
            "does_not_claim_world_model": True,
            "blocked_rows_are_not_pass_rows": True,
            "failed_rows_remain_failed": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "captures_video": False,
            "creates_ground_truth": False,
            "runs_handoff": False,
            "runs_identity_eval": False,
            "runs_prediction_eval": False,
            "runs_intervention_eval": False,
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
    return validate_objectstate_public_interaction_reality_rows_summary(payload)


def validate_objectstate_public_interaction_reality_rows_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("public interaction reality rows summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_PUBLIC_INTERACTION_REALITY_ROWS_SCHEMA:
        raise ValueError(
            "unsupported public interaction reality rows schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_public_interaction_reality_rows":
        raise ValueError("public interaction reality rows kind is unsupported")
    if payload.get("status") != "objectstate_public_interaction_reality_rows_reviewable":
        raise ValueError("public interaction reality rows status is unsupported")
    if payload.get("source_schema") != OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_HANDOFF_SCHEMA:
        raise ValueError("public interaction reality rows source_schema is unsupported")
    if payload.get("source_kind") != _PUBLIC_INTERACTION_SOURCE_KIND:
        raise ValueError("public interaction reality rows source_kind must be public_replay")
    _validate_candidate_payload(payload.get("candidate"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("public interaction reality rows require exactly three rows")
    if payload.get("row_count") != len(rows):
        raise ValueError("public interaction reality row_count must match rows")
    if {row.get("evidence_kind") for row in rows} != set(_EVIDENCE_KINDS):
        raise ValueError(
            "public interaction rows must include identity, prediction and intervention"
        )
    if any(row.get("source_kind") != _PUBLIC_INTERACTION_SOURCE_KIND for row in rows):
        raise ValueError("public interaction rows must be source_kind=public_replay")
    gate = payload.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError("public interaction reality rows require gate")
    validate_objectstate_reality_gate_summary(dict(gate))
    if rows != gate.get("rows"):
        raise ValueError(
            "public interaction reality rows must match gate-derived rows"
        )
    if payload.get("pass_row_count") != gate.get("pass_row_count"):
        raise ValueError("public interaction pass_row_count must match gate")
    if payload.get("fail_row_count") != gate.get("fail_row_count"):
        raise ValueError("public interaction fail_row_count must match gate")
    if payload.get("blocked_row_count") != gate.get("blocked_row_count"):
        raise ValueError("public interaction blocked_row_count must match gate")
    if payload.get("row_count") != gate.get("row_count"):
        raise ValueError("public interaction row_count must match gate")
    if not isinstance(payload.get("blocked_rows_markdown"), str):
        raise ValueError("public interaction rows require blocked_rows_markdown")
    gates = payload.get("source_handoff_gates")
    if not isinstance(gates, Mapping) or any(
        not isinstance(value, bool) for value in gates.values()
    ):
        raise ValueError("public interaction rows require source_handoff_gates")
    if not isinstance(payload.get("row_sources"), list):
        raise ValueError("public interaction rows require row_sources")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("public interaction rows require issues")
    claim_policy = payload.get("claim_policy", {})
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("imports_existing_controlled_reality_handoff_summary")
        or not claim_policy.get("converts_public_interaction_rows_to_public_replay")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_run_handoff")
        or not claim_policy.get("does_not_run_eval")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_create_public_samples")
        or not claim_policy.get("does_not_claim_counterfactual_proof")
        or not claim_policy.get("does_not_claim_world_model")
        or not claim_policy.get("blocked_rows_are_not_pass_rows")
        or not claim_policy.get("failed_rows_remain_failed")
    ):
        raise ValueError("public interaction rows summary must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "public interaction rows cannot download, capture, create GT, run "
            "handoff/eval/model, reconstruct, train, write public samples, replay, "
            "diffuse, or mutate viewer policy"
        )
    return dict(payload)


def _public_interaction_row_from_controlled_row(
    row: Mapping[str, Any],
    *,
    index: int,
    candidate_has_action_gt: bool,
    source_summary_ref: str | None,
) -> ObjectStateRealityRow:
    ground_truth = row.get("ground_truth", {})
    if not isinstance(ground_truth, Mapping):
        raise ValueError("controlled handoff row requires ground_truth")
    metrics = dict(row.get("metrics", {}))
    evidence_kind = str(row["evidence_kind"])
    status = str(row["status"])
    block_reason = row.get("block_reason")
    failure_reason = row.get("failure_reason")
    if evidence_kind == "intervention":
        metrics.setdefault("action_challenge_present", True)
        if not candidate_has_action_gt:
            status = "blocked"
            block_reason = (
                "public dataset candidate does not provide native action ground truth"
            )
            failure_reason = None
    return validate_objectstate_reality_row(
        ObjectStateRealityRow(
            row_id=(
                f"{row['sample_id']}:public_interaction:"
                f"{row['evidence_kind']}:{index:03d}"
            ),
            sample_id=str(row["sample_id"]),
            source_kind=_PUBLIC_INTERACTION_SOURCE_KIND,
            evidence_kind=evidence_kind,
            status=status,
            object_category=str(row["object_category"]),
            scenario=str(row["scenario"]),
            observation_modalities=tuple(str(item) for item in row["observation_modalities"]),
            artifact_refs=tuple(_artifact_refs(row, source_summary_ref)),
            metrics=metrics,
            has_identity_gt=bool(ground_truth.get("identity")),
            has_pose_gt=bool(ground_truth.get("pose")),
            has_action_gt=(
                bool(ground_truth.get("action")) and bool(candidate_has_action_gt)
            ),
            has_timestamp=bool(ground_truth.get("timestamp")),
            license=str(row.get("license", "unknown")),
            block_reason=block_reason,
            failure_reason=failure_reason,
        )
    )


def _artifact_refs(row: Mapping[str, Any], source_summary_ref: str | None) -> list[str]:
    refs = [str(item) for item in row.get("artifact_refs", ())]
    if source_summary_ref:
        refs.append(source_summary_ref)
    return refs


def _public_interaction_candidate(candidate_id: str):
    normalized = str(candidate_id).strip()
    for candidate in default_objectstate_public_dataset_candidates():
        checked = validate_objectstate_public_dataset_candidate(candidate)
        if checked.candidate_id == normalized:
            if checked.source_kind != "public_interaction_dataset":
                raise ValueError(
                    f"candidate is not a public interaction dataset: {candidate_id}"
                )
            return checked
    raise ValueError(f"unknown public interaction candidate id: {candidate_id}")


def _row_sources(
    rows: Sequence[ObjectStateRealityRow],
    *,
    source_summary_ref: str | None,
) -> list[dict[str, Any]]:
    return [
        {
            "row_id": row.row_id,
            "sample_id": row.sample_id,
            "evidence_kind": row.evidence_kind,
            "source_kind": row.source_kind,
            "source_summary_ref": source_summary_ref,
            "artifact_refs": list(row.artifact_refs),
        }
        for row in rows
    ]


def _issues_from_handoff(handoff: Mapping[str, Any]) -> list[str]:
    issues = list(str(item) for item in handoff.get("issues", ()))
    if handoff["status"] != "objectstate_controlled_reality_bundle_handoff_pass":
        issues.append("source controlled reality handoff did not pass")
    issues.append(
        "public interaction rows are observed action evidence, not randomized counterfactual proof"
    )
    return issues


def _validate_candidate_payload(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("public interaction rows require candidate payload")
    if payload.get("source_kind") != "public_interaction_dataset":
        raise ValueError("candidate payload must be public_interaction_dataset")
    ground_truth = payload.get("ground_truth")
    if not isinstance(ground_truth, Mapping) or not isinstance(
        ground_truth.get("action"), bool
    ):
        raise ValueError("candidate payload must declare whether action ground truth exists")


__all__ = (
    "OBJECTSTATE_PUBLIC_INTERACTION_REALITY_ROWS_SCHEMA",
    "read_objectstate_public_interaction_handoff_summary",
    "objectstate_public_interaction_reality_rows_from_handoff",
    "objectstate_public_interaction_reality_rows_summary",
    "validate_objectstate_public_interaction_reality_rows_summary",
)
