from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from objgauss.evaluation.objectstate_reality_gate import (
    ObjectStateRealityGateReport,
    ObjectStateRealityRow,
    evaluate_objectstate_reality_gate,
    objectstate_reality_blocked_rows_markdown,
    validate_objectstate_reality_gate_summary,
)

OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA = "objgauss-objectstate-public-artifact-rows-v1"

__all__ = (
    "OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA",
    "ObjectStateRealityPublicArtifact",
    "default_objectstate_reality_public_artifacts",
    "objectstate_reality_rows_from_public_artifacts",
    "evaluate_public_artifact_reality_gate",
    "objectstate_reality_public_rows_summary",
    "validate_objectstate_reality_public_rows_summary",
    "validate_objectstate_reality_public_artifact",
)


@dataclass(frozen=True)
class ObjectStateRealityPublicArtifact:
    sample_id: str
    object_category: str
    scenario: str
    artifact_refs: tuple[str, ...]
    license: str
    observation_modalities: tuple[str, ...] = ("gaussian", "object_id")
    source_kind: str = "public_replay"
    has_identity_gt: bool = False
    has_pose_gt: bool = False
    has_action_gt: bool = False
    has_timestamp: bool = False

    def as_dict(self) -> dict[str, Any]:
        artifact = validate_objectstate_reality_public_artifact(self)
        return {
            "sample_id": artifact.sample_id,
            "source_kind": artifact.source_kind,
            "object_category": artifact.object_category,
            "scenario": artifact.scenario,
            "observation_modalities": list(artifact.observation_modalities),
            "artifact_refs": list(artifact.artifact_refs),
            "license": artifact.license,
            "ground_truth": {
                "identity": bool(artifact.has_identity_gt),
                "pose": bool(artifact.has_pose_gt),
                "action": bool(artifact.has_action_gt),
                "timestamp": bool(artifact.has_timestamp),
            },
        }


def default_objectstate_reality_public_artifacts() -> tuple[ObjectStateRealityPublicArtifact, ...]:
    return (
        ObjectStateRealityPublicArtifact(
            sample_id="real-sample-v2-sample-aware-lego",
            object_category="lego",
            scenario="sample_aware_object_layer_preview",
            artifact_refs=(
                "src/modelCatalog.js:real-sample-v2-sample-aware-lego",
                "public/samples/objgauss-real-sample-v2-sample-aware-lego.ply",
                "public/samples/lego_alpha_proxy.splat",
            ),
            license="NeRF official example data; training/research use only",
            observation_modalities=("gaussian", "splat", "object_id"),
        ),
        ObjectStateRealityPublicArtifact(
            sample_id="polyhaven-chair",
            object_category="chair",
            scenario="cc0_single_object_gaussian_demo",
            artifact_refs=(
                "src/modelCatalog.js:polyhaven-chair",
                "public/samples/polyhaven_chair_demo_objects.ply",
                "public/samples/polyhaven_chair_demo.splat",
            ),
            license="CC0 Poly Haven School Chair derived training output",
            observation_modalities=("gaussian", "splat", "object_id"),
        ),
        ObjectStateRealityPublicArtifact(
            sample_id="nike-real-splat-demo",
            object_category="shoe",
            scenario="local_real_splat_object_edit_smoke",
            artifact_refs=(
                "src/modelCatalog.js:nike-real-splat-demo",
                "public/samples/nike_objects.ply",
                "public/samples/nike.splat",
            ),
            license="mixed upstream source license; local testing only",
            observation_modalities=("gaussian", "splat", "object_id"),
        ),
        ObjectStateRealityPublicArtifact(
            sample_id="plush",
            object_category="toy",
            scenario="local_real_splat_object_edit_smoke",
            artifact_refs=(
                "src/modelCatalog.js:plush",
                "public/samples/plush_objects.ply",
                "public/samples/plush.splat",
            ),
            license="mixed upstream source license; local testing only",
            observation_modalities=("gaussian", "splat", "object_id"),
        ),
    )


def objectstate_reality_rows_from_public_artifacts(
    artifacts: Sequence[ObjectStateRealityPublicArtifact] | None = None,
) -> tuple[ObjectStateRealityRow, ...]:
    resolved_artifacts = tuple(
        default_objectstate_reality_public_artifacts()
        if artifacts is None
        else artifacts
    )
    if not resolved_artifacts:
        raise ValueError("public artifact rows require at least one artifact")
    rows: list[ObjectStateRealityRow] = []
    for artifact in resolved_artifacts:
        checked = validate_objectstate_reality_public_artifact(artifact)
        rows.extend(_blocked_rows_for_artifact(checked))
    return tuple(rows)


def evaluate_public_artifact_reality_gate(
    artifacts: Sequence[ObjectStateRealityPublicArtifact] | None = None,
    *,
    synthetic_smoke_passed: bool = True,
) -> ObjectStateRealityGateReport:
    return evaluate_objectstate_reality_gate(
        objectstate_reality_rows_from_public_artifacts(artifacts),
        synthetic_smoke_passed=bool(synthetic_smoke_passed),
    )


def objectstate_reality_public_rows_summary(
    artifacts: Sequence[ObjectStateRealityPublicArtifact] | None = None,
    *,
    synthetic_smoke_passed: bool = True,
) -> dict[str, Any]:
    resolved_artifacts = tuple(
        default_objectstate_reality_public_artifacts()
        if artifacts is None
        else artifacts
    )
    checked_artifacts = tuple(
        validate_objectstate_reality_public_artifact(artifact)
        for artifact in resolved_artifacts
    )
    report = evaluate_public_artifact_reality_gate(
        checked_artifacts,
        synthetic_smoke_passed=bool(synthetic_smoke_passed),
    )
    payload = {
        "schema": OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA,
        "kind": "objectstate_reality_public_artifact_rows",
        "artifact_count": len(checked_artifacts),
        "artifacts": [artifact.as_dict() for artifact in checked_artifacts],
        "row_count": len(report.rows),
        "blocked_row_count": len(report.blocked_rows),
        "pass_row_count": len(report.pass_rows),
        "fail_row_count": len(report.fail_rows),
        "gate": report.as_dict(),
        "blocked_rows_markdown": objectstate_reality_blocked_rows_markdown(report),
        "claim_policy": {
            "public_artifact_rows_are_reality_evidence_candidates": True,
            "current_rows_are_blocked_not_pass": True,
            "object_id_is_not_identity_ground_truth": True,
            "does_not_claim_real_world_state_variable": True,
        },
        "non_goals": {
            "writes_public_samples": False,
            "submits_generated_outputs": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_reality_public_rows_summary(payload)


def validate_objectstate_reality_public_rows_summary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("objectstate reality public rows summary must be a dict")
    if payload.get("schema") != OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA:
        raise ValueError(f"unsupported objectstate reality public rows schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_reality_public_artifact_rows":
        raise ValueError("objectstate reality public rows kind is unsupported")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("objectstate reality public rows require artifacts")
    if payload.get("artifact_count") != len(artifacts):
        raise ValueError("artifact_count must match artifacts")
    gate = payload.get("gate")
    if not isinstance(gate, dict):
        raise ValueError("objectstate reality public rows require gate")
    validate_objectstate_reality_gate_summary(gate)
    if payload.get("row_count") != gate.get("row_count"):
        raise ValueError("row_count must match gate row_count")
    if payload.get("blocked_row_count") != gate.get("blocked_row_count"):
        raise ValueError("blocked_row_count must match gate blocked_row_count")
    if payload.get("pass_row_count") != gate.get("pass_row_count"):
        raise ValueError("pass_row_count must match gate pass_row_count")
    if payload.get("fail_row_count") != gate.get("fail_row_count"):
        raise ValueError("fail_row_count must match gate fail_row_count")
    if not isinstance(payload.get("blocked_rows_markdown"), str):
        raise ValueError("blocked_rows_markdown must be a string")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("current_rows_are_blocked_not_pass")
        or not claim_policy.get("object_id_is_not_identity_ground_truth")
        or not claim_policy.get("does_not_claim_real_world_state_variable")
    ):
        raise ValueError("public rows summary must keep blocked/object_id/no-world-state claims")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("writes_public_samples")
        or non_goals.get("submits_generated_outputs")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError("public rows summary cannot write outputs, train, replay, diffuse, or mutate viewer policy")
    return payload


def validate_objectstate_reality_public_artifact(
    artifact: ObjectStateRealityPublicArtifact,
) -> ObjectStateRealityPublicArtifact:
    if not isinstance(artifact, ObjectStateRealityPublicArtifact):
        raise TypeError("artifact must be ObjectStateRealityPublicArtifact")
    if not artifact.sample_id:
        raise ValueError("sample_id must be non-empty")
    if artifact.source_kind != "public_replay":
        raise ValueError("public artifact rows must use source_kind=public_replay")
    if not artifact.object_category:
        raise ValueError("object_category must be non-empty")
    if not artifact.scenario:
        raise ValueError("scenario must be non-empty")
    refs = _non_empty_string_tuple(artifact.artifact_refs, "artifact_refs")
    modalities = _non_empty_string_tuple(
        artifact.observation_modalities,
        "observation_modalities",
    )
    if not artifact.license:
        raise ValueError("license must be non-empty")
    return ObjectStateRealityPublicArtifact(
        sample_id=str(artifact.sample_id),
        object_category=str(artifact.object_category),
        scenario=str(artifact.scenario),
        artifact_refs=refs,
        license=str(artifact.license),
        observation_modalities=modalities,
        source_kind=artifact.source_kind,
        has_identity_gt=bool(artifact.has_identity_gt),
        has_pose_gt=bool(artifact.has_pose_gt),
        has_action_gt=bool(artifact.has_action_gt),
        has_timestamp=bool(artifact.has_timestamp),
    )


def _blocked_rows_for_artifact(
    artifact: ObjectStateRealityPublicArtifact,
) -> tuple[ObjectStateRealityRow, ...]:
    return (
        _blocked_row(
            artifact,
            evidence_kind="identity",
            block_reason=(
                "missing timestamped cross-view or occlusion identity ground truth; "
                "object_id labels are renderer addresses, not physical identity GT"
            ),
        ),
        _blocked_row(
            artifact,
            evidence_kind="prediction",
            block_reason=(
                "missing timestamped 6DoF pose tracks and history-vs-state "
                "future-pose targets"
            ),
        ),
        _blocked_row(
            artifact,
            evidence_kind="intervention",
            block_reason=(
                "missing action events, 6DoF intervention outcomes and "
                "counterfactual targets"
            ),
        ),
    )


def _blocked_row(
    artifact: ObjectStateRealityPublicArtifact,
    *,
    evidence_kind: str,
    block_reason: str,
) -> ObjectStateRealityRow:
    return ObjectStateRealityRow(
        row_id=f"{artifact.sample_id}:{evidence_kind}:blocked",
        sample_id=artifact.sample_id,
        source_kind=artifact.source_kind,
        evidence_kind=evidence_kind,
        status="blocked",
        object_category=artifact.object_category,
        scenario=artifact.scenario,
        observation_modalities=artifact.observation_modalities,
        artifact_refs=artifact.artifact_refs,
        metrics={},
        has_identity_gt=artifact.has_identity_gt,
        has_pose_gt=artifact.has_pose_gt,
        has_action_gt=artifact.has_action_gt,
        has_timestamp=artifact.has_timestamp,
        license=artifact.license,
        block_reason=block_reason,
    )


def _non_empty_string_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    normalized = tuple(str(value) for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{name} must contain non-empty strings")
    return normalized
