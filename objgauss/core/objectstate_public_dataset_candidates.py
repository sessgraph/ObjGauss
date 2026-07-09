from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_capture import (
    objectstate_controlled_capture_summary,
    read_objectstate_controlled_capture_manifest,
)
from objgauss.core.objectstate_controlled_capture_intervention_action_gt import (
    objectstate_controlled_capture_intervention_action_gt_readiness,
)
from objgauss.core.objectstate_controlled_intervention_eval import (
    read_objectstate_controlled_intervention_candidates,
)
from objgauss.core.objectstate_controlled_prediction_eval import (
    read_objectstate_controlled_prediction_candidates,
)

OBJECTSTATE_PUBLIC_DATASET_CANDIDATES_SCHEMA = (
    "objgauss-objectstate-public-dataset-candidates-v1"
)
OBJECTSTATE_PUBLIC_DATASET_CANDIDATE_SCHEMA = (
    "objgauss-objectstate-public-dataset-candidate-v1"
)
OBJECTSTATE_PUBLIC_INTERACTION_ROUTE_AUDIT_SCHEMA = (
    "objgauss-objectstate-public-interaction-route-audit-v1"
)

OBJECTSTATE_PUBLIC_DATASET_GATE_KINDS = (
    "identity_persistence",
    "occlusion_recovery",
    "view_invariance",
    "predictive_sufficiency",
    "counterfactual_interface",
)
OBJECTSTATE_PUBLIC_DATASET_COVERAGE_VALUES = (
    "ready_with_adapter",
    "partial",
    "blocked",
)
OBJECTSTATE_PUBLIC_DATASET_SOURCE_KINDS = (
    "public_pose_dataset",
    "public_interaction_dataset",
)


@dataclass(frozen=True)
class ObjectStatePublicDatasetCandidate:
    candidate_id: str
    name: str
    source_url: str
    source_kind: str
    source_license: str
    access: str
    download_size_note: str
    recommended_role: str
    object_category: str
    sequence_kind: str
    observation_modalities: tuple[str, ...]
    gate_coverage: Mapping[str, str]
    blockers: tuple[str, ...]
    next_actions: tuple[str, ...]
    source_notes: tuple[str, ...]
    has_identity_gt: bool
    has_pose_gt: bool
    has_action_gt: bool
    has_timestamp_or_frame_order: bool
    has_camera_gt: bool
    has_occlusion_or_visibility_signal: bool
    has_multiview_or_view_change: bool
    has_object_models: bool
    has_gaussian_evidence: bool = False
    schema: str = OBJECTSTATE_PUBLIC_DATASET_CANDIDATE_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        candidate = validate_objectstate_public_dataset_candidate(self)
        return {
            "schema": candidate.schema,
            "candidate_id": candidate.candidate_id,
            "name": candidate.name,
            "source_url": candidate.source_url,
            "source_kind": candidate.source_kind,
            "source_license": candidate.source_license,
            "access": candidate.access,
            "download_size_note": candidate.download_size_note,
            "recommended_role": candidate.recommended_role,
            "object_category": candidate.object_category,
            "sequence_kind": candidate.sequence_kind,
            "observation_modalities": list(candidate.observation_modalities),
            "ground_truth": {
                "identity": bool(candidate.has_identity_gt),
                "pose_6d": bool(candidate.has_pose_gt),
                "action": bool(candidate.has_action_gt),
                "timestamp_or_frame_order": bool(
                    candidate.has_timestamp_or_frame_order
                ),
                "camera": bool(candidate.has_camera_gt),
                "occlusion_or_visibility": bool(
                    candidate.has_occlusion_or_visibility_signal
                ),
                "multiview_or_view_change": bool(
                    candidate.has_multiview_or_view_change
                ),
                "object_models": bool(candidate.has_object_models),
                "gaussian_evidence": bool(candidate.has_gaussian_evidence),
            },
            "gate_coverage": dict(candidate.gate_coverage),
            "blockers": list(candidate.blockers),
            "next_actions": list(candidate.next_actions),
            "source_notes": list(candidate.source_notes),
        }


def default_objectstate_public_dataset_candidates() -> (
    tuple[ObjectStatePublicDatasetCandidate, ...]
):
    return (
        ObjectStatePublicDatasetCandidate(
            candidate_id="bop-ycbv-keyframes",
            name="BOP YCB-V / YCB-Video",
            source_url="https://bop.felk.cvut.cz/datasets/#YCB-V",
            source_kind="public_pose_dataset",
            source_license=(
                "MIT per BOP dataset page; verify original YCB object/model terms "
                "before redistribution"
            ),
            access="BOP HuggingFace dataset archives",
            download_size_note=(
                "Use base archive, object models, and one small real scene/keyframe "
                "subset first; do not import full archives into git."
            ),
            recommended_role="first public identity/prediction adapter candidate",
            object_category="21 household YCB objects",
            sequence_kind="cluttered RGB-D video/keyframe pose dataset",
            observation_modalities=("rgb", "depth", "mask", "6d_pose", "3d_models"),
            has_identity_gt=True,
            has_pose_gt=True,
            has_action_gt=False,
            has_timestamp_or_frame_order=True,
            has_camera_gt=True,
            has_occlusion_or_visibility_signal=True,
            has_multiview_or_view_change=True,
            has_object_models=True,
            gate_coverage={
                "identity_persistence": "ready_with_adapter",
                "occlusion_recovery": "partial",
                "view_invariance": "partial",
                "predictive_sufficiency": "partial",
                "counterfactual_interface": "blocked",
            },
            blockers=(
                "no action or intervention ground truth",
                "BOP keyframes may be sparse; ordered video adapter must verify continuity",
                "no Gaussian reconstruction evidence until local frames are reconstructed",
            ),
            next_actions=(
                "select one small YCB-V scene/keyframe subset outside git",
                "write an adapter to controlled capture manifest rows",
                "produce per-frame Gaussian evidence under ignored outputs/",
            ),
            source_notes=(
                "BOP lists YCB-V as 21 YCB objects captured in 92 videos.",
                "BOP provides labels, 6D poses, boxes, masks, models, and download archives.",
            ),
        ),
        ObjectStatePublicDatasetCandidate(
            candidate_id="bop-hopev2",
            name="BOP HOPE / HOPEv2",
            source_url="https://bop.felk.cvut.cz/datasets/#HOPE",
            source_kind="public_pose_dataset",
            source_license="CC BY-SA 4.0 per BOP dataset page",
            access="BOP HuggingFace dataset archives",
            download_size_note=(
                "Start with validation or Vicon test subset; keep archives outside git."
            ),
            recommended_role="lighting/view/occlusion robustness candidate",
            object_category="toy grocery household objects",
            sequence_kind="cluttered household/office scenes with lighting and moving camera",
            observation_modalities=("rgb", "depth", "mask", "6d_pose", "3d_models"),
            has_identity_gt=True,
            has_pose_gt=True,
            has_action_gt=False,
            has_timestamp_or_frame_order=True,
            has_camera_gt=True,
            has_occlusion_or_visibility_signal=True,
            has_multiview_or_view_change=True,
            has_object_models=True,
            gate_coverage={
                "identity_persistence": "ready_with_adapter",
                "occlusion_recovery": "partial",
                "view_invariance": "ready_with_adapter",
                "predictive_sufficiency": "partial",
                "counterfactual_interface": "blocked",
            },
            blockers=(
                "no action or counterfactual intervention outcomes",
                "dynamic onboarding ground truth availability is split by frame type",
                "no Gaussian reconstruction evidence until local conversion is done",
            ),
            next_actions=(
                "prefer one Vicon/moving-camera subset for view invariance",
                "map BOP pose/mask fields into controlled capture annotations",
                "record CC BY-SA inheritance before any public artifact claim",
            ),
            source_notes=(
                "BOP describes clutter, multiple lighting variants, and HOPEv2 moving-camera data.",
                "BOP states training/validation pose ground truth is publicly available.",
            ),
        ),
        ObjectStatePublicDatasetCandidate(
            candidate_id="bop-tudl",
            name="BOP TUD-L",
            source_url="https://bop.felk.cvut.cz/datasets/#TUD-L",
            source_kind="public_pose_dataset",
            source_license="CC BY-SA 4.0 per BOP dataset page",
            access="BOP HuggingFace dataset archives",
            download_size_note=(
                "Small candidate because it has only three moving objects; keep archives "
                "outside git."
            ),
            recommended_role="small lighting/motion sanity candidate",
            object_category="three moving objects",
            sequence_kind="moving-object image sequences under eight lighting conditions",
            observation_modalities=("rgb", "6d_pose", "3d_models"),
            has_identity_gt=True,
            has_pose_gt=True,
            has_action_gt=False,
            has_timestamp_or_frame_order=True,
            has_camera_gt=True,
            has_occlusion_or_visibility_signal=False,
            has_multiview_or_view_change=True,
            has_object_models=True,
            gate_coverage={
                "identity_persistence": "ready_with_adapter",
                "occlusion_recovery": "blocked",
                "view_invariance": "ready_with_adapter",
                "predictive_sufficiency": "partial",
                "counterfactual_interface": "blocked",
            },
            blockers=(
                "too small for category generalization",
                "does not provide a strong occlusion recovery scenario",
                "no action or intervention ground truth",
                "no Gaussian reconstruction evidence until local conversion is done",
            ),
            next_actions=(
                "use only as a cheap adapter smoke after YCB-V or HOPEv2",
                "verify frame ordering and camera metadata before predictive rows",
            ),
            source_notes=(
                "BOP describes TUD-L as three moving objects under eight lighting conditions.",
            ),
        ),
        ObjectStatePublicDatasetCandidate(
            candidate_id="hot3d-clips",
            name="HOT3D-Clips",
            source_url="https://facebookresearch.github.io/hot3d/",
            source_kind="public_interaction_dataset",
            source_license="HOT3D license agreement",
            access="HOT3D/HOT3D-Clips downloads and toolkit",
            download_size_note=(
                "Use one 150-frame clip first; full HOT3D is large and agreement-gated."
            ),
            recommended_role="first action-like interaction candidate after pose adapter",
            object_category="33 rigid household/office objects with hands",
            sequence_kind="egocentric multi-view hand-object interaction clips",
            observation_modalities=(
                "rgb",
                "monochrome",
                "hand_pose",
                "object_pose",
                "camera_pose",
                "3d_models",
            ),
            has_identity_gt=True,
            has_pose_gt=True,
            has_action_gt=True,
            has_timestamp_or_frame_order=True,
            has_camera_gt=True,
            has_occlusion_or_visibility_signal=True,
            has_multiview_or_view_change=True,
            has_object_models=True,
            gate_coverage={
                "identity_persistence": "ready_with_adapter",
                "occlusion_recovery": "ready_with_adapter",
                "view_invariance": "ready_with_adapter",
                "predictive_sufficiency": "ready_with_adapter",
                "counterfactual_interface": "partial",
            },
            blockers=(
                "license agreement must be reviewed before redistribution",
                "egocentric fisheye/multi-stream format needs dedicated adapter",
                "observed interactions are not randomized counterfactual trials",
                "no Gaussian reconstruction evidence until local conversion is done",
            ),
            next_actions=(
                "defer until BOP pose adapter proves the controlled manifest path",
                "select one clip with clear pick-up/put-down interaction",
                "treat action rows as observational intervention candidates, not causal proof",
            ),
            source_notes=(
                "HOT3D reports 833 minutes, 30 FPS, multi-view recordings, object/hand/camera 3D poses.",
                "HOT3D-Clips are 150-frame clips with ground-truth annotations for modeled objects and hands.",
            ),
        ),
        ObjectStatePublicDatasetCandidate(
            candidate_id="dexycb",
            name="DexYCB",
            source_url="https://dex-ycb.github.io/",
            source_kind="public_interaction_dataset",
            source_license="CC BY-NC 4.0",
            access="DexYCB project downloads and toolkit",
            download_size_note=(
                "Full compressed dataset is listed as 119G; BOP-format subset is "
                "listed as 1.2G."
            ),
            recommended_role="non-commercial hand-grasp stress candidate",
            object_category="YCB objects in human hand grasping sequences",
            sequence_kind="hand grasping / object pose benchmark",
            observation_modalities=(
                "rgb",
                "6d_pose",
                "2d_detection",
                "keypoints",
                "hand_pose",
                "3d_models",
            ),
            has_identity_gt=True,
            has_pose_gt=True,
            has_action_gt=False,
            has_timestamp_or_frame_order=True,
            has_camera_gt=True,
            has_occlusion_or_visibility_signal=True,
            has_multiview_or_view_change=True,
            has_object_models=True,
            gate_coverage={
                "identity_persistence": "ready_with_adapter",
                "occlusion_recovery": "partial",
                "view_invariance": "partial",
                "predictive_sufficiency": "partial",
                "counterfactual_interface": "blocked",
            },
            blockers=(
                "CC BY-NC 4.0 blocks public commercial demo claims",
                "large download footprint",
                "hand grasping is action-like but not a counterfactual action label contract",
                "no Gaussian reconstruction evidence until local conversion is done",
            ),
            next_actions=(
                "use only after smaller BOP candidates if hand occlusion stress is needed",
                "never mix this candidate into commercial/public demo eligibility",
            ),
            source_notes=(
                "DexYCB lists tasks including 6D object pose and 3D hand pose.",
                "DexYCB project page lists CC BY-NC 4.0 and 119G full download size.",
            ),
        ),
    )


def objectstate_public_dataset_candidates_audit(
    candidates: Sequence[ObjectStatePublicDatasetCandidate] | None = None,
) -> dict[str, Any]:
    resolved_candidates = tuple(
        default_objectstate_public_dataset_candidates()
        if candidates is None
        else candidates
    )
    if not resolved_candidates:
        raise ValueError("public dataset candidate audit requires at least one candidate")
    checked = tuple(
        validate_objectstate_public_dataset_candidate(candidate)
        for candidate in resolved_candidates
    )
    coverage_counts = _coverage_counts(checked)
    hard_blockers = list(_hard_blockers(checked))
    payload = {
        "schema": OBJECTSTATE_PUBLIC_DATASET_CANDIDATES_SCHEMA,
        "kind": "objectstate_public_dataset_candidate_audit",
        "status": "objectstate_public_dataset_candidates_audited",
        "candidate_count": len(checked),
        "candidates": [candidate.as_dict() for candidate in checked],
        "recommended_order": (
            "bop-ycbv-keyframes",
            "bop-hopev2",
            "bop-tudl",
            "hot3d-clips",
            "dexycb",
        ),
        "recommended_first": "bop-ycbv-keyframes",
        "recommended_action_candidate": "hot3d-clips",
        "coverage_counts": coverage_counts,
        "readiness": {
            "has_identity_pose_dataset": any(
                candidate.has_identity_gt and candidate.has_pose_gt
                for candidate in checked
            ),
            "has_action_like_dataset": any(candidate.has_action_gt for candidate in checked),
            "has_direct_gaussian_evidence": any(
                candidate.has_gaussian_evidence for candidate in checked
            ),
            "has_direct_phase1_ready_dataset": False,
            "requires_controlled_capture_adapter": True,
            "requires_local_gaussian_reconstruction": True,
            "requires_license_review_before_public_release": True,
        },
        "hard_blockers": hard_blockers,
        "next_actions": (
            "start with one BOP YCB-V or HOPEv2 scene subset outside git",
            "write/import a controlled capture manifest adapter for RGB, pose, masks, cameras, and frame order",
            "generate local per-frame Gaussian evidence under ignored outputs/",
            "run identity and prediction rows before using HOT3D action-like rows",
            "keep counterfactual claims blocked until action-conditioned outcomes are evaluated",
        ),
        "claim_policy": {
            "candidate_audit_only": True,
            "does_not_download_datasets": True,
            "does_not_create_reality_rows": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
            "blocked_rows_are_not_pass_rows": True,
            "public_dataset_is_not_public_demo_asset": True,
        },
        "non_goals": {
            "downloads_datasets": False,
            "writes_outputs": False,
            "writes_public_samples": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_public_dataset_candidates_audit(payload)


def objectstate_public_dataset_candidates_markdown(summary: Mapping[str, Any]) -> str:
    payload = validate_objectstate_public_dataset_candidates_audit(summary)
    lines = [
        "# ObjectState Public Dataset Candidate Audit",
        "",
        f"- schema: `{payload['schema']}`",
        f"- candidates: `{payload['candidate_count']}`",
        f"- recommended first: `{payload['recommended_first']}`",
        f"- recommended action candidate: `{payload['recommended_action_candidate']}`",
        "",
        "| candidate | role | license | identity | occlusion | view | prediction | action |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in payload["candidates"]:
        coverage = candidate["gate_coverage"]
        lines.append(
            "| "
            + " | ".join(
                (
                    candidate["candidate_id"],
                    candidate["recommended_role"],
                    candidate["source_license"],
                    coverage["identity_persistence"],
                    coverage["occlusion_recovery"],
                    coverage["view_invariance"],
                    coverage["predictive_sufficiency"],
                    coverage["counterfactual_interface"],
                )
            )
            + " |"
        )
    lines.extend(("", "## Hard Blockers", ""))
    for blocker in payload["hard_blockers"]:
        lines.append(f"- {blocker}")
    lines.extend(("", "## Next Actions", ""))
    for action in payload["next_actions"]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def objectstate_public_interaction_route_audit(
    *,
    candidate_id: str = "hot3d-clips",
    dataset_root: str | Path | None = None,
    capture_manifest: str | Path | None = None,
    candidate_artifact: str | Path | None = None,
    prediction_candidates: str | Path | None = None,
    intervention_candidates: str | Path | None = None,
    candidates: Sequence[ObjectStatePublicDatasetCandidate] | None = None,
) -> dict[str, Any]:
    resolved_candidates = tuple(
        default_objectstate_public_dataset_candidates()
        if candidates is None
        else candidates
    )
    checked_candidates = tuple(
        validate_objectstate_public_dataset_candidate(candidate)
        for candidate in resolved_candidates
    )
    candidate = _candidate_by_id(candidate_id, checked_candidates)
    root_path = Path(dataset_root) if dataset_root is not None else None
    resolved_paths = _public_interaction_route_paths(
        root_path,
        capture_manifest=capture_manifest,
        candidate_artifact=candidate_artifact,
        prediction_candidates=prediction_candidates,
        intervention_candidates=intervention_candidates,
    )
    capture_record = _capture_manifest_record(resolved_paths["capture_manifest"])
    prediction_record = _candidate_json_record(
        resolved_paths["prediction_candidates"],
        reader=read_objectstate_controlled_prediction_candidates,
    )
    intervention_record = _candidate_json_record(
        resolved_paths["intervention_candidates"],
        reader=read_objectstate_controlled_intervention_candidates,
    )
    artifact_record = _plain_file_record(resolved_paths["candidate_artifact"])
    sample_ids = _sample_id_binding(
        capture_record=capture_record,
        prediction_record=prediction_record,
        intervention_record=intervention_record,
    )
    readiness = {
        "candidate_is_public_interaction_dataset": (
            candidate.source_kind == "public_interaction_dataset"
        ),
        "candidate_has_action_gt": bool(candidate.has_action_gt),
        "dataset_root_present": bool(root_path is not None and root_path.exists()),
        "capture_manifest_present": bool(capture_record["present"]),
        "capture_manifest_valid": bool(capture_record["valid"]),
        "capture_identity_ready": bool(capture_record["readiness"].get("identity_stage_ready")),
        "capture_prediction_ready": bool(
            capture_record["readiness"].get("prediction_stage_ready")
        ),
        "capture_intervention_ready": bool(
            capture_record["readiness"].get("intervention_stage_ready")
        ),
        "capture_intervention_action_gt_ready": bool(
            capture_record["readiness"].get("intervention_action_gt_ready")
        ),
        "gaussian_evidence_declared": bool(
            capture_record["readiness"].get("real_gaussian_reconstruction_present")
        ),
        "candidate_artifact_present": bool(artifact_record["present"]),
        "prediction_candidates_valid": bool(prediction_record["valid"]),
        "intervention_candidates_valid": bool(intervention_record["valid"]),
        "sample_ids_match": bool(sample_ids["match"]),
        "prediction_sample_id_match": bool(sample_ids["prediction_match"]),
        "intervention_sample_id_match": bool(sample_ids["intervention_match"]),
    }
    readiness["controlled_reality_handoff_ready"] = bool(
        readiness["candidate_is_public_interaction_dataset"]
        and readiness["candidate_has_action_gt"]
        and readiness["dataset_root_present"]
        and readiness["capture_intervention_ready"]
        and readiness["capture_intervention_action_gt_ready"]
        and readiness["gaussian_evidence_declared"]
        and readiness["candidate_artifact_present"]
        and readiness["prediction_candidates_valid"]
        and readiness["intervention_candidates_valid"]
        and readiness["sample_ids_match"]
    )
    readiness["identity_accounting_ready"] = bool(
        readiness["candidate_is_public_interaction_dataset"]
        and readiness["candidate_has_action_gt"]
        and readiness["dataset_root_present"]
        and readiness["capture_manifest_valid"]
        and readiness["capture_identity_ready"]
        and readiness["gaussian_evidence_declared"]
        and readiness["candidate_artifact_present"]
    )
    readiness["prediction_accounting_ready"] = bool(
        readiness["identity_accounting_ready"]
        and readiness["capture_prediction_ready"]
        and readiness["prediction_candidates_valid"]
        and readiness["prediction_sample_id_match"]
    )
    readiness["intervention_accounting_ready"] = bool(
        readiness["controlled_reality_handoff_ready"]
    )
    status = _public_interaction_route_status(readiness)
    payload = {
        "schema": OBJECTSTATE_PUBLIC_INTERACTION_ROUTE_AUDIT_SCHEMA,
        "kind": "objectstate_public_interaction_route_audit",
        "status": status,
        "accounting_route_status": _public_interaction_accounting_route_status(
            readiness
        ),
        "candidate": candidate.as_dict(),
        "dataset_root": _path_string(root_path),
        "paths": {
            key: _path_string(path)
            for key, path in resolved_paths.items()
        },
        "file_records": {
            "capture_manifest": capture_record,
            "candidate_artifact": artifact_record,
            "prediction_candidates": prediction_record,
            "intervention_candidates": intervention_record,
        },
        "sample_id_binding": sample_ids,
        "readiness": readiness,
        "hard_blockers": _public_interaction_route_blockers(readiness),
        "next_actions": _public_interaction_route_next_actions(readiness, resolved_paths),
        "claim_policy": {
            "route_audit_only": True,
            "does_not_download_datasets": True,
            "does_not_create_ground_truth": True,
            "does_not_create_reality_rows": True,
            "does_not_claim_intervention_pass": True,
            "does_not_claim_counterfactual_proof": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_datasets": False,
            "writes_outputs": False,
            "writes_public_samples": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "runs_intervention_model": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_public_interaction_route_audit(payload)


def objectstate_public_interaction_route_markdown(summary: Mapping[str, Any]) -> str:
    payload = validate_objectstate_public_interaction_route_audit(summary)
    readiness = payload["readiness"]
    lines = [
        "# ObjectState Public Interaction Route Audit",
        "",
        f"- schema: `{payload['schema']}`",
        f"- status: `{payload['status']}`",
        f"- accounting route status: `{payload['accounting_route_status']}`",
        f"- candidate: `{payload['candidate']['candidate_id']}`",
        f"- handoff ready: `{str(readiness['controlled_reality_handoff_ready']).lower()}`",
        "",
        "| prerequisite | ready |",
        "| --- | --- |",
    ]
    for key in (
        "candidate_has_action_gt",
        "dataset_root_present",
        "capture_identity_ready",
        "capture_prediction_ready",
        "capture_intervention_ready",
        "capture_intervention_action_gt_ready",
        "gaussian_evidence_declared",
        "candidate_artifact_present",
        "prediction_candidates_valid",
        "intervention_candidates_valid",
        "prediction_sample_id_match",
        "intervention_sample_id_match",
        "sample_ids_match",
        "identity_accounting_ready",
        "prediction_accounting_ready",
        "intervention_accounting_ready",
    ):
        lines.append(f"| {key} | {str(readiness[key]).lower()} |")
    lines.extend(("", "## Hard Blockers", ""))
    for blocker in payload["hard_blockers"]:
        lines.append(f"- {blocker}")
    lines.extend(("", "## Next Actions", ""))
    for action in payload["next_actions"]:
        lines.append(f"- {action}")
    lines.extend(("", "## Claim Boundary", ""))
    lines.append("- This audit does not create a reality row or pass evidence.")
    lines.append("- HOT3D-style observed interactions are action-like, not randomized counterfactual trials.")
    return "\n".join(lines) + "\n"


def validate_objectstate_public_dataset_candidates_audit(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("public dataset candidate audit must be a mapping")
    if payload.get("schema") != OBJECTSTATE_PUBLIC_DATASET_CANDIDATES_SCHEMA:
        raise ValueError(
            "unsupported public dataset candidate schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_public_dataset_candidate_audit":
        raise ValueError("public dataset candidate audit kind is unsupported")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("public dataset candidate audit requires candidates")
    if payload.get("candidate_count") != len(candidates):
        raise ValueError("candidate_count must match candidates")
    for candidate in candidates:
        _validate_candidate_payload(candidate)
    recommended_order = payload.get("recommended_order")
    if not isinstance(recommended_order, (list, tuple)) or not recommended_order:
        raise ValueError("public dataset candidate audit requires recommended_order")
    ids = {candidate["candidate_id"] for candidate in candidates}
    if any(candidate_id not in ids for candidate_id in recommended_order):
        raise ValueError("recommended_order contains an unknown candidate id")
    if payload.get("recommended_first") not in ids:
        raise ValueError("recommended_first must be a known candidate id")
    if payload.get("recommended_action_candidate") not in ids:
        raise ValueError("recommended_action_candidate must be a known candidate id")
    coverage_counts = payload.get("coverage_counts")
    if not isinstance(coverage_counts, Mapping):
        raise ValueError("public dataset candidate audit requires coverage_counts")
    for gate in OBJECTSTATE_PUBLIC_DATASET_GATE_KINDS:
        counts = coverage_counts.get(gate)
        if not isinstance(counts, Mapping):
            raise ValueError(f"coverage_counts missing gate {gate}")
        for value in OBJECTSTATE_PUBLIC_DATASET_COVERAGE_VALUES:
            if not isinstance(counts.get(value), int):
                raise ValueError(f"coverage_counts {gate}.{value} must be int")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("public dataset candidate audit requires readiness")
    for key in (
        "has_identity_pose_dataset",
        "has_action_like_dataset",
        "has_direct_gaussian_evidence",
        "has_direct_phase1_ready_dataset",
        "requires_controlled_capture_adapter",
        "requires_local_gaussian_reconstruction",
        "requires_license_review_before_public_release",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"readiness requires bool {key}")
    if readiness["has_direct_phase1_ready_dataset"]:
        raise ValueError("candidate audit cannot mark a direct Phase 1 ready dataset")
    if readiness["has_direct_gaussian_evidence"]:
        raise ValueError("default public candidates must not claim Gaussian evidence")
    hard_blockers = payload.get("hard_blockers")
    if not isinstance(hard_blockers, list) or not hard_blockers:
        raise ValueError("public dataset candidate audit requires hard_blockers")
    next_actions = payload.get("next_actions")
    if not isinstance(next_actions, (list, tuple)) or not next_actions:
        raise ValueError("public dataset candidate audit requires next_actions")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("candidate_audit_only")
        or not claim_policy.get("does_not_download_datasets")
        or not claim_policy.get("does_not_create_reality_rows")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
        or not claim_policy.get("blocked_rows_are_not_pass_rows")
        or not claim_policy.get("public_dataset_is_not_public_demo_asset")
    ):
        raise ValueError("public dataset candidate audit must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("downloads_datasets")
        or non_goals.get("writes_outputs")
        or non_goals.get("writes_public_samples")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError(
            "public dataset candidate audit cannot download, write outputs, "
            "train, replay, diffuse, or mutate viewer defaults"
        )
    return dict(payload)


def validate_objectstate_public_interaction_route_audit(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("public interaction route audit must be a mapping")
    if payload.get("schema") != OBJECTSTATE_PUBLIC_INTERACTION_ROUTE_AUDIT_SCHEMA:
        raise ValueError(
            "unsupported public interaction route audit schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_public_interaction_route_audit":
        raise ValueError("public interaction route audit kind is unsupported")
    if payload.get("status") not in {
        "objectstate_public_interaction_route_unsupported_candidate",
        "objectstate_public_interaction_route_blocked_no_local_dataset",
        "objectstate_public_interaction_route_capture_required",
        "objectstate_public_interaction_route_intervention_gt_required",
        "objectstate_public_interaction_route_gaussian_evidence_required",
        "objectstate_public_interaction_route_candidate_artifacts_required",
        "objectstate_public_interaction_route_sample_binding_blocked",
        "objectstate_public_interaction_route_handoff_ready",
    }:
        raise ValueError("public interaction route audit status is unsupported")
    if payload.get("accounting_route_status") not in {
        "handoff_ready",
        "identity_ready",
        "prediction_ready",
        "evidence_incomplete",
    }:
        raise ValueError("public interaction accounting route status is unsupported")
    _validate_candidate_payload(payload.get("candidate"))
    paths = payload.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("public interaction route audit requires paths")
    records = payload.get("file_records")
    if not isinstance(records, Mapping):
        raise ValueError("public interaction route audit requires file_records")
    for key in (
        "capture_manifest",
        "candidate_artifact",
        "prediction_candidates",
        "intervention_candidates",
    ):
        _validate_file_record(records.get(key), key)
    sample_id_binding = payload.get("sample_id_binding")
    if not isinstance(sample_id_binding, Mapping):
        raise ValueError("public interaction route audit requires sample_id_binding")
    if not isinstance(sample_id_binding.get("match"), bool):
        raise ValueError("sample_id_binding.match must be bool")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("public interaction route audit requires readiness")
    for key in (
        "candidate_is_public_interaction_dataset",
        "candidate_has_action_gt",
        "dataset_root_present",
        "capture_manifest_present",
        "capture_manifest_valid",
        "capture_identity_ready",
        "capture_prediction_ready",
        "capture_intervention_ready",
        "capture_intervention_action_gt_ready",
        "gaussian_evidence_declared",
        "candidate_artifact_present",
        "prediction_candidates_valid",
        "intervention_candidates_valid",
        "sample_ids_match",
        "prediction_sample_id_match",
        "intervention_sample_id_match",
        "controlled_reality_handoff_ready",
        "identity_accounting_ready",
        "prediction_accounting_ready",
        "intervention_accounting_ready",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"public interaction readiness requires bool {key}")
    expected_status = _public_interaction_route_status(readiness)
    if payload["status"] != expected_status:
        raise ValueError("public interaction route status must match readiness")
    expected_accounting_status = _public_interaction_accounting_route_status(readiness)
    if payload["accounting_route_status"] != expected_accounting_status:
        raise ValueError(
            "public interaction accounting route status must match readiness"
        )
    if not isinstance(payload.get("hard_blockers"), list):
        raise ValueError("public interaction route audit hard_blockers must be a list")
    next_actions = payload.get("next_actions")
    if not isinstance(next_actions, (list, tuple)) or not next_actions:
        raise ValueError("public interaction route audit requires next_actions")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("route_audit_only")
        or not claim_policy.get("does_not_download_datasets")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_create_reality_rows")
        or not claim_policy.get("does_not_claim_intervention_pass")
        or not claim_policy.get("does_not_claim_counterfactual_proof")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("public interaction route audit must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("downloads_datasets")
        or non_goals.get("writes_outputs")
        or non_goals.get("writes_public_samples")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("runs_intervention_model")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError(
            "public interaction route audit cannot download, write outputs, "
            "train, run intervention models, replay, diffuse, or mutate viewer defaults"
        )
    if readiness["controlled_reality_handoff_ready"] and payload["hard_blockers"]:
        raise ValueError("handoff-ready interaction route cannot have hard blockers")
    return dict(payload)


def validate_objectstate_public_dataset_candidate(
    candidate: ObjectStatePublicDatasetCandidate,
) -> ObjectStatePublicDatasetCandidate:
    if not isinstance(candidate, ObjectStatePublicDatasetCandidate):
        raise TypeError("candidate must be ObjectStatePublicDatasetCandidate")
    if candidate.schema != OBJECTSTATE_PUBLIC_DATASET_CANDIDATE_SCHEMA:
        raise ValueError(f"unsupported candidate schema: {candidate.schema}")
    for field_name in (
        "candidate_id",
        "name",
        "source_url",
        "source_kind",
        "source_license",
        "access",
        "download_size_note",
        "recommended_role",
        "object_category",
        "sequence_kind",
    ):
        value = getattr(candidate, field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"candidate {field_name} must be non-empty")
    if candidate.source_kind not in OBJECTSTATE_PUBLIC_DATASET_SOURCE_KINDS:
        raise ValueError(f"unsupported public dataset source kind: {candidate.source_kind}")
    if not candidate.source_url.startswith(("https://", "http://")):
        raise ValueError("candidate source_url must be http(s)")
    modalities = _non_empty_string_tuple(
        candidate.observation_modalities,
        "observation_modalities",
    )
    blockers = _non_empty_string_tuple(candidate.blockers, "blockers")
    next_actions = _non_empty_string_tuple(candidate.next_actions, "next_actions")
    source_notes = _non_empty_string_tuple(candidate.source_notes, "source_notes")
    coverage = _validate_gate_coverage(candidate.gate_coverage)
    if not candidate.has_identity_gt or not candidate.has_pose_gt:
        raise ValueError("public dataset candidates must include identity and pose GT")
    return ObjectStatePublicDatasetCandidate(
        candidate_id=candidate.candidate_id.strip(),
        name=candidate.name.strip(),
        source_url=candidate.source_url.strip(),
        source_kind=candidate.source_kind,
        source_license=candidate.source_license.strip(),
        access=candidate.access.strip(),
        download_size_note=candidate.download_size_note.strip(),
        recommended_role=candidate.recommended_role.strip(),
        object_category=candidate.object_category.strip(),
        sequence_kind=candidate.sequence_kind.strip(),
        observation_modalities=modalities,
        gate_coverage=coverage,
        blockers=blockers,
        next_actions=next_actions,
        source_notes=source_notes,
        has_identity_gt=bool(candidate.has_identity_gt),
        has_pose_gt=bool(candidate.has_pose_gt),
        has_action_gt=bool(candidate.has_action_gt),
        has_timestamp_or_frame_order=bool(candidate.has_timestamp_or_frame_order),
        has_camera_gt=bool(candidate.has_camera_gt),
        has_occlusion_or_visibility_signal=bool(
            candidate.has_occlusion_or_visibility_signal
        ),
        has_multiview_or_view_change=bool(candidate.has_multiview_or_view_change),
        has_object_models=bool(candidate.has_object_models),
        has_gaussian_evidence=bool(candidate.has_gaussian_evidence),
        schema=candidate.schema,
    )


def _validate_candidate_payload(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("candidate payload must be mapping")
    if payload.get("schema") != OBJECTSTATE_PUBLIC_DATASET_CANDIDATE_SCHEMA:
        raise ValueError("candidate payload schema is unsupported")
    for key in (
        "candidate_id",
        "name",
        "source_url",
        "source_kind",
        "source_license",
        "access",
        "download_size_note",
        "recommended_role",
        "object_category",
        "sequence_kind",
    ):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"candidate payload requires {key}")
    if payload["source_kind"] not in OBJECTSTATE_PUBLIC_DATASET_SOURCE_KINDS:
        raise ValueError("candidate payload source_kind is unsupported")
    if not isinstance(payload.get("observation_modalities"), list) or not payload[
        "observation_modalities"
    ]:
        raise ValueError("candidate payload requires observation_modalities")
    _validate_gate_coverage(payload.get("gate_coverage"))
    ground_truth = payload.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        raise ValueError("candidate payload requires ground_truth")
    for key in (
        "identity",
        "pose_6d",
        "action",
        "timestamp_or_frame_order",
        "camera",
        "occlusion_or_visibility",
        "multiview_or_view_change",
        "object_models",
        "gaussian_evidence",
    ):
        if not isinstance(ground_truth.get(key), bool):
            raise ValueError(f"candidate ground_truth requires bool {key}")
    for key in ("blockers", "next_actions", "source_notes"):
        if not isinstance(payload.get(key), list) or not payload[key]:
            raise ValueError(f"candidate payload requires {key}")


def _validate_gate_coverage(coverage: Mapping[str, str] | Any) -> dict[str, str]:
    if not isinstance(coverage, Mapping):
        raise ValueError("gate_coverage must be a mapping")
    result: dict[str, str] = {}
    for gate in OBJECTSTATE_PUBLIC_DATASET_GATE_KINDS:
        value = coverage.get(gate)
        if value not in OBJECTSTATE_PUBLIC_DATASET_COVERAGE_VALUES:
            raise ValueError(f"unsupported coverage for {gate}: {value}")
        result[gate] = str(value)
    extra = set(coverage) - set(OBJECTSTATE_PUBLIC_DATASET_GATE_KINDS)
    if extra:
        raise ValueError(f"unsupported gate coverage keys: {sorted(extra)}")
    return result


def _coverage_counts(
    candidates: Sequence[ObjectStatePublicDatasetCandidate],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        gate: {value: 0 for value in OBJECTSTATE_PUBLIC_DATASET_COVERAGE_VALUES}
        for gate in OBJECTSTATE_PUBLIC_DATASET_GATE_KINDS
    }
    for candidate in candidates:
        for gate, value in candidate.gate_coverage.items():
            counts[gate][value] += 1
    return counts


def _hard_blockers(
    candidates: Sequence[ObjectStatePublicDatasetCandidate],
) -> tuple[str, ...]:
    blockers = [
        "no public candidate directly supplies ObjGauss per-frame Gaussian evidence",
        "all public candidates require an adapter into the controlled capture manifest contract",
        "counterfactual rows remain blocked until action-conditioned outcomes are evaluated",
        "public dataset license terms must be reviewed before any public demo or redistribution claim",
    ]
    if not any(candidate.has_action_gt for candidate in candidates):
        blockers.append("no action-like public candidate was found")
    return tuple(blockers)


def _candidate_by_id(
    candidate_id: str,
    candidates: Sequence[ObjectStatePublicDatasetCandidate],
) -> ObjectStatePublicDatasetCandidate:
    normalized = str(candidate_id).strip()
    for candidate in candidates:
        if candidate.candidate_id == normalized:
            return candidate
    raise ValueError(f"unknown public dataset candidate id: {candidate_id}")


def _public_interaction_route_paths(
    dataset_root: Path | None,
    *,
    capture_manifest: str | Path | None,
    candidate_artifact: str | Path | None,
    prediction_candidates: str | Path | None,
    intervention_candidates: str | Path | None,
) -> dict[str, Path | None]:
    return {
        "capture_manifest": _resolve_route_path(
            dataset_root,
            capture_manifest,
            "capture-manifest.json",
        ),
        "candidate_artifact": _resolve_route_path(
            dataset_root,
            candidate_artifact,
            "objectstates.json",
        ),
        "prediction_candidates": _resolve_route_path(
            dataset_root,
            prediction_candidates,
            "reality-candidates/prediction-candidates.json",
        ),
        "intervention_candidates": _resolve_route_path(
            dataset_root,
            intervention_candidates,
            "reality-candidates/intervention-candidates.json",
        ),
    }


def _resolve_route_path(
    dataset_root: Path | None,
    explicit_path: str | Path | None,
    default_relative: str,
) -> Path | None:
    if explicit_path is not None:
        return Path(explicit_path)
    if dataset_root is None:
        return None
    return dataset_root / default_relative


def _capture_manifest_record(path: Path | None) -> dict[str, Any]:
    record = _plain_file_record(path)
    if not record["present"]:
        return {
            **record,
            "valid": False,
            "sample_id": None,
            "readiness": {},
            "intervention_action_gt": None,
            "error": record.get("error"),
        }
    try:
        manifest = read_objectstate_controlled_capture_manifest(path)  # type: ignore[arg-type]
        summary = objectstate_controlled_capture_summary(manifest)
    except Exception as exc:  # pragma: no cover - exact parser messages vary.
        return {
            **record,
            "valid": False,
            "sample_id": None,
            "readiness": {},
            "intervention_action_gt": None,
            "error": str(exc),
        }
    intervention_action_gt = (
        objectstate_controlled_capture_intervention_action_gt_readiness(manifest)
    )
    readiness = dict(summary["readiness"])
    readiness["intervention_action_gt_ready"] = bool(
        intervention_action_gt["ready"]
    )
    return {
        **record,
        "valid": True,
        "sample_id": manifest["sample"]["sample_id"],
        "frame_count": summary["frame_count"],
        "action_count": summary["action_count"],
        "readiness": readiness,
        "intervention_action_gt": intervention_action_gt,
        "error": None,
    }


def _candidate_json_record(path: Path | None, *, reader: Any) -> dict[str, Any]:
    record = _plain_file_record(path)
    if not record["present"]:
        return {**record, "valid": False, "sample_id": None, "error": record.get("error")}
    try:
        payload = reader(path)
    except Exception as exc:  # pragma: no cover - exact parser messages vary.
        return {**record, "valid": False, "sample_id": None, "error": str(exc)}
    return {
        **record,
        "valid": True,
        "sample_id": payload["sample_id"],
        "error": None,
    }


def _plain_file_record(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "present": False,
            "is_file": False,
            "size_bytes": 0,
            "error": "path not provided",
        }
    exists = path.exists()
    is_file = path.is_file()
    size = path.stat().st_size if is_file else 0
    error = None
    if not exists:
        error = "path does not exist"
    elif not is_file:
        error = "path is not a file"
    return {
        "path": str(path),
        "present": bool(is_file),
        "is_file": bool(is_file),
        "size_bytes": int(size),
        "error": error,
    }


def _sample_id_binding(
    *,
    capture_record: Mapping[str, Any],
    prediction_record: Mapping[str, Any],
    intervention_record: Mapping[str, Any],
) -> dict[str, Any]:
    sample_ids = {
        "capture_manifest": capture_record.get("sample_id"),
        "prediction_candidates": prediction_record.get("sample_id"),
        "intervention_candidates": intervention_record.get("sample_id"),
    }
    capture_id = sample_ids["capture_manifest"]
    prediction_id = sample_ids["prediction_candidates"]
    intervention_id = sample_ids["intervention_candidates"]
    prediction_match = bool(capture_id and prediction_id and capture_id == prediction_id)
    intervention_match = bool(
        capture_id and intervention_id and capture_id == intervention_id
    )
    present_ids = [value for value in sample_ids.values() if value]
    return {
        "sample_ids": sample_ids,
        "prediction_match": prediction_match,
        "intervention_match": intervention_match,
        "match": bool(present_ids) and len(set(present_ids)) == 1 and len(present_ids) == 3,
    }


def _public_interaction_route_status(readiness: Mapping[str, bool]) -> str:
    if (
        not readiness.get("candidate_is_public_interaction_dataset")
        or not readiness.get("candidate_has_action_gt")
    ):
        return "objectstate_public_interaction_route_unsupported_candidate"
    if not readiness.get("dataset_root_present"):
        return "objectstate_public_interaction_route_blocked_no_local_dataset"
    if not readiness.get("capture_manifest_valid"):
        return "objectstate_public_interaction_route_capture_required"
    if not readiness.get("capture_intervention_ready"):
        return "objectstate_public_interaction_route_intervention_gt_required"
    if not readiness.get("capture_intervention_action_gt_ready"):
        return "objectstate_public_interaction_route_intervention_gt_required"
    if not readiness.get("gaussian_evidence_declared"):
        return "objectstate_public_interaction_route_gaussian_evidence_required"
    if (
        not readiness.get("candidate_artifact_present")
        or not readiness.get("prediction_candidates_valid")
        or not readiness.get("intervention_candidates_valid")
    ):
        return "objectstate_public_interaction_route_candidate_artifacts_required"
    if not readiness.get("sample_ids_match"):
        return "objectstate_public_interaction_route_sample_binding_blocked"
    return "objectstate_public_interaction_route_handoff_ready"


def _public_interaction_accounting_route_status(
    readiness: Mapping[str, bool],
) -> str:
    if readiness.get("intervention_accounting_ready"):
        return "handoff_ready"
    if readiness.get("prediction_accounting_ready"):
        return "prediction_ready"
    if readiness.get("identity_accounting_ready"):
        return "identity_ready"
    return "evidence_incomplete"


def _public_interaction_route_blockers(
    readiness: Mapping[str, bool],
) -> list[str]:
    blockers: list[str] = []
    if not readiness["candidate_is_public_interaction_dataset"]:
        blockers.append("selected candidate is not a public interaction dataset")
    if not readiness["candidate_has_action_gt"]:
        blockers.append("selected candidate does not advertise action ground truth")
    if not readiness["dataset_root_present"]:
        blockers.append("local public interaction dataset root is missing")
    if not readiness["capture_manifest_valid"]:
        blockers.append("controlled capture manifest is missing or invalid")
    elif not readiness["capture_intervention_ready"]:
        blockers.append("capture manifest is not intervention-ready")
    elif not readiness["capture_intervention_action_gt_ready"]:
        blockers.append(
            "capture manifest lacks usable intervention action GT"
        )
    if not readiness["gaussian_evidence_declared"]:
        blockers.append("capture manifest does not declare per-frame Gaussian evidence")
    if not readiness["candidate_artifact_present"]:
        blockers.append("ObjectState candidate artifact is missing")
    if not readiness["prediction_candidates_valid"]:
        blockers.append("prediction candidates JSON is missing or invalid")
    if not readiness["intervention_candidates_valid"]:
        blockers.append("intervention candidates JSON is missing or invalid")
    if (
        readiness["prediction_candidates_valid"]
        and readiness["intervention_candidates_valid"]
        and not readiness["sample_ids_match"]
    ):
        blockers.append("capture/prediction/intervention sample_id values do not match")
    return blockers


def _public_interaction_route_next_actions(
    readiness: Mapping[str, bool],
    paths: Mapping[str, Path | None],
) -> list[str]:
    actions: list[str] = []
    if not readiness["dataset_root_present"]:
        actions.append(
            "place one license-reviewed public interaction clip outside git and rerun this route audit"
        )
    if not readiness["capture_manifest_valid"]:
        actions.append(
            "adapt the clip into the controlled capture manifest contract at "
            f"{_route_target(paths['capture_manifest'], 'capture-manifest.json')}"
        )
    elif not readiness["capture_intervention_ready"]:
        actions.append(
            "fill timestamped object pose tracks and action events until intervention_stage_ready=true"
        )
    elif not readiness["capture_intervention_action_gt_ready"]:
        actions.append(
            "fill action rows with non-zero vectors whose intervals cover "
            "object pose transitions until intervention_action_gt_ready=true"
        )
    if not readiness["gaussian_evidence_declared"]:
        actions.append(
            "write per-frame Gaussian evidence refs into the capture manifest before handoff"
        )
    if not readiness["candidate_artifact_present"]:
        actions.append(
            "write the ObjectState candidate artifact at "
            f"{_route_target(paths['candidate_artifact'], 'objectstates.json')}"
        )
    if not readiness["prediction_candidates_valid"]:
        actions.append(
            "write evaluator-ready prediction candidates at "
            f"{_route_target(paths['prediction_candidates'], 'reality-candidates/prediction-candidates.json')}"
        )
    if not readiness["intervention_candidates_valid"]:
        actions.append(
            "write evaluator-ready action-conditioned candidates at "
            f"{_route_target(paths['intervention_candidates'], 'reality-candidates/intervention-candidates.json')}"
        )
    if not readiness["sample_ids_match"]:
        actions.append("bind capture, prediction and intervention files to the same sample_id")
    actions.append(
        "run controlled-reality-bundle-handoff only after this audit reports handoff_ready"
    )
    actions.append(
        "keep counterfactual proof blocked unless observed actions have an explicit counterfactual evaluation design"
    )
    return actions


def _validate_file_record(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} file record must be a mapping")
    for key in ("present", "is_file"):
        if not isinstance(value.get(key), bool):
            raise ValueError(f"{name}.{key} must be bool")
    if not isinstance(value.get("size_bytes"), int):
        raise ValueError(f"{name}.size_bytes must be int")


def _path_string(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _route_target(path: Path | None, default_relative: str) -> str:
    if path is not None:
        return str(path)
    return f"<dataset_root>/{default_relative}"


def _non_empty_string_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence) or not values:
        raise ValueError(f"{name} must be a non-empty sequence")
    result = tuple(str(value).strip() for value in values)
    if any(not value for value in result):
        raise ValueError(f"{name} cannot contain empty values")
    return result
