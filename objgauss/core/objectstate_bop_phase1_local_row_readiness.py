from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_bop_identity_route_audit import (
    OBJECTSTATE_BOP_IDENTITY_ROUTE_AUDIT_SCHEMA,
    objectstate_bop_identity_route_audit,
    validate_objectstate_bop_identity_route_audit_summary,
)
from objgauss.core.objectstate_bop_phase1_route_audit import (
    OBJECTSTATE_BOP_PHASE1_ROUTE_AUDIT_SCHEMA,
    objectstate_bop_phase1_route_audit,
    validate_objectstate_bop_phase1_route_audit_summary,
)

OBJECTSTATE_BOP_PHASE1_LOCAL_ROW_READINESS_SCHEMA = (
    "objgauss-objectstate-bop-phase1-local-row-readiness-v1"
)


def objectstate_bop_phase1_local_row_readiness(
    scene_root: str | Path,
    *,
    output_root: str | Path,
    sample_id: str,
    candidate_artifact: str | Path | None = None,
    identity_dir: str | Path = "identity-handoff",
    dataset_id: str = "bop-ycbv",
    object_category: str = "bop_objects",
    scenario: str = "bop_pose_sequence",
    fps: float = 30.0,
    license_text: str = "BOP dataset terms; verify source dataset license before redistribution",
    rgb_dir: str = "rgb",
    gaussian_dir: str = "gaussians",
    max_frames: int | None = None,
    frame_step: int = 1,
    check_artifact_refs: bool = False,
    min_rgb_bytes: int = 1,
    min_gaussian_bytes: int = 1,
    require_frame_formats: bool = True,
    hash_files: bool = False,
    min_identity_scenario_frames: int = 3,
    min_occlusion_fraction: float = 0.5,
    min_view_conditions: int = 2,
    min_lighting_conditions: int = 2,
    min_camera_motion_m: float = 0.01,
) -> dict[str, Any]:
    scene = Path(scene_root)
    out = Path(output_root)
    identity = objectstate_bop_identity_route_audit(
        scene,
        output_root=out,
        sample_id=sample_id,
        candidate_artifact=candidate_artifact,
        identity_dir=identity_dir,
        dataset_id=dataset_id,
        object_category=object_category,
        scenario=scenario,
        fps=fps,
        license_text=license_text,
        rgb_dir=rgb_dir,
        gaussian_dir=gaussian_dir,
        max_frames=max_frames,
        frame_step=frame_step,
        check_artifact_refs=check_artifact_refs,
        min_rgb_bytes=min_rgb_bytes,
        min_gaussian_bytes=min_gaussian_bytes,
        require_frame_formats=require_frame_formats,
        hash_files=hash_files,
        min_identity_scenario_frames=min_identity_scenario_frames,
        min_occlusion_fraction=min_occlusion_fraction,
        min_view_conditions=min_view_conditions,
        min_lighting_conditions=min_lighting_conditions,
        min_camera_motion_m=min_camera_motion_m,
    )
    prediction = objectstate_bop_phase1_route_audit(
        scene,
        output_root=out,
        sample_id=sample_id,
        dataset_id=dataset_id,
        object_category=object_category,
        scenario=scenario,
        fps=fps,
        license_text=license_text,
        rgb_dir=rgb_dir,
        gaussian_dir=gaussian_dir,
        max_frames=max_frames,
        frame_step=frame_step,
        check_artifact_refs=check_artifact_refs,
        min_rgb_bytes=min_rgb_bytes,
        min_gaussian_bytes=min_gaussian_bytes,
        require_frame_formats=require_frame_formats,
        hash_files=hash_files,
    )
    readiness = _readiness(identity, prediction)
    status = _status(readiness)
    payload = {
        "schema": OBJECTSTATE_BOP_PHASE1_LOCAL_ROW_READINESS_SCHEMA,
        "kind": "objectstate_bop_phase1_local_row_readiness",
        "status": status,
        "blocking_stage": _blocking_stage(readiness),
        "scene_root": str(scene),
        "output_root": str(out),
        "sample_id": sample_id,
        "identity_route_schema": OBJECTSTATE_BOP_IDENTITY_ROUTE_AUDIT_SCHEMA,
        "prediction_route_schema": OBJECTSTATE_BOP_PHASE1_ROUTE_AUDIT_SCHEMA,
        "routes": {
            "identity": identity,
            "prediction": prediction,
        },
        "readiness": readiness,
        "files": _files(identity, prediction),
        "hard_blockers": _hard_blockers(identity, prediction, readiness),
        "next_actions": _next_actions(identity, prediction, readiness),
        "issues": _issues(identity, prediction, readiness),
        "claim_policy": {
            "read_only_local_row_audit": True,
            "runs_identity_route_audit": True,
            "runs_prediction_route_audit": True,
            "requires_bop_acceptance": True,
            "requires_phase1_gaussian_evidence": True,
            "checks_candidate_artifact_route": True,
            "checks_existing_evidence_packages": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_run_identity_handoff": True,
            "does_not_run_prediction_handoff": True,
            "does_not_run_identity_eval": True,
            "does_not_run_prediction_eval": True,
            "does_not_train_model": True,
            "does_not_relax_identity_gate": True,
            "does_not_claim_intervention_gate": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "creates_ground_truth": False,
            "reconstructs_gaussians": False,
            "runs_identity_handoff": False,
            "runs_prediction_handoff": False,
            "runs_identity_eval": False,
            "runs_prediction_eval": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_bop_phase1_local_row_readiness_summary(payload)


def validate_objectstate_bop_phase1_local_row_readiness_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP Phase 1 local row readiness summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_PHASE1_LOCAL_ROW_READINESS_SCHEMA:
        raise ValueError(
            "unsupported BOP Phase 1 local row readiness schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_phase1_local_row_readiness":
        raise ValueError("BOP Phase 1 local row readiness kind is unsupported")
    routes = payload.get("routes")
    if not isinstance(routes, Mapping):
        raise ValueError("BOP Phase 1 local row readiness requires routes")
    identity = validate_objectstate_bop_identity_route_audit_summary(
        routes.get("identity")
    )
    prediction = validate_objectstate_bop_phase1_route_audit_summary(
        routes.get("prediction")
    )
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping) or not readiness:
        raise ValueError("BOP Phase 1 local row readiness requires readiness")
    if any(not isinstance(value, bool) for value in readiness.values()):
        raise ValueError("BOP Phase 1 local row readiness readiness values must be bool")
    if dict(readiness) != _readiness(identity, prediction):
        raise ValueError("BOP Phase 1 local row readiness gates mismatch routes")
    if payload.get("status") != _status(readiness):
        raise ValueError("BOP Phase 1 local row readiness status mismatch")
    if payload.get("blocking_stage") != _blocking_stage(readiness):
        raise ValueError("BOP Phase 1 local row readiness blocking_stage mismatch")
    for key in ("scene_root", "output_root", "sample_id"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"BOP Phase 1 local row readiness requires {key}")
    if payload.get("identity_route_schema") != OBJECTSTATE_BOP_IDENTITY_ROUTE_AUDIT_SCHEMA:
        raise ValueError("BOP Phase 1 local row readiness identity schema mismatch")
    if payload.get("prediction_route_schema") != OBJECTSTATE_BOP_PHASE1_ROUTE_AUDIT_SCHEMA:
        raise ValueError("BOP Phase 1 local row readiness prediction schema mismatch")
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("BOP Phase 1 local row readiness requires files")
    for key in (
        "candidate_artifact",
        "identity_evidence_package_summary",
        "prediction_evidence_package_summary",
        "phase1_evidence_ledger",
    ):
        if not isinstance(files.get(key), str) or not files[key]:
            raise ValueError(f"BOP Phase 1 local row readiness missing file {key}")
    for key in ("hard_blockers", "next_actions", "issues"):
        values = payload.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"BOP Phase 1 local row readiness {key} must be strings")
    claim_policy = payload.get("claim_policy")
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("read_only_local_row_audit")
        or not claim_policy.get("runs_identity_route_audit")
        or not claim_policy.get("runs_prediction_route_audit")
        or not claim_policy.get("requires_bop_acceptance")
        or not claim_policy.get("requires_phase1_gaussian_evidence")
        or not claim_policy.get("checks_candidate_artifact_route")
        or not claim_policy.get("checks_existing_evidence_packages")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_run_identity_handoff")
        or not claim_policy.get("does_not_run_prediction_handoff")
        or not claim_policy.get("does_not_run_identity_eval")
        or not claim_policy.get("does_not_run_prediction_eval")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_relax_identity_gate")
        or not claim_policy.get("does_not_claim_intervention_gate")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP Phase 1 local row readiness must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP Phase 1 local row readiness cannot claim downloads, GT, "
            "reconstruction, handoff, eval, training, public samples, replay, "
            "diffusion, or viewer mutation"
        )
    return dict(payload)


def _readiness(
    identity: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, bool]:
    identity_ready = identity["readiness"]
    prediction_ready = prediction["readiness"]
    return {
        "bop_acceptance_available": bool(
            identity_ready["bop_acceptance_available"]
            and prediction_ready["bop_acceptance_available"]
        ),
        "bop_acceptance_pass": bool(
            identity_ready["bop_acceptance_pass"]
            and prediction_ready["bop_acceptance_pass"]
        ),
        "phase1_gaussian_evidence_ready": bool(
            identity_ready["phase1_gaussian_evidence_ready"]
            and prediction_ready["phase1_gaussian_evidence_ready"]
        ),
        "candidate_artifact_present": bool(
            identity_ready["candidate_artifact_present"]
        ),
        "candidate_artifact_valid": bool(identity_ready["candidate_artifact_valid"]),
        "candidate_artifact_binding_ready": bool(
            identity_ready["candidate_artifact_binding_ready"]
        ),
        "identity_scenario_metadata_ready": bool(
            identity_ready["identity_scenario_metadata_ready"]
        ),
        "identity_route_ready_for_handoff": bool(
            identity_ready["route_ready_for_identity_handoff"]
        ),
        "prediction_route_ready_for_handoff": bool(
            prediction_ready["route_ready_for_prediction_handoff"]
        ),
        "identity_evidence_reviewable": bool(
            identity_ready["route_has_reviewable_identity_evidence"]
        ),
        "prediction_evidence_reviewable": bool(
            prediction_ready["route_has_reviewable_prediction_evidence"]
        ),
        "phase1_has_any_reviewable_evidence": bool(
            identity_ready["route_has_reviewable_identity_evidence"]
            or prediction_ready["route_has_reviewable_prediction_evidence"]
        ),
        "phase1_identity_prediction_reviewable": bool(
            identity_ready["route_has_reviewable_identity_evidence"]
            and prediction_ready["route_has_reviewable_prediction_evidence"]
        ),
        "phase1_ready_for_any_handoff": bool(
            identity_ready["route_ready_for_identity_handoff"]
            or prediction_ready["route_ready_for_prediction_handoff"]
        ),
    }


def _status(readiness: Mapping[str, bool]) -> str:
    prefix = "objectstate_bop_phase1_local_row_readiness"
    if readiness["phase1_identity_prediction_reviewable"]:
        return f"{prefix}_identity_prediction_reviewable"
    if readiness["prediction_evidence_reviewable"]:
        return f"{prefix}_prediction_reviewable"
    if readiness["identity_evidence_reviewable"]:
        return f"{prefix}_identity_reviewable"
    if not readiness["bop_acceptance_pass"] or not readiness["phase1_gaussian_evidence_ready"]:
        return f"{prefix}_blocked"
    if not readiness["candidate_artifact_binding_ready"]:
        return f"{prefix}_candidate_required"
    if not readiness["identity_scenario_metadata_ready"]:
        return f"{prefix}_identity_scenario_blocked"
    if (
        readiness["identity_route_ready_for_handoff"]
        and readiness["prediction_route_ready_for_handoff"]
    ):
        return f"{prefix}_identity_prediction_handoff_ready"
    if readiness["prediction_route_ready_for_handoff"]:
        return f"{prefix}_prediction_handoff_ready"
    return f"{prefix}_blocked"


def _blocking_stage(readiness: Mapping[str, bool]) -> str:
    if readiness["phase1_identity_prediction_reviewable"]:
        return "identity_prediction_reviewable"
    if readiness["prediction_evidence_reviewable"]:
        return "prediction_reviewable_identity_pending"
    if readiness["identity_evidence_reviewable"]:
        return "identity_reviewable_prediction_pending"
    if not readiness["bop_acceptance_available"]:
        return "local_bop_scene"
    if not readiness["bop_acceptance_pass"]:
        return "bop_acceptance"
    if not readiness["phase1_gaussian_evidence_ready"]:
        return "phase1_gaussian_evidence"
    if not readiness["candidate_artifact_binding_ready"]:
        return "candidate_artifact"
    if not readiness["identity_scenario_metadata_ready"]:
        return "identity_scenario_metadata"
    if readiness["phase1_ready_for_any_handoff"]:
        return "handoff_ready"
    return "unknown"


def _files(
    identity: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, str]:
    identity_files = identity["files"]
    prediction_files = prediction["files"]
    return {
        "candidate_artifact": identity_files["candidate_artifact"],
        "identity_evidence_package_summary": identity_files[
            "identity_evidence_package_summary"
        ],
        "prediction_evidence_package_summary": prediction_files[
            "prediction_evidence_package_summary"
        ],
        "phase1_evidence_ledger": prediction_files["phase1_evidence_ledger"],
    }


def _hard_blockers(
    identity: Mapping[str, Any],
    prediction: Mapping[str, Any],
    readiness: Mapping[str, bool],
) -> list[str]:
    blockers = []
    if readiness["phase1_has_any_reviewable_evidence"]:
        if not readiness["identity_evidence_reviewable"]:
            blockers.append("identity evidence remains missing or not reviewable")
        if not readiness["prediction_evidence_reviewable"]:
            blockers.append("prediction evidence remains missing or not reviewable")
        blockers.append("intervention and counterfactual gates remain separate")
        return _dedupe(blockers)
    blockers.extend(str(item) for item in identity["hard_blockers"])
    blockers.extend(str(item) for item in prediction["hard_blockers"])
    if readiness["prediction_route_ready_for_handoff"] and not readiness[
        "identity_route_ready_for_handoff"
    ]:
        blockers.append(
            "prediction route is handoff-ready before identity route; Stage 1 identity evidence still needs the candidate/scenario gate"
        )
    return _dedupe(blockers)


def _next_actions(
    identity: Mapping[str, Any],
    prediction: Mapping[str, Any],
    readiness: Mapping[str, bool],
) -> list[str]:
    if readiness["phase1_identity_prediction_reviewable"]:
        return [
            "review identity and prediction rows in phase1-evidence-ledger.json",
            "keep intervention/counterfactual evidence on a separate action-capable route",
        ]
    if readiness["prediction_evidence_reviewable"]:
        return [
            "prediction evidence is reviewable; resolve identity route blockers next",
            *_as_strings(identity["next_actions"]),
        ]
    if readiness["identity_evidence_reviewable"]:
        return [
            "identity evidence is reviewable; run the BOP prediction baseline handoff next",
            *_as_strings(prediction["next_actions"]),
        ]
    if not readiness["bop_acceptance_pass"] or not readiness["phase1_gaussian_evidence_ready"]:
        return _dedupe(prediction["next_actions"])
    if not readiness["candidate_artifact_binding_ready"]:
        return _dedupe(identity["next_actions"])
    if not readiness["identity_scenario_metadata_ready"]:
        return _dedupe(identity["next_actions"])
    actions = []
    if readiness["identity_route_ready_for_handoff"]:
        actions.extend(_as_strings(identity["next_actions"]))
    if readiness["prediction_route_ready_for_handoff"]:
        actions.extend(_as_strings(prediction["next_actions"]))
    return _dedupe(actions)


def _issues(
    identity: Mapping[str, Any],
    prediction: Mapping[str, Any],
    readiness: Mapping[str, bool],
) -> list[str]:
    issues = []
    issues.extend(f"identity:{item}" for item in identity["issues"])
    issues.extend(f"prediction:{item}" for item in prediction["issues"])
    for gate, passed in readiness.items():
        if not passed:
            issues.append(f"readiness gate failed: {gate}")
    return _dedupe(issues)


def _as_strings(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return [str(value) for value in values]


def _dedupe(values: Any) -> list[str]:
    result = []
    seen: set[str] = set()
    for value in _as_strings(values):
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
