from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from objgauss.core.objectstate_bop_capture_adapter import (
    OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
    objectstate_bop_capture_acceptance_summary,
    validate_objectstate_bop_capture_acceptance_summary,
)
from objgauss.core.objectstate_controlled_identity_evidence_package import (
    OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA,
    validate_objectstate_controlled_identity_evidence_package_summary,
)
from objgauss.core.objectstate_phase1_evidence_ledger import (
    OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA,
    validate_objectstate_phase1_evidence_ledger_summary,
)
from objgauss.core.trainable_artifact import validate_trainable_kernel_model_artifact

OBJECTSTATE_BOP_IDENTITY_ROUTE_AUDIT_SCHEMA = (
    "objgauss-objectstate-bop-identity-route-audit-v1"
)

Validator = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def objectstate_bop_identity_route_audit(
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
    condition_sidecar: str | Path | None = None,
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
    identity_root = _resolve_identity_dir(out, identity_dir)
    candidate_path = Path(candidate_artifact) if candidate_artifact else out / "objectstates.json"
    acceptance, acceptance_issue = _acceptance_or_issue(
        scene,
        sample_id=sample_id,
        dataset_id=dataset_id,
        object_category=object_category,
        scenario=scenario,
        fps=fps,
        license_text=license_text,
        rgb_dir=rgb_dir,
        gaussian_dir=gaussian_dir,
        condition_sidecar=condition_sidecar,
        max_frames=max_frames,
        frame_step=frame_step,
        check_artifact_refs=check_artifact_refs,
        min_rgb_bytes=min_rgb_bytes,
        min_gaussian_bytes=min_gaussian_bytes,
        require_frame_formats=require_frame_formats,
        hash_files=hash_files,
    )
    scenario_audit = _identity_scenario_metadata_audit(
        acceptance["manifest"] if acceptance else None,
        min_frames=min_identity_scenario_frames,
        min_occlusion_fraction=min_occlusion_fraction,
        min_view_conditions=min_view_conditions,
        min_lighting_conditions=min_lighting_conditions,
        min_camera_motion_m=min_camera_motion_m,
    )
    candidate_record = _candidate_artifact_record(
        candidate_path,
        acceptance["manifest"] if acceptance else None,
    )
    identity_record = _summary_record(
        identity_root / "identity-evidence-package-summary.json",
        expected_schema=OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA,
        validator=validate_objectstate_controlled_identity_evidence_package_summary,
    )
    phase1_record = _summary_record(
        out / "phase1-evidence-ledger.json",
        expected_schema=OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA,
        validator=validate_objectstate_phase1_evidence_ledger_summary,
    )
    readiness = _readiness(
        acceptance,
        scenario_audit=scenario_audit,
        candidate_record=candidate_record,
        identity_record=identity_record,
        phase1_record=phase1_record,
    )
    status = _status(readiness)
    payload = {
        "schema": OBJECTSTATE_BOP_IDENTITY_ROUTE_AUDIT_SCHEMA,
        "kind": "objectstate_bop_identity_route_audit",
        "status": status,
        "scene_root": str(scene),
        "output_root": str(out),
        "identity_dir": str(identity_root),
        "sample_id": sample_id,
        "acceptance_schema": OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
        "identity_evidence_package_schema": (
            OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA
        ),
        "phase1_evidence_ledger_schema": OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA,
        "files": {
            "candidate_artifact": str(candidate_path),
            "identity_evidence_package_summary": str(
                identity_root / "identity-evidence-package-summary.json"
            ),
            "phase1_evidence_ledger": str(out / "phase1-evidence-ledger.json"),
        },
        "readiness": readiness,
        "records": {
            "candidate_artifact": candidate_record,
            "identity_evidence_package": identity_record,
            "phase1_evidence_ledger": phase1_record,
        },
        "identity_scenario_metadata_audit": scenario_audit,
        "acceptance": acceptance,
        "issues": _issues(
            acceptance_issue=acceptance_issue,
            scenario_audit=scenario_audit,
            candidate_record=candidate_record,
            identity_record=identity_record,
            phase1_record=phase1_record,
            readiness=readiness,
        ),
        "hard_blockers": _hard_blockers(
            acceptance,
            acceptance_issue=acceptance_issue,
            scenario_audit=scenario_audit,
            candidate_record=candidate_record,
            identity_record=identity_record,
            phase1_record=phase1_record,
            readiness=readiness,
        ),
        "next_actions": _next_actions(
            acceptance,
            scene,
            out,
            candidate_path,
            sample_id=sample_id,
            dataset_id=dataset_id,
            scenario_audit=scenario_audit,
            candidate_record=candidate_record,
            identity_record=identity_record,
            phase1_record=phase1_record,
            readiness=readiness,
        ),
        "claim_policy": {
            "read_only_route_audit": True,
            "runs_bop_acceptance_audit": True,
            "checks_candidate_artifact_file": True,
            "checks_identity_scenario_metadata": True,
            "checks_existing_identity_evidence_package": True,
            "checks_existing_phase1_evidence_ledger": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_run_identity_handoff": True,
            "does_not_run_identity_eval": True,
            "does_not_train_model": True,
            "does_not_relax_identity_gate": True,
            "does_not_claim_prediction_gate": True,
            "does_not_claim_intervention_gate": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "creates_ground_truth": False,
            "reconstructs_gaussians": False,
            "runs_identity_handoff": False,
            "runs_identity_eval": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_bop_identity_route_audit_summary(payload)


def validate_objectstate_bop_identity_route_audit_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP identity route audit summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_IDENTITY_ROUTE_AUDIT_SCHEMA:
        raise ValueError(
            f"unsupported BOP identity route audit schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_identity_route_audit":
        raise ValueError("BOP identity route audit kind is unsupported")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping) or not readiness:
        raise ValueError("BOP identity route audit requires readiness")
    if any(not isinstance(value, bool) for value in readiness.values()):
        raise ValueError("BOP identity route audit readiness values must be bool")
    if payload.get("status") != _status(readiness):
        raise ValueError("BOP identity route audit status mismatch")
    for key in ("scene_root", "output_root", "identity_dir", "sample_id"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"BOP identity route audit requires {key}")
    if payload.get("acceptance_schema") != OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA:
        raise ValueError("BOP identity route audit acceptance_schema mismatch")
    if (
        payload.get("identity_evidence_package_schema")
        != OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA
    ):
        raise ValueError("BOP identity route audit identity evidence schema mismatch")
    if payload.get("phase1_evidence_ledger_schema") != OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA:
        raise ValueError("BOP identity route audit ledger schema mismatch")
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("BOP identity route audit requires files")
    for key in (
        "candidate_artifact",
        "identity_evidence_package_summary",
        "phase1_evidence_ledger",
    ):
        if not isinstance(files.get(key), str) or not files[key]:
            raise ValueError(f"BOP identity route audit missing file path {key}")
    records = payload.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("BOP identity route audit requires records")
    _validate_record(records.get("candidate_artifact"), allow_payload=False)
    _validate_record(records.get("identity_evidence_package"), allow_payload=True)
    _validate_record(records.get("phase1_evidence_ledger"), allow_payload=True)
    _validate_scenario_audit(payload.get("identity_scenario_metadata_audit"))
    acceptance = payload.get("acceptance")
    if acceptance is not None:
        validate_objectstate_bop_capture_acceptance_summary(acceptance)
    for key in ("issues", "hard_blockers", "next_actions"):
        values = payload.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"BOP identity route audit {key} must be string list")
    claim_policy = payload.get("claim_policy")
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("read_only_route_audit")
        or not claim_policy.get("runs_bop_acceptance_audit")
        or not claim_policy.get("checks_candidate_artifact_file")
        or not claim_policy.get("checks_identity_scenario_metadata")
        or not claim_policy.get("checks_existing_identity_evidence_package")
        or not claim_policy.get("checks_existing_phase1_evidence_ledger")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_run_identity_handoff")
        or not claim_policy.get("does_not_run_identity_eval")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_relax_identity_gate")
        or not claim_policy.get("does_not_claim_prediction_gate")
        or not claim_policy.get("does_not_claim_intervention_gate")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP identity route audit must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP identity route audit cannot claim downloads, GT, reconstruction, "
            "handoff, eval, training, public samples, replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def _acceptance_or_issue(
    scene: Path,
    **kwargs: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        acceptance = objectstate_bop_capture_acceptance_summary(
            scene,
            require_gaussian_files=True,
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - route audit reports blocked inputs.
        return None, f"BOP acceptance audit failed: {exc}"
    return acceptance, None


def _candidate_artifact_record(
    path: Path,
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    issues: list[str] = []
    exists = path.exists()
    is_file = exists and path.is_file()
    schema = None
    schema_ok = False
    validator_ok = False
    binding_ok = False
    status = None
    if not is_file:
        issues.append("candidate artifact file is missing")
    else:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - route audit reports malformed inputs.
            issues.append(f"invalid JSON: {exc}")
        else:
            if not isinstance(payload, Mapping):
                issues.append("candidate artifact JSON must be an object")
            else:
                schema = payload.get("schema")
                try:
                    validate_trainable_kernel_model_artifact(payload)
                except Exception as exc:  # noqa: BLE001 - keep audit tolerant.
                    issues.append(f"validator failed: {exc}")
                else:
                    validator_ok = True
                    schema_ok = True
                    status = payload.get("status")
                    binding_ok = _artifact_binding_ok(payload, manifest, issues)
    return {
        "path": str(path),
        "exists": bool(exists),
        "is_file": bool(is_file),
        "size_bytes": int(path.stat().st_size) if is_file else 0,
        "schema": schema,
        "expected_schema": "objgauss-trainable-kernel-model-artifact-v1",
        "schema_ok": schema_ok,
        "validator_ok": validator_ok,
        "binding_ok": binding_ok,
        "status": status,
        "payload": None,
        "issues": issues,
    }


def _artifact_binding_ok(
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    issues: list[str],
) -> bool:
    if manifest is None:
        issues.append("cannot check artifact frame binding without accepted BOP manifest")
        return False
    object_states = payload.get("object_states")
    frames = manifest.get("frames") if isinstance(manifest, Mapping) else None
    if not isinstance(object_states, Sequence) or isinstance(object_states, (str, bytes)):
        issues.append("candidate artifact object_states must be a sequence")
        return False
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        issues.append("accepted BOP manifest frames must be a sequence")
        return False
    if len(object_states) != len(frames):
        issues.append(
            "candidate artifact object_states frame count does not match BOP capture frames"
        )
        return False
    for index, frame in enumerate(object_states):
        if not isinstance(frame, Mapping) or int(frame.get("frame_index", index)) != index:
            issues.append("candidate artifact object_states frame_index mismatch")
            return False
    return True


def _summary_record(
    path: Path,
    *,
    expected_schema: str,
    validator: Validator,
) -> dict[str, Any]:
    issues: list[str] = []
    exists = path.exists()
    is_file = exists and path.is_file()
    schema = None
    status = None
    schema_ok = False
    validator_ok = False
    payload: Mapping[str, Any] | None = None
    if not is_file:
        issues.append("summary file is missing")
    else:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - route audit reports malformed inputs.
            issues.append(f"invalid JSON: {exc}")
        else:
            if not isinstance(loaded, Mapping):
                issues.append("summary JSON must be an object")
            else:
                payload = loaded
                schema = loaded.get("schema")
                schema_ok = schema == expected_schema
                if not schema_ok:
                    issues.append(
                        f"schema mismatch: expected {expected_schema}, got {schema}"
                    )
                try:
                    checked = validator(loaded)
                except Exception as exc:  # noqa: BLE001 - keep audit tolerant.
                    issues.append(f"validator failed: {exc}")
                else:
                    validator_ok = True
                    status = checked.get("status")
                    payload = checked
    return {
        "path": str(path),
        "exists": bool(exists),
        "is_file": bool(is_file),
        "size_bytes": int(path.stat().st_size) if is_file else 0,
        "schema": schema,
        "expected_schema": expected_schema,
        "schema_ok": schema_ok,
        "validator_ok": validator_ok,
        "binding_ok": None,
        "status": status,
        "payload": dict(payload) if payload is not None else None,
        "issues": issues,
    }


def _identity_scenario_metadata_audit(
    manifest: Mapping[str, Any] | None,
    *,
    min_frames: int,
    min_occlusion_fraction: float,
    min_view_conditions: int,
    min_lighting_conditions: int,
    min_camera_motion_m: float,
) -> dict[str, Any]:
    if manifest is None:
        readiness = {
            "min_frame_count_met": False,
            "occlusion_reappearance_present": False,
            "min_view_conditions_met": False,
            "min_lighting_conditions_met": False,
            "camera_motion_present": False,
        }
        return {
            "status": "objectstate_bop_identity_route_scenario_metadata_blocked",
            "readiness": readiness,
            "requirements": _scenario_requirements(
                min_frames,
                min_occlusion_fraction,
                min_view_conditions,
                min_lighting_conditions,
                min_camera_motion_m,
            ),
            "scenario_coverage": _empty_scenario_coverage(),
            "issues": ["cannot audit identity scenario metadata without accepted BOP manifest"],
        }
    frames = tuple(manifest["frames"])
    coverage = _scenario_coverage(frames)
    occlusion_reappearance = _occlusion_reappearance_present(
        frames,
        min_occlusion_fraction=min_occlusion_fraction,
    )
    readiness = {
        "min_frame_count_met": len(frames) >= min_frames,
        "occlusion_reappearance_present": occlusion_reappearance,
        "min_view_conditions_met": coverage["view_condition_count"] >= min_view_conditions,
        "min_lighting_conditions_met": (
            coverage["lighting_condition_count"] >= min_lighting_conditions
        ),
        "camera_motion_present": (
            coverage["camera_pose_count"] >= 2
            and coverage["max_camera_translation_m"] >= min_camera_motion_m
        ),
    }
    issues = []
    if not readiness["min_frame_count_met"]:
        issues.append(f"identity route requires at least {min_frames} frames")
    if not readiness["occlusion_reappearance_present"]:
        issues.append("identity route requires clear-visible/occluded/reappeared metadata")
    if not readiness["min_lighting_conditions_met"]:
        issues.append(
            f"identity route requires at least {min_lighting_conditions} lighting conditions"
        )
    if not readiness["camera_motion_present"]:
        issues.append(
            "identity route requires camera_pose metadata with sufficient translation"
        )
    return {
        "status": (
            "objectstate_bop_identity_route_scenario_metadata_ready"
            if all(readiness.values())
            else "objectstate_bop_identity_route_scenario_metadata_blocked"
        ),
        "readiness": readiness,
        "requirements": _scenario_requirements(
            min_frames,
            min_occlusion_fraction,
            min_view_conditions,
            min_lighting_conditions,
            min_camera_motion_m,
        ),
        "scenario_coverage": coverage,
        "issues": issues,
    }


def _readiness(
    acceptance: Mapping[str, Any] | None,
    *,
    scenario_audit: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    identity_record: Mapping[str, Any],
    phase1_record: Mapping[str, Any],
) -> dict[str, bool]:
    acceptance_pass = bool(
        acceptance
        and acceptance["status"] == "objectstate_bop_capture_acceptance_pass"
    )
    gaussian_ready = bool(
        acceptance and acceptance["readiness"]["phase1_gaussian_evidence_ready"]
    )
    identity_reviewable = bool(
        identity_record["validator_ok"]
        and str(identity_record.get("status", "")).endswith("_reviewable")
    )
    phase1_payload = phase1_record.get("payload")
    phase1_identity_reviewable = bool(
        phase1_record["validator_ok"]
        and isinstance(phase1_payload, Mapping)
        and phase1_payload["phase1_evidence_gates"]["identity_evidence_reviewable"]
    )
    candidate_ready = bool(
        candidate_record["validator_ok"] and candidate_record["binding_ok"]
    )
    scenario_ready = bool(
        scenario_audit["status"]
        == "objectstate_bop_identity_route_scenario_metadata_ready"
    )
    return {
        "bop_acceptance_available": acceptance is not None,
        "bop_acceptance_pass": acceptance_pass,
        "phase1_gaussian_evidence_ready": gaussian_ready,
        "candidate_artifact_present": bool(candidate_record["is_file"]),
        "candidate_artifact_valid": bool(candidate_record["validator_ok"]),
        "candidate_artifact_binding_ready": bool(candidate_record["binding_ok"]),
        "identity_scenario_metadata_ready": scenario_ready,
        "identity_evidence_package_present": bool(identity_record["is_file"]),
        "identity_evidence_package_reviewable": identity_reviewable,
        "phase1_evidence_ledger_present": bool(phase1_record["is_file"]),
        "phase1_evidence_ledger_identity_reviewable": phase1_identity_reviewable,
        "route_ready_for_identity_handoff": bool(
            acceptance_pass and gaussian_ready and candidate_ready and scenario_ready
        ),
        "route_has_reviewable_identity_evidence": bool(
            identity_reviewable and phase1_identity_reviewable
        ),
    }


def _status(readiness: Mapping[str, bool]) -> str:
    if readiness["route_has_reviewable_identity_evidence"]:
        return "objectstate_bop_identity_route_audit_identity_reviewable"
    if readiness["route_ready_for_identity_handoff"]:
        return "objectstate_bop_identity_route_audit_handoff_ready"
    return "objectstate_bop_identity_route_audit_blocked"


def _issues(
    *,
    acceptance_issue: str | None,
    scenario_audit: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    identity_record: Mapping[str, Any],
    phase1_record: Mapping[str, Any],
    readiness: Mapping[str, bool],
) -> list[str]:
    issues = []
    if acceptance_issue:
        issues.append(acceptance_issue)
    for issue in scenario_audit["issues"]:
        issues.append(f"identity_scenario_metadata: {issue}")
    for label, record in (
        ("candidate_artifact", candidate_record),
        ("identity_evidence_package", identity_record),
        ("phase1_evidence_ledger", phase1_record),
    ):
        for issue in record["issues"]:
            issues.append(f"{label}:{record['path']}: {issue}")
    for gate, passed in readiness.items():
        if not passed:
            issues.append(f"readiness gate failed: {gate}")
    return issues


def _hard_blockers(
    acceptance: Mapping[str, Any] | None,
    *,
    acceptance_issue: str | None,
    scenario_audit: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    identity_record: Mapping[str, Any],
    phase1_record: Mapping[str, Any],
    readiness: Mapping[str, bool],
) -> list[str]:
    blockers = []
    if readiness["route_has_reviewable_identity_evidence"]:
        return [
            "BOP identity route only covers identity evidence; prediction and intervention gates remain separate"
        ]
    if acceptance_issue:
        blockers.append("local BOP scene cannot be accepted yet")
    if acceptance:
        blockers.extend(str(blocker) for blocker in acceptance["hard_blockers"])
    if not readiness["candidate_artifact_binding_ready"]:
        blockers.append("candidate ObjectState artifact is missing, invalid, or not frame-bound")
    if not readiness["identity_scenario_metadata_ready"]:
        blockers.append("BOP identity route lacks required scenario challenge metadata")
    if readiness["route_ready_for_identity_handoff"]:
        if not identity_record["validator_ok"]:
            blockers.append("identity evidence package is not reviewable yet")
        if not phase1_record["validator_ok"]:
            blockers.append("Phase 1 evidence ledger is not reviewable yet")
    return blockers


def _next_actions(
    acceptance: Mapping[str, Any] | None,
    scene: Path,
    out: Path,
    candidate_path: Path,
    *,
    sample_id: str,
    dataset_id: str,
    scenario_audit: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    identity_record: Mapping[str, Any],
    phase1_record: Mapping[str, Any],
    readiness: Mapping[str, bool],
) -> list[str]:
    if readiness["route_has_reviewable_identity_evidence"]:
        return [
            "review identity row status in phase1-evidence-ledger.json",
            f"keep prediction evidence on the separate BOP prediction route for dataset {dataset_id}",
        ]
    if not acceptance:
        return [
            f"place a local BOP scene under {scene} with scene_camera.json, scene_gt.json, and rgb/",
            "rerun audit-bop-identity-route after the local scene files exist",
        ]
    if not readiness["phase1_gaussian_evidence_ready"]:
        return list(acceptance["next_actions"])
    if not candidate_record["validator_ok"]:
        return [
            f"create or export a trainable ObjectState artifact at {candidate_path}",
            "rerun audit-bop-identity-route after the candidate artifact exists",
        ]
    if not candidate_record["binding_ok"]:
        return [
            "regenerate candidate artifact so object_states frame count and frame_index match the BOP capture manifest",
        ]
    if scenario_audit["status"] != "objectstate_bop_identity_route_scenario_metadata_ready":
        return [
            "use a controlled identity capture or enriched manifest with occlusion reappearance, multiple lighting conditions, and camera_pose motion metadata",
            "do not relax the identity scenario gate just to make BOP identity evidence pass",
        ]
    if not identity_record["validator_ok"]:
        return [
            "run: uv run objgauss object-state controlled-identity-handoff "
            f"{out / 'capture-manifest.json'} {candidate_path} --output-dir {out / 'identity-handoff'} "
            f"--capture-root {scene} --candidate-id {sample_id}-identity-candidate "
            f"--artifact-ref {candidate_path}"
        ]
    if not phase1_record["validator_ok"]:
        return [
            "run: uv run objgauss object-state audit-phase1-evidence-ledger "
            f"--identity-summary {out / 'identity-handoff' / 'identity-evidence-package-summary.json'} "
            f"--summary-output {out / 'phase1-evidence-ledger.json'} --require-reviewable"
        ]
    return [
        "review identity evidence package status and rerun the route audit",
        f"keep prediction evidence on the separate BOP prediction route for dataset {dataset_id}",
    ]


def _resolve_identity_dir(output_root: Path, identity_dir: str | Path) -> Path:
    path = Path(identity_dir)
    return path if path.is_absolute() else output_root / path


def _scenario_requirements(
    min_frames: int,
    min_occlusion_fraction: float,
    min_view_conditions: int,
    min_lighting_conditions: int,
    min_camera_motion_m: float,
) -> dict[str, Any]:
    return {
        "min_frames": int(min_frames),
        "min_occlusion_fraction": float(min_occlusion_fraction),
        "min_view_conditions": int(min_view_conditions),
        "min_lighting_conditions": int(min_lighting_conditions),
        "min_camera_motion_m": float(min_camera_motion_m),
    }


def _empty_scenario_coverage() -> dict[str, Any]:
    return {
        "view_ids": [],
        "view_condition_count": 0,
        "lighting_ids": [],
        "lighting_condition_count": 0,
        "camera_pose_count": 0,
        "max_camera_translation_m": 0.0,
    }


def _scenario_coverage(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    view_ids: set[str] = set()
    lighting_ids: set[str] = set()
    camera_positions: list[Sequence[float]] = []
    for frame in frames:
        condition = frame.get("condition", {})
        if not isinstance(condition, Mapping):
            continue
        view_id = condition.get("view_id")
        lighting_id = condition.get("lighting_id")
        if isinstance(view_id, str) and view_id:
            view_ids.add(view_id)
        if isinstance(lighting_id, str) and lighting_id:
            lighting_ids.add(lighting_id)
        camera_pose = condition.get("camera_pose")
        if isinstance(camera_pose, Mapping):
            position = camera_pose.get("position")
            if (
                isinstance(position, Sequence)
                and not isinstance(position, (str, bytes))
                and len(position) == 3
            ):
                camera_positions.append([float(component) for component in position])
    return {
        "view_ids": sorted(view_ids),
        "view_condition_count": len(view_ids),
        "lighting_ids": sorted(lighting_ids),
        "lighting_condition_count": len(lighting_ids),
        "camera_pose_count": len(camera_positions),
        "max_camera_translation_m": _max_camera_translation(camera_positions),
    }


def _occlusion_reappearance_present(
    frames: Sequence[Mapping[str, Any]],
    *,
    min_occlusion_fraction: float,
) -> bool:
    tracks: dict[str, list[tuple[int, bool]]] = {}
    for index, frame in enumerate(frames):
        for item in frame["objects"]:
            occlusion_fraction = float(item.get("occlusion_fraction", 0.0))
            visible = bool(item.get("visible", True))
            occluded = (not visible) or occlusion_fraction >= min_occlusion_fraction
            tracks.setdefault(str(item["object_id"]), []).append((index, occluded))
    for observations in tracks.values():
        occluded_indices = [index for index, occluded in observations if occluded]
        clear_indices = [index for index, occluded in observations if not occluded]
        if any(
            any(clear < occluded for clear in clear_indices)
            and any(clear > occluded for clear in clear_indices)
            for occluded in occluded_indices
        ):
            return True
    return False


def _max_camera_translation(camera_positions: Sequence[Sequence[float]]) -> float:
    max_distance = 0.0
    for left_index, left in enumerate(camera_positions):
        for right in camera_positions[left_index + 1 :]:
            squared = sum(
                (float(left[axis]) - float(right[axis])) ** 2 for axis in range(3)
            )
            max_distance = max(max_distance, squared**0.5)
    return float(max_distance)


def _validate_scenario_audit(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("BOP identity route audit requires scenario metadata audit")
    if value.get("status") not in {
        "objectstate_bop_identity_route_scenario_metadata_ready",
        "objectstate_bop_identity_route_scenario_metadata_blocked",
    }:
        raise ValueError("BOP identity route scenario metadata status unsupported")
    readiness = value.get("readiness")
    if not isinstance(readiness, Mapping) or any(
        not isinstance(item, bool) for item in readiness.values()
    ):
        raise ValueError("BOP identity route scenario readiness must be bool mapping")
    if not isinstance(value.get("requirements"), Mapping):
        raise ValueError("BOP identity route scenario requires requirements")
    coverage = value.get("scenario_coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("BOP identity route scenario requires coverage")
    for key in ("view_ids", "lighting_ids"):
        if not isinstance(coverage.get(key), list):
            raise ValueError(f"BOP identity route scenario coverage requires {key}")
    if not isinstance(value.get("issues"), list):
        raise ValueError("BOP identity route scenario issues must be list")


def _validate_record(record: Any, *, allow_payload: bool) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("BOP identity route audit records must be mappings")
    for key in ("path", "expected_schema"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"BOP identity route audit record requires {key}")
    for key in ("exists", "is_file", "schema_ok", "validator_ok"):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"BOP identity route audit record requires bool {key}")
    binding_ok = record.get("binding_ok")
    if binding_ok is not None and not isinstance(binding_ok, bool):
        raise ValueError("BOP identity route audit binding_ok must be bool or null")
    if isinstance(record.get("size_bytes"), bool) or not isinstance(
        record.get("size_bytes"), int
    ):
        raise ValueError("BOP identity route audit record size_bytes must be int")
    if not allow_payload and record.get("payload") is not None:
        raise ValueError("BOP identity route audit candidate record cannot embed payload")
    if not isinstance(record.get("issues"), list):
        raise ValueError("BOP identity route audit record issues must be list")
