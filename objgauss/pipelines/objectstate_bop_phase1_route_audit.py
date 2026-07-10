from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from objgauss.datasets.objectstate_bop_capture_adapter import (
    BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
    OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
    objectstate_bop_capture_acceptance_summary,
    validate_objectstate_bop_capture_acceptance_summary,
)
from objgauss.pipelines.objectstate_controlled_prediction_evidence_package import (
    OBJECTSTATE_CONTROLLED_PREDICTION_EVIDENCE_PACKAGE_SCHEMA,
    validate_objectstate_controlled_prediction_evidence_package_summary,
)
from objgauss.pipelines.objectstate_phase1_evidence_ledger import (
    OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA,
    validate_objectstate_phase1_evidence_ledger_summary,
)

OBJECTSTATE_BOP_PHASE1_ROUTE_AUDIT_SCHEMA = (
    "objgauss-objectstate-bop-phase1-route-audit-v1"
)

Validator = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def objectstate_bop_phase1_route_audit(
    scene_root: str | Path,
    *,
    output_root: str | Path,
    sample_id: str,
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
    identity_policy: str = BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    pose_track_max_distance_m: float = DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
    check_artifact_refs: bool = False,
    min_rgb_bytes: int = 1,
    min_gaussian_bytes: int = 1,
    require_frame_formats: bool = True,
    hash_files: bool = False,
) -> dict[str, Any]:
    scene = Path(scene_root)
    out = Path(output_root)
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
        identity_policy=identity_policy,
        pose_track_max_distance_m=pose_track_max_distance_m,
        check_artifact_refs=check_artifact_refs,
        min_rgb_bytes=min_rgb_bytes,
        min_gaussian_bytes=min_gaussian_bytes,
        require_frame_formats=require_frame_formats,
        hash_files=hash_files,
    )
    prediction_package_path = (
        out / "reality-candidates" / "prediction-evidence-package-summary.json"
    )
    phase1_ledger_path = out / "phase1-evidence-ledger.json"
    prediction_record = _summary_record(
        prediction_package_path,
        expected_schema=OBJECTSTATE_CONTROLLED_PREDICTION_EVIDENCE_PACKAGE_SCHEMA,
        validator=validate_objectstate_controlled_prediction_evidence_package_summary,
    )
    phase1_record = _summary_record(
        phase1_ledger_path,
        expected_schema=OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA,
        validator=validate_objectstate_phase1_evidence_ledger_summary,
    )
    readiness = _readiness(
        acceptance,
        prediction_record=prediction_record,
        phase1_record=phase1_record,
    )
    status = _status(readiness)
    payload = {
        "schema": OBJECTSTATE_BOP_PHASE1_ROUTE_AUDIT_SCHEMA,
        "kind": "objectstate_bop_phase1_route_audit",
        "status": status,
        "scene_root": str(scene),
        "output_root": str(out),
        "sample_id": sample_id,
        "acceptance_schema": OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
        "prediction_evidence_package_schema": (
            OBJECTSTATE_CONTROLLED_PREDICTION_EVIDENCE_PACKAGE_SCHEMA
        ),
        "phase1_evidence_ledger_schema": OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA,
        "files": {
            "prediction_evidence_package_summary": str(prediction_package_path),
            "phase1_evidence_ledger": str(phase1_ledger_path),
        },
        "readiness": readiness,
        "records": {
            "prediction_evidence_package": prediction_record,
            "phase1_evidence_ledger": phase1_record,
        },
        "acceptance": acceptance,
        "issues": _issues(
            acceptance_issue=acceptance_issue,
            prediction_record=prediction_record,
            phase1_record=phase1_record,
            readiness=readiness,
        ),
        "hard_blockers": _hard_blockers(
            acceptance,
            acceptance_issue=acceptance_issue,
            prediction_record=prediction_record,
            phase1_record=phase1_record,
            readiness=readiness,
        ),
        "next_actions": _next_actions(
            acceptance,
            scene,
            out,
            sample_id=sample_id,
            dataset_id=dataset_id,
            prediction_record=prediction_record,
            phase1_record=phase1_record,
            readiness=readiness,
        ),
        "claim_policy": {
            "read_only_route_audit": True,
            "runs_bop_acceptance_audit": True,
            "checks_existing_prediction_evidence_package": True,
            "checks_existing_phase1_evidence_ledger": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_run_prediction_handoff": True,
            "does_not_run_prediction_eval": True,
            "does_not_train_model": True,
            "does_not_claim_identity_gate": True,
            "does_not_claim_intervention_gate": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "creates_ground_truth": False,
            "reconstructs_gaussians": False,
            "runs_prediction_handoff": False,
            "runs_prediction_eval": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_bop_phase1_route_audit_summary(payload)


def validate_objectstate_bop_phase1_route_audit_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP Phase 1 route audit summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_PHASE1_ROUTE_AUDIT_SCHEMA:
        raise ValueError(
            f"unsupported BOP Phase 1 route audit schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_phase1_route_audit":
        raise ValueError("BOP Phase 1 route audit kind is unsupported")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping) or not readiness:
        raise ValueError("BOP Phase 1 route audit requires readiness")
    if any(not isinstance(value, bool) for value in readiness.values()):
        raise ValueError("BOP Phase 1 route audit readiness values must be bool")
    expected_status = _status(readiness)
    if payload.get("status") != expected_status:
        raise ValueError("BOP Phase 1 route audit status mismatch")
    for key in ("scene_root", "output_root", "sample_id"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"BOP Phase 1 route audit requires {key}")
    if payload.get("acceptance_schema") != OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA:
        raise ValueError("BOP Phase 1 route audit acceptance_schema mismatch")
    if (
        payload.get("prediction_evidence_package_schema")
        != OBJECTSTATE_CONTROLLED_PREDICTION_EVIDENCE_PACKAGE_SCHEMA
    ):
        raise ValueError(
            "BOP Phase 1 route audit prediction evidence schema mismatch"
        )
    if payload.get("phase1_evidence_ledger_schema") != OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA:
        raise ValueError("BOP Phase 1 route audit ledger schema mismatch")
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("BOP Phase 1 route audit requires files")
    for key in ("prediction_evidence_package_summary", "phase1_evidence_ledger"):
        if not isinstance(files.get(key), str) or not files[key]:
            raise ValueError(f"BOP Phase 1 route audit missing file path {key}")
    records = payload.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("BOP Phase 1 route audit requires records")
    _validate_record(records.get("prediction_evidence_package"))
    _validate_record(records.get("phase1_evidence_ledger"))
    acceptance = payload.get("acceptance")
    if acceptance is not None:
        validate_objectstate_bop_capture_acceptance_summary(acceptance)
    for key in ("issues", "hard_blockers", "next_actions"):
        values = payload.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"BOP Phase 1 route audit {key} must be string list")
    claim_policy = payload.get("claim_policy")
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("read_only_route_audit")
        or not claim_policy.get("runs_bop_acceptance_audit")
        or not claim_policy.get("checks_existing_prediction_evidence_package")
        or not claim_policy.get("checks_existing_phase1_evidence_ledger")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_run_prediction_handoff")
        or not claim_policy.get("does_not_run_prediction_eval")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_claim_identity_gate")
        or not claim_policy.get("does_not_claim_intervention_gate")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP Phase 1 route audit must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP Phase 1 route audit cannot claim downloads, GT, reconstruction, "
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
        "status": status,
        "payload": dict(payload) if payload is not None else None,
        "issues": issues,
    }


def _readiness(
    acceptance: Mapping[str, Any] | None,
    *,
    prediction_record: Mapping[str, Any],
    phase1_record: Mapping[str, Any],
) -> dict[str, bool]:
    acceptance_pass = bool(
        acceptance
        and acceptance["status"] == "objectstate_bop_capture_acceptance_pass"
    )
    gaussian_ready = bool(
        acceptance and acceptance["readiness"]["phase1_gaussian_evidence_ready"]
    )
    prediction_reviewable = bool(
        prediction_record["validator_ok"]
        and str(prediction_record.get("status", "")).endswith("_reviewable")
    )
    phase1_payload = phase1_record.get("payload")
    phase1_prediction_reviewable = bool(
        phase1_record["validator_ok"]
        and isinstance(phase1_payload, Mapping)
        and phase1_payload["phase1_evidence_gates"]["prediction_evidence_reviewable"]
    )
    return {
        "bop_acceptance_available": acceptance is not None,
        "bop_acceptance_pass": acceptance_pass,
        "phase1_gaussian_evidence_ready": gaussian_ready,
        "prediction_evidence_package_present": bool(prediction_record["is_file"]),
        "prediction_evidence_package_reviewable": prediction_reviewable,
        "phase1_evidence_ledger_present": bool(phase1_record["is_file"]),
        "phase1_evidence_ledger_prediction_reviewable": phase1_prediction_reviewable,
        "route_ready_for_prediction_handoff": bool(acceptance_pass and gaussian_ready),
        "route_has_reviewable_prediction_evidence": bool(
            prediction_reviewable and phase1_prediction_reviewable
        ),
    }


def _status(readiness: Mapping[str, bool]) -> str:
    if readiness["route_has_reviewable_prediction_evidence"]:
        return "objectstate_bop_phase1_route_audit_prediction_reviewable"
    if readiness["route_ready_for_prediction_handoff"]:
        return "objectstate_bop_phase1_route_audit_handoff_ready"
    return "objectstate_bop_phase1_route_audit_blocked"


def _issues(
    *,
    acceptance_issue: str | None,
    prediction_record: Mapping[str, Any],
    phase1_record: Mapping[str, Any],
    readiness: Mapping[str, bool],
) -> list[str]:
    issues = []
    if acceptance_issue:
        issues.append(acceptance_issue)
    for label, record in (
        ("prediction_evidence_package", prediction_record),
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
    prediction_record: Mapping[str, Any],
    phase1_record: Mapping[str, Any],
    readiness: Mapping[str, bool],
) -> list[str]:
    blockers: list[str] = []
    if acceptance_issue:
        blockers.append("local BOP scene cannot be accepted yet")
    if acceptance:
        blockers.extend(str(blocker) for blocker in acceptance["hard_blockers"])
    if readiness["route_ready_for_prediction_handoff"]:
        if not prediction_record["validator_ok"]:
            blockers.append("prediction evidence package is not reviewable yet")
        if not phase1_record["validator_ok"]:
            blockers.append("Phase 1 evidence ledger is not reviewable yet")
    if readiness["route_has_reviewable_prediction_evidence"]:
        blockers.append(
            "BOP route only covers prediction evidence; identity and intervention gates remain unproven"
        )
    return blockers


def _next_actions(
    acceptance: Mapping[str, Any] | None,
    scene: Path,
    out: Path,
    *,
    sample_id: str,
    dataset_id: str,
    prediction_record: Mapping[str, Any],
    phase1_record: Mapping[str, Any],
    readiness: Mapping[str, bool],
) -> list[str]:
    if not acceptance:
        return [
            f"place a local BOP scene under {scene} with scene_camera.json, scene_gt.json, and rgb/",
            "rerun audit-bop-phase1-route after the local scene files exist",
        ]
    if not readiness["route_ready_for_prediction_handoff"]:
        return list(acceptance["next_actions"])
    if not prediction_record["validator_ok"]:
        return [
            "run: uv run objgauss object-state bop-prediction-baseline-handoff "
            f"{scene} --output-root {out} --sample-id {sample_id} "
            f"--dataset-id {dataset_id} --require-reviewable"
        ]
    if not phase1_record["validator_ok"]:
        return [
            "run: uv run objgauss object-state audit-phase1-evidence-ledger "
            f"--prediction-summary {out / 'reality-candidates' / 'prediction-evidence-package-summary.json'} "
            f"--summary-output {out / 'phase1-evidence-ledger.json'} --require-reviewable"
        ]
    return [
        "review prediction row status in phase1-evidence-ledger.json",
        "do not claim identity, intervention, counterfactual, or world-model evidence from this BOP route",
    ]


def _validate_record(record: Any) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("BOP Phase 1 route audit records must be mappings")
    for key in ("path", "expected_schema"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"BOP Phase 1 route audit record requires {key}")
    for key in ("exists", "is_file", "schema_ok", "validator_ok"):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"BOP Phase 1 route audit record requires bool {key}")
    if isinstance(record.get("size_bytes"), bool) or not isinstance(
        record.get("size_bytes"), int
    ):
        raise ValueError("BOP Phase 1 route audit record size_bytes must be int")
    if not isinstance(record.get("issues"), list):
        raise ValueError("BOP Phase 1 route audit record issues must be list")


__all__ = (
    "OBJECTSTATE_BOP_PHASE1_ROUTE_AUDIT_SCHEMA",
    "objectstate_bop_phase1_route_audit",
    "validate_objectstate_bop_phase1_route_audit_summary",
)
