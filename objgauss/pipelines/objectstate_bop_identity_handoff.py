from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from objgauss.datasets.objectstate_bop_capture_adapter import (
    BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
    OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
    objectstate_bop_capture_acceptance_summary,
    validate_objectstate_bop_capture_acceptance_summary,
)
from objgauss.evaluation.objectstate_controlled_identity_eval import (
    ObjectStateControlledIdentityThresholds,
)
from objgauss.pipelines.objectstate_controlled_identity_evidence_package import (
    OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA,
    objectstate_controlled_identity_evidence_package,
    validate_objectstate_controlled_identity_evidence_package_summary,
)
from objgauss.pipelines.objectstate_controlled_identity_handoff import (
    OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA,
    objectstate_controlled_identity_handoff,
    validate_objectstate_controlled_identity_handoff_summary,
)
from objgauss.pipelines.objectstate_identity_prediction_adapter import (
    read_trainable_kernel_identity_source,
)
from objgauss.pipelines.objectstate_phase1_evidence_ledger import (
    OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA,
    objectstate_phase1_evidence_ledger,
    validate_objectstate_phase1_evidence_ledger_summary,
)

OBJECTSTATE_BOP_IDENTITY_HANDOFF_SCHEMA = (
    "objgauss-objectstate-bop-identity-handoff-v1"
)


def objectstate_bop_identity_handoff(
    scene_root: str | Path,
    *,
    output_root: str | Path,
    sample_id: str,
    candidate_artifact: str | Path,
    identity_dir: str | Path | None = None,
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
    candidate_id: str | None = None,
    candidate_source: str = "trainable_kernel_objectstate_raw_track_adapter",
    max_centroid_distance: float | None = None,
    check_artifact_refs: bool = False,
    min_rgb_bytes: int = 1,
    min_gaussian_bytes: int = 1,
    require_frame_formats: bool = True,
    hash_files: bool = False,
    min_candidate_artifact_bytes: int = 1,
    hash_candidate_artifact: bool = False,
    min_identity_scenario_frames: int = 3,
    min_occlusion_fraction: float = 0.5,
    min_view_conditions: int = 2,
    min_lighting_conditions: int = 2,
    min_camera_motion_m: float = 0.01,
    identity_thresholds: ObjectStateControlledIdentityThresholds | None = None,
    synthetic_smoke_passed: bool = True,
    min_real_or_public_rows: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    scene = Path(scene_root)
    out = Path(output_root)
    artifact_path = Path(candidate_artifact)
    package_dir = Path(identity_dir) if identity_dir is not None else out / "identity-handoff"
    _ensure_output_paths(out, package_dir, force=force)

    acceptance = objectstate_bop_capture_acceptance_summary(
        scene,
        sample_id=sample_id,
        dataset_id=dataset_id,
        object_category=object_category,
        scenario=scenario,
        fps=fps,
        license_text=license_text,
        rgb_dir=rgb_dir,
        max_frames=max_frames,
        frame_step=frame_step,
        include_gaussian_refs=True,
        gaussian_dir=gaussian_dir,
        condition_sidecar=condition_sidecar,
        require_gaussian_files=True,
        identity_policy=identity_policy,
        pose_track_max_distance_m=pose_track_max_distance_m,
        check_artifact_refs=check_artifact_refs,
        min_rgb_bytes=min_rgb_bytes,
        min_gaussian_bytes=min_gaussian_bytes,
        require_frame_formats=require_frame_formats,
        hash_files=hash_files,
    )
    artifact = read_trainable_kernel_identity_source(artifact_path)
    handoff = objectstate_controlled_identity_handoff(
        acceptance["manifest"],
        artifact,
        candidate_id=candidate_id,
        source=candidate_source,
        artifact_refs=(str(artifact_path),),
        max_centroid_distance=max_centroid_distance,
        identity_thresholds=identity_thresholds,
        synthetic_smoke_passed=synthetic_smoke_passed,
        min_real_or_public_rows=min_real_or_public_rows,
        capture_root=scene,
        check_artifact_refs=check_artifact_refs,
        min_rgb_bytes=min_rgb_bytes,
        min_gaussian_bytes=min_gaussian_bytes,
        require_frame_formats=require_frame_formats,
        hash_files=hash_files,
        candidate_artifact_path=artifact_path,
        min_candidate_artifact_bytes=min_candidate_artifact_bytes,
        hash_candidate_artifact=hash_candidate_artifact,
        min_identity_scenario_frames=min_identity_scenario_frames,
        min_occlusion_fraction=min_occlusion_fraction,
        min_view_conditions=min_view_conditions,
        min_lighting_conditions=min_lighting_conditions,
        min_camera_motion_m=min_camera_motion_m,
    )

    files = _write_outputs(
        out,
        package_dir,
        acceptance=acceptance,
        handoff=handoff,
    )
    identity_evidence_package = objectstate_controlled_identity_evidence_package(
        package_dir
    )
    _write_json(
        Path(files["identity_evidence_package_summary"]),
        identity_evidence_package,
    )
    phase1_evidence_ledger = objectstate_phase1_evidence_ledger(
        identity_summaries=(files["identity_evidence_package_summary"],),
    )
    _write_json(Path(files["phase1_evidence_ledger"]), phase1_evidence_ledger)

    reviewability_gates = {
        "bop_acceptance_pass": (
            acceptance["status"] == "objectstate_bop_capture_acceptance_pass"
        ),
        "phase1_gaussian_evidence_ready": bool(
            acceptance["readiness"]["phase1_gaussian_evidence_ready"]
        ),
        "candidate_artifact_file_audit_pass": (
            handoff["candidate_artifact_file_audit"]["status"]
            == "objectstate_controlled_candidate_artifact_file_audit_pass"
        ),
        "candidate_artifact_ref_match": bool(
            handoff["candidate_artifact_ref_match"]["matches"]
        ),
        "identity_scenario_audit_pass": (
            handoff["identity_scenario_audit"]["status"]
            == "objectstate_controlled_identity_scenario_audit_pass"
        ),
        "identity_eval_row_present": bool(
            handoff["identity_eval"]["metrics"]["predicted_pair_count"] > 0
        ),
        "identity_evidence_package_reviewable": (
            identity_evidence_package["status"]
            == "objectstate_controlled_identity_evidence_package_reviewable"
        ),
        "phase1_evidence_ledger_identity_reviewable": bool(
            phase1_evidence_ledger["phase1_evidence_gates"][
                "identity_evidence_reviewable"
            ]
        ),
    }
    pass_gates = {
        "identity_handoff_pass": (
            handoff["status"] == "objectstate_controlled_identity_handoff_pass"
        ),
        "identity_eval_pass": (
            handoff["identity_eval"]["status"]
            == "objectstate_controlled_identity_eval_pass"
        ),
        "identity_only_reality_gate_pass": (
            handoff["controlled_real_summary"]["gate"]["status"]
            == "objectstate_reality_gate_pass"
        ),
    }
    status = (
        "objectstate_bop_identity_handoff_reviewable"
        if all(reviewability_gates.values())
        else "objectstate_bop_identity_handoff_incomplete"
    )
    payload = {
        "schema": OBJECTSTATE_BOP_IDENTITY_HANDOFF_SCHEMA,
        "kind": "objectstate_bop_identity_handoff",
        "status": status,
        "acceptance_schema": OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
        "identity_handoff_schema": OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA,
        "identity_evidence_package_schema": (
            OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA
        ),
        "phase1_evidence_ledger_schema": OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA,
        "scene_root": str(scene),
        "output_root": str(out),
        "identity_dir": str(package_dir),
        "sample_id": sample_id,
        "candidate": {
            "candidate_id": handoff["candidate"]["candidate_id"],
            "source": handoff["candidate"]["source"],
            "artifact_ref": str(artifact_path),
        },
        "reviewability_gates": reviewability_gates,
        "pass_gates": pass_gates,
        "row_counts": {
            "identity_predictions": handoff["identity_eval"]["metrics"][
                "predicted_pair_count"
            ],
            "controlled_real_pass_rows": handoff["controlled_real_summary"][
                "pass_row_count"
            ],
            "controlled_real_fail_rows": handoff["controlled_real_summary"][
                "fail_row_count"
            ],
            "controlled_real_blocked_rows": handoff["controlled_real_summary"][
                "blocked_row_count"
            ],
        },
        "files": files,
        "acceptance": acceptance,
        "identity_handoff": handoff,
        "identity_evidence_package": identity_evidence_package,
        "phase1_evidence_ledger_summary": phase1_evidence_ledger,
        "issues": _handoff_issues(reviewability_gates),
        "claim_policy": {
            "orchestrates_local_bop_identity_handoff": True,
            "requires_gaussian_files": True,
            "requires_candidate_artifact": True,
            "requires_identity_scenario_audit": True,
            "runs_identity_eval": True,
            "runs_evidence_package_audit": True,
            "runs_phase1_evidence_ledger": True,
            "reviewable_allows_identity_pass_or_fail": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_train_model": True,
            "does_not_claim_prediction_or_intervention_gate": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "captures_video": False,
            "creates_ground_truth": False,
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
    return validate_objectstate_bop_identity_handoff_summary(payload)


def validate_objectstate_bop_identity_handoff_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP identity handoff summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_IDENTITY_HANDOFF_SCHEMA:
        raise ValueError(
            "unsupported BOP identity handoff schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_identity_handoff":
        raise ValueError("BOP identity handoff kind is unsupported")
    gates = payload.get("reviewability_gates")
    if not isinstance(gates, Mapping) or not gates:
        raise ValueError("BOP identity handoff requires reviewability gates")
    if any(not isinstance(value, bool) for value in gates.values()):
        raise ValueError("BOP identity handoff reviewability gates must be bool")
    expected_status = (
        "objectstate_bop_identity_handoff_reviewable"
        if all(gates.values())
        else "objectstate_bop_identity_handoff_incomplete"
    )
    if payload.get("status") != expected_status:
        raise ValueError("BOP identity handoff status mismatch")
    for key in (
        "scene_root",
        "output_root",
        "identity_dir",
        "sample_id",
    ):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"BOP identity handoff requires {key}")
    pass_gates = payload.get("pass_gates")
    if not isinstance(pass_gates, Mapping) or not pass_gates:
        raise ValueError("BOP identity handoff requires pass gates")
    if any(not isinstance(value, bool) for value in pass_gates.values()):
        raise ValueError("BOP identity handoff pass gates must be bool")
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("BOP identity handoff requires files")
    for key in (
        "capture_manifest",
        "bop_acceptance_summary",
        "bop_file_audit",
        "bop_missing_files",
        "controlled_real_seed",
        "capture_file_audit",
        "capture_missing_files",
        "candidate_artifact_file_audit",
        "identity_scenario_audit",
        "identity_predictions",
        "identity_eval_summary",
        "controlled_real_identity",
        "controlled_real_identity_summary",
        "blocked_rows",
        "identity_handoff_summary",
        "identity_evidence_package_summary",
        "phase1_evidence_ledger",
    ):
        if not isinstance(files.get(key), str) or not files[key]:
            raise ValueError(f"BOP identity handoff missing file {key}")
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("BOP identity handoff requires row counts")
    prediction_count = row_counts.get("identity_predictions")
    if isinstance(prediction_count, bool) or not isinstance(prediction_count, int):
        raise ValueError("BOP identity handoff identity prediction count invalid")
    if prediction_count < 0:
        raise ValueError("BOP identity handoff identity prediction count negative")
    validate_objectstate_bop_capture_acceptance_summary(payload.get("acceptance"))
    validate_objectstate_controlled_identity_handoff_summary(
        payload.get("identity_handoff")
    )
    validate_objectstate_controlled_identity_evidence_package_summary(
        payload.get("identity_evidence_package")
    )
    validate_objectstate_phase1_evidence_ledger_summary(
        payload.get("phase1_evidence_ledger_summary")
    )
    claim_policy = payload.get("claim_policy")
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("orchestrates_local_bop_identity_handoff")
        or not claim_policy.get("requires_gaussian_files")
        or not claim_policy.get("requires_candidate_artifact")
        or not claim_policy.get("requires_identity_scenario_audit")
        or not claim_policy.get("runs_identity_eval")
        or not claim_policy.get("runs_evidence_package_audit")
        or not claim_policy.get("runs_phase1_evidence_ledger")
        or not claim_policy.get("reviewable_allows_identity_pass_or_fail")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_claim_prediction_or_intervention_gate")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP identity handoff must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP identity handoff cannot claim downloads, capture, GT, Gaussian "
            "reconstruction, tracking, prediction, intervention, training, public "
            "samples, replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def _write_outputs(
    out: Path,
    package_dir: Path,
    *,
    acceptance: Mapping[str, Any],
    handoff: Mapping[str, Any],
) -> dict[str, str]:
    files = {
        "capture_manifest": package_dir / "capture-manifest.json",
        "bop_acceptance_summary": out / "bop-acceptance-summary.json",
        "bop_file_audit": out / "bop-file-audit.json",
        "bop_missing_files": out / "bop-missing-files.md",
        "controlled_real_seed": out / "controlled-real-seed.json",
        "capture_file_audit": package_dir / "capture-file-audit.json",
        "capture_missing_files": package_dir / "capture-missing-files.md",
        "candidate_artifact_file_audit": (
            package_dir / "candidate-artifact-file-audit.json"
        ),
        "identity_scenario_audit": package_dir / "identity-scenario-audit.json",
        "identity_predictions": package_dir / "identity-predictions.json",
        "identity_eval_summary": package_dir / "identity-eval-summary.json",
        "controlled_real_identity": package_dir / "controlled-real.json",
        "controlled_real_identity_summary": package_dir / "controlled-real-summary.json",
        "blocked_rows": package_dir / "blocked-rows.md",
        "identity_handoff_summary": package_dir / "handoff-summary.json",
        "identity_evidence_package_summary": (
            package_dir / "identity-evidence-package-summary.json"
        ),
        "phase1_evidence_ledger": out / "phase1-evidence-ledger.json",
    }
    _write_json(files["capture_manifest"], acceptance["manifest"])
    _write_json(files["bop_acceptance_summary"], acceptance)
    _write_json(files["bop_file_audit"], acceptance["file_audit"])
    files["bop_missing_files"].write_text(
        acceptance["file_audit"]["missing_files_markdown"],
        encoding="utf-8",
    )
    _write_json(files["controlled_real_seed"], acceptance["controlled_real_manifest_seed"])
    _write_json(files["capture_file_audit"], handoff["capture_file_audit"])
    files["capture_missing_files"].write_text(
        handoff["capture_file_audit"]["missing_files_markdown"],
        encoding="utf-8",
    )
    _write_json(
        files["candidate_artifact_file_audit"],
        handoff["candidate_artifact_file_audit"],
    )
    _write_json(files["identity_scenario_audit"], handoff["identity_scenario_audit"])
    _write_json(files["identity_predictions"], handoff["identity_predictions"])
    _write_json(files["identity_eval_summary"], handoff["identity_eval"])
    _write_json(files["controlled_real_identity"], handoff["controlled_real_manifest"])
    _write_json(
        files["controlled_real_identity_summary"],
        handoff["controlled_real_summary"],
    )
    files["blocked_rows"].write_text(
        handoff["controlled_real_summary"]["blocked_rows_markdown"],
        encoding="utf-8",
    )
    _write_json(files["identity_handoff_summary"], handoff)
    return {key: str(value) for key, value in files.items()}


def _ensure_output_paths(out: Path, package_dir: Path, *, force: bool) -> None:
    files = (
        out / "bop-acceptance-summary.json",
        out / "bop-file-audit.json",
        out / "bop-missing-files.md",
        out / "controlled-real-seed.json",
        out / "phase1-evidence-ledger.json",
        package_dir / "capture-manifest.json",
        package_dir / "capture-file-audit.json",
        package_dir / "capture-missing-files.md",
        package_dir / "candidate-artifact-file-audit.json",
        package_dir / "identity-scenario-audit.json",
        package_dir / "identity-predictions.json",
        package_dir / "identity-eval-summary.json",
        package_dir / "controlled-real.json",
        package_dir / "controlled-real-summary.json",
        package_dir / "blocked-rows.md",
        package_dir / "handoff-summary.json",
        package_dir / "identity-evidence-package-summary.json",
    )
    existing = [path for path in files if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "BOP identity handoff refuses to overwrite existing files: "
            + ", ".join(str(path) for path in existing)
        )
    out.mkdir(parents=True, exist_ok=True)
    package_dir.mkdir(parents=True, exist_ok=True)


def _handoff_issues(gates: Mapping[str, bool]) -> list[str]:
    labels = {
        "bop_acceptance_pass": "BOP acceptance did not pass",
        "phase1_gaussian_evidence_ready": "phase1 Gaussian evidence is not ready",
        "candidate_artifact_file_audit_pass": "candidate artifact file audit did not pass",
        "candidate_artifact_ref_match": "candidate artifact ref did not match audited file",
        "identity_scenario_audit_pass": "identity scenario audit did not pass",
        "identity_eval_row_present": "identity eval did not produce prediction rows",
        "identity_evidence_package_reviewable": (
            "identity evidence package is not reviewable"
        ),
        "phase1_evidence_ledger_identity_reviewable": (
            "phase1 evidence ledger does not expose reviewable identity evidence"
        ),
    }
    return [message for key, message in labels.items() if not gates.get(key)]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
