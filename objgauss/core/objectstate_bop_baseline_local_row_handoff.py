from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from objgauss.core.objectstate_bop_baseline_candidate import (
    OBJECTSTATE_BOP_BASELINE_CANDIDATE_SCHEMA,
    write_objectstate_bop_gaussian_centroid_baseline_candidate,
    validate_objectstate_bop_baseline_candidate_summary,
)
from objgauss.core.objectstate_bop_local_row_handoff import (
    OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA,
    objectstate_bop_local_row_handoff,
    validate_objectstate_bop_local_row_handoff_summary,
)
from objgauss.core.objectstate_controlled_identity_eval import (
    ObjectStateControlledIdentityThresholds,
)
from objgauss.core.objectstate_controlled_prediction_eval import (
    ObjectStateControlledPredictionThresholds,
)

OBJECTSTATE_BOP_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA = (
    "objgauss-objectstate-bop-baseline-local-row-handoff-v1"
)


def objectstate_bop_baseline_local_row_handoff(
    scene_root: str | Path,
    *,
    output_root: str | Path,
    sample_id: str,
    candidate_artifact: str | Path | None = None,
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
    baseline_candidate_id: str = "bop-gaussian-centroid-baseline",
    identity_candidate_source: str = "bop_gaussian_centroid_single_state_baseline",
    max_centroid_distance: float | None = None,
    prediction_policy: str = "constant_velocity",
    prediction_candidate_id: str = "bop-constant-velocity-baseline",
    prediction_candidate_source: str = "controlled-prediction-baseline",
    prediction_confidence: float = 0.5,
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
    prediction_thresholds: ObjectStateControlledPredictionThresholds | None = None,
    synthetic_smoke_passed: bool = True,
    min_real_or_public_rows: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    scene = Path(scene_root)
    out = Path(output_root)
    artifact_path = (
        Path(candidate_artifact) if candidate_artifact is not None else out / "objectstates.json"
    )
    _ensure_output_paths(out, artifact_path=artifact_path, force=force)

    baseline = write_objectstate_bop_gaussian_centroid_baseline_candidate(
        scene,
        output=artifact_path,
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
        candidate_id=baseline_candidate_id,
        force=force,
    )
    local_row = objectstate_bop_local_row_handoff(
        scene,
        output_root=out,
        sample_id=sample_id,
        candidate_artifact=artifact_path,
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
        identity_candidate_id=baseline_candidate_id,
        identity_candidate_source=identity_candidate_source,
        max_centroid_distance=max_centroid_distance,
        prediction_policy=prediction_policy,
        prediction_candidate_id=prediction_candidate_id,
        prediction_candidate_source=prediction_candidate_source,
        prediction_confidence=prediction_confidence,
        check_artifact_refs=check_artifact_refs,
        min_rgb_bytes=min_rgb_bytes,
        min_gaussian_bytes=min_gaussian_bytes,
        require_frame_formats=require_frame_formats,
        hash_files=hash_files,
        min_candidate_artifact_bytes=min_candidate_artifact_bytes,
        hash_candidate_artifact=hash_candidate_artifact,
        min_identity_scenario_frames=min_identity_scenario_frames,
        min_occlusion_fraction=min_occlusion_fraction,
        min_view_conditions=min_view_conditions,
        min_lighting_conditions=min_lighting_conditions,
        min_camera_motion_m=min_camera_motion_m,
        identity_thresholds=identity_thresholds,
        prediction_thresholds=prediction_thresholds,
        synthetic_smoke_passed=synthetic_smoke_passed,
        min_real_or_public_rows=min_real_or_public_rows,
        force=force,
    )

    reviewability = {
        "baseline_candidate_written": (
            baseline["status"] == "objectstate_bop_baseline_candidate_written"
        ),
        "baseline_candidate_ready_for_identity_handoff": bool(
            baseline["readiness"]["ready_for_identity_handoff"]
        ),
        "local_row_identity_handoff_reviewable": bool(
            local_row["reviewability_gates"]["identity_handoff_reviewable"]
        ),
        "local_row_prediction_handoff_reviewable": bool(
            local_row["reviewability_gates"]["prediction_handoff_reviewable"]
        ),
        "phase1_evidence_ledger_identity_reviewable": bool(
            local_row["reviewability_gates"][
                "phase1_evidence_ledger_identity_reviewable"
            ]
        ),
        "phase1_evidence_ledger_prediction_reviewable": bool(
            local_row["reviewability_gates"][
                "phase1_evidence_ledger_prediction_reviewable"
            ]
        ),
    }
    status = (
        "objectstate_bop_baseline_local_row_handoff_reviewable"
        if all(reviewability.values())
        else "objectstate_bop_baseline_local_row_handoff_incomplete"
    )
    payload = {
        "schema": OBJECTSTATE_BOP_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA,
        "kind": "objectstate_bop_baseline_local_row_handoff",
        "status": status,
        "baseline_candidate_schema": OBJECTSTATE_BOP_BASELINE_CANDIDATE_SCHEMA,
        "local_row_handoff_schema": OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA,
        "scene_root": str(scene),
        "output_root": str(out),
        "sample_id": sample_id,
        "candidate_artifact": str(artifact_path),
        "reviewability_gates": reviewability,
        "pass_gates": dict(local_row["pass_gates"]),
        "row_counts": {
            "baseline_frames": baseline["row_counts"]["frames"],
            "baseline_states": baseline["row_counts"]["states"],
            "baseline_total_gaussians": baseline["row_counts"]["total_gaussians"],
            "identity_predictions": local_row["row_counts"]["identity_predictions"],
            "prediction_candidates": local_row["row_counts"][
                "prediction_candidates"
            ],
        },
        "files": {
            "candidate_artifact": str(artifact_path),
            "phase1_evidence_ledger": local_row["files"]["phase1_evidence_ledger"],
            "identity_evidence_package_summary": local_row["files"][
                "identity_evidence_package_summary"
            ],
            "prediction_evidence_package_summary": local_row["files"][
                "prediction_evidence_package_summary"
            ],
        },
        "baseline_candidate": baseline,
        "local_row_handoff": local_row,
        "phase1_evidence_ledger_summary": local_row[
            "phase1_evidence_ledger_summary"
        ],
        "issues": _handoff_issues(reviewability, local_row["issues"]),
        "claim_policy": {
            "writes_baseline_candidate_artifact": True,
            "runs_local_row_handoff": True,
            "requires_existing_bop_scene": True,
            "requires_existing_gaussian_evidence": True,
            "uses_gaussian_centroid_only_for_objectstate_baseline": True,
            "baseline_expected_to_be_negative_evidence": True,
            "reviewable_allows_metric_pass_or_fail": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_use_bop_pose_gt_for_objectstate_prediction": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_train_model": True,
            "does_not_claim_intervention_gate": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "captures_video": False,
            "creates_ground_truth": False,
            "uses_pose_gt_for_objectstate_prediction": False,
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
    return validate_objectstate_bop_baseline_local_row_handoff_summary(payload)


def validate_objectstate_bop_baseline_local_row_handoff_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP baseline local-row handoff summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA:
        raise ValueError(
            "unsupported BOP baseline local-row handoff schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_baseline_local_row_handoff":
        raise ValueError("BOP baseline local-row handoff kind is unsupported")
    if payload.get("baseline_candidate_schema") != OBJECTSTATE_BOP_BASELINE_CANDIDATE_SCHEMA:
        raise ValueError("BOP baseline local-row candidate schema mismatch")
    if payload.get("local_row_handoff_schema") != OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA:
        raise ValueError("BOP baseline local-row handoff schema mismatch")
    for key in ("scene_root", "output_root", "sample_id", "candidate_artifact"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"BOP baseline local-row handoff requires {key}")
    gates = payload.get("reviewability_gates")
    if not isinstance(gates, Mapping) or not gates:
        raise ValueError("BOP baseline local-row handoff requires reviewability gates")
    if any(not isinstance(value, bool) for value in gates.values()):
        raise ValueError("BOP baseline local-row handoff reviewability gates must be bool")
    expected_status = (
        "objectstate_bop_baseline_local_row_handoff_reviewable"
        if all(gates.values())
        else "objectstate_bop_baseline_local_row_handoff_incomplete"
    )
    if payload.get("status") != expected_status:
        raise ValueError("BOP baseline local-row handoff status mismatch")
    pass_gates = payload.get("pass_gates")
    if not isinstance(pass_gates, Mapping) or not pass_gates:
        raise ValueError("BOP baseline local-row handoff requires pass gates")
    if any(not isinstance(value, bool) for value in pass_gates.values()):
        raise ValueError("BOP baseline local-row handoff pass gates must be bool")
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("BOP baseline local-row handoff requires row counts")
    for key in (
        "baseline_frames",
        "baseline_states",
        "baseline_total_gaussians",
        "identity_predictions",
        "prediction_candidates",
    ):
        value = row_counts.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"BOP baseline local-row handoff invalid count: {key}")
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("BOP baseline local-row handoff requires files")
    for key in (
        "candidate_artifact",
        "phase1_evidence_ledger",
        "identity_evidence_package_summary",
        "prediction_evidence_package_summary",
    ):
        if not isinstance(files.get(key), str) or not files[key]:
            raise ValueError(f"BOP baseline local-row handoff missing file {key}")
    validate_objectstate_bop_baseline_candidate_summary(
        payload.get("baseline_candidate")
    )
    validate_objectstate_bop_local_row_handoff_summary(
        payload.get("local_row_handoff")
    )
    claim_policy = payload.get("claim_policy")
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("writes_baseline_candidate_artifact")
        or not claim_policy.get("runs_local_row_handoff")
        or not claim_policy.get("requires_existing_bop_scene")
        or not claim_policy.get("requires_existing_gaussian_evidence")
        or not claim_policy.get("uses_gaussian_centroid_only_for_objectstate_baseline")
        or not claim_policy.get("baseline_expected_to_be_negative_evidence")
        or not claim_policy.get("reviewable_allows_metric_pass_or_fail")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_use_bop_pose_gt_for_objectstate_prediction")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_claim_intervention_gate")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP baseline local-row handoff must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP baseline local-row handoff cannot claim downloads, capture, GT, "
            "pose-GT ObjectState prediction, reconstruction, tracking, learned "
            "prediction, intervention, training, public samples, replay, diffusion, "
            "or viewer mutation"
        )
    return dict(payload)


def _ensure_output_paths(
    output_root: Path,
    *,
    artifact_path: Path,
    force: bool,
) -> None:
    files = (
        artifact_path,
        output_root / "phase1-evidence-ledger.json",
        output_root / "bop-local-row-handoff-summary.json",
        output_root / "identity-handoff" / "identity-evidence-package-summary.json",
        output_root / "identity-handoff" / "handoff-summary.json",
        output_root / "reality-candidates" / "prediction-evidence-package-summary.json",
        output_root / "reality-candidates" / "prediction-eval-summary.json",
    )
    existing = [path for path in files if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "BOP baseline local-row handoff refuses to overwrite existing files: "
            + ", ".join(str(path) for path in existing)
        )
    output_root.mkdir(parents=True, exist_ok=True)


def _handoff_issues(
    gates: Mapping[str, bool],
    local_row_issues: Any,
) -> list[str]:
    labels = {
        "baseline_candidate_written": "baseline candidate artifact was not written",
        "baseline_candidate_ready_for_identity_handoff": (
            "baseline candidate is not ready for identity handoff"
        ),
        "local_row_identity_handoff_reviewable": (
            "local-row identity handoff is not reviewable"
        ),
        "local_row_prediction_handoff_reviewable": (
            "local-row prediction handoff is not reviewable"
        ),
        "phase1_evidence_ledger_identity_reviewable": (
            "phase1 evidence ledger does not expose reviewable identity evidence"
        ),
        "phase1_evidence_ledger_prediction_reviewable": (
            "phase1 evidence ledger does not expose reviewable prediction evidence"
        ),
    }
    issues = [message for key, message in labels.items() if not gates.get(key)]
    if isinstance(local_row_issues, list):
        issues.extend(str(item) for item in local_row_issues)
    return issues
