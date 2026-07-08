from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from objgauss.core.objectstate_bop_capture_adapter import (
    BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
)
from objgauss.core.objectstate_bop_identity_handoff import (
    OBJECTSTATE_BOP_IDENTITY_HANDOFF_SCHEMA,
    objectstate_bop_identity_handoff,
    validate_objectstate_bop_identity_handoff_summary,
)
from objgauss.core.objectstate_bop_prediction_baseline_handoff import (
    OBJECTSTATE_BOP_PREDICTION_BASELINE_HANDOFF_SCHEMA,
    objectstate_bop_prediction_baseline_handoff,
    validate_objectstate_bop_prediction_baseline_handoff_summary,
)
from objgauss.core.objectstate_controlled_identity_eval import (
    ObjectStateControlledIdentityThresholds,
)
from objgauss.core.objectstate_controlled_prediction_eval import (
    ObjectStateControlledPredictionThresholds,
)
from objgauss.core.objectstate_phase1_evidence_ledger import (
    OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA,
    objectstate_phase1_evidence_ledger,
    validate_objectstate_phase1_evidence_ledger_summary,
)

OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA = (
    "objgauss-objectstate-bop-local-row-handoff-v1"
)


def objectstate_bop_local_row_handoff(
    scene_root: str | Path,
    *,
    output_root: str | Path,
    sample_id: str,
    candidate_artifact: str | Path,
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
    identity_candidate_id: str | None = None,
    identity_candidate_source: str = "trainable_kernel_objectstate_nearest_pose_adapter",
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
    artifact_path = Path(candidate_artifact)
    _ensure_output_paths(out, force=force)

    prediction_handoff = objectstate_bop_prediction_baseline_handoff(
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
        condition_sidecar=condition_sidecar,
        max_frames=max_frames,
        frame_step=frame_step,
        identity_policy=identity_policy,
        pose_track_max_distance_m=pose_track_max_distance_m,
        policy=prediction_policy,
        candidate_id=prediction_candidate_id,
        candidate_source=prediction_candidate_source,
        artifact_ref=str(artifact_path),
        confidence=prediction_confidence,
        check_artifact_refs=check_artifact_refs,
        min_rgb_bytes=min_rgb_bytes,
        min_gaussian_bytes=min_gaussian_bytes,
        require_frame_formats=require_frame_formats,
        hash_files=hash_files,
        prediction_thresholds=prediction_thresholds,
        force=True,
    )
    identity_handoff = objectstate_bop_identity_handoff(
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
        identity_policy=identity_policy,
        pose_track_max_distance_m=pose_track_max_distance_m,
        candidate_id=identity_candidate_id,
        candidate_source=identity_candidate_source,
        max_centroid_distance=max_centroid_distance,
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
        synthetic_smoke_passed=synthetic_smoke_passed,
        min_real_or_public_rows=min_real_or_public_rows,
        force=True,
    )

    merged_ledger = objectstate_phase1_evidence_ledger(
        identity_summaries=(
            identity_handoff["files"]["identity_evidence_package_summary"],
        ),
        prediction_summaries=(
            prediction_handoff["files"]["prediction_evidence_package_summary"],
        ),
    )
    merged_ledger_path = out / "phase1-evidence-ledger.json"
    _write_json(merged_ledger_path, merged_ledger)

    reviewability = {
        "identity_handoff_reviewable": (
            identity_handoff["status"]
            == "objectstate_bop_identity_handoff_reviewable"
        ),
        "prediction_handoff_reviewable": (
            prediction_handoff["status"]
            == "objectstate_bop_prediction_baseline_handoff_reviewable"
        ),
        "phase1_evidence_ledger_identity_reviewable": bool(
            merged_ledger["phase1_evidence_gates"]["identity_evidence_reviewable"]
        ),
        "phase1_evidence_ledger_prediction_reviewable": bool(
            merged_ledger["phase1_evidence_gates"]["prediction_evidence_reviewable"]
        ),
        "same_sample_scope": merged_ledger["sample_scope"]["sample_ids"]
        == [sample_id],
    }
    pass_gates = {
        "identity_handoff_pass": bool(
            identity_handoff["pass_gates"]["identity_handoff_pass"]
        ),
        "prediction_eval_pass": (
            prediction_handoff["prediction_eval_summary"]["status"]
            == "objectstate_controlled_prediction_eval_pass"
        ),
    }
    status = (
        "objectstate_bop_local_row_handoff_reviewable"
        if all(reviewability.values())
        else "objectstate_bop_local_row_handoff_incomplete"
    )
    payload = {
        "schema": OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA,
        "kind": "objectstate_bop_local_row_handoff",
        "status": status,
        "identity_handoff_schema": OBJECTSTATE_BOP_IDENTITY_HANDOFF_SCHEMA,
        "prediction_handoff_schema": OBJECTSTATE_BOP_PREDICTION_BASELINE_HANDOFF_SCHEMA,
        "phase1_evidence_ledger_schema": OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA,
        "scene_root": str(scene),
        "output_root": str(out),
        "sample_id": sample_id,
        "candidate_artifact": str(artifact_path),
        "reviewability_gates": reviewability,
        "pass_gates": pass_gates,
        "files": {
            "phase1_evidence_ledger": str(merged_ledger_path),
            "identity_evidence_package_summary": identity_handoff["files"][
                "identity_evidence_package_summary"
            ],
            "prediction_evidence_package_summary": prediction_handoff["files"][
                "prediction_evidence_package_summary"
            ],
            "identity_handoff_summary": identity_handoff["files"][
                "identity_handoff_summary"
            ],
            "prediction_eval_summary": prediction_handoff["files"][
                "prediction_eval_summary"
            ],
        },
        "row_counts": {
            "identity_predictions": identity_handoff["row_counts"][
                "identity_predictions"
            ],
            "prediction_candidates": prediction_handoff["row_counts"][
                "prediction_candidates"
            ],
        },
        "identity_handoff": identity_handoff,
        "prediction_handoff": prediction_handoff,
        "phase1_evidence_ledger_summary": merged_ledger,
        "issues": _handoff_issues(reviewability),
        "claim_policy": {
            "orchestrates_local_bop_identity_and_prediction_handoff": True,
            "requires_gaussian_files": True,
            "requires_candidate_artifact_for_identity": True,
            "uses_prediction_baseline": True,
            "runs_identity_eval": True,
            "runs_prediction_eval": True,
            "runs_phase1_evidence_ledger": True,
            "reviewable_allows_metric_pass_or_fail": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_train_model": True,
            "does_not_claim_intervention_gate": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "captures_video": False,
            "creates_ground_truth": False,
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
    return validate_objectstate_bop_local_row_handoff_summary(payload)


def validate_objectstate_bop_local_row_handoff_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP local row handoff summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA:
        raise ValueError(
            "unsupported BOP local row handoff schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_local_row_handoff":
        raise ValueError("BOP local row handoff kind is unsupported")
    reviewability = payload.get("reviewability_gates")
    if not isinstance(reviewability, Mapping) or not reviewability:
        raise ValueError("BOP local row handoff requires reviewability gates")
    if any(not isinstance(value, bool) for value in reviewability.values()):
        raise ValueError("BOP local row handoff reviewability gates must be bool")
    expected_status = (
        "objectstate_bop_local_row_handoff_reviewable"
        if all(reviewability.values())
        else "objectstate_bop_local_row_handoff_incomplete"
    )
    if payload.get("status") != expected_status:
        raise ValueError("BOP local row handoff status mismatch")
    pass_gates = payload.get("pass_gates")
    if not isinstance(pass_gates, Mapping) or not pass_gates:
        raise ValueError("BOP local row handoff requires pass gates")
    if any(not isinstance(value, bool) for value in pass_gates.values()):
        raise ValueError("BOP local row handoff pass gates must be bool")
    for key in ("scene_root", "output_root", "sample_id", "candidate_artifact"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"BOP local row handoff requires {key}")
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("BOP local row handoff requires files")
    for key in (
        "phase1_evidence_ledger",
        "identity_evidence_package_summary",
        "prediction_evidence_package_summary",
        "identity_handoff_summary",
        "prediction_eval_summary",
    ):
        if not isinstance(files.get(key), str) or not files[key]:
            raise ValueError(f"BOP local row handoff missing file {key}")
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("BOP local row handoff requires row counts")
    for key in ("identity_predictions", "prediction_candidates"):
        value = row_counts.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"BOP local row handoff invalid row count: {key}")
    validate_objectstate_bop_identity_handoff_summary(payload.get("identity_handoff"))
    validate_objectstate_bop_prediction_baseline_handoff_summary(
        payload.get("prediction_handoff")
    )
    validate_objectstate_phase1_evidence_ledger_summary(
        payload.get("phase1_evidence_ledger_summary")
    )
    claim_policy = payload.get("claim_policy")
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("orchestrates_local_bop_identity_and_prediction_handoff")
        or not claim_policy.get("requires_gaussian_files")
        or not claim_policy.get("requires_candidate_artifact_for_identity")
        or not claim_policy.get("uses_prediction_baseline")
        or not claim_policy.get("runs_identity_eval")
        or not claim_policy.get("runs_prediction_eval")
        or not claim_policy.get("runs_phase1_evidence_ledger")
        or not claim_policy.get("reviewable_allows_metric_pass_or_fail")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_claim_intervention_gate")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP local row handoff must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP local row handoff cannot claim downloads, capture, GT, Gaussian "
            "reconstruction, tracking, learned prediction, intervention, training, "
            "public samples, replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def _ensure_output_paths(output_root: Path, *, force: bool) -> None:
    files = (
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
            "BOP local row handoff refuses to overwrite existing files: "
            + ", ".join(str(path) for path in existing)
        )
    output_root.mkdir(parents=True, exist_ok=True)


def _handoff_issues(gates: Mapping[str, bool]) -> list[str]:
    labels = {
        "identity_handoff_reviewable": "identity handoff is not reviewable",
        "prediction_handoff_reviewable": "prediction handoff is not reviewable",
        "phase1_evidence_ledger_identity_reviewable": (
            "phase1 evidence ledger does not expose reviewable identity evidence"
        ),
        "phase1_evidence_ledger_prediction_reviewable": (
            "phase1 evidence ledger does not expose reviewable prediction evidence"
        ),
        "same_sample_scope": "identity and prediction evidence are not scoped to the same sample",
    }
    return [message for key, message in labels.items() if not gates.get(key)]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
