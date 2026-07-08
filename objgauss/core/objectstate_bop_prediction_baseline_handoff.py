from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from objgauss.core.objectstate_bop_capture_adapter import (
    OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
    objectstate_bop_capture_acceptance_summary,
    validate_objectstate_bop_capture_acceptance_summary,
)
from objgauss.core.objectstate_controlled_prediction_baseline import (
    OBJECTSTATE_CONTROLLED_PREDICTION_BASELINE_SCHEMA,
    write_objectstate_controlled_prediction_baseline_candidates,
    validate_objectstate_controlled_prediction_baseline_summary,
)
from objgauss.core.objectstate_controlled_prediction_eval import (
    OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA,
    ObjectStateControlledPredictionThresholds,
    evaluate_objectstate_controlled_prediction_candidates,
    validate_objectstate_controlled_prediction_eval_summary,
)
from objgauss.core.objectstate_controlled_prediction_evidence_package import (
    OBJECTSTATE_CONTROLLED_PREDICTION_EVIDENCE_PACKAGE_SCHEMA,
    objectstate_controlled_prediction_evidence_package,
    validate_objectstate_controlled_prediction_evidence_package_summary,
)
from objgauss.core.objectstate_controlled_reality_candidate_template import (
    OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA,
    write_objectstate_controlled_reality_candidate_templates_from_manifest,
    validate_objectstate_controlled_reality_candidate_template_summary,
)

OBJECTSTATE_BOP_PREDICTION_BASELINE_HANDOFF_SCHEMA = (
    "objgauss-objectstate-bop-prediction-baseline-handoff-v1"
)


def objectstate_bop_prediction_baseline_handoff(
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
    max_frames: int | None = None,
    frame_step: int = 1,
    policy: str = "constant_velocity",
    candidate_id: str = "bop-constant-velocity-baseline",
    candidate_source: str = "controlled-prediction-baseline",
    artifact_ref: str | None = None,
    confidence: float = 0.5,
    check_artifact_refs: bool = False,
    min_rgb_bytes: int = 1,
    min_gaussian_bytes: int = 1,
    require_frame_formats: bool = True,
    hash_files: bool = False,
    prediction_thresholds: ObjectStateControlledPredictionThresholds | None = None,
    force: bool = False,
) -> dict[str, Any]:
    scene = Path(scene_root)
    out = Path(output_root)
    candidate_dir = out / "reality-candidates"
    _ensure_output_paths(out, candidate_dir, force=force)
    effective_artifact_ref = artifact_ref or str(out / "objectstates.json")

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
        require_gaussian_files=True,
        check_artifact_refs=check_artifact_refs,
        min_rgb_bytes=min_rgb_bytes,
        min_gaussian_bytes=min_gaussian_bytes,
        require_frame_formats=require_frame_formats,
        hash_files=hash_files,
    )
    capture_manifest_path = out / "capture-manifest.json"
    acceptance_path = out / "bop-acceptance-summary.json"
    file_audit_path = out / "bop-file-audit.json"
    missing_files_path = out / "bop-missing-files.md"
    controlled_real_seed_path = out / "controlled-real-seed.json"
    _write_json(capture_manifest_path, acceptance["manifest"])
    _write_json(acceptance_path, acceptance)
    _write_json(file_audit_path, acceptance["file_audit"])
    missing_files_path.write_text(
        acceptance["file_audit"]["missing_files_markdown"],
        encoding="utf-8",
    )
    _write_json(controlled_real_seed_path, acceptance["controlled_real_manifest_seed"])

    template_summary = write_objectstate_controlled_reality_candidate_templates_from_manifest(
        capture_manifest_path,
        output_dir=candidate_dir,
        candidate_id=candidate_id,
        candidate_source=candidate_source,
        artifact_ref=effective_artifact_ref,
        force=force,
    )
    template_summary_path = candidate_dir / "template-summary.json"
    _write_json(template_summary_path, template_summary)

    baseline_summary = write_objectstate_controlled_prediction_baseline_candidates(
        capture_manifest_path,
        candidate_dir / "prediction-candidates.template.json",
        output_dir=candidate_dir,
        policy=policy,
        candidate_id=candidate_id,
        candidate_source=candidate_source,
        artifact_ref=effective_artifact_ref,
        confidence=confidence,
        force=force,
    )
    baseline_summary_path = candidate_dir / "prediction-baseline-summary.json"
    _write_json(baseline_summary_path, baseline_summary)
    _write_json(
        candidate_dir / "prediction-finalize-summary.json",
        baseline_summary["prediction_finalize_summary"],
    )

    prediction_candidates = _read_json(candidate_dir / "prediction-candidates.json")
    prediction_eval = evaluate_objectstate_controlled_prediction_candidates(
        acceptance["manifest"],
        prediction_candidates,
        thresholds=prediction_thresholds,
    )
    prediction_eval_path = candidate_dir / "prediction-eval-summary.json"
    controlled_real_prediction_path = candidate_dir / "controlled-real-prediction.json"
    _write_json(prediction_eval_path, prediction_eval)
    _write_json(
        controlled_real_prediction_path,
        prediction_eval["controlled_real_manifest"],
    )

    evidence_package = objectstate_controlled_prediction_evidence_package(
        out,
        candidate_dir=candidate_dir,
    )
    evidence_package_path = candidate_dir / "prediction-evidence-package-summary.json"
    _write_json(evidence_package_path, evidence_package)

    files = {
        "capture_manifest": capture_manifest_path,
        "bop_acceptance_summary": acceptance_path,
        "bop_file_audit": file_audit_path,
        "bop_missing_files": missing_files_path,
        "controlled_real_seed": controlled_real_seed_path,
        "template_summary": template_summary_path,
        "prediction_baseline_summary": baseline_summary_path,
        "filled_prediction_template": Path(
            baseline_summary["files"]["filled_prediction_template"]
        ),
        "prediction_candidates": candidate_dir / "prediction-candidates.json",
        "prediction_finalize_summary": candidate_dir / "prediction-finalize-summary.json",
        "prediction_eval_summary": prediction_eval_path,
        "controlled_real_prediction": controlled_real_prediction_path,
        "prediction_evidence_package_summary": evidence_package_path,
    }
    readiness = {
        "bop_acceptance_pass": (
            acceptance["status"] == "objectstate_bop_capture_acceptance_pass"
        ),
        "phase1_gaussian_evidence_ready": bool(
            acceptance["readiness"]["phase1_gaussian_evidence_ready"]
        ),
        "template_ready": (
            template_summary["status"]
            == "objectstate_controlled_reality_candidate_template_ready"
        ),
        "baseline_candidates_ready": (
            baseline_summary["status"]
            == "objectstate_controlled_prediction_baseline_candidates_ready"
        ),
        "prediction_eval_ready": bool(prediction_eval["metrics"]["prediction_count"] > 0),
        "prediction_evidence_package_reviewable": (
            evidence_package["status"]
            == "objectstate_controlled_prediction_evidence_package_reviewable"
        ),
    }
    status = (
        "objectstate_bop_prediction_baseline_handoff_reviewable"
        if all(readiness.values())
        else "objectstate_bop_prediction_baseline_handoff_incomplete"
    )
    payload = {
        "schema": OBJECTSTATE_BOP_PREDICTION_BASELINE_HANDOFF_SCHEMA,
        "kind": "objectstate_bop_prediction_baseline_handoff",
        "status": status,
        "acceptance_schema": OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
        "template_schema": OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA,
        "baseline_schema": OBJECTSTATE_CONTROLLED_PREDICTION_BASELINE_SCHEMA,
        "prediction_eval_schema": OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA,
        "evidence_package_schema": (
            OBJECTSTATE_CONTROLLED_PREDICTION_EVIDENCE_PACKAGE_SCHEMA
        ),
        "scene_root": str(scene),
        "output_root": str(out),
        "candidate_dir": str(candidate_dir),
        "sample_id": sample_id,
        "candidate": {
            "candidate_id": candidate_id,
            "source": candidate_source,
            "artifact_ref": effective_artifact_ref,
        },
        "policy": {
            "name": policy,
            "uses_target_pose_values": False,
        },
        "readiness": readiness,
        "row_counts": {
            "prediction_drafts": template_summary["row_counts"]["prediction_drafts"],
            "prediction_candidates": baseline_summary["row_counts"][
                "prediction_candidates"
            ],
        },
        "files": {key: str(value) for key, value in files.items()},
        "acceptance": acceptance,
        "template_summary": template_summary,
        "prediction_baseline_summary": baseline_summary,
        "prediction_eval_summary": prediction_eval,
        "prediction_evidence_package": evidence_package,
        "issues": _handoff_issues(readiness),
        "claim_policy": {
            "orchestrates_local_bop_prediction_handoff": True,
            "requires_gaussian_files": True,
            "generates_baseline_candidates": True,
            "runs_prediction_eval": True,
            "runs_evidence_package_audit": True,
            "uses_target_pose_values_for_eval_only": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_train_model": True,
            "does_not_claim_learned_model": True,
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
    return validate_objectstate_bop_prediction_baseline_handoff_summary(payload)


def validate_objectstate_bop_prediction_baseline_handoff_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP prediction baseline handoff summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_PREDICTION_BASELINE_HANDOFF_SCHEMA:
        raise ValueError(
            "unsupported BOP prediction baseline handoff schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_prediction_baseline_handoff":
        raise ValueError("BOP prediction baseline handoff kind is unsupported")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping) or not readiness:
        raise ValueError("BOP prediction baseline handoff requires readiness")
    if any(not isinstance(value, bool) for value in readiness.values()):
        raise ValueError("BOP prediction baseline handoff readiness values must be bool")
    expected_status = (
        "objectstate_bop_prediction_baseline_handoff_reviewable"
        if all(readiness.values())
        else "objectstate_bop_prediction_baseline_handoff_incomplete"
    )
    if payload.get("status") != expected_status:
        raise ValueError("BOP prediction baseline handoff status mismatch")
    for key in (
        "scene_root",
        "output_root",
        "candidate_dir",
        "sample_id",
    ):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"BOP prediction baseline handoff requires {key}")
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("BOP prediction baseline handoff requires files")
    for key in (
        "capture_manifest",
        "bop_acceptance_summary",
        "bop_file_audit",
        "bop_missing_files",
        "template_summary",
        "prediction_baseline_summary",
        "prediction_candidates",
        "prediction_finalize_summary",
        "prediction_eval_summary",
        "controlled_real_prediction",
        "prediction_evidence_package_summary",
    ):
        if not isinstance(files.get(key), str) or not files[key]:
            raise ValueError(f"BOP prediction baseline handoff missing file {key}")
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("BOP prediction baseline handoff requires row counts")
    for key in ("prediction_drafts", "prediction_candidates"):
        value = row_counts.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"BOP prediction baseline handoff row count invalid: {key}")
    policy = payload.get("policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("name") not in {"constant_velocity", "hold"}
        or policy.get("uses_target_pose_values") is not False
    ):
        raise ValueError("BOP prediction baseline handoff policy invalid")
    validate_objectstate_bop_capture_acceptance_summary(payload.get("acceptance"))
    validate_objectstate_controlled_reality_candidate_template_summary(
        payload.get("template_summary")
    )
    validate_objectstate_controlled_prediction_baseline_summary(
        payload.get("prediction_baseline_summary")
    )
    validate_objectstate_controlled_prediction_eval_summary(
        payload.get("prediction_eval_summary")
    )
    validate_objectstate_controlled_prediction_evidence_package_summary(
        payload.get("prediction_evidence_package")
    )
    claim_policy = payload.get("claim_policy")
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("orchestrates_local_bop_prediction_handoff")
        or not claim_policy.get("requires_gaussian_files")
        or not claim_policy.get("generates_baseline_candidates")
        or not claim_policy.get("runs_prediction_eval")
        or not claim_policy.get("runs_evidence_package_audit")
        or not claim_policy.get("uses_target_pose_values_for_eval_only")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_claim_learned_model")
        or not claim_policy.get("does_not_claim_intervention_gate")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP prediction baseline handoff must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP prediction baseline handoff cannot claim downloads, capture, GT, "
            "Gaussian reconstruction, tracking, learned prediction, intervention, "
            "training, public samples, replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def _ensure_output_paths(output_root: Path, candidate_dir: Path, *, force: bool) -> None:
    files = (
        output_root / "capture-manifest.json",
        output_root / "bop-acceptance-summary.json",
        output_root / "bop-file-audit.json",
        output_root / "bop-missing-files.md",
        output_root / "controlled-real-seed.json",
        candidate_dir / "template-summary.json",
        candidate_dir / "prediction-baseline-summary.json",
        candidate_dir / "prediction-finalize-summary.json",
        candidate_dir / "prediction-eval-summary.json",
        candidate_dir / "controlled-real-prediction.json",
        candidate_dir / "prediction-evidence-package-summary.json",
    )
    existing = [path for path in files if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "BOP prediction baseline handoff refuses to overwrite existing files: "
            + ", ".join(str(path) for path in existing)
        )
    candidate_dir.mkdir(parents=True, exist_ok=True)


def _handoff_issues(readiness: Mapping[str, bool]) -> list[str]:
    labels = {
        "bop_acceptance_pass": "BOP acceptance did not pass",
        "phase1_gaussian_evidence_ready": "phase1 Gaussian evidence is not ready",
        "template_ready": "prediction template generation is not ready",
        "baseline_candidates_ready": "baseline prediction candidates are not ready",
        "prediction_eval_ready": "prediction eval did not produce candidate metrics",
        "prediction_evidence_package_reviewable": (
            "prediction evidence package is not reviewable"
        ),
    }
    return [message for key, message in labels.items() if not readiness.get(key)]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
